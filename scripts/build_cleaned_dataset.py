#!/usr/bin/env python3
"""
Confident learning-based data cleaning.

Cross-references model predictions with pct_rating to identify and remove
contradictory labels. Flags entries that satisfy either:
  1) The model prediction is incorrect and low-confidence, or
  2) The label contradicts the human pct_rating.

Creates a text-only cleaned variant when only text inputs are provided:
  - textconf:   remove papers flagged by the text contradiction rule

If both text and vision inputs are provided, also creates:
  - visionconf: remove papers flagged by the vision contradiction rule
  - bothconf:   remove papers flagged by BOTH text AND vision rules

Usage:
    python scripts/build_cleaned_dataset.py \
        --text-predictions results/data_cleaning/text_train_predictions.jsonl \
        --vision-predictions results/data_cleaning/vision_train_predictions.jsonl \
        --text-dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered \
        --vision-dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480 \
        --output-report results/data_cleaning/cleaning_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

DATA_DIR = Path("data")

BOXED_RE = re.compile(r"\\boxed\{(Accept|Reject)\}")
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_ACCEPT_PCT_THRESHOLD = 0.6
DEFAULT_REJECT_PCT_THRESHOLD = 0.4


# ── helpers ──────────────────────────────────────────────────────────────────

def _sanitize_surrogates(obj):
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: _sanitize_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_surrogates(v) for v in obj]
    return obj


def load_source_data(dataset_name, split):
    path = DATA_DIR / f"{dataset_name}_{split}" / "data.json"
    with open(path) as f:
        return json.load(f)


def write_dataset(data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _sanitize_surrogates(data)
    with open(output_dir / "data.json", "w") as f:
        json.dump(data, f, indent=None, ensure_ascii=False)


def update_dataset_info(new_entries):
    info_path = DATA_DIR / "dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)
    info.update(new_entries)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")


def make_dataset_info_entry(dataset_name, split, is_vision):
    entry = {
        "file_name": f"{dataset_name}_{split}/data.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    }
    if is_vision:
        entry["columns"]["images"] = "images"
    return entry


# ── prediction loading ───────────────────────────────────────────────────────

def parse_accept_reject(text: str) -> str | None:
    """Extract Accept or Reject from a string containing \\boxed{...}."""
    m = BOXED_RE.search(text)
    return m.group(1) if m else None


def load_predictions(jsonl_path: str) -> list[dict]:
    """Load prediction JSONL from vllm_infer.py."""
    preds = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    return preds


def find_decision_token_idx(all_logprobs: list[list[float]]) -> int | None:
    """Find the output token position that best tracks the final decision."""
    if not all_logprobs:
        return None

    width = max(len(row) for row in all_logprobs)
    padded = []
    for row in all_logprobs:
        if len(row) < width:
            row = row + [row[-1]] * (width - len(row))
        padded.append(row)

    variances = []
    for col in range(width):
        values = [row[col] for row in padded]
        mean = sum(values) / len(values)
        variances.append(sum((v - mean) ** 2 for v in values) / len(values))
    return max(range(width), key=variances.__getitem__)


def extract_model_confidence(prediction: dict, decision_token_idx: int | None) -> float | None:
    """Extract probability-like confidence from the decision token logprob."""
    token_logprobs = prediction.get("token_logprobs") or []
    if not token_logprobs or decision_token_idx is None:
        return None

    conf_idx = min(decision_token_idx, len(token_logprobs) - 1)
    return math.exp(token_logprobs[conf_idx])


def summarize_prediction_fields(predictions: list[dict]) -> dict:
    """Summarize which optional prediction fields are present."""
    total = len(predictions)
    if total == 0:
        return {"count": 0, "predict_present": 0, "label_present": 0, "token_logprobs_present": 0}

    return {
        "count": total,
        "predict_present": sum(1 for p in predictions if p.get("predict") is not None),
        "label_present": sum(1 for p in predictions if p.get("label") is not None),
        "token_logprobs_present": sum(1 for p in predictions if p.get("token_logprobs") is not None),
    }


# ── core logic ───────────────────────────────────────────────────────────────

def build_entry_records(train_data, predictions):
    """
    Merge training data metadata with model predictions.
    Returns list of dicts with: index, submission_id, year, pct_rating,
    ground_truth, model_pred, model_confidence, and the original data entry.
    """
    assert len(train_data) == len(predictions), (
        f"Mismatch: {len(train_data)} training entries vs {len(predictions)} predictions"
    )

    decision_token_idx = find_decision_token_idx(
        [pred["token_logprobs"] for pred in predictions if pred.get("token_logprobs")]
    )

    records = []
    for i, (entry, pred) in enumerate(zip(train_data, predictions)):
        meta = entry.get("_metadata", {})
        submission_id = meta.get("submission_id")
        year = meta.get("year")
        pct_rating = meta.get("pct_rating")

        # Ground truth from metadata
        gt_label = meta.get("answer")
        if gt_label is None:
            # Fall back to parsing from prediction label field
            gt_label = parse_accept_reject(pred.get("label", ""))

        model_pred = parse_accept_reject(pred.get("predict", ""))
        model_confidence = extract_model_confidence(pred, decision_token_idx)

        records.append({
            "index": i,
            "submission_id": submission_id,
            "year": year,
            "pct_rating": pct_rating,
            "ground_truth": gt_label,
            "model_pred": model_pred,
            "model_confidence": model_confidence,
            "entry": entry,
        })

    return records, decision_token_idx


def collect_flagged_ids(
    records,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    accept_pct_threshold=DEFAULT_ACCEPT_PCT_THRESHOLD,
    reject_pct_threshold=DEFAULT_REJECT_PCT_THRESHOLD,
):
    """Collect contradiction-based flagged ids and reason-specific subsets."""
    incorrect_low_conf = set()
    pct_contradiction = set()

    for r in records:
        gt = r["ground_truth"]
        pred = r["model_pred"]
        pct = r["pct_rating"]
        conf = r["model_confidence"]
        sid = r["submission_id"]

        if sid is None or gt is None:
            continue

        if pred is not None and pred != gt and conf is not None and conf < confidence_threshold:
            incorrect_low_conf.add(sid)

        if pct is not None:
            if gt == "Accept" and pct < accept_pct_threshold:
                pct_contradiction.add(sid)
            elif gt == "Reject" and pct > reject_pct_threshold:
                pct_contradiction.add(sid)

    flagged = incorrect_low_conf | pct_contradiction
    return {
        "flagged": flagged,
        "incorrect_low_conf": incorrect_low_conf,
        "pct_contradiction": pct_contradiction,
        "overlap": incorrect_low_conf & pct_contradiction,
    }


def summarize_record_coverage(records, decision_token_idx):
    """Summarize data availability relevant to the contradiction rules."""
    return {
        "count": len(records),
        "decision_token_idx": decision_token_idx,
        "ground_truth_present": sum(1 for r in records if r["ground_truth"] is not None),
        "model_pred_present": sum(1 for r in records if r["model_pred"] is not None),
        "model_confidence_present": sum(1 for r in records if r["model_confidence"] is not None),
        "pct_rating_present": sum(1 for r in records if r["pct_rating"] is not None),
    }


def remove_flagged(train_data, records, flagged_ids):
    """Return filtered training data with flagged submission_ids removed."""
    cleaned = []
    for r in records:
        if r["submission_id"] not in flagged_ids:
            cleaned.append(r["entry"])
    return cleaned


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Contradiction-based data cleaning"
    )
    parser.add_argument("--text-predictions", required=True,
                        help="Path to text model predictions JSONL")
    parser.add_argument("--vision-predictions", default=None,
                        help="Path to vision model predictions JSONL (optional)")
    parser.add_argument("--text-dataset", required=True,
                        help="Base name of text training dataset")
    parser.add_argument("--vision-dataset", default=None,
                        help="Base name of vision training dataset (optional)")
    parser.add_argument("--output-report", required=True,
                        help="Path to save cleaning report JSON")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    has_vision = args.vision_predictions is not None and args.vision_dataset is not None

    print("Loading text training data...")
    text_train = load_source_data(args.text_dataset, "train")
    print(f"  {len(text_train)} entries")

    print("Loading text predictions...")
    text_preds = load_predictions(args.text_predictions)
    print(f"  {len(text_preds)} predictions")

    vision_train = None
    vision_preds = None
    if has_vision:
        print("Loading vision training data...")
        vision_train = load_source_data(args.vision_dataset, "train")
        print(f"  {len(vision_train)} entries")

        print("Loading vision predictions...")
        vision_preds = load_predictions(args.vision_predictions)
        print(f"  {len(vision_preds)} predictions")
    else:
        print("\nNo vision predictions provided — creating textconf variant only.")

    # ── Build records ────────────────────────────────────────────────────
    text_records, text_decision_token_idx = build_entry_records(text_train, text_preds)
    vision_records, vision_decision_token_idx = build_entry_records(vision_train, vision_preds) if has_vision else ([], None)

    # ── Flag contradictory entries ───────────────────────────────────────
    print(
        "\nUsing contradiction thresholds: "
        f"confidence<{DEFAULT_CONFIDENCE_THRESHOLD}, "
        f"Accept pct_rating<{DEFAULT_ACCEPT_PCT_THRESHOLD}, "
        f"Reject pct_rating>{DEFAULT_REJECT_PCT_THRESHOLD}"
    )
    text_reasons = collect_flagged_ids(text_records)
    vision_reasons = collect_flagged_ids(vision_records) if has_vision else None
    text_flagged = text_reasons["flagged"]
    vision_flagged = vision_reasons["flagged"] if has_vision else set()

    # For bothconf: intersection of flagged submission_ids across modalities
    text_sids = {r["submission_id"] for r in text_records}
    vision_sids = {r["submission_id"] for r in vision_records} if has_vision else set()
    shared_sids = text_sids & vision_sids if has_vision else set()

    both_flagged = (text_flagged & vision_flagged) & shared_sids if has_vision else set()

    print(f"\nFlagged by text contradiction rule:   {len(text_flagged)}")
    print(f"  Incorrect + low confidence:         {len(text_reasons['incorrect_low_conf'])}")
    print(f"  pct_rating contradiction:           {len(text_reasons['pct_contradiction'])}")
    if has_vision:
        print(f"Flagged by vision contradiction rule: {len(vision_flagged)}")
        print(f"Flagged by BOTH (intersection):       {len(both_flagged)}")
        print(f"Shared submission_ids across modalities: {len(shared_sids)}")

    # ── Create cleaned datasets ──────────────────────────────────────────
    variants = {"textconf": text_flagged}
    if has_vision:
        variants["visionconf"] = vision_flagged
        variants["bothconf"] = both_flagged

    dataset_info_entries = {}
    report = {
        "mode": "text+vision" if has_vision else "text-only",
        "thresholds": {
            "confidence_lt": DEFAULT_CONFIDENCE_THRESHOLD,
            "accept_pct_rating_lt": DEFAULT_ACCEPT_PCT_THRESHOLD,
            "reject_pct_rating_gt": DEFAULT_REJECT_PCT_THRESHOLD,
        },
        "text_flagged_count": len(text_flagged),
        "prediction_fields": {
            "text": summarize_prediction_fields(text_preds),
        },
        "record_coverage": {
            "text": summarize_record_coverage(text_records, text_decision_token_idx),
        },
        "text_flagged_ids": sorted(text_flagged),
        "text_flagging_reasons": {
            "incorrect_low_conf_count": len(text_reasons["incorrect_low_conf"]),
            "pct_contradiction_count": len(text_reasons["pct_contradiction"]),
            "overlap_count": len(text_reasons["overlap"]),
            "incorrect_low_conf_ids": sorted(text_reasons["incorrect_low_conf"]),
            "pct_contradiction_ids": sorted(text_reasons["pct_contradiction"]),
            "overlap_ids": sorted(text_reasons["overlap"]),
        },
        "variants": {},
    }
    if has_vision:
        report.update({
            "vision_flagged_count": len(vision_flagged),
            "both_flagged_count": len(both_flagged),
            "shared_submission_ids": len(shared_sids),
            "vision_flagged_ids": sorted(vision_flagged),
            "both_flagged_ids": sorted(both_flagged),
        })
        report["prediction_fields"]["vision"] = summarize_prediction_fields(vision_preds)
        report["record_coverage"]["vision"] = summarize_record_coverage(
            vision_records, vision_decision_token_idx
        )
        report["vision_flagging_reasons"] = {
            "incorrect_low_conf_count": len(vision_reasons["incorrect_low_conf"]),
            "pct_contradiction_count": len(vision_reasons["pct_contradiction"]),
            "overlap_count": len(vision_reasons["overlap"]),
            "incorrect_low_conf_ids": sorted(vision_reasons["incorrect_low_conf"]),
            "pct_contradiction_ids": sorted(vision_reasons["pct_contradiction"]),
            "overlap_ids": sorted(vision_reasons["overlap"]),
        }

    for variant_name, flagged_ids in variants.items():
        print(f"\n── Variant: {variant_name} ──")

        # Text modality
        text_cleaned = remove_flagged(text_train, text_records, flagged_ids)
        text_out_name = f"{args.text_dataset}_cleaned_{variant_name}"
        text_out_dir = DATA_DIR / f"{text_out_name}_train"
        write_dataset(text_cleaned, text_out_dir)
        print(f"  Text: {len(text_train)} → {len(text_cleaned)} "
              f"(removed {len(text_train) - len(text_cleaned)})")

        # Register text train
        entry_key = f"{text_out_name}_train"
        dataset_info_entries[entry_key] = make_dataset_info_entry(
            text_out_name, "train", is_vision=False
        )
        # Register text validation and test (unchanged, point to source)
        for split in ["validation", "test"]:
            split_key = f"{text_out_name}_{split}"
            dataset_info_entries[split_key] = make_dataset_info_entry(
                args.text_dataset, split, is_vision=False
            )

        # Vision modality
        if has_vision:
            vision_cleaned = remove_flagged(vision_train, vision_records, flagged_ids)
            vision_out_name = f"{args.vision_dataset}_cleaned_{variant_name}"
            vision_out_dir = DATA_DIR / f"{vision_out_name}_train"
            write_dataset(vision_cleaned, vision_out_dir)
            print(f"  Vision: {len(vision_train)} → {len(vision_cleaned)} "
                  f"(removed {len(vision_train) - len(vision_cleaned)})")

            # Register vision train
            entry_key = f"{vision_out_name}_train"
            dataset_info_entries[entry_key] = make_dataset_info_entry(
                vision_out_name, "train", is_vision=True
            )
            # Register vision validation and test (unchanged, point to source)
            for split in ["validation", "test"]:
                split_key = f"{vision_out_name}_{split}"
                dataset_info_entries[split_key] = make_dataset_info_entry(
                    args.vision_dataset, split, is_vision=True
                )

        report["variants"][variant_name] = {
            "text_original": len(text_train),
            "text_cleaned": len(text_cleaned),
            "text_removed": len(text_train) - len(text_cleaned),
        }
        if has_vision:
            report["variants"][variant_name].update({
                "vision_original": len(vision_train),
                "vision_cleaned": len(vision_cleaned),
                "vision_removed": len(vision_train) - len(vision_cleaned),
            })

    # ── Update dataset_info.json ─────────────────────────────────────────
    print(f"\nRegistering {len(dataset_info_entries)} dataset entries...")
    update_dataset_info(dataset_info_entries)

    # ── Save report ──────────────────────────────────────────────────────
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Cleaning report saved to {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
