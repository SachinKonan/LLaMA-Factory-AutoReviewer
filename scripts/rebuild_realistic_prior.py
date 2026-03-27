"""Rebuild realistic-prior agent benchmarks with correctly aligned vision predictions.

The original build_agent_sample.py paired the TEXT test split (2495 rows) with
VISION predictions (2498 rows), causing misalignment after index 1186.  All 133
extra rejects fell after that index and got the wrong paper's prediction.

This script uses the VISION test split (2498 rows) to build a correct
pred_lookup, then re-selects 133 extra rejects targeting ~63.9% reject recall
(matching 2026-only vision model performance) and mean confidence ~0.67.

Outputs `_2` variants of the realistic prior dirs and ground truth files.

Usage:
  python scripts/rebuild_realistic_prior.py
"""

import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

# ── Paths ──────────────────────────────────────────────────────────────────
METADATA_DIR = Path("data/massive_metadata_v7_5")

# Correctly use VISION test split for alignment with vision predictions
VISION_TEST_SPLIT = Path(
    "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json"
)
VISION_PREDICTIONS = Path(
    "results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl"
)

# Base 200 test papers (no-model variant, paper content only)
BASE_NO_MODEL_DIR = Path("coding_agents_2026_codex_base")
BASE_WITH_MODEL_DIR = Path("coding_agents_2026_codex_with_best_model_and_model_prior_base")
BASE_GROUND_TRUTH = Path("data/agent_ground_truth_2026.json")

# Existing run dirs (for copying AGENTS.override.md + main.py)
EXISTING_RUN_NO_MODEL = Path("coding_agents_2026_codex_realistic_prior")
EXISTING_RUN_WITH_MODEL = Path("coding_agents_2026_codex_with_best_model_and_model_prior_realistic_prior")

# Output dirs
OUT_BASE_NO_MODEL = Path("coding_agents_2026_codex_realistic_prior_base_2")
OUT_BASE_WITH_MODEL = Path("coding_agents_2026_codex_with_best_model_and_model_prior_realistic_prior_base_2")
OUT_RUN_NO_MODEL = Path("coding_agents_2026_codex_realistic_prior_2")
OUT_RUN_WITH_MODEL = Path("coding_agents_2026_codex_with_best_model_and_model_prior_realistic_prior_2")
OUT_GT_NO_MODEL = Path("data/agent_ground_truth_2026_realistic_prior_2.json")
OUT_GT_WITH_MODEL = Path("data/agent_ground_truth_2026_realistic_prior_with_model_2.json")

# ── Constants ──────────────────────────────────────────────────────────────
ADDED_REJECTS = 133
TARGET_REJECT_RECALL = 0.639  # 2026-only vision model reject recall
TARGET_CONFIDENCE = 0.67
SHUFFLE_SEED = 20260311
SEED_SEARCH_MAX = 10000

KEEP_FIELDS = {
    "type", "text", "text_level", "text_format",
    "page_idx", "bbox",
    "image_caption", "image_footnote",
    "table_body", "table_caption", "table_footnote",
    "sub_type", "list_items",
}
LEAK_PATTERNS = [
    "we accept", "we reject", "paper is accepted", "paper is rejected",
    "decision: accept", "decision: reject",
    "overall rating", "reviewer confidence",
    "score:", "rating:",
]


# ── Helpers (from build_agent_sample.py) ───────────────────────────────────
def sanitize_element(elem):
    text = elem.get("text", "")
    if isinstance(text, str):
        text_lower = text.lower()
        for pattern in LEAK_PATTERNS:
            if pattern in text_lower:
                return None
    cleaned = {k: v for k, v in elem.items() if k in KEEP_FIELDS and v is not None}
    if "type" not in cleaned:
        return None
    return cleaned


def extract_headers(content_list):
    return [
        elem["text"]
        for elem in content_list
        if elem.get("text_level") is not None and elem.get("text")
    ]


def build_paper_data(sub_id, content_json):
    content_list_raw = json.loads(content_json) if isinstance(content_json, str) else content_json
    content_list = [c for elem in content_list_raw if (c := sanitize_element(elem)) is not None]
    headers = extract_headers(content_list)
    return {"content_list": content_list, "headers": headers}


def extract_prediction_label(text):
    text_lower = (text or "").lower()
    accept_pos = text_lower.rfind("accept")
    reject_pos = text_lower.rfind("reject")
    if accept_pos == -1 and reject_pos == -1:
        return "unknown"
    return "accept" if accept_pos > reject_pos else "reject"


def find_decision_token_idx(all_logprobs):
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


# ── Core: build correctly-aligned vision prediction lookup ─────────────────
def load_vision_prediction_lookup():
    """Load vision test split + vision predictions, correctly aligned."""
    with open(VISION_TEST_SPLIT) as f:
        vision_rows = json.load(f)
    predictions = []
    with open(VISION_PREDICTIONS) as f:
        for line in f:
            predictions.append(json.loads(line))

    assert len(vision_rows) == len(predictions), (
        f"Vision test split ({len(vision_rows)}) != predictions ({len(predictions)})"
    )

    paired = list(zip(vision_rows, predictions))
    decision_token_idx = find_decision_token_idx(
        [pred["token_logprobs"] for _, pred in paired if pred.get("token_logprobs")]
    )

    pred_lookup = {}
    for row, pred in paired:
        submission_id = row["_metadata"]["submission_id"]
        decision = extract_prediction_label(pred.get("predict", ""))
        if decision == "unknown":
            continue
        token_logprobs = pred.get("token_logprobs") or []
        if not token_logprobs:
            continue
        conf_idx = min(decision_token_idx, len(token_logprobs) - 1)
        pred_lookup[submission_id] = {
            "model_prediction": decision,
            "model_confidence": round(math.exp(token_logprobs[conf_idx]), 4),
        }

    print(f"Vision pred_lookup: {len(pred_lookup)} papers with predictions")
    return pred_lookup, vision_rows


def main():
    print("=" * 70)
    print("Rebuilding realistic-prior benchmarks with correct vision alignment")
    print("=" * 70)

    # 1. Build correctly-aligned prediction lookup
    pred_lookup, vision_rows = load_vision_prediction_lookup()

    # 2. Load base 200 test papers (no-model variant)
    with open(BASE_NO_MODEL_DIR / "TEST_SAMPLE.json") as f:
        base_test_no_model = json.load(f)
    with open(BASE_NO_MODEL_DIR / "TRAIN_SAMPLE.json") as f:
        train_sample = json.load(f)
    with open(BASE_GROUND_TRUTH) as f:
        base_ground_truth = json.load(f)

    base_ids = set(base_test_no_model.keys())
    train_ids = set(train_sample.keys())
    print(f"\nBase 200 test papers: {len(base_ids)}")
    print(f"Train papers: {len(train_ids)}")

    # Verify base 200 reject recall with correct lookup
    base_reject_correct = sum(
        1 for sid, label in base_ground_truth.items()
        if label == "reject" and pred_lookup.get(sid, {}).get("model_prediction") == "reject"
    )
    base_reject_total = sum(1 for label in base_ground_truth.values() if label == "reject")
    print(f"Base reject recall (correct lookup): {base_reject_correct}/{base_reject_total} = {base_reject_correct/base_reject_total:.4f}")

    # 3. Build reject pool from VISION test split (2026 rejects not in train or base test)
    existing_ids = train_ids | base_ids
    reject_pool_rows = [
        row for row in vision_rows
        if row["_metadata"]["year"] == 2026
        and row["_metadata"]["answer"].lower() == "reject"
        and row["_metadata"]["submission_id"] not in existing_ids
    ]
    # Filter to those with predictions
    reject_pool_rows = [
        row for row in reject_pool_rows
        if row["_metadata"]["submission_id"] in pred_lookup
    ]
    print(f"Reject pool (2026, with preds, excl train+base): {len(reject_pool_rows)}")
    assert len(reject_pool_rows) >= ADDED_REJECTS, (
        f"Not enough rejects: need {ADDED_REJECTS}, have {len(reject_pool_rows)}"
    )

    # 4. Seed search: target ~63.9% reject recall AND mean confidence ~0.67
    #    Base has 68/100 correct rejects.
    #    Total rejects will be 100 + 133 = 233
    #    Target: 233 * 0.639 ≈ 148.9 correct → need ~81 from extras
    target_extra_correct = round(233 * TARGET_REJECT_RECALL) - base_reject_correct
    print(f"\nTarget: {base_reject_correct} + ~{target_extra_correct} = ~{base_reject_correct + target_extra_correct} / 233 reject recall")

    base_conf_sum = sum(
        pred_lookup[sid]["model_confidence"]
        for sid in base_test_no_model
        if sid in pred_lookup
    )
    base_with_preds = sum(1 for sid in base_test_no_model if sid in pred_lookup)
    print(f"Base papers with predictions: {base_with_preds}/200")

    best = None
    for seed in range(SEED_SEARCH_MAX):
        rng = random.Random(seed)
        sampled_rows = rng.sample(reject_pool_rows, ADDED_REJECTS)
        sampled_ids = [row["_metadata"]["submission_id"] for row in sampled_rows]

        # Count how many extra rejects the model correctly predicts as reject
        extra_correct = sum(
            1 for sid in sampled_ids
            if pred_lookup[sid]["model_prediction"] == "reject"
        )
        total_reject_correct = base_reject_correct + extra_correct
        actual_recall = total_reject_correct / 233

        # Mean confidence over all 333 papers
        added_conf_sum = sum(pred_lookup[sid]["model_confidence"] for sid in sampled_ids)
        final_mean_conf = (base_conf_sum + added_conf_sum) / (base_with_preds + ADDED_REJECTS)

        # Combined score: primarily match reject recall, secondarily match confidence
        recall_err = abs(actual_recall - TARGET_REJECT_RECALL)
        conf_err = abs(final_mean_conf - TARGET_CONFIDENCE)
        score = recall_err * 10 + conf_err  # weight recall more heavily

        candidate = (score, recall_err, conf_err, seed, actual_recall, final_mean_conf, sampled_rows)
        if best is None or candidate < best:
            best = candidate

    score, recall_err, conf_err, chosen_seed, actual_recall, final_mean_conf, added_rows = best
    print(f"\nSeed search results (searched {SEED_SEARCH_MAX} seeds):")
    print(f"  Chosen seed: {chosen_seed}")
    print(f"  Reject recall: {actual_recall:.4f} (target {TARGET_REJECT_RECALL:.4f}, err {recall_err:.4f})")
    print(f"  Mean confidence: {final_mean_conf:.4f} (target {TARGET_CONFIDENCE:.4f}, err {conf_err:.4f})")

    # 5. Load metadata for building paper content
    print("\nLoading massive_metadata_v7_5...")
    ds = load_from_disk(str(METADATA_DIR))
    meta_index = {}
    for i in range(len(ds)):
        meta_index[ds[i]["submission_id"]] = i
    print(f"Indexed {len(meta_index)} submission_ids")

    # 6. Build test samples
    # NO-MODEL variant: base 200 (paper content only) + 133 extra rejects (paper content only)
    test_no_model = dict(base_test_no_model)
    gt_no_model = dict(base_ground_truth)

    # WITH-MODEL variant: rebuild ALL 333 papers with correct vision predictions
    test_with_model = {}
    gt_with_model = dict(base_ground_truth)

    # First, rebuild base 200 for with-model from the no-model base + correct pred_lookup
    for sid, entry in base_test_no_model.items():
        wm_entry = dict(entry)  # copy paper content
        if sid in pred_lookup:
            wm_entry["model_prediction"] = pred_lookup[sid]["model_prediction"]
            wm_entry["model_confidence"] = pred_lookup[sid]["model_confidence"]
        test_with_model[sid] = wm_entry

    # Add 133 extra rejects to both variants
    for row in added_rows:
        sid = row["_metadata"]["submission_id"]
        idx = meta_index[sid]
        paper_entry = build_paper_data(sid, ds[idx]["content_list_json"])

        # No-model: just paper content
        test_no_model[sid] = paper_entry
        gt_no_model[sid] = "reject"

        # With-model: paper content + predictions
        wm_entry = dict(paper_entry)
        wm_entry["model_prediction"] = pred_lookup[sid]["model_prediction"]
        wm_entry["model_confidence"] = pred_lookup[sid]["model_confidence"]
        test_with_model[sid] = wm_entry
        gt_with_model[sid] = "reject"

    # Shuffle both consistently
    shuffled_ids = list(test_no_model.keys())
    random.Random(SHUFFLE_SEED).shuffle(shuffled_ids)

    test_no_model = {sid: test_no_model[sid] for sid in shuffled_ids}
    gt_no_model = {sid: gt_no_model[sid] for sid in shuffled_ids}
    test_with_model = {sid: test_with_model[sid] for sid in shuffled_ids}
    gt_with_model = {sid: gt_with_model[sid] for sid in shuffled_ids}

    # 7. Write base dirs
    for out_dir, test_data, gt_data, gt_path, variant in [
        (OUT_BASE_NO_MODEL, test_no_model, gt_no_model, OUT_GT_NO_MODEL, "no-model"),
        (OUT_BASE_WITH_MODEL, test_with_model, gt_with_model, OUT_GT_WITH_MODEL, "with-model"),
    ]:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Copy train sample from base
        shutil.copy2(BASE_NO_MODEL_DIR / "TRAIN_SAMPLE.json", out_dir / "TRAIN_SAMPLE.json")
        with open(out_dir / "TEST_SAMPLE.json", "w") as f:
            json.dump(test_data, f)
        with open(gt_path, "w") as f:
            json.dump(gt_data, f, indent=2)
        counts = Counter(gt_data.values())
        print(f"\n[{variant}] Wrote {out_dir}/TEST_SAMPLE.json")
        print(f"  Wrote {gt_path}")
        print(f"  Labels: {dict(sorted(counts.items()))}")
        print(f"  Total: {len(test_data)}")

    # 8. Create agent run dirs (copy base + AGENTS.override.md + main.py)
    for base_dir, run_dir, existing_run_dir, variant in [
        (OUT_BASE_NO_MODEL, OUT_RUN_NO_MODEL, EXISTING_RUN_NO_MODEL, "no-model"),
        (OUT_BASE_WITH_MODEL, OUT_RUN_WITH_MODEL, EXISTING_RUN_WITH_MODEL, "with-model"),
    ]:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        shutil.copytree(base_dir, run_dir)
        # Copy agent run files from existing run dirs
        for fname in ["AGENTS.override.md", "main.py"]:
            src = existing_run_dir / fname
            if src.exists():
                shutil.copy2(src, run_dir / fname)
                print(f"[{variant}] Copied {fname} from {existing_run_dir}")
            else:
                print(f"[{variant}] WARNING: {src} not found")

    # 9. Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Verify ALL 333 with-model predictions match vision-aligned pred_lookup
    mismatches = 0
    for sid, entry in test_with_model.items():
        if "model_prediction" in entry:
            expected = pred_lookup.get(sid)
            if expected is None:
                print(f"  MISSING: {sid} not in pred_lookup")
                mismatches += 1
            elif entry["model_prediction"] != expected["model_prediction"]:
                print(f"  MISMATCH: {sid} got {entry['model_prediction']} expected {expected['model_prediction']}")
                mismatches += 1
            elif entry["model_confidence"] != expected["model_confidence"]:
                print(f"  CONF MISMATCH: {sid} got {entry['model_confidence']} expected {expected['model_confidence']}")
                mismatches += 1
    print(f"Prediction alignment check: {mismatches} mismatches (expect 0)")

    # Verify reject recall
    reject_correct = sum(
        1 for sid, label in gt_with_model.items()
        if label == "reject" and test_with_model[sid].get("model_prediction") == "reject"
    )
    reject_total = sum(1 for label in gt_with_model.values() if label == "reject")
    print(f"Reject recall: {reject_correct}/{reject_total} = {reject_correct/reject_total:.4f} (target ~{TARGET_REJECT_RECALL:.4f})")

    # Verify mean confidence
    confs = [entry["model_confidence"] for entry in test_with_model.values() if "model_confidence" in entry]
    mean_conf = sum(confs) / len(confs) if confs else 0
    print(f"Mean confidence: {mean_conf:.4f} (target ~{TARGET_CONFIDENCE:.4f})")

    # Verify counts
    counts = Counter(gt_with_model.values())
    print(f"Total papers: {len(test_with_model)} (expect 333)")
    print(f"Labels: {dict(sorted(counts.items()))} (expect accept=100, reject=233)")

    print("\nDone!")


if __name__ == "__main__":
    main()
