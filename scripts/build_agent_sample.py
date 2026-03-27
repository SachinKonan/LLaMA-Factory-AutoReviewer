"""Build train/test samples for the coding agent benchmark.

Samples from the actual train/test splits so there is zero overlap:
- TRAIN_SAMPLE.json: 400 labeled papers (100 accept + 100 reject per year, 2025+2026)
- TEST_SAMPLE.json:  200 unlabeled papers (50 accept + 50 reject per year, 2025+2026)
- agent_ground_truth.json: labels for the 200 test papers

Usage:
  python scripts/build_agent_sample.py
  python scripts/build_agent_sample.py --mode 2026
  python scripts/build_agent_sample.py --mode 2026_realistic_prior \
      --output-dir coding_agents_2026_codex_realistic_prior_base
  python scripts/build_agent_sample.py --mode 2026_realistic_prior \
      --with-model-prior \
      --output-dir coding_agents_2026_codex_with_best_model_and_model_prior_realistic_prior_base
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

METADATA_DIR = Path("data/massive_metadata_v7_5")
TRAIN_SPLIT = Path(
    "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train/data.json"
)
TEST_SPLIT = Path(
    "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json"
)
VISION_PREDICTIONS = Path(
    "results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl"
)
REALISTIC_BASE_DIR = Path("coding_agents_2026_codex_base")
REALISTIC_PRIOR_BASE_DIR = Path("coding_agents_2026_codex_with_best_model_and_model_prior_base")

# Mode configs: (target_years, train_per_year_per_label, test_per_year_per_label, output_dir)
MODE_CONFIGS = {
    "default": {
        "target_years": [2025, 2026],
        "train_per_label": 100,   # per year -> 4 groups -> 400
        "test_per_label": 50,     # per year -> 4 groups -> 200
        "output_dir": Path("coding_agents"),
    },
    "2026": {
        "target_years": [2026],
        "train_per_label": 200,   # 1 year -> 2 groups -> 400
        "test_per_label": 100,    # 1 year -> 2 groups -> 200
        "output_dir": Path("coding_agents_2026_base"),
    },
    "2026_realistic_prior": {
        "target_years": [2026],
        "output_dir": Path("coding_agents_2026_codex_realistic_prior_base"),
    },
}

SEED = 42  # default mode
SEED_2026 = 83  # 2026 mode: chosen for representative model accuracy (~67% train+test)  # default mode
SEED_2026 = 23  # 2026 mode: chosen for representative model accuracy (~66.5% train+test)
REALISTIC_SHUFFLE_SEED = 20260311
REALISTIC_TARGET_CONFIDENCE = 0.67
REALISTIC_ADDED_REJECTS = 133

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


def sanitize_element(elem):
    """Remove leaky fields and return cleaned element, or None to skip."""
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
    """Extract ordered list of header strings from content elements."""
    return [
        elem["text"]
        for elem in content_list
        if elem.get("text_level") is not None and elem.get("text")
    ]


def load_split_ids(split_path, target_years):
    """Load a split JSON and return {(year, label_lower): [submission_id, ...]}."""
    with open(split_path) as f:
        data = json.load(f)
    groups = {}
    for item in data:
        m = item["_metadata"]
        year = m["year"]
        if year not in target_years:
            continue
        label = m["answer"].lower()  # "Accept" -> "accept"
        groups.setdefault((year, label), []).append(m["submission_id"])
    return groups


def load_split_rows(split_path):
    with open(split_path) as f:
        return json.load(f)


def build_paper_data(sub_id, content_json, label=None):
    """Build sanitized paper entry from raw content JSON."""
    content_list_raw = json.loads(content_json) if isinstance(content_json, str) else content_json
    content_list = [c for elem in content_list_raw if (c := sanitize_element(elem)) is not None]
    headers = extract_headers(content_list)
    entry = {"content_list": content_list, "headers": headers}
    if label is not None:
        entry["label"] = label
    return entry


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


def load_prediction_lookup():
    split_rows = load_split_rows(TEST_SPLIT)
    predictions = []
    with open(VISION_PREDICTIONS) as f:
        for line in f:
            predictions.append(json.loads(line))

    paired = list(zip(split_rows, predictions))
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

    return pred_lookup


def load_realistic_core(with_model_prior):
    source_dir = REALISTIC_PRIOR_BASE_DIR if with_model_prior else REALISTIC_BASE_DIR
    with open(source_dir / "TRAIN_SAMPLE.json") as f:
        train_sample = json.load(f)
    with open(source_dir / "TEST_SAMPLE.json") as f:
        test_sample = json.load(f)
    with open(Path("data") / "agent_ground_truth_2026.json") as f:
        ground_truth = json.load(f)
    return train_sample, test_sample, ground_truth


def build_realistic_prior_samples(output_dir, with_model_prior, seed_search_max):
    print(f"Building additive realistic-prior benchmark -> {output_dir}")
    print(f"with_model_prior={with_model_prior} target_confidence={REALISTIC_TARGET_CONFIDENCE:.4f}")
    train_sample, base_test_sample, base_ground_truth = load_realistic_core(with_model_prior)

    split_rows = load_split_rows(TEST_SPLIT)
    all_2026_reject_rows = [
        row for row in split_rows
        if row["_metadata"]["year"] == 2026 and row["_metadata"]["answer"].lower() == "reject"
    ]

    existing_ids = set(train_sample) | set(base_test_sample)
    reject_pool_rows = [
        row for row in all_2026_reject_rows
        if row["_metadata"]["submission_id"] not in existing_ids
    ]
    assert len(reject_pool_rows) >= REALISTIC_ADDED_REJECTS, (
        f"Not enough extra rejects: need {REALISTIC_ADDED_REJECTS}, have {len(reject_pool_rows)}"
    )

    pred_lookup = load_prediction_lookup() if with_model_prior else {}
    if with_model_prior:
        base_conf_sum = sum(item["model_confidence"] for item in base_test_sample.values())
        best = None
        for seed in range(seed_search_max):
            rng = random.Random(seed)
            sampled_rows = rng.sample(reject_pool_rows, REALISTIC_ADDED_REJECTS)
            sampled_ids = [row["_metadata"]["submission_id"] for row in sampled_rows]
            if not all(sid in pred_lookup for sid in sampled_ids):
                continue
            added_conf_sum = sum(pred_lookup[sid]["model_confidence"] for sid in sampled_ids)
            final_mean_conf = (base_conf_sum + added_conf_sum) / (len(base_test_sample) + len(sampled_ids))
            distance = abs(final_mean_conf - REALISTIC_TARGET_CONFIDENCE)
            candidate = (distance, seed, final_mean_conf, sampled_rows)
            if best is None or candidate < best:
                best = candidate
        assert best is not None, "Could not find a reject-only sample with prediction metadata"
        _, chosen_seed, final_mean_conf, added_rows = best
        print(
            f"Seed search: tried {seed_search_max} seeds, chose seed={chosen_seed}, "
            f"final_mean_conf={final_mean_conf:.4f}, abs_err={abs(final_mean_conf - REALISTIC_TARGET_CONFIDENCE):.4f}"
        )
    else:
        rng = random.Random(SEED_2026)
        added_rows = rng.sample(reject_pool_rows, REALISTIC_ADDED_REJECTS)
        chosen_seed = SEED_2026
        print(f"Using seed={chosen_seed} for no-prior additive reject sample")

    ds = load_from_disk(str(METADATA_DIR))
    meta_index = {}
    for i in range(len(ds)):
        meta_index[ds[i]["submission_id"]] = i

    test_sample = dict(base_test_sample)
    ground_truth = dict(base_ground_truth)
    for row in added_rows:
        submission_id = row["_metadata"]["submission_id"]
        idx = meta_index[submission_id]
        entry = build_paper_data(submission_id, ds[idx]["content_list_json"])
        if with_model_prior:
            entry.update(pred_lookup[submission_id])
        test_sample[submission_id] = entry
        ground_truth[submission_id] = "reject"

    shuffled_ids = list(test_sample.keys())
    random.Random(REALISTIC_SHUFFLE_SEED).shuffle(shuffled_ids)
    shuffled_test_sample = {sid: test_sample[sid] for sid in shuffled_ids}
    shuffled_ground_truth = {sid: ground_truth[sid] for sid in shuffled_ids}

    output_dir.mkdir(parents=True, exist_ok=True)
    output_train = output_dir / "TRAIN_SAMPLE.json"
    output_test = output_dir / "TEST_SAMPLE.json"
    output_ground_truth = Path("data") / (
        "agent_ground_truth_2026_realistic_prior_with_model.json"
        if with_model_prior else
        "agent_ground_truth_2026_realistic_prior.json"
    )

    with open(output_train, "w") as f:
        json.dump(train_sample, f)
    with open(output_test, "w") as f:
        json.dump(shuffled_test_sample, f)
    with open(output_ground_truth, "w") as f:
        json.dump(shuffled_ground_truth, f, indent=2)

    counts = Counter(shuffled_ground_truth.values())
    print(f"Wrote {output_train}")
    print(f"Wrote {output_test}")
    print(f"Wrote {output_ground_truth}")
    print(f"Shuffle seed: {REALISTIC_SHUFFLE_SEED}")
    print(f"Test labels: {dict(sorted(counts.items()))}")
    print(f"Total test papers: {len(shuffled_test_sample)}")


def main():
    parser = argparse.ArgumentParser(description="Build agent benchmark samples")
    parser.add_argument("--mode", choices=list(MODE_CONFIGS.keys()), default="default",
                        help="Sample mode: 'default' (2025+2026) or '2026' (2026-only)")
    parser.add_argument(
        "--with-model-prior",
        action="store_true",
        help="For realistic-prior mode, include model_prediction/model_confidence in TEST_SAMPLE.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the configured output directory.",
    )
    parser.add_argument(
        "--seed-search-max",
        type=int,
        default=5000,
        help="For realistic-prior + model-prior mode, search seeds in range [0, N).",
    )
    args = parser.parse_args()

    cfg = MODE_CONFIGS[args.mode]
    target_years = cfg["target_years"]
    output_dir = args.output_dir or cfg["output_dir"]

    if args.mode == "2026_realistic_prior":
        build_realistic_prior_samples(
            output_dir=output_dir,
            with_model_prior=args.with_model_prior,
            seed_search_max=args.seed_search_max,
        )
        return

    train_per_label = cfg["train_per_label"]
    test_per_label = cfg["test_per_label"]

    output_train = output_dir / "TRAIN_SAMPLE.json"
    output_test = output_dir / "TEST_SAMPLE.json"
    output_ground_truth = Path("data") / f"agent_ground_truth{'_2026' if args.mode == '2026' else ''}.json"

    print(f"Mode: {args.mode}  Years: {target_years}")
    print(f"Train: {train_per_label} per label per year  Test: {test_per_label} per label per year")
    print(f"Output dir: {output_dir}")

    # Load metadata and index by submission_id
    print("\nLoading massive_metadata_v7_5...")
    ds = load_from_disk(str(METADATA_DIR))
    print(f"Total rows: {len(ds)}")

    meta_index = {}
    for i in range(len(ds)):
        meta_index[ds[i]["submission_id"]] = i
    print(f"Indexed {len(meta_index)} unique submission_ids")

    # Load splits
    print("Loading train/test splits...")
    if args.mode == "2026":
        # For 2026-only mode, draw BOTH agent train and test from the model's test split
        # so that model inference predictions are available for all papers.
        all_groups = load_split_ids(TEST_SPLIT, target_years)
        print(f"\nTest split pool ({'/'.join(str(y) for y in target_years)}):")
        for k in sorted(all_groups.keys()):
            print(f"  {k}: {len(all_groups[k])}")
    else:
        train_groups = load_split_ids(TRAIN_SPLIT, target_years)
        test_groups = load_split_ids(TEST_SPLIT, target_years)
        print(f"\nTrain split ({'/'.join(str(y) for y in target_years)}):")
        for k in sorted(train_groups.keys()):
            print(f"  {k}: {len(train_groups[k])}")
        print(f"Test split ({'/'.join(str(y) for y in target_years)}):")
        for k in sorted(test_groups.keys()):
            print(f"  {k}: {len(test_groups[k])}")

    # Sample
    rng = random.Random(SEED_2026 if args.mode == "2026" else SEED)
    train_ids = {}  # sub_id -> label
    test_ids = {}

    if args.mode == "2026":
        # Both train and test sampled from the same pool (model's test split),
        # with no overlap between them.
        for year in target_years:
            for label in ["accept", "reject"]:
                key = (year, label)
                pool = all_groups[key]
                needed = train_per_label + test_per_label
                assert len(pool) >= needed, f"Not enough for {key}: need {needed}, have {len(pool)}"
                sampled = rng.sample(pool, needed)
                for sid in sampled[:train_per_label]:
                    train_ids[sid] = label
                for sid in sampled[train_per_label:]:
                    test_ids[sid] = label
    else:
        for year in target_years:
            for label in ["accept", "reject"]:
                key = (year, label)
                # Train
                pool = train_groups[key]
                assert len(pool) >= train_per_label, f"Not enough train for {key}: {len(pool)}"
                for sid in rng.sample(pool, train_per_label):
                    train_ids[sid] = label
                # Test
                pool = test_groups[key]
                assert len(pool) >= test_per_label, f"Not enough test for {key}: {len(pool)}"
                for sid in rng.sample(pool, test_per_label):
                    test_ids[sid] = label

    assert not (set(train_ids) & set(test_ids)), "Train/test overlap!"
    print(f"\nSampled {len(train_ids)} train, {len(test_ids)} test (0 overlap)")

    # Build outputs
    print("Building TRAIN_SAMPLE.json...")
    train_sample = {}
    for sub_id, label in train_ids.items():
        idx = meta_index[sub_id]
        train_sample[sub_id] = build_paper_data(sub_id, ds[idx]["content_list_json"], label=label)

    print("Building TEST_SAMPLE.json...")
    test_sample = {}
    ground_truth = {}
    for sub_id, label in test_ids.items():
        idx = meta_index[sub_id]
        test_sample[sub_id] = build_paper_data(sub_id, ds[idx]["content_list_json"])
        ground_truth[sub_id] = label

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_train, "w") as f:
        json.dump(train_sample, f)
    print(f"Wrote {output_train} ({output_train.stat().st_size / 1e6:.1f} MB)")

    with open(output_test, "w") as f:
        json.dump(test_sample, f)
    print(f"Wrote {output_test} ({output_test.stat().st_size / 1e6:.1f} MB)")

    with open(output_ground_truth, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Wrote {output_ground_truth}")

    # Summary
    train_labels = Counter(train_ids.values())
    test_labels = Counter(test_ids.values())
    print(f"\n--- Summary ---")
    print(f"Train: {len(train_sample)} papers — {dict(sorted(train_labels.items()))}")
    print(f"Test:  {len(test_sample)} papers — {dict(sorted(test_labels.items()))}")


if __name__ == "__main__":
    main()
