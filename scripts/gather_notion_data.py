#!/usr/bin/env python3
"""
Gather experiment status, timing, and per-year accuracy data for Notion tables.
Outputs JSON for each experiment group.
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SAVES_ROOT = Path("saves/final_sweep_v7_datasweepv3")
RESULTS_ROOT = Path("results/final_sweep_v7_datasweepv3")
DATA_DIR = Path("data")

# Map experiment group -> list of variant dirs
EXPERIMENT_GROUPS = {
    "optim_search": SAVES_ROOT / "optim_search",
    "optim_search_6epochs": SAVES_ROOT / "optim_search_6epochs",
    "wd_sweep": SAVES_ROOT / "wd_sweep",
    "wd_sweep_expdecay": SAVES_ROOT / "wd_sweep_expdecay",
    "wd_sweep_2epoch": SAVES_ROOT / "wd_sweep_2epoch",
    "wd_sweep_2epoch_3epochexp": SAVES_ROOT / "wd_sweep_2epoch_3epochexp",
}

# Dataset mapping for per-year analysis
DATASET_MAP = {
    # Text variants
    "text": "iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered",
    "vision": "iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered",
}


def get_mtime_hours(path):
    """Get modification time as hours-ago from a reference, or just epoch timestamp."""
    try:
        return os.path.getmtime(str(path))
    except:
        return None


def read_touch(path):
    """Read touch file content (typically a job ID or result path)."""
    try:
        return Path(path).read_text().strip()
    except:
        return None


def time_diff_hours(t1, t2):
    """Compute time difference in hours between two epoch timestamps."""
    if t1 is None or t2 is None:
        return None
    return round(abs(t2 - t1) / 3600, 1)


def extract_prediction(text):
    text_lower = text.lower().strip()
    if text_lower == "y": return "accept"
    if text_lower == "n": return "reject"
    if "\\boxed{accept}" in text_lower or "boxed{accept}" in text_lower: return "accept"
    if "\\boxed{reject}" in text_lower or "boxed{reject}" in text_lower: return "reject"
    if "\\boxed{yes}" in text_lower or "boxed{yes}" in text_lower or "\\boxed{y}" in text_lower: return "accept"
    if "\\boxed{no}" in text_lower or "boxed{no}" in text_lower or "\\boxed{n}" in text_lower: return "reject"
    yes_pos = text_lower.rfind("yes")
    no_pos = text_lower.rfind("no")
    accept_pos = text_lower.rfind("accept")
    reject_pos = text_lower.rfind("reject")
    positions = [(p, l) for p, l in [(yes_pos,"accept"),(no_pos,"reject"),(accept_pos,"accept"),(reject_pos,"reject")] if p != -1]
    if positions:
        positions.sort(key=lambda x: x[0])
        return positions[-1][1]
    return "unknown"


def extract_label(text):
    text_lower = text.lower().strip()
    if text_lower == "y": return "accept"
    if text_lower == "n": return "reject"
    if "accept" in text_lower or "yes" == text_lower: return "accept"
    if "reject" in text_lower or "no" == text_lower: return "reject"
    return "unknown"


def compute_per_year_accuracy(pred_file, test_data):
    """Compute per-year accuracy from predictions and test data."""
    predictions = []
    with open(pred_file) as f:
        for line in f:
            predictions.append(json.loads(line))

    min_len = min(len(predictions), len(test_data))
    predictions = predictions[:min_len]
    test_items = test_data[:min_len]

    year_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, pred in enumerate(predictions):
        metadata = test_items[i].get("_metadata", {})
        year = metadata.get("year")
        if year is None:
            continue
        pred_label = extract_prediction(pred.get("predict", ""))
        true_label = extract_label(pred.get("label", ""))
        if pred_label == "unknown" or true_label == "unknown":
            continue
        year_stats[year]["total"] += 1
        if pred_label == true_label:
            year_stats[year]["correct"] += 1

    result = {}
    overall_correct = 0
    overall_total = 0
    for year, stats in sorted(year_stats.items()):
        acc = round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] > 0 else None
        result[str(year)] = {"accuracy": acc, "n": stats["total"]}
        overall_correct += stats["correct"]
        overall_total += stats["total"]

    result["overall"] = {
        "accuracy": round(overall_correct / overall_total * 100, 1) if overall_total > 0 else None,
        "n": overall_total
    }
    return result


def detect_modality(variant_name):
    """Detect if variant is text or vision."""
    if "vision" in variant_name:
        return "vision"
    return "text"


def gather_variant_data(variant_dir, results_dir):
    """Gather complete status data for one variant."""
    variant_name = variant_dir.name
    data = {
        "variant": variant_name,
        "modality": detect_modality(variant_name),
        "checkpoints": {},
    }

    # Train job ID
    trainjob_touch = variant_dir / ".trainjob.touch"
    data["train_job_id"] = read_touch(trainjob_touch)

    # Training start time (from .trainjob.touch modification time)
    train_start = get_mtime_hours(trainjob_touch)

    # Find checkpoints
    ckpt_dirs = sorted(
        [d for d in variant_dir.glob("checkpoint-*") if d.is_dir()],
        key=lambda d: int(d.name.split("-")[1])
    )

    prev_ckpt_time = train_start

    for ckpt_dir in ckpt_dirs:
        step = int(ckpt_dir.name.split("-")[1])
        ckpt_data = {"step": step}

        # Train status
        ckpt_mtime = get_mtime_hours(ckpt_dir / "config.json")  # Use config.json as proxy
        if ckpt_mtime is None:
            ckpt_mtime = get_mtime_hours(ckpt_dir)
        ckpt_data["train_done"] = ckpt_dir.exists()
        ckpt_data["train_time_h"] = time_diff_hours(prev_ckpt_time, ckpt_mtime)
        prev_ckpt_time = ckpt_mtime

        # Train-infer (training accuracy callback)
        train_result_file = results_dir / f"train-ckpt-{step}.json"
        if train_result_file.exists():
            try:
                train_result = json.loads(train_result_file.read_text())
                ckpt_data["train_infer_done"] = True
                ckpt_data["train_infer_accuracy"] = round(train_result.get("sft_accuracy", 0) * 100, 1)
                runtime = train_result.get("eval_runtime_seconds", train_result.get("eval_runtime"))
                ckpt_data["train_infer_time_h"] = round(runtime / 3600, 1) if runtime else None
            except:
                ckpt_data["train_infer_done"] = False
                ckpt_data["train_infer_time_h"] = None
        else:
            ckpt_data["train_infer_done"] = False
            ckpt_data["train_infer_time_h"] = None

        # Test-infer
        infer_touch = ckpt_dir / ".infer.touch"
        inferdone_touch = ckpt_dir / ".inferdone.touch"
        ckpt_data["test_infer_job_id"] = read_touch(infer_touch)

        if inferdone_touch.exists():
            ckpt_data["test_infer_done"] = True
            infer_start = get_mtime_hours(infer_touch)
            infer_end = get_mtime_hours(inferdone_touch)
            ckpt_data["test_infer_time_h"] = time_diff_hours(infer_start, infer_end)
        elif infer_touch.exists():
            ckpt_data["test_infer_done"] = False  # Running
            ckpt_data["test_infer_time_h"] = None
            ckpt_data["test_infer_running"] = True
        else:
            ckpt_data["test_infer_done"] = False
            ckpt_data["test_infer_time_h"] = None

        # Cleaned
        ckpt_data["cleaned"] = (ckpt_dir / ".cleaned.touch").exists()

        # Per-year test accuracy
        pred_file = results_dir / f"finetuned-ckpt-{step}.jsonl"
        if pred_file.exists():
            ckpt_data["has_test_results"] = True
        else:
            ckpt_data["has_test_results"] = False

        data["checkpoints"][str(step)] = ckpt_data

    return data


def gather_per_year_results(experiment_group, results_base_dir):
    """Gather per-year accuracy results for all variants in an experiment group."""
    results = {}

    for variant_dir in sorted(results_base_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant_name = variant_dir.name
        if variant_name.startswith("_"):
            continue

        modality = detect_modality(variant_name)
        dataset_name = DATASET_MAP.get(modality)
        if not dataset_name:
            continue

        test_data_path = DATA_DIR / f"{dataset_name}_test" / "data.json"
        if not test_data_path.exists():
            continue

        try:
            test_data = json.loads(test_data_path.read_text())
        except:
            continue

        variant_results = {}
        for pred_file in sorted(variant_dir.glob("finetuned-ckpt-*.jsonl"),
                                 key=lambda f: int(re.search(r"ckpt-(\d+)", f.name).group(1))):
            step = re.search(r"ckpt-(\d+)", pred_file.name).group(1)
            try:
                variant_results[step] = compute_per_year_accuracy(pred_file, test_data)
            except Exception as e:
                print(f"  Error computing accuracy for {pred_file}: {e}", file=sys.stderr)

        if variant_results:
            results[variant_name] = variant_results

    return results


def main():
    for group_name, saves_dir in EXPERIMENT_GROUPS.items():
        if not saves_dir.exists():
            print(f"Skipping {group_name}: saves dir not found", file=sys.stderr)
            continue

        results_dir = RESULTS_ROOT / group_name
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"Processing: {group_name}", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)

        group_data = {
            "experiment": group_name,
            "variants": [],
            "per_year_results": {},
        }

        # Gather per-variant checkpoint status
        for variant_dir in sorted(saves_dir.iterdir()):
            if not variant_dir.is_dir() or variant_dir.name.startswith("."):
                continue
            variant_results_dir = results_dir / variant_dir.name
            print(f"  Variant: {variant_dir.name}", file=sys.stderr)
            variant_data = gather_variant_data(variant_dir, variant_results_dir)
            group_data["variants"].append(variant_data)

        # Gather per-year accuracy results
        if results_dir.exists():
            group_data["per_year_results"] = gather_per_year_results(group_name, results_dir)

        # Output JSON for this group
        print(f"=== {group_name} ===")
        print(json.dumps(group_data, indent=2))
        print()


if __name__ == "__main__":
    main()
