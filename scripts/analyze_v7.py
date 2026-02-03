#!/usr/bin/env python3
"""
Analyze v7 sweep results with per-year breakdown.

Shows combined tables across all checkpoints:
1. Per-year metrics
2. Overall metrics

Each cell shows values from all checkpoints as: ckpt1/ckpt2/ckpt3...

Usage:
    python scripts/analyze_v7.py results/final_sweep_v7/balanced_clean/
    python scripts/analyze_v7.py --all  # Analyze all available results
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


# Dataset mappings: short_name -> dataset_name
DATASET_MAPPINGS = {
    # Text-only
    "balanced_clean": "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7",
    "balanced_title_abstract": "iclr_2020_2025_85_5_10_split7_balanced_clean_title_abstract_binary_noreviews_v7",
    "balanced_trainagreeing": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7",
    "balanced_2024_2025": "iclr_2024_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7",
    "balanced_2017_2025": "iclr_2017_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7",
    # Vision
    "balanced_vision": "iclr_2020_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_trainagreeing_vision": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "balanced_2024_2025_vision": "iclr_2024_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_2017_2025_vision": "iclr_2017_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_no2024_vision": "iclr_2020_2023_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_trainagreeing_no2024_vision": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    # Images
    "balanced_clean_images": "iclr_2020_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7",
    "balanced_trainagreeing_images": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_images_binary_noreviews_v7",
    "balanced_2024_2025_images": "iclr_2024_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7",
    "balanced_2017_2025_images": "iclr_2017_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7",
}

DATA_DIR = Path("data")


def load_predictions(pred_file: Path) -> list[dict]:
    """Load predictions from jsonl file."""
    predictions = []
    with open(pred_file) as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def load_test_dataset(dataset_name: str) -> list[dict]:
    """Load test dataset with metadata."""
    path = DATA_DIR / f"{dataset_name}_test" / "data.json"
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")

    with open(path) as f:
        return json.load(f)


def extract_prediction(text: str) -> str:
    """Extract Accept/Reject from model prediction."""
    text_lower = text.lower()

    # Look for boxed answers first
    if "\\boxed{accept}" in text_lower or "boxed{accept}" in text_lower:
        return "accept"
    if "\\boxed{reject}" in text_lower or "boxed{reject}" in text_lower:
        return "reject"

    # Fallback: look for accept/reject keywords
    if "accept" in text_lower and "reject" not in text_lower:
        return "accept"
    if "reject" in text_lower and "accept" not in text_lower:
        return "reject"

    # If both or neither, check which comes last (final answer)
    accept_pos = text_lower.rfind("accept")
    reject_pos = text_lower.rfind("reject")

    if accept_pos > reject_pos:
        return "accept"
    elif reject_pos > accept_pos:
        return "reject"

    return "unknown"


def extract_label(text: str) -> str:
    """Extract Accept/Reject from ground truth label."""
    text_lower = text.lower().strip()
    if "accept" in text_lower:
        return "accept"
    if "reject" in text_lower:
        return "reject"
    return "unknown"


def compute_stats(predictions: list[dict], test_data: list[dict]) -> dict:
    """Compute per-year statistics."""
    if len(predictions) != len(test_data):
        print(f"Warning: predictions ({len(predictions)}) != test_data ({len(test_data)})")
        min_len = min(len(predictions), len(test_data))
        predictions = predictions[:min_len]
        test_data = test_data[:min_len]

    year_stats = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0})

    for pred, data in zip(predictions, test_data):
        metadata = data.get("_metadata", {})
        year = metadata.get("year")
        if year is None:
            continue

        pred_label = extract_prediction(pred.get("predict", ""))
        true_label = extract_label(pred.get("label", ""))

        if pred_label == "unknown" or true_label == "unknown":
            continue

        year_stats[year]["total"] += 1

        if true_label == "accept":
            if pred_label == "accept":
                year_stats[year]["tp"] += 1
            else:
                year_stats[year]["fn"] += 1
        else:
            if pred_label == "reject":
                year_stats[year]["tn"] += 1
            else:
                year_stats[year]["fp"] += 1

    return dict(year_stats)


def calc_metrics(stats: dict) -> dict:
    """Calculate accuracy, accept recall, reject recall from stats."""
    tp = stats.get("tp", 0)
    tn = stats.get("tn", 0)
    fp = stats.get("fp", 0)
    fn = stats.get("fn", 0)
    total = stats.get("total", 0)

    if total == 0:
        return {"accuracy": None, "accept_recall": None, "reject_recall": None, "n": 0}

    accuracy = (tp + tn) / total * 100 if total > 0 else None
    accept_recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else None
    reject_recall = tn / (tn + fp) * 100 if (tn + fp) > 0 else None

    return {
        "accuracy": accuracy,
        "accept_recall": accept_recall,
        "reject_recall": reject_recall,
        "n": total,
    }


def aggregate_stats(year_stats: dict, years: list[int]) -> dict:
    """Aggregate stats across specified years."""
    agg = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0}
    for year in years:
        if year in year_stats:
            stats = year_stats[year]
            agg["tp"] += stats["tp"]
            agg["tn"] += stats["tn"]
            agg["fp"] += stats["fp"]
            agg["fn"] += stats["fn"]
            agg["total"] += stats["total"]
    return agg


def format_combined_value(values: list[Optional[float]], fmt: str = ".1f") -> str:
    """Format multiple values as val1/val2/val3."""
    formatted = []
    for v in values:
        if v is None:
            formatted.append("-")
        else:
            formatted.append(f"{v:{fmt}}")
    return "/".join(formatted)


def get_ckpt_short_name(filename: str) -> str:
    """Extract short checkpoint name from filename."""
    match = re.search(r"ckpt-(\d+)", filename)
    if match:
        return match.group(1)
    return filename.replace(".jsonl", "")


def print_combined_per_year_table(all_year_stats: list[dict], ckpt_names: list[str]):
    """Print combined per-year metrics table."""
    print("\n--- Per-Year Metrics ---")
    print(f"Checkpoints: {' / '.join(ckpt_names)}")

    # Determine which years have data
    all_years = set()
    for year_stats in all_year_stats:
        all_years.update(year_stats.keys())
    years = sorted(all_years)

    if not years:
        print("No data available.")
        return

    # Compute metrics for each checkpoint and year
    all_metrics = []
    for year_stats in all_year_stats:
        metrics_by_year = {}
        for y in years:
            if y in year_stats:
                metrics_by_year[y] = calc_metrics(year_stats[y])
            else:
                metrics_by_year[y] = calc_metrics({})
        all_metrics.append(metrics_by_year)

    # Calculate column width
    num_ckpts = len(ckpt_names)
    col_width = max(8, num_ckpts * 5 + 2)

    # Header
    year_cols = [str(y) for y in years]
    header = f"{'Metric':<20}" + "".join(f"{y:>{col_width}}" for y in year_cols)
    print(header)
    print("-" * len(header))

    # Accuracy
    row = f"{'Accuracy (%)':<20}"
    for y in years:
        values = [m[y]["accuracy"] for m in all_metrics]
        row += f"{format_combined_value(values):>{col_width}}"
    print(row)

    # Accept Recall
    row = f"{'Accept Recall (%)':<20}"
    for y in years:
        values = [m[y]["accept_recall"] for m in all_metrics]
        row += f"{format_combined_value(values):>{col_width}}"
    print(row)

    # Reject Recall
    row = f"{'Reject Recall (%)':<20}"
    for y in years:
        values = [m[y]["reject_recall"] for m in all_metrics]
        row += f"{format_combined_value(values):>{col_width}}"
    print(row)

    # N
    row = f"{'N':<20}"
    for y in years:
        val = all_metrics[0][y]["n"]
        row += f"{val:>{col_width}}"
    print(row)


def print_combined_overall_table(all_year_stats: list[dict], ckpt_names: list[str]):
    """Print combined overall aggregated metrics."""
    print("\n--- Overall Metrics ---")
    print(f"Checkpoints: {' / '.join(ckpt_names)}")

    # Compute metrics for each checkpoint
    all_metrics = []
    for year_stats in all_year_stats:
        all_years = sorted(year_stats.keys())
        stats_all = aggregate_stats(year_stats, all_years)
        all_metrics.append(calc_metrics(stats_all))

    values = [m["accuracy"] for m in all_metrics]
    print(f"{'Accuracy (%):':<20}{format_combined_value(values)}")

    values = [m["accept_recall"] for m in all_metrics]
    print(f"{'Accept Recall (%):':<20}{format_combined_value(values)}")

    values = [m["reject_recall"] for m in all_metrics]
    print(f"{'Reject Recall (%):':<20}{format_combined_value(values)}")

    print(f"{'N:':<20}{all_metrics[0]['n']}")


def analyze_directory(result_dir: Path, data_root: Path):
    """Analyze all checkpoints in a result directory with combined tables."""
    short_name = result_dir.name

    if short_name not in DATASET_MAPPINGS:
        print(f"Error: Unknown dataset '{short_name}'")
        print(f"  Extracted short_name: {short_name}")
        return

    dataset_name = DATASET_MAPPINGS[short_name]
    data_dir = data_root / f"{dataset_name}_test"

    if not data_dir.exists():
        print(f"Error: Test data directory not found: {data_dir}")
        return

    # Find all checkpoint files, sorted numerically
    def ckpt_sort_key(f):
        match = re.search(r"ckpt-(\d+)", f.name)
        return int(match.group(1)) if match else float('inf')

    ckpt_files = sorted(result_dir.glob("*.jsonl"), key=ckpt_sort_key)

    if not ckpt_files:
        print(f"No checkpoint files found in {result_dir}")
        return

    # Load test data once
    try:
        test_data = load_test_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # Load and compute stats for all checkpoints
    all_year_stats = []
    ckpt_names = []
    for ckpt_file in ckpt_files:
        try:
            predictions = load_predictions(ckpt_file)
            year_stats = compute_stats(predictions, test_data)
            all_year_stats.append(year_stats)
            ckpt_names.append(get_ckpt_short_name(ckpt_file.name))
        except Exception as e:
            print(f"Error loading {ckpt_file}: {e}")
            continue

    if not all_year_stats:
        print(f"No valid checkpoints found in {result_dir}")
        return

    # Print header
    print("=" * 80)
    print(f"{short_name}")
    print(f"Checkpoints: {', '.join([f.name for f in ckpt_files])}")
    print("=" * 80)

    # Print combined tables
    print_combined_per_year_table(all_year_stats, ckpt_names)
    print_combined_overall_table(all_year_stats, ckpt_names)
    print()


def analyze_all(data_root: Path):
    """Analyze all result directories."""
    results_dirs = [
        Path("results/final_sweep_v7"),
        Path("results/final_sweep_v7_pli"),
    ]

    for results_dir in results_dirs:
        if not results_dir.exists():
            continue

        print(f"\nScanning: {results_dir}")

        for short_name in sorted(DATASET_MAPPINGS.keys()):
            result_dir = results_dir / short_name
            if result_dir.exists():
                analyze_directory(result_dir, data_root)


def main():
    parser = argparse.ArgumentParser(description="Analyze v7 sweep results with per-year breakdown")
    parser.add_argument("path", nargs="?", type=Path, help="Result directory")
    parser.add_argument("--all", action="store_true", help="Analyze all available results")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for test data (default: data)",
    )
    args = parser.parse_args()

    if args.all:
        analyze_all(args.data_root)
    elif args.path:
        if args.path.is_dir():
            analyze_directory(args.path, args.data_root)
        else:
            print(f"Error: {args.path} is not a directory")
            print("Please provide a directory path (e.g., results/final_sweep_v7/balanced_clean/)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
