#!/usr/bin/env python3
"""
Analyze testing_v8 results.

Shows combined tables across all checkpoints:
1. Per-year metrics (2020-2026)
2. Year group metrics (<=2024 vs >2024)
3. Overall metrics

Each cell shows values from all checkpoints as: ckpt1/ckpt2/ckpt3...
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Dataset mappings
DATASET_MAPPINGS = {
    # Binary
    "26ood_vision": "2020_2025_train_2020_2026_valtest_vision_binary_noreviews_v8",
    "26ood_rm24_vision": "2020_2023_2025_train_2020_2026_valtest_vision_binary_noreviews_v8",
    "balanced_vision": "2020_2026_balanced_vision_binary_noreviews_v8",
    "26ood_vision_bz32": "2020_2025_train_2020_2026_valtest_vision_binary_noreviews_v8",
    "26ood_text": "2020_2025_train_2020_2026_valtest_clean_binary_noreviews_v8",
    "26ood_rm24_text": "2020_2023_2025_train_2020_2026_valtest_clean_binary_noreviews_v8",
    "balanced_text": "2020_2026_balanced_clean_binary_noreviews_v8",
    "26ood_text_bz32": "2020_2025_train_2020_2026_valtest_clean_binary_noreviews_v8",
    # Ratingbinary
    "ood_text_rating": "2020_2025_train_2020_2026_valtest_clean_ratingbinary_noreviews_v8",
    "balanced_text_rating": "2020_2026_balanced_clean_ratingbinary_noreviews_v8",
    "ood_vision_rating": "2020_2025_train_2020_2026_valtest_vision_ratingbinary_noreviews_v8",
    "balanced_vision_rating": "2020_2026_balanced_vision_ratingbinary_noreviews_v8",
    # V7 no2024 vision
    "balanced_no2024_vision": "iclr_2020_2023_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_trainagreeing_no2024_vision": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
}


def extract_binary(text: str) -> Optional[str]:
    """Extract Accept/Reject from \\boxed{Accept} or \\boxed{Reject}."""
    match = re.search(r"\\boxed\{(Accept|Reject)\}", text)
    if match:
        return match.group(1)
    return None


def extract_ratingbinary(text: str) -> Tuple[Optional[float], Optional[str]]:
    """Extract (rating, binary) from \\boxed{0.6,Accept}."""
    match = re.search(r"\\boxed\{([0-9.]+)\s*,\s*(Accept|Reject)\}", text)
    if match:
        try:
            rating = float(match.group(1))
            binary = match.group(2)
            return rating, binary
        except ValueError:
            return None, None
    return None, None


def is_ratingbinary(short_name: str) -> bool:
    """Return True if the dataset is ratingbinary format."""
    return "rating" in short_name


def load_predictions(pred_file: str) -> List[Dict]:
    """Load predictions from a JSONL file."""
    predictions = []
    with open(pred_file, "r") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def load_test_data(data_dir: str) -> List[Dict]:
    """Load test data from JSON file."""
    data_file = os.path.join(data_dir, "data.json")
    with open(data_file, "r") as f:
        return json.load(f)


def compute_stats(
    predictions: List[Dict], test_data: List[Dict], is_rating: bool
) -> Dict:
    """
    Compute per-year statistics.

    Returns: {year: {tp, tn, fp, fn, total, rating_errors: []}}
    """
    if len(predictions) != len(test_data):
        print(f"Warning: predictions ({len(predictions)}) != test_data ({len(test_data)})")

    year_stats = defaultdict(lambda: {
        "tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0, "rating_errors": []
    })

    for i, (pred_item, test_item) in enumerate(zip(predictions, test_data)):
        metadata = test_item.get("_metadata", {})
        year = metadata.get("year")
        gt_answer_raw = metadata.get("answer")
        gt_rating = metadata.get("pct_rating")

        if year is None:
            continue

        pred_text = pred_item.get("predict", "")

        if is_rating:
            pred_rating, pred_binary = extract_ratingbinary(pred_text)
            # For ratingbinary, gt_answer is "0.9,Accept" format
            if gt_answer_raw and "," in str(gt_answer_raw):
                gt_binary = gt_answer_raw.split(",")[-1].strip()
            else:
                gt_binary = gt_answer_raw
        else:
            pred_binary = extract_binary(pred_text)
            pred_rating = None
            gt_binary = gt_answer_raw

        stats = year_stats[year]
        stats["total"] += 1

        if pred_binary is None:
            # Can't parse prediction, count as wrong
            if gt_binary == "Accept":
                stats["fn"] += 1
            else:
                stats["fp"] += 1
            continue

        # Binary classification metrics
        if gt_binary == "Accept" and pred_binary == "Accept":
            stats["tp"] += 1
        elif gt_binary == "Reject" and pred_binary == "Reject":
            stats["tn"] += 1
        elif gt_binary == "Accept" and pred_binary == "Reject":
            stats["fn"] += 1
        else:  # gt_binary == "Reject" and pred_binary == "Accept"
            stats["fp"] += 1

        # Rating error
        if is_rating and pred_rating is not None and gt_rating is not None:
            stats["rating_errors"].append(abs(pred_rating - gt_rating))

    return dict(year_stats)


def calc_metrics(stats: Dict) -> Dict:
    """Calculate accuracy, accept recall, reject recall, and rating MAE from stats."""
    tp = stats.get("tp", 0)
    tn = stats.get("tn", 0)
    fp = stats.get("fp", 0)
    fn = stats.get("fn", 0)
    total = stats.get("total", 0)
    rating_errors = stats.get("rating_errors", [])

    if total == 0:
        return {
            "accuracy": None,
            "accept_recall": None,
            "reject_recall": None,
            "rating_mae": None,
            "n": 0
        }

    accuracy = (tp + tn) / total * 100 if total > 0 else None
    accept_recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else None
    reject_recall = tn / (tn + fp) * 100 if (tn + fp) > 0 else None
    rating_mae = sum(rating_errors) / len(rating_errors) if rating_errors else None

    return {
        "accuracy": accuracy,
        "accept_recall": accept_recall,
        "reject_recall": reject_recall,
        "rating_mae": rating_mae,
        "n": total
    }


def aggregate_stats(year_stats: Dict, years: List[int]) -> Dict:
    """Aggregate stats across specified years."""
    agg = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0, "rating_errors": []}
    for year in years:
        if year in year_stats:
            stats = year_stats[year]
            agg["tp"] += stats["tp"]
            agg["tn"] += stats["tn"]
            agg["fp"] += stats["fp"]
            agg["fn"] += stats["fn"]
            agg["total"] += stats["total"]
            agg["rating_errors"].extend(stats["rating_errors"])
    return agg


def format_combined_value(values: List[Optional[float]], fmt: str = ".1f") -> str:
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
    # e.g., "finetuned-ckpt-1597.jsonl" -> "1597"
    match = re.search(r"ckpt-(\d+)", filename)
    if match:
        return match.group(1)
    return filename.replace(".jsonl", "")


def print_combined_per_year_table(
    all_year_stats: List[Dict],
    ckpt_names: List[str],
    is_rating: bool
):
    """Print combined per-year metrics table."""
    print("\n--- Per-Year Metrics ---")
    print(f"Checkpoints: {' / '.join(ckpt_names)}")

    # Determine which years have data (union of all checkpoints)
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

    # Calculate column width based on number of checkpoints
    num_ckpts = len(ckpt_names)
    # Each value is ~4 chars, plus "/" separators
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

    # Rating MAE (only for ratingbinary)
    if is_rating:
        row = f"{'Rating MAE':<20}"
        for y in years:
            values = [m[y]["rating_mae"] for m in all_metrics]
            row += f"{format_combined_value(values, '.2f'):>{col_width}}"
        print(row)

    # N (sample count) - same for all checkpoints, just show once
    row = f"{'N':<20}"
    for y in years:
        # Use first checkpoint's N (should be same for all)
        val = all_metrics[0][y]["n"]
        row += f"{val:>{col_width}}"
    print(row)


def print_combined_year_group_table(
    all_year_stats: List[Dict],
    ckpt_names: List[str],
    is_rating: bool
):
    """Print combined year group metrics table (<=2024 vs >2024)."""
    print("\n--- Year Group Metrics ---")
    print(f"Checkpoints: {' / '.join(ckpt_names)}")

    # Calculate column width
    num_ckpts = len(ckpt_names)
    col_width = max(12, num_ckpts * 5 + 2)

    # Compute metrics for each checkpoint
    all_metrics_le = []
    all_metrics_gt = []
    for year_stats in all_year_stats:
        all_years = sorted(year_stats.keys())
        years_le_2024 = [y for y in all_years if y <= 2024]
        years_gt_2024 = [y for y in all_years if y > 2024]

        stats_le = aggregate_stats(year_stats, years_le_2024)
        stats_gt = aggregate_stats(year_stats, years_gt_2024)

        all_metrics_le.append(calc_metrics(stats_le))
        all_metrics_gt.append(calc_metrics(stats_gt))

    # Header
    header = f"{'Metric':<20}{' <=2024':>{col_width}}{'  >2024':>{col_width}}"
    print(header)
    print("-" * len(header))

    # Accuracy
    row = f"{'Accuracy (%)':<20}"
    values_le = [m["accuracy"] for m in all_metrics_le]
    values_gt = [m["accuracy"] for m in all_metrics_gt]
    row += f"{format_combined_value(values_le):>{col_width}}"
    row += f"{format_combined_value(values_gt):>{col_width}}"
    print(row)

    # Accept Recall
    row = f"{'Accept Recall (%)':<20}"
    values_le = [m["accept_recall"] for m in all_metrics_le]
    values_gt = [m["accept_recall"] for m in all_metrics_gt]
    row += f"{format_combined_value(values_le):>{col_width}}"
    row += f"{format_combined_value(values_gt):>{col_width}}"
    print(row)

    # Reject Recall
    row = f"{'Reject Recall (%)':<20}"
    values_le = [m["reject_recall"] for m in all_metrics_le]
    values_gt = [m["reject_recall"] for m in all_metrics_gt]
    row += f"{format_combined_value(values_le):>{col_width}}"
    row += f"{format_combined_value(values_gt):>{col_width}}"
    print(row)

    # Rating MAE
    if is_rating:
        row = f"{'Rating MAE':<20}"
        values_le = [m["rating_mae"] for m in all_metrics_le]
        values_gt = [m["rating_mae"] for m in all_metrics_gt]
        row += f"{format_combined_value(values_le, '.2f'):>{col_width}}"
        row += f"{format_combined_value(values_gt, '.2f'):>{col_width}}"
        print(row)

    # N
    row = f"{'N':<20}"
    row += f"{all_metrics_le[0]['n']:>{col_width}}"
    row += f"{all_metrics_gt[0]['n']:>{col_width}}"
    print(row)


def print_combined_overall_table(
    all_year_stats: List[Dict],
    ckpt_names: List[str],
    is_rating: bool
):
    """Print combined overall aggregated metrics."""
    print("\n--- Overall Metrics ---")
    print(f"Checkpoints: {' / '.join(ckpt_names)}")

    # Compute metrics for each checkpoint
    all_metrics = []
    for year_stats in all_year_stats:
        all_years = sorted(year_stats.keys())
        stats_all = aggregate_stats(year_stats, all_years)
        all_metrics.append(calc_metrics(stats_all))

    # Print combined values
    values = [m["accuracy"] for m in all_metrics]
    print(f"{'Accuracy (%):':<20}{format_combined_value(values)}")

    values = [m["accept_recall"] for m in all_metrics]
    print(f"{'Accept Recall (%):':<20}{format_combined_value(values)}")

    values = [m["reject_recall"] for m in all_metrics]
    print(f"{'Reject Recall (%):':<20}{format_combined_value(values)}")

    if is_rating:
        values = [m["rating_mae"] for m in all_metrics]
        print(f"{'Rating MAE:':<20}{format_combined_value(values, '.2f')}")

    print(f"{'N:':<20}{all_metrics[0]['n']}")


def analyze_directory(result_dir: str, data_root: str):
    """Analyze all checkpoints in a result directory with combined tables."""
    result_path = Path(result_dir)
    short_name = result_path.name

    if short_name not in DATASET_MAPPINGS:
        print(f"Error: Unknown dataset '{short_name}'")
        print(f"Known datasets: {list(DATASET_MAPPINGS.keys())}")
        return

    dataset_name = DATASET_MAPPINGS[short_name]
    data_dir = os.path.join(data_root, f"{dataset_name}_test")

    if not os.path.exists(data_dir):
        print(f"Error: Test data directory not found: {data_dir}")
        return

    # Find all checkpoint files, sorted numerically by checkpoint number
    def ckpt_sort_key(f):
        match = re.search(r"ckpt-(\d+)", f.name)
        return int(match.group(1)) if match else 0
    ckpt_files = sorted(result_path.glob("*.jsonl"), key=ckpt_sort_key)

    if not ckpt_files:
        print(f"No checkpoint files found in {result_dir}")
        return

    is_rating = is_ratingbinary(short_name)

    # Load test data once
    test_data = load_test_data(data_dir)

    # Load and compute stats for all checkpoints
    all_year_stats = []
    ckpt_names = []
    for ckpt_file in ckpt_files:
        predictions = load_predictions(str(ckpt_file))
        year_stats = compute_stats(predictions, test_data, is_rating)
        all_year_stats.append(year_stats)
        ckpt_names.append(get_ckpt_short_name(ckpt_file.name))

    # Print header
    print("=" * 80)
    print(f"{short_name}")
    print(f"Checkpoints: {', '.join([f.name for f in ckpt_files])}")
    print("=" * 80)

    # Print combined tables
    print_combined_per_year_table(all_year_stats, ckpt_names, is_rating)
    print_combined_year_group_table(all_year_stats, ckpt_names, is_rating)
    print_combined_overall_table(all_year_stats, ckpt_names, is_rating)
    print()


def analyze_all(results_root: str, data_root: str):
    """Analyze all result directories."""
    results_path = Path(results_root)

    for short_name in sorted(DATASET_MAPPINGS.keys()):
        result_dir = results_path / short_name
        if result_dir.exists():
            analyze_directory(str(result_dir), data_root)


def main():
    parser = argparse.ArgumentParser(description="Analyze testing_v8 results")
    parser.add_argument(
        "result_dir",
        nargs="?",
        help="Path to result directory (e.g., results/testing_v8/balanced_vision/)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all result directories"
    )
    parser.add_argument(
        "--results-root",
        default="results/testing_v8",
        help="Root directory for results (default: results/testing_v8)"
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory for test data (default: data)"
    )

    args = parser.parse_args()

    if args.all:
        analyze_all(args.results_root, args.data_root)
    elif args.result_dir:
        analyze_directory(args.result_dir, args.data_root)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
