#!/usr/bin/env python3
"""
Analyze v7 sweep results with per-year breakdown.

Shows accuracy, accept recall, and reject recall by year with an overall column.

Usage:
    python scripts/analyze_v7.py results/final_sweep_v7/balanced_clean/finetuned-ckpt-1069.jsonl
    python scripts/analyze_v7.py results/final_sweep_v7/balanced_clean/  # Analyze all checkpoints in dir
    python scripts/analyze_v7.py --all  # Analyze all available results
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

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


def compute_metrics_by_year(predictions: list[dict], test_data: list[dict]) -> dict:
    """Compute accuracy, accept recall, reject recall by year."""
    if len(predictions) != len(test_data):
        print(f"Warning: predictions ({len(predictions)}) != test_data ({len(test_data)})")
        # Use minimum length
        min_len = min(len(predictions), len(test_data))
        predictions = predictions[:min_len]
        test_data = test_data[:min_len]

    # Group by year
    year_stats = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0})

    for pred, data in zip(predictions, test_data):
        metadata = data.get("_metadata", {})
        year = metadata.get("year", "unknown")

        pred_label = extract_prediction(pred.get("predict", ""))
        true_label = extract_label(pred.get("label", ""))

        if pred_label == "unknown" or true_label == "unknown":
            continue

        year_stats[year]["total"] += 1

        if true_label == "accept":
            if pred_label == "accept":
                year_stats[year]["tp"] += 1  # True positive (correctly predicted accept)
            else:
                year_stats[year]["fn"] += 1  # False negative (missed accept)
        else:  # true_label == "reject"
            if pred_label == "reject":
                year_stats[year]["tn"] += 1  # True negative (correctly predicted reject)
            else:
                year_stats[year]["fp"] += 1  # False positive (wrong accept)

    return dict(year_stats)


def compute_overall_stats(year_stats: dict) -> dict:
    """Compute overall stats from year stats."""
    overall = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0}
    for stats in year_stats.values():
        for key in overall:
            overall[key] += stats[key]
    return overall


def calc_metrics(stats: dict) -> tuple[float, float, float]:
    """Calculate accuracy, accept recall, reject recall from stats."""
    tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
    total = stats["total"]

    accuracy = 100 * (tp + tn) / total if total > 0 else 0
    accept_recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR
    reject_recall = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR

    return accuracy, accept_recall, reject_recall


def print_table(year_stats: dict, title: str):
    """Print a formatted table of metrics by year."""
    # Sort years
    years = sorted([y for y in year_stats.keys() if y != "unknown"])

    # Compute overall
    overall_stats = compute_overall_stats(year_stats)

    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    # Header
    header = f"{'Metric':<20}"
    for year in years:
        header += f"{year:>10}"
    header += f"{'Overall':>12}"
    print(header)
    print("-" * len(header))

    # Compute metrics for each year
    year_metrics = {}
    for year in years:
        year_metrics[year] = calc_metrics(year_stats[year])
    overall_metrics = calc_metrics(overall_stats)

    # Accuracy row
    row = f"{'Accuracy (%)':<20}"
    for year in years:
        row += f"{year_metrics[year][0]:>10.1f}"
    row += f"{overall_metrics[0]:>12.1f}"
    print(row)

    # Accept Recall row
    row = f"{'Accept Recall (%)':<20}"
    for year in years:
        row += f"{year_metrics[year][1]:>10.1f}"
    row += f"{overall_metrics[1]:>12.1f}"
    print(row)

    # Reject Recall row
    row = f"{'Reject Recall (%)':<20}"
    for year in years:
        row += f"{year_metrics[year][2]:>10.1f}"
    row += f"{overall_metrics[2]:>12.1f}"
    print(row)

    # Sample count row
    row = f"{'N':<20}"
    for year in years:
        row += f"{year_stats[year]['total']:>10}"
    row += f"{overall_stats['total']:>12}"
    print(row)

    print()


def analyze_file(pred_file: Path):
    """Analyze a single prediction file."""
    # Extract short_name from path
    # Expected format: results/final_sweep_v7[_pli]/{short_name}/finetuned*.jsonl
    parts = pred_file.parts
    short_name = None
    for i, part in enumerate(parts):
        if part.startswith("final_sweep_v7"):
            if i + 1 < len(parts):
                short_name = parts[i + 1]
                break

    if not short_name or short_name not in DATASET_MAPPINGS:
        print(f"Error: Could not determine dataset for {pred_file}")
        print(f"  Extracted short_name: {short_name}")
        return

    dataset_name = DATASET_MAPPINGS[short_name]

    # Load data
    try:
        predictions = load_predictions(pred_file)
        test_data = load_test_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Compute metrics
    year_stats = compute_metrics_by_year(predictions, test_data)

    # Print table
    title = f"{pred_file.name} ({short_name})"
    print_table(year_stats, title)


def find_prediction_files(path: Path) -> list[Path]:
    """Find all prediction files in a path."""
    if path.is_file():
        return [path]
    elif path.is_dir():
        return sorted(path.glob("finetuned*.jsonl"))
    return []


def main():
    parser = argparse.ArgumentParser(description="Analyze v7 sweep results with per-year breakdown")
    parser.add_argument("path", nargs="?", type=Path, help="Prediction file or directory")
    parser.add_argument("--all", action="store_true", help="Analyze all available results")
    args = parser.parse_args()

    if args.all:
        # Find all results
        results_dirs = [
            Path("results/final_sweep_v7"),
            Path("results/final_sweep_v7_pli"),
        ]
        for results_dir in results_dirs:
            if not results_dir.exists():
                continue
            for model_dir in sorted(results_dir.iterdir()):
                if model_dir.is_dir():
                    pred_files = find_prediction_files(model_dir)
                    for pred_file in pred_files:
                        analyze_file(pred_file)
    elif args.path:
        pred_files = find_prediction_files(args.path)
        if not pred_files:
            print(f"No prediction files found at {args.path}")
            return
        for pred_file in pred_files:
            analyze_file(pred_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
