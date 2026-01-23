#!/usr/bin/env python3
"""
Analyze ablation inference results and generate a CSV summary.

Metrics computed:
- Dataset name, size
- Overall accuracy
- Accept recall, Reject recall
- Accuracy by year (2020-2025)
"""

import json
import re
import csv
from pathlib import Path
from collections import defaultdict


def extract_decision(text: str) -> str | None:
    """Extract Accept/Reject decision from model output or label."""
    if not text:
        return None

    text = text.strip()

    # Try boxed format first
    boxed_match = re.search(r'\\boxed\{(Accept|Reject)\}', text, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).capitalize()

    # Try JSON format
    json_match = re.search(r'"decision"\s*:\s*"(Accept|Reject)"', text, re.IGNORECASE)
    if json_match:
        return json_match.group(1).capitalize()

    # Simple keyword match (last resort)
    text_lower = text.lower()
    if 'accept' in text_lower and 'reject' not in text_lower:
        return 'Accept'
    if 'reject' in text_lower and 'accept' not in text_lower:
        return 'Reject'

    # Check if the entire text is just Accept or Reject
    if text_lower.strip() in ['accept', 'reject']:
        return text.strip().capitalize()

    return None


def load_metadata_mapping(base_dataset_path: Path) -> list[dict]:
    """Load metadata from base dataset."""
    data_file = base_dataset_path / "data.json"
    if not data_file.exists():
        return []

    with open(data_file) as f:
        data = json.load(f)

    return [item.get("_metadata", {}) for item in data]


def analyze_predictions(pred_file: Path, metadata_list: list[dict]) -> dict:
    """Analyze a single predictions file."""
    results = {
        "dataset_name": pred_file.stem.replace("_predictions", ""),
        "dataset_size": 0,
        "true_acceptance_rate": 0.0,
        "predicted_acceptance_rate": 0.0,
        "accuracy": 0.0,
        "accept_recall": 0.0,
        "reject_recall": 0.0,
    }

    # Initialize year stats
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    for year in years:
        results[f"metrics_{year}"] = ""

    if not pred_file.exists():
        return results

    # Load predictions
    predictions = []
    with open(pred_file) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    results["dataset_size"] = len(predictions)

    if not predictions:
        return results

    # Count stats
    total_correct = 0
    total = 0

    # For recall calculation
    true_accepts = 0
    true_rejects = 0
    correct_accepts = 0
    correct_rejects = 0
    predicted_accepts = 0

    # Per-year stats
    year_correct = defaultdict(int)
    year_total = defaultdict(int)
    year_true_accepts = defaultdict(int)
    year_true_rejects = defaultdict(int)
    year_correct_accepts = defaultdict(int)
    year_correct_rejects = defaultdict(int)

    for i, pred in enumerate(predictions):
        # Get ground truth from label
        label = pred.get("label", "")
        ground_truth = extract_decision(label)

        # Get prediction - handle both string and list
        pred_text = pred.get("predict", "")
        if isinstance(pred_text, list):
            pred_text = pred_text[0] if pred_text else ""
        predicted = extract_decision(pred_text)

        if ground_truth is None:
            continue

        total += 1

        # Get year from metadata if available
        year = None
        if i < len(metadata_list):
            year = metadata_list[i].get("year")

        # Track predicted accepts
        if predicted == "Accept":
            predicted_accepts += 1

        # Track recalls
        if ground_truth == "Accept":
            true_accepts += 1
            if year:
                year_true_accepts[year] += 1
            if predicted == "Accept":
                correct_accepts += 1
                if year:
                    year_correct_accepts[year] += 1
        elif ground_truth == "Reject":
            true_rejects += 1
            if year:
                year_true_rejects[year] += 1
            if predicted == "Reject":
                correct_rejects += 1
                if year:
                    year_correct_rejects[year] += 1

        # Track accuracy
        if predicted == ground_truth:
            total_correct += 1
            if year:
                year_correct[year] += 1

        if year:
            year_total[year] += 1

    # Compute metrics
    if total > 0:
        results["true_acceptance_rate"] = round(true_accepts / total * 100, 2)
        results["predicted_acceptance_rate"] = round(predicted_accepts / total * 100, 2)
        results["accuracy"] = round(total_correct / total * 100, 2)

    if true_accepts > 0:
        results["accept_recall"] = round(correct_accepts / true_accepts * 100, 2)

    if true_rejects > 0:
        results["reject_recall"] = round(correct_rejects / true_rejects * 100, 2)

    # Per-year metrics: accuracy/accept_recall/reject_recall(n=size)
    for year in years:
        if year_total[year] > 0:
            y_acc = round(year_correct[year] / year_total[year] * 100, 1)
            y_acc_recall = round(year_correct_accepts[year] / year_true_accepts[year] * 100, 1) if year_true_accepts[year] > 0 else 0.0
            y_rej_recall = round(year_correct_rejects[year] / year_true_rejects[year] * 100, 1) if year_true_rejects[year] > 0 else 0.0
            results[f"metrics_{year}"] = f"{y_acc}/{y_acc_recall}/{y_rej_recall}(n={year_total[year]})"

    return results


def get_base_dataset_for_ablation(ablation_name: str) -> str:
    """Determine the base dataset path for an ablation dataset."""
    # Map ablation datasets to their base datasets
    if "vision" in ablation_name:
        base = "iclr_2020_2025_85_5_10_split6_original_vision_binary_noreviews_v6_test"
    else:
        base = "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_test"
    return base


def main():
    project_dir = Path(__file__).parent.parent.parent
    results_dir = project_dir / "ablations" / "results"
    data_dir = project_dir / "data"
    output_file = project_dir / "ablations" / "results" / "analysis_summary.csv"

    # Find all prediction files
    pred_files = sorted(results_dir.glob("*_predictions.jsonl"))

    if not pred_files:
        print("No prediction files found!")
        return

    print(f"Found {len(pred_files)} prediction files")

    # Cache metadata for base datasets
    metadata_cache = {}

    all_results = []

    for pred_file in pred_files:
        print(f"Analyzing: {pred_file.name}")

        # Determine base dataset and load metadata
        ablation_name = pred_file.stem.replace("_predictions", "")
        base_dataset = get_base_dataset_for_ablation(ablation_name)

        if base_dataset not in metadata_cache:
            base_path = data_dir / base_dataset
            metadata_cache[base_dataset] = load_metadata_mapping(base_path)

        metadata_list = metadata_cache[base_dataset]

        # Analyze
        results = analyze_predictions(pred_file, metadata_list)
        all_results.append(results)

        # Print summary
        print(f"  Size: {results['dataset_size']}, True AR: {results['true_acceptance_rate']}%, Pred AR: {results['predicted_acceptance_rate']}%, "
              f"Accuracy: {results['accuracy']}%, Accept Recall: {results['accept_recall']}%, Reject Recall: {results['reject_recall']}%")

    # Write CSV
    if all_results:
        fieldnames = [
            "dataset_name", "dataset_size", "true_acceptance_rate", "predicted_acceptance_rate",
            "accuracy", "accept_recall", "reject_recall",
            "metrics_2020", "metrics_2021", "metrics_2022",
            "metrics_2023", "metrics_2024", "metrics_2025"
        ]

        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\nResults saved to: {output_file}")

    # Also print a formatted table
    print("\n" + "=" * 140)
    print(f"{'Dataset':<70} {'Size':>6} {'T_AR':>6} {'P_AR':>6} {'Acc':>6} {'Acc+':>6} {'Rej+':>6}")
    print("=" * 140)
    for r in all_results:
        name = r["dataset_name"]
        # Shorten the name for display
        short_name = name.replace("iclr_2020_2025_85_5_10_split6_original_", "").replace("_binary_noreviews_v6_test", "")
        print(f"{short_name:<70} {r['dataset_size']:>6} {r['true_acceptance_rate']:>5.1f}% {r['predicted_acceptance_rate']:>5.1f}% {r['accuracy']:>5.1f}% {r['accept_recall']:>5.1f}% {r['reject_recall']:>5.1f}%")
    print("=" * 140)


if __name__ == "__main__":
    main()
