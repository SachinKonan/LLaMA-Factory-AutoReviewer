#!/usr/bin/env python3
"""
Compute metrics and generate plots for inference scaling experiments.

Metrics:
- Overall accuracy
- Accept recall (sensitivity)
- Reject recall (specificity)
- In-distribution (2020-2024) vs out-of-distribution (2025)
- By year breakdown
- JSON validity rate
- Field presence and correctness

Usage:
    python compute_metrics.py --results_dir ./results --output_dir ./metrics
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Expected JSON schema fields for the new prompt format
EXPECTED_FIELDS = {
    "summary": str,
    "questions": str,
    "limitations": str,
    "strengths": str,
    "weaknesses": str,
    "ethical_concerns": bool,
    "soundness": int,  # 1-5
    "presentation": int,  # 1-5
    "contribution": int,  # 1-5
    "overall": int,  # 1-10
    "confidence": int,  # 1-5
    "decision": str,  # "accept" or "reject"
}

SCORE_RANGES = {
    "soundness": (1, 5),
    "presentation": (1, 5),
    "contribution": (1, 5),
    "overall": (1, 10),
    "confidence": (1, 5),
}


def parse_boxed_from_text(text: str) -> Tuple[Optional[str], str]:
    """
    Parse boxed decision from model output text.

    Returns:
        Tuple of (decision or None, error_message or "")
    """
    if not text:
        return None, "empty_response"

    # Match \boxed{Accept} or \boxed{Reject}
    match = re.search(r'\\boxed\{(Accept|Reject)\}', text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize(), ""

    return None, "no_boxed_found"


def compute_boxed_metrics(predictions_path: str) -> Dict:
    """
    Compute boxed format validity from predictions file (for original prompt).

    Args:
        predictions_path: Path to predictions.jsonl file

    Returns:
        Dictionary with boxed-related metrics
    """
    metrics = {
        "total_predictions": 0,
        "valid_boxed_count": 0,
        "invalid_boxed_count": 0,
        "boxed_validity_rate": 0.0,
        "error_types": defaultdict(int),
        "decision_distribution": {"Accept": 0, "Reject": 0},
    }

    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            predictions = data.get("predict", [])
            if isinstance(predictions, str):
                predictions = [predictions]

            for pred in predictions:
                metrics["total_predictions"] += 1

                decision, error = parse_boxed_from_text(pred)

                if decision is not None:
                    metrics["valid_boxed_count"] += 1
                    metrics["decision_distribution"][decision] += 1
                else:
                    metrics["invalid_boxed_count"] += 1
                    metrics["error_types"][error] += 1

    # Compute rates
    total = metrics["total_predictions"]
    if total > 0:
        metrics["boxed_validity_rate"] = metrics["valid_boxed_count"] / total

    # Convert defaultdict to regular dict for JSON serialization
    metrics["error_types"] = dict(metrics["error_types"])

    return metrics


def parse_json_from_text(text: str) -> Tuple[Optional[Dict], str]:
    """
    Parse JSON from model output text.

    Returns:
        Tuple of (parsed_dict or None, error_message or "")
    """
    if not text:
        return None, "empty_response"

    # Try to find JSON block in markdown code fence
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1)), ""
        except json.JSONDecodeError as e:
            return None, f"json_parse_error_in_fence: {str(e)}"

    # Try to find raw JSON object
    try:
        # Find the last occurrence of a JSON-like structure
        json_start = text.rfind('{')
        json_end = text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = text[json_start:json_end]
            return json.loads(json_str), ""
    except json.JSONDecodeError as e:
        return None, f"json_parse_error: {str(e)}"

    return None, "no_json_found"


def validate_json_fields(parsed: Dict) -> Dict[str, Any]:
    """
    Validate JSON fields against expected schema.

    Returns:
        Dictionary with field validation results
    """
    result = {
        "fields_present": [],
        "fields_missing": [],
        "fields_correct_type": [],
        "fields_wrong_type": [],
        "scores_in_range": [],
        "scores_out_of_range": [],
    }

    for field, expected_type in EXPECTED_FIELDS.items():
        if field in parsed:
            result["fields_present"].append(field)
            value = parsed[field]

            # Check type
            if expected_type == int:
                # Allow int or float that can be int
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    result["fields_correct_type"].append(field)

                    # Check range for score fields
                    if field in SCORE_RANGES:
                        min_val, max_val = SCORE_RANGES[field]
                        if min_val <= int(value) <= max_val:
                            result["scores_in_range"].append(field)
                        else:
                            result["scores_out_of_range"].append(field)
                else:
                    result["fields_wrong_type"].append(field)
            elif expected_type == bool:
                if isinstance(value, bool):
                    result["fields_correct_type"].append(field)
                else:
                    result["fields_wrong_type"].append(field)
            elif expected_type == str:
                if isinstance(value, str):
                    result["fields_correct_type"].append(field)
                    # For decision field, also check valid values
                    if field == "decision":
                        if value.lower() in ["accept", "reject"]:
                            result["scores_in_range"].append("decision_valid")
                        else:
                            result["scores_out_of_range"].append("decision_invalid")
                else:
                    result["fields_wrong_type"].append(field)
        else:
            result["fields_missing"].append(field)

    return result


def compute_json_metrics(predictions_path: str) -> Dict:
    """
    Compute JSON validity and field metrics from predictions file.

    Args:
        predictions_path: Path to predictions.jsonl file

    Returns:
        Dictionary with JSON-related metrics
    """
    metrics = {
        "total_predictions": 0,
        "valid_json_count": 0,
        "invalid_json_count": 0,
        "json_validity_rate": 0.0,
        "error_types": defaultdict(int),
        "field_presence_rate": {},
        "field_type_correctness_rate": {},
        "score_validity_rate": {},
        "avg_fields_present": 0.0,
    }

    field_presence_counts = defaultdict(int)
    field_type_correct_counts = defaultdict(int)
    score_valid_counts = defaultdict(int)
    total_fields_present = 0

    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            predictions = data.get("predict", [])
            if isinstance(predictions, str):
                predictions = [predictions]

            for pred in predictions:
                metrics["total_predictions"] += 1

                parsed, error = parse_json_from_text(pred)

                if parsed is not None:
                    metrics["valid_json_count"] += 1
                    validation = validate_json_fields(parsed)

                    total_fields_present += len(validation["fields_present"])

                    for field in validation["fields_present"]:
                        field_presence_counts[field] += 1
                    for field in validation["fields_correct_type"]:
                        field_type_correct_counts[field] += 1
                    for field in validation["scores_in_range"]:
                        score_valid_counts[field] += 1
                else:
                    metrics["invalid_json_count"] += 1
                    metrics["error_types"][error] += 1

    # Compute rates
    total = metrics["total_predictions"]
    if total > 0:
        metrics["json_validity_rate"] = metrics["valid_json_count"] / total
        metrics["avg_fields_present"] = total_fields_present / total

        for field in EXPECTED_FIELDS:
            metrics["field_presence_rate"][field] = field_presence_counts[field] / total
            if field_presence_counts[field] > 0:
                metrics["field_type_correctness_rate"][field] = (
                    field_type_correct_counts[field] / field_presence_counts[field]
                )

        for field in SCORE_RANGES:
            if field_presence_counts[field] > 0:
                metrics["score_validity_rate"][field] = (
                    score_valid_counts[field] / field_presence_counts[field]
                )

        # Decision validity
        if field_presence_counts.get("decision", 0) > 0:
            metrics["score_validity_rate"]["decision"] = (
                score_valid_counts.get("decision_valid", 0) / field_presence_counts["decision"]
            )

    # Convert defaultdict to regular dict for JSON serialization
    metrics["error_types"] = dict(metrics["error_types"])

    return metrics


def load_results(results_path: str) -> List[Dict]:
    """Load results from a JSONL file."""
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return results


def load_metadata(dataset_path: str) -> List[Dict]:
    """Load metadata from original dataset."""
    json_path = os.path.join(dataset_path, "data.json")
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [d.get("_metadata", {}) for d in data]


def compute_metrics(
    results: List[Dict],
    metadata: Optional[List[Dict]] = None
) -> Dict:
    """
    Compute metrics from results.

    Args:
        results: List of result dictionaries with 'prediction', 'ground_truth', 'correct'
        metadata: Optional list of metadata dictionaries with 'year', 'answer', etc.

    Returns:
        Dictionary containing computed metrics
    """
    # Filter valid results
    valid_results = [r for r in results if r.get("correct") is not None]

    if not valid_results:
        return {"error": "No valid results to compute metrics"}

    # Basic counts
    total = len(valid_results)
    correct = sum(1 for r in valid_results if r["correct"])

    # Confusion matrix components
    tp = sum(1 for r in valid_results if r["prediction"] == "Accept" and r["ground_truth"] == "Accept")
    tn = sum(1 for r in valid_results if r["prediction"] == "Reject" and r["ground_truth"] == "Reject")
    fp = sum(1 for r in valid_results if r["prediction"] == "Accept" and r["ground_truth"] == "Reject")
    fn = sum(1 for r in valid_results if r["prediction"] == "Reject" and r["ground_truth"] == "Accept")

    # Compute metrics
    accuracy = correct / total if total > 0 else 0
    accept_recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
    reject_recall = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    accept_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    reject_precision = tn / (tn + fn) if (tn + fn) > 0 else 0

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accept_recall": accept_recall,
        "reject_recall": reject_recall,
        "accept_precision": accept_precision,
        "reject_precision": reject_precision,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

    # Year-wise breakdown if metadata is available
    if metadata and len(metadata) == len(results):
        year_metrics = defaultdict(lambda: {"total": 0, "correct": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0})

        for result, meta in zip(results, metadata):
            if result.get("correct") is None:
                continue

            year = meta.get("year", "unknown")
            year_metrics[year]["total"] += 1
            if result["correct"]:
                year_metrics[year]["correct"] += 1

            pred = result.get("prediction")
            gt = result.get("ground_truth")
            if pred == "Accept" and gt == "Accept":
                year_metrics[year]["tp"] += 1
            elif pred == "Reject" and gt == "Reject":
                year_metrics[year]["tn"] += 1
            elif pred == "Accept" and gt == "Reject":
                year_metrics[year]["fp"] += 1
            elif pred == "Reject" and gt == "Accept":
                year_metrics[year]["fn"] += 1

        # Compute per-year metrics
        metrics["by_year"] = {}
        for year, counts in sorted(year_metrics.items()):
            y_total = counts["total"]
            y_tp, y_tn, y_fp, y_fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]

            metrics["by_year"][year] = {
                "total": y_total,
                "correct": counts["correct"],
                "accuracy": counts["correct"] / y_total if y_total > 0 else 0,
                "accept_recall": y_tp / (y_tp + y_fn) if (y_tp + y_fn) > 0 else 0,
                "reject_recall": y_tn / (y_tn + y_fp) if (y_tn + y_fp) > 0 else 0
            }

        # In-distribution (2020-2024) vs Out-of-distribution (2025)
        in_dist = {"total": 0, "correct": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
        out_dist = {"total": 0, "correct": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}

        for year, counts in year_metrics.items():
            if year == "unknown":
                continue
            target = out_dist if year >= 2025 else in_dist
            for key in ["total", "correct", "tp", "tn", "fp", "fn"]:
                target[key] += counts[key]

        for dist_name, dist_counts in [("in_distribution", in_dist), ("out_distribution", out_dist)]:
            d_total = dist_counts["total"]
            d_tp, d_tn, d_fp, d_fn = dist_counts["tp"], dist_counts["tn"], dist_counts["fp"], dist_counts["fn"]

            metrics[dist_name] = {
                "total": d_total,
                "correct": dist_counts["correct"],
                "accuracy": dist_counts["correct"] / d_total if d_total > 0 else 0,
                "accept_recall": d_tp / (d_tp + d_fn) if (d_tp + d_fn) > 0 else 0,
                "reject_recall": d_tn / (d_tn + d_fp) if (d_tn + d_fp) > 0 else 0
            }

    return metrics


def print_json_metrics(metrics: Dict, name: str = ""):
    """Print JSON validity metrics in a formatted way."""
    print(f"\n{'='*70}")
    print(f"JSON Metrics: {name}")
    print(f"{'='*70}")

    print(f"\nJSON Validity:")
    print(f"  Total predictions: {metrics['total_predictions']}")
    print(f"  Valid JSON: {metrics['valid_json_count']} ({metrics['json_validity_rate']:.2%})")
    print(f"  Invalid JSON: {metrics['invalid_json_count']}")
    print(f"  Avg fields present: {metrics['avg_fields_present']:.1f}/{len(EXPECTED_FIELDS)}")

    if metrics.get("error_types"):
        print(f"\nError Types:")
        for error, count in sorted(metrics["error_types"].items(), key=lambda x: -x[1]):
            print(f"  {error}: {count}")

    if metrics.get("field_presence_rate"):
        print(f"\nField Presence Rates:")
        for field, rate in sorted(metrics["field_presence_rate"].items()):
            print(f"  {field}: {rate:.2%}")

    if metrics.get("score_validity_rate"):
        print(f"\nScore Validity Rates (in valid range):")
        for field, rate in sorted(metrics["score_validity_rate"].items()):
            print(f"  {field}: {rate:.2%}")


def print_metrics(metrics: Dict, name: str = ""):
    """Print metrics in a formatted way."""
    print(f"\n{'='*70}")
    print(f"Metrics: {name}")
    print(f"{'='*70}")

    print(f"\nOverall Performance:")
    print(f"  Total samples: {metrics['total']}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"  Accept Recall: {metrics['accept_recall']:.4f}")
    print(f"  Reject Recall: {metrics['reject_recall']:.4f}")
    print(f"  Accept Precision: {metrics['accept_precision']:.4f}")
    print(f"  Reject Precision: {metrics['reject_precision']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TP (Accept->Accept): {metrics['tp']}")
    print(f"  TN (Reject->Reject): {metrics['tn']}")
    print(f"  FP (Reject->Accept): {metrics['fp']}")
    print(f"  FN (Accept->Reject): {metrics['fn']}")

    if "in_distribution" in metrics:
        print(f"\nIn-Distribution (2020-2024):")
        in_d = metrics["in_distribution"]
        print(f"  Accuracy: {in_d['accuracy']:.4f} ({in_d['correct']}/{in_d['total']})")
        print(f"  Accept Recall: {in_d['accept_recall']:.4f}")
        print(f"  Reject Recall: {in_d['reject_recall']:.4f}")

    if "out_distribution" in metrics:
        print(f"\nOut-of-Distribution (2025):")
        out_d = metrics["out_distribution"]
        print(f"  Accuracy: {out_d['accuracy']:.4f} ({out_d['correct']}/{out_d['total']})")
        print(f"  Accept Recall: {out_d['accept_recall']:.4f}")
        print(f"  Reject Recall: {out_d['reject_recall']:.4f}")

    if "by_year" in metrics:
        print(f"\nBy Year:")
        for year, y_metrics in sorted(metrics["by_year"].items()):
            print(f"  {year}: Acc={y_metrics['accuracy']:.4f}, "
                  f"AccRecall={y_metrics['accept_recall']:.4f}, "
                  f"RejRecall={y_metrics['reject_recall']:.4f} "
                  f"(n={y_metrics['total']})")


def generate_plots(all_metrics: Dict[str, Dict], output_dir: str):
    """Generate comparison plots for all configurations."""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    configs = list(all_metrics.keys())
    accuracies = [m["accuracy"] for m in all_metrics.values()]
    accept_recalls = [m["accept_recall"] for m in all_metrics.values()]
    reject_recalls = [m["reject_recall"] for m in all_metrics.values()]

    # Bar plot: Overall metrics comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(configs))
    width = 0.25

    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='#2ecc71')
    bars2 = ax.bar(x, accept_recalls, width, label='Accept Recall', color='#3498db')
    bars3 = ax.bar(x + width, reject_recalls, width, label='Reject Recall', color='#e74c3c')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Inference Scaling: Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150)
    plt.close()

    # In-distribution vs Out-of-distribution plot
    in_dist_acc = []
    out_dist_acc = []
    valid_configs = []

    for config, metrics in all_metrics.items():
        if "in_distribution" in metrics and "out_distribution" in metrics:
            in_dist_acc.append(metrics["in_distribution"]["accuracy"])
            out_dist_acc.append(metrics["out_distribution"]["accuracy"])
            valid_configs.append(config)

    if valid_configs:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(valid_configs))
        width = 0.35

        bars1 = ax.bar(x - width/2, in_dist_acc, width, label='In-Distribution (2020-2024)', color='#3498db')
        bars2 = ax.bar(x + width/2, out_dist_acc, width, label='Out-of-Distribution (2025)', color='#e74c3c')

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Accuracy')
        ax.set_title('In-Distribution vs Out-of-Distribution Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_configs, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'), dpi=150)
        plt.close()

    # Year-wise accuracy plot
    year_data = defaultdict(dict)
    for config, metrics in all_metrics.items():
        if "by_year" in metrics:
            for year, y_metrics in metrics["by_year"].items():
                if year != "unknown":
                    year_data[year][config] = y_metrics["accuracy"]

    if year_data:
        years = sorted(year_data.keys())
        fig, ax = plt.subplots(figsize=(12, 6))

        for config in configs:
            accs = [year_data[year].get(config, 0) for year in years]
            ax.plot(years, accs, marker='o', label=config)

        ax.set_xlabel('Year')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Year')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_year.png'), dpi=150)
        plt.close()

    print(f"\nPlots saved to: {output_dir}")


def create_summary_table(all_metrics: Dict[str, Dict], output_path: str):
    """Create a summary table of all metrics."""
    rows = []
    for config, metrics in all_metrics.items():
        row = {
            "Configuration": config,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Accept Recall": f"{metrics['accept_recall']:.4f}",
            "Reject Recall": f"{metrics['reject_recall']:.4f}",
            "Accept Precision": f"{metrics['accept_precision']:.4f}",
            "Reject Precision": f"{metrics['reject_precision']:.4f}",
            "Total": metrics["total"]
        }

        if "in_distribution" in metrics:
            row["In-Dist Acc"] = f"{metrics['in_distribution']['accuracy']:.4f}"
        if "out_distribution" in metrics:
            row["Out-Dist Acc"] = f"{metrics['out_distribution']['accuracy']:.4f}"

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSummary table saved to: {output_path}")

    # Also print as markdown (if tabulate is available)
    try:
        print("\n" + df.to_markdown(index=False))
    except ImportError:
        print("\n" + df.to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description="Compute metrics and generate plots")
    parser.add_argument("--results_dir", type=str, default="./inference_scaling/results",
                        help="Directory containing results")
    parser.add_argument("--output_dir", type=str, default="./inference_scaling/metrics",
                        help="Output directory for metrics and plots")
    parser.add_argument("--base_data_dir", type=str,
                        default="/n/fs/vision-mix/sk7524/LLaMA-Factory/data",
                        help="Base directory containing original datasets (for metadata)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = {}
    all_json_metrics = {}

    # Find all result files
    results_dir = Path(args.results_dir)

    # Expected structure: results/{modality}/{prompt_variant}/results_{strategy}.jsonl
    for modality_dir in results_dir.iterdir():
        if not modality_dir.is_dir():
            continue

        modality = modality_dir.name

        for variant_dir in modality_dir.iterdir():
            if not variant_dir.is_dir():
                continue

            variant = variant_dir.name

            # Compute format validity metrics from predictions.jsonl
            predictions_file = variant_dir / "predictions.jsonl"
            if predictions_file.exists():
                config_name = f"{modality}/{variant}"
                if variant in ["new", "new_fewshot"]:
                    # JSON format for new prompts
                    json_metrics = compute_json_metrics(str(predictions_file))
                    all_json_metrics[config_name] = json_metrics
                    print_json_metrics(json_metrics, config_name)
                elif variant == "original":
                    # Boxed format for original prompt
                    boxed_metrics = compute_boxed_metrics(str(predictions_file))
                    # Store in same format for unified plotting
                    all_json_metrics[config_name] = {
                        "total_predictions": boxed_metrics["total_predictions"],
                        "valid_json_count": boxed_metrics["valid_boxed_count"],
                        "invalid_json_count": boxed_metrics["invalid_boxed_count"],
                        "json_validity_rate": boxed_metrics["boxed_validity_rate"],
                        "format_type": "boxed",
                        "error_types": boxed_metrics["error_types"],
                        "decision_distribution": boxed_metrics["decision_distribution"],
                    }
                    print(f"\n{'='*70}")
                    print(f"Boxed Metrics: {config_name}")
                    print(f"{'='*70}")
                    print(f"  Total predictions: {boxed_metrics['total_predictions']}")
                    print(f"  Valid boxed: {boxed_metrics['valid_boxed_count']} ({boxed_metrics['boxed_validity_rate']:.2%})")
                    print(f"  Invalid: {boxed_metrics['invalid_boxed_count']}")

            # Find result files
            for result_file in variant_dir.glob("results_*.jsonl"):
                strategy = result_file.stem.replace("results_", "")
                config_name = f"{modality}/{variant}/{strategy}"

                results = load_results(str(result_file))

                # Try to load metadata from original dataset
                base_dataset = f"iclr_2020_2025_85_5_10_split6_balanced_{modality}_binary_noreviews_v6_test"
                metadata_path = os.path.join(args.base_data_dir, base_dataset)
                metadata = load_metadata(metadata_path)

                if len(metadata) != len(results):
                    print(f"Warning: Metadata length mismatch for {config_name}, skipping year analysis")
                    metadata = None

                metrics = compute_metrics(results, metadata)
                all_metrics[config_name] = metrics
                print_metrics(metrics, config_name)

    if all_metrics:
        # Generate plots
        generate_plots(all_metrics, args.output_dir)

        # Create summary table
        summary_path = os.path.join(args.output_dir, "summary.csv")
        create_summary_table(all_metrics, summary_path)

        # Save all metrics as JSON
        metrics_path = os.path.join(args.output_dir, "all_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nAll metrics saved to: {metrics_path}")

        # Save JSON metrics separately
        if all_json_metrics:
            json_metrics_path = os.path.join(args.output_dir, "json_metrics.json")
            with open(json_metrics_path, "w") as f:
                json.dump(all_json_metrics, f, indent=2)
            print(f"JSON metrics saved to: {json_metrics_path}")

            # Create JSON metrics summary table
            json_summary_rows = []
            for config, jm in all_json_metrics.items():
                format_type = jm.get("format_type", "json")
                row = {
                    "Configuration": config,
                    "Format": format_type,
                    "Validity Rate": f"{jm['json_validity_rate']:.2%}",
                    "Total Predictions": jm["total_predictions"],
                }
                # Add JSON-specific fields if available
                if format_type == "json" or "avg_fields_present" in jm:
                    row["Avg Fields Present"] = f"{jm.get('avg_fields_present', 0):.1f}/{len(EXPECTED_FIELDS)}"
                    row["Decision Present"] = f"{jm.get('field_presence_rate', {}).get('decision', 0):.2%}"
                    row["Decision Valid"] = f"{jm.get('score_validity_rate', {}).get('decision', 0):.2%}"
                json_summary_rows.append(row)
            json_df = pd.DataFrame(json_summary_rows)
            json_summary_path = os.path.join(args.output_dir, "json_summary.csv")
            json_df.to_csv(json_summary_path, index=False)
            print(f"JSON summary saved to: {json_summary_path}")
            print("\nJSON Metrics Summary:")
            try:
                print(json_df.to_markdown(index=False))
            except ImportError:
                print(json_df.to_string(index=False))
    else:
        print("\nNo results found. Please run inference first.")


if __name__ == "__main__":
    main()
