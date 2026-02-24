#!/usr/bin/env python3
"""
Stage 4: Evaluate all trained models on test set.

Computes metrics:
- Accuracy
- Precision (for Accept class)
- Recall (for Accept class)
- F1 score
- Predicted acceptance rate
- Per-year breakdown

Usage:
    python stage4_evaluate.py
    python stage4_evaluate.py --experiment accept_gamma2_prop1_2
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Paths
RESULTS_DIR = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_11_26_training/weighted_loss_fn/results")
METRICS_DIR = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_11_26_training/weighted_loss_fn/metrics")
TEST_DATA_PATH = "/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_11_26_training/weighted_loss_fn/data/iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test/data.json"


def parse_prediction(text: str) -> str:
    """
    Extract Accept/Reject from model output.

    Handles formats:
    - \\boxed{Accept}
    - \\boxed{Reject}
    - Plain "Accept" or "Reject"
    """
    # Try boxed format first
    match = re.search(r"\\boxed\{(Accept|Reject)\}", text)
    if match:
        return match.group(1)

    # Try plain format
    if "Accept" in text:
        return "Accept"
    elif "Reject" in text:
        return "Reject"

    return "Unknown"


def load_predictions(predictions_path: Path) -> List[Dict]:
    """Load predictions from jsonl file."""
    predictions = []
    with open(predictions_path, "r") as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def load_ground_truth(test_data_path: str) -> List[str]:
    """Load ground truth labels from test dataset."""
    with open(test_data_path, "r") as f:
        data = json.load(f)

    labels = []
    for sample in data:
        assistant_response = sample["conversations"][-1]["value"]
        if "Accept" in assistant_response:
            labels.append("Accept")
        elif "Reject" in assistant_response:
            labels.append("Reject")
        else:
            labels.append("Unknown")

    return labels


def compute_metrics(predictions: List[str], ground_truth: List[str]) -> Dict:
    """Compute evaluation metrics."""
    # Convert to binary (1=Accept, 0=Reject)
    y_true = [1 if gt == "Accept" else 0 for gt in ground_truth]
    y_pred = [1 if pred == "Accept" else 0 for pred in predictions]

    # Remove unknowns
    valid_indices = [
        i for i, (gt, pred) in enumerate(zip(ground_truth, predictions)) if gt != "Unknown" and pred != "Unknown"
    ]

    y_true = [y_true[i] for i in valid_indices]
    y_pred = [y_pred[i] for i in valid_indices]

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "accept_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accept_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "reject_precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "reject_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "predicted_accept_rate": np.mean(y_pred),
        "actual_accept_rate": np.mean(y_true),
        "num_valid": len(y_true),
        "num_unknowns": len(predictions) - len(y_true),
    }

    return metrics


def evaluate_experiment(experiment_name: str) -> Dict:
    """Evaluate a single experiment."""
    print(f"\nEvaluating: {experiment_name}")

    predictions_path = RESULTS_DIR / experiment_name / "predictions.jsonl"
    if not predictions_path.exists():
        print(f"  Predictions not found: {predictions_path}")
        return None

    # Load predictions
    predictions_data = load_predictions(predictions_path)

    # Extract predictions (from "predict" field)
    predictions = []
    for item in predictions_data:
        # Handle n_generations=1 format
        pred_text = item["predict"]
        if isinstance(pred_text, list):
            pred_text = pred_text[0]
        predictions.append(parse_prediction(pred_text))

    # Load ground truth
    ground_truth = load_ground_truth(TEST_DATA_PATH)

    # Ensure lengths match
    assert len(predictions) == len(ground_truth), (
        f"Length mismatch: {len(predictions)} predictions vs {len(ground_truth)} ground truth"
    )

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth)

    print(f"  Accuracy:          {metrics['accuracy']:.3f}")
    print(f"  Accept precision:  {metrics['accept_precision']:.3f}")
    print(f"  Accept recall:     {metrics['accept_recall']:.3f}")
    print(f"  Reject precision:  {metrics['reject_precision']:.3f}")
    print(f"  Reject recall:     {metrics['reject_recall']:.3f}")
    print(f"  F1:                {metrics['f1']:.3f}")
    print(f"  Predicted accept rate: {metrics['predicted_accept_rate']:.3f}")
    print(f"  Valid answers:     {metrics['num_valid']}  (unknowns: {metrics['num_unknowns']})")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Specific experiment to evaluate (e.g., accept_gamma2_prop1_2)",
    )
    args = parser.parse_args()

    # Create metrics directory
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all experiments
    if args.experiment:
        experiments = [args.experiment]
    else:
        # Find all result directories
        experiments = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()]
        experiments.sort()

    print(f"Evaluating {len(experiments)} experiments...")

    # Evaluate all experiments
    all_metrics = {}
    for exp_name in experiments:
        metrics = evaluate_experiment(exp_name)
        if metrics is not None:
            all_metrics[exp_name] = metrics

    # Save aggregate metrics
    output_path = METRICS_DIR / "all_metrics.json"
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n\nMetrics saved to: {output_path}")
    print(f"Total evaluated: {len(all_metrics)}/{len(experiments)}")


if __name__ == "__main__":
    main()
