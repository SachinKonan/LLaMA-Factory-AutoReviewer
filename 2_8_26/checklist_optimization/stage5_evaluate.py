#!/usr/bin/env python3
"""
Stage 5: Evaluate Optimal Checklist.

Applies the optimized checklist to paper-level evaluations (aggregated from
real ICLR reviews) and computes comprehensive metrics:
- Overall accuracy, precision, recall, F1
- Per-question correlation with ground truth
- Comparison with baseline (raw LLM predictions from B2 standard)

Usage:
    # Full evaluation
    python 2_8_26/checklist_optimization/stage5_evaluate.py \
        --checklist 2_8_26/checklist_optimization/data/optimal_checklist.json \
        --evaluations 2_8_26/checklist_optimization/data/paper_evaluations.jsonl \
        --output_metrics 2_8_26/checklist_optimization/metrics/checklist_metrics.json \
        --output_predictions 2_8_26/checklist_optimization/results/final_predictions.jsonl

    # Debug mode
    python 2_8_26/checklist_optimization/stage5_evaluate.py \
        --checklist 2_8_26/checklist_optimization/data/optimal_checklist.json \
        --evaluations 2_8_26/checklist_optimization/data/paper_evaluations.jsonl \
        --output_metrics 2_8_26/checklist_optimization/metrics/checklist_metrics.json \
        --output_predictions 2_8_26/checklist_optimization/results/final_predictions.jsonl \
        --debug
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    compute_accuracy,
    compute_point_biserial,
    load_json,
    load_jsonl,
    load_results,
    save_json,
    save_jsonl,
)


# ============================================================================
# Checklist Application
# ============================================================================

def apply_checklist(
    checklist: dict,
    evaluations: list[dict],
) -> tuple[list[str], list[float]]:
    """Apply checklist to evaluations to generate predictions.

    Args:
        checklist: Optimal checklist dict with questions and threshold
        evaluations: List of evaluation entries with answers

    Returns:
        Tuple of (predictions, checklist_scores)
    """
    question_ids = [q["id"] for q in checklist["questions"]]
    threshold = checklist.get("threshold", 0.5)

    checklist_scores = []
    predictions = []

    for eval_entry in evaluations:
        answers = eval_entry.get("answers", {})

        # Count yes answers
        yes_count = sum(1 for qid in question_ids if answers.get(qid, "No") == "Yes")
        score = yes_count / len(question_ids) if question_ids else 0.0

        checklist_scores.append(score)
        prediction = "Accept" if score >= threshold else "Reject"
        predictions.append(prediction)

    return predictions, checklist_scores


def optimize_threshold(
    checklist_scores: list[float],
    ground_truth: list[str],
) -> tuple[float, float]:
    """Find optimal threshold that maximizes accuracy.

    Args:
        checklist_scores: Checklist scores (fraction of yes answers)
        ground_truth: Ground truth labels

    Returns:
        Tuple of (optimal_threshold, accuracy_at_threshold)
    """
    # Try different thresholds
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_accuracy = 0.0

    for threshold in thresholds:
        predictions = ["Accept" if score >= threshold else "Reject" for score in checklist_scores]
        accuracy = compute_accuracy(predictions, ground_truth)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_per_question_correlation(
    checklist: dict,
    evaluations: list[dict],
    ground_truth: list[str],
) -> dict:
    """Compute point-biserial correlation for each question.

    Args:
        checklist: Optimal checklist dict
        evaluations: List of evaluation entries
        ground_truth: Ground truth labels

    Returns:
        Dict mapping question ID to correlation metrics
    """
    correlations = {}
    binary_gt = [1 if gt == "Accept" else 0 for gt in ground_truth]

    for q in checklist["questions"]:
        qid = q["id"]

        # Extract answers for this question
        answers = [eval_entry["answers"].get(qid, "No") for eval_entry in evaluations]
        binary_answers = [1 if a == "Yes" else 0 for a in answers]

        # Compute point-biserial correlation
        corr, pval = compute_point_biserial(binary_answers, binary_gt)

        correlations[qid] = {
            "correlation": float(corr),
            "p_value": float(pval),
            "text": q["text"],
            "category": q.get("category", "unknown"),
            "fraction_yes": sum(binary_answers) / len(binary_answers),
        }

    return correlations


def compute_metrics(
    predictions: list[str],
    ground_truth: list[str],
) -> dict:
    """Compute classification metrics.

    Args:
        predictions: Predicted labels
        ground_truth: Ground truth labels

    Returns:
        Dict with accuracy, precision, recall, F1
    """
    # Confusion matrix
    tp = sum(1 for p, gt in zip(predictions, ground_truth) if p == "Accept" and gt == "Accept")
    tn = sum(1 for p, gt in zip(predictions, ground_truth) if p == "Reject" and gt == "Reject")
    fp = sum(1 for p, gt in zip(predictions, ground_truth) if p == "Accept" and gt == "Reject")
    fn = sum(1 for p, gt in zip(predictions, ground_truth) if p == "Reject" and gt == "Accept")

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        }
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 5: Evaluate optimal checklist")
    parser.add_argument(
        "--checklist",
        type=str,
        required=True,
        help="Path to optimal_checklist.json",
    )
    parser.add_argument(
        "--evaluations",
        type=str,
        required=True,
        help="Path to paper_evaluations.jsonl (from stage 4, contains answers + decision)",
    )
    parser.add_argument(
        "--input_results",
        type=str,
        default=None,
        help="Optional path to results_single.jsonl (if not using paper_evaluations decision field)",
    )
    parser.add_argument(
        "--output_metrics",
        type=str,
        required=True,
        help="Output path for metrics JSON",
    )
    parser.add_argument(
        "--output_predictions",
        type=str,
        required=True,
        help="Output path for predictions JSONL",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )

    args = parser.parse_args()

    if args.debug:
        print("DEBUG MODE")

    # Load inputs
    print(f"\nLoading checklist from: {args.checklist}")
    checklist = load_json(args.checklist)
    print(f"  Checklist has {len(checklist['questions'])} questions")

    print(f"\nLoading evaluations from: {args.evaluations}")
    evaluations = load_jsonl(args.evaluations)
    print(f"  Loaded {len(evaluations)} evaluations")

    # Get ground truth from evaluations (paper_evaluations.jsonl has decision field)
    # or from a separate results file
    if args.input_results:
        print(f"\nLoading ground truth from: {args.input_results}")
        results = load_results(args.input_results)
        ground_truth = [r.get("ground_truth", "") for r in results]
    else:
        print(f"\nExtracting ground truth from evaluations (decision field)...")
        ground_truth = [e.get("decision", "") for e in evaluations]
    print(f"  Loaded {len(ground_truth)} ground truth labels")

    # Ensure alignment
    if len(evaluations) != len(ground_truth):
        print(f"\nWarning: Mismatched lengths (evaluations={len(evaluations)}, ground_truth={len(ground_truth)})")
        min_len = min(len(evaluations), len(ground_truth))
        evaluations = evaluations[:min_len]
        ground_truth = ground_truth[:min_len]
        print(f"  Truncated to {min_len} samples")

    # Apply checklist with default threshold
    print(f"\nApplying checklist (default threshold={checklist.get('threshold', 0.5)})...")
    predictions, checklist_scores = apply_checklist(checklist, evaluations)

    # Optimize threshold
    print("\nOptimizing threshold...")
    optimal_threshold, optimal_accuracy = optimize_threshold(checklist_scores, ground_truth)
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print(f"  Accuracy at optimal threshold: {optimal_accuracy:.4f}")

    # Update checklist with optimal threshold
    checklist["threshold"] = optimal_threshold

    # Re-apply with optimal threshold
    predictions = ["Accept" if score >= optimal_threshold else "Reject" for score in checklist_scores]

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, ground_truth)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")

    # Per-question correlation
    print("\nComputing per-question correlations...")
    correlations = compute_per_question_correlation(checklist, evaluations, ground_truth)

    # Sort by absolute correlation
    sorted_questions = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]["correlation"]),
        reverse=True
    )

    print("\nTop 10 most correlated questions:")
    for i, (qid, stats) in enumerate(sorted_questions[:10], 1):
        print(f"  {i}. [{qid}] corr={stats['correlation']:+.3f}, p={stats['p_value']:.4f}")
        print(f"     {stats['text']}")

    # Build final metrics dict
    final_metrics = {
        "overall": metrics,
        "optimal_threshold": optimal_threshold,
        "question_correlations": correlations,
        "checklist": {
            "n_questions": len(checklist["questions"]),
            "questions": checklist["questions"],
        }
    }

    # Save metrics
    print(f"\nSaving metrics to: {args.output_metrics}")
    save_json(final_metrics, args.output_metrics)

    # Save predictions
    print(f"Saving predictions to: {args.output_predictions}")
    prediction_entries = []
    for i, (pred, score) in enumerate(zip(predictions, checklist_scores)):
        prediction_entries.append({
            "review_idx": i,
            "prediction": pred,
            "ground_truth": ground_truth[i],
            "correct": pred == ground_truth[i],
            "checklist_score": score,
        })
    save_jsonl(prediction_entries, args.output_predictions)

    print("\nDone!")


if __name__ == "__main__":
    main()
