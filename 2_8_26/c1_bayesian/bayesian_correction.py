#!/usr/bin/env python3
"""
C1: Apply Bayesian decision correction to model predictions.

Uses estimated confusion matrix priors to compute:
    P(True Label | Prediction) via Bayes' rule

Then finds the optimal decision threshold to maximize accuracy.

Usage:
    python 2_8_26/c1_bayesian/bayesian_correction.py \
        --input results.jsonl \
        --priors 2_8_26/c1_bayesian/results/priors.json \
        --output 2_8_26/c1_bayesian/results/corrected_results.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.analysis_utils import compute_accuracy, compute_metrics_summary, load_results, save_metrics_json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def apply_bayes(prediction, priors, base_rate_accept=0.3):
    """Apply Bayes' rule to compute P(True Accept | Prediction).

    P(True Accept | Pred Accept) = P(Pred Accept | True Accept) * P(True Accept)
                                   / P(Pred Accept)

    Args:
        prediction: "Accept" or "Reject"
        priors: Dict with conditional probabilities
        base_rate_accept: Prior probability of acceptance (ICLR ~30%)

    Returns:
        P(True Accept | Prediction)
    """
    base_rate_reject = 1.0 - base_rate_accept

    p_pred_a_given_ta = priors["P_pred_accept_given_true_accept"]
    p_pred_a_given_tr = priors["P_pred_accept_given_true_reject"]
    p_pred_r_given_ta = priors["P_pred_reject_given_true_accept"]
    p_pred_r_given_tr = priors["P_pred_reject_given_true_reject"]

    if prediction == "Accept":
        # P(True Accept | Pred Accept)
        numerator = p_pred_a_given_ta * base_rate_accept
        denominator = (p_pred_a_given_ta * base_rate_accept +
                       p_pred_a_given_tr * base_rate_reject)
    elif prediction == "Reject":
        # P(True Accept | Pred Reject)
        numerator = p_pred_r_given_ta * base_rate_accept
        denominator = (p_pred_r_given_ta * base_rate_accept +
                       p_pred_r_given_tr * base_rate_reject)
    else:
        return None

    if denominator == 0:
        return None

    return numerator / denominator


def correct_predictions(results, priors, threshold=0.5, base_rate_accept=0.3):
    """Apply Bayesian correction to all predictions.

    For each prediction, compute P(True Accept | Prediction).
    If P(True Accept | Prediction) >= threshold → Accept, else Reject.
    """
    corrected = []
    for r in results:
        pred = r.get("prediction")
        gt = r.get("ground_truth")

        p_accept = apply_bayes(pred, priors, base_rate_accept)

        corrected_pred = None
        if p_accept is not None:
            corrected_pred = "Accept" if p_accept >= threshold else "Reject"

        corrected.append({
            "original_prediction": pred,
            "corrected_prediction": corrected_pred,
            "p_true_accept": p_accept,
            "ground_truth": gt,
            "prediction": corrected_pred,  # For compatibility with analysis_utils
            "correct": corrected_pred == gt if (corrected_pred and gt) else None,
        })

    return corrected


def find_optimal_threshold(results, priors, base_rate_accept=0.3):
    """Grid search for the threshold that maximizes accuracy."""
    best_threshold = 0.5
    best_accuracy = 0

    thresholds = np.arange(0.1, 0.9, 0.01)
    accuracies = []

    for t in thresholds:
        corrected = correct_predictions(results, priors, threshold=t,
                                        base_rate_accept=base_rate_accept)
        acc = compute_accuracy(corrected)
        accuracies.append(acc)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = t

    return best_threshold, best_accuracy, list(thresholds), accuracies


def main():
    parser = argparse.ArgumentParser(description="C1: Bayesian decision correction")
    parser.add_argument("--input", type=str, required=True,
                        help="Input results JSONL file")
    parser.add_argument("--priors", type=str, default=None,
                        help="Priors JSON file (from estimate_priors.py)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output corrected results JSONL file")
    parser.add_argument("--base_rate", type=float, default=0.3,
                        help="Prior acceptance rate (default: 0.3 for ICLR)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (if None, find optimal)")
    # Manual priors (used if --priors file not provided)
    parser.add_argument("--fpr", type=float, default=None,
                        help="Manual FPR: P(Pred Accept | True Reject)")
    parser.add_argument("--fnr", type=float, default=None,
                        help="Manual FNR: P(Pred Reject | True Accept)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load results
    results = load_results(args.input)
    print(f"Loaded {len(results)} results from {args.input}")

    # Load or construct priors
    if args.priors:
        with open(args.priors, "r") as f:
            priors = json.load(f)
        print(f"Loaded priors from {args.priors}")
    elif args.fpr is not None and args.fnr is not None:
        priors = {
            "P_pred_accept_given_true_accept": 1.0 - args.fnr,
            "P_pred_accept_given_true_reject": args.fpr,
            "P_pred_reject_given_true_accept": args.fnr,
            "P_pred_reject_given_true_reject": 1.0 - args.fpr,
        }
        print(f"Using manual priors: FPR={args.fpr}, FNR={args.fnr}")
    else:
        # Use working_plan_3.md estimates
        priors = {
            "P_pred_accept_given_true_accept": 0.5,
            "P_pred_accept_given_true_reject": 0.2,
            "P_pred_reject_given_true_accept": 0.5,
            "P_pred_reject_given_true_reject": 0.8,
        }
        print("Using default priors from working_plan_3.md:")
        print(f"  P(Pred Reject | True Accept) ≈ 50% (Harshness)")
        print(f"  P(Pred Accept | True Reject) ≈ 20% (Optimism)")

    # Original metrics
    orig_metrics = compute_metrics_summary(results)
    print(f"\nOriginal Metrics:")
    print(f"  Accuracy:        {orig_metrics['accuracy']:.1%}")
    print(f"  Acceptance Rate: {orig_metrics['acceptance_rate']:.1%}")

    # Find optimal threshold or use specified one
    if args.threshold is not None:
        threshold = args.threshold
        corrected = correct_predictions(results, priors, threshold, args.base_rate)
        accuracy = compute_accuracy(corrected)
    else:
        print("\nSearching for optimal threshold...")
        threshold, accuracy, thresholds, accuracies = find_optimal_threshold(
            results, priors, args.base_rate
        )
        corrected = correct_predictions(results, priors, threshold, args.base_rate)

    corr_metrics = compute_metrics_summary(corrected)
    print(f"\nCorrected Metrics (threshold={threshold:.2f}):")
    print(f"  Accuracy:        {corr_metrics['accuracy']:.1%}")
    print(f"  Acceptance Rate: {corr_metrics['acceptance_rate']:.1%}")
    print(f"  Accept Recall:   {corr_metrics['accept_recall']:.1%}")
    print(f"  Reject Recall:   {corr_metrics['reject_recall']:.1%}")

    # Save corrected results
    output_path = args.output or os.path.join(OUTPUT_DIR, "corrected_results.jsonl")
    with open(output_path, "w") as f:
        for r in corrected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved corrected results to {output_path}")

    # Save summary
    summary = {
        "priors": priors,
        "base_rate_accept": args.base_rate,
        "optimal_threshold": threshold,
        "original_metrics": orig_metrics,
        "corrected_metrics": corr_metrics,
    }
    save_metrics_json(summary, os.path.join(OUTPUT_DIR, "correction_summary.json"))


if __name__ == "__main__":
    main()
