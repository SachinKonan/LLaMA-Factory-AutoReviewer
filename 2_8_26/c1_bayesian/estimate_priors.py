#!/usr/bin/env python3
"""
C1: Estimate confusion matrix priors from validation set predictions.

Computes:
- P(Pred Accept | True Accept) = True Positive Rate
- P(Pred Accept | True Reject) = False Positive Rate (optimism)
- P(Pred Reject | True Accept) = False Negative Rate (harshness)
- P(Pred Reject | True Reject) = True Negative Rate

These priors are used by bayesian_correction.py to apply Bayes' rule.

Usage:
    python 2_8_26/c1_bayesian/estimate_priors.py --input results.jsonl
    python 2_8_26/c1_bayesian/estimate_priors.py --input_dir inference_scaling/results
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.analysis_utils import compute_confusion_matrix, load_results, save_metrics_json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def estimate_priors_from_results(results):
    """Estimate confusion matrix rates from results."""
    cm = compute_confusion_matrix(results)
    total = sum(cm.values())

    # Actual class counts
    actual_accept = cm["TP"] + cm["FN"]
    actual_reject = cm["FP"] + cm["TN"]

    priors = {
        "confusion_matrix": cm,
        "total_samples": total,
        "actual_accept": actual_accept,
        "actual_reject": actual_reject,
        "base_rate_accept": actual_accept / total if total > 0 else 0,
        "base_rate_reject": actual_reject / total if total > 0 else 0,
    }

    # Conditional probabilities
    if actual_accept > 0:
        priors["P_pred_accept_given_true_accept"] = cm["TP"] / actual_accept
        priors["P_pred_reject_given_true_accept"] = cm["FN"] / actual_accept
    else:
        priors["P_pred_accept_given_true_accept"] = 0
        priors["P_pred_reject_given_true_accept"] = 0

    if actual_reject > 0:
        priors["P_pred_accept_given_true_reject"] = cm["FP"] / actual_reject
        priors["P_pred_reject_given_true_reject"] = cm["TN"] / actual_reject
    else:
        priors["P_pred_accept_given_true_reject"] = 0
        priors["P_pred_reject_given_true_reject"] = 0

    return priors


def estimate_from_directory(results_dir):
    """Estimate priors from all result files in a directory."""
    all_priors = {}

    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".jsonl"):
                path = os.path.join(root, f)
                rel_path = os.path.relpath(path, results_dir)

                results = load_results(path)
                if len(results) < 10:
                    continue

                priors = estimate_priors_from_results(results)
                all_priors[rel_path] = priors
                print(f"  {rel_path}: n={priors['total_samples']}, "
                      f"FPR={priors['P_pred_accept_given_true_reject']:.1%}, "
                      f"FNR={priors['P_pred_reject_given_true_accept']:.1%}")

    return all_priors


def main():
    parser = argparse.ArgumentParser(description="C1: Estimate Bayesian priors")
    parser.add_argument("--input", type=str, default=None,
                        help="Single results JSONL file")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of results to scan")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for priors")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.input:
        print(f"Estimating priors from: {args.input}")
        results = load_results(args.input)
        priors = estimate_priors_from_results(results)

        print(f"\nEstimated Confusion Matrix:")
        cm = priors["confusion_matrix"]
        print(f"  TP={cm['TP']}  FN={cm['FN']}  | True Accept={priors['actual_accept']}")
        print(f"  FP={cm['FP']}  TN={cm['TN']}  | True Reject={priors['actual_reject']}")
        print(f"\nConditional Probabilities:")
        print(f"  P(Pred Accept | True Accept) = {priors['P_pred_accept_given_true_accept']:.3f}")
        print(f"  P(Pred Accept | True Reject) = {priors['P_pred_accept_given_true_reject']:.3f}  [Optimism]")
        print(f"  P(Pred Reject | True Accept) = {priors['P_pred_reject_given_true_accept']:.3f}  [Harshness]")
        print(f"  P(Pred Reject | True Reject) = {priors['P_pred_reject_given_true_reject']:.3f}")

        output = args.output or os.path.join(OUTPUT_DIR, "priors.json")
        save_metrics_json(priors, output)

    elif args.input_dir:
        print(f"Scanning directory: {args.input_dir}")
        all_priors = estimate_from_directory(args.input_dir)

        output = args.output or os.path.join(OUTPUT_DIR, "all_priors.json")
        save_metrics_json(all_priors, output)

    else:
        # Default: scan inference_scaling results
        default_dir = "inference_scaling/results"
        if os.path.exists(default_dir):
            print(f"Scanning default directory: {default_dir}")
            all_priors = estimate_from_directory(default_dir)
            save_metrics_json(all_priors, os.path.join(OUTPUT_DIR, "all_priors.json"))
        else:
            print("No input specified and no default results found.")
            print("Usage: python estimate_priors.py --input results.jsonl")
            print("   or: python estimate_priors.py --input_dir inference_scaling/results")


if __name__ == "__main__":
    main()
