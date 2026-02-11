#!/usr/bin/env python3
"""
C1 Analysis: Bayesian correction before/after comparison.

Generates:
- roc_curve.png: ROC curve with optimal threshold
- confusion_before_after.png: Side-by-side confusion matrices

Usage:
    python 2_8_26/c1_bayesian/analyze.py
    python 2_8_26/c1_bayesian/analyze.py --input inference_scaling/results/clean/new/results_single.jsonl
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from shared.analysis_utils import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_metrics_summary,
    load_results,
    plot_confusion_matrix,
    save_metrics_json,
    setup_plot_style,
)
from bayesian_correction import apply_bayes, correct_predictions, find_optimal_threshold
from estimate_priors import estimate_priors_from_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")


def plot_threshold_curve(thresholds, accuracies, optimal_threshold, save_path):
    """Plot accuracy vs threshold curve."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds, accuracies, linewidth=2, color="#3498db")
    ax.axvline(x=optimal_threshold, color="red", linestyle="--",
               label=f"Optimal: {optimal_threshold:.2f}")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Accuracy")
    ax.set_title("C1: Bayesian Correction — Accuracy vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_before_after(orig_cm, corr_cm, save_path):
    """Plot side-by-side confusion matrices."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cm, title in [(ax1, orig_cm, "Before Correction"),
                           (ax2, corr_cm, "After Correction")]:
        matrix = np.array([[cm["TP"], cm["FN"]], [cm["FP"], cm["TN"]]])
        im = ax.imshow(matrix, cmap="Blues")

        labels = ["Accept", "Reject"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(title)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=16)

    fig.suptitle("C1: Bayesian Correction — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="C1: Bayesian correction analysis")
    parser.add_argument("--input", type=str,
                        default="2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl",
                        help="Original results to correct")
    parser.add_argument("--base_rate", type=float, default=0.5,
                        help="Prior acceptance rate (0.5 for balanced dataset)")
    args = parser.parse_args()

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(args.input):
        print(f"Results file not found: {args.input}")
        print("Run inference first, then re-run this analysis.")
        return

    # Load results
    results = load_results(args.input)
    print(f"Loaded {len(results)} results from {args.input}")

    # Estimate priors from the data itself (self-estimate)
    priors = estimate_priors_from_results(results)
    print(f"\nEstimated priors:")
    print(f"  FPR (Optimism):  {priors['P_pred_accept_given_true_reject']:.1%}")
    print(f"  FNR (Harshness): {priors['P_pred_reject_given_true_accept']:.1%}")

    # Find optimal threshold
    optimal_threshold, best_acc, thresholds, accuracies = find_optimal_threshold(
        results, priors, args.base_rate
    )

    # Apply correction
    corrected = correct_predictions(results, priors, optimal_threshold, args.base_rate)

    # Metrics
    orig_metrics = compute_metrics_summary(results)
    corr_metrics = compute_metrics_summary(corrected)

    print(f"\n{'='*60}")
    print(f"Before Correction:  Accuracy={orig_metrics['accuracy']:.1%}, "
          f"Accept Rate={orig_metrics['acceptance_rate']:.1%}")
    print(f"After Correction:   Accuracy={corr_metrics['accuracy']:.1%}, "
          f"Accept Rate={corr_metrics['acceptance_rate']:.1%}")
    print(f"Optimal Threshold:  {optimal_threshold:.2f}")
    print(f"{'='*60}")

    # Plots
    plot_threshold_curve(thresholds, accuracies, optimal_threshold,
                         os.path.join(METRICS_DIR, "roc_curve.png"))
    plot_before_after(orig_metrics["confusion_matrix"], corr_metrics["confusion_matrix"],
                      os.path.join(METRICS_DIR, "confusion_before_after.png"))

    # Save
    save_metrics_json({
        "original": orig_metrics,
        "corrected": corr_metrics,
        "priors": priors,
        "optimal_threshold": optimal_threshold,
        "base_rate": args.base_rate,
    }, os.path.join(METRICS_DIR, "c1_metrics.json"))

    # Save corrected results
    with open(os.path.join(RESULTS_DIR, "corrected_results.jsonl"), "w") as f:
        for r in corrected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
