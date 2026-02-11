#!/usr/bin/env python3
"""
H1 Analysis: Compare base model vs instruct model predictions.

Generates:
- acceptance_rate_base_vs_instruct.png: Acceptance rate comparison
- accuracy_base_vs_instruct.png: Accuracy comparison
- confusion_base.png / confusion_instruct.png: Confusion matrices
- disagreement_analysis.png: Analysis of papers where models disagree
- venn_agreement.png: Venn-style diagram of agree/disagree × correct/incorrect

Usage:
    python 2_8_26/h1_base_model/analyze.py
    python 2_8_26/h1_base_model/analyze.py --instruct_results 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.analysis_utils import (
    compute_acceptance_rate,
    compute_accuracy,
    compute_confusion_matrix,
    compute_metrics_summary,
    load_results,
    plot_bar_comparison,
    plot_confusion_matrix,
    save_metrics_json,
    setup_plot_style,
)

METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")

# Default instruct results: B2 standard/clean uses instruct model on same v7 split
DEFAULT_INSTRUCT = "2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl"


def align_results(base_results, instruct_results):
    """Align results by index, keeping only entries where both have valid predictions."""
    aligned_base = []
    aligned_inst = []
    for b, inst in zip(base_results, instruct_results):
        if b.get("prediction") and inst.get("prediction"):
            aligned_base.append(b)
            aligned_inst.append(inst)
    return aligned_base, aligned_inst


def compute_agreement_stats(base_results, instruct_results):
    """Compute detailed agreement/disagreement statistics."""
    stats = {
        "both_correct": [],
        "both_wrong": [],
        "base_correct_instruct_wrong": [],
        "base_wrong_instruct_correct": [],
        "both_accept_correct": [],   # both predict accept, GT=accept
        "both_accept_wrong": [],     # both predict accept, GT=reject
        "both_reject_correct": [],   # both predict reject, GT=reject
        "both_reject_wrong": [],     # both predict reject, GT=accept
        "base_accept_instruct_reject_gt_accept": [],
        "base_accept_instruct_reject_gt_reject": [],
        "base_reject_instruct_accept_gt_accept": [],
        "base_reject_instruct_accept_gt_reject": [],
    }

    for i, (b, inst) in enumerate(zip(base_results, instruct_results)):
        b_pred = b.get("prediction")
        i_pred = inst.get("prediction")
        gt = b.get("ground_truth")

        b_correct = b_pred == gt
        i_correct = i_pred == gt
        agree = b_pred == i_pred

        entry = {"index": i, "base_pred": b_pred, "instruct_pred": i_pred,
                 "ground_truth": gt, "base_correct": b_correct, "instruct_correct": i_correct}

        if b_correct and i_correct:
            stats["both_correct"].append(entry)
        elif not b_correct and not i_correct:
            stats["both_wrong"].append(entry)
        elif b_correct and not i_correct:
            stats["base_correct_instruct_wrong"].append(entry)
        else:
            stats["base_wrong_instruct_correct"].append(entry)

        # Detailed agreement breakdown
        if agree:
            if b_pred == "Accept":
                if gt == "Accept":
                    stats["both_accept_correct"].append(entry)
                else:
                    stats["both_accept_wrong"].append(entry)
            else:
                if gt == "Reject":
                    stats["both_reject_correct"].append(entry)
                else:
                    stats["both_reject_wrong"].append(entry)
        else:
            if b_pred == "Accept" and i_pred == "Reject":
                if gt == "Accept":
                    stats["base_accept_instruct_reject_gt_accept"].append(entry)
                else:
                    stats["base_accept_instruct_reject_gt_reject"].append(entry)
            elif b_pred == "Reject" and i_pred == "Accept":
                if gt == "Accept":
                    stats["base_reject_instruct_accept_gt_accept"].append(entry)
                else:
                    stats["base_reject_instruct_accept_gt_reject"].append(entry)

    return stats


def plot_venn_agreement(stats, total, save_path):
    """Plot a Venn-style diagram showing agreement/disagreement × correct/incorrect.

    Four quadrants:
    - Both correct (agree, correct)
    - Both wrong (agree, wrong)
    - Only base correct (disagree, base right)
    - Only instruct correct (disagree, instruct right)
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    n_both_correct = len(stats["both_correct"])
    n_both_wrong = len(stats["both_wrong"])
    n_base_only = len(stats["base_correct_instruct_wrong"])
    n_inst_only = len(stats["base_wrong_instruct_correct"])

    # Draw two overlapping circles
    circle_base = plt.Circle((-0.35, 0), 1.0, fill=False, edgecolor="#e74c3c",
                              linewidth=3, linestyle="-", label="Base Correct")
    circle_inst = plt.Circle((0.35, 0), 1.0, fill=False, edgecolor="#3498db",
                              linewidth=3, linestyle="-", label="Instruct Correct")
    ax.add_patch(circle_base)
    ax.add_patch(circle_inst)

    # Fill regions with semi-transparent colors
    # Overlap (both correct) - green
    from matplotlib.patches import FancyBboxPatch
    ax.text(0, 0.15, f"{n_both_correct}", fontsize=28, fontweight="bold",
            ha="center", va="center", color="#27ae60")
    ax.text(0, -0.15, f"Both Correct", fontsize=13, ha="center", va="center", color="#27ae60")
    ax.text(0, -0.40, f"({n_both_correct/total:.1%})", fontsize=11, ha="center", va="center",
            color="#27ae60", style="italic")

    # Base only correct (left crescent)
    ax.text(-0.95, 0.15, f"{n_base_only}", fontsize=24, fontweight="bold",
            ha="center", va="center", color="#e74c3c")
    ax.text(-0.95, -0.15, f"Only Base\nCorrect", fontsize=11, ha="center", va="center",
            color="#e74c3c")
    ax.text(-0.95, -0.50, f"({n_base_only/total:.1%})", fontsize=10, ha="center", va="center",
            color="#e74c3c", style="italic")

    # Instruct only correct (right crescent)
    ax.text(0.95, 0.15, f"{n_inst_only}", fontsize=24, fontweight="bold",
            ha="center", va="center", color="#3498db")
    ax.text(0.95, -0.15, f"Only Instruct\nCorrect", fontsize=11, ha="center", va="center",
            color="#3498db")
    ax.text(0.95, -0.50, f"({n_inst_only/total:.1%})", fontsize=10, ha="center", va="center",
            color="#3498db", style="italic")

    # Outside both circles (both wrong)
    ax.text(0, -1.25, f"{n_both_wrong} Both Wrong ({n_both_wrong/total:.1%})",
            fontsize=16, fontweight="bold", ha="center", va="center", color="#7f8c8d")

    # Labels
    ax.text(-1.35, 1.05, "Base Model", fontsize=14, fontweight="bold", color="#e74c3c")
    ax.text(0.75, 1.05, "Instruct Model", fontsize=14, fontweight="bold", color="#3498db")

    # Title with agreement stats
    n_agree = n_both_correct + n_both_wrong
    n_disagree = n_base_only + n_inst_only
    ax.set_title(f"H1: Base vs Instruct Agreement (n={total})\n"
                 f"Agree: {n_agree} ({n_agree/total:.1%})  |  "
                 f"Disagree: {n_disagree} ({n_disagree/total:.1%})",
                 fontsize=15, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_disagreement_detail(stats, save_path):
    """Plot detailed breakdown of disagreements split by ground truth correctness."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Disagreement counts by type and GT
    categories = [
        "Base Accept\nInstruct Reject\n(GT=Accept)",
        "Base Accept\nInstruct Reject\n(GT=Reject)",
        "Base Reject\nInstruct Accept\n(GT=Accept)",
        "Base Reject\nInstruct Accept\n(GT=Reject)",
    ]
    values = [
        len(stats["base_accept_instruct_reject_gt_accept"]),
        len(stats["base_accept_instruct_reject_gt_reject"]),
        len(stats["base_reject_instruct_accept_gt_accept"]),
        len(stats["base_reject_instruct_accept_gt_reject"]),
    ]
    # Green = base correct, Blue = instruct correct
    colors = ["#27ae60", "#e74c3c", "#3498db", "#7f8c8d"]
    bar_labels = ["Base ✓", "Both ✗", "Instruct ✓", "Both ✗"]

    bars = axes[0].bar(range(len(categories)), values, color=colors,
                        edgecolor="black", linewidth=0.5)
    for bar, val, bl in zip(bars, values, bar_labels):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val}\n({bl})", ha="center", va="bottom", fontsize=10)
    axes[0].set_xticks(range(len(categories)))
    axes[0].set_xticklabels(categories, fontsize=8)
    axes[0].set_ylabel("Number of Papers")
    axes[0].set_title("Disagreement Breakdown by Type & GT")

    # Right: Agreement pie chart
    n_agree_correct = len(stats["both_correct"])
    n_agree_wrong = len(stats["both_wrong"])
    n_base_only = len(stats["base_correct_instruct_wrong"])
    n_inst_only = len(stats["base_wrong_instruct_correct"])

    pie_labels = [f"Both Correct\n({n_agree_correct})",
                  f"Both Wrong\n({n_agree_wrong})",
                  f"Only Base Correct\n({n_base_only})",
                  f"Only Instruct Correct\n({n_inst_only})"]
    pie_values = [n_agree_correct, n_agree_wrong, n_base_only, n_inst_only]
    pie_colors = ["#27ae60", "#7f8c8d", "#e74c3c", "#3498db"]

    axes[1].pie(pie_values, labels=pie_labels, colors=pie_colors,
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    axes[1].set_title("Agreement/Correctness Distribution")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="H1: Base vs Instruct model analysis")
    parser.add_argument("--base_results", type=str,
                        default="2_8_26/h1_base_model/results/results_single.jsonl",
                        help="Path to base model results")
    parser.add_argument("--instruct_results", type=str,
                        default=DEFAULT_INSTRUCT,
                        help="Path to instruct model results (same split as base)")
    args = parser.parse_args()

    os.makedirs(METRICS_DIR, exist_ok=True)

    # Load results
    print("Loading base model results...")
    base_results = load_results(args.base_results)
    print(f"  Loaded {len(base_results)} base model results")

    print("Loading instruct model results...")
    if not os.path.exists(args.instruct_results):
        print(f"  Warning: Instruct results not found at {args.instruct_results}")
        print("  Running base-only analysis...")
        instruct_results = None
    else:
        instruct_results = load_results(args.instruct_results)
        print(f"  Loaded {len(instruct_results)} instruct model results")

    # Compute metrics for each model independently
    base_metrics = compute_metrics_summary(base_results)
    print(f"\nBase Model Metrics:")
    print(f"  Accuracy: {base_metrics['accuracy']:.1%}")
    print(f"  Acceptance Rate: {base_metrics['acceptance_rate']:.1%}")
    print(f"  Accept Recall: {base_metrics['accept_recall']:.1%}")
    print(f"  Reject Recall: {base_metrics['reject_recall']:.1%}")

    all_metrics = {"base_model": base_metrics}

    if instruct_results:
        instruct_metrics = compute_metrics_summary(instruct_results)
        print(f"\nInstruct Model Metrics:")
        print(f"  Accuracy: {instruct_metrics['accuracy']:.1%}")
        print(f"  Acceptance Rate: {instruct_metrics['acceptance_rate']:.1%}")
        print(f"  Accept Recall: {instruct_metrics['accept_recall']:.1%}")
        print(f"  Reject Recall: {instruct_metrics['reject_recall']:.1%}")

        all_metrics["instruct_model"] = instruct_metrics

        # Plot 1: Acceptance rate comparison
        plot_bar_comparison(
            {"Base Model\n(Qwen2.5-7B)": base_metrics["acceptance_rate"],
             "Instruct Model\n(Qwen2.5-7B-Instruct)": instruct_metrics["acceptance_rate"]},
            "H1: Acceptance Rate — Base vs Instruct",
            "Acceptance Rate",
            os.path.join(METRICS_DIR, "acceptance_rate_base_vs_instruct.png"),
            ylim=(0, 1.1),
        )

        # Plot 2: Accuracy comparison
        plot_bar_comparison(
            {"Base Model": base_metrics["accuracy"],
             "Instruct Model": instruct_metrics["accuracy"]},
            "H1: Accuracy — Base vs Instruct",
            "Accuracy",
            os.path.join(METRICS_DIR, "accuracy_base_vs_instruct.png"),
            ylim=(0, 1.1),
        )

        # Plot 3: Confusion matrices
        plot_confusion_matrix(base_metrics["confusion_matrix"],
                              "Base Model Confusion Matrix",
                              os.path.join(METRICS_DIR, "confusion_base.png"))
        plot_confusion_matrix(instruct_metrics["confusion_matrix"],
                              "Instruct Model Confusion Matrix",
                              os.path.join(METRICS_DIR, "confusion_instruct.png"))

        # Align results for agreement analysis (only entries where both have predictions)
        if len(base_results) == len(instruct_results):
            aligned_base, aligned_inst = align_results(base_results, instruct_results)
            print(f"\nAligned {len(aligned_base)} papers with valid predictions from both models")

            # Compute agreement stats
            stats = compute_agreement_stats(aligned_base, aligned_inst)
            total = len(aligned_base)

            n_agree = len(stats["both_correct"]) + len(stats["both_wrong"])
            n_disagree = len(stats["base_correct_instruct_wrong"]) + len(stats["base_wrong_instruct_correct"])

            print(f"\n{'='*60}")
            print(f"AGREEMENT ANALYSIS (n={total})")
            print(f"{'='*60}")
            print(f"  Models agree: {n_agree} ({n_agree/total:.1%})")
            print(f"    Both correct:  {len(stats['both_correct'])} ({len(stats['both_correct'])/total:.1%})")
            print(f"    Both wrong:    {len(stats['both_wrong'])} ({len(stats['both_wrong'])/total:.1%})")
            print(f"  Models disagree: {n_disagree} ({n_disagree/total:.1%})")
            print(f"    Only base correct:     {len(stats['base_correct_instruct_wrong'])} ({len(stats['base_correct_instruct_wrong'])/total:.1%})")
            print(f"    Only instruct correct: {len(stats['base_wrong_instruct_correct'])} ({len(stats['base_wrong_instruct_correct'])/total:.1%})")

            print(f"\n  Disagreement detail:")
            print(f"    Base=Accept, Instruct=Reject, GT=Accept: {len(stats['base_accept_instruct_reject_gt_accept'])}")
            print(f"    Base=Accept, Instruct=Reject, GT=Reject: {len(stats['base_accept_instruct_reject_gt_reject'])}")
            print(f"    Base=Reject, Instruct=Accept, GT=Accept: {len(stats['base_reject_instruct_accept_gt_accept'])}")
            print(f"    Base=Reject, Instruct=Accept, GT=Reject: {len(stats['base_reject_instruct_accept_gt_reject'])}")

            all_metrics["agreement"] = {
                "total_aligned": total,
                "both_correct": len(stats["both_correct"]),
                "both_wrong": len(stats["both_wrong"]),
                "base_only_correct": len(stats["base_correct_instruct_wrong"]),
                "instruct_only_correct": len(stats["base_wrong_instruct_correct"]),
                "agreement_rate": n_agree / total,
                "disagreement_detail": {
                    "base_accept_instruct_reject_gt_accept": len(stats["base_accept_instruct_reject_gt_accept"]),
                    "base_accept_instruct_reject_gt_reject": len(stats["base_accept_instruct_reject_gt_reject"]),
                    "base_reject_instruct_accept_gt_accept": len(stats["base_reject_instruct_accept_gt_accept"]),
                    "base_reject_instruct_accept_gt_reject": len(stats["base_reject_instruct_accept_gt_reject"]),
                },
            }

            # Plot 4: Venn-style agreement diagram
            plot_venn_agreement(stats, total,
                                os.path.join(METRICS_DIR, "venn_agreement.png"))

            # Plot 5: Detailed disagreement analysis
            plot_disagreement_detail(stats,
                                      os.path.join(METRICS_DIR, "disagreement_analysis.png"))
        else:
            print(f"\nWarning: Result counts differ ({len(base_results)} vs {len(instruct_results)})")
            print("  Cannot do per-paper agreement analysis. Use --instruct_results with same-split results.")
    else:
        plot_confusion_matrix(base_metrics["confusion_matrix"],
                              "Base Model Confusion Matrix",
                              os.path.join(METRICS_DIR, "confusion_base.png"))

    # Save all metrics
    save_metrics_json(all_metrics, os.path.join(METRICS_DIR, "h1_metrics.json"))
    print("\nDone!")


if __name__ == "__main__":
    main()
