#!/usr/bin/env python3
"""
Analyze and Plot Checklist Optimization Results.

Generates 4 plots:
1. Optimization curve: beam search progress
2. Question importance: top questions by correlation
3. Correlation heatmap: questions vs. accept/reject
4. Comparison plot: checklist vs. LLM baseline

Usage:
    # Default paths
    python 2_8_26/checklist_optimization/analyze.py

    # Custom paths
    python 2_8_26/checklist_optimization/analyze.py \
        --checklist data/optimal_checklist.json \
        --metrics metrics/checklist_metrics.json \
        --baseline_results 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_json, load_jsonl, load_results


# ============================================================================
# Plot 1: Optimization Curve
# ============================================================================

def plot_optimization_curve(trace_path: str, output_path: str):
    """Plot beam search optimization progress.

    Args:
        trace_path: Path to beam_search_trace.jsonl
        output_path: Output path for PNG
    """
    print(f"\nGenerating optimization curve...")

    # Load trace
    trace = load_jsonl(trace_path)

    steps = [t["step"] for t in trace]
    scores = [t["best_score"] for t in trace]
    sizes = [len(t["best_subset"]) for t in trace]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Score vs. step
    ax1.plot(steps, scores, marker="o", linewidth=2)
    ax1.set_xlabel("Beam Search Step")
    ax1.set_ylabel("Composite Score")
    ax1.set_title("Beam Search Optimization Progress")
    ax1.grid(alpha=0.3)

    # Subset size vs. step
    ax2.plot(steps, sizes, marker="s", color="orange", linewidth=2)
    ax2.set_xlabel("Beam Search Step")
    ax2.set_ylabel("Checklist Size (# Questions)")
    ax2.set_title("Checklist Size Growth")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to: {output_path}")
    plt.close()


# ============================================================================
# Plot 2: Question Importance Rankings
# ============================================================================

def plot_question_importance(metrics: dict, output_path: str, top_k: int = 20):
    """Plot top questions by correlation with ground truth.

    Args:
        metrics: Metrics dict with question_correlations
        output_path: Output path for PNG
        top_k: Number of top questions to show
    """
    print(f"\nGenerating question importance rankings...")

    # Extract correlations
    correlations = metrics.get("question_correlations", {})
    if not correlations:
        print("  Warning: No question correlations found")
        return

    # Sort by absolute correlation
    sorted_items = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]["correlation"]),
        reverse=True
    )[:top_k]

    # Prepare data
    question_texts = []
    corr_values = []
    categories = []

    for qid, stats in sorted_items:
        # Truncate long question texts
        text = stats.get("text", qid)
        if len(text) > 60:
            text = text[:57] + "..."
        question_texts.append(text)
        corr_values.append(stats["correlation"])
        categories.append(stats.get("category", "unknown"))

    # Color map by category
    category_colors = {
        "novelty": "#1f77b4",
        "soundness": "#ff7f0e",
        "clarity": "#2ca02c",
        "validation": "#d62728",
        "impact": "#9467bd",
        "unknown": "#8c564b",
    }
    colors = [category_colors.get(cat, "#8c564b") for cat in categories]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(question_texts))
    bars = ax.barh(y_pos, corr_values, color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(question_texts, fontsize=9)
    ax.set_xlabel("Point-Biserial Correlation")
    ax.set_title(f"Top {top_k} Questions by Correlation with Accept/Reject")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    # Legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7)
                       for color in category_colors.values()]
    ax.legend(legend_elements, category_colors.keys(), loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to: {output_path}")
    plt.close()


# ============================================================================
# Plot 3: Correlation Heatmap
# ============================================================================

def plot_correlation_heatmap(metrics: dict, output_path: str):
    """Plot heatmap of question correlations.

    Args:
        metrics: Metrics dict with question_correlations
        output_path: Output path for PNG
    """
    print(f"\nGenerating correlation heatmap...")

    # Extract correlations
    correlations = metrics.get("question_correlations", {})
    if not correlations:
        print("  Warning: No question correlations found")
        return

    # Sort by correlation (descending)
    sorted_items = sorted(
        correlations.items(),
        key=lambda x: x[1]["correlation"],
        reverse=True
    )

    # Prepare data
    question_labels = []
    corr_values = []
    yes_fractions = []

    for qid, stats in sorted_items:
        # Truncate question text
        text = stats.get("text", qid)
        if len(text) > 50:
            text = text[:47] + "..."
        question_labels.append(f"[{qid}] {text}")
        corr_values.append(stats["correlation"])
        yes_fractions.append(stats["fraction_yes"])

    # Create 2-column matrix (correlation, yes_fraction)
    data = np.array([corr_values, yes_fractions]).T

    # Plot
    fig, ax = plt.subplots(figsize=(8, max(10, len(question_labels) * 0.3)))

    sns.heatmap(
        data,
        xticklabels=["Correlation", "Fraction Yes"],
        yticklabels=question_labels,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Value"},
        ax=ax,
    )

    ax.set_title("Question Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to: {output_path}")
    plt.close()


# ============================================================================
# Plot 4: Comparison Plot
# ============================================================================

def plot_comparison(metrics: dict, baseline_results_path: str, output_path: str):
    """Plot checklist performance vs. LLM baseline.

    Args:
        metrics: Checklist metrics dict
        baseline_results_path: Path to baseline results_single.jsonl
        output_path: Output path for PNG
    """
    print(f"\nGenerating comparison plot...")

    # Load baseline
    baseline_results = load_results(baseline_results_path)
    baseline_correct = sum(1 for r in baseline_results if r.get("correct", False))
    baseline_accuracy = baseline_correct / len(baseline_results)

    baseline_tp = sum(1 for r in baseline_results
                      if r.get("prediction") == "Accept" and r.get("ground_truth") == "Accept")
    baseline_tn = sum(1 for r in baseline_results
                      if r.get("prediction") == "Reject" and r.get("ground_truth") == "Reject")
    baseline_fp = sum(1 for r in baseline_results
                      if r.get("prediction") == "Accept" and r.get("ground_truth") == "Reject")
    baseline_fn = sum(1 for r in baseline_results
                      if r.get("prediction") == "Reject" and r.get("ground_truth") == "Accept")

    baseline_precision = baseline_tp / (baseline_tp + baseline_fp) if (baseline_tp + baseline_fp) > 0 else 0
    baseline_recall = baseline_tp / (baseline_tp + baseline_fn) if (baseline_tp + baseline_fn) > 0 else 0

    # Checklist metrics
    checklist_metrics = metrics.get("overall", {})
    checklist_accuracy = checklist_metrics.get("accuracy", 0)
    checklist_precision = checklist_metrics.get("precision", 0)
    checklist_recall = checklist_metrics.get("recall", 0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ["Accuracy", "Precision", "Recall"]
    baseline_values = [baseline_accuracy, baseline_precision, baseline_recall]
    checklist_values = [checklist_accuracy, checklist_precision, checklist_recall]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, baseline_values, width, label="LLM Baseline", alpha=0.8)
    bars2 = ax.bar(x + width / 2, checklist_values, width, label="Optimized Checklist", alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Checklist vs. LLM Baseline Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze checklist optimization results")
    parser.add_argument(
        "--checklist",
        type=str,
        default="2_8_26/checklist_optimization/data/optimal_checklist.json",
        help="Path to optimal_checklist.json",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="2_8_26/checklist_optimization/metrics/checklist_metrics.json",
        help="Path to checklist_metrics.json",
    )
    parser.add_argument(
        "--baseline_results",
        type=str,
        default="2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl",
        help="Path to baseline results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="2_8_26/checklist_optimization/metrics",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metrics
    print(f"Loading metrics from: {args.metrics}")
    metrics = load_json(args.metrics)

    # Plot 1: Optimization curve
    trace_path = args.checklist.replace("data/optimal_checklist.json", "results/beam_search_trace.jsonl")
    if os.path.exists(trace_path):
        plot_optimization_curve(
            trace_path,
            os.path.join(args.output_dir, "optimization_curve.png")
        )
    else:
        print(f"\nWarning: Trace file not found: {trace_path}")

    # Plot 2: Question importance
    plot_question_importance(
        metrics,
        os.path.join(args.output_dir, "question_importance.png")
    )

    # Plot 3: Correlation heatmap
    plot_correlation_heatmap(
        metrics,
        os.path.join(args.output_dir, "correlation_heatmap.png")
    )

    # Plot 4: Comparison plot
    if os.path.exists(args.baseline_results):
        plot_comparison(
            metrics,
            args.baseline_results,
            os.path.join(args.output_dir, "comparison_plot.png")
        )
    else:
        print(f"\nWarning: Baseline results not found: {args.baseline_results}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
