#!/usr/bin/env python3
"""
B1 Analysis: Compare PDR (meta-review) vs majority voting vs single strategy.

Generates:
- strategy_comparison.png: Accuracy of single vs majority vs PDR
- strategy_agreement.png: Pairwise agreement heatmap

Usage:
    python 2_8_26/b1_pdr/analyze.py
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.analysis_utils import (
    compute_acceptance_rate,
    compute_accuracy,
    compute_metrics_summary,
    load_results,
    save_metrics_json,
    setup_plot_style,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")

MODALITIES = ["clean", "clean_images", "vision"]
STRATEGIES = ["single", "majority", "metareview"]
STRATEGY_LABELS = {"single": "Single", "majority": "Majority\nVote", "metareview": "PDR\n(Meta-Review)"}


def load_strategy_results():
    """Load results for all modalities and strategies."""
    all_results = {}
    for modality in MODALITIES:
        all_results[modality] = {}
        for strategy in STRATEGIES:
            path = os.path.join(RESULTS_DIR, modality, f"results_{strategy}.jsonl")
            if os.path.exists(path):
                all_results[modality][strategy] = load_results(path)
                print(f"  Loaded {modality}/{strategy}: {len(all_results[modality][strategy])} results")
            else:
                print(f"  Missing: {modality}/{strategy}")
    return all_results


def plot_strategy_comparison(all_results, save_path):
    """Plot accuracy comparison across strategies and modalities."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(MODALITIES))
    width = 0.25
    colors = ["#3498db", "#e67e22", "#2ecc71"]

    for i, strategy in enumerate(STRATEGIES):
        accuracies = []
        for modality in MODALITIES:
            if strategy in all_results.get(modality, {}):
                accuracies.append(compute_accuracy(all_results[modality][strategy]))
            else:
                accuracies.append(0)

        bars = ax.bar(x + i * width, accuracies, width, label=STRATEGY_LABELS[strategy],
                      color=colors[i], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, accuracies):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", "\n") for m in MODALITIES])
    ax.set_ylabel("Accuracy")
    ax.set_title("B1: Strategy Comparison — Single vs Majority vs PDR (Meta-Review)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_strategy_agreement(all_results, save_path):
    """Plot pairwise agreement heatmap between strategies."""
    setup_plot_style()
    fig, axes = plt.subplots(1, len(MODALITIES), figsize=(5 * len(MODALITIES), 5))
    if len(MODALITIES) == 1:
        axes = [axes]

    for ax, modality in zip(axes, MODALITIES):
        results = all_results.get(modality, {})
        n_strategies = len(STRATEGIES)
        agreement = np.zeros((n_strategies, n_strategies))

        for i, s1 in enumerate(STRATEGIES):
            for j, s2 in enumerate(STRATEGIES):
                if s1 in results and s2 in results:
                    r1 = results[s1]
                    r2 = results[s2]
                    n = min(len(r1), len(r2))
                    if n > 0:
                        agree = sum(1 for a, b in zip(r1[:n], r2[:n])
                                    if a.get("prediction") == b.get("prediction"))
                        agreement[i, j] = agree / n

        im = ax.imshow(agreement, cmap="YlGn", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(n_strategies))
        ax.set_yticks(range(n_strategies))
        ax.set_xticklabels([STRATEGY_LABELS[s].replace("\n", " ") for s in STRATEGIES], fontsize=9)
        ax.set_yticklabels([STRATEGY_LABELS[s].replace("\n", " ") for s in STRATEGIES], fontsize=9)
        ax.set_title(modality.replace("_", " ").title())

        for i in range(n_strategies):
            for j in range(n_strategies):
                ax.text(j, i, f"{agreement[i, j]:.1%}", ha="center", va="center", fontsize=10)

    fig.suptitle("B1: Pairwise Strategy Agreement", fontsize=14)
    fig.colorbar(im, ax=axes, label="Agreement Rate", shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="B1: PDR analysis")
    args = parser.parse_args()

    os.makedirs(METRICS_DIR, exist_ok=True)

    print("Loading results...")
    all_results = load_strategy_results()

    # Compute metrics
    all_metrics = {}
    print(f"\n{'='*60}")
    print("B1: PDR Strategy Metrics")
    print(f"{'='*60}")

    for modality in MODALITIES:
        all_metrics[modality] = {}
        if modality not in all_results:
            continue

        for strategy in STRATEGIES:
            if strategy in all_results[modality]:
                metrics = compute_metrics_summary(all_results[modality][strategy])
                all_metrics[modality][strategy] = metrics
                print(f"\n{modality}/{strategy}:")
                print(f"  Accuracy:        {metrics['accuracy']:.1%}")
                print(f"  Acceptance Rate: {metrics['acceptance_rate']:.1%}")

    # Plots
    print(f"\n{'='*60}")
    print("Generating plots...")
    plot_strategy_comparison(all_results, os.path.join(METRICS_DIR, "strategy_comparison.png"))
    plot_strategy_agreement(all_results, os.path.join(METRICS_DIR, "strategy_agreement.png"))

    save_metrics_json(all_metrics, os.path.join(METRICS_DIR, "b1_metrics.json"))
    print("\nDone!")


if __name__ == "__main__":
    main()
