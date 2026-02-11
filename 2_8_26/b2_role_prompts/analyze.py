#!/usr/bin/env python3
"""
B2 Analysis: Role-playing prompts and Strategy D analysis.

Generates:
- acceptance_rate_by_role.png: Acceptance rate per role
- soundness_presentation_grid.png: 2×2 heatmap per role (if score data available)
- strategy_d_comparison.png: Strategy D vs individual roles

Usage:
    python 2_8_26/b2_role_prompts/analyze.py
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
    plot_bar_comparison,
    save_metrics_json,
    setup_plot_style,
)
from shared.prompt_templates import (
    CRITICAL_MODIFIER,
    ENTHUSIASTIC_MODIFIER,
    STANDARD_MODIFIER,
    STRATEGY_D_SYSTEM_PROMPT,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")

MODALITIES = ["clean", "clean_images", "vision"]
ROLES = ["critical", "enthusiastic", "standard"]
ROLE_COLORS = {
    "critical": "#e74c3c",
    "enthusiastic": "#2ecc71",
    "standard": "#3498db",
    "strategy_d": "#9b59b6",
    "strategy_d_critical": "#8e44ad"
}

# Short versions of prompts for legend (truncate to first sentence)
ROLE_PROMPTS = {
    "critical": CRITICAL_MODIFIER.split('.')[0].replace("IMPORTANT: ", "").strip(),
    "enthusiastic": ENTHUSIASTIC_MODIFIER.split('.')[0].replace("IMPORTANT: ", "").strip(),
    "standard": "No modifier (baseline)",
    "strategy_d": "Synthesize critical + enthusiastic perspectives",
    "strategy_d_critical": "Weigh weaknesses vs strengths, reject when unclear"
}


def load_all_results():
    """Load results for all modality/role combinations."""
    all_results = {}
    for modality in MODALITIES:
        all_results[modality] = {}
        for role in ROLES + ["strategy_d", "strategy_d_critical"]:
            path = os.path.join(RESULTS_DIR, modality, role, "results_single.jsonl")
            if os.path.exists(path):
                all_results[modality][role] = load_results(path)
                print(f"  Loaded {modality}/{role}: {len(all_results[modality][role])} results")
    return all_results


def plot_acceptance_rate_by_role(all_results, save_path):
    """Plot acceptance rate by role across modalities."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(MODALITIES))
    all_labels = ROLES + ["strategy_d", "strategy_d_critical"]
    width = 0.16

    for i, role in enumerate(all_labels):
        rates = []
        for modality in MODALITIES:
            if role in all_results.get(modality, {}):
                rates.append(compute_acceptance_rate(all_results[modality][role]))
            else:
                rates.append(0)

        # Create legend label with prompt
        role_name = role.replace("_", " ").title()
        prompt = ROLE_PROMPTS.get(role, "")
        legend_label = f"{role_name}\n({prompt})" if prompt else role_name

        bars = ax.bar(x + i * width, rates, width, label=legend_label,
                      color=ROLE_COLORS.get(role, "gray"), edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, rates):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.0%}", ha="center", va="bottom", fontsize=8, rotation=45)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([m.replace("_", "\n") for m in MODALITIES])
    ax.set_ylabel("Acceptance Rate", fontsize=12)
    ax.set_title("B2: Acceptance Rate by Reviewer Role", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.axhline(y=0.3, color="blue", linestyle="--", alpha=0.5, label="ICLR baseline (~30%)")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_strategy_d_comparison(all_results, save_path):
    """Plot Strategy D vs individual roles."""
    setup_plot_style()

    n_modalities = sum(1 for m in MODALITIES if m in all_results and all_results[m])
    if n_modalities == 0:
        print("No results to plot for Strategy D comparison")
        return

    fig, axes = plt.subplots(1, n_modalities, figsize=(6 * n_modalities, 6), squeeze=False)

    plot_idx = 0
    for modality in MODALITIES:
        if modality not in all_results or not all_results[modality]:
            continue

        ax = axes[0, plot_idx]
        results = all_results[modality]

        labels = []
        accuracies = []
        colors = []

        for role in ROLES + ["strategy_d", "strategy_d_critical"]:
            if role in results:
                labels.append(role.replace("_", " ").title())
                accuracies.append(compute_accuracy(results[role]))
                colors.append(ROLE_COLORS.get(role, "gray"))

        bars = ax.bar(labels, accuracies, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylabel("Accuracy")
        ax.set_title(modality.replace("_", " ").title())
        ax.set_ylim(0, max(accuracies) * 1.2 if accuracies else 1)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=30)

        plot_idx += 1

    fig.suptitle("B2: Accuracy by Role (incl. Strategy D)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="B2: Role prompts analysis")
    args = parser.parse_args()

    os.makedirs(METRICS_DIR, exist_ok=True)

    print("Loading results...")
    all_results = load_all_results()

    # Compute metrics
    all_metrics = {}
    print(f"\n{'='*60}")
    print("B2: Role Prompt Metrics")
    print(f"{'='*60}")

    for modality in MODALITIES:
        all_metrics[modality] = {}
        for role in ROLES + ["strategy_d", "strategy_d_critical"]:
            if role in all_results.get(modality, {}):
                metrics = compute_metrics_summary(all_results[modality][role])
                all_metrics[modality][role] = metrics
                print(f"\n{modality}/{role}:")
                print(f"  Accuracy:        {metrics['accuracy']:.1%}")
                print(f"  Acceptance Rate: {metrics['acceptance_rate']:.1%}")
                print(f"  Accept Recall:   {metrics['accept_recall']:.1%}")
                print(f"  Reject Recall:   {metrics['reject_recall']:.1%}")

    # Plots
    print(f"\n{'='*60}")
    print("Generating plots...")

    plot_acceptance_rate_by_role(all_results,
                                 os.path.join(METRICS_DIR, "acceptance_rate_by_role.png"))
    plot_strategy_d_comparison(all_results,
                                os.path.join(METRICS_DIR, "strategy_d_comparison.png"))

    save_metrics_json(all_metrics, os.path.join(METRICS_DIR, "b2_metrics.json"))
    print("\nDone!")


if __name__ == "__main__":
    main()
