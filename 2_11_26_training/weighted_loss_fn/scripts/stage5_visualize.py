#!/usr/bin/env python3
"""
Stage 5: Visualize experiment results.

Creates plots:
1. Predicted acceptance rate vs gamma (grouped by variant and proportion)
2. Accuracy vs gamma (grouped by variant and proportion)
3. Precision-Recall curves per configuration
4. Heatmaps of metrics across experiment dimensions

Usage:
    python stage5_visualize.py
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Paths
METRICS_DIR = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_11_26_training/weighted_loss_fn/metrics")
PLOTS_DIR = METRICS_DIR / "plots"

# Plot styling
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def load_metrics() -> Dict:
    """Load all metrics from JSON file."""
    metrics_path = METRICS_DIR / "all_metrics.json"
    with open(metrics_path, "r") as f:
        return json.load(f)


def parse_experiment_name(exp_name: str) -> Dict:
    """
    Parse experiment name into components.

    Example: "accept_gamma2.0_prop1_2" -> {variant: "accept", gamma: 2.0, proportion: "1_2"}
    """
    parts = exp_name.split("_")

    # Extract variant
    variant = parts[0]

    # Extract gamma
    gamma_str = [p for p in parts if p.startswith("gamma")][0]
    gamma = float(gamma_str.replace("gamma", ""))

    # Extract proportion
    prop_idx = parts.index("prop")
    proportion = "_".join(parts[prop_idx + 1 :])

    return {
        "variant": variant,
        "gamma": gamma,
        "proportion": proportion,
    }


def create_dataframe(all_metrics: Dict) -> pd.DataFrame:
    """Convert metrics dict to pandas DataFrame."""
    rows = []

    for exp_name, metrics in all_metrics.items():
        exp_params = parse_experiment_name(exp_name)

        row = {
            "experiment": exp_name,
            **exp_params,
            **metrics,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_acceptance_rate_vs_gamma(df: pd.DataFrame, output_path: Path):
    """Plot predicted acceptance rate vs gamma, grouped by variant and proportion."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, variant in enumerate(["accept", "reject"]):
        ax = axes[i]
        df_variant = df[df["variant"] == variant]

        for proportion in sorted(df_variant["proportion"].unique()):
            df_prop = df_variant[df_variant["proportion"] == proportion]
            df_prop = df_prop.sort_values("gamma")

            ax.plot(
                df_prop["gamma"],
                df_prop["predicted_accept_rate"],
                marker="o",
                label=f"Proportion {proportion.replace('_', ':')}",
                linewidth=2,
            )

        ax.set_xlabel("Gamma", fontsize=14)
        ax.set_ylabel("Predicted Accept Rate", fontsize=14)
        ax.set_title(f"Variant: Weight {variant.capitalize()}s", fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_vs_gamma(df: pd.DataFrame, output_path: Path):
    """Plot accuracy vs gamma, grouped by variant and proportion."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, variant in enumerate(["accept", "reject"]):
        ax = axes[i]
        df_variant = df[df["variant"] == variant]

        for proportion in sorted(df_variant["proportion"].unique()):
            df_prop = df_variant[df_variant["proportion"] == proportion]
            df_prop = df_prop.sort_values("gamma")

            ax.plot(
                df_prop["gamma"],
                df_prop["accuracy"],
                marker="s",
                label=f"Proportion {proportion.replace('_', ':')}",
                linewidth=2,
            )

        ax.set_xlabel("Gamma", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_title(f"Variant: Weight {variant.capitalize()}s", fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_heatmap_metric(df: pd.DataFrame, metric: str, output_path: Path):
    """Plot heatmap of a metric across all experiment dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for i, variant in enumerate(["accept", "reject"]):
        ax = axes[i]
        df_variant = df[df["variant"] == variant]

        # Pivot for heatmap (gamma × proportion)
        pivot = df_variant.pivot(index="gamma", columns="proportion", values=metric)

        # Sort columns by proportion value
        prop_order = sorted(pivot.columns, key=lambda x: eval(x.replace("_", "/")))
        pivot = pivot[prop_order]

        # Rename columns for readability
        pivot.columns = [c.replace("_", ":") for c in pivot.columns]

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            ax=ax,
            cbar_kws={"label": metric.replace("_", " ").title()},
            vmin=df[metric].min(),
            vmax=df[metric].max(),
        )

        ax.set_title(f"Variant: Weight {variant.capitalize()}s", fontsize=16)
        ax.set_xlabel("Accept:Reject Proportion", fontsize=14)
        ax.set_ylabel("Gamma", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_precision_recall(df: pd.DataFrame, output_path: Path):
    """Plot precision vs recall for all experiments."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Color by variant, marker by gamma, size by proportion
    for variant in ["accept", "reject"]:
        df_variant = df[df["variant"] == variant]

        ax.scatter(
            df_variant["recall"],
            df_variant["precision"],
            label=f"Weight {variant.capitalize()}s",
            s=100,
            alpha=0.7,
        )

    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title("Precision-Recall Trade-off Across Experiments", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add diagonal line (F1=0.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="F1=0.5")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load metrics
    print("Loading metrics...")
    all_metrics = load_metrics()

    # Convert to DataFrame
    print("Converting to DataFrame...")
    df = create_dataframe(all_metrics)

    print(f"Loaded {len(df)} experiments")
    print(f"Variants: {df['variant'].unique()}")
    print(f"Gammas: {sorted(df['gamma'].unique())}")
    print(f"Proportions: {sorted(df['proportion'].unique())}")

    # Generate plots
    print("\nGenerating plots...")

    plot_acceptance_rate_vs_gamma(df, PLOTS_DIR / "acceptance_rate_vs_gamma.png")
    plot_accuracy_vs_gamma(df, PLOTS_DIR / "accuracy_vs_gamma.png")
    plot_precision_recall(df, PLOTS_DIR / "precision_recall.png")

    # Heatmaps for each metric
    for metric in ["accuracy", "predicted_accept_rate", "f1", "precision", "recall"]:
        plot_heatmap_metric(df, metric, PLOTS_DIR / f"heatmap_{metric}.png")

    print("\nVisualization complete!")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
