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
    try:
        parts = exp_name.split("_")
        variant = parts[0]
        gamma = None
        proportion = None
        # Handle baseline
        if variant == "baseline":
            # e.g. baseline_gamma1.0_prop1_2
            gamma_str = [p for p in parts if p.startswith("gamma")]
            if gamma_str:
                gamma = float(gamma_str[0].replace("gamma", ""))
            prop_idx = next((i for i, p in enumerate(parts) if p.startswith("prop")), None)
            if prop_idx is not None:
                prop_num = parts[prop_idx].replace("prop", "")
                if prop_idx + 1 < len(parts) and parts[prop_idx + 1].isdigit():
                    proportion = f"{prop_num}_{parts[prop_idx + 1]}"
                else:
                    proportion = prop_num
        else:
            gamma_str = [p for p in parts if p.startswith("gamma")]
            if gamma_str:
                gamma = float(gamma_str[0].replace("gamma", ""))
            prop_idx = next((i for i, p in enumerate(parts) if p.startswith("prop")), None)
            if prop_idx is not None:
                prop_num = parts[prop_idx].replace("prop", "")
                if prop_idx + 1 < len(parts) and parts[prop_idx + 1].isdigit():
                    proportion = f"{prop_num}_{parts[prop_idx + 1]}"
                else:
                    proportion = prop_num
        if gamma is None or proportion is None:
            raise ValueError("Missing gamma or proportion")
        return {
            "variant": variant,
            "gamma": gamma,
            "proportion": proportion,
        }
    except (IndexError, ValueError) as e:
        print(f"  WARNING: Could not parse experiment name '{exp_name}': {e}")
        return None


def create_dataframe(all_metrics: Dict) -> pd.DataFrame:
    """Convert metrics dict to pandas DataFrame."""
    rows = []

    for exp_name, metrics in all_metrics.items():
        exp_params = parse_experiment_name(exp_name)
        if exp_params is None:
            continue

        row = {
            "experiment": exp_name,
            **exp_params,
            **metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Derived columns
    total = df["num_valid"]
    actual_accepts = df["actual_accept_rate"] * total
    predicted_accepts = df["predicted_accept_rate"] * total
    tp = df["accept_recall"] * actual_accepts  # TP for accept class
    tn = total - predicted_accepts - actual_accepts + tp
    actual_rejects = total - actual_accepts

    # accept_recall and reject_recall are already in metrics, skip recomputation
    df["acceptance_rate"] = df["predicted_accept_rate"]

    return df


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
        recall_col = f"{variant}_recall"
        precision_col = f"{variant}_precision"

        ax.scatter(
            df_variant[recall_col],
            df_variant[precision_col],
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
    for metric in ["accuracy", "predicted_accept_rate", "f1"]:
        plot_heatmap_metric(df, metric, PLOTS_DIR / f"heatmap_{metric}.png")

    # Custom gamma comparison plot
    plot_gamma_comparison(df, PLOTS_DIR / "gamma_comparison.png")

    print("\nVisualization complete!")
    print(f"Plots saved to: {PLOTS_DIR}")
def plot_gamma_comparison(df: pd.DataFrame, output_path: Path):
    """Plot 6 metrics with custom x-axis: accept/reject gammas and baseline."""
    # Define x-axis order and labels
    x_order = [
        ("accept", 8.0), ("accept", 4.0), ("accept", 3.0), ("accept", 2.0),
        ("baseline", 1.0),
        ("reject", 2.0), ("reject", 3.0), ("reject", 4.0), ("reject", 8.0)
    ]
    x_labels = [
        "accept_8", "accept_4", "accept_3", "accept_2",
        "baseline", "reject_2", "reject_3", "reject_4", "reject_8"
    ]

    # Only use prop1_2
    prop = "1_2"
    metrics = [
        ("accuracy", "Accuracy", "Accuracy"),
        ("reject_recall", "Reject Recall", "Recall (Reject)"),
        ("accept_recall", "Accept Recall", "Recall (Accept)"),
        ("f1", "F1 Score", "F1"),
        ("num_valid", "Number Valid", "Samples"),
        ("acceptance_rate", "Acceptance Rate", "Accept Rate"),
    ]

    # Prepare values for each metric
    values = {m[0]: [] for m in metrics}
    for variant, gamma in x_order:
        row = df[(df["variant"] == variant) & (df["gamma"] == gamma) & (df["proportion"] == prop)]
        if len(row) == 0:
            for m in metrics:
                values[m[0]].append(np.nan)
            continue
        row = row.iloc[0]
        for col, _, _ in metrics:
            values[col].append(row[col] if col in row.index else np.nan)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        ax.plot(x_labels, values[metric], marker="o", linewidth=2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Gamma Variant", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
