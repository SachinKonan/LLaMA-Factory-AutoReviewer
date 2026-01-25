#!/usr/bin/env python3
"""
Generate model comparison figures for the LaTeX report.

Generates:
- data_variant_bars.pdf: Comparison of data variants (balanced, original, trainagreeing)
- modality_comparison.pdf: Comparison of modalities (clean, vision, clean_images)
- all_models_comparison.pdf: Complete comparison of all model configurations

Uses:
- results/subset_analysis.csv - Main results file

Usage:
    python scripts/latex_analysis/fig_model_comparison.py
"""

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Matplotlib styling
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

# Sizes
LABELSIZE = 14
TITLESIZE = 16
LEGENDSIZE = 12
TICKSIZE = 12

# Colors
CLEAN_COLOR = "#2ecc71"
VISION_COLOR = "#e74c3c"
CLEAN_IMAGES_COLOR = "#3498db"
BALANCED_COLOR = "#27ae60"
ORIGINAL_COLOR = "#e67e22"
TRAINAGREEING_COLOR = "#9b59b6"


def parse_combined_metrics(val: str) -> dict:
    """Parse combined column format: acc/accept_recall/reject_recall."""
    try:
        parts = val.split("/")
        return {
            "accuracy": float(parts[0]),
            "accept_recall": float(parts[1]),
            "reject_recall": float(parts[2]),
        }
    except:
        return {"accuracy": 0, "accept_recall": 0, "reject_recall": 0}


def parse_year_metrics(val: str) -> dict:
    """Parse year column format: acc/acc_rec/rej_rec(n=N)."""
    try:
        # Extract metrics and count
        match = re.match(r"([\d.]+)/([\d.]+)/([\d.]+)\(n=(\d+)\)", val)
        if match:
            return {
                "accuracy": float(match.group(1)),
                "accept_recall": float(match.group(2)),
                "reject_recall": float(match.group(3)),
                "count": int(match.group(4)),
            }
        return None
    except:
        return None


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load results from subset_analysis.csv."""
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    return df


def categorize_result(result_name: str) -> dict:
    """Categorize result by data variant and modality."""
    categories = {}

    # Data variant
    if "original" in result_name.lower():
        categories["variant"] = "original"
    elif "trainagreeing" in result_name.lower():
        categories["variant"] = "trainagreeing"
    elif "balanced" in result_name.lower():
        categories["variant"] = "balanced"
    else:
        categories["variant"] = "unknown"

    # Modality
    if "clean_images" in result_name.lower():
        categories["modality"] = "clean_images"
    elif "vision" in result_name.lower():
        categories["modality"] = "vision"
    elif "clean" in result_name.lower():
        categories["modality"] = "clean"
    else:
        categories["modality"] = "unknown"

    # Year range
    if "iclr17" in result_name.lower():
        categories["year_range"] = "2017-2025"
    elif "iclr20" in result_name.lower() or "2020_2025" in result_name.lower():
        categories["year_range"] = "2020-2025"
    else:
        categories["year_range"] = "unknown"

    return categories


def plot_data_variant_comparison(df: pd.DataFrame, output_path: Path):
    """Plot comparison of data variants (balanced, original, trainagreeing)."""
    # Filter to (full) subset and iclr20 results
    df_full = df[(df["subset"] == "(full)") & (df["result"].str.contains("iclr20", case=False))]

    if len(df_full) == 0:
        print("No data for variant comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Group by variant and modality
    results_by_variant = {}
    for _, row in df_full.iterrows():
        cat = categorize_result(row["result"])
        variant = cat["variant"]
        modality = cat["modality"]

        if variant not in results_by_variant:
            results_by_variant[variant] = {}

        metrics = parse_combined_metrics(row["combined"])
        results_by_variant[variant][modality] = metrics

    # Left: Accuracy comparison
    ax1 = axes[0]
    variants = ["balanced", "original", "trainagreeing"]
    modalities = ["clean", "vision"]
    variant_colors = {"balanced": BALANCED_COLOR, "original": ORIGINAL_COLOR, "trainagreeing": TRAINAGREEING_COLOR}

    x = np.arange(len(modalities))
    width = 0.25
    multiplier = 0

    for variant in variants:
        if variant not in results_by_variant:
            continue

        accuracies = [results_by_variant[variant].get(m, {}).get("accuracy", 0) for m in modalities]
        offset = width * multiplier
        bars = ax1.bar(x + offset, accuracies, width, label=variant.capitalize(),
                      color=variant_colors.get(variant, "#888888"), alpha=0.8)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{acc:.2f}", ha='center', fontsize=TICKSIZE - 1)

        multiplier += 1

    ax1.set_ylabel("Accuracy", fontsize=LABELSIZE)
    ax1.set_title("Accuracy by Data Variant and Modality", fontsize=TITLESIZE)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(["Text", "Vision"], fontsize=LABELSIZE)
    ax1.legend(fontsize=LEGENDSIZE)
    ax1.tick_params(axis='both', labelsize=TICKSIZE)
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax1.set_ylim(0.5, 0.8)

    # Right: Accept/Reject recall comparison
    ax2 = axes[1]

    # Plot accept recall (solid) and reject recall (hatched) for balanced clean
    metrics_list = []
    labels = []

    for variant in variants:
        if variant in results_by_variant and "clean" in results_by_variant[variant]:
            m = results_by_variant[variant]["clean"]
            metrics_list.append(m)
            labels.append(variant.capitalize())

    x = np.arange(len(labels))
    width = 0.35

    acc_recall = [m.get("accept_recall", 0) for m in metrics_list]
    rej_recall = [m.get("reject_recall", 0) for m in metrics_list]

    bars1 = ax2.bar(x - width/2, acc_recall, width, label='Accept Recall', color=CLEAN_COLOR, alpha=0.8)
    bars2 = ax2.bar(x + width/2, rej_recall, width, label='Reject Recall', color=VISION_COLOR, alpha=0.8)

    # Add value labels
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha='center', fontsize=TICKSIZE - 1)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha='center', fontsize=TICKSIZE - 1)

    ax2.set_ylabel("Recall", fontsize=LABELSIZE)
    ax2.set_title("Accept/Reject Recall by Data Variant (Text)", fontsize=TITLESIZE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=LABELSIZE)
    ax2.legend(fontsize=LEGENDSIZE)
    ax2.tick_params(axis='both', labelsize=TICKSIZE)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax2.set_ylim(0.5, 0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_modality_comparison(df: pd.DataFrame, output_path: Path):
    """Plot comparison of modalities (clean, vision, clean_images)."""
    # Filter to (full) subset and balanced results
    df_balanced = df[(df["subset"] == "(full)") & (df["result"].str.contains("balanced", case=False))]

    if len(df_balanced) == 0:
        print("No data for modality comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect metrics by modality
    modality_metrics = {}
    for _, row in df_balanced.iterrows():
        cat = categorize_result(row["result"])
        modality = cat["modality"]

        if modality == "unknown":
            continue

        metrics = parse_combined_metrics(row["combined"])
        if modality not in modality_metrics or metrics["accuracy"] > modality_metrics[modality]["accuracy"]:
            modality_metrics[modality] = metrics

    modalities = ["clean", "vision", "clean_images"]
    display_names = ["Text", "Vision", "Text+Images"]
    modality_colors = {"clean": CLEAN_COLOR, "vision": VISION_COLOR, "clean_images": CLEAN_IMAGES_COLOR}

    x = np.arange(len(modalities))
    width = 0.25

    # Plot accuracy, accept recall, reject recall
    available_modalities = [m for m in modalities if m in modality_metrics]
    display_available = [display_names[modalities.index(m)] for m in available_modalities]

    x = np.arange(len(available_modalities))

    accuracies = [modality_metrics[m]["accuracy"] for m in available_modalities]
    acc_recalls = [modality_metrics[m]["accept_recall"] for m in available_modalities]
    rej_recalls = [modality_metrics[m]["reject_recall"] for m in available_modalities]

    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='#34495e', alpha=0.9)
    bars2 = ax.bar(x, acc_recalls, width, label='Accept Recall', color=CLEAN_COLOR, alpha=0.9)
    bars3 = ax.bar(x + width, rej_recalls, width, label='Reject Recall', color=VISION_COLOR, alpha=0.9)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"{bar.get_height():.2f}", ha='center', fontsize=TICKSIZE - 1)

    ax.set_ylabel("Score", fontsize=LABELSIZE)
    ax.set_title("Model Performance by Modality (Balanced Data)", fontsize=TITLESIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(display_available, fontsize=LABELSIZE)
    ax.legend(fontsize=LEGENDSIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_ylim(0.5, 0.85)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_models_comparison(df: pd.DataFrame, output_path: Path):
    """Plot comparison of all model configurations."""
    # Filter to (full) subset
    df_full = df[df["subset"] == "(full)"].copy()

    if len(df_full) == 0:
        print("No data for all models comparison")
        return

    # Parse metrics
    df_full["accuracy"] = df_full["combined"].apply(lambda x: parse_combined_metrics(x)["accuracy"])

    # Sort by accuracy
    df_full = df_full.sort_values("accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(14, max(6, len(df_full) * 0.4)))

    # Create labels
    labels = []
    colors = []
    for _, row in df_full.iterrows():
        cat = categorize_result(row["result"])
        label = f"{row['result']}"
        labels.append(label)

        # Color by modality
        if cat["modality"] == "vision":
            colors.append(VISION_COLOR)
        elif cat["modality"] == "clean_images":
            colors.append(CLEAN_IMAGES_COLOR)
        else:
            colors.append(CLEAN_COLOR)

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, df_full["accuracy"], color=colors, alpha=0.8)

    # Add value labels
    for bar, acc in zip(bars, df_full["accuracy"]):
        ax.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f"{acc:.2f}",
               va='center', fontsize=TICKSIZE - 1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=TICKSIZE - 2)
    ax.set_xlabel("Accuracy", fontsize=LABELSIZE)
    ax.set_title("All Model Configurations Comparison", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    ax.set_xlim(0.6, 0.75)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CLEAN_COLOR, label='Text'),
        Patch(facecolor=VISION_COLOR, label='Vision'),
        Patch(facecolor=CLEAN_IMAGES_COLOR, label='Text+Images'),
    ]
    ax.legend(handles=legend_elements, fontsize=LEGENDSIZE, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_indist_vs_ood(df: pd.DataFrame, output_path: Path):
    """Plot in-distribution vs OOD comparison."""
    # Filter to (full) subset
    df_full = df[df["subset"] == "(full)"].copy()

    if len(df_full) == 0:
        print("No data for in-dist vs OOD comparison")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Parse in_dist and ood columns
    results = []
    for _, row in df_full.iterrows():
        in_dist = parse_year_metrics(row.get("in_dist", ""))
        ood = parse_year_metrics(row.get("ood", ""))

        if in_dist and ood:
            cat = categorize_result(row["result"])
            results.append({
                "name": row["result"],
                "modality": cat["modality"],
                "variant": cat["variant"],
                "in_dist": in_dist["accuracy"],
                "ood": ood["accuracy"],
                "gap": in_dist["accuracy"] - ood["accuracy"],
            })

    if not results:
        print("No in-dist/OOD data available")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("ood", ascending=True)

    x = np.arange(len(results_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, results_df["in_dist"], width, label='In-Distribution', color=CLEAN_COLOR, alpha=0.8)
    bars2 = ax.bar(x + width/2, results_df["ood"], width, label='OOD (2025)', color=VISION_COLOR, alpha=0.8)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f"{bar.get_height():.2f}", ha='center', fontsize=TICKSIZE - 2, rotation=90)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f"{bar.get_height():.2f}", ha='center', fontsize=TICKSIZE - 2, rotation=90)

    ax.set_ylabel("Accuracy", fontsize=LABELSIZE)
    ax.set_title("In-Distribution vs OOD (2025) Performance", fontsize=TITLESIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([r[:25] for r in results_df["name"]], fontsize=TICKSIZE - 2, rotation=45, ha='right')
    ax.legend(fontsize=LEGENDSIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_ylim(0.55, 0.80)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    CSV_PATH = Path("results/subset_analysis.csv")
    OUTPUT_DIR = Path("figures/latex/models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...")
    df = load_results(CSV_PATH)
    print(f"Loaded {len(df)} result entries")

    if len(df) == 0:
        print("No results found")
        return

    # Generate figures
    print("\nGenerating figures...")

    # 1. Data variant comparison
    plot_data_variant_comparison(df, OUTPUT_DIR / "data_variant_comparison.pdf")

    # 2. Modality comparison
    plot_modality_comparison(df, OUTPUT_DIR / "modality_comparison.pdf")

    # 3. All models comparison
    plot_all_models_comparison(df, OUTPUT_DIR / "all_models_comparison.pdf")

    # 4. In-dist vs OOD
    plot_indist_vs_ood(df, OUTPUT_DIR / "indist_vs_ood.pdf")

    # Print summary
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)

    df_full = df[df["subset"] == "(full)"].copy()
    if len(df_full) > 0:
        df_full["accuracy"] = df_full["combined"].apply(lambda x: parse_combined_metrics(x)["accuracy"])
        print("\nTop 5 configurations by accuracy:")
        for _, row in df_full.nlargest(5, "accuracy").iterrows():
            print(f"  {row['result']}: {row['accuracy']:.3f}")

    print("\n" + "="*60)
    print("Done!")


if __name__ == "__main__":
    main()
