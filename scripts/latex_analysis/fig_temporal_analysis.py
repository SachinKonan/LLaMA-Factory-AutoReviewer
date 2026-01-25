#!/usr/bin/env python3
"""
Generate temporal analysis figures for the LaTeX report.

Generates:
- year_accuracy_line.pdf: Line plot of accuracy by year for different models
- indist_ood_bars.pdf: Grouped bars comparing in-distribution vs OOD
- year_heatmap.pdf: Heatmap of accuracy by model × year

Uses:
- results/subset_analysis.csv - Contains year-by-year breakdown

Usage:
    python scripts/latex_analysis/fig_temporal_analysis.py
"""

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def parse_year_metrics(val: str) -> dict | None:
    """Parse year column format: acc/acc_rec/rej_rec(n=N)."""
    if pd.isna(val) or val == "N/A":
        return None
    try:
        match = re.match(r"([\d.]+)/([\d.]+)/([\d.]+)\(n=(\d+)\)", str(val))
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

    # Modality
    if "clean_images" in result_name.lower():
        categories["modality"] = "clean_images"
    elif "vision" in result_name.lower():
        categories["modality"] = "vision"
    elif "clean" in result_name.lower():
        categories["modality"] = "clean"
    else:
        categories["modality"] = "unknown"

    # Data variant
    if "original" in result_name.lower():
        categories["variant"] = "original"
    elif "trainagreeing" in result_name.lower():
        categories["variant"] = "trainagreeing"
    elif "balanced" in result_name.lower():
        categories["variant"] = "balanced"
    else:
        categories["variant"] = "unknown"

    return categories


def extract_year_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year-by-year accuracy data from results."""
    year_columns = [c for c in df.columns if c.startswith("y20")]
    years = sorted([int(c[1:]) for c in year_columns])

    records = []
    for _, row in df.iterrows():
        if row["subset"] != "(full)":
            continue

        cat = categorize_result(row["result"])

        for year in years:
            col = f"y{year}"
            if col in df.columns:
                metrics = parse_year_metrics(row[col])
                if metrics:
                    records.append({
                        "result": row["result"],
                        "modality": cat["modality"],
                        "variant": cat["variant"],
                        "year": year,
                        "accuracy": metrics["accuracy"],
                        "accept_recall": metrics["accept_recall"],
                        "reject_recall": metrics["reject_recall"],
                        "count": metrics["count"],
                    })

    return pd.DataFrame(records)


def plot_year_accuracy_line(year_df: pd.DataFrame, output_path: Path):
    """Plot line chart of accuracy by year for different models."""
    if len(year_df) == 0:
        print("No year data available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Select representative models
    target_models = [
        ("iclr20_balanced_clean", "Balanced Clean", CLEAN_COLOR, "-"),
        ("iclr20_balanced_vision", "Balanced Vision", VISION_COLOR, "-"),
        ("iclr20_trainagreeing_clean", "Trainagreeing Clean", CLEAN_COLOR, "--"),
        ("iclr20_trainagreeing_vision", "Trainagreeing Vision", VISION_COLOR, "--"),
    ]

    for result_pattern, label, color, linestyle in target_models:
        df_model = year_df[year_df["result"].str.contains(result_pattern, case=False)]
        if len(df_model) == 0:
            continue

        df_model = df_model.sort_values("year")
        ax.plot(df_model["year"], df_model["accuracy"], marker='o', linestyle=linestyle,
               color=color, label=label, linewidth=2, markersize=8)

    ax.set_xlabel("Year", fontsize=LABELSIZE)
    ax.set_ylabel("Accuracy", fontsize=LABELSIZE)
    ax.set_title("Model Accuracy by Year", fontsize=TITLESIZE)
    ax.legend(fontsize=LEGENDSIZE, loc='upper right')
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(0.55, 0.85)

    # Mark 2025 as OOD
    ax.axvspan(2024.5, 2025.5, alpha=0.1, color='red', label='OOD')
    ax.axvline(2024.5, color='red', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_indist_ood_bars(df: pd.DataFrame, output_path: Path):
    """Plot grouped bars comparing in-distribution vs OOD."""
    # Filter to (full) subset
    df_full = df[df["subset"] == "(full)"].copy()

    if len(df_full) == 0:
        print("No data for in-dist vs OOD comparison")
        return

    # Parse in_dist and ood columns
    results = []
    for _, row in df_full.iterrows():
        in_dist = parse_year_metrics(row.get("in_dist", ""))
        ood = parse_year_metrics(row.get("ood", ""))

        if in_dist and ood:
            cat = categorize_result(row["result"])
            # Create short name
            short_name = f"{cat['variant']}_{cat['modality']}"
            results.append({
                "name": short_name,
                "full_name": row["result"],
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

    # Remove duplicates (keep first occurrence)
    results_df = results_df.drop_duplicates(subset=["name"])
    results_df = results_df.sort_values("ood", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(results_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, results_df["in_dist"], width, label='In-Distribution (<2025)',
                  color=CLEAN_COLOR, alpha=0.8)
    bars2 = ax.bar(x + width/2, results_df["ood"], width, label='OOD (2025)',
                  color=VISION_COLOR, alpha=0.8)

    # Add value labels and gap
    for i, (bar1, bar2, gap) in enumerate(zip(bars1, bars2, results_df["gap"])):
        ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
               f"{bar1.get_height():.2f}", ha='center', fontsize=TICKSIZE - 2)
        ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
               f"{bar2.get_height():.2f}", ha='center', fontsize=TICKSIZE - 2)

    ax.set_ylabel("Accuracy", fontsize=LABELSIZE)
    ax.set_title("In-Distribution vs OOD (2025) Performance", fontsize=TITLESIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["name"], fontsize=TICKSIZE - 1, rotation=30, ha='right')
    ax.legend(fontsize=LEGENDSIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_ylim(0.55, 0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_year_heatmap(year_df: pd.DataFrame, output_path: Path):
    """Plot heatmap of accuracy by model × year."""
    if len(year_df) == 0:
        print("No year data for heatmap")
        return

    # Create pivot table
    # Use short names for models
    year_df["short_name"] = year_df.apply(
        lambda r: f"{r['variant']}_{r['modality']}", axis=1
    )

    pivot = year_df.pivot_table(values="accuracy", index="short_name", columns="year", aggfunc="first")

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))

    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0.65,
               vmin=0.55, vmax=0.80, ax=ax, cbar_kws={"label": "Accuracy"})

    ax.set_xlabel("Year", fontsize=LABELSIZE)
    ax.set_ylabel("Model Configuration", fontsize=LABELSIZE)
    ax.set_title("Accuracy Heatmap by Model and Year", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)

    # Mark 2025 column
    if 2025 in pivot.columns:
        col_idx = list(pivot.columns).index(2025)
        ax.axvline(col_idx, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvline(col_idx + 1, color='red', linestyle='--', linewidth=2, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_gap_analysis(df: pd.DataFrame, output_path: Path):
    """Plot analysis of the gap between in-dist and OOD performance."""
    df_full = df[df["subset"] == "(full)"].copy()

    results = []
    for _, row in df_full.iterrows():
        in_dist = parse_year_metrics(row.get("in_dist", ""))
        ood = parse_year_metrics(row.get("ood", ""))

        if in_dist and ood:
            cat = categorize_result(row["result"])
            results.append({
                "modality": cat["modality"],
                "variant": cat["variant"],
                "in_dist": in_dist["accuracy"],
                "ood": ood["accuracy"],
                "gap": (in_dist["accuracy"] - ood["accuracy"]) * 100,  # Percentage points
            })

    if not results:
        return

    results_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Gap by modality
    ax1 = axes[0]
    modality_gap = results_df.groupby("modality")["gap"].mean()
    modality_colors = {"clean": CLEAN_COLOR, "vision": VISION_COLOR, "clean_images": CLEAN_IMAGES_COLOR}

    bars = ax1.bar(modality_gap.index, modality_gap.values,
                  color=[modality_colors.get(m, "#888888") for m in modality_gap.index], alpha=0.8)

    for bar, gap in zip(bars, modality_gap.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{gap:.1f}pp", ha='center', fontsize=LABELSIZE)

    ax1.set_ylabel("Gap (percentage points)", fontsize=LABELSIZE)
    ax1.set_title("In-Dist to OOD Gap by Modality", fontsize=TITLESIZE)
    ax1.tick_params(axis='both', labelsize=TICKSIZE)
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Right: Gap by variant
    ax2 = axes[1]
    variant_gap = results_df.groupby("variant")["gap"].mean()
    variant_colors = {"balanced": "#27ae60", "original": "#e67e22", "trainagreeing": "#9b59b6"}

    bars = ax2.bar(variant_gap.index, variant_gap.values,
                  color=[variant_colors.get(v, "#888888") for v in variant_gap.index], alpha=0.8)

    for bar, gap in zip(bars, variant_gap.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{gap:.1f}pp", ha='center', fontsize=LABELSIZE)

    ax2.set_ylabel("Gap (percentage points)", fontsize=LABELSIZE)
    ax2.set_title("In-Dist to OOD Gap by Data Variant", fontsize=TITLESIZE)
    ax2.tick_params(axis='both', labelsize=TICKSIZE)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

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

    # Extract year data
    print("\nExtracting year-by-year data...")
    year_df = extract_year_data(df)
    print(f"Extracted {len(year_df)} year entries")

    # Generate figures
    print("\nGenerating figures...")

    # 1. Year accuracy line plot
    plot_year_accuracy_line(year_df, OUTPUT_DIR / "year_accuracy_line.pdf")

    # 2. In-dist vs OOD bars
    plot_indist_ood_bars(df, OUTPUT_DIR / "indist_ood_bars.pdf")

    # 3. Year heatmap
    plot_year_heatmap(year_df, OUTPUT_DIR / "year_heatmap.pdf")

    # 4. Gap analysis
    plot_gap_analysis(df, OUTPUT_DIR / "gap_analysis.pdf")

    # Print summary
    print("\n" + "="*60)
    print("Temporal Analysis Summary")
    print("="*60)

    if len(year_df) > 0:
        print("\nAccuracy by Year (averaged across models):")
        year_avg = year_df.groupby("year")["accuracy"].mean()
        for year, acc in year_avg.items():
            print(f"  {year}: {acc:.3f}")

        print("\n2025 (OOD) Performance:")
        ood_df = year_df[year_df["year"] == 2025]
        if len(ood_df) > 0:
            print(f"  Mean: {ood_df['accuracy'].mean():.3f}")
            print(f"  Best: {ood_df.loc[ood_df['accuracy'].idxmax(), 'result']} ({ood_df['accuracy'].max():.3f})")

    print("\n" + "="*60)
    print("Done!")


if __name__ == "__main__":
    main()
