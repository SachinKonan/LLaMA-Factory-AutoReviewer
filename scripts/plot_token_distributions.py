#!/usr/bin/env python3
"""Plot token distribution statistics across conferences and years.

Reads CSV files from figures/token_stats/ and creates a subplot figure
showing distributions of pages, text tokens, and vision tokens by year
and conference.

Usage:
    python scripts/plot_token_distributions.py
    python scripts/plot_token_distributions.py --output figures/token_distributions.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATS_DIR = Path(__file__).resolve().parent.parent / "figures" / "token_stats"

# Expected CSV files
TEXT_DATASETS = [
    "iclr_2017_2019_original_text_v7",
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_v7_filtered",
    "nips_2021_2025_original_text_v7_noref",
    "icml_colm_2024_2025_clean_binary_noref",
]
VISION_DATASETS = [
    "iclr_2017_2019_original_vision_v7",
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_v7_filtered",
    "nips_2021_2025_original_vision_v7",
    "icml_colm_2024_2025_vision_binary",
]

# Consistent conference labels and colors
CONFERENCE_MAP = {
    "iclr": "ICLR",
    "nips": "NeurIPS",
    "neurips": "NeurIPS",
    "icml": "ICML/COLM",
    "colm": "ICML/COLM",
}
CONFERENCE_COLORS = {
    "ICLR": "#2196F3",
    "NeurIPS": "#FF9800",
    "ICML/COLM": "#4CAF50",
}


def load_all_csvs(dataset_names):
    """Load and concatenate all CSV files for given dataset names."""
    frames = []
    for name in dataset_names:
        path = STATS_DIR / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["dataset"] = name
            frames.append(df)
            print(f"Loaded {name}: {len(df)} entries")
        else:
            print(f"WARNING: {path} not found, skipping")
    if not frames:
        raise FileNotFoundError("No CSV files found in " + str(STATS_DIR))
    df = pd.concat(frames, ignore_index=True)
    df["conference_label"] = df["conference"].map(CONFERENCE_MAP).fillna(df["conference"])
    return df


def add_boxplot_stats(ax, data_groups, positions, colors, width=0.6):
    """Draw box plots with mean markers and min/max annotations."""
    bps = []
    for data, pos, color in zip(data_groups, positions, colors):
        if len(data) == 0:
            continue
        bp = ax.boxplot(
            [data],
            positions=[pos],
            widths=width,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5),
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(facecolor=color, alpha=0.7, edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
        )
        bps.append(bp)

        # Annotate mean above the box
        mean_val = np.mean(data)
        q75 = np.percentile(data, 75)
        iqr = q75 - np.percentile(data, 25)
        whisker_top = min(q75 + 1.5 * iqr, np.max(data))
        ax.annotate(
            f"{mean_val:.0f}",
            xy=(pos, whisker_top),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
            color=color,
            fontweight="bold",
        )
    return bps


def plot_metric(ax, df, metric, title, ylabel):
    """Plot one metric as grouped box plots by year and conference."""
    conferences = sorted(df["conference_label"].unique(), key=lambda c: list(CONFERENCE_COLORS.keys()).index(c) if c in CONFERENCE_COLORS else 99)
    years = sorted(df["year"].unique())

    n_conf = len(conferences)
    group_width = 0.8
    box_width = group_width / max(n_conf, 1)

    positions_all = []
    data_all = []
    colors_all = []

    for yi, year in enumerate(years):
        for ci, conf in enumerate(conferences):
            subset = df[(df["year"] == year) & (df["conference_label"] == conf)]
            if len(subset) == 0:
                continue
            pos = yi * (n_conf + 1) + ci
            positions_all.append(pos)
            data_all.append(subset[metric].values)
            colors_all.append(CONFERENCE_COLORS.get(conf, "#999999"))

    add_boxplot_stats(ax, data_all, positions_all, colors_all, width=box_width)

    # X-axis: center of each year group
    tick_positions = [yi * (n_conf + 1) + (n_conf - 1) / 2 for yi in range(len(years))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(years, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add min/max as text table below
    # Print summary stats to console
    print(f"\n{title}:")
    print(f"  {'Year':<6} {'Conference':<12} {'N':>6} {'Mean':>8} {'Min':>8} {'Max':>8} {'Median':>8}")
    for year in years:
        for conf in conferences:
            subset = df[(df["year"] == year) & (df["conference_label"] == conf)]
            if len(subset) == 0:
                continue
            vals = subset[metric].values
            print(f"  {year:<6} {conf:<12} {len(vals):>6} {np.mean(vals):>8.0f} {np.min(vals):>8} {np.max(vals):>8} {np.median(vals):>8.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/token_distributions.png")
    args = parser.parse_args()

    print("Loading text datasets...")
    df_text = load_all_csvs(TEXT_DATASETS)
    print(f"\nLoading vision datasets...")
    df_vision = load_all_csvs(VISION_DATASETS)

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle("Token & Page Distributions by Year and Conference", fontsize=14, fontweight="bold", y=0.98)

    # Row 1: Pages (vision datasets)
    plot_metric(axes[0], df_vision, "num_pages", "Number of Pages (Vision Datasets)", "Pages")

    # Row 2: Text tokens (text datasets)
    plot_metric(axes[1], df_text, "text_tokens", "Text Tokens (Text Datasets)", "Tokens")

    # Row 3: Vision tokens (vision datasets)
    plot_metric(axes[2], df_vision, "vision_tokens", "Vision Tokens (Vision Datasets)", "Tokens")

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, edgecolor="black", alpha=0.7, label=l) for l, c in CONFERENCE_COLORS.items()]
    legend_patches.append(plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                                      markeredgecolor="black", markersize=6, label="Mean"))
    fig.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(0.98, 0.97), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
