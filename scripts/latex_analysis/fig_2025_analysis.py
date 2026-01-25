#!/usr/bin/env python3
"""
Generate 2025 analysis figures - investigating why 2025 is harder.

USES ONLY NORMALIZED METRICS:
- pct_rating: Percentile rating within year
- citation_normalized_by_year: Percentile citation within year
- rating_std: Reviewer disagreement

Generates:
- 2024_vs_2025_features.pdf: Feature comparison between 2024 and 2025
- 2025_difficulty_explanation.pdf: Explanation of 2025 difficulty

Usage:
    python scripts/latex_analysis/fig_2025_analysis.py
"""

import json
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
ACCEPT_COLOR = "#4CAF50"
REJECT_COLOR = "#F44336"
COLOR_2024 = "#3498db"
COLOR_2025 = "#e74c3c"


def load_dataset(data_dir: Path, dataset_name: str, split: str) -> list[dict]:
    """Load dataset and return list of entries with metadata."""
    path = data_dir / f"{dataset_name}_{split}" / "data.json"
    if not path.exists():
        print(f"Warning: {path} not found")
        return []

    with open(path) as f:
        data = json.load(f)
    return data


def extract_features(entries: list[dict]) -> pd.DataFrame:
    """Extract features from dataset entries.

    ONLY uses normalized metrics - no raw citations or mean_rating.
    """
    features = []

    for entry in entries:
        metadata = entry.get("_metadata", {})
        answer = metadata.get("answer", "").lower()
        year = metadata.get("year", 0)

        # Get ratings for std calculation
        ratings = metadata.get("ratings", [])
        rating_std = np.std(ratings) if len(ratings) > 1 else 0

        # Get content length
        conversations = entry.get("conversations", [])
        human_content = ""
        for msg in conversations:
            if msg.get("from") == "human":
                human_content = msg.get("value", "")
                break
        tokens = len(human_content) // 4

        features.append({
            "submission_id": metadata.get("submission_id", ""),
            "year": year,
            "label": 1 if answer == "accept" else 0,
            "tokens": tokens,
            # NORMALIZED METRICS ONLY
            "pct_rating": metadata.get("pct_rating", np.nan),
            "citation_normalized": metadata.get("citation_normalized_by_year", np.nan),
            "rating_std": rating_std,
        })

    return pd.DataFrame(features)


def plot_2024_vs_2025_features(df: pd.DataFrame, output_path: Path):
    """Plot feature comparison between 2024 and 2025 using ONLY normalized metrics."""
    df_2024 = df[df["year"] == 2024]
    df_2025 = df[df["year"] == 2025]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Only normalized metrics
    features = ["pct_rating", "citation_normalized", "rating_std", "tokens"]
    feature_names = [
        "Pct Rating\n(Percentile within Year)",
        "Citation Normalized\n(Percentile within Year)",
        "Rating Std Dev\n(Reviewer Disagreement)",
        "Token Count"
    ]

    for idx, (feature, name) in enumerate(zip(features, feature_names)):
        ax = axes[idx // 2, idx % 2]

        data_2024 = df_2024[feature].dropna()
        data_2025 = df_2025[feature].dropna()

        if len(data_2024) == 0 or len(data_2025) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name, fontsize=TITLESIZE)
            continue

        # Create violin plot
        data = [data_2024.values, data_2025.values]
        positions = [0, 1]

        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLOR_2024 if i == 0 else COLOR_2025)
            pc.set_alpha(0.7)

        # Add means as text
        y_max = max(data_2024.max(), data_2025.max())
        ax.text(0, y_max * 1.05, f"$\\mu$={data_2024.mean():.3f}",
               ha='center', fontsize=TICKSIZE)
        ax.text(1, y_max * 1.05, f"$\\mu$={data_2025.mean():.3f}",
               ha='center', fontsize=TICKSIZE)

        ax.set_ylabel(name.split('\n')[0], fontsize=LABELSIZE)
        ax.set_xticks(positions)
        ax.set_xticklabels(["2024", "2025"], fontsize=LABELSIZE)
        ax.set_title(name, fontsize=TITLESIZE - 1)
        ax.tick_params(axis='both', labelsize=TICKSIZE)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle("2024 vs 2025: Normalized Feature Comparison", fontsize=TITLESIZE + 2, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_normalized_by_decision(df: pd.DataFrame, output_path: Path):
    """Plot normalized metrics by decision for 2024 vs 2025."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    years = [2024, 2025]

    # 1. Pct Rating by decision
    ax1 = axes[0]
    x = np.arange(len(years))
    width = 0.35

    accept_pct = [df[(df["year"]==y) & (df["label"]==1)]["pct_rating"].mean() for y in years]
    reject_pct = [df[(df["year"]==y) & (df["label"]==0)]["pct_rating"].mean() for y in years]

    ax1.bar(x - width/2, accept_pct, width, label='Accept', color=ACCEPT_COLOR, alpha=0.8)
    ax1.bar(x + width/2, reject_pct, width, label='Reject', color=REJECT_COLOR, alpha=0.8)

    ax1.set_xlabel("Year", fontsize=LABELSIZE)
    ax1.set_ylabel("Mean Pct Rating", fontsize=LABELSIZE)
    ax1.set_title("Pct Rating by Decision\n(Higher = Better Percentile)", fontsize=TITLESIZE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.legend(fontsize=LEGENDSIZE)
    ax1.tick_params(axis='both', labelsize=TICKSIZE)
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 2. Citation Normalized by decision
    ax2 = axes[1]
    accept_cit = [df[(df["year"]==y) & (df["label"]==1)]["citation_normalized"].mean() for y in years]
    reject_cit = [df[(df["year"]==y) & (df["label"]==0)]["citation_normalized"].mean() for y in years]

    ax2.bar(x - width/2, accept_cit, width, label='Accept', color=ACCEPT_COLOR, alpha=0.8)
    ax2.bar(x + width/2, reject_cit, width, label='Reject', color=REJECT_COLOR, alpha=0.8)

    ax2.set_xlabel("Year", fontsize=LABELSIZE)
    ax2.set_ylabel("Mean Citation Normalized", fontsize=LABELSIZE)
    ax2.set_title("Citation Percentile by Decision\n(Normalized within Year)", fontsize=TITLESIZE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.legend(fontsize=LEGENDSIZE)
    ax2.tick_params(axis='both', labelsize=TICKSIZE)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 3. Rating Std by decision
    ax3 = axes[2]
    accept_std = [df[(df["year"]==y) & (df["label"]==1)]["rating_std"].mean() for y in years]
    reject_std = [df[(df["year"]==y) & (df["label"]==0)]["rating_std"].mean() for y in years]

    ax3.bar(x - width/2, accept_std, width, label='Accept', color=ACCEPT_COLOR, alpha=0.8)
    ax3.bar(x + width/2, reject_std, width, label='Reject', color=REJECT_COLOR, alpha=0.8)

    ax3.set_xlabel("Year", fontsize=LABELSIZE)
    ax3.set_ylabel("Mean Rating Std", fontsize=LABELSIZE)
    ax3.set_title("Reviewer Disagreement by Decision\n(Higher = More Disagreement)", fontsize=TITLESIZE)
    ax3.set_xticks(x)
    ax3.set_xticklabels(years)
    ax3.legend(fontsize=LEGENDSIZE)
    ax3.tick_params(axis='both', labelsize=TICKSIZE)
    ax3.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_2025_difficulty_explanation(df: pd.DataFrame, output_path: Path):
    """Plot explanation of why 2025 might be harder - using normalized metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Feature correlation with label by year
    ax1 = axes[0]
    years = sorted(df["year"].unique())
    features = ["pct_rating", "citation_normalized", "rating_std"]
    feature_names = ["Pct Rating", "Citation Norm", "Rating Std"]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    x = np.arange(len(years))
    width = 0.25

    for i, (feature, fname, color) in enumerate(zip(features, feature_names, colors)):
        correlations = []
        for year in years:
            year_df = df[df["year"] == year]
            feat_data = year_df[feature].dropna()
            label_data = year_df.loc[feat_data.index, "label"]
            if len(feat_data) > 10:
                corr = feat_data.corr(label_data)
                correlations.append(corr if not np.isnan(corr) else 0)
            else:
                correlations.append(0)

        ax1.bar(x + i * width - width, correlations, width, label=fname, color=color, alpha=0.8)

    ax1.set_xlabel("Year", fontsize=LABELSIZE)
    ax1.set_ylabel("Correlation with Accept/Reject", fontsize=LABELSIZE)
    ax1.set_title("Feature Predictiveness by Year\n(Normalized Metrics Only)", fontsize=TITLESIZE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.legend(fontsize=LEGENDSIZE)
    ax1.tick_params(axis='both', labelsize=TICKSIZE)
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # Right: Summary statistics table
    ax2 = axes[1]
    ax2.axis('off')

    # Compute actual stats
    df_2024 = df[df["year"] == 2024]
    df_2025 = df[df["year"] == 2025]

    stats_text = f"""
    2024 vs 2025 Comparison (Normalized Metrics)
    ════════════════════════════════════════════

    Pct Rating (percentile within year):
      2024 Accept: {df_2024[df_2024['label']==1]['pct_rating'].mean():.3f}
      2024 Reject: {df_2024[df_2024['label']==0]['pct_rating'].mean():.3f}
      2025 Accept: {df_2025[df_2025['label']==1]['pct_rating'].mean():.3f}
      2025 Reject: {df_2025[df_2025['label']==0]['pct_rating'].mean():.3f}

    Citation Normalized (percentile within year):
      2024 Accept: {df_2024[df_2024['label']==1]['citation_normalized'].mean():.3f}
      2024 Reject: {df_2024[df_2024['label']==0]['citation_normalized'].mean():.3f}
      2025 Accept: {df_2025[df_2025['label']==1]['citation_normalized'].mean():.3f}
      2025 Reject: {df_2025[df_2025['label']==0]['citation_normalized'].mean():.3f}

    Rating Std (reviewer disagreement):
      2024 Accept: {df_2024[df_2024['label']==1]['rating_std'].mean():.2f}
      2024 Reject: {df_2024[df_2024['label']==0]['rating_std'].mean():.2f}
      2025 Accept: {df_2025[df_2025['label']==1]['rating_std'].mean():.2f}
      2025 Reject: {df_2025[df_2025['label']==0]['rating_std'].mean():.2f}

    Sample sizes: 2024={len(df_2024)}, 2025={len(df_2025)}
    """

    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("figures/latex/ablations")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset_name = "iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6"

    print("Loading dataset...")
    test_data = load_dataset(DATA_DIR, dataset_name, "test")

    if not test_data:
        print("Error: Could not load test data")
        return

    # Extract features
    print("Extracting features...")
    df = extract_features(test_data)
    print(f"Dataset size: {len(df)}")

    # Generate figures
    print("\nGenerating figures (NORMALIZED METRICS ONLY)...")

    # 1. 2024 vs 2025 feature comparison
    plot_2024_vs_2025_features(df, OUTPUT_DIR / "2024_vs_2025_features.pdf")

    # 2. Normalized metrics by decision
    plot_normalized_by_decision(df, OUTPUT_DIR / "normalized_by_decision.pdf")

    # 3. Difficulty explanation
    plot_2025_difficulty_explanation(df, OUTPUT_DIR / "2025_difficulty_explanation.pdf")

    # Print summary
    print("\n" + "="*60)
    print("2025 Analysis Summary (NORMALIZED METRICS)")
    print("="*60)

    df_2024 = df[df["year"] == 2024]
    df_2025 = df[df["year"] == 2025]

    print(f"\nSample sizes: 2024={len(df_2024)}, 2025={len(df_2025)}")

    print("\nPct Rating (percentile, should separate accept/reject):")
    print(f"  2024 accept: {df_2024[df_2024['label']==1]['pct_rating'].mean():.3f}")
    print(f"  2024 reject: {df_2024[df_2024['label']==0]['pct_rating'].mean():.3f}")
    print(f"  2025 accept: {df_2025[df_2025['label']==1]['pct_rating'].mean():.3f}")
    print(f"  2025 reject: {df_2025[df_2025['label']==0]['pct_rating'].mean():.3f}")

    print("\nCitation Normalized (percentile within year):")
    print(f"  2024 accept: {df_2024[df_2024['label']==1]['citation_normalized'].mean():.3f}")
    print(f"  2024 reject: {df_2024[df_2024['label']==0]['citation_normalized'].mean():.3f}")
    print(f"  2025 accept: {df_2025[df_2025['label']==1]['citation_normalized'].mean():.3f}")
    print(f"  2025 reject: {df_2025[df_2025['label']==0]['citation_normalized'].mean():.3f}")

    print("\n" + "="*60)
    print("Done!")


if __name__ == "__main__":
    main()
