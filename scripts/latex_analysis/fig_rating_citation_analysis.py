#!/usr/bin/env python3
"""
Generate rating analysis figures for the LaTeX report.

ONLY uses normalized metrics (pct_rating) - no raw ratings or citations.

Generates:
- pct_rating_violin_by_decision.pdf: Violin plot of pct_rating (accept vs reject)
- pct_rating_by_year_violin.pdf: Violin per year showing pct_rating distribution
- rating_std_violin_by_decision.pdf: Violin plot of reviewer disagreement
- pct_rating_scatter.pdf: pct_rating vs rating_std scatter

Mathematical definitions:
- pct_rating: Percentile rank of rating within conference/year
  pct_rating_i = rank(r_i in year y) / count(papers in year y)

Usage:
    python scripts/latex_analysis/fig_rating_citation_analysis.py
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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
PALETTE = {0: REJECT_COLOR, 1: ACCEPT_COLOR}


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
    """Extract features from dataset entries."""
    features = []

    for entry in entries:
        metadata = entry.get("_metadata", {})
        answer = metadata.get("answer", "").lower()
        year = metadata.get("year", 0)

        # Rating features
        ratings = metadata.get("ratings", [])
        rating_std = np.std(ratings) if len(ratings) > 1 else 0
        pct_rating = metadata.get("pct_rating", np.nan)
        citation_normalized = metadata.get("citation_normalized_by_year", np.nan)

        features.append({
            "submission_id": metadata.get("submission_id", ""),
            "answer": answer,
            "label": 1 if answer == "accept" else 0,
            "decision": "Accept" if answer == "accept" else "Reject",
            "year": year,
            "rating_std": rating_std,
            "pct_rating": pct_rating,
            "citation_normalized": citation_normalized,
            "num_reviewers": len(ratings),
        })

    return pd.DataFrame(features)


def plot_pct_rating_violin_by_decision(df: pd.DataFrame, output_path: Path):
    """Plot violin plot of pct_rating by decision (accept vs reject)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter valid data
    plot_df = df[df["pct_rating"].notna()].copy()

    # Create violin plot
    parts = ax.violinplot(
        [plot_df[plot_df["label"] == 0]["pct_rating"].values,
         plot_df[plot_df["label"] == 1]["pct_rating"].values],
        positions=[0, 1],
        showmeans=True,
        showmedians=True,
        widths=0.8
    )

    # Color the violins
    colors = [REJECT_COLOR, ACCEPT_COLOR]
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # Style mean and median lines
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('white')

    # Add statistics
    accept_pct = plot_df[plot_df["label"] == 1]["pct_rating"]
    reject_pct = plot_df[plot_df["label"] == 0]["pct_rating"]

    # Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(accept_pct, reject_pct, alternative='two-sided')

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        f"Reject\n$\\mu$={reject_pct.mean():.3f}\n$n$={len(reject_pct)}",
        f"Accept\n$\\mu$={accept_pct.mean():.3f}\n$n$={len(accept_pct)}"
    ], fontsize=LABELSIZE)

    ax.set_ylabel("Percentile Rating (pct\\_rating)", fontsize=LABELSIZE)
    ax.set_title("Distribution of Normalized Reviewer Ratings by Decision\n" +
                 f"(Mann-Whitney $U$={u_stat:.0f}, $p$<{max(p_value, 1e-100):.1e})", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Add formula annotation
    formula_text = r"$\text{pct\_rating}_i = \frac{\text{rank}(r_i \text{ in year } y)}{|\text{papers in year } y|}$"
    ax.text(0.5, 0.02, formula_text, transform=ax.transAxes, fontsize=10,
            ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pct_rating_by_year_violin(df: pd.DataFrame, output_path: Path):
    """Plot violin plots of pct_rating by year, split by decision."""
    fig, ax = plt.subplots(figsize=(14, 6))

    plot_df = df[df["pct_rating"].notna()].copy()
    years = sorted(plot_df["year"].unique())

    # Create positions for violins
    positions_reject = [i * 2 - 0.3 for i in range(len(years))]
    positions_accept = [i * 2 + 0.3 for i in range(len(years))]

    # Plot reject violins
    reject_data = [plot_df[(plot_df["year"] == y) & (plot_df["label"] == 0)]["pct_rating"].values
                  for y in years]
    parts_reject = ax.violinplot(reject_data, positions=positions_reject, widths=0.5,
                                  showmeans=True, showmedians=False)
    for pc in parts_reject['bodies']:
        pc.set_facecolor(REJECT_COLOR)
        pc.set_alpha(0.7)
    parts_reject['cmeans'].set_color('black')

    # Plot accept violins
    accept_data = [plot_df[(plot_df["year"] == y) & (plot_df["label"] == 1)]["pct_rating"].values
                  for y in years]
    parts_accept = ax.violinplot(accept_data, positions=positions_accept, widths=0.5,
                                  showmeans=True, showmedians=False)
    for pc in parts_accept['bodies']:
        pc.set_facecolor(ACCEPT_COLOR)
        pc.set_alpha(0.7)
    parts_accept['cmeans'].set_color('black')

    # Add mean annotations
    for i, year in enumerate(years):
        reject_mean = np.mean(reject_data[i]) if len(reject_data[i]) > 0 else np.nan
        accept_mean = np.mean(accept_data[i]) if len(accept_data[i]) > 0 else np.nan
        if not np.isnan(reject_mean):
            ax.text(positions_reject[i], reject_mean + 0.05, f"{reject_mean:.2f}",
                   ha='center', fontsize=TICKSIZE - 2, color=REJECT_COLOR)
        if not np.isnan(accept_mean):
            ax.text(positions_accept[i], accept_mean + 0.05, f"{accept_mean:.2f}",
                   ha='center', fontsize=TICKSIZE - 2, color=ACCEPT_COLOR)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ACCEPT_COLOR, alpha=0.7, label='Accept'),
                      Patch(facecolor=REJECT_COLOR, alpha=0.7, label='Reject')]
    ax.legend(handles=legend_elements, fontsize=LEGENDSIZE, loc='upper right')

    ax.set_xticks([i * 2 for i in range(len(years))])
    ax.set_xticklabels([str(y) for y in years], fontsize=LABELSIZE)
    ax.set_xlabel("Year", fontsize=LABELSIZE)
    ax.set_ylabel("Percentile Rating (pct\\_rating)", fontsize=LABELSIZE)
    ax.set_title("Normalized Rating Distribution by Year and Decision", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_rating_std_violin_by_decision(df: pd.DataFrame, output_path: Path):
    """Plot violin plot of rating standard deviation (reviewer disagreement) by decision."""
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_df = df[df["rating_std"].notna()].copy()

    # Create violin plot
    parts = ax.violinplot(
        [plot_df[plot_df["label"] == 0]["rating_std"].values,
         plot_df[plot_df["label"] == 1]["rating_std"].values],
        positions=[0, 1],
        showmeans=True,
        showmedians=True,
        widths=0.8
    )

    # Color the violins
    colors = [REJECT_COLOR, ACCEPT_COLOR]
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('white')

    # Add statistics
    accept_std = plot_df[plot_df["label"] == 1]["rating_std"]
    reject_std = plot_df[plot_df["label"] == 0]["rating_std"]

    # t-test for difference in means
    t_stat, p_value = stats.ttest_ind(accept_std, reject_std)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        f"Reject\n$\\mu$={reject_std.mean():.2f}\n$\\sigma$={reject_std.std():.2f}",
        f"Accept\n$\\mu$={accept_std.mean():.2f}\n$\\sigma$={accept_std.std():.2f}"
    ], fontsize=LABELSIZE)

    ax.set_ylabel("Rating Standard Deviation $\\sigma_r$", fontsize=LABELSIZE)
    ax.set_title("Reviewer Disagreement by Decision\n" +
                 f"($t$={t_stat:.2f}, $p$={p_value:.3f})", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Add formula annotation
    formula_text = r"$\sigma_r = \text{std}(\text{reviewer ratings})$"
    ax.text(0.5, 0.95, formula_text, transform=ax.transAxes, fontsize=10,
            ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pct_rating_scatter(df: pd.DataFrame, output_path: Path):
    """Plot scatter of pct_rating vs rating_std colored by decision."""
    fig, ax = plt.subplots(figsize=(10, 8))

    plot_df = df[df["pct_rating"].notna() & df["rating_std"].notna()].copy()
    accept_df = plot_df[plot_df["label"] == 1]
    reject_df = plot_df[plot_df["label"] == 0]

    # Add jitter
    jitter_x = 0.02
    jitter_y = 0.05

    ax.scatter(
        reject_df["pct_rating"] + np.random.uniform(-jitter_x, jitter_x, len(reject_df)),
        reject_df["rating_std"] + np.random.uniform(-jitter_y, jitter_y, len(reject_df)),
        c=REJECT_COLOR, alpha=0.4, s=30, label='Reject'
    )
    ax.scatter(
        accept_df["pct_rating"] + np.random.uniform(-jitter_x, jitter_x, len(accept_df)),
        accept_df["rating_std"] + np.random.uniform(-jitter_y, jitter_y, len(accept_df)),
        c=ACCEPT_COLOR, alpha=0.4, s=30, label='Accept'
    )

    # Add means as larger markers
    ax.scatter([accept_df["pct_rating"].mean()], [accept_df["rating_std"].mean()],
              c=ACCEPT_COLOR, marker='*', s=400, edgecolors='black', linewidths=1,
              label=f'Accept Mean ({accept_df["pct_rating"].mean():.2f}, {accept_df["rating_std"].mean():.2f})')
    ax.scatter([reject_df["pct_rating"].mean()], [reject_df["rating_std"].mean()],
              c=REJECT_COLOR, marker='*', s=400, edgecolors='black', linewidths=1,
              label=f'Reject Mean ({reject_df["pct_rating"].mean():.2f}, {reject_df["rating_std"].mean():.2f})')

    # Calculate correlation
    corr, p_corr = stats.pearsonr(plot_df["pct_rating"], plot_df["rating_std"])

    ax.set_xlabel("Percentile Rating (pct\\_rating)", fontsize=LABELSIZE)
    ax.set_ylabel("Rating Standard Deviation $\\sigma_r$", fontsize=LABELSIZE)
    ax.set_title(f"Rating Characteristics by Decision\n(Pearson $r$={corr:.3f}, $p$={p_corr:.3e})",
                fontsize=TITLESIZE)
    ax.legend(fontsize=LEGENDSIZE - 1, loc='upper right')
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_rating_std_by_year(df: pd.DataFrame, output_path: Path):
    """Plot rating std distribution by year."""
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_df = df[df["rating_std"].notna()].copy()
    years = sorted(plot_df["year"].unique())

    # Create positions for violins
    positions = list(range(len(years)))

    # Plot violins
    data = [plot_df[plot_df["year"] == y]["rating_std"].values for y in years]
    parts = ax.violinplot(data, positions=positions, widths=0.7,
                          showmeans=True, showmedians=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('black')

    # Add mean annotations
    for i, year in enumerate(years):
        mean_val = np.mean(data[i]) if len(data[i]) > 0 else np.nan
        if not np.isnan(mean_val):
            ax.text(i, mean_val + 0.1, f"{mean_val:.2f}",
                   ha='center', fontsize=TICKSIZE - 1)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(y) for y in years], fontsize=LABELSIZE)
    ax.set_xlabel("Year", fontsize=LABELSIZE)
    ax.set_ylabel("Rating Standard Deviation $\\sigma_r$", fontsize=LABELSIZE)
    ax.set_title("Reviewer Disagreement Over Time", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("figures/latex/data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset_name = "iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6"

    print("Loading datasets...")
    # Load train + test for comprehensive analysis
    train_data = load_dataset(DATA_DIR, dataset_name, "train")
    test_data = load_dataset(DATA_DIR, dataset_name, "test")
    val_data = load_dataset(DATA_DIR, dataset_name, "validation")

    all_data = train_data + test_data + val_data

    if not all_data:
        print("Error: Could not load data")
        return

    # Extract features
    print("Extracting features...")
    df = extract_features(all_data)

    print(f"Dataset size: {len(df)}")
    print(f"  Accept: {len(df[df['label'] == 1])}")
    print(f"  Reject: {len(df[df['label'] == 0])}")

    # Generate figures
    print("\nGenerating figures...")

    # 1. Pct rating violin by decision
    plot_pct_rating_violin_by_decision(df, OUTPUT_DIR / "pct_rating_violin_by_decision.pdf")

    # 2. Pct rating by year violin
    plot_pct_rating_by_year_violin(df, OUTPUT_DIR / "pct_rating_by_year_violin.pdf")

    # 3. Rating std violin by decision
    plot_rating_std_violin_by_decision(df, OUTPUT_DIR / "rating_std_violin_by_decision.pdf")

    # 4. Pct rating scatter (pct_rating vs rating_std)
    plot_pct_rating_scatter(df, OUTPUT_DIR / "pct_rating_scatter.pdf")

    # 5. Rating std by year
    plot_rating_std_by_year(df, OUTPUT_DIR / "rating_std_by_year.pdf")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Rating Statistics (Normalized Metrics Only)")
    print("=" * 60)

    accept_df = df[df["label"] == 1]
    reject_df = df[df["label"] == 0]

    print(f"\nPct Rating (Percentile within year):")
    print(f"  Accept: mean={accept_df['pct_rating'].mean():.3f}, std={accept_df['pct_rating'].std():.3f}")
    print(f"  Reject: mean={reject_df['pct_rating'].mean():.3f}, std={reject_df['pct_rating'].std():.3f}")

    # Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(
        accept_df['pct_rating'].dropna(),
        reject_df['pct_rating'].dropna()
    )
    print(f"  Mann-Whitney U={u_stat:.0f}, p={p_value:.2e}")

    print(f"\nRating Std (Reviewer Disagreement):")
    print(f"  Accept: mean={accept_df['rating_std'].mean():.3f}, std={accept_df['rating_std'].std():.3f}")
    print(f"  Reject: mean={reject_df['rating_std'].mean():.3f}, std={reject_df['rating_std'].std():.3f}")

    # t-test
    t_stat, p_value = stats.ttest_ind(
        accept_df['rating_std'].dropna(),
        reject_df['rating_std'].dropna()
    )
    print(f"  t={t_stat:.2f}, p={p_value:.3f}")

    print(f"\nPct Rating by Year:")
    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year]
        accept_mean = year_df[year_df["label"] == 1]["pct_rating"].mean()
        reject_mean = year_df[year_df["label"] == 0]["pct_rating"].mean()
        print(f"  {year}: Accept={accept_mean:.3f}, Reject={reject_mean:.3f}, "
              f"n={len(year_df)}")

    print("\n" + "=" * 60)
    print("Done! All figures use ONLY normalized metrics (pct_rating).")
    print("Raw ratings and citations have been removed per plan.")


if __name__ == "__main__":
    main()
