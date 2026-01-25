#!/usr/bin/env python3
"""
Generate data distribution figures for the LaTeX report.

Generates:
- year_distribution.pdf: Stacked bar chart of submissions per year (accept/reject)
- token_violin.pdf: Violin plot of token counts for accept vs reject
- correlation_heatmap.pdf: Correlation heatmap of features vs decision

Usage:
    python scripts/latex_analysis/fig_data_distribution.py
"""

import json
import re
from collections import defaultdict
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
ACCEPT_COLOR = "#4CAF50"  # Green
REJECT_COLOR = "#F44336"  # Red


def load_dataset(data_dir: Path, dataset_name: str, split: str) -> list[dict]:
    """Load dataset and return list of entries with metadata."""
    path = data_dir / f"{dataset_name}_{split}" / "data.json"
    if not path.exists():
        print(f"Warning: {path} not found")
        return []

    with open(path) as f:
        data = json.load(f)
    return data


def extract_all_features(entries: list[dict]) -> pd.DataFrame:
    """Extract all features from dataset entries."""
    features = []

    # Pattern to find markdown image references
    image_pattern = re.compile(r'!\[[^\]]*\]\(images/([^)]+)\)')

    for entry in entries:
        metadata = entry.get("_metadata", {})
        answer = metadata.get("answer", "").lower()
        year = metadata.get("year", 0)

        # Get human message content
        conversations = entry.get("conversations", [])
        human_content = ""
        for msg in conversations:
            if msg.get("from") == "human":
                human_content = msg.get("value", "")
                break

        # Calculate features
        tokens = len(human_content) // 4  # Approximate token count
        pages = human_content.count("<image>")  # Page count for vision
        images = len(image_pattern.findall(human_content))  # Image count for clean_images

        # Metadata features - ONLY normalized metrics
        ratings = metadata.get("ratings", [])
        rating_std = np.std(ratings) if len(ratings) > 1 else 0
        pct_rating = metadata.get("pct_rating", 0)

        features.append({
            "submission_id": metadata.get("submission_id", ""),
            "answer": answer,
            "label": 1 if answer == "accept" else 0,
            "year": year,
            "tokens": tokens,
            "pages": pages,
            "images": images,
            "rating_std": rating_std,
            "pct_rating": pct_rating,
        })

    return pd.DataFrame(features)


def plot_year_distribution(df: pd.DataFrame, output_path: Path):
    """Plot stacked bar chart of submissions per year."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by year and answer
    year_accept = df[df["label"] == 1].groupby("year").size()
    year_reject = df[df["label"] == 0].groupby("year").size()

    years = sorted(df["year"].unique())
    accepts = [year_accept.get(y, 0) for y in years]
    rejects = [year_reject.get(y, 0) for y in years]

    x = np.arange(len(years))
    width = 0.6

    # Stacked bar chart
    bars_accept = ax.bar(x, accepts, width, label='Accept', color=ACCEPT_COLOR)
    bars_reject = ax.bar(x, rejects, width, bottom=accepts, label='Reject', color=REJECT_COLOR)

    # Add count labels
    for i, (a, r) in enumerate(zip(accepts, rejects)):
        if a > 0:
            ax.text(i, a / 2, str(a), ha='center', va='center',
                   fontsize=TICKSIZE - 2, fontweight='bold', color='white')
        if r > 0:
            ax.text(i, a + r / 2, str(r), ha='center', va='center',
                   fontsize=TICKSIZE - 2, fontweight='bold', color='white')

    # Calculate overall acceptance rate
    total_accept = sum(accepts)
    total_reject = sum(rejects)
    total = total_accept + total_reject
    acc_rate = 100.0 * total_accept / total if total > 0 else 0

    ax.set_xlabel("Year", fontsize=LABELSIZE)
    ax.set_ylabel("Number of Submissions", fontsize=LABELSIZE)
    ax.set_title(f"Test Set Distribution by Year (Acceptance Rate: {acc_rate:.1f}\\%)", fontsize=TITLESIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.legend(fontsize=LEGENDSIZE, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_token_violin(df: pd.DataFrame, output_path: Path):
    """Plot violin plot of token counts for accept vs reject."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Prepare data
    accept_tokens = df[df["label"] == 1]["tokens"].values / 1000  # Convert to k
    reject_tokens = df[df["label"] == 0]["tokens"].values / 1000

    data = [accept_tokens, reject_tokens]
    positions = [0, 1]

    # Create violin plot
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(ACCEPT_COLOR if i == 0 else REJECT_COLOR)
        pc.set_alpha(0.7)

    # Add box plot inside
    bp = ax.boxplot(data, positions=positions, widths=0.15, patch_artist=True,
                    showfliers=False, showmeans=False)
    for patch, color in zip(bp['boxes'], [ACCEPT_COLOR, REJECT_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    # Add mean annotations
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.text(positions[i], np.max(d) + 2, f"$\\mu$={mean_val:.1f}k",
               ha='center', fontsize=TICKSIZE)

    ax.set_ylabel("Token Count (thousands)", fontsize=LABELSIZE)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Accept", "Reject"], fontsize=LABELSIZE)
    ax.set_title("Token Length Distribution by Decision", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path):
    """Plot correlation heatmap of features vs decision.

    NOTE: Only uses normalized metrics (pct_rating). Raw citations and mean_rating removed.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Select features for correlation - ONLY normalized metrics
    features = ["label", "tokens", "pages", "images", "rating_std", "pct_rating"]

    # Filter to available features
    available = [f for f in features if f in df.columns and df[f].notna().sum() > 0]

    # Compute correlation matrix
    corr_df = df[available].corr()

    # Rename for display
    rename_map = {
        "label": "Accept",
        "tokens": "Tokens",
        "pages": "Pages",
        "images": "Images",
        "rating_std": "Rating Std",
        "pct_rating": "Pct Rating",
    }
    corr_df = corr_df.rename(index=rename_map, columns=rename_map)

    # Plot heatmap
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        annot_kws={"size": TICKSIZE},
    )

    ax.set_title("Feature Correlation Matrix", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_by_decision(df: pd.DataFrame, feature: str, ylabel: str, title: str, output_path: Path):
    """Generic function to plot feature distribution by decision."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    accept_data = df[df["label"] == 1][feature].dropna()
    reject_data = df[df["label"] == 0][feature].dropna()

    # Left: KDE plot
    ax1 = axes[0]
    if len(accept_data) > 1:
        sns.kdeplot(accept_data, ax=ax1, color=ACCEPT_COLOR, label=f"Accept ($\\mu$={accept_data.mean():.1f})", fill=True, alpha=0.3)
    if len(reject_data) > 1:
        sns.kdeplot(reject_data, ax=ax1, color=REJECT_COLOR, label=f"Reject ($\\mu$={reject_data.mean():.1f})", fill=True, alpha=0.3)

    ax1.set_xlabel(ylabel, fontsize=LABELSIZE)
    ax1.set_ylabel("Density", fontsize=LABELSIZE)
    ax1.set_title(f"{title} Distribution", fontsize=TITLESIZE)
    ax1.legend(fontsize=LEGENDSIZE)
    ax1.tick_params(axis='both', labelsize=TICKSIZE)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Right: Box plot by year
    ax2 = axes[1]
    years = sorted(df["year"].unique())
    positions = []
    labels = []
    data_to_plot = []
    colors = []

    for i, year in enumerate(years):
        year_accept = df[(df["year"] == year) & (df["label"] == 1)][feature].dropna()
        year_reject = df[(df["year"] == year) & (df["label"] == 0)][feature].dropna()

        if len(year_accept) > 0:
            data_to_plot.append(year_accept.values)
            positions.append(i * 3)
            labels.append(f"{year}\nAcc")
            colors.append(ACCEPT_COLOR)
        if len(year_reject) > 0:
            data_to_plot.append(year_reject.values)
            positions.append(i * 3 + 1)
            labels.append(f"{year}\nRej")
            colors.append(REJECT_COLOR)

    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_ylabel(ylabel, fontsize=LABELSIZE)
    ax2.set_title(f"{title} by Year and Decision", fontsize=TITLESIZE)
    ax2.tick_params(axis='both', labelsize=TICKSIZE - 2)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Custom x-ticks
    year_centers = [i * 3 + 0.5 for i in range(len(years))]
    ax2.set_xticks(year_centers)
    ax2.set_xticklabels([str(y) for y in years])

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
    train_data = load_dataset(DATA_DIR, dataset_name, "train")
    test_data = load_dataset(DATA_DIR, dataset_name, "test")

    if not test_data:
        print("Error: Could not load test data")
        return

    # Extract features
    print("Extracting features...")
    df_test = extract_all_features(test_data)
    df_train = extract_all_features(train_data)
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    print(f"Test set size: {len(df_test)}")
    print(f"Train set size: {len(df_train)}")
    print(f"Total: {len(df_all)}")

    # Generate figures
    print("\nGenerating figures...")

    # 1. Year distribution (stacked bar)
    plot_year_distribution(df_test, OUTPUT_DIR / "year_distribution.pdf")

    # 2. Token violin plot
    plot_token_violin(df_test, OUTPUT_DIR / "token_violin.pdf")

    # 3. Correlation heatmap
    plot_correlation_heatmap(df_test, OUTPUT_DIR / "correlation_heatmap.pdf")

    # NOTE: Citation distribution removed - only using normalized metrics (pct_rating)

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics (Test Set)")
    print("="*60)

    accept_df = df_test[df_test["label"] == 1]
    reject_df = df_test[df_test["label"] == 0]

    print(f"\nAccepts: {len(accept_df)}, Rejects: {len(reject_df)}")
    print(f"\nTokens (mean):")
    print(f"  Accept: {accept_df['tokens'].mean():.1f}")
    print(f"  Reject: {reject_df['tokens'].mean():.1f}")

    # NOTE: Only using normalized metrics (pct_rating), not raw mean_rating or citations
    print(f"\nPct Rating (normalized within year):")
    print(f"  Accept: {accept_df['pct_rating'].mean():.3f}")
    print(f"  Reject: {reject_df['pct_rating'].mean():.3f}")

    print(f"\nRating Std (reviewer disagreement):")
    print(f"  Accept: {accept_df['rating_std'].mean():.2f}")
    print(f"  Reject: {reject_df['rating_std'].mean():.2f}")

    print("\n" + "="*60)
    print("Done!")


if __name__ == "__main__":
    main()
