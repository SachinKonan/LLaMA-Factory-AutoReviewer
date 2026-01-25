#!/usr/bin/env python3
"""
Generate TF-IDF baseline analysis figures for the LaTeX report.

Generates:
- tfidf_year_accuracy.pdf: TF-IDF accuracy by year
- feature_importance.pdf: Top discriminative TF-IDF features

Uses:
- results/_tfidf/*.csv - TF-IDF per-year results
- results/_tfidf/iclr_2020_2025_features.csv - Top features

Usage:
    python scripts/latex_analysis/fig_tfidf_analysis.py
"""

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
TFIDF_COLOR = "#888888"


def load_tfidf_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all TF-IDF result files."""
    results = {}
    for csv_file in results_dir.glob("*.csv"):
        if csv_file.name.endswith("_features.csv"):
            continue  # Skip feature files
        name = csv_file.stem
        df = pd.read_csv(csv_file)
        results[name] = df
    return results


def plot_tfidf_year_accuracy(df: pd.DataFrame, output_path: Path, title_suffix: str = ""):
    """Plot TF-IDF accuracy by year."""
    # Filter to yearly rows (exclude Total)
    df_years = df[df["year"] != "Total"].copy()
    df_years["year"] = df_years["year"].astype(int)
    df_years = df_years.sort_values("year")

    fig, ax = plt.subplots(figsize=(10, 6))

    years = df_years["year"].values
    accuracy = df_years["accuracy"].values
    accept_recall = df_years["accept_recall"].values
    reject_recall = df_years["reject_recall"].values

    x = np.arange(len(years))
    width = 0.25

    bars1 = ax.bar(x - width, accuracy, width, label='Accuracy', color=TFIDF_COLOR, alpha=0.8)
    bars2 = ax.bar(x, accept_recall, width, label='Accept Recall', color=ACCEPT_COLOR, alpha=0.8)
    bars3 = ax.bar(x + width, reject_recall, width, label='Reject Recall', color=REJECT_COLOR, alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=TICKSIZE - 2)

    ax.set_xlabel("Year", fontsize=LABELSIZE)
    ax.set_ylabel("Score", fontsize=LABELSIZE)
    ax.set_title(f"TF-IDF Baseline Performance by Year{title_suffix}", fontsize=TITLESIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(fontsize=LEGENDSIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    # Add total accuracy as horizontal line
    total_row = df[df["year"] == "Total"]
    if len(total_row) > 0:
        total_acc = total_row["accuracy"].values[0]
        ax.axhline(total_acc, color='black', linestyle='--', alpha=0.7, linewidth=1.5,
                  label=f'Overall: {total_acc:.2f}')
        ax.legend(fontsize=LEGENDSIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_importance(features_path: Path, output_path: Path, top_n: int = 15):
    """Plot top discriminative TF-IDF features."""
    df = pd.read_csv(features_path)

    accept_features = df["accept"].dropna().head(top_n).tolist()
    reject_features = df["reject"].dropna().head(top_n).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Left: Accept features
    ax1 = axes[0]
    y_pos = np.arange(len(accept_features))
    ax1.barh(y_pos, np.arange(len(accept_features), 0, -1), color=ACCEPT_COLOR, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(accept_features)
    ax1.set_xlabel("Importance Rank", fontsize=LABELSIZE)
    ax1.set_title("Top Features for Accept", fontsize=TITLESIZE)
    ax1.tick_params(axis='both', labelsize=TICKSIZE - 1)
    ax1.invert_yaxis()

    # Right: Reject features
    ax2 = axes[1]
    y_pos = np.arange(len(reject_features))
    ax2.barh(y_pos, np.arange(len(reject_features), 0, -1), color=REJECT_COLOR, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(reject_features)
    ax2.set_xlabel("Importance Rank", fontsize=LABELSIZE)
    ax2.set_title("Top Features for Reject", fontsize=TITLESIZE)
    ax2.tick_params(axis='both', labelsize=TICKSIZE - 1)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_tfidf_comparison(results: dict[str, pd.DataFrame], output_path: Path):
    """Plot comparison of TF-IDF across different data splits."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get total accuracy for each configuration
    configs = []
    accuracies = []
    accept_recalls = []
    reject_recalls = []

    for name, df in results.items():
        total_row = df[df["year"] == "Total"]
        if len(total_row) > 0:
            configs.append(name.replace("iclr_", "").replace("_", " "))
            accuracies.append(total_row["accuracy"].values[0])
            accept_recalls.append(total_row["accept_recall"].values[0])
            reject_recalls.append(total_row["reject_recall"].values[0])

    x = np.arange(len(configs))
    width = 0.25

    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', color=TFIDF_COLOR, alpha=0.8)
    bars2 = ax.bar(x, accept_recalls, width, label='Accept Recall', color=ACCEPT_COLOR, alpha=0.8)
    bars3 = ax.bar(x + width, reject_recalls, width, label='Reject Recall', color=REJECT_COLOR, alpha=0.8)

    ax.set_xlabel("Configuration", fontsize=LABELSIZE)
    ax.set_ylabel("Score", fontsize=LABELSIZE)
    ax.set_title("TF-IDF Performance Across Configurations", fontsize=TITLESIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha='right')
    ax.legend(fontsize=LEGENDSIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_tfidf_metrics_detail(df: pd.DataFrame, output_path: Path):
    """Plot detailed TF-IDF metrics including precision and F1."""
    # Filter to yearly rows
    df_years = df[df["year"] != "Total"].copy()
    df_years["year"] = df_years["year"].astype(int)
    df_years = df_years.sort_values("year")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    years = df_years["year"].values

    # Top left: Accuracy
    ax1 = axes[0, 0]
    ax1.plot(years, df_years["accuracy"], 'o-', color=TFIDF_COLOR, linewidth=2, markersize=8)
    ax1.fill_between(years, df_years["accuracy"], alpha=0.2, color=TFIDF_COLOR)
    ax1.set_xlabel("Year", fontsize=LABELSIZE)
    ax1.set_ylabel("Accuracy", fontsize=LABELSIZE)
    ax1.set_title("Accuracy by Year", fontsize=TITLESIZE)
    ax1.tick_params(axis='both', labelsize=TICKSIZE)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_ylim(0.5, 0.7)

    # Top right: Recall
    ax2 = axes[0, 1]
    ax2.plot(years, df_years["accept_recall"], 'o-', color=ACCEPT_COLOR, linewidth=2, markersize=8, label='Accept')
    ax2.plot(years, df_years["reject_recall"], 's--', color=REJECT_COLOR, linewidth=2, markersize=8, label='Reject')
    ax2.set_xlabel("Year", fontsize=LABELSIZE)
    ax2.set_ylabel("Recall", fontsize=LABELSIZE)
    ax2.set_title("Recall by Year", fontsize=TITLESIZE)
    ax2.legend(fontsize=LEGENDSIZE)
    ax2.tick_params(axis='both', labelsize=TICKSIZE)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_ylim(0.4, 0.8)

    # Bottom left: Precision
    ax3 = axes[1, 0]
    ax3.plot(years, df_years["accept_precision"], 'o-', color=ACCEPT_COLOR, linewidth=2, markersize=8, label='Accept')
    ax3.plot(years, df_years["reject_precision"], 's--', color=REJECT_COLOR, linewidth=2, markersize=8, label='Reject')
    ax3.set_xlabel("Year", fontsize=LABELSIZE)
    ax3.set_ylabel("Precision", fontsize=LABELSIZE)
    ax3.set_title("Precision by Year", fontsize=TITLESIZE)
    ax3.legend(fontsize=LEGENDSIZE)
    ax3.tick_params(axis='both', labelsize=TICKSIZE)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.set_ylim(0.5, 0.7)

    # Bottom right: F1
    ax4 = axes[1, 1]
    ax4.plot(years, df_years["accept_f1"], 'o-', color=ACCEPT_COLOR, linewidth=2, markersize=8, label='Accept')
    ax4.plot(years, df_years["reject_f1"], 's--', color=REJECT_COLOR, linewidth=2, markersize=8, label='Reject')
    ax4.set_xlabel("Year", fontsize=LABELSIZE)
    ax4.set_ylabel("F1 Score", fontsize=LABELSIZE)
    ax4.set_title("F1 Score by Year", fontsize=TITLESIZE)
    ax4.legend(fontsize=LEGENDSIZE)
    ax4.tick_params(axis='both', labelsize=TICKSIZE)
    ax4.grid(True, linestyle='--', alpha=0.3)
    ax4.set_ylim(0.5, 0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    RESULTS_DIR = Path("results/_tfidf")
    OUTPUT_DIR = Path("figures/latex/baseline")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_DIR.exists():
        print(f"TF-IDF results directory not found: {RESULTS_DIR}")
        return

    # Load all TF-IDF results
    print("Loading TF-IDF results...")
    results = load_tfidf_results(RESULTS_DIR)
    print(f"Loaded {len(results)} result files: {list(results.keys())}")

    # Generate figures
    print("\nGenerating figures...")

    # 1. Main TF-IDF year accuracy (2020-2025)
    main_result = "iclr_2020_2025"
    if main_result in results:
        plot_tfidf_year_accuracy(results[main_result], OUTPUT_DIR / "tfidf_year_accuracy.pdf")
        plot_tfidf_metrics_detail(results[main_result], OUTPUT_DIR / "tfidf_metrics_detail.pdf")

    # 2. Feature importance
    features_path = RESULTS_DIR / "iclr_2020_2025_features.csv"
    if features_path.exists():
        plot_feature_importance(features_path, OUTPUT_DIR / "tfidf_feature_importance.pdf")
    else:
        print(f"Features file not found: {features_path}")

    # 3. Comparison across configurations
    if len(results) > 1:
        plot_tfidf_comparison(results, OUTPUT_DIR / "tfidf_comparison.pdf")

    # 4. Additional configurations
    for name, df in results.items():
        if name != main_result and not name.endswith("_features"):
            safe_name = name.replace("/", "_")
            plot_tfidf_year_accuracy(df, OUTPUT_DIR / f"tfidf_year_accuracy_{safe_name}.pdf",
                                    title_suffix=f" ({name})")

    # Print summary
    print("\n" + "="*60)
    print("TF-IDF Summary")
    print("="*60)

    for name, df in results.items():
        total_row = df[df["year"] == "Total"]
        if len(total_row) > 0:
            acc = total_row["accuracy"].values[0]
            acc_rec = total_row["accept_recall"].values[0]
            rej_rec = total_row["reject_recall"].values[0]
            print(f"\n{name}:")
            print(f"  Accuracy: {acc:.3f}")
            print(f"  Accept Recall: {acc_rec:.3f}")
            print(f"  Reject Recall: {rej_rec:.3f}")

    print("\n" + "="*60)
    print("Done!")


if __name__ == "__main__":
    main()
