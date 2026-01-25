#!/usr/bin/env python3
"""
Extract and analyze structural features from paper content.

Generates:
- structural_features_by_decision.pdf: Violin plots of each feature by accept/reject
- feature_correlation_matrix.pdf: Correlation between all structural features
- feature_summary_stats.tex: Table of mean/std for each feature by decision

Usage:
    python scripts/latex_analysis/fig_structural_features.py
"""

import json
import re
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
LABELSIZE = 12
TITLESIZE = 14
LEGENDSIZE = 10
TICKSIZE = 10

# Colors
ACCEPT_COLOR = "#4CAF50"
REJECT_COLOR = "#F44336"

# Feature extraction patterns
FEATURE_PATTERNS = {
    # Math environments (LaTeX style)
    "equation_env": r'\\begin\{(equation|align|gather|multline|eqnarray)\*?\}',
    "inline_math": r'(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)',  # $...$ but not $$
    "display_math": r'\$\$[^$]+\$\$|\\\[[^\]]*\\\]',

    # Theorem-like environments (theoretical papers)
    "theorem_env": r'\\begin\{(theorem|lemma|proposition|corollary|definition|remark|proof)\}',

    # Visual elements
    "figure_marker": r'<image>|!\[.*?\]\(.*?\)|\\includegraphics',
    "table_marker": r'\\begin\{table\}|\\begin\{tabular\}|\|[^\|]+\|[^\|]+\|',

    # Algorithm/code
    "algorithm_marker": r'\\begin\{algorithm\}|\\begin\{lstlisting\}|```',

    # References and citations
    "citation": r'\\cite\{[^}]+\}|\([A-Z][a-z]+ et al\.,? \d{4}\)|\([A-Z][a-z]+, \d{4}\)',
}


def extract_structural_features(content: str) -> dict:
    """Extract LaTeX/structural features from paper content."""
    features = {}

    # Count pattern matches
    for name, pattern in FEATURE_PATTERNS.items():
        try:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            features[name] = len(matches)
        except re.error:
            features[name] = 0

    # Additional structural features
    # Sections (markdown style from converted papers)
    features["sections"] = len(re.findall(r'^#+\s+\S', content, re.MULTILINE))

    # Word/token count (approximate)
    features["word_count"] = len(content.split())

    # Paragraph count
    features["paragraph_count"] = len(re.split(r'\n\s*\n', content))

    # Bullet/list items
    features["list_items"] = len(re.findall(r'^[\s]*[-•*]\s+', content, re.MULTILINE))

    # Number of equations (combined)
    features["total_equations"] = (
        features.get("equation_env", 0) +
        features.get("inline_math", 0) +
        features.get("display_math", 0)
    )

    # Has significant math (binary)
    features["has_heavy_math"] = 1 if features["total_equations"] > 20 else 0

    # Has figures
    features["has_figures"] = 1 if features.get("figure_marker", 0) > 0 else 0

    # Has tables
    features["has_tables"] = 1 if features.get("table_marker", 0) > 0 else 0

    return features


def load_dataset_with_features(data_dir: Path, dataset_name: str, split: str) -> pd.DataFrame:
    """Load dataset and extract structural features."""
    path = data_dir / f"{dataset_name}_{split}" / "data.json"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    with open(path) as f:
        data = json.load(f)

    records = []
    for entry in data:
        metadata = entry.get("_metadata", {})
        conversations = entry.get("conversations", [])

        # Find the paper content (usually in human message)
        content = ""
        for conv in conversations:
            if conv.get("from") == "human":
                content = conv.get("value", "")
                break

        # Extract structural features
        features = extract_structural_features(content)

        # Add metadata
        features["submission_id"] = metadata.get("submission_id", "")
        features["answer"] = metadata.get("answer", "").lower()
        features["label"] = 1 if features["answer"] == "accept" else 0
        features["decision"] = "Accept" if features["answer"] == "accept" else "Reject"
        features["year"] = metadata.get("year", 0)
        features["pct_rating"] = metadata.get("pct_rating", np.nan)

        records.append(features)

    return pd.DataFrame(records)


def plot_structural_features_by_decision(df: pd.DataFrame, output_path: Path):
    """Plot violin plots of structural features by decision."""
    # Select features to plot
    feature_cols = [
        "total_equations", "inline_math", "display_math",
        "figure_marker", "table_marker", "citation",
        "sections", "word_count", "list_items"
    ]

    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(feature_cols):
        ax = axes[idx]

        accept_data = df[df["label"] == 1][col].dropna()
        reject_data = df[df["label"] == 0][col].dropna()

        # Create violin plot
        parts = ax.violinplot(
            [reject_data.values, accept_data.values],
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

        # Statistical test
        try:
            t_stat, p_value = stats.mannwhitneyu(accept_data, reject_data, alternative='two-sided')
            sig = "*" if p_value < 0.05 else ""
            sig += "*" if p_value < 0.01 else ""
            sig += "*" if p_value < 0.001 else ""
        except:
            p_value = 1.0
            sig = ""

        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            f"Reject\n$\\mu$={reject_data.mean():.1f}",
            f"Accept\n$\\mu$={accept_data.mean():.1f}"
        ], fontsize=TICKSIZE)

        # Clean up column name for title
        col_display = col.replace("_", " ").title()
        ax.set_title(f"{col_display} {sig}\n($p$={p_value:.2e})", fontsize=TITLESIZE - 2)
        ax.tick_params(axis='both', labelsize=TICKSIZE)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Hide unused axes
    for idx in range(len(feature_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_correlation_matrix(df: pd.DataFrame, output_path: Path):
    """Plot correlation matrix between structural features."""
    feature_cols = [
        "total_equations", "inline_math", "figure_marker",
        "table_marker", "citation", "sections", "word_count"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Compute correlation matrix
    corr_matrix = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                cbar_kws={"label": "Pearson Correlation"})

    # Clean up labels
    labels = [c.replace("_", " ").title() for c in feature_cols]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=TICKSIZE)
    ax.set_yticklabels(labels, rotation=0, fontsize=TICKSIZE)
    ax.set_title("Structural Feature Correlations", fontsize=TITLESIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_features_vs_pct_rating(df: pd.DataFrame, output_path: Path):
    """Plot structural features vs pct_rating."""
    feature_cols = ["total_equations", "figure_marker", "citation", "word_count"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, col in enumerate(feature_cols):
        ax = axes[idx]

        valid_df = df[df["pct_rating"].notna()].copy()

        # Scatter with color by decision
        accept_df = valid_df[valid_df["label"] == 1]
        reject_df = valid_df[valid_df["label"] == 0]

        ax.scatter(reject_df["pct_rating"], reject_df[col],
                  c=REJECT_COLOR, alpha=0.3, s=20, label="Reject")
        ax.scatter(accept_df["pct_rating"], accept_df[col],
                  c=ACCEPT_COLOR, alpha=0.3, s=20, label="Accept")

        # Compute correlation
        corr, p_corr = stats.pearsonr(valid_df["pct_rating"].values, valid_df[col].values)

        col_display = col.replace("_", " ").title()
        ax.set_xlabel("Percentile Rating", fontsize=LABELSIZE)
        ax.set_ylabel(col_display, fontsize=LABELSIZE)
        ax.set_title(f"{col_display} vs Rating\n($r$={corr:.3f}, $p$={p_corr:.2e})", fontsize=TITLESIZE - 1)
        ax.legend(fontsize=LEGENDSIZE - 1, loc='upper right')
        ax.tick_params(axis='both', labelsize=TICKSIZE)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_feature_summary_table(df: pd.DataFrame, output_path: Path):
    """Generate LaTeX table of feature summary statistics."""
    feature_cols = [
        ("total_equations", "Total Equations"),
        ("inline_math", "Inline Math"),
        ("display_math", "Display Math"),
        ("figure_marker", "Figures"),
        ("table_marker", "Tables"),
        ("citation", "Citations"),
        ("sections", "Sections"),
        ("word_count", "Word Count"),
        ("list_items", "List Items"),
    ]

    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Structural Feature Statistics by Decision}")
    latex.append(r"\label{tab:structural_features}")
    latex.append(r"\begin{tabular}{l|rr|rr|rr}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Feature} & \multicolumn{2}{c|}{\textbf{Accept}} & \multicolumn{2}{c|}{\textbf{Reject}} & \multicolumn{2}{c}{\textbf{Difference}} \\")
    latex.append(r" & Mean & Std & Mean & Std & $\Delta$ & $p$-value \\")
    latex.append(r"\hline")

    accept_df = df[df["label"] == 1]
    reject_df = df[df["label"] == 0]

    for col, display_name in feature_cols:
        if col not in df.columns:
            continue

        accept_mean = accept_df[col].mean()
        accept_std = accept_df[col].std()
        reject_mean = reject_df[col].mean()
        reject_std = reject_df[col].std()
        diff = accept_mean - reject_mean

        try:
            _, p_value = stats.mannwhitneyu(
                accept_df[col].dropna(),
                reject_df[col].dropna(),
                alternative='two-sided'
            )
        except:
            p_value = 1.0

        # Format p-value
        if p_value < 0.001:
            p_str = "$<$0.001"
        else:
            p_str = f"{p_value:.3f}"

        latex.append(f"{display_name} & {accept_mean:.1f} & {accept_std:.1f} & "
                    f"{reject_mean:.1f} & {reject_std:.1f} & {diff:+.1f} & {p_str} \\\\")

    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"Saved: {output_path}")


def main():
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("figures/latex/analysis")
    TABLE_DIR = Path("figures/latex/tables")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    dataset_name = "iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6"

    print("Loading datasets and extracting features...")
    # Load all splits
    dfs = []
    for split in ["train", "test", "validation"]:
        df_split = load_dataset_with_features(DATA_DIR, dataset_name, split)
        if len(df_split) > 0:
            dfs.append(df_split)
            print(f"  {split}: {len(df_split)} entries")

    if not dfs:
        print("Error: No data loaded")
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal entries: {len(df)}")
    print(f"  Accept: {len(df[df['label'] == 1])}")
    print(f"  Reject: {len(df[df['label'] == 0])}")

    # Print feature statistics
    print("\nFeature Extraction Summary:")
    for col in ["total_equations", "figure_marker", "table_marker", "citation", "word_count"]:
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.1f}, max={df[col].max()}")

    # Generate figures
    print("\nGenerating figures...")

    # 1. Structural features by decision
    plot_structural_features_by_decision(df, OUTPUT_DIR / "structural_features_by_decision.pdf")

    # 2. Feature correlation matrix
    plot_feature_correlation_matrix(df, OUTPUT_DIR / "feature_correlation_matrix.pdf")

    # 3. Features vs pct_rating
    plot_features_vs_pct_rating(df, OUTPUT_DIR / "features_vs_pct_rating.pdf")

    # 4. Generate summary table
    generate_feature_summary_table(df, TABLE_DIR / "feature_summary_stats.tex")

    # Save extracted features for other scripts
    features_path = OUTPUT_DIR / "structural_features.csv"
    feature_cols = [c for c in df.columns if c not in ["answer"]]
    df[feature_cols].to_csv(features_path, index=False)
    print(f"Saved features CSV: {features_path}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
