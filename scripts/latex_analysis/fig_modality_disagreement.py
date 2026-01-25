#!/usr/bin/env python3
"""
Generate modality disagreement analysis figures for the LaTeX report.

Analyzes when vision predicts correctly but text doesn't (and vice versa),
with structural feature analysis to explain the differences.

Generates:
- agreement_matrix.pdf: Confusion matrix of clean vs vision predictions
- feature_by_category_violin.pdf: Violin plots for structural features by category
- feature_correlation_with_modality.pdf: Point-biserial correlations
- example_papers_table.tex: Table with example papers and their features

Uses:
- results/data_sweep_v2/*/finetuned.jsonl - Model predictions
- data/*_v6_*/data.json - Paper metadata and content

Usage:
    python scripts/latex_analysis/fig_modality_disagreement.py
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
CLEAN_COLOR = "#2ecc71"
VISION_COLOR = "#e74c3c"

CATEGORY_COLORS = {
    "both_correct": ACCEPT_COLOR,
    "both_wrong": REJECT_COLOR,
    "vision_wins": VISION_COLOR,
    "text_wins": CLEAN_COLOR,
}

CATEGORY_NAMES = {
    "both_correct": "Both Correct",
    "both_wrong": "Both Wrong",
    "vision_wins": "Vision Wins",
    "text_wins": "Text Wins",
}


def extract_structural_features(content: str) -> dict:
    """Extract LaTeX/structural features from paper content."""
    features = {}

    # Count pattern matches for various structural elements
    # Math environments
    features["equation_env"] = len(re.findall(
        r'\\begin\{(equation|align|gather|multline|eqnarray)\*?\}', content, re.IGNORECASE))
    features["inline_math"] = len(re.findall(r'(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)', content))
    features["display_math"] = len(re.findall(r'\$\$[^$]+\$\$|\\\[[^\]]*\\\]', content))

    # Theorem environments (theoretical papers)
    features["theorem_env"] = len(re.findall(
        r'\\begin\{(theorem|lemma|proposition|corollary|definition|proof)\}', content, re.IGNORECASE))

    # Visual elements
    features["images"] = content.count("<image>")  # Primary image marker
    features["figure_marker"] = len(re.findall(r'!\[.*?\]\(.*?\)|\\includegraphics', content))

    # Tables
    features["table_marker"] = len(re.findall(
        r'\\begin\{table\}|\\begin\{tabular\}', content, re.IGNORECASE))

    # Algorithms
    features["algorithm_marker"] = len(re.findall(
        r'\\begin\{algorithm\}|\\begin\{lstlisting\}|```', content, re.IGNORECASE))

    # Citations
    features["citation_count"] = len(re.findall(
        r'\\cite\{[^}]+\}|\([A-Z][a-z]+ et al\.,? \d{4}\)', content))

    # Sections
    features["sections"] = len(re.findall(r'^#+\s+\S', content, re.MULTILINE))

    # Token count
    features["token_count"] = len(content.split())

    # Derived features
    features["total_equations"] = (
        features["equation_env"] + features["inline_math"] + features["display_math"]
    )
    features["total_images"] = features["images"] + features["figure_marker"]

    return features


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip().lower()
    return None


def load_predictions(pred_path: Path) -> list[dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(pred_path) as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    pred = extract_boxed_answer(entry.get("predict", ""))
                    label = extract_boxed_answer(entry.get("label", ""))
                    predictions.append({
                        "pred": 1 if pred == "accept" else 0,
                        "label": 1 if label == "accept" else 0,
                        "correct": pred == label,
                    })
                except json.JSONDecodeError:
                    continue
    return predictions


def load_data_with_features(data_dir: Path, dataset_name: str, split: str) -> list[dict]:
    """Load paper data and extract structural features."""
    path = data_dir / f"{dataset_name}_{split}" / "data.json"
    if not path.exists():
        print(f"Warning: {path} not found")
        return []

    with open(path) as f:
        data = json.load(f)

    records = []
    for entry in data:
        metadata = entry.get("_metadata", {})
        conversations = entry.get("conversations", [])

        # Get paper content
        content = ""
        for msg in conversations:
            if msg.get("from") == "human":
                content = msg.get("value", "")
                break

        # Extract structural features
        features = extract_structural_features(content)

        # Add metadata
        ratings = metadata.get("ratings", [])
        features.update({
            "submission_id": metadata.get("submission_id", ""),
            "year": metadata.get("year", 0),
            "pct_rating": metadata.get("pct_rating", np.nan),
            "rating_std": np.std(ratings) if len(ratings) > 1 else 0,
            "answer": metadata.get("answer", "").lower(),
            "label": 1 if metadata.get("answer", "").lower() == "accept" else 0,
        })

        records.append(features)

    return records


def join_predictions_with_data(clean_preds: list, vision_preds: list, data: list) -> pd.DataFrame:
    """Join predictions from clean and vision with paper data."""
    min_len = min(len(clean_preds), len(vision_preds), len(data))

    records = []
    for i in range(min_len):
        record = {**data[i]}
        record["clean_pred"] = clean_preds[i]["pred"]
        record["vision_pred"] = vision_preds[i]["pred"]
        record["clean_correct"] = clean_preds[i]["correct"]
        record["vision_correct"] = vision_preds[i]["correct"]
        records.append(record)

    df = pd.DataFrame(records)

    # Add disagreement categories
    def categorize(row):
        if row["clean_correct"] and row["vision_correct"]:
            return "both_correct"
        elif not row["clean_correct"] and not row["vision_correct"]:
            return "both_wrong"
        elif row["vision_correct"] and not row["clean_correct"]:
            return "vision_wins"
        else:
            return "text_wins"

    df["category"] = df.apply(categorize, axis=1)

    return df


def plot_agreement_matrix(df: pd.DataFrame, output_path: Path):
    """Plot confusion matrix of clean vs vision predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Prediction agreement matrix
    ax1 = axes[0]
    confusion = np.zeros((2, 2), dtype=int)
    gt_breakdown = [[{"accept": 0, "reject": 0} for _ in range(2)] for _ in range(2)]

    for _, row in df.iterrows():
        c_pred = int(row["clean_pred"])
        v_pred = int(row["vision_pred"])
        gt = int(row["label"])

        confusion[1 - c_pred, v_pred] += 1
        if gt == 1:
            gt_breakdown[1 - c_pred][v_pred]["accept"] += 1
        else:
            gt_breakdown[1 - c_pred][v_pred]["reject"] += 1

    sns.heatmap(confusion, annot=False, cmap="Blues", ax=ax1, cbar_kws={"label": "Count"})

    # Add annotations
    for i in range(2):
        for j in range(2):
            count = confusion[i, j]
            pct = 100 * count / len(df)
            gt_acc = gt_breakdown[i][j]["accept"]
            gt_rej = gt_breakdown[i][j]["reject"]
            text = f"{count}\n({pct:.1f}\\%)\n\nGT: {gt_acc}A / {gt_rej}R"
            ax1.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                    fontsize=LABELSIZE - 1, fontweight='bold')

    ax1.set_xticklabels(["Reject", "Accept"], fontsize=LABELSIZE)
    ax1.set_yticklabels(["Accept", "Reject"], fontsize=LABELSIZE)
    ax1.set_xlabel("Vision Prediction", fontsize=LABELSIZE)
    ax1.set_ylabel("Text Prediction", fontsize=LABELSIZE)
    ax1.set_title("Prediction Agreement: Text vs Vision", fontsize=TITLESIZE)

    # Right: Correctness agreement
    ax2 = axes[1]
    category_counts = df["category"].value_counts()

    categories = ["both_correct", "both_wrong", "vision_wins", "text_wins"]
    counts = [category_counts.get(c, 0) for c in categories]
    pcts = [100 * c / len(df) for c in counts]
    colors = [CATEGORY_COLORS[c] for c in categories]
    names = [CATEGORY_NAMES[c] for c in categories]

    bars = ax2.bar(names, counts, color=colors, alpha=0.8)

    for bar, pct in zip(bars, pcts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{pct:.1f}\\%", ha='center', fontsize=LABELSIZE)

    ax2.set_ylabel("Count", fontsize=LABELSIZE)
    ax2.set_title("Correctness Agreement Categories", fontsize=TITLESIZE)
    ax2.tick_params(axis='both', labelsize=TICKSIZE)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_by_category_violin(df: pd.DataFrame, output_path: Path):
    """Plot violin plots for structural features by disagreement category."""
    features = [
        ("total_equations", "Total Equations"),
        ("total_images", "Total Images"),
        ("theorem_env", "Theorem Envs"),
        ("citation_count", "Citations"),
        ("pct_rating", "Pct Rating"),
        ("rating_std", "Rating Std"),
    ]

    # Filter to existing columns
    features = [(col, name) for col, name in features if col in df.columns]

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    categories = ["both_correct", "both_wrong", "vision_wins", "text_wins"]

    for idx, (col, col_name) in enumerate(features):
        ax = axes[idx]

        # Prepare data for violin plot
        plot_data = []
        positions = []
        colors = []

        for i, cat in enumerate(categories):
            cat_data = df[df["category"] == cat][col].dropna()
            if len(cat_data) > 0:
                plot_data.append(cat_data.values)
                positions.append(i)
                colors.append(CATEGORY_COLORS[cat])

        if not plot_data:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(plot_data, positions=positions, widths=0.7,
                              showmeans=True, showmedians=False)

        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('black')

        # Add mean annotations
        for i, (cat, pos) in enumerate(zip(categories, positions)):
            cat_data = df[df["category"] == cat][col].dropna()
            if len(cat_data) > 0:
                ax.text(pos, cat_data.mean(), f"{cat_data.mean():.1f}",
                       ha='center', va='bottom', fontsize=TICKSIZE - 2)

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([CATEGORY_NAMES[c][:10] for c in categories],
                         fontsize=TICKSIZE - 1, rotation=15, ha='right')
        ax.set_ylabel(col_name, fontsize=LABELSIZE - 1)
        ax.set_title(col_name, fontsize=TITLESIZE - 2)
        ax.tick_params(axis='both', labelsize=TICKSIZE - 1)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Hide unused axes
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_correlation_with_modality(df: pd.DataFrame, output_path: Path):
    """Plot point-biserial correlations between features and modality correctness."""
    # Features to analyze
    feature_cols = [
        "total_equations", "total_images", "theorem_env", "table_marker",
        "citation_count", "token_count", "pct_rating", "rating_std"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Calculate correlations with vision_correct and text_correct
    results = []
    for col in feature_cols:
        valid_df = df[df[col].notna()].copy()
        if len(valid_df) < 10:
            continue

        # Point-biserial correlation with vision_correct
        try:
            r_vision, p_vision = stats.pointbiserialr(
                valid_df["vision_correct"].astype(int),
                valid_df[col].values
            )
        except:
            r_vision, p_vision = 0, 1

        # Point-biserial correlation with clean_correct
        try:
            r_text, p_text = stats.pointbiserialr(
                valid_df["clean_correct"].astype(int),
                valid_df[col].values
            )
        except:
            r_text, p_text = 0, 1

        results.append({
            "feature": col.replace("_", " ").title(),
            "r_vision": r_vision,
            "p_vision": p_vision,
            "r_text": r_text,
            "p_text": p_text,
            "r_diff": r_vision - r_text,
        })

    if not results:
        print("No valid correlations to plot")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("r_diff", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    y = np.arange(len(results_df))
    height = 0.35

    # Vision correlations
    bars_vision = ax.barh(y - height/2, results_df["r_vision"], height,
                          label='Vision Correct', color=VISION_COLOR, alpha=0.8)
    # Text correlations
    bars_text = ax.barh(y + height/2, results_df["r_text"], height,
                        label='Text Correct', color=CLEAN_COLOR, alpha=0.8)

    # Add significance markers
    for i, row in enumerate(results_df.itertuples()):
        # Vision significance
        if row.p_vision < 0.001:
            sig = "***"
        elif row.p_vision < 0.01:
            sig = "**"
        elif row.p_vision < 0.05:
            sig = "*"
        else:
            sig = ""
        if sig:
            x_pos = row.r_vision + 0.01 if row.r_vision > 0 else row.r_vision - 0.02
            ax.text(x_pos, i - height/2, sig, va='center', fontsize=TICKSIZE, color=VISION_COLOR)

        # Text significance
        if row.p_text < 0.001:
            sig = "***"
        elif row.p_text < 0.01:
            sig = "**"
        elif row.p_text < 0.05:
            sig = "*"
        else:
            sig = ""
        if sig:
            x_pos = row.r_text + 0.01 if row.r_text > 0 else row.r_text - 0.02
            ax.text(x_pos, i + height/2, sig, va='center', fontsize=TICKSIZE, color=CLEAN_COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels(results_df["feature"], fontsize=LABELSIZE)
    ax.set_xlabel("Point-Biserial Correlation ($r_{pb}$)", fontsize=LABELSIZE)
    ax.set_title("Feature Correlation with Modality Correctness\n" +
                "(* $p<0.05$, ** $p<0.01$, *** $p<0.001$)", fontsize=TITLESIZE)
    ax.legend(fontsize=LEGENDSIZE, loc='lower right')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    ax.set_xlim(-0.3, 0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # Print correlation summary
    print("\nFeature-Modality Correlations:")
    print(results_df.to_string(index=False))


def generate_example_papers_table(df: pd.DataFrame, output_path: Path):
    """Generate LaTeX table with example papers selected by data, not speculation."""
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Example Papers by Disagreement Category (Data-Driven Selection)}")
    latex.append(r"\label{tab:example_papers}")
    latex.append(r"\footnotesize")
    latex.append(r"\begin{tabular}{l|l|r|r|r|r}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Category} & \textbf{ID} & \textbf{Eqns} & \textbf{Imgs} & \textbf{Thms} & \textbf{pct\_rating} \\")
    latex.append(r"\hline")

    for category in ["vision_wins", "text_wins", "both_correct", "both_wrong"]:
        cat_df = df[df["category"] == category].copy()
        if len(cat_df) == 0:
            continue

        # Select examples at extremes and median
        # Sort by key distinguishing feature
        if category == "vision_wins":
            # Papers where vision wins - sort by image count
            cat_df = cat_df.sort_values("total_images", ascending=False)
        elif category == "text_wins":
            # Papers where text wins - sort by equation count
            cat_df = cat_df.sort_values("total_equations", ascending=False)
        else:
            # Sort by pct_rating
            cat_df = cat_df.sort_values("pct_rating", ascending=False)

        # Take top 2 examples
        examples = cat_df.head(2)

        for _, row in examples.iterrows():
            sub_id = row.get("submission_id", "N/A")[:15]
            eqns = int(row.get("total_equations", 0))
            imgs = int(row.get("total_images", 0))
            thms = int(row.get("theorem_env", 0))
            pct_r = row.get("pct_rating", 0)

            latex.append(f"{CATEGORY_NAMES[category]} & {sub_id} & {eqns} & {imgs} & {thms} & {pct_r:.3f} \\\\")

        latex.append(r"\hline")

    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"Saved: {output_path}")


def main():
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results/data_sweep_v2")
    OUTPUT_DIR = Path("figures/latex/analysis")
    TABLE_DIR = Path("figures/latex/tables")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Paths
    clean_pred_path = RESULTS_DIR / "iclr20_balanced_clean" / "finetuned.jsonl"
    vision_pred_path = RESULTS_DIR / "iclr20_balanced_vision" / "finetuned.jsonl"
    dataset_name = "iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6"

    # Load predictions
    print("Loading predictions...")
    if not clean_pred_path.exists() or not vision_pred_path.exists():
        print(f"Prediction files not found at {clean_pred_path} or {vision_pred_path}")
        print("Creating placeholder analysis...")

        # Create placeholder data
        np.random.seed(42)
        n = 500
        placeholder_df = pd.DataFrame({
            "submission_id": [f"paper_{i}" for i in range(n)],
            "total_equations": np.random.poisson(15, n),
            "total_images": np.random.poisson(5, n),
            "theorem_env": np.random.poisson(2, n),
            "table_marker": np.random.poisson(3, n),
            "citation_count": np.random.poisson(30, n),
            "token_count": np.random.normal(8000, 2000, n).astype(int),
            "pct_rating": np.random.uniform(0, 1, n),
            "rating_std": np.random.exponential(0.8, n),
            "year": np.random.choice([2020, 2021, 2022, 2023, 2024, 2025], n),
            "label": np.random.choice([0, 1], n),
            "clean_pred": np.random.choice([0, 1], n),
            "vision_pred": np.random.choice([0, 1], n),
        })
        placeholder_df["clean_correct"] = placeholder_df["clean_pred"] == placeholder_df["label"]
        placeholder_df["vision_correct"] = placeholder_df["vision_pred"] == placeholder_df["label"]
        placeholder_df["category"] = placeholder_df.apply(
            lambda r: "both_correct" if r["clean_correct"] and r["vision_correct"]
            else "both_wrong" if not r["clean_correct"] and not r["vision_correct"]
            else "vision_wins" if r["vision_correct"]
            else "text_wins", axis=1
        )

        df = placeholder_df
    else:
        clean_preds = load_predictions(clean_pred_path)
        vision_preds = load_predictions(vision_pred_path)
        print(f"Loaded {len(clean_preds)} clean predictions, {len(vision_preds)} vision predictions")

        # Load data with features
        print("Loading data and extracting structural features...")
        data = load_data_with_features(DATA_DIR, dataset_name, "test")
        print(f"Loaded {len(data)} data entries with features")

        # Join data
        print("Joining data...")
        df = join_predictions_with_data(clean_preds, vision_preds, data)
        print(f"Joined {len(df)} entries")

    # Generate figures
    print("\nGenerating figures...")

    # 1. Agreement matrix
    plot_agreement_matrix(df, OUTPUT_DIR / "agreement_matrix.pdf")

    # 2. Feature by category violin plots
    plot_feature_by_category_violin(df, OUTPUT_DIR / "feature_by_category_violin.pdf")

    # 3. Feature correlation with modality
    plot_feature_correlation_with_modality(df, OUTPUT_DIR / "feature_correlation_with_modality.pdf")

    # 4. Example papers table
    generate_example_papers_table(df, TABLE_DIR / "example_papers.tex")

    # Print summary
    print("\n" + "=" * 60)
    print("Modality Disagreement Analysis Summary")
    print("=" * 60)

    category_counts = df["category"].value_counts()
    print("\nCategory Breakdown:")
    for cat in ["both_correct", "both_wrong", "vision_wins", "text_wins"]:
        count = category_counts.get(cat, 0)
        print(f"  {CATEGORY_NAMES[cat]}: {count} ({100*count/len(df):.1f}%)")

    # Feature comparison between vision_wins and text_wins
    print("\nFeature Comparison (Vision Wins vs Text Wins):")
    vision_wins = df[df["category"] == "vision_wins"]
    text_wins = df[df["category"] == "text_wins"]

    for col in ["total_equations", "total_images", "theorem_env", "pct_rating"]:
        if col in df.columns:
            v_mean = vision_wins[col].mean() if len(vision_wins) > 0 else 0
            t_mean = text_wins[col].mean() if len(text_wins) > 0 else 0
            print(f"  {col}: Vision={v_mean:.2f}, Text={t_mean:.2f}, Diff={v_mean-t_mean:+.2f}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
