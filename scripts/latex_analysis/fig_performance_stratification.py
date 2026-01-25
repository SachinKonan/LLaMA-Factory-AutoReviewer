#!/usr/bin/env python3
"""
Generate performance stratification figures - accuracy by paper characteristics.

Analyzes what types of papers the model is good/bad at predicting.

Mathematical definitions:
- Accuracy within stratum S: Acc(S) = |{x ∈ S : ŷ_x = y_x}| / |S|
- Rating disagreement stratum: S_σ = {x : σ_r(x) ∈ [σ_min, σ_max)}
- Borderline papers: B = {x : 0.4 ≤ pct_rating(x) ≤ 0.6}

Generates:
- acc_vs_rating_std_with_ci.pdf: Accuracy vs reviewer disagreement with 95% CI
- acc_vs_pct_rating_violin.pdf: Violin + accuracy overlay by pct_rating
- acc_heatmap_2d.pdf: 2D heatmap with cell counts
- borderline_analysis.pdf: Deep dive into borderline papers

Usage:
    python scripts/latex_analysis/fig_performance_stratification.py
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
from scipy.stats import binomtest

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


def load_metadata(data_dir: Path, dataset_name: str, split: str) -> pd.DataFrame:
    """Load paper metadata from data.json."""
    path = data_dir / f"{dataset_name}_{split}" / "data.json"
    if not path.exists():
        return pd.DataFrame()

    with open(path) as f:
        data = json.load(f)

    records = []
    for entry in data:
        metadata = entry.get("_metadata", {})
        conversations = entry.get("conversations", [])

        human_content = ""
        for msg in conversations:
            if msg.get("from") == "human":
                human_content = msg.get("value", "")
                break

        tokens = len(human_content.split())
        ratings = metadata.get("ratings", [])
        rating_std = np.std(ratings) if len(ratings) > 1 else 0

        records.append({
            "submission_id": metadata.get("submission_id", ""),
            "year": metadata.get("year", 0),
            "tokens": tokens,
            "rating_std": rating_std,
            "pct_rating": metadata.get("pct_rating", np.nan),
        })

    return pd.DataFrame(records)


def join_predictions(preds: list[dict], metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Join predictions with metadata."""
    min_len = min(len(preds), len(metadata_df))
    df = metadata_df.iloc[:min_len].copy()
    df["pred"] = [p["pred"] for p in preds[:min_len]]
    df["label"] = [p["label"] for p in preds[:min_len]]
    df["correct"] = [p["correct"] for p in preds[:min_len]]
    return df


def compute_accuracy_with_ci(correct_series: pd.Series) -> dict:
    """Compute accuracy with 95% confidence interval."""
    n = len(correct_series)
    correct = correct_series.sum()
    acc = correct / n if n > 0 else 0

    if n > 0:
        ci = binomtest(int(correct), n).proportion_ci(confidence_level=0.95)
        return {"accuracy": acc, "ci_low": ci.low, "ci_high": ci.high, "n": n}
    return {"accuracy": 0, "ci_low": 0, "ci_high": 0, "n": 0}


def plot_acc_vs_rating_std_with_ci(df: pd.DataFrame, output_path: Path):
    """Plot accuracy vs reviewer disagreement with 95% CI bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bins
    bins = [0, 0.5, 1.0, 1.5, 2.0, 5.0]
    bin_labels = ["$\\sigma_r < 0.5$", "$0.5 \\leq \\sigma_r < 1.0$",
                  "$1.0 \\leq \\sigma_r < 1.5$", "$1.5 \\leq \\sigma_r < 2.0$",
                  "$\\sigma_r \\geq 2.0$"]

    df["rating_std_bin"] = pd.cut(df["rating_std"], bins=bins,
                                   labels=range(len(bin_labels)), right=False)

    # Calculate accuracy with CI per bin
    results = []
    for i, label in enumerate(bin_labels):
        bin_data = df[df["rating_std_bin"] == i]["correct"]
        if len(bin_data) > 0:
            stats_dict = compute_accuracy_with_ci(bin_data)
            stats_dict["bin"] = i
            stats_dict["label"] = label
            results.append(stats_dict)

    results_df = pd.DataFrame(results)

    # Plot bars with error bars
    x = np.arange(len(results_df))
    bars = ax.bar(x, results_df["accuracy"], color=CLEAN_COLOR, alpha=0.7, width=0.6)

    # Add 95% CI error bars
    yerr_lower = results_df["accuracy"] - results_df["ci_low"]
    yerr_upper = results_df["ci_high"] - results_df["accuracy"]
    ax.errorbar(x, results_df["accuracy"], yerr=[yerr_lower, yerr_upper],
               fmt='none', color='black', capsize=5, capthick=2)

    # Add annotations
    for i, row in results_df.iterrows():
        ax.text(row["bin"], row["ci_high"] + 0.02,
               f"{row['accuracy']:.1%}\n[{row['ci_low']:.1%}-{row['ci_high']:.1%}]\n(n={row['n']})",
               ha='center', fontsize=TICKSIZE - 1, va='bottom')

    ax.set_xlabel("Rating Standard Deviation $\\sigma_r$ (Reviewer Disagreement)", fontsize=LABELSIZE)
    ax.set_ylabel("Accuracy", fontsize=LABELSIZE)
    ax.set_title("Model Accuracy vs Reviewer Disagreement (with 95\\% CI)\n" +
                r"$\text{Acc}(S_{\sigma}) = |\{x \in S_{\sigma} : \hat{y}_x = y_x\}| / |S_{\sigma}|$",
                fontsize=TITLESIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["label"], fontsize=TICKSIZE - 1)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_ylim(0.4, 0.9)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_acc_vs_pct_rating_violin(df: pd.DataFrame, output_path: Path):
    """Plot violin + accuracy overlay by pct_rating quintiles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Define quintiles
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0-0.2\nClear Reject", "0.2-0.4", "0.4-0.6\nBorderline",
                  "0.6-0.8", "0.8-1.0\nClear Accept"]

    df["pct_bin"] = pd.cut(df["pct_rating"], bins=bins, labels=range(len(bin_labels)), right=True)

    # Left: Violin plot showing pct_rating distribution by correctness
    ax1 = axes[0]

    correct_df = df[df["correct"] == True]
    incorrect_df = df[df["correct"] == False]

    parts = ax1.violinplot([correct_df["pct_rating"].dropna().values,
                            incorrect_df["pct_rating"].dropna().values],
                           positions=[0, 1], widths=0.8, showmeans=True, showmedians=True)

    colors = [ACCEPT_COLOR, REJECT_COLOR]
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('white')

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([
        f"Correct\n$\\mu$={correct_df['pct_rating'].mean():.3f}\nn={len(correct_df)}",
        f"Incorrect\n$\\mu$={incorrect_df['pct_rating'].mean():.3f}\nn={len(incorrect_df)}"
    ], fontsize=LABELSIZE)
    ax1.set_ylabel("Percentile Rating (pct\\_rating)", fontsize=LABELSIZE)
    ax1.set_title("pct\\_rating Distribution by Prediction Correctness", fontsize=TITLESIZE)
    ax1.tick_params(axis='both', labelsize=TICKSIZE)
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Mann-Whitney test
    u_stat, p_value = stats.mannwhitneyu(correct_df["pct_rating"].dropna(),
                                         incorrect_df["pct_rating"].dropna())
    ax1.text(0.5, 0.02, f"Mann-Whitney $U$={u_stat:.0f}, $p$<{max(p_value, 1e-100):.1e}",
            transform=ax1.transAxes, ha='center', fontsize=TICKSIZE)

    # Right: Accuracy by pct_rating bin with CI
    ax2 = axes[1]

    results = []
    for i, label in enumerate(bin_labels):
        bin_data = df[df["pct_bin"] == i]["correct"]
        if len(bin_data) > 0:
            stats_dict = compute_accuracy_with_ci(bin_data)
            stats_dict["bin"] = i
            stats_dict["label"] = label
            results.append(stats_dict)

    results_df = pd.DataFrame(results)

    # Color by difficulty (borderline = red)
    colors = [ACCEPT_COLOR, '#f39c12', REJECT_COLOR, '#f39c12', ACCEPT_COLOR]

    x = np.arange(len(results_df))
    bars = ax2.bar(x, results_df["accuracy"], color=[colors[int(r)] for r in results_df["bin"]],
                   alpha=0.7, width=0.6)

    # Add CI error bars
    yerr_lower = results_df["accuracy"] - results_df["ci_low"]
    yerr_upper = results_df["ci_high"] - results_df["accuracy"]
    ax2.errorbar(x, results_df["accuracy"], yerr=[yerr_lower, yerr_upper],
                fmt='none', color='black', capsize=5, capthick=2)

    # Add annotations
    for i, row in results_df.iterrows():
        ax2.text(row["bin"], row["ci_high"] + 0.02,
                f"{row['accuracy']:.1%}\n(n={row['n']})",
                ha='center', fontsize=TICKSIZE - 1, va='bottom')

    ax2.set_xlabel("Percentile Rating (pct\\_rating)", fontsize=LABELSIZE)
    ax2.set_ylabel("Accuracy", fontsize=LABELSIZE)
    ax2.set_title("Model Accuracy by Rating Position\n" +
                 r"$B = \{x : 0.4 \leq \text{pct\_rating}(x) \leq 0.6\}$ (Borderline)",
                 fontsize=TITLESIZE)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["label"] for _, r in results_df.iterrows()], fontsize=TICKSIZE - 2)
    ax2.tick_params(axis='both', labelsize=TICKSIZE)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax2.set_ylim(0.4, 0.95)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_acc_heatmap_2d(df: pd.DataFrame, output_path: Path):
    """Plot 2D heatmap of accuracy by rating_std × pct_rating with cell counts."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create bins
    std_bins = [0, 0.5, 1.0, 1.5, 5.0]
    std_labels = ["$\\sigma_r<0.5$", "0.5-1.0", "1.0-1.5", "$\\sigma_r\\geq1.5$"]

    pct_bins = [0, 0.3, 0.5, 0.7, 1.0]
    pct_labels = ["pct<0.3", "0.3-0.5", "0.5-0.7", "pct>0.7"]

    df["std_bin"] = pd.cut(df["rating_std"], bins=std_bins, labels=std_labels, right=False)
    df["pct_bin"] = pd.cut(df["pct_rating"], bins=pct_bins, labels=pct_labels, right=True)

    # Create pivot table for accuracy
    pivot = df.pivot_table(values="correct", index="pct_bin", columns="std_bin", aggfunc="mean")
    counts = df.pivot_table(values="correct", index="pct_bin", columns="std_bin", aggfunc="count")

    # Reorder index for logical display (low pct at top)
    pivot = pivot.reindex(pct_labels[::-1])
    counts = counts.reindex(pct_labels[::-1])

    # Create annotation with acc and count
    annot = pivot.copy()
    for i in range(len(pct_labels)):
        for j in range(len(std_labels)):
            try:
                acc_val = pivot.iloc[i, j]
                count_val = counts.iloc[i, j]
                if not np.isnan(acc_val):
                    annot.iloc[i, j] = f"{acc_val:.2f}\nn={int(count_val)}"
                else:
                    annot.iloc[i, j] = "N/A"
            except:
                annot.iloc[i, j] = "N/A"

    # Plot heatmap
    sns.heatmap(pivot, annot=annot.values, fmt='', cmap="RdYlGn", center=0.6,
               vmin=0.45, vmax=0.85, ax=ax, cbar_kws={"label": "Accuracy"},
               annot_kws={"fontsize": TICKSIZE - 1})

    ax.set_xlabel("Rating Std Dev $\\sigma_r$ (Reviewer Disagreement)", fontsize=LABELSIZE)
    ax.set_ylabel("Percentile Rating (pct\\_rating)", fontsize=LABELSIZE)
    ax.set_title("Accuracy by Rating Position and Reviewer Disagreement\n" +
                r"$\text{Acc}(S) = |\{x \in S : \hat{y}_x = y_x\}| / |S|$",
                fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_borderline_analysis(df: pd.DataFrame, output_path: Path):
    """Deep dive into borderline papers (pct_rating 0.4-0.6)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    borderline = df[(df["pct_rating"] >= 0.4) & (df["pct_rating"] <= 0.6)]
    non_borderline = df[(df["pct_rating"] < 0.4) | (df["pct_rating"] > 0.6)]

    # Left: Comparison
    ax1 = axes[0]

    borderline_stats = compute_accuracy_with_ci(borderline["correct"])
    non_borderline_stats = compute_accuracy_with_ci(non_borderline["correct"])

    categories = ["Borderline\n$0.4 \\leq $ pct $\\leq 0.6$",
                  "Clear Cases\npct $< 0.4$ or $> 0.6$"]
    accs = [borderline_stats["accuracy"], non_borderline_stats["accuracy"]]
    ci_lows = [borderline_stats["ci_low"], non_borderline_stats["ci_low"]]
    ci_highs = [borderline_stats["ci_high"], non_borderline_stats["ci_high"]]
    ns = [borderline_stats["n"], non_borderline_stats["n"]]

    colors = [REJECT_COLOR, ACCEPT_COLOR]
    x = np.arange(2)
    bars = ax1.bar(x, accs, color=colors, alpha=0.7, width=0.5)

    yerr_lower = [accs[i] - ci_lows[i] for i in range(2)]
    yerr_upper = [ci_highs[i] - accs[i] for i in range(2)]
    ax1.errorbar(x, accs, yerr=[yerr_lower, yerr_upper],
                fmt='none', color='black', capsize=8, capthick=2)

    for i, (acc, ci_l, ci_h, n) in enumerate(zip(accs, ci_lows, ci_highs, ns)):
        ax1.text(i, ci_h + 0.02, f"{acc:.1%}\n[{ci_l:.1%}-{ci_h:.1%}]\n(n={n})",
                ha='center', fontsize=LABELSIZE, va='bottom')

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=LABELSIZE)
    ax1.set_ylabel("Accuracy", fontsize=LABELSIZE)
    ax1.set_title("Borderline vs Clear Cases", fontsize=TITLESIZE)
    ax1.tick_params(axis='both', labelsize=TICKSIZE)
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax1.set_ylim(0.4, 0.95)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Statistical test
    contingency = pd.DataFrame({
        'borderline': [borderline["correct"].sum(), len(borderline) - borderline["correct"].sum()],
        'clear': [non_borderline["correct"].sum(), len(non_borderline) - non_borderline["correct"].sum()]
    }, index=['correct', 'incorrect'])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    ax1.text(0.5, 0.02, f"$\\chi^2$={chi2:.2f}, $p$={p_value:.4f}",
            transform=ax1.transAxes, ha='center', fontsize=TICKSIZE)

    # Right: Breakdown within borderline by label
    ax2 = axes[1]

    borderline_accept = borderline[borderline["label"] == 1]
    borderline_reject = borderline[borderline["label"] == 0]

    accept_stats = compute_accuracy_with_ci(borderline_accept["correct"])
    reject_stats = compute_accuracy_with_ci(borderline_reject["correct"])

    categories2 = ["Borderline\nAccept (GT)", "Borderline\nReject (GT)"]
    accs2 = [accept_stats["accuracy"], reject_stats["accuracy"]]
    ci_lows2 = [accept_stats["ci_low"], reject_stats["ci_low"]]
    ci_highs2 = [accept_stats["ci_high"], reject_stats["ci_high"]]
    ns2 = [accept_stats["n"], reject_stats["n"]]

    colors2 = [ACCEPT_COLOR, REJECT_COLOR]
    bars2 = ax2.bar(x, accs2, color=colors2, alpha=0.7, width=0.5)

    yerr_lower2 = [accs2[i] - ci_lows2[i] for i in range(2)]
    yerr_upper2 = [ci_highs2[i] - accs2[i] for i in range(2)]
    ax2.errorbar(x, accs2, yerr=[yerr_lower2, yerr_upper2],
                fmt='none', color='black', capsize=8, capthick=2)

    for i, (acc, ci_l, ci_h, n) in enumerate(zip(accs2, ci_lows2, ci_highs2, ns2)):
        ax2.text(i, ci_h + 0.02, f"{acc:.1%}\n[{ci_l:.1%}-{ci_h:.1%}]\n(n={n})",
                ha='center', fontsize=LABELSIZE, va='bottom')

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories2, fontsize=LABELSIZE)
    ax2.set_ylabel("Recall", fontsize=LABELSIZE)
    ax2.set_title("Within Borderline: Accept vs Reject Recall", fontsize=TITLESIZE)
    ax2.tick_params(axis='both', labelsize=TICKSIZE)
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax2.set_ylim(0.4, 0.95)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_stratification_table(df: pd.DataFrame, output_path: Path):
    """Generate LaTeX table of stratification statistics."""
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Model Accuracy by Paper Stratum}")
    latex.append(r"\label{tab:stratification}")
    latex.append(r"\footnotesize")
    latex.append(r"\begin{tabular}{l|r|r|c}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Stratum Definition} & \textbf{n} & \textbf{Accuracy} & \textbf{95\% CI} \\")
    latex.append(r"\hline")

    # Define strata
    strata = [
        ("Overall", df["correct"]),
        ("Low disagreement ($\\sigma_r < 0.5$)", df[df["rating_std"] < 0.5]["correct"]),
        ("High disagreement ($\\sigma_r \\geq 1.5$)", df[df["rating_std"] >= 1.5]["correct"]),
        ("Clear reject (pct $< 0.2$)", df[df["pct_rating"] < 0.2]["correct"]),
        ("Borderline ($0.4 \\leq$ pct $\\leq 0.6$)", df[(df["pct_rating"] >= 0.4) & (df["pct_rating"] <= 0.6)]["correct"]),
        ("Clear accept (pct $> 0.8$)", df[df["pct_rating"] > 0.8]["correct"]),
    ]

    for name, data in strata:
        stats_dict = compute_accuracy_with_ci(data)
        latex.append(f"{name} & {stats_dict['n']} & {stats_dict['accuracy']:.1%} & "
                    f"[{stats_dict['ci_low']:.1%}-{stats_dict['ci_high']:.1%}] \\\\")

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
    pred_path = RESULTS_DIR / "iclr20_balanced_clean" / "finetuned.jsonl"
    dataset_name = "iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6"

    # Load data
    print("Loading predictions...")
    if pred_path.exists():
        preds = load_predictions(pred_path)
        print(f"Loaded {len(preds)} predictions")
    else:
        print("Predictions not found, using simulated data")
        np.random.seed(42)
        n = 1000
        preds = [{"pred": np.random.choice([0, 1]), "label": np.random.choice([0, 1]),
                 "correct": np.random.random() < 0.66} for _ in range(n)]

    print("Loading metadata...")
    metadata_df = load_metadata(DATA_DIR, dataset_name, "test")

    if len(metadata_df) == 0:
        print("Metadata not found, using simulated data")
        n = len(preds)
        metadata_df = pd.DataFrame({
            "tokens": np.random.normal(5000, 1500, n).astype(int),
            "rating_std": np.random.exponential(0.8, n),
            "pct_rating": np.random.beta(2, 2, n),
        })

    # Join data
    print("Joining data...")
    df = join_predictions(preds, metadata_df)
    print(f"Joined {len(df)} entries")

    # Generate figures
    print("\nGenerating figures...")

    # 1. Accuracy vs rating std with CI
    plot_acc_vs_rating_std_with_ci(df, OUTPUT_DIR / "acc_vs_rating_std_with_ci.pdf")

    # 2. Accuracy vs pct_rating violin
    plot_acc_vs_pct_rating_violin(df, OUTPUT_DIR / "acc_vs_pct_rating_violin.pdf")

    # 3. 2D heatmap
    plot_acc_heatmap_2d(df, OUTPUT_DIR / "acc_heatmap_2d.pdf")

    # 4. Borderline analysis
    plot_borderline_analysis(df, OUTPUT_DIR / "borderline_analysis.pdf")

    # 5. Stratification table
    generate_stratification_table(df, TABLE_DIR / "stratification_stats.tex")

    # Print summary
    print("\n" + "=" * 60)
    print("Performance Stratification Summary (with 95% CI)")
    print("=" * 60)

    overall_stats = compute_accuracy_with_ci(df["correct"])
    print(f"\nOverall: {overall_stats['accuracy']:.1%} "
          f"[{overall_stats['ci_low']:.1%}-{overall_stats['ci_high']:.1%}] (n={overall_stats['n']})")

    borderline = df[(df["pct_rating"] >= 0.4) & (df["pct_rating"] <= 0.6)]
    clear = df[(df["pct_rating"] < 0.4) | (df["pct_rating"] > 0.6)]

    borderline_stats = compute_accuracy_with_ci(borderline["correct"])
    clear_stats = compute_accuracy_with_ci(clear["correct"])

    print(f"\nBorderline (0.4-0.6): {borderline_stats['accuracy']:.1%} "
          f"[{borderline_stats['ci_low']:.1%}-{borderline_stats['ci_high']:.1%}] (n={borderline_stats['n']})")
    print(f"Clear cases: {clear_stats['accuracy']:.1%} "
          f"[{clear_stats['ci_low']:.1%}-{clear_stats['ci_high']:.1%}] (n={clear_stats['n']})")

    low_std = df[df["rating_std"] < 0.5]
    high_std = df[df["rating_std"] >= 1.5]

    low_std_stats = compute_accuracy_with_ci(low_std["correct"])
    high_std_stats = compute_accuracy_with_ci(high_std["correct"])

    print(f"\nLow disagreement (<0.5): {low_std_stats['accuracy']:.1%} "
          f"[{low_std_stats['ci_low']:.1%}-{low_std_stats['ci_high']:.1%}] (n={low_std_stats['n']})")
    print(f"High disagreement (>=1.5): {high_std_stats['accuracy']:.1%} "
          f"[{high_std_stats['ci_low']:.1%}-{high_std_stats['ci_high']:.1%}] (n={high_std_stats['n']})")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
