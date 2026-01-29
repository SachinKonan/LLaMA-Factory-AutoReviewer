#!/usr/bin/env python3
"""
RL ArXiv Comparison - Compare different model architectures with GRPO training.

Generates plots:
1. overall_accuracy.png - Bar chart of overall accuracy for 3 variants with eval
2. accuracy_by_year.png - 1x3 line plot (Accuracy, Accept Recall, Reject Recall by year)
3. training_reward.png - Training reward curves (first 50 steps)
4. training_entropy.png - Policy entropy curves
5. training_generation_length.png - Generation length over training steps
6. accuracy_by_length.png - Accuracy by paper length (text tokens) per model

Generates tables:
1. metrics_table.csv - Comparison of metrics across methods
"""

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

# ID/OOD year definitions
ID_YEARS = [2020, 2021, 2022, 2023, 2024]
OOD_YEARS = [2025]

# Configuration
VARIANTS = {
    "Qwen3-8B": {
        "eval_pattern": "arxiv_qwen3_8b_step",
        "training_col": "arxiv_qwen3_8b_grpo_4033544",
    },
    "Qwen3-4B": {
        "eval_pattern": "arxiv_qwen3_4b_step50",
        "training_col": "arxiv_train_qwen3_4b_3_turns_1000gen_gs10_cosdecay_3816027",
    },
    "Qwen2.5-7B": {
        "eval_pattern": "arxiv_qwen2.5_7b_step45_ailab",
        "training_col": "arxiv_qwen2.5_7b_grpo_3964391_0",
    },
    "Qwen2.5-Bz256": {
        "eval_pattern": "arxiv_qwen2.5_7b_bz256",
        "training_col": "arxiv_qwen2.5_7b_bz256_grpo_3991957_0",
    },
    "Qwen3-4B-Balanced": {
        "eval_pattern": "qwen3_4b_balanced",
        "training_col": "arxiv_qwen3_4b_balanced_4139080",
    }
}

COLORS = {
    "Qwen3-8B": "#1f77b4",      # Blue
    "Qwen3-4B": "#ff7f0e",      # Orange
    "Qwen2.5-7B": "#2ca02c",    # Green
    "Qwen2.5-Bz256": "#d62728", # Red
    "Qwen3-4B-Balanced": "#9467bd", # Purple
}

OUTPUT_DIR = Path("results/summarized_investigation/rl_arxiv")
DATA_DIR = Path("results/_rl")

YEAR_COLUMNS = ["y2020", "y2021", "y2022", "y2023", "y2024", "y2025"]


def parse_metric(s: str) -> dict | None:
    """Parse 'acc/acc_recall/rej_recall(n=count)' format."""
    if pd.isna(s) or not s:
        return None
    match = re.match(r'([\d.]+)/([\d.]+)/([\d.]+)\(n=(\d+)\)', s)
    if match:
        return {
            'accuracy': float(match.group(1)),
            'accept_recall': float(match.group(2)),
            'reject_recall': float(match.group(3)),
            'n': int(match.group(4))
        }
    return None


def load_eval_data() -> dict:
    """Load eval CSV and map to variant names."""
    df = pd.read_csv(DATA_DIR / "arxiv_eval.csv")

    eval_data = {}
    for variant_name, config in VARIANTS.items():
        pattern = config["eval_pattern"]
        if pattern is None:
            continue

        # Find matching row
        for _, row in df.iterrows():
            if pattern in row["file"]:
                eval_data[variant_name] = {
                    "file": row["file"],
                    "mv_combined": parse_metric(row["mv_combined"]),
                    "years": {}
                }
                for year_col in YEAR_COLUMNS:
                    eval_data[variant_name]["years"][year_col] = parse_metric(row[year_col])
                break

    return eval_data


def load_training_data() -> pd.DataFrame:
    """Load training reward CSV for all 4 variants."""
    df = pd.read_csv(DATA_DIR / "arxiv_training.csv")

    # Rename columns to variant names
    rename_map = {"Step": "step"}
    for variant_name, config in VARIANTS.items():
        col_prefix = config["training_col"]
        reward_col = f"{col_prefix} - reward/avg_raw_reward"
        if reward_col in df.columns:
            rename_map[reward_col] = variant_name

    df = df.rename(columns=rename_map)

    # Keep only step and variant columns
    cols_to_keep = ["step"] + [v for v in VARIANTS.keys() if v in df.columns]
    df = df[cols_to_keep]

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_entropy_data() -> pd.DataFrame:
    """Load entropy data from the entropy file."""
    df = pd.read_csv(DATA_DIR / "arxiv_entropy")

    # Rename columns to variant names
    rename_map = {"Step": "step"}
    for variant_name, config in VARIANTS.items():
        col_prefix = config["training_col"]
        entropy_col = f"{col_prefix} - policy/policy_entropy"
        if entropy_col in df.columns:
            rename_map[entropy_col] = variant_name

    df = df.rename(columns=rename_map)

    # Keep only step and variant columns
    cols_to_keep = ["step"] + [v for v in VARIANTS.keys() if v in df.columns]
    df = df[cols_to_keep]

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_generation_length_data() -> pd.DataFrame:
    """Load generation length (avg_num_tokens) data."""
    df = pd.read_csv(DATA_DIR / "arxiv_avg_num_tokens.csv")

    # Rename columns to variant names
    rename_map = {"Step": "step"}
    for variant_name, config in VARIANTS.items():
        col_prefix = config["training_col"]
        tokens_col = f"{col_prefix} - generate/avg_num_tokens"
        if tokens_col in df.columns:
            rename_map[tokens_col] = variant_name

    df = df.rename(columns=rename_map)

    # Keep only step and variant columns
    cols_to_keep = ["step"] + [v for v in VARIANTS.keys() if v in df.columns]
    df = df[cols_to_keep]

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def plot_overall_accuracy(eval_data: dict) -> None:
    """Bar chart of overall accuracy for all variants (empty for those without eval)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    all_variants = list(VARIANTS.keys())
    x_positions = range(len(all_variants))

    # Get accuracies (None for those without eval)
    accuracies = []
    for v in all_variants:
        if v in eval_data and eval_data[v]["mv_combined"]:
            accuracies.append(eval_data[v]["mv_combined"]["accuracy"])
        else:
            accuracies.append(None)

    # Plot bars only for variants with eval
    max_acc = 0
    for i, (variant, acc) in enumerate(zip(all_variants, accuracies)):
        if acc is not None:
            bar = ax.bar(i, acc, color=COLORS[variant], edgecolor='black', linewidth=1.2)
            ax.text(i, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
            max_acc = max(max_acc, acc)
        else:
            # Draw empty bar outline for missing eval
            ax.bar(i, 0, color='none', edgecolor='gray', linewidth=1.5, linestyle='--')
            ax.text(i, 0.02, 'N/A', ha='center', va='bottom',
                   fontsize=10, color='gray', style='italic')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_variants, fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Model Variant', fontsize=12)
    ax.set_title('Overall Accuracy by Model Variant (mv_combined)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max_acc * 1.15 if max_acc > 0 else 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overall_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'overall_accuracy.png'}")


def plot_accuracy_by_year(eval_data: dict, variant_filter: list[str] | None = None, output_suffix: str = "") -> None:
    """1x4 line plot: Accuracy, Accept Recall, Reject Recall, Pred Accept Rate by year.

    Uses x markers for ID years, o markers for OOD years.
    - Qwen3 models: all years are ID (all o markers, filled)
    - Qwen2.5 models: 2020-2024 ID (x), 2025 OOD (o, hollow)

    Args:
        eval_data: Evaluation data dictionary
        variant_filter: Optional list of variant names to include (if None, includes all)
        output_suffix: Suffix to add to output filename (e.g., "_qwen4b" -> "accuracy_by_year_qwen4b.png")
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # For pred_accept_rate, we derive from accept_recall and reject_recall
    # Formula (assuming balanced data): pred_accept_rate = (accept_recall + 1 - reject_recall) / 2
    metrics = ['accuracy', 'accept_recall', 'reject_recall', 'pred_accept_rate']
    titles = ['Accuracy', 'Accept Recall', 'Reject Recall', 'Pred. Accept Rate']
    all_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Filter variants if specified
    if variant_filter:
        variants_with_eval = [v for v in variant_filter if v in eval_data]
    else:
        variants_with_eval = [v for v in VARIANTS.keys() if v in eval_data]

    for ax, metric, title in zip(axes, metrics, titles):
        for variant in variants_with_eval:
            years = []
            values = []
            for year_col in YEAR_COLUMNS:
                year_data = eval_data[variant]["years"].get(year_col)
                year = int(year_col[1:])  # Extract year from "y2020" etc.
                if year_data:
                    years.append(year)
                    if metric == 'pred_accept_rate':
                        # Derive from accept_recall and reject_recall
                        acc_rec = year_data['accept_recall']
                        rej_rec = year_data['reject_recall']
                        values.append((acc_rec + 1 - rej_rec) / 2)
                    else:
                        values.append(year_data[metric])

            if not years:
                continue

            color = COLORS[variant]

            # Plot line
            ax.plot(years, values, '-', color=color, linewidth=2, label=variant)

            # Determine if Qwen3 model (all years ID) or Qwen2.5 (2025 is OOD)
            is_qwen3 = 'Qwen3' in variant

            if is_qwen3:
                # All years are ID for Qwen3 - use filled circles
                ax.scatter(years, values, marker='o', s=80, color=color, zorder=5)
            else:
                # Qwen2.5: ID years (x), OOD years (hollow o)
                id_years = [y for y in years if y in ID_YEARS]
                id_vals = [values[years.index(y)] for y in id_years]
                if id_years:
                    ax.scatter(id_years, id_vals, marker='x', s=80, color=color, zorder=5)

                ood_years = [y for y in years if y in OOD_YEARS]
                ood_vals = [values[years.index(y)] for y in ood_years]
                if ood_years:
                    ax.scatter(ood_years, ood_vals, marker='o', s=80, color=color, zorder=5,
                              facecolors='white', edgecolors=color, linewidths=2)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} by Year', fontsize=14, fontweight='bold')
        ax.set_xticks(all_years)
        ax.set_xticklabels([str(y) for y in all_years])
        ax.set_ylim(0.2, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.3)

    # Single legend for all subplots at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(variants_with_eval),
               fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_name = f"accuracy_by_year{output_suffix}.png"
    plt.savefig(OUTPUT_DIR / output_name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / output_name}")


def plot_training_reward(training_data: pd.DataFrame) -> None:
    """Training reward curves for first 50 steps."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to first 50 steps
    df = training_data[training_data['step'] <= 50].copy()

    for variant in VARIANTS.keys():
        if variant in df.columns:
            valid_data = df[['step', variant]].dropna()
            if not valid_data.empty:
                ax.plot(valid_data['step'], valid_data[variant],
                       label=variant, color=COLORS[variant], linewidth=2)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Average Raw Reward', fontsize=12)
    ax.set_title('Training Reward Curves (GRPO)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, 50)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_reward.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'training_reward.png'}")


def plot_training_entropy(entropy_data: pd.DataFrame) -> None:
    """Policy entropy curves showing collapse over training."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to first 50 steps
    df = entropy_data[entropy_data['step'] <= 50].copy()

    for variant in VARIANTS.keys():
        if variant in df.columns:
            valid_data = df[['step', variant]].dropna()
            if not valid_data.empty:
                ax.plot(valid_data['step'], valid_data[variant],
                       label=variant, color=COLORS[variant], linewidth=2)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Policy Entropy', fontsize=12)
    ax.set_title('Policy Entropy During Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, 50)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_entropy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'training_entropy.png'}")


def plot_training_generation_length(gen_length_data: pd.DataFrame) -> None:
    """Generation length curves over training steps."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to first 50 steps
    df = gen_length_data[gen_length_data['step'] <= 50].copy()

    for variant in VARIANTS.keys():
        if variant in df.columns:
            valid_data = df[['step', variant]].dropna()
            if not valid_data.empty:
                ax.plot(valid_data['step'], valid_data[variant],
                       label=variant, color=COLORS[variant], linewidth=2)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Average Number of Tokens', fontsize=12)
    ax.set_title('Generation Length During Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, 50)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_generation_length.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'training_generation_length.png'}")


def normalize_binary(answer: str | None) -> str | None:
    """Normalize answer to 'accepted' or 'rejected'."""
    if answer is None:
        return None
    answer = answer.lower().strip()
    if "accept" in answer:
        return "accepted"
    if "reject" in answer:
        return "rejected"
    return None


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    if not text:
        return None
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1)
    return None


def get_majority_vote(predictions: list[str], tie_breaker: str = "rejected") -> str | None:
    """Get majority vote from list of predictions."""
    votes = [normalize_binary(extract_boxed_answer(p)) for p in predictions]
    votes = [v for v in votes if v is not None]
    if not votes:
        return None
    counts = Counter(votes)
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]
    if len(winners) > 1:
        return tie_breaker
    return winners[0]


def load_test_metadata() -> pd.DataFrame:
    """Load test data metadata with submission_id, year, and ground truth."""
    test_data_path = Path("data/iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6_test/data.json")
    with open(test_data_path) as f:
        test_data = json.load(f)

    rows = []
    for i, item in enumerate(test_data):
        meta = item.get("_metadata", {})
        rows.append({
            "index": i,
            "submission_id": meta.get("submission_id"),
            "year": meta.get("year"),
            "ground_truth": normalize_binary(meta.get("answer")),
        })
    return pd.DataFrame(rows)


def load_paper_lengths() -> pd.DataFrame:
    """Load paper lengths from massive_metadata.csv."""
    metadata_path = Path("data/massive_metadata.csv")
    df = pd.read_csv(metadata_path, usecols=["submission_id", "num_text_tokens"])
    return df


def load_predictions_for_variant(variant_name: str, config: dict) -> pd.DataFrame | None:
    """Load predictions for a variant and compute correctness."""
    pattern = config["eval_pattern"]
    if pattern is None:
        return None

    # Find the prediction file
    results_dir = Path("/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/results")
    pred_file = None
    for f in results_dir.glob("*.jsonl"):
        if pattern in f.name:
            pred_file = f
            break

    if pred_file is None:
        return None

    # Load predictions
    predictions = []
    with open(pred_file) as f:
        for line in f:
            item = json.loads(line)
            preds = item.get("predict", [])
            if isinstance(preds, list):
                mv = get_majority_vote(preds)
            else:
                mv = normalize_binary(extract_boxed_answer(preds))
            predictions.append(mv)

    return pd.DataFrame({"index": range(len(predictions)), "prediction": predictions})


def plot_accuracy_by_length(variants_config: dict, colors: dict) -> None:
    """Plot binned accuracy vs paper length with regression line and correlation."""
    from scipy import stats

    print("Loading test metadata and paper lengths...")
    test_meta = load_test_metadata()
    paper_lengths = load_paper_lengths()

    # Merge test metadata with paper lengths
    df = test_meta.merge(paper_lengths, on="submission_id", how="left")

    # Load predictions for each variant (using embedded labels for correct alignment)
    variant_predictions = {}
    for variant_name, config in variants_config.items():
        preds = load_predictions_with_labels(variant_name, config)
        if preds is not None:
            preds = preds.merge(paper_lengths, on="submission_id", how="left")
            variant_predictions[variant_name] = preds

    if not variant_predictions:
        print("No predictions found for any variant, skipping accuracy_by_length plot")
        return

    years = sorted(df["year"].dropna().unique().astype(int))
    n_years = len(years)
    n_models = len(variant_predictions)
    model_names = list(variant_predictions.keys())

    # Define length bins
    bin_size = 2000
    bin_edges = np.arange(0, 30000, bin_size)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # nrows = years, ncols = models
    fig, axes = plt.subplots(n_years, n_models, figsize=(4.5 * n_models, 3.5 * n_years))
    if n_years == 1:
        axes = axes.reshape(1, -1)
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for y_idx, year in enumerate(years):
        for m_idx, variant_name in enumerate(model_names):
            ax = axes[y_idx, m_idx]
            preds_df = variant_predictions[variant_name]
            color = colors.get(variant_name, "#333333")

            # Filter to this year
            year_data = preds_df[preds_df["year"] == year].copy()
            year_data = year_data.dropna(subset=["num_text_tokens", "prediction", "ground_truth"])

            if len(year_data) < 10:
                ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
                continue

            # Compute correctness
            year_data["correct"] = (year_data["prediction"] == year_data["ground_truth"]).astype(int)

            # Bin by length and compute accuracy per bin
            year_data["length_bin"] = pd.cut(year_data["num_text_tokens"], bins=bin_edges, labels=False)
            bin_stats = year_data.groupby("length_bin").agg(
                accuracy=("correct", "mean"),
                count=("correct", "count")
            ).reset_index()

            # Get valid bins with enough samples
            bin_stats = bin_stats[bin_stats["count"] >= 5]
            if len(bin_stats) < 3:
                ax.text(0.5, 0.5, "Insufficient bins", ha='center', va='center', transform=ax.transAxes)
                continue

            x = bin_centers[bin_stats["length_bin"].astype(int)] / 1000  # k tokens
            y = bin_stats["accuracy"].values
            sizes = bin_stats["count"].values

            # Scatter plot (size proportional to count)
            ax.scatter(x, y, s=sizes * 3, alpha=0.7, color=color, edgecolor='white', linewidth=0.5)

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_line = np.linspace(0, 25, 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color='black', linewidth=2, linestyle='-',
                   label=f'r = {r_value:.3f}')

            # Mean paper length as vertical dotted line
            mean_tokens = year_data["num_text_tokens"].mean() / 1000
            ax.axvline(x=mean_tokens, color='gray', linestyle=':', linewidth=2,
                      label=f'Mean = {mean_tokens:.1f}k')

            # Labels
            ax.set_xlabel("Length (k tokens)", fontsize=10)
            if m_idx == 0:
                ax.set_ylabel(f"{year}\nAccuracy", fontsize=11, fontweight="bold")
            if y_idx == 0:
                ax.set_title(f"{variant_name}", fontsize=12, fontweight="bold")

            ax.set_ylim(0.3, 0.85)
            ax.set_xlim(0, 25)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=0.3)

            # Legend on each subplot
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.suptitle("Accuracy vs Paper Length (Binned, with Linear Regression)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_by_length.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'accuracy_by_length.png'}")


def load_train_metadata() -> pd.DataFrame:
    """Load train data metadata with submission_id and ground truth."""
    train_data_path = Path("data/iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6_train/data.json")
    with open(train_data_path) as f:
        train_data = json.load(f)

    rows = []
    for i, item in enumerate(train_data):
        meta = item.get("_metadata", {})
        rows.append({
            "index": i,
            "submission_id": meta.get("submission_id"),
            "year": meta.get("year"),
            "ground_truth": normalize_binary(meta.get("answer")),
        })
    return pd.DataFrame(rows)


def load_predictions_with_labels(variant_name: str, config: dict) -> pd.DataFrame | None:
    """Load predictions using embedded labels (not index alignment)."""
    pattern = config["eval_pattern"]
    if pattern is None:
        return None

    # Find the prediction file
    results_dir = Path("/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/results")
    pred_file = None
    for f in results_dir.glob("*.jsonl"):
        if pattern in f.name:
            pred_file = f
            break

    if pred_file is None:
        return None

    # Load predictions with labels
    rows = []
    with open(pred_file) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            label = item.get("label", "")
            preds = item.get("predict", [])
            metadata = item.get("_metadata", {})

            gt = normalize_binary(extract_boxed_answer(label))
            if isinstance(preds, list):
                pred = get_majority_vote(preds)
            else:
                pred = normalize_binary(extract_boxed_answer(preds))

            rows.append({
                "index": i,
                "prediction": pred,
                "ground_truth": gt,
                "submission_id": metadata.get("submission_id"),
                "year": metadata.get("year"),
            })

    return pd.DataFrame(rows)


def plot_prediction_analysis(variants_config: dict, colors: dict) -> None:
    """Plot 2 x years: top=accuracy, bottom=pred acceptance rate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("Loading train/test metadata and paper lengths for accuracy analysis...")

    # Load train and test metadata
    train_meta = load_train_metadata()
    test_meta = load_test_metadata()
    paper_lengths = load_paper_lengths()

    # Merge with paper lengths
    train_df = train_meta.merge(paper_lengths, on="submission_id", how="left")
    test_df = test_meta.merge(paper_lengths, on="submission_id", how="left")

    train_df["gt_binary"] = (train_df["ground_truth"] == "accepted").astype(int)
    test_df["gt_binary"] = (test_df["ground_truth"] == "accepted").astype(int)

    # Train logistic regression on train set (tokens -> label)
    train_df_clean = train_df.dropna(subset=["num_text_tokens", "gt_binary"])
    test_df_clean = test_df.dropna(subset=["num_text_tokens", "gt_binary"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df_clean[["num_text_tokens"]])
    y_train = train_df_clean["gt_binary"]

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)

    # Load predictions for each variant (using embedded labels)
    variant_data = {}
    for variant_name, config in variants_config.items():
        preds_df = load_predictions_with_labels(variant_name, config)
        if preds_df is not None and len(preds_df) > 0:
            preds_df = preds_df.merge(paper_lengths, on="submission_id", how="left")
            variant_data[variant_name] = preds_df
            print(f"  Loaded {variant_name}: {len(preds_df)} samples")

    if not variant_data:
        print("No predictions found, skipping prediction analysis plot")
        return

    years = sorted(test_df["year"].dropna().unique().astype(int))
    n_years = len(years)
    model_names = list(variant_data.keys())
    n_models = len(model_names)

    # Create figure: 2 rows x n_years columns
    fig, axes = plt.subplots(2, n_years, figsize=(4 * n_years, 8), sharey='row')

    bar_width = 0.8 / (n_models + 1)  # +1 for LR baseline

    # Define colors for each model
    model_colors = ["#7f7f7f"]  # Gray for LR baseline
    for name in model_names:
        model_colors.append(colors.get(name, "#333333"))

    for y_idx, year in enumerate(years):
        ax_acc = axes[0, y_idx]  # Top row: accuracy
        ax_rate = axes[1, y_idx]  # Bottom row: pred acceptance rate

        # Get test data for this year
        year_test = test_df_clean[test_df_clean["year"] == year].copy()
        if len(year_test) == 0:
            continue

        # LR baseline for this year
        X_year = scaler.transform(year_test[["num_text_tokens"]])
        lr_preds = lr_model.predict(X_year)
        lr_accuracy = (lr_preds == year_test["gt_binary"]).mean()
        lr_accept_rate = lr_preds.mean()

        # Collect all accuracies and acceptance rates
        all_names = ["LR(Tokens)"] + model_names
        all_accs = [lr_accuracy]
        all_accept_rates = [lr_accept_rate]

        for variant_name in model_names:
            preds_df = variant_data[variant_name]
            year_preds = preds_df[preds_df["year"] == year].copy()
            year_preds = year_preds.dropna(subset=["prediction", "ground_truth"])

            if len(year_preds) == 0:
                all_accs.append(0)
                all_accept_rates.append(0)
                continue

            # Model accuracy
            year_preds["correct"] = year_preds["prediction"] == year_preds["ground_truth"]
            model_acc = year_preds["correct"].mean()
            all_accs.append(model_acc)

            # Pred acceptance rate
            pred_accept_rate = (year_preds["prediction"] == "accepted").mean()
            all_accept_rates.append(pred_accept_rate)

        # Plot accuracy bars (top row)
        x = np.arange(len(all_names))
        bars_acc = ax_acc.bar(x, all_accs, width=0.7, color=model_colors, alpha=0.8, edgecolor='white')

        # Add values above bars
        for bar, acc in zip(bars_acc, all_accs):
            ax_acc.annotate(f'{acc:.0%}',
                           xy=(bar.get_x() + bar.get_width() / 2, acc),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax_acc.set_ylabel("Accuracy" if y_idx == 0 else "", fontsize=11)
        ax_acc.set_title(f"{year} (n={len(year_test)})", fontsize=12, fontweight="bold")
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(all_names, fontsize=8, rotation=45, ha='right')
        ax_acc.set_ylim(0.4, 0.85)
        ax_acc.yaxis.set_major_formatter(PercentFormatter(1))
        ax_acc.grid(axis="y", alpha=0.3)

        # Plot acceptance rate bars (bottom row)
        bars_rate = ax_rate.bar(x, all_accept_rates, width=0.7, color=model_colors, alpha=0.8, edgecolor='white')

        # Add values above bars
        for bar, rate in zip(bars_rate, all_accept_rates):
            ax_rate.annotate(f'{rate:.0%}',
                            xy=(bar.get_x() + bar.get_width() / 2, rate),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Add ground truth acceptance rate line
        gt_accept_rate = year_test["gt_binary"].mean()
        ax_rate.axhline(y=gt_accept_rate, color='red', linestyle='--', linewidth=2,
                       label=f'GT Rate: {gt_accept_rate:.0%}')

        ax_rate.set_ylabel("Pred Accept Rate" if y_idx == 0 else "", fontsize=11)
        ax_rate.set_xlabel("Method", fontsize=10)
        ax_rate.set_xticks(x)
        ax_rate.set_xticklabels(all_names, fontsize=8, rotation=45, ha='right')
        ax_rate.set_ylim(0, 0.9)  # Start from 0 to show all bars
        ax_rate.yaxis.set_major_formatter(PercentFormatter(1))
        ax_rate.grid(axis="y", alpha=0.3)
        if y_idx == 0:
            ax_rate.legend(loc='upper right', fontsize=9)

    # Add row labels
    fig.text(0.02, 0.72, "Accuracy", fontsize=14, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.28, "Pred Accept\nRate", fontsize=14, fontweight='bold', rotation=90, va='center')

    plt.suptitle("Model Accuracy & Predicted Acceptance Rate by Year\n(LR baseline trained on train set tokens)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    plt.savefig(OUTPUT_DIR / "prediction_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'prediction_analysis.png'}")


def create_metrics_table(eval_data: dict) -> None:
    """Create metrics table CSV with all variants (empty cells for those without eval)."""
    rows = []

    for variant in VARIANTS.keys():
        if variant in eval_data:
            data = eval_data[variant]
            mv = data["mv_combined"]

            # Calculate ID accuracy (average of y2020-y2024)
            id_years = ["y2020", "y2021", "y2022", "y2023", "y2024"]
            id_accuracies = []
            for y in id_years:
                year_data = data["years"].get(y)
                if year_data:
                    id_accuracies.append(year_data["accuracy"])
            id_acc = sum(id_accuracies) / len(id_accuracies) if id_accuracies else None

            # OOD accuracy (y2025)
            ood_data = data["years"].get("y2025")
            ood_acc = ood_data["accuracy"] if ood_data else None

            rows.append({
                "Method": variant,
                "Overall Accuracy": f"{mv['accuracy']:.0%}" if mv else "",
                "ID Accuracy (2020-2024)": f"{id_acc:.0%}" if id_acc else "",
                "OOD Accuracy (2025)": f"{ood_acc:.0%}" if ood_acc else "",
                "Accept Recall": f"{mv['accept_recall']:.0%}" if mv else "",
                "Reject Recall": f"{mv['reject_recall']:.0%}" if mv else "",
            })
        else:
            # No eval data - leave empty
            rows.append({
                "Method": variant,
                "Overall Accuracy": "",
                "ID Accuracy (2020-2024)": "",
                "OOD Accuracy (2025)": "",
                "Accept Recall": "",
                "Reject Recall": "",
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "metrics_table.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'metrics_table.csv'}")

    # Also print to console
    print("\nMetrics Table:")
    print(df.to_string(index=False))


def main():
    """Generate all plots and tables."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    eval_data = load_eval_data()
    training_data = load_training_data()
    entropy_data = load_entropy_data()
    gen_length_data = load_generation_length_data()

    print(f"Found eval data for: {list(eval_data.keys())}")
    print(f"Training data columns: {[c for c in training_data.columns if c != 'step']}")
    print(f"Entropy data columns: {[c for c in entropy_data.columns if c != 'step']}")
    print(f"Generation length columns: {[c for c in gen_length_data.columns if c != 'step']}")

    print("\nGenerating plots...")

    # Plot 1: Overall accuracy bar chart
    plot_overall_accuracy(eval_data)

    # Plot 2a: Accuracy by year - Qwen3-4B vs Qwen3-4B-Balanced comparison
    plot_accuracy_by_year(eval_data, variant_filter=["Qwen3-4B", "Qwen3-4B-Balanced"], output_suffix="_qwen4b")

    # Plot 2b: Accuracy by year - all models
    plot_accuracy_by_year(eval_data)

    # Plot 3: Training reward curves
    plot_training_reward(training_data)

    # Plot 4: Policy entropy curves
    plot_training_entropy(entropy_data)

    # Plot 5: Generation length curves
    plot_training_generation_length(gen_length_data)

    # Plot 6: Accuracy by paper length
    plot_accuracy_by_length(VARIANTS, COLORS)

    # Plot 7: Prediction analysis (R² comparison)
    plot_prediction_analysis(VARIANTS, COLORS)

    print("\nGenerating tables...")

    # Table 1: Metrics comparison table
    create_metrics_table(eval_data)

    print("\nAll outputs generated successfully!")


if __name__ == "__main__":
    main()
