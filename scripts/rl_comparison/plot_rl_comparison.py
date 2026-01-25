#!/usr/bin/env python3
"""
RL ArXiv Comparison - Compare different model architectures with GRPO training.

Generates plots:
1. overall_accuracy.png - Bar chart of overall accuracy for 3 variants with eval
2. accuracy_by_year.png - 1x3 line plot (Accuracy, Accept Recall, Reject Recall by year)
3. training_reward.png - Training reward curves (first 50 steps)
4. training_entropy.png - Policy entropy curves
5. training_generation_length.png - Generation length over training steps

Generates tables:
1. metrics_table.csv - Comparison of metrics across methods
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

# ID/OOD year definitions
ID_YEARS = [2020, 2021, 2022, 2023, 2024]
OOD_YEARS = [2025]

# Configuration
VARIANTS = {
    "Qwen3-8B": {
        "eval_pattern": "qwen3_8b_step",
        "training_col": "arxiv_qwen3_8b_grpo_4033544",
    },
    "Qwen3-4B": {
        "eval_pattern": "qwen3_4b",
        "training_col": "arxiv_train_qwen3_4b_3_turns_1000gen_gs10_cosdecay_3816027",
    },
    "Qwen2.5-7B": {
        "eval_pattern": "qwen2.5_7b_step",
        "training_col": "arxiv_qwen2.5_7b_grpo_3964391_0",
    },
    "Qwen2.5-Bz256": {
        "eval_pattern": "qwen2.5_7b_bz256",
        "training_col": "arxiv_qwen2.5_7b_bz256_grpo_3991957_0",
    },
}

COLORS = {
    "Qwen3-8B": "#1f77b4",      # Blue
    "Qwen3-4B": "#ff7f0e",      # Orange
    "Qwen2.5-7B": "#2ca02c",    # Green
    "Qwen2.5-Bz256": "#d62728", # Red
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


def plot_accuracy_by_year(eval_data: dict) -> None:
    """1x4 line plot: Accuracy, Accept Recall, Reject Recall, Pred Accept Rate by year.

    Uses x markers for ID years, o markers for OOD years.
    - Qwen3 models: all years are ID (all o markers, filled)
    - Qwen2.5 models: 2020-2024 ID (x), 2025 OOD (o, hollow)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # For pred_accept_rate, we derive from accept_recall and reject_recall
    # Formula (assuming balanced data): pred_accept_rate = (accept_recall + 1 - reject_recall) / 2
    metrics = ['accuracy', 'accept_recall', 'reject_recall', 'pred_accept_rate']
    titles = ['Accuracy', 'Accept Recall', 'Reject Recall', 'Pred. Accept Rate']
    all_years = [2020, 2021, 2022, 2023, 2024, 2025]

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

    plt.savefig(OUTPUT_DIR / "accuracy_by_year.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'accuracy_by_year.png'}")


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

    # Plot 2: Accuracy by year (1x3 subplots)
    plot_accuracy_by_year(eval_data)

    # Plot 3: Training reward curves
    plot_training_reward(training_data)

    # Plot 4: Policy entropy curves
    plot_training_entropy(entropy_data)

    # Plot 5: Generation length curves
    plot_training_generation_length(gen_length_data)

    print("\nGenerating tables...")

    # Table 1: Metrics comparison table
    create_metrics_table(eval_data)

    print("\nAll outputs generated successfully!")


if __name__ == "__main__":
    main()
