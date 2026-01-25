#!/usr/bin/env python3
"""
Generate visualizations for long-context modality investigation:
What is the optimal ordering of paper representations for long-context models?

Compares two orderings:
- Text + Vision: Full paper text followed by page images
- Vision + Text: Page images followed by full paper text
"""

import json
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# Reference dataset for year lookups (v6 has year metadata)
YEAR_REFERENCE_DATA = "data/iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6_test/data.json"

# Configuration
VARIANTS = {
    "Text + Vision": {
        "results": "results/data_sweep_long_context/balanced_clean_vision/finetuned.jsonl",
        "test_data": "data/iclr_2020_2025_85_5_10_split6_balanced_clean_vision_binary_noreviews_v6_test/data.json",
        "train_data": "data/iclr_2020_2025_85_5_10_split6_balanced_clean_vision_binary_noreviews_v6_train/data.json",
        "train_log": "saves/data_sweep_long_context/balanced_clean_vision/trainer_state.json",
    },
    "Vision + Text": {
        "results": "results/data_sweep_long_context/balanced_vision_clean/finetuned.jsonl",
        "test_data": "data/iclr_2020_2025_85_5_10_split6_balanced_vision_clean_binary_noreviews_v6_test/data.json",
        "train_data": "data/iclr_2020_2025_85_5_10_split6_balanced_vision_clean_binary_noreviews_v6_train/data.json",
        "train_log": "saves/data_sweep_long_context/balanced_vision_clean/trainer_state.json",
    },
}
OUTPUT_DIR = Path("results/summarized_investigation/long_context_modality")
ID_YEARS = [2020, 2021, 2022, 2023, 2024]
OOD_YEARS = [2025]

# Colors for variants
COLORS = {
    "Text + Vision": "#1f77b4",   # Blue
    "Vision + Text": "#ff7f0e",   # Orange
}


def extract_boxed_answer(text: str) -> str:
    """Parse \\boxed{} format to extract Accept/Reject."""
    if text is None:
        return None
    # Match \boxed{Accept} or \boxed{Reject}
    match = re.search(r'\\boxed\{(\w+)\}', text)
    if match:
        return match.group(1)
    # Fallback: check if Accept or Reject appears
    if 'Accept' in text:
        return 'Accept'
    if 'Reject' in text:
        return 'Reject'
    return None


def normalize_label(label: str) -> str:
    """Convert to accepted/rejected format."""
    if label is None:
        return None
    label = label.lower().strip()
    if label in ['accept', 'accepted']:
        return 'accepted'
    if label in ['reject', 'rejected']:
        return 'rejected'
    return label


def load_year_reference() -> dict:
    """Load submission_id -> year mapping from reference dataset."""
    base_dir = Path(__file__).parent.parent.parent
    ref_file = base_dir / YEAR_REFERENCE_DATA
    with open(ref_file, 'r') as f:
        ref_data = json.load(f)

    id_to_year = {}
    for item in ref_data:
        meta = item.get('_metadata', {})
        sid = meta.get('submission_id')
        year = meta.get('year')
        if sid and year:
            id_to_year[sid] = year
    return id_to_year


def load_predictions_with_metadata(results_path: str, test_data_path: str, use_year_reference: bool = False) -> pd.DataFrame:
    """Join predictions with test data metadata."""
    base_dir = Path(__file__).parent.parent.parent

    # Load year reference if needed
    year_lookup = load_year_reference() if use_year_reference else {}

    # Load test data to get metadata (year, ground truth answer)
    test_data_file = base_dir / test_data_path
    with open(test_data_file, 'r') as f:
        test_data = json.load(f)

    # Extract metadata from test data
    metadata_list = []
    for i, item in enumerate(test_data):
        meta = item.get('_metadata', {})
        sid = meta.get('submission_id')
        # Use year from metadata, or lookup from reference
        year = meta.get('year')
        if year is None and sid and sid in year_lookup:
            year = year_lookup[sid]
        metadata_list.append({
            'index': i,
            'year': year,
            'ground_truth': meta.get('answer'),
            'submission_id': sid,
        })
    metadata_df = pd.DataFrame(metadata_list)

    # Load predictions from JSONL
    results_file = base_dir / results_path
    predictions = []
    with open(results_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            pred = extract_boxed_answer(data.get('predict', ''))
            label = extract_boxed_answer(data.get('label', ''))
            predictions.append({
                'index': i,
                'prediction': pred,
                'label_from_results': label,
            })
    predictions_df = pd.DataFrame(predictions)

    # Merge predictions with metadata
    df = pd.merge(metadata_df, predictions_df, on='index')

    # Normalize labels
    df['pred_normalized'] = df['prediction'].apply(normalize_label)
    df['gt_normalized'] = df['ground_truth'].apply(normalize_label)

    # Compute correctness
    df['correct'] = df['pred_normalized'] == df['gt_normalized']

    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute accuracy, accept/reject recall metrics."""
    metrics = {}

    # Overall accuracy
    metrics['overall_accuracy'] = df['correct'].mean()

    # ID accuracy (2020-2024)
    id_mask = df['year'].isin(ID_YEARS)
    metrics['id_accuracy'] = df[id_mask]['correct'].mean() if id_mask.any() else np.nan

    # OOD accuracy (2025)
    ood_mask = df['year'].isin(OOD_YEARS)
    metrics['ood_accuracy'] = df[ood_mask]['correct'].mean() if ood_mask.any() else np.nan

    # Accept recall (true positives among actual accepts)
    accept_mask = df['gt_normalized'] == 'accepted'
    if accept_mask.any():
        metrics['accept_recall'] = df[accept_mask]['correct'].mean()
    else:
        metrics['accept_recall'] = np.nan

    # Reject recall (true negatives among actual rejects)
    reject_mask = df['gt_normalized'] == 'rejected'
    if reject_mask.any():
        metrics['reject_recall'] = df[reject_mask]['correct'].mean()
    else:
        metrics['reject_recall'] = np.nan

    # Per-year accuracy
    metrics['accuracy_by_year'] = {}
    metrics['accept_recall_by_year'] = {}
    metrics['reject_recall_by_year'] = {}
    metrics['pred_accept_rate_by_year'] = {}
    for year in sorted(df['year'].dropna().unique()):
        year_mask = df['year'] == year
        year_df = df[year_mask]
        metrics['accuracy_by_year'][int(year)] = year_df['correct'].mean()

        # Accept recall for this year
        accept_year = year_df[year_df['gt_normalized'] == 'accepted']
        if len(accept_year) > 0:
            metrics['accept_recall_by_year'][int(year)] = accept_year['correct'].mean()

        # Reject recall for this year
        reject_year = year_df[year_df['gt_normalized'] == 'rejected']
        if len(reject_year) > 0:
            metrics['reject_recall_by_year'][int(year)] = reject_year['correct'].mean()

        # Predicted acceptance rate for this year
        pred_accepts = (year_df['pred_normalized'] == 'accepted').sum()
        metrics['pred_accept_rate_by_year'][int(year)] = pred_accepts / len(year_df) if len(year_df) > 0 else np.nan

    # Sample count
    metrics['n_samples'] = len(df)

    return metrics


def load_training_log(train_log_path: str) -> pd.DataFrame:
    """Load training loss from trainer_state.json."""
    base_dir = Path(__file__).parent.parent.parent
    log_file = base_dir / train_log_path

    with open(log_file, 'r') as f:
        state = json.load(f)

    log_history = state.get('log_history', [])

    # Extract only entries with loss
    loss_entries = [entry for entry in log_history if 'loss' in entry]

    df = pd.DataFrame(loss_entries)
    return df


def plot_data_table(all_metrics: dict, output_dir: Path):
    """Save metrics as CSV and PNG table."""
    # Prepare data for table
    # Format: (label, key, is_percentage)
    rows = [
        ('Training Size', 'n_train', False),
        ('Testing Size', 'n_samples', False),
        ('Overall Accuracy', 'overall_accuracy', True),
        ('ID Accuracy (2020-2024)', 'id_accuracy', True),
        ('OOD Accuracy (2025)', 'ood_accuracy', True),
    ]

    data = {}
    for variant in VARIANTS.keys():
        data[variant] = []
        for label, key, is_pct in rows:
            val = all_metrics[variant].get(key, np.nan)
            if is_pct:
                data[variant].append(f"{round(val * 100)}%" if not np.isnan(val) else "N/A")
            else:
                data[variant].append(f"{int(val):,}" if not np.isnan(val) else "N/A")

    # Create DataFrame
    df = pd.DataFrame(data, index=[r[0] for r in rows])

    # Save CSV
    csv_path = output_dir / 'metrics_table.csv'
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    # Create table figure
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(df.columns),
        rowColours=['#f0f0f0'] * len(df.index),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title('Long-Context Modality: Text+Vision vs Vision+Text', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    png_path = output_dir / 'metrics_table.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_recall_bars(all_metrics: dict, output_dir: Path):
    """Create grouped bar chart for Accept/Reject recall."""
    variants = list(VARIANTS.keys())
    x = np.arange(len(variants))
    width = 0.35

    accept_recalls = [all_metrics[v]['accept_recall'] for v in variants]
    reject_recalls = [all_metrics[v]['reject_recall'] for v in variants]

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x - width/2, accept_recalls, width, label='Accept Recall', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, reject_recalls, width, label='Reject Recall', color='#e74c3c', edgecolor='black')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{round(height * 100)}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Ordering', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Accept/Reject Recall: Text+Vision vs Vision+Text', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(labelsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    png_path = output_dir / 'recall_bars.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_accuracy_time_series(all_metrics: dict, output_dir: Path):
    """Create 1x4 line chart of accuracy, accept recall, reject recall, pred accept rate by year."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Get all years across all variants
    all_years = set()
    for variant in VARIANTS.keys():
        all_years.update(all_metrics[variant]['accuracy_by_year'].keys())
    all_years = sorted(all_years)

    # Define the four metrics to plot
    metric_configs = [
        ('accuracy_by_year', 'Accuracy', 'Accuracy by Year'),
        ('accept_recall_by_year', 'Accept Recall', 'Accept Recall by Year'),
        ('reject_recall_by_year', 'Reject Recall', 'Reject Recall by Year'),
        ('pred_accept_rate_by_year', 'Pred. Accept Rate', 'Predicted Accept Rate by Year'),
    ]

    for ax, (metric_key, ylabel, title) in zip(axes, metric_configs):
        for variant in VARIANTS.keys():
            metric_data = all_metrics[variant].get(metric_key, {})
            years = []
            values = []
            for year in all_years:
                val = metric_data.get(year)
                if val is not None:
                    years.append(year)
                    values.append(val)

            if not years:
                continue

            color = COLORS[variant]

            # Plot line
            ax.plot(years, values, '-', color=color, linewidth=2, label=variant)

            # ID markers (x)
            id_years = [y for y in years if y in ID_YEARS]
            id_vals = [values[years.index(y)] for y in id_years]
            if id_years:
                ax.scatter(id_years, id_vals, marker='x', s=80, color=color, zorder=5)

            # OOD markers (o)
            ood_years = [y for y in years if y in OOD_YEARS]
            ood_vals = [values[years.index(y)] for y in ood_years]
            if ood_years:
                ax.scatter(ood_years, ood_vals, marker='o', s=80, color=color, zorder=5,
                          facecolors='white', edgecolors=color, linewidths=2)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(all_years)
        ax.set_xticklabels([str(y) for y in all_years])
        ax.set_ylim(0.2, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.3)

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(VARIANTS), fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    png_path = output_dir / 'accuracy_by_year.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_training_curves(output_dir: Path):
    """Create training loss curves from trainer_state.json."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for variant, config in VARIANTS.items():
        try:
            df = load_training_log(config['train_log'])

            if df.empty or 'loss' not in df.columns:
                print(f"Warning: No loss data for {variant}")
                continue

            # Normalize progress to 0-1
            max_step = df['step'].max()
            progress = df['step'] / max_step

            color = COLORS[variant]
            ax.plot(progress, df['loss'], '-', color=color, linewidth=1.5,
                   label=variant, alpha=0.8)
        except Exception as e:
            print(f"Warning: Could not load training log for {variant}: {e}")

    ax.set_xlabel('Training Progress (normalized)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curves: Text+Vision vs Vision+Text', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    png_path = output_dir / 'training_curves.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def main():
    """Orchestrate all plots."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load data and compute metrics for all variants
    base_dir = Path(__file__).parent.parent.parent
    all_metrics = {}
    for variant, config in VARIANTS.items():
        print(f"\nProcessing: {variant}")
        try:
            use_year_ref = config.get('use_year_reference', False)
            df = load_predictions_with_metadata(config['results'], config['test_data'], use_year_ref)
            metrics = compute_metrics(df)

            # Load training data size
            train_data_path = base_dir / config['train_data']
            with open(train_data_path, 'r') as f:
                train_data = json.load(f)
            metrics['n_train'] = len(train_data)

            all_metrics[variant] = metrics
            print(f"  Training Size: {metrics['n_train']:,}")
            print(f"  Testing Size: {metrics['n_samples']:,}")
            print(f"  Overall Accuracy: {metrics['overall_accuracy']:.1%}")
            print(f"  ID Accuracy: {metrics['id_accuracy']:.1%}")
            print(f"  OOD Accuracy: {metrics['ood_accuracy']:.1%}")
        except Exception as e:
            print(f"  Error: {e}")
            raise

    print("\n" + "="*50)
    print("Generating visualizations...")
    print("="*50)

    # Generate all plots
    plot_data_table(all_metrics, OUTPUT_DIR)
    plot_recall_bars(all_metrics, OUTPUT_DIR)
    plot_accuracy_time_series(all_metrics, OUTPUT_DIR)
    plot_training_curves(OUTPUT_DIR)

    print("\n" + "="*50)
    print("All visualizations generated successfully!")
    print("="*50)

    # List output files
    print("\nOutput files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
