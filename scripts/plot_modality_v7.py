#!/usr/bin/env python3
"""
Generate visualizations for v7 modality sweep results.

Compares three modalities across multiple dataset variants:
- clean: Full Paper (text only)
- images: Full Paper w/ Figures (text + embedded figures)
- vision: Full Paper as Images (paper rendered as images)

Analyzes results across different dataset configurations:
- balanced (2020-2025)
- balanced_2017_2025
- balanced_2024_2025
- balanced_trainagreeing
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# Reuse dataset mappings from analyze_v7.py
DATASET_MAPPINGS = {
    # Text-only (clean)
    "balanced_clean": "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7",
    "balanced_title_abstract": "iclr_2020_2025_85_5_10_split7_balanced_clean_title_abstract_binary_noreviews_v7",
    "balanced_trainagreeing": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7",
    "balanced_2024_2025": "iclr_2024_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7",
    "balanced_2017_2025": "iclr_2017_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7",
    # Vision
    "balanced_vision": "iclr_2020_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_trainagreeing_vision": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "balanced_2024_2025_vision": "iclr_2024_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_2017_2025_vision": "iclr_2017_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    # Images
    "balanced_clean_images": "iclr_2020_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7",
    "balanced_trainagreeing_images": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_images_binary_noreviews_v7",
    "balanced_2024_2025_images": "iclr_2024_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7",
    "balanced_2017_2025_images": "iclr_2017_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7",
    # Yes/No format and no2024 variants
    "balanced_trainagreeing_no2024_vision": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "balanced_trainagreeing_no2024_vision_yesno": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_yesno_noreviews_v7",
}

# Dataset groups for comparison (base_dataset -> modality -> short_name)
DATASET_GROUPS = {
    "balanced": {
        "clean": "balanced_clean",
        "images": "balanced_clean_images",
        "vision": "balanced_vision",
    },
    "balanced_2017_2025": {
        "clean": "balanced_2017_2025",
        "images": "balanced_2017_2025_images",
        "vision": "balanced_2017_2025_vision",
    },
    "balanced_2024_2025": {
        "clean": "balanced_2024_2025",
        "images": "balanced_2024_2025_images",
        "vision": "balanced_2024_2025_vision",
    },
    "balanced_trainagreeing": {
        "clean": "balanced_trainagreeing",
        "images": "balanced_trainagreeing_images",
        "vision": "balanced_trainagreeing_vision",
    },
}

# Result directories
RESULTS_DIRS = {
    "main": Path("results/final_sweep_v7"),
    "pli": Path("results/final_sweep_v7_pli"),  # images variants only
}

# Save directories (for trainer logs)
SAVES_DIRS = {
    "main": Path("saves/final_sweep_v7"),
    "pli": Path("saves/final_sweep_v7_pli"),
}

DATA_DIR = Path("data")
OUTPUT_DIR = Path("results/summarized_investigation/modality_v7")

# Color scheme matching existing plots
MODALITY_COLORS = {
    "clean": "#1f77b4",      # Blue - "Full Paper"
    "images": "#ff7f0e",     # Orange - "Full Paper w/ Figures"
    "vision": "#2ca02c",     # Green - "Full Paper as Images"
}

MODALITY_LABELS = {
    "clean": "Full Paper",
    "images": "Full Paper w/ Figures",
    "vision": "Full Paper as Images",
}

# ID vs OOD years
ID_YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
OOD_YEARS = [2025]


def extract_prediction(text: str) -> str:
    """Extract Accept/Reject from model prediction.

    Handles accept/reject, yes/no, and Y/N formats.
    """
    text_lower = text.lower().strip()

    # Single letter format (Y/N)
    if text_lower == "y":
        return "accept"
    if text_lower == "n":
        return "reject"

    # Look for boxed answers (accept/reject and yes/no)
    if "\\boxed{accept}" in text_lower or "boxed{accept}" in text_lower:
        return "accept"
    if "\\boxed{reject}" in text_lower or "boxed{reject}" in text_lower:
        return "reject"
    if "\\boxed{yes}" in text_lower or "boxed{yes}" in text_lower or "\\boxed{y}" in text_lower:
        return "accept"
    if "\\boxed{no}" in text_lower or "boxed{no}" in text_lower or "\\boxed{n}" in text_lower:
        return "reject"

    # Fallback: look for accept/reject/yes/no keywords
    yes_pos = text_lower.rfind("yes")
    no_pos = text_lower.rfind("no")
    accept_pos = text_lower.rfind("accept")
    reject_pos = text_lower.rfind("reject")

    # Find the last occurring keyword
    positions = [
        (yes_pos, "accept"),
        (no_pos, "reject"),
        (accept_pos, "accept"),
        (reject_pos, "reject"),
    ]
    # Filter out -1 (not found)
    positions = [(pos, label) for pos, label in positions if pos != -1]

    if positions:
        # Return the label of the last occurring keyword
        positions.sort(key=lambda x: x[0])
        return positions[-1][1]

    return "unknown"


def extract_label(text: str) -> str:
    """Extract Accept/Reject from ground truth label.

    Handles accept/reject, yes/no, and Y/N formats.
    """
    text_lower = text.lower().strip()

    # Single letter format (Y/N)
    if text_lower == "y":
        return "accept"
    if text_lower == "n":
        return "reject"

    # Full word formats
    if "accept" in text_lower or "yes" == text_lower:
        return "accept"
    if "reject" in text_lower or "no" == text_lower:
        return "reject"
    return "unknown"


def load_predictions(pred_file: Path) -> list[dict]:
    """Load predictions from jsonl file."""
    predictions = []
    with open(pred_file) as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def load_test_dataset(dataset_name: str) -> list[dict]:
    """Load test dataset with metadata."""
    path = DATA_DIR / f"{dataset_name}_test" / "data.json"
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")

    with open(path) as f:
        return json.load(f)


def compute_metrics_by_year(predictions: list[dict], test_data: list[dict]) -> dict:
    """Compute accuracy, accept recall, reject recall by year."""
    if len(predictions) != len(test_data):
        print(f"Warning: predictions ({len(predictions)}) != test_data ({len(test_data)})")
        min_len = min(len(predictions), len(test_data))
        predictions = predictions[:min_len]
        test_data = test_data[:min_len]

    # Group by year
    year_stats = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0})

    for pred, data in zip(predictions, test_data):
        metadata = data.get("_metadata", {})
        year = metadata.get("year", "unknown")

        pred_label = extract_prediction(pred.get("predict", ""))
        true_label = extract_label(pred.get("label", ""))

        if pred_label == "unknown" or true_label == "unknown":
            continue

        year_stats[year]["total"] += 1

        if true_label == "accept":
            if pred_label == "accept":
                year_stats[year]["tp"] += 1
            else:
                year_stats[year]["fn"] += 1
        else:  # true_label == "reject"
            if pred_label == "reject":
                year_stats[year]["tn"] += 1
            else:
                year_stats[year]["fp"] += 1

    return dict(year_stats)


def compute_overall_stats(year_stats: dict) -> dict:
    """Compute overall stats from year stats."""
    overall = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0}
    for year, stats in year_stats.items():
        if year == "unknown":
            continue
        for key in overall:
            overall[key] += stats[key]
    return overall


def calc_metrics(stats: dict) -> dict:
    """Calculate accuracy, accept recall, reject recall from stats."""
    tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
    total = stats["total"]

    accuracy = (tp + tn) / total if total > 0 else 0
    accept_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    reject_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    pred_accept_rate = (tp + fp) / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "accept_recall": accept_recall,
        "reject_recall": reject_recall,
        "pred_accept_rate": pred_accept_rate,
        "n_samples": total,
    }


def find_variant_dir(short_name: str) -> Path | None:
    """Find the directory containing results for a variant."""
    # Check main results dir first
    main_dir = RESULTS_DIRS["main"] / short_name
    if main_dir.exists():
        return main_dir

    # Check pli dir for images variants
    pli_dir = RESULTS_DIRS["pli"] / short_name
    if pli_dir.exists():
        return pli_dir

    return None


def find_all_checkpoints(variant_dir: Path) -> list[Path]:
    """Find all checkpoint files in a variant directory."""
    return sorted(variant_dir.glob("finetuned*.jsonl"))


def extract_step_from_filename(filename: str) -> int | None:
    """Extract step number from checkpoint filename.

    Returns None for finetuned.jsonl (final checkpoint) - caller should look up actual step.
    """
    # finetuned-ckpt-1069.jsonl -> 1069
    # finetuned.jsonl -> None (caller should look up from trainer log)
    if "ckpt" in filename:
        parts = filename.replace(".jsonl", "").split("-")
        return int(parts[-1])
    return None  # Final checkpoint - need to look up from trainer log


def get_final_step_from_trainer_log(short_name: str) -> int | None:
    """Get the total training steps from trainer_log.jsonl."""
    # Check main saves dir first
    for key in ["main", "pli"]:
        log_path = SAVES_DIRS[key] / short_name / "trainer_log.jsonl"
        if log_path.exists():
            try:
                # Read last line to get total_steps
                with open(log_path, 'r') as f:
                    last_line = None
                    for line in f:
                        last_line = line
                    if last_line:
                        data = json.loads(last_line)
                        return data.get("total_steps") or data.get("current_steps")
            except Exception as e:
                print(f"Warning: Could not read trainer log {log_path}: {e}")
    return None


def find_best_checkpoint(short_name: str) -> dict | None:
    """Find the best checkpoint for a variant based on overall accuracy."""
    variant_dir = find_variant_dir(short_name)
    if not variant_dir:
        return None

    dataset_name = DATASET_MAPPINGS.get(short_name)
    if not dataset_name:
        return None

    try:
        test_data = load_test_dataset(dataset_name)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return None

    checkpoint_files = find_all_checkpoints(variant_dir)
    if not checkpoint_files:
        return None

    # Get final step from trainer log (for finetuned.jsonl)
    final_step = get_final_step_from_trainer_log(short_name)

    best_result = None
    best_accuracy = -1

    for ckpt_file in checkpoint_files:
        try:
            predictions = load_predictions(ckpt_file)
            year_stats = compute_metrics_by_year(predictions, test_data)
            overall_stats = compute_overall_stats(year_stats)
            metrics = calc_metrics(overall_stats)

            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                step = extract_step_from_filename(ckpt_file.name)
                if step is None:  # Final checkpoint
                    step = final_step if final_step else 0
                best_result = {
                    "short_name": short_name,
                    "checkpoint": ckpt_file.name,
                    "step": step,
                    "year_stats": year_stats,
                    "overall_stats": overall_stats,
                    "metrics": metrics,
                }
        except Exception as e:
            print(f"Warning: Error processing {ckpt_file}: {e}")
            continue

    return best_result


def load_all_checkpoints(short_name: str) -> list[dict]:
    """Load metrics for all checkpoints of a variant."""
    variant_dir = find_variant_dir(short_name)
    if not variant_dir:
        return []

    dataset_name = DATASET_MAPPINGS.get(short_name)
    if not dataset_name:
        return []

    try:
        test_data = load_test_dataset(dataset_name)
    except FileNotFoundError:
        return []

    checkpoint_files = find_all_checkpoints(variant_dir)

    # Get final step from trainer log (for finetuned.jsonl)
    final_step = get_final_step_from_trainer_log(short_name)

    results = []

    for ckpt_file in checkpoint_files:
        try:
            predictions = load_predictions(ckpt_file)
            year_stats = compute_metrics_by_year(predictions, test_data)
            overall_stats = compute_overall_stats(year_stats)
            metrics = calc_metrics(overall_stats)
            step = extract_step_from_filename(ckpt_file.name)
            if step is None:  # Final checkpoint
                step = final_step if final_step else 0

            results.append({
                "checkpoint": ckpt_file.name,
                "step": step,
                "metrics": metrics,
            })
        except Exception as e:
            print(f"Warning: Error processing {ckpt_file}: {e}")
            continue

    return sorted(results, key=lambda x: x["step"])


def compute_metrics_for_year(year_stats: dict, year: int) -> dict:
    """Compute metrics for a specific year."""
    if year not in year_stats:
        return {"accuracy": np.nan, "accept_recall": np.nan, "reject_recall": np.nan, "pred_accept_rate": np.nan}
    return calc_metrics(year_stats[year])


def plot_recall_bars(all_best_results: dict, output_dir: Path):
    """Create grouped bar chart for Accept/Reject recall across all dataset groups."""
    # Prepare data: for each dataset group, show bars for each modality
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (group_name, modalities) in zip(axes, DATASET_GROUPS.items()):
        modality_names = []
        accept_recalls = []
        reject_recalls = []
        accuracies = []

        for modality in ["clean", "images", "vision"]:
            short_name = modalities.get(modality)
            if not short_name:
                continue

            result = all_best_results.get(short_name)
            if result:
                modality_names.append(MODALITY_LABELS[modality])
                accept_recalls.append(result["metrics"]["accept_recall"])
                reject_recalls.append(result["metrics"]["reject_recall"])
                accuracies.append(result["metrics"]["accuracy"])
            else:
                modality_names.append(MODALITY_LABELS[modality])
                accept_recalls.append(0)
                reject_recalls.append(0)
                accuracies.append(0)

        x = np.arange(len(modality_names))
        width = 0.25

        bars1 = ax.bar(x - width, accept_recalls, width, label='Accept Recall', color='#2ecc71', edgecolor='black')
        bars2 = ax.bar(x, reject_recalls, width, label='Reject Recall', color='#e74c3c', edgecolor='black')
        bars3 = ax.bar(x + width, accuracies, width, label='Accuracy', color='#3498db', edgecolor='black')

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.0%}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Rate', fontsize=10)
        ax.set_title(f'{group_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modality_names, rotation=15, ha='right', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=0.3)

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    png_path = output_dir / 'recall_bars.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_accuracy_by_year(all_best_results: dict, group_name: str, modalities: dict, output_dir: Path):
    """Create 1x4 subplot showing accuracy, accept recall, reject recall, pred accept rate by year."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Collect all years across modalities
    all_years = set()
    for modality, short_name in modalities.items():
        result = all_best_results.get(short_name)
        if result:
            all_years.update(y for y in result["year_stats"].keys() if y != "unknown")
    all_years = sorted(all_years)

    if not all_years:
        plt.close()
        return

    metric_configs = [
        ('accuracy', 'Accuracy', 'Accuracy by Year'),
        ('accept_recall', 'Accept Recall', 'Accept Recall by Year'),
        ('reject_recall', 'Reject Recall', 'Reject Recall by Year'),
        ('pred_accept_rate', 'Pred. Accept Rate', 'Predicted Accept Rate by Year'),
    ]

    # Collect overall and 2025 accuracy for textbox (only for accuracy subplot)
    accuracy_summary = []

    for ax, (metric_key, ylabel, title) in zip(axes, metric_configs):
        for modality in ["clean", "images", "vision"]:
            short_name = modalities.get(modality)
            if not short_name:
                continue

            result = all_best_results.get(short_name)
            if not result:
                continue

            years = []
            values = []
            for year in all_years:
                metrics = compute_metrics_for_year(result["year_stats"], year)
                val = metrics.get(metric_key)
                if val is not None and not np.isnan(val):
                    years.append(year)
                    values.append(val)

            if not years:
                continue

            color = MODALITY_COLORS[modality]

            # Plot line
            ax.plot(years, values, '-', color=color, linewidth=2, label=MODALITY_LABELS[modality])

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

            # Collect accuracy summary for textbox (only on first metric - accuracy)
            if metric_key == 'accuracy':
                overall_acc = result["metrics"]["accuracy"]
                acc_2025 = compute_metrics_for_year(result["year_stats"], 2025).get("accuracy", np.nan)
                accuracy_summary.append((modality, overall_acc, acc_2025, color))

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(all_years)
        ax.set_xticklabels([str(y) for y in all_years], rotation=45)
        ax.set_ylim(0.2, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)

    # Add textbox with Overall/2025 accuracy to the first subplot (Accuracy by Year)
    if accuracy_summary:
        textbox_lines = ["Overall / 2025:"]
        for modality, overall_acc, acc_2025, color in accuracy_summary:
            label = MODALITY_LABELS[modality]
            if np.isnan(acc_2025):
                textbox_lines.append(f"{label}: {overall_acc:.0%} / N/A")
            else:
                textbox_lines.append(f"{label}: {overall_acc:.0%} / {acc_2025:.0%}")
        textbox_text = "\n".join(textbox_lines)

        # Add textbox to the first axes (Accuracy by Year)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        axes[0].text(0.02, 0.02, textbox_text, transform=axes[0].transAxes, fontsize=9,
                     verticalalignment='bottom', horizontalalignment='left', bbox=props)

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle(f'Metrics by Year: {group_name}', fontsize=16, fontweight='bold', y=1.08)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    png_path = output_dir / f'accuracy_by_year_{group_name}.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_checkpoint_accuracy(all_checkpoint_results: dict, group_name: str, modalities: dict, output_dir: Path):
    """Create line plot showing accuracy vs checkpoint step for each modality."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for modality in ["clean", "images", "vision"]:
        short_name = modalities.get(modality)
        if not short_name:
            continue

        results = all_checkpoint_results.get(short_name, [])
        if not results:
            continue

        steps = [r["step"] for r in results]
        accuracies = [r["metrics"]["accuracy"] for r in results]

        color = MODALITY_COLORS[modality]
        ax.plot(steps, accuracies, '-o', color=color, linewidth=2, markersize=6,
               label=MODALITY_LABELS[modality])

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Checkpoint Accuracy: {group_name}', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='best', fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    png_path = output_dir / f'checkpoint_accuracy_{group_name}.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def save_metrics_table(all_best_results: dict, output_dir: Path):
    """Save summary metrics table as CSV."""
    rows = []

    for group_name, modalities in DATASET_GROUPS.items():
        for modality in ["clean", "images", "vision"]:
            short_name = modalities.get(modality)
            if not short_name:
                continue

            result = all_best_results.get(short_name)
            if result:
                metrics = result["metrics"]
                rows.append({
                    "dataset_group": group_name,
                    "modality": modality,
                    "short_name": short_name,
                    "best_checkpoint": result["checkpoint"],
                    "step": result["step"],
                    "accuracy": metrics["accuracy"],
                    "accept_recall": metrics["accept_recall"],
                    "reject_recall": metrics["reject_recall"],
                    "pred_accept_rate": metrics["pred_accept_rate"],
                    "n_samples": metrics["n_samples"],
                })
            else:
                rows.append({
                    "dataset_group": group_name,
                    "modality": modality,
                    "short_name": short_name,
                    "best_checkpoint": "N/A",
                    "step": np.nan,
                    "accuracy": np.nan,
                    "accept_recall": np.nan,
                    "reject_recall": np.nan,
                    "pred_accept_rate": np.nan,
                    "n_samples": np.nan,
                })

    df = pd.DataFrame(rows)
    csv_path = output_dir / 'metrics_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return df


def load_test_data_with_rating(dataset_name: str) -> pd.DataFrame:
    """Load test dataset with pct_rating metadata."""
    path = DATA_DIR / f"{dataset_name}_test" / "data.json"
    if not path.exists():
        return pd.DataFrame()

    with open(path) as f:
        data = json.load(f)

    rows = []
    for i, item in enumerate(data):
        meta = item.get("_metadata", {})
        rows.append({
            "index": i,
            "submission_id": meta.get("submission_id"),
            "year": meta.get("year"),
            "ground_truth": "accepted" if "accept" in str(meta.get("answer", "")).lower() else "rejected",
            "pct_rating": meta.get("pct_rating"),
        })
    return pd.DataFrame(rows)


def load_predictions_for_variant(short_name: str) -> pd.DataFrame | None:
    """Load predictions for a variant from its best checkpoint."""
    variant_dir = find_variant_dir(short_name)
    if not variant_dir:
        return None

    dataset_name = DATASET_MAPPINGS.get(short_name)
    if not dataset_name:
        return None

    # Find best checkpoint
    checkpoint_files = find_all_checkpoints(variant_dir)
    if not checkpoint_files:
        return None

    # Load test data
    try:
        test_data = load_test_dataset(dataset_name)
    except FileNotFoundError:
        return None

    # Find best checkpoint by accuracy
    best_preds = None
    best_accuracy = -1

    for ckpt_file in checkpoint_files:
        try:
            preds = load_predictions(ckpt_file)
            year_stats = compute_metrics_by_year(preds, test_data)
            overall_stats = compute_overall_stats(year_stats)
            metrics = calc_metrics(overall_stats)

            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_preds = preds
        except Exception:
            continue

    if best_preds is None:
        return None

    # Build prediction dataframe
    rows = []
    for i, (pred, data) in enumerate(zip(best_preds, test_data)):
        meta = data.get("_metadata", {})
        pred_text = pred.get("predict", "")
        pred_label = extract_prediction(pred_text)

        rows.append({
            "index": i,
            "submission_id": meta.get("submission_id"),
            "year": meta.get("year"),
            "prediction": "accepted" if pred_label == "accept" else "rejected",
            "ground_truth": "accepted" if "accept" in str(meta.get("answer", "")).lower() else "rejected",
            "pct_rating": meta.get("pct_rating"),
        })

    return pd.DataFrame(rows)


def load_trainer_log(short_name: str) -> list[dict] | None:
    """Load trainer log data for a variant."""
    for key in ["main", "pli"]:
        log_path = SAVES_DIRS[key] / short_name / "trainer_log.jsonl"
        if log_path.exists():
            try:
                logs = []
                with open(log_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line))
                return logs
            except Exception as e:
                print(f"Warning: Could not read trainer log {log_path}: {e}")
    return None


def plot_loss_curves(output_dir: Path):
    """Plot training loss curves for balanced clean, images, and vision modalities.

    Creates a 1x3 subplot with one panel per modality, x-axis normalized by percentage.
    """
    print("Generating loss curves plot...")

    # Use the "balanced" dataset group
    modalities = DATASET_GROUPS["balanced"]
    modality_order = ["clean", "images", "vision"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    has_any_data = False
    for idx, modality in enumerate(modality_order):
        ax = axes[idx]
        short_name = modalities.get(modality)
        if not short_name:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        logs = load_trainer_log(short_name)
        if not logs:
            print(f"  No trainer log found for {short_name}")
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        # Use percentage field for x-axis
        percentages = [entry["percentage"] for entry in logs if "percentage" in entry and "loss" in entry]
        losses = [entry["loss"] for entry in logs if "percentage" in entry and "loss" in entry]

        if not percentages:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        has_any_data = True
        color = MODALITY_COLORS[modality]
        ax.plot(percentages, losses, '-', color=color, linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Training Progress (%)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(MODALITY_LABELS[modality], fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 100)

    if not has_any_data:
        print("  No loss data found, skipping loss curves plot")
        plt.close()
        return

    plt.suptitle('Training Loss Curves (balanced dataset)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    png_path = output_dir / 'loss_curves_balanced.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_accuracy_by_rating(all_best_results: dict, output_dir: Path):
    """Plot accuracy vs pct_rating (review score percentile) as bar chart.

    Rows: All years, 2020-2024 (ID), 2025 (OOD)
    Columns: Modalities (clean, images, vision)
    """
    print("Generating accuracy by rating plot...")

    # Only use the "balanced" dataset group (2020-2025 data)
    modalities = DATASET_GROUPS["balanced"]
    modality_names = ["clean", "images", "vision"]

    # Load predictions for each modality
    variant_predictions = {}
    for modality in modality_names:
        short_name = modalities.get(modality)
        if short_name:
            preds = load_predictions_for_variant(short_name)
            if preds is not None:
                variant_predictions[modality] = preds

    if not variant_predictions:
        print("No predictions found, skipping accuracy_by_rating plot")
        return

    # Define rating bins
    bin_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

    # Row configurations
    row_configs = [
        ("All Years", None),
        ("2020-2024 (ID)", [2020, 2021, 2022, 2023, 2024]),
        ("2025 (OOD)", [2025]),
    ]
    n_rows = len(row_configs)
    n_cols = len(modality_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for r_idx, (row_label, year_filter) in enumerate(row_configs):
        for m_idx, modality in enumerate(modality_names):
            ax = axes[r_idx, m_idx]
            color = MODALITY_COLORS[modality]

            if modality not in variant_predictions:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                continue

            preds_df = variant_predictions[modality]

            # Filter by years
            if year_filter is None:
                year_data = preds_df.copy()
            else:
                year_data = preds_df[preds_df["year"].isin(year_filter)].copy()

            year_data = year_data.dropna(subset=["pct_rating", "prediction", "ground_truth"])

            if len(year_data) < 10:
                ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
                continue

            # Compute correctness
            year_data["correct"] = (year_data["prediction"] == year_data["ground_truth"]).astype(int)

            # Bin by pct_rating
            year_data["rating_bin"] = pd.cut(year_data["pct_rating"], bins=bin_edges, labels=bin_labels, include_lowest=True)
            bin_stats = year_data.groupby("rating_bin", observed=True).agg(
                accuracy=("correct", "mean"),
                count=("correct", "count")
            ).reindex(bin_labels)

            # Bar chart
            x = np.arange(len(bin_labels))
            accuracies = bin_stats["accuracy"].values
            ax.bar(x, accuracies, color=color, edgecolor='white', linewidth=0.5, alpha=0.8)

            # Add value labels on bars
            for i, (acc, cnt) in enumerate(zip(bin_stats["accuracy"], bin_stats["count"])):
                if not np.isnan(acc):
                    ax.annotate(f'{acc:.0%}',
                               xy=(i, acc),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)

            # Labels
            ax.set_xlabel("Rating Percentile", fontsize=10)
            if m_idx == 0:
                ax.set_ylabel(f"{row_label}\nAccuracy", fontsize=11, fontweight="bold")
            if r_idx == 0:
                ax.set_title(f"{MODALITY_LABELS[modality]}", fontsize=12, fontweight="bold")

            ax.set_ylim(0, 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Accuracy by Rating Percentile (balanced dataset)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_rating.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'accuracy_by_rating.png'}")


def plot_yesno_accuracy_by_year(output_dir: Path):
    """Plot 1x4 grid comparing Accept/Reject vs Y/N format (both no2024 trainagreeing vision)."""
    print("Generating yesno comparison accuracy by year...")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Load results for both variants
    variants = {
        "Accept/Reject": "balanced_trainagreeing_no2024_vision",
        "Y/N": "balanced_trainagreeing_no2024_vision_yesno",
    }

    colors = {
        "Accept/Reject": "#3498db",  # Blue
        "Y/N": "#e74c3c",            # Red
    }

    results = {}
    for label, short_name in variants.items():
        result = find_best_checkpoint(short_name)
        if result:
            results[label] = result

    if not results:
        plt.close()
        return

    # Collect all years
    all_years = set()
    for result in results.values():
        all_years.update(y for y in result["year_stats"].keys() if y != "unknown")
    all_years = sorted(all_years)

    # Prepare data for all metrics
    data_by_variant = {}
    for variant_label in variants.keys():
        if variant_label not in results:
            continue

        result = results[variant_label]
        data_by_variant[variant_label] = {
            "years": [],
            "accuracy": [],
            "accept_recall": [],
            "reject_recall": [],
            "pred_accept_rate": [],
        }

        for year in all_years:
            if year not in result["year_stats"]:
                continue
            metrics = calc_metrics(result["year_stats"][year])
            data_by_variant[variant_label]["years"].append(year)
            data_by_variant[variant_label]["accuracy"].append(metrics["accuracy"])
            data_by_variant[variant_label]["accept_recall"].append(metrics["accept_recall"])
            data_by_variant[variant_label]["reject_recall"].append(metrics["reject_recall"])
            data_by_variant[variant_label]["pred_accept_rate"].append(metrics["pred_accept_rate"])

    # Compute overall and 2025 metrics for text box
    overall_2025_metrics = {}
    for variant_label in variants.keys():
        if variant_label not in results:
            continue

        result = results[variant_label]
        overall = result["metrics"]
        year_2025_metrics = calc_metrics(result["year_stats"][2025]) if 2025 in result["year_stats"] else {}

        overall_2025_metrics[variant_label] = {
            "overall": overall,
            "2025": year_2025_metrics,
        }

    # Plot each metric
    metric_configs = [
        ("accuracy", "Accuracy by Year", 0),
        ("accept_recall", "Accept Recall by Year", 1),
        ("reject_recall", "Reject Recall by Year", 2),
        ("pred_accept_rate", "Predicted Accept Rate by Year", 3),
    ]

    for metric_key, metric_title, idx in metric_configs:
        ax = axes[idx]

        for variant_label in variants.keys():
            if variant_label not in data_by_variant:
                continue

            data = data_by_variant[variant_label]
            if not data["years"]:
                continue

            color = colors[variant_label]
            ax.plot(data["years"], data[metric_key], '-o', color=color,
                   linewidth=2, markersize=8, label=variant_label)

        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel(metric_title.split(' by')[0], fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight='bold')
        ax.set_xticks(all_years)
        ax.set_xticklabels([str(y) for y in all_years], rotation=45)
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)

        # Add text box with overall/2025 metrics
        text_lines = ["Overall / 2025:"]
        for variant_label in variants.keys():
            if variant_label not in overall_2025_metrics:
                continue

            metrics = overall_2025_metrics[variant_label]
            overall_val = metrics["overall"].get(metric_key, 0)
            year_2025_val = metrics["2025"].get(metric_key, 0)

            text_lines.append(f"{variant_label}: {overall_val:.0%} / {year_2025_val:.0%}")

        text_str = "\n".join(text_lines)
        ax.text(0.02, 0.02, text_str, transform=ax.transAxes,
               fontsize=8, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Metrics by Year: Trainagreeing No2024 Vision (Accept/Reject vs Y/N)',
                 fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()

    png_path = output_dir / 'accuracy_by_year_balanced_trainagreeing_yesnocomparison.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_yesno_checkpoint_accuracy(output_dir: Path):
    """Plot test accuracy over checkpoints (yesno comparison)."""
    print("Generating yesno comparison checkpoint accuracy...")

    fig, ax = plt.subplots(figsize=(10, 6))

    variants = {
        "Accept/Reject": "balanced_trainagreeing_no2024_vision",
        "Y/N": "balanced_trainagreeing_no2024_vision_yesno",
    }

    colors = {
        "Accept/Reject": "#3498db",
        "Y/N": "#e74c3c",
    }

    has_data = False
    for variant_label, short_name in variants.items():
        results = load_all_checkpoints(short_name)

        if not results:
            continue

        # Sort by step and assign sequential epochs
        results.sort(key=lambda x: x["step"])
        epochs = list(range(1, len(results) + 1))
        accuracies = [r["metrics"]["accuracy"] for r in results]

        has_data = True
        color = colors[variant_label]
        ax.plot(epochs, accuracies, '-o', color=color, linewidth=2,
               markersize=6, label=variant_label)

    if not has_data:
        plt.close()
        return

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Test Accuracy Over Checkpoints: Trainagreeing No2024 Vision (Accept/Reject vs Y/N)',
                fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.60, 0.80)

    plt.tight_layout()

    png_path = output_dir / 'checkpoint_accuracy_balanced_trainagreeing_yesnocomparison.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_yesno_loss_curves(output_dir: Path):
    """Plot training loss curves comparing Accept/Reject vs Y/N format."""
    print("Generating yesno comparison loss curves...")

    fig, ax = plt.subplots(figsize=(10, 6))

    variants = {
        "Accept/Reject": "balanced_trainagreeing_no2024_vision",
        "Y/N": "balanced_trainagreeing_no2024_vision_yesno",
    }

    colors = {
        "Accept/Reject": "#3498db",
        "Y/N": "#e74c3c",
    }

    has_data = False
    for variant_label, short_name in variants.items():
        logs = load_trainer_log(short_name)

        if not logs:
            continue

        percentages = [entry["percentage"] for entry in logs
                      if "percentage" in entry and "loss" in entry]
        losses = [entry["loss"] for entry in logs
                 if "percentage" in entry and "loss" in entry]

        if not percentages:
            continue

        has_data = True
        color = colors[variant_label]
        ax.plot(percentages, losses, '-', color=color, linewidth=2,
               alpha=0.8, label=variant_label)

    if not has_data:
        plt.close()
        return

    ax.set_xlabel('Training Progress (%)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curves: Trainagreeing No2024 Vision (Accept/Reject vs Y/N)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.8)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    png_path = output_dir / 'loss_curves_balanced_trainagreeing_yesnocomparison.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def main():
    """Orchestrate all plots."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Find best checkpoint for each variant
    print("\n" + "=" * 60)
    print("Finding best checkpoints for each variant...")
    print("=" * 60)

    all_best_results = {}
    all_checkpoint_results = {}

    for group_name, modalities in DATASET_GROUPS.items():
        print(f"\n{group_name}:")
        for modality, short_name in modalities.items():
            print(f"  Processing {short_name}...", end=" ")

            # Find best checkpoint
            best_result = find_best_checkpoint(short_name)
            if best_result:
                all_best_results[short_name] = best_result
                print(f"Best: {best_result['checkpoint']} ({best_result['metrics']['accuracy']:.1%})")
            else:
                print("Not found")

            # Load all checkpoints for training curve
            checkpoint_results = load_all_checkpoints(short_name)
            if checkpoint_results:
                all_checkpoint_results[short_name] = checkpoint_results

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    # 1. Recall bars (all groups in one figure)
    plot_recall_bars(all_best_results, OUTPUT_DIR)

    # 2. Accuracy by year plots (one per dataset group)
    for group_name, modalities in DATASET_GROUPS.items():
        plot_accuracy_by_year(all_best_results, group_name, modalities, OUTPUT_DIR)

    # 3. Checkpoint accuracy plots (one per dataset group)
    for group_name, modalities in DATASET_GROUPS.items():
        plot_checkpoint_accuracy(all_checkpoint_results, group_name, modalities, OUTPUT_DIR)

    # 4. Summary metrics table
    df = save_metrics_table(all_best_results, OUTPUT_DIR)

    # 5. Accuracy by rating percentile
    plot_accuracy_by_rating(all_best_results, OUTPUT_DIR)

    # 6. Loss curves
    plot_loss_curves(OUTPUT_DIR)

    # 7. Yes/No comparison plots (trainagreeing no2024 vision)
    print("\n" + "=" * 60)
    print("Generating Yes/No comparison plots...")
    print("=" * 60)
    plot_yesno_accuracy_by_year(OUTPUT_DIR)
    plot_yesno_checkpoint_accuracy(OUTPUT_DIR)
    plot_yesno_loss_curves(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)

    # List output files
    print("\nOutput files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}")

    # Print summary table
    print("\nSummary Metrics:")
    print("-" * 80)
    for _, row in df.iterrows():
        if not np.isnan(row["accuracy"]):
            print(f"{row['dataset_group']:25} {row['modality']:10} Acc={row['accuracy']:.1%} "
                  f"AccRecall={row['accept_recall']:.1%} RejRecall={row['reject_recall']:.1%}")


if __name__ == "__main__":
    main()
