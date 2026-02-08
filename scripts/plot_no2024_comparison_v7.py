#!/usr/bin/env python3
"""
Generate visualizations comparing 2024-included vs 2024-excluded datasets.

Comparisons:
1. balanced_vision (2020-2025) vs balanced_no2024_vision (2020,2023,2025)
2. balanced_trainagreeing_vision (2020-2025) vs balanced_trainagreeing_no2024_vision (2020,2023,2025)

For each comparison, generates:
- accuracy_by_year: Performance by year (with2024 vs no2024)
- checkpoint_accuracy: 1x2 grid (train acc, test acc) over checkpoints
- loss_curves: Training loss over time
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# Dataset mappings
DATASET_MAPPINGS = {
    "balanced_vision": "iclr_2020_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_no2024_vision": "iclr_2020_2023_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_trainagreeing_vision": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "balanced_trainagreeing_no2024_vision": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "balanced_trainagreeing_no2024_vision_yesno_trainacc": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_yesno_noreviews_v7",
}

# Comparison pairs
COMPARISON_PAIRS = {
    "balanced": {
        "with2024": "balanced_vision",
        "no2024": "balanced_no2024_vision",
        "title": "Balanced (Vision)",
    },
    "balanced_trainagreeing": {
        "with2024": "balanced_trainagreeing_vision",
        "no2024": "balanced_trainagreeing_no2024_vision",
        "title": "Balanced Train-Agreeing (Vision)",
    },
    "balanced_trainagreeing_yesnocomparison": {
        "with2024": "balanced_trainagreeing_no2024_vision",
        "no2024": "balanced_trainagreeing_no2024_vision_yesno_trainacc",
        "title": "Balanced Train-Agreeing No2024 (Accept/Reject vs Y/N)",
    },
}

# Directories
RESULTS_DIR = Path("results/final_sweep_v7")
SAVES_DIR = Path("saves/final_sweep_v7")
DATA_DIR = Path("data")
OUTPUT_DIR = Path("results/summarized_investigation/no2024_v7")

# Color scheme
COLORS = {
    "with2024": "#3498db",  # Blue - includes 2024
    "no2024": "#e74c3c",    # Red - excludes 2024
}

LABELS = {
    "with2024": "With 2024 Data",
    "no2024": "Without 2024 Data",
}


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
    """Compute accuracy metrics by year."""
    if len(predictions) != len(test_data):
        min_len = min(len(predictions), len(test_data))
        predictions = predictions[:min_len]
        test_data = test_data[:min_len]

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
        else:
            if pred_label == "reject":
                year_stats[year]["tn"] += 1
            else:
                year_stats[year]["fp"] += 1

    return dict(year_stats)


def calc_metrics(stats: dict) -> dict:
    """Calculate accuracy metrics from stats."""
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


def compute_overall_stats(year_stats: dict) -> dict:
    """Compute overall stats from year stats."""
    overall = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0}
    for year, stats in year_stats.items():
        if year == "unknown":
            continue
        for key in overall:
            overall[key] += stats[key]
    return overall


def find_all_test_checkpoints(short_name: str) -> list[Path]:
    """Find all test checkpoint prediction files (finetuned-ckpt-*.jsonl)."""
    variant_dir = RESULTS_DIR / short_name
    if not variant_dir.exists():
        return []
    return sorted(variant_dir.glob("finetuned*.jsonl"))


def find_all_train_checkpoints(short_name: str) -> list[Path]:
    """Find all train checkpoint accuracy files (train-ckpt-*.json)."""
    variant_dir = RESULTS_DIR / short_name
    if not variant_dir.exists():
        return []
    return sorted(variant_dir.glob("train-ckpt-*.json"))


def extract_step_from_filename(filename: str) -> int | None:
    """Extract step number from checkpoint filename."""
    if "ckpt" in filename:
        parts = filename.replace(".jsonl", "").replace(".json", "").split("-")
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            return None
    return None


def get_final_step_from_trainer_log(short_name: str) -> int | None:
    """Get total training steps from trainer_log.jsonl."""
    log_path = SAVES_DIR / short_name / "trainer_log.jsonl"
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                last_line = None
                for line in f:
                    last_line = line
                if last_line:
                    data = json.loads(last_line)
                    return data.get("total_steps") or data.get("current_steps")
        except Exception:
            pass
    return None


def load_trainer_log(short_name: str) -> list[dict] | None:
    """Load trainer log data."""
    log_path = SAVES_DIR / short_name / "trainer_log.jsonl"
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


def load_checkpoint_metrics(short_name: str) -> dict:
    """Load test accuracy and training accuracy for all checkpoints.

    Returns dict with:
    - test_accuracy: list of (epoch, accuracy) tuples
    - train_accuracy: list of (epoch, accuracy) tuples

    Epochs are assigned as sequential integers (1, 2, 3, ...) based on checkpoint order.
    """
    result = {"test_accuracy": [], "train_accuracy": []}

    # Load test accuracy from prediction files (finetuned-ckpt-*.jsonl)
    dataset_name = DATASET_MAPPINGS.get(short_name)
    if dataset_name:
        try:
            test_data = load_test_dataset(dataset_name)
            checkpoint_files = find_all_test_checkpoints(short_name)
            final_step = get_final_step_from_trainer_log(short_name)

            # Collect (step, accuracy) pairs
            test_checkpoints = []
            for ckpt_file in checkpoint_files:
                try:
                    predictions = load_predictions(ckpt_file)
                    year_stats = compute_metrics_by_year(predictions, test_data)
                    overall_stats = compute_overall_stats(year_stats)
                    metrics = calc_metrics(overall_stats)

                    step = extract_step_from_filename(ckpt_file.name)
                    if step is None:
                        step = final_step if final_step else 0

                    test_checkpoints.append((step, metrics["accuracy"]))
                except Exception:
                    continue

            # Sort by step and assign sequential epoch numbers
            test_checkpoints.sort(key=lambda x: x[0])
            result["test_accuracy"] = [(i + 1, acc) for i, (step, acc) in enumerate(test_checkpoints)]

        except FileNotFoundError:
            pass

    # Load training accuracy from train-ckpt-*.json files
    train_checkpoint_files = find_all_train_checkpoints(short_name)
    train_checkpoints = []
    for train_ckpt_file in train_checkpoint_files:
        try:
            with open(train_ckpt_file, 'r') as f:
                data = json.load(f)
                step = extract_step_from_filename(train_ckpt_file.name)
                if step is None:
                    continue

                # Extract accuracy from the JSON (try multiple field names)
                accuracy = data.get("accuracy") or data.get("sft_accuracy") or data.get("cls_accuracy", 0)
                train_checkpoints.append((step, accuracy))
        except Exception as e:
            print(f"Warning: Could not read {train_ckpt_file}: {e}")
            continue

    # Sort by step and assign sequential epoch numbers
    train_checkpoints.sort(key=lambda x: x[0])
    result["train_accuracy"] = [(i + 1, acc) for i, (step, acc) in enumerate(train_checkpoints)]

    return result


def load_best_checkpoint_metrics(short_name: str) -> dict | None:
    """Load metrics for the best checkpoint."""
    dataset_name = DATASET_MAPPINGS.get(short_name)
    if not dataset_name:
        return None

    try:
        test_data = load_test_dataset(dataset_name)
    except FileNotFoundError:
        return None

    checkpoint_files = find_all_test_checkpoints(short_name)
    if not checkpoint_files:
        return None

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
                best_result = {
                    "year_stats": year_stats,
                    "metrics": metrics,
                }
        except Exception:
            continue

    return best_result


def compute_metrics_for_year(year_stats: dict, year: int) -> dict:
    """Compute metrics for a specific year."""
    if year not in year_stats:
        return {"accuracy": np.nan}
    return calc_metrics(year_stats[year])


def plot_accuracy_by_year(pair_name: str, pair_config: dict, output_dir: Path):
    """Plot 1x4 grid: accuracy, accept recall, reject recall, predicted acceptance rate by year."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Use custom labels for yesno comparison
    if "yesno" in pair_name:
        labels = {"with2024": "Accept/Reject", "no2024": "Y/N"}
    else:
        labels = LABELS

    results = {}
    for variant_key in ["with2024", "no2024"]:
        short_name = pair_config[variant_key]
        result = load_best_checkpoint_metrics(short_name)
        if result:
            results[variant_key] = result

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
    for variant_key in ["with2024", "no2024"]:
        if variant_key not in results:
            continue

        result = results[variant_key]
        data_by_variant[variant_key] = {
            "years": [],
            "accuracy": [],
            "accept_recall": [],
            "reject_recall": [],
            "pred_accept_rate": [],
        }

        for year in all_years:
            metrics = compute_metrics_for_year(result["year_stats"], year)
            acc = metrics.get("accuracy")
            if acc is not None and not np.isnan(acc):
                data_by_variant[variant_key]["years"].append(year)
                data_by_variant[variant_key]["accuracy"].append(acc)
                data_by_variant[variant_key]["accept_recall"].append(metrics.get("accept_recall", 0))
                data_by_variant[variant_key]["reject_recall"].append(metrics.get("reject_recall", 0))
                data_by_variant[variant_key]["pred_accept_rate"].append(metrics.get("pred_accept_rate", 0))

    # Compute overall and 2025 metrics for text box
    overall_2025_metrics = {}
    for variant_key in ["with2024", "no2024"]:
        if variant_key not in results:
            continue

        result = results[variant_key]

        # Overall metrics
        overall = result["metrics"]

        # 2025 metrics
        year_2025_metrics = compute_metrics_for_year(result["year_stats"], 2025)

        overall_2025_metrics[variant_key] = {
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

        for variant_key in ["with2024", "no2024"]:
            if variant_key not in data_by_variant:
                continue

            data = data_by_variant[variant_key]
            if not data["years"]:
                continue

            color = COLORS[variant_key]
            variant_label = labels[variant_key]

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
        for variant_key in ["with2024", "no2024"]:
            if variant_key not in overall_2025_metrics:
                continue

            metrics = overall_2025_metrics[variant_key]
            overall_val = metrics["overall"].get(metric_key, 0)
            year_2025_val = metrics["2025"].get(metric_key, 0)

            label = labels[variant_key]
            text_lines.append(f"{label}: {overall_val:.0%} / {year_2025_val:.0%}")

        text_str = "\n".join(text_lines)
        ax.text(0.02, 0.02, text_str, transform=ax.transAxes,
               fontsize=8, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'Metrics by Year: {pair_config["title"]}',
                 fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()

    png_path = output_dir / f'accuracy_by_year_{pair_name}.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_checkpoint_accuracy(pair_name: str, pair_config: dict, output_dir: Path):
    """Plot 1x2 grid: train accuracy and test accuracy over checkpoints.

    Note: Train accuracy only available for variants with train-ckpt-*.json files.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Use custom labels for yesno comparison
    if "yesno" in pair_name:
        labels = {"with2024": "Accept/Reject", "no2024": "Y/N"}
    else:
        labels = LABELS

    metrics_data = {}
    for variant_key in ["with2024", "no2024"]:
        short_name = pair_config[variant_key]
        metrics_data[variant_key] = load_checkpoint_metrics(short_name)

    # Left subplot: Training Accuracy
    ax_train = axes[0]
    has_train_data = False
    all_epochs_train = []
    all_accuracies_train = []
    for variant_key in ["with2024", "no2024"]:
        data = metrics_data.get(variant_key, {})
        train_acc = data.get("train_accuracy", [])

        if train_acc:
            has_train_data = True
            epochs, accuracies = zip(*train_acc)  # Already in epochs from the file
            all_epochs_train.extend(epochs)
            all_accuracies_train.extend(accuracies)

            color = COLORS[variant_key]
            variant_label = labels[variant_key]
            ax_train.plot(epochs, accuracies, '-o', color=color, linewidth=2,
                         markersize=6, alpha=0.8, label=variant_label)

    if has_train_data:
        ax_train.set_xlabel('Epoch', fontsize=12)
        ax_train.set_ylabel('Training Accuracy', fontsize=12)
        ax_train.set_title('Training Accuracy Over Checkpoints', fontsize=13, fontweight='bold')
        ax_train.yaxis.set_major_formatter(PercentFormatter(1))
        ax_train.legend(loc='best', fontsize=11)
        ax_train.grid(alpha=0.3)
        # Set y-limits based on data range, with padding
        max_acc = max(all_accuracies_train)
        ax_train.set_ylim(0.55, min(1.05, max_acc + 0.05))  # Leave room at top
        # More granular x-ticks
        if all_epochs_train:
            max_epoch = max(all_epochs_train)
            ax_train.set_xticks(np.arange(1, int(max_epoch) + 2, 1))
    else:
        ax_train.text(0.5, 0.5, 'No training checkpoint data available',
                     ha='center', va='center', transform=ax_train.transAxes,
                     fontsize=12, color='gray')
        ax_train.set_title('Training Accuracy Over Checkpoints', fontsize=13, fontweight='bold')

    # Right subplot: Test Accuracy
    ax_test = axes[1]
    has_test_data = False
    all_epochs_test = []
    for variant_key in ["with2024", "no2024"]:
        data = metrics_data.get(variant_key, {})
        test_acc = data.get("test_accuracy", [])

        if test_acc:
            has_test_data = True
            epochs, accuracies = zip(*test_acc)  # Already sequential integers from load_checkpoint_metrics
            all_epochs_test.extend(epochs)

            color = COLORS[variant_key]
            variant_label = labels[variant_key]
            ax_test.plot(epochs, accuracies, '-o', color=color, linewidth=2,
                        markersize=6, label=variant_label)

    if has_test_data:
        ax_test.set_xlabel('Epoch', fontsize=12)
        ax_test.set_ylabel('Test Accuracy', fontsize=12)
        ax_test.set_title('Test Accuracy Over Checkpoints', fontsize=13, fontweight='bold')
        ax_test.yaxis.set_major_formatter(PercentFormatter(1))
        ax_test.legend(loc='best', fontsize=11)
        ax_test.grid(alpha=0.3)
        ax_test.set_ylim(0.60, 0.80)  # Centered around the data with more granularity
        # More granular x-ticks
        if all_epochs_test:
            max_epoch = max(all_epochs_test)
            ax_test.set_xticks(np.arange(1, int(max_epoch) + 2, 1))
    else:
        ax_test.text(0.5, 0.5, 'No test checkpoint data available',
                    ha='center', va='center', transform=ax_test.transAxes,
                    fontsize=12, color='gray')
        ax_test.set_title('Test Accuracy Over Checkpoints', fontsize=13, fontweight='bold')

    plt.suptitle(f'Checkpoint Accuracy: {pair_config["title"]}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    png_path = output_dir / f'checkpoint_accuracy_{pair_name}.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_loss_curves(pair_name: str, pair_config: dict, output_dir: Path):
    """Plot training loss curves comparing with2024 vs no2024."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use custom labels for yesno comparison
    if "yesno" in pair_name:
        labels = {"with2024": "Accept/Reject", "no2024": "Y/N"}
    else:
        labels = LABELS

    has_data = False
    for variant_key in ["with2024", "no2024"]:
        short_name = pair_config[variant_key]
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
        color = COLORS[variant_key]
        variant_label = labels[variant_key]
        ax.plot(percentages, losses, '-', color=color, linewidth=2,
               alpha=0.8, label=variant_label)

    if not has_data:
        plt.close()
        return

    ax.set_xlabel('Training Progress (%)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training Loss Curves: {pair_config["title"]}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.8)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    png_path = output_dir / f'loss_curves_{pair_name}.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def main():
    """Generate all comparison plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("Generating no2024 comparison visualizations...")
    print("=" * 60)

    for pair_name, pair_config in COMPARISON_PAIRS.items():
        print(f"\n{pair_config['title']}:")
        print(f"  {LABELS['with2024']}: {pair_config['with2024']}")
        print(f"  {LABELS['no2024']}: {pair_config['no2024']}")

        # Generate plots for this comparison
        plot_accuracy_by_year(pair_name, pair_config, OUTPUT_DIR)
        plot_checkpoint_accuracy(pair_name, pair_config, OUTPUT_DIR)
        plot_loss_curves(pair_name, pair_config, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)

    print("\nOutput files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()