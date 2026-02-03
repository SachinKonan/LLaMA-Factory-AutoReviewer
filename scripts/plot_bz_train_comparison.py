#!/usr/bin/env python3
"""
Compare training loss and test accuracy between batch sizes (16 vs 32) for clean and vision modalities.

Creates:
1. A 2x2 subplot for training loss curves (rows: clean vs vision, columns: bz16 vs bz32)
2. A bar chart showing test accuracy comparison

Output: results/summarized_investigation/old_hyperparam_investigation/
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuration
SAVES_DIRS = {
    ("clean", "bz16"): Path("saves/final_sweep_v7/balanced_clean"),
    ("vision", "bz16"): Path("saves/final_sweep_v7/balanced_vision"),
    ("clean", "bz32"): Path("saves/hyperparam_sweep_pli_v2/lr3.0e-6_b32_clean"),
    ("vision", "bz32"): Path("saves/hyperparam_sweep_pli_v2/lr3.0e-6_b32_vision"),
}

RESULTS_DIRS = {
    ("clean", "bz16"): Path("results/final_sweep_v7/balanced_clean"),
    ("vision", "bz16"): Path("results/final_sweep_v7/balanced_vision"),
    ("clean", "bz32"): Path("results/hyperparam_sweep_pli_v2/lr3.0e-6_b32_clean"),
    ("vision", "bz32"): Path("results/hyperparam_sweep_pli_v2/lr3.0e-6_b32_vision"),
}

OUTPUT_DIR = Path("results/summarized_investigation/old_hyperparam_investigation")

# Colors
COLORS = {
    "bz16": "#1f77b4",  # Blue
    "bz32": "#ff7f0e",  # Orange
}

MODALITY_LABELS = {
    "clean": "Full Paper (Text)",
    "vision": "Full Paper as Images",
}

BZ_LABELS = {
    "bz16": "Batch Size 16",
    "bz32": "Batch Size 32",
}


def load_trainer_log(save_dir: Path) -> list[dict] | None:
    """Load trainer log data."""
    log_path = save_dir / "trainer_log.jsonl"
    if not log_path.exists():
        print(f"  Warning: trainer_log.jsonl not found at {log_path}")
        return None

    try:
        logs = []
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        return logs
    except Exception as e:
        print(f"  Warning: Could not read trainer log {log_path}: {e}")
        return None


def extract_prediction(text: str) -> str:
    """Extract Accept/Reject from model prediction."""
    text_lower = text.lower()

    if "\\boxed{accept}" in text_lower or "boxed{accept}" in text_lower:
        return "accept"
    if "\\boxed{reject}" in text_lower or "boxed{reject}" in text_lower:
        return "reject"

    if "accept" in text_lower and "reject" not in text_lower:
        return "accept"
    if "reject" in text_lower and "accept" not in text_lower:
        return "reject"

    accept_pos = text_lower.rfind("accept")
    reject_pos = text_lower.rfind("reject")

    if accept_pos > reject_pos:
        return "accept"
    elif reject_pos > accept_pos:
        return "reject"

    return "unknown"


def load_predictions_and_compute_accuracy(results_dir: Path) -> float | None:
    """Load predictions and compute accuracy."""
    # Find prediction files
    pred_files = sorted(results_dir.glob("finetuned*.jsonl"))
    if not pred_files:
        print(f"  Warning: No prediction files found in {results_dir}")
        return None

    # Use the last checkpoint (or finetuned.jsonl)
    # Sort by checkpoint number, with finetuned.jsonl (no number) being last
    def sort_key(f):
        match = re.search(r"ckpt-(\d+)", f.name)
        return int(match.group(1)) if match else float('inf')

    pred_files = sorted(pred_files, key=sort_key)
    pred_file = pred_files[-1]  # Use last/final checkpoint

    print(f"  Using prediction file: {pred_file.name}")

    try:
        correct = 0
        total = 0
        with open(pred_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                pred = extract_prediction(data.get("predict", ""))
                label = extract_prediction(data.get("label", ""))

                if pred != "unknown" and label != "unknown":
                    total += 1
                    if pred == label:
                        correct += 1

        if total == 0:
            return None

        return correct / total

    except Exception as e:
        print(f"  Warning: Error computing accuracy: {e}")
        return None


def plot_training_loss_grid(output_dir: Path):
    """Create 2x2 subplot for training loss curves."""
    print("Generating training loss grid...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    modalities = ["clean", "vision"]
    batch_sizes = ["bz16", "bz32"]

    for row, modality in enumerate(modalities):
        for col, bz in enumerate(batch_sizes):
            ax = axes[row, col]
            save_dir = SAVES_DIRS.get((modality, bz))

            if not save_dir or not save_dir.exists():
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                continue

            logs = load_trainer_log(save_dir)
            if not logs:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                continue

            # Use percentage for x-axis
            percentages = [entry["percentage"] for entry in logs if "percentage" in entry and "loss" in entry]
            losses = [entry["loss"] for entry in logs if "percentage" in entry and "loss" in entry]

            if not percentages:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                continue

            color = COLORS[bz]
            ax.plot(percentages, losses, '-', color=color, linewidth=1.5, alpha=0.8)

            ax.set_xlabel('Training Progress (%)', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(f'{MODALITY_LABELS[modality]} - {BZ_LABELS[bz]}', fontsize=11, fontweight='bold')
            ax.tick_params(labelsize=9)
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=0)
            ax.set_xlim(0, 100)

    plt.suptitle('Training Loss Comparison: Batch Size 16 vs 32', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / 'training_loss_grid.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_test_accuracy_bars(output_dir: Path):
    """Create bar chart for test accuracy comparison."""
    print("Generating test accuracy bar chart...")

    modalities = ["clean", "vision"]
    batch_sizes = ["bz16", "bz32"]

    # Compute accuracies
    accuracies = {}
    for modality in modalities:
        for bz in batch_sizes:
            results_dir = RESULTS_DIRS.get((modality, bz))
            if results_dir and results_dir.exists():
                print(f"Computing accuracy for {modality} {bz}...")
                acc = load_predictions_and_compute_accuracy(results_dir)
                accuracies[(modality, bz)] = acc
            else:
                accuracies[(modality, bz)] = None

    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(modalities))
    width = 0.35

    bz16_accs = [accuracies.get((m, "bz16")) for m in modalities]
    bz32_accs = [accuracies.get((m, "bz32")) for m in modalities]

    # Replace None with 0 for plotting
    bz16_accs_plot = [a if a is not None else 0 for a in bz16_accs]
    bz32_accs_plot = [a if a is not None else 0 for a in bz32_accs]

    bars1 = ax.bar(x - width/2, bz16_accs_plot, width, label=BZ_LABELS["bz16"], color=COLORS["bz16"], edgecolor='black')
    bars2 = ax.bar(x + width/2, bz32_accs_plot, width, label=BZ_LABELS["bz32"], color=COLORS["bz32"], edgecolor='black')

    # Add value labels
    for bars, accs in [(bars1, bz16_accs), (bars2, bz32_accs)]:
        for bar, acc in zip(bars, accs):
            if acc is not None:
                height = bar.get_height()
                ax.annotate(f'{acc:.1%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Test Accuracy: Batch Size 16 vs 32', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODALITY_LABELS[m] for m in modalities], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Format y-axis as percentage
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / 'test_accuracy_comparison.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")

    # Print summary
    print("\nAccuracy Summary:")
    print("-" * 50)
    for modality in modalities:
        for bz in batch_sizes:
            acc = accuracies.get((modality, bz))
            if acc is not None:
                print(f"  {MODALITY_LABELS[modality]:30} {BZ_LABELS[bz]:15} {acc:.1%}")
            else:
                print(f"  {MODALITY_LABELS[modality]:30} {BZ_LABELS[bz]:15} N/A")


def main():
    """Generate all plots."""
    print("=" * 60)
    print("Batch Size Comparison: Training Loss and Test Accuracy")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Training loss grid (2x2)
    plot_training_loss_grid(OUTPUT_DIR)

    # 2. Test accuracy bar chart
    plot_test_accuracy_bars(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Done! Output files:")
    print("=" * 60)
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f}")


if __name__ == "__main__":
    main()
