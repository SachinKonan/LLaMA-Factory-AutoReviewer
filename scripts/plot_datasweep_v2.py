#!/usr/bin/env python3
"""Plot training loss for data_sweep_v2 experiments.

Layout: 2 rows (clean, vision) x 6 columns (dataset variants)
Each subplot shows training loss vs % completion with min value annotated.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import time


# Dataset configurations matching the sbatch script
DATASETS = [
    ("iclr17_balanced", "ICLR 2017-2025\nBalanced"),
    ("iclr20_balanced", "ICLR 2020-2025\nBalanced"),
    ("iclr20_trainagreeing", "ICLR 2020-2025\nTrain Agreeing"),
    ("iclr20_original", "ICLR 2020-2025\nOriginal"),
    ("iclr_nips_balanced", "ICLR+NeurIPS\nBalanced"),
    ("iclr_nips_accepts", "ICLR+NeurIPS\nNIPS Accepts"),
]

MODALITIES = ["clean", "vision"]


def load_training_log(save_dir: Path) -> tuple:
    """Load training log from trainer_log.jsonl.

    Returns: (entries, is_recently_modified)
    """
    log_path = save_dir / "trainer_log.jsonl"

    if not log_path.exists():
        return None, False

    # Check if file was modified in the last 5 minutes (300 seconds)
    mtime = log_path.stat().st_mtime
    is_recently_modified = (time.time() - mtime) < 300

    entries = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    return entries, is_recently_modified


def extract_loss_data(entries: list) -> tuple:
    """Extract training loss and progress from trainer log.

    Returns: (pct_complete, losses, min_loss)
    """
    if entries is None or len(entries) == 0:
        return None, None, None

    pct_complete = []
    losses = []

    for entry in entries:
        if "loss" in entry and "percentage" in entry:
            pct_complete.append(entry["percentage"])
            losses.append(entry["loss"])

    if not losses:
        return None, None, None

    return pct_complete, losses, min(losses) if losses else None


def plot_datasweep_v2(base_dir: Path, output_path: Path = None):
    """Create the 2x6 plot grid."""
    fig, axes = plt.subplots(2, 6, figsize=(20, 8), squeeze=False)

    for col_idx, (short_name, title) in enumerate(DATASETS):
        for row_idx, modality in enumerate(MODALITIES):
            ax = axes[row_idx, col_idx]

            # Construct save directory path
            save_dir = base_dir / f"{short_name}_{modality}"

            # Load and extract data
            entries, is_running = load_training_log(save_dir)
            pct_complete, losses, min_loss = extract_loss_data(entries)

            if pct_complete is not None and losses is not None:
                ax.plot(pct_complete, losses, linewidth=1.5, color='#1f77b4')

                # Mark minimum point
                min_idx = np.argmin(losses)
                ax.scatter([pct_complete[min_idx]], [losses[min_idx]],
                          color='red', s=50, zorder=5)
                ax.annotate(f'min: {min_loss:.4f}',
                           xy=(pct_complete[min_idx], losses[min_idx]),
                           xytext=(5, 10), textcoords='offset points',
                           fontsize=8, color='red')

                ax.set_xlim(0, 100)

                # Mark if still running
                if is_running:
                    ax.axvline(x=pct_complete[-1], color='orange', linestyle='--',
                              linewidth=2, alpha=0.7)
                    ax.text(0.98, 0.98, 'RUNNING', transform=ax.transAxes,
                           ha='right', va='top', fontsize=8, color='orange',
                           fontweight='bold', bbox=dict(boxstyle='round',
                           facecolor='white', edgecolor='orange', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, color='gray')

            # Labels
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if row_idx == 1:
                ax.set_xlabel('% Complete', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f'{modality.upper()}\nTraining Loss', fontsize=10)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    plt.suptitle('Data Sweep v2: Training Loss by Dataset and Modality',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    return fig


def print_summary(base_dir: Path):
    """Print summary table of min losses."""
    print("\n" + "="*80)
    print("Data Sweep v2 Summary - Minimum Training Loss")
    print("="*80)

    header = f"{'Dataset':<25} {'Clean':<20} {'Vision':<20}"
    print(header)
    print("-"*80)

    for short_name, title in DATASETS:
        clean_title = title.replace('\n', ' ')
        row = f"{clean_title:<25}"

        for modality in MODALITIES:
            save_dir = base_dir / f"{short_name}_{modality}"
            entries, is_running = load_training_log(save_dir)
            _, _, min_loss = extract_loss_data(entries)

            if min_loss is not None:
                status = " (running)" if is_running else ""
                row += f" {min_loss:.4f}{status:<11}"
            else:
                row += f" {'N/A':<20}"

        print(row)

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Plot data_sweep_v2 training losses")
    parser.add_argument("--base_dir", type=str,
                       default="saves/data_sweep_v2",
                       help="Base directory containing sweep results")
    parser.add_argument("--output", type=str,
                       default="results/data_sweep_v2/training_loss_grid.png",
                       help="Output path for the plot")
    parser.add_argument("--no-save", action="store_true",
                       help="Show plot instead of saving")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist")
        return

    # Print summary
    print_summary(base_dir)

    # Create output directory if needed
    output_path = None if args.no_save else Path(args.output)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create plot
    plot_datasweep_v2(base_dir, output_path)


if __name__ == "__main__":
    main()