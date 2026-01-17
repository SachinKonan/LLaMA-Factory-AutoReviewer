#!/usr/bin/env python3
"""Plot training loss for hyperparam_sweep_v2 experiments.

Layout: Dynamically sized grid based on discovered configurations.
Groups by modality (clean/vision) and compares different lr/batch combinations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import time
import re
from collections import defaultdict


def parse_config_name(name: str) -> dict:
    """Parse config name like 'lr3.0e-6_b32_clean' into components."""
    match = re.match(r'lr([\d.e-]+)_b(\d+)_(\w+)', name)
    if match:
        return {
            'lr': match.group(1),
            'batch': int(match.group(2)),
            'modality': match.group(3),
            'name': name
        }
    return None


def discover_configs(base_dir: Path) -> list:
    """Discover all config directories in the base directory."""
    configs = []
    if not base_dir.exists():
        return configs

    for d in base_dir.iterdir():
        if d.is_dir():
            parsed = parse_config_name(d.name)
            if parsed:
                configs.append(parsed)

    return configs


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


def plot_hyperparam_v2(base_dir: Path, output_path: Path = None):
    """Create the plot grid based on discovered configurations."""
    configs = discover_configs(base_dir)

    if not configs:
        print(f"No configurations found in {base_dir}")
        return None

    # Group configs by modality
    modalities = sorted(set(c['modality'] for c in configs))

    # Get unique lr/batch combinations
    lr_batch_combos = sorted(set((c['lr'], c['batch']) for c in configs),
                              key=lambda x: (float(x[0]), x[1]))

    n_rows = len(modalities)
    n_cols = max(len(lr_batch_combos), 1)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    # Color palette for different configurations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for row_idx, modality in enumerate(modalities):
        modality_configs = [c for c in configs if c['modality'] == modality]

        for col_idx, (lr, batch) in enumerate(lr_batch_combos):
            ax = axes[row_idx, col_idx]

            # Find matching config
            matching = [c for c in modality_configs if c['lr'] == lr and c['batch'] == batch]

            if matching:
                config = matching[0]
                save_dir = base_dir / config['name']

                entries, is_running = load_training_log(save_dir)
                pct_complete, losses, min_loss = extract_loss_data(entries)

                if pct_complete is not None and losses is not None:
                    ax.plot(pct_complete, losses, linewidth=1.5,
                           color=colors[col_idx % len(colors)])

                    # Mark minimum point
                    min_idx = np.argmin(losses)
                    ax.scatter([pct_complete[min_idx]], [losses[min_idx]],
                              color='red', s=50, zorder=5)
                    ax.annotate(f'min: {min_loss:.4f}',
                               xy=(pct_complete[min_idx], losses[min_idx]),
                               xytext=(5, 10), textcoords='offset points',
                               fontsize=9, color='red')

                    ax.set_xlim(0, 100)

                    # Mark if still running
                    if is_running:
                        ax.axvline(x=pct_complete[-1], color='orange', linestyle='--',
                                  linewidth=2, alpha=0.7)
                        ax.text(0.98, 0.98, 'RUNNING', transform=ax.transAxes,
                               ha='right', va='top', fontsize=9, color='orange',
                               fontweight='bold', bbox=dict(boxstyle='round',
                               facecolor='white', edgecolor='orange', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                           ha='center', va='center', fontsize=12, color='gray')
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, color='gray')

            # Labels
            if row_idx == 0:
                ax.set_title(f'LR={lr}, Batch={batch}', fontsize=11, fontweight='bold')
            if row_idx == n_rows - 1:
                ax.set_xlabel('% Complete', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f'{modality.upper()}\nTraining Loss', fontsize=11)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)

    plt.suptitle('Hyperparameter Sweep v2: Training Loss',
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
    configs = discover_configs(base_dir)

    if not configs:
        print(f"No configurations found in {base_dir}")
        return

    print("\n" + "=" * 80)
    print("Hyperparameter Sweep v2 Summary - Minimum Training Loss")
    print("=" * 80)

    # Group by modality
    modalities = sorted(set(c['modality'] for c in configs))

    for modality in modalities:
        print(f"\n{modality.upper()}:")
        print("-" * 60)
        print(f"{'Config':<30} {'Min Loss':<15} {'Status':<15}")
        print("-" * 60)

        modality_configs = sorted([c for c in configs if c['modality'] == modality],
                                   key=lambda x: (float(x['lr']), x['batch']))

        for config in modality_configs:
            save_dir = base_dir / config['name']
            entries, is_running = load_training_log(save_dir)
            _, _, min_loss = extract_loss_data(entries)

            status = "running" if is_running else ("complete" if min_loss else "no data")
            loss_str = f"{min_loss:.4f}" if min_loss else "N/A"

            print(f"lr={config['lr']}, b={config['batch']:<5} {loss_str:<15} {status:<15}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Plot hyperparam_sweep_v2 training losses")
    parser.add_argument("--base_dir", type=str,
                       default="saves/hyperparam_sweep_pli_v2",
                       help="Base directory containing sweep results")
    parser.add_argument("--output", type=str,
                       default="results/hyperparam_sweep_pli_v2/training_loss_grid.png",
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
    plot_hyperparam_v2(base_dir, output_path)


if __name__ == "__main__":
    main()
