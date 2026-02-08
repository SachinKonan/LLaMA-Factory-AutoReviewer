#!/usr/bin/env python3
"""
Plot training curves for LR experiment runs.

Reads trainer_log.jsonl files from saves/lr_experiment_v7/ and creates
comparison plots for different learning rate configurations.

Usage:
    python scripts/plot_lr_experiment.py
    python scripts/plot_lr_experiment.py --output figures/lr_experiment.png
    python scripts/plot_lr_experiment.py --live  # Auto-refresh for monitoring
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Hardcoded mapping: experiment_name -> raw log file path
# These are the .out files that contain accuracy metrics for CLS training
EXPERIMENT_LOG_FILES = {
    # Old text experiments (LR sweep)
    "text_trainagreeing_lr_2e5": "logs/lr_experiment_v7/4442026_0.out",
    "text_trainagreeing_lr_2e6": "logs/lr_experiment_v7/4442026_1.out",
    "text_trainagreeing_lr_2e5_backbone_2e6": "logs/lr_experiment_v7/4442026_2.out",
    "text_trainagreeing_lr_2.5e6_bs32_4epoch": "logs/lr_experiment_v7/4475015.out",
    # Old vision experiments (LR sweep)
    "vision_trainagreeing_lr_2e5": "logs/lr_experiment_v7/4450300_0.out",
    "vision_trainagreeing_lr_2e6": "logs/lr_experiment_v7/4450300_1.out",
    "vision_trainagreeing_lr_2e5_backbone_2e6": "logs/lr_experiment_v7/4450300_2.out",
    "vision_trainagreeing_lr_2.5e6_bs32_4epoch": "logs/lr_experiment_v7/4475014.out",
    # New experiments (no2024, bs16, 3epoch)
    "vision_trainagreeing_no2024_head2e5_bb2e6_bs16_3epoch": "logs/lr_experiment_v7/4486374_0.out",
    "vision_balanced_no2024_head2e5_bb2e6_bs16_3epoch": "logs/lr_experiment_v7/4486374_1.out",
    "text_trainagreeing_no2024_lr_1.75e6_bs16_3epoch": "logs/lr_experiment_v7/4486375_0.out",
    "text_balanced_no2024_lr_1.75e6_bs16_3epoch": "logs/lr_experiment_v7/4486375_1.out",
    # Original trainagreeing experiments (jobs 4524591-4524596)
    "trainagreeing_original_no2024_text_cls": "logs/lr_experiment_v7/4524591.out",
    "trainagreeing_original_no2024_text_cls_rating": "logs/lr_experiment_v7/4524592.out",
    "trainagreeing_original_no2024_text_sft": "logs/lr_experiment_v7/4524593.out",
    "trainagreeing_original_no2024_vision_cls": "logs/lr_experiment_v7/4524594.out",
    "trainagreeing_original_no2024_vision_cls_rating": "logs/lr_experiment_v7/4524595.out",
    "trainagreeing_original_no2024_vision_sft": "logs/lr_experiment_v7/4524596.out",
}

# Experiment groupings for separate plots
OLD_EXPERIMENTS = [
    "text_trainagreeing_lr_2e5",
    "text_trainagreeing_lr_2e6",
    "text_trainagreeing_lr_2e5_backbone_2e6",
    "vision_trainagreeing_lr_2e5",
    "vision_trainagreeing_lr_2e6",
    "vision_trainagreeing_lr_2e5_backbone_2e6",
    "text_trainagreeing_lr_2.5e6_bs32_4epoch",
    "vision_trainagreeing_lr_2.5e6_bs32_4epoch",
]

NEW_EXPERIMENTS = [
    "text_trainagreeing_no2024_lr_1.75e6_bs16_3epoch",
    "text_balanced_no2024_lr_1.75e6_bs16_3epoch",
    "vision_trainagreeing_no2024_head2e5_bb2e6_bs16_3epoch",
    "vision_balanced_no2024_head2e5_bb2e6_bs16_3epoch",
]


def load_trainer_log(log_file: Path) -> pd.DataFrame:
    """Load trainer_log.jsonl into a DataFrame."""
    records = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def parse_raw_log_for_metrics(log_file: Path) -> dict:
    """Parse a raw .out log file to extract train and eval metrics for CLS training.

    CLS trainer logs accuracy to stdout but LogCallback doesn't capture it.
    This function parses the raw output to get accuracy data.

    Returns dict with separate lists for train and eval metrics:
        - epoch, loss, accuracy (training - many points)
        - eval_epoch, eval_loss, eval_accuracy (eval - one per epoch)
    """
    metrics = {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'eval_epoch': [],
        'eval_loss': [],
        'eval_accuracy': [],
    }

    if not log_file.exists():
        return metrics

    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Match training lines: {'loss': 0.6786, ..., 'accuracy': 0.606, ..., 'epoch': 0.08}
                # Must have 'accuracy' but NOT 'eval_loss'
                if "'accuracy':" in line and "'epoch':" in line and "'eval_loss'" not in line:
                    try:
                        match = re.search(r"\{[^}]+\}", line)
                        if match:
                            json_str = match.group().replace("'", '"')
                            data = json.loads(json_str)
                            if 'accuracy' in data and 'epoch' in data:
                                metrics['epoch'].append(data['epoch'])
                                metrics['accuracy'].append(data['accuracy'])
                                metrics['loss'].append(data.get('loss', data.get('cls_loss')))
                    except (json.JSONDecodeError, AttributeError):
                        continue

                # Match eval lines: {'eval_loss': ..., 'eval_accuracy': ..., 'epoch': ...}
                # Also handle cases with only eval_accuracy (CLS training)
                elif "'eval_accuracy':" in line and "'epoch':" in line and "'eval_runtime':" in line:
                    try:
                        match = re.search(r"\{[^}]+\}", line)
                        if match:
                            json_str = match.group().replace("'", '"')
                            data = json.loads(json_str)
                            if 'eval_accuracy' in data and 'epoch' in data:
                                metrics['eval_epoch'].append(data['epoch'])
                                metrics['eval_loss'].append(data.get('eval_loss', 0))
                                metrics['eval_accuracy'].append(data.get('eval_accuracy', 0))
                    except (json.JSONDecodeError, AttributeError):
                        continue
    except Exception:
        pass

    return metrics


def parse_experiment_name(exp_name: str) -> Dict[str, str]:
    """Parse experiment name to extract model type and config."""
    info = {
        'model_type': 'unknown',
        'dataset': 'unknown',
        'lr_config': 'unknown',
        'short_name': exp_name,
    }

    # Handle trainagreeing_original experiments
    if exp_name.startswith('trainagreeing_original_no2024_'):
        parts = exp_name.split('_')
        if 'text' in parts:
            info['model_type'] = 'text'
        elif 'vision' in parts:
            info['model_type'] = 'vision'

        info['dataset'] = 'trainagreeing_no2024'

        # Extract method (cls, cls_rating, sft)
        if 'sft' in parts:
            info['lr_config'] = 'SFT'
            info['short_name'] = 'SFT'
        elif 'cls_rating' in exp_name:
            info['lr_config'] = 'CLS+Rating'
            info['short_name'] = 'CLS+Rating'
        elif 'cls' in parts:
            info['lr_config'] = 'CLS'
            info['short_name'] = 'CLS'

        return info

    if exp_name.startswith('text_'):
        info['model_type'] = 'text'
        remainder = exp_name[5:]  # Remove 'text_'
    elif exp_name.startswith('vision_'):
        info['model_type'] = 'vision'
        remainder = exp_name[7:]  # Remove 'vision_'
    else:
        return info

    # Parse dataset and LR config
    # Examples:
    #   trainagreeing_lr_2e5
    #   trainagreeing_no2024_lr_1.75e6_bs16_3epoch
    #   balanced_no2024_head2e5_bb2e6_bs16_3epoch

    if 'trainagreeing' in remainder:
        info['dataset'] = 'trainagreeing'
        ds_short = 'agree'
    elif 'balanced' in remainder:
        info['dataset'] = 'balanced'
        ds_short = 'bal'
    else:
        ds_short = 'unk'

    # Extract LR config for short name
    if 'head2e5_bb2e6' in remainder or 'head2e-5_bb2e-6' in remainder:
        info['lr_config'] = 'head=2e-5, bb=2e-6'
        info['short_name'] = f"{ds_short}_h2e5_bb2e6"
    elif 'lr_2e5_backbone_2e6' in remainder:
        info['lr_config'] = 'head=2e-5, bb=2e-6'
        info['short_name'] = f"{ds_short}_h2e5_bb2e6"
    elif 'lr_2e5' in remainder:
        info['lr_config'] = 'LR=2e-5 (uniform)'
        info['short_name'] = f"{ds_short}_lr2e5"
    elif 'lr_2e6' in remainder:
        info['lr_config'] = 'LR=2e-6 (uniform)'
        info['short_name'] = f"{ds_short}_lr2e6"
    elif 'lr_1.75e6' in remainder or 'lr_1.75e-6' in remainder:
        info['lr_config'] = 'LR=1.75e-6'
        info['short_name'] = f"{ds_short}_lr1.75e6"
    elif 'lr_2.5e6' in remainder or 'lr_2.5e-6' in remainder:
        info['lr_config'] = 'LR=2.5e-6'
        info['short_name'] = f"{ds_short}_lr2.5e6"

    # Add batch size info if present
    if 'bs16' in remainder:
        info['short_name'] += '_bs16'
    elif 'bs32' in remainder:
        info['short_name'] += '_bs32'

    return info


def load_all_experiments(saves_dir: Path, base_dir: Path = None) -> Dict[str, Tuple[Dict, pd.DataFrame]]:
    """Load all experiment logs from saves directory.

    Args:
        saves_dir: Directory containing experiment saves (trainer_log.jsonl)
        base_dir: Base directory for resolving relative paths in EXPERIMENT_LOG_FILES
    """
    experiments = {}

    if base_dir is None:
        base_dir = Path(__file__).parent.parent  # Repo root

    if not saves_dir.exists():
        print(f"Saves directory not found: {saves_dir}")
        return experiments

    for exp_dir in sorted(saves_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        log_file = exp_dir / "trainer_log.jsonl"
        if not log_file.exists():
            continue

        df = load_trainer_log(log_file)
        if df.empty:
            continue

        exp_name = exp_dir.name
        info = parse_experiment_name(exp_name)

        # Try to get accuracy from raw log file if available
        if exp_name in EXPERIMENT_LOG_FILES:
            raw_log_path = base_dir / EXPERIMENT_LOG_FILES[exp_name]
            metrics = parse_raw_log_for_metrics(raw_log_path)
            if metrics['epoch']:
                # Convert to DataFrame for compatibility
                df = pd.DataFrame({
                    'epoch': metrics['epoch'],
                    'loss': metrics['loss'],
                    'accuracy': metrics['accuracy'],
                })
                # Store eval metrics separately in info
                info['eval_epoch'] = metrics['eval_epoch']
                info['eval_loss'] = metrics['eval_loss']
                info['eval_accuracy'] = metrics['eval_accuracy']
                has_eval = len(metrics['eval_epoch']) > 0
                print(f"Loaded: {exp_name} ({len(df)} steps, with accuracy" + (f", {len(metrics['eval_epoch'])} eval points)" if has_eval else ")"))
            else:
                print(f"Loaded: {exp_name} ({len(df)} steps, no accuracy)")
        else:
            print(f"Loaded: {exp_name} ({len(df)} steps, no raw log mapping)")

        experiments[exp_name] = (info, df)

    return experiments


def plot_training_curves(
    experiments: Dict[str, Tuple[Dict, pd.DataFrame]],
    output_file: Optional[Path] = None,
    show: bool = True,
    title_prefix: str = "",
):
    """Create comparison plots for a set of experiments.

    Creates a 2x4 grid:
    - Row 1: Text model (train_loss, train_acc, eval_loss, eval_acc)
    - Row 2: Vision model (train_loss, train_acc, eval_loss, eval_acc)
    """
    if not experiments:
        print("No experiments to plot.")
        return

    # Separate by model type
    text_exps = {k: v for k, v in experiments.items() if v[0]['model_type'] == 'text'}
    vision_exps = {k: v for k, v in experiments.items() if v[0]['model_type'] == 'vision'}

    # Create 2x4 figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def plot_metric(ax, exps: Dict, metric: str, ylabel: str, title: str, is_eval: bool = False):
        """Plot a single metric for a set of experiments.

        Args:
            is_eval: If True, plot eval metrics (discrete points with markers)
        """
        if not exps:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        has_data = False
        for idx, (exp_name, (info, df)) in enumerate(sorted(exps.items())):
            label = info['short_name']
            color = colors[idx % len(colors)]

            if is_eval:
                # Eval metrics are stored in info dict, not df
                eval_key = metric  # e.g., 'eval_loss', 'eval_accuracy'
                epoch_key = 'eval_epoch'
                if epoch_key in info and eval_key in info:
                    x = info[epoch_key]
                    y = info[eval_key]
                    if len(x) > 0 and len(y) > 0:
                        ax.plot(x, y, label=label, color=color, alpha=0.9,
                               linewidth=2, marker='o', markersize=8)
                        has_data = True
            else:
                # Training metrics are in df
                if metric not in df.columns:
                    continue

                # Filter out NaN values
                valid_mask = df[metric].notna()
                if not valid_mask.any():
                    continue

                x = df.loc[valid_mask, 'epoch'] if 'epoch' in df.columns else df.loc[valid_mask, 'current_steps']
                y = df.loc[valid_mask, metric]

                ax.plot(x, y, label=label, color=color, alpha=0.8, linewidth=1.5)
                has_data = True

        if not has_data:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix}{title}" if title_prefix else title)
        if has_data:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    # Row 0: Text model
    plot_metric(axes[0, 0], text_exps, 'loss', 'Loss', 'Text - Train Loss')
    plot_metric(axes[0, 1], text_exps, 'accuracy', 'Accuracy', 'Text - Train Acc')
    plot_metric(axes[0, 2], text_exps, 'eval_loss', 'Loss', 'Text - Eval Loss', is_eval=True)
    plot_metric(axes[0, 3], text_exps, 'eval_accuracy', 'Accuracy', 'Text - Eval Acc', is_eval=True)

    # Row 1: Vision model
    plot_metric(axes[1, 0], vision_exps, 'loss', 'Loss', 'Vision - Train Loss')
    plot_metric(axes[1, 1], vision_exps, 'accuracy', 'Accuracy', 'Vision - Train Acc')
    plot_metric(axes[1, 2], vision_exps, 'eval_loss', 'Loss', 'Vision - Eval Loss', is_eval=True)
    plot_metric(axes[1, 3], vision_exps, 'eval_accuracy', 'Accuracy', 'Vision - Eval Acc', is_eval=True)

    plt.tight_layout()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_old_and_new_separately(
    experiments: Dict[str, Tuple[Dict, pd.DataFrame]],
    output_dir: Path,
    show: bool = True,
):
    """Create separate plots for old and new experiments."""
    # Split into old and new
    old_exps = {k: v for k, v in experiments.items() if k in OLD_EXPERIMENTS}
    new_exps = {k: v for k, v in experiments.items() if k in NEW_EXPERIMENTS}

    # Plot old experiments
    if old_exps:
        print(f"\n=== OLD EXPERIMENTS ({len(old_exps)}) ===")
        plot_training_curves(
            old_exps,
            output_file=output_dir / "lr_experiment_old.png",
            show=show,
            title_prefix="[OLD] ",
        )

    # Plot new experiments
    if new_exps:
        print(f"\n=== NEW EXPERIMENTS ({len(new_exps)}) ===")
        plot_training_curves(
            new_exps,
            output_file=output_dir / "lr_experiment_new.png",
            show=show,
            title_prefix="[NEW] ",
        )


def print_current_status(experiments: Dict[str, Tuple[Dict, pd.DataFrame]]):
    """Print current training status for all experiments."""
    print("\n" + "=" * 90)
    print("CURRENT TRAINING STATUS")
    print("=" * 90)
    print(f"{'Experiment':<50} {'Epoch':>8} {'Loss':>10} {'Accuracy':>10} {'Steps':>12}")
    print("-" * 90)

    for exp_name, (info, df) in sorted(experiments.items()):
        if df.empty:
            continue

        last = df.iloc[-1]
        epoch = last.get('epoch', 0)
        loss = last.get('loss', float('nan'))
        acc = last.get('accuracy', float('nan'))
        steps = f"{int(last.get('current_steps', 0))}/{int(last.get('total_steps', 0))}"

        acc_str = f"{acc:.4f}" if not np.isnan(acc) else "n/a"
        print(f"{exp_name:<50} {epoch:>8.2f} {loss:>10.4f} {acc_str:>10} {steps:>12}")

    print("=" * 90 + "\n")


def live_monitor(saves_dir: Path, interval: int = 30):
    """Live monitoring mode - refresh plots periodically."""
    print(f"Live monitoring mode. Refresh every {interval}s. Press Ctrl+C to stop.")

    plt.ion()  # Interactive mode
    fig = None

    try:
        while True:
            experiments = load_all_experiments(saves_dir)

            if experiments:
                print_current_status(experiments)

                if fig:
                    plt.close(fig)

                plot_training_curves(experiments, show=False)
                plt.pause(0.1)

            print(f"Next refresh in {interval}s...")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopping live monitor.")
        plt.close('all')


def main():
    parser = argparse.ArgumentParser(description='Plot LR experiment training curves')
    parser.add_argument('--saves-dir', type=Path,
                        default=Path('saves/lr_experiment_v7'),
                        help='Directory containing experiment saves')
    parser.add_argument('--output-dir', '-o', type=Path,
                        default=Path('figures'),
                        help='Output directory for plots')
    parser.add_argument('--live', action='store_true',
                        help='Live monitoring mode')
    parser.add_argument('--interval', type=int, default=60,
                        help='Refresh interval for live mode (seconds)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (just save)')
    parser.add_argument('--combined', action='store_true',
                        help='Create a single combined plot instead of separate old/new')
    args = parser.parse_args()

    if args.live:
        live_monitor(args.saves_dir, args.interval)
    else:
        experiments = load_all_experiments(args.saves_dir)

        if not experiments:
            print("No experiments found. Check if training has started.")
            return

        print_current_status(experiments)

        if args.combined:
            # Single combined plot
            plot_training_curves(
                experiments,
                output_file=args.output_dir / "lr_experiment_curves.png",
                show=not args.no_show
            )
        else:
            # Separate plots for old and new experiments
            plot_old_and_new_separately(
                experiments,
                output_dir=args.output_dir,
                show=not args.no_show
            )


if __name__ == '__main__':
    main()
