#!/usr/bin/env python3
"""
Analyze and plot training curves for lr_experiment_text_trainagreeing and lr_experiment_vision_trainagreeing runs.

Creates a 2x2 grid:
- Row 1: Text experiments
- Row 2: Vision experiments
- Column 1: Training Loss
- Column 2: Training Accuracy / P(correct)
"""

import json
import re
import ast
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import argparse


def parse_training_log_from_output(log_file: Path) -> dict:
    """Parse training metrics from a .out log file.

    Returns dict with lists of step, loss, accuracy values.
    Includes both training metrics and eval metrics at epoch boundaries.
    """
    metrics = {
        'steps': [],
        'loss': [],
        'accuracy': [],
        'epoch': [],
        'lr': [],
        # SFT-specific metrics (if available)
        'sft_accuracy': [],
        'sft_p_correct_mean': [],
        # Rating-specific decomposed losses (if available)
        'cls_loss': [],
        'loss_bce': [],
        'loss_rating': [],
        # Eval metrics (at epoch boundaries)
        'eval_epoch': [],
        'eval_loss': [],
        'eval_accuracy': [],
        'eval_f1': [],
    }

    if not log_file.exists():
        print(f"Warning: {log_file} does not exist")
        return metrics

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue

            try:
                data = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue

            # Training metrics (has 'loss' and 'accuracy' but not 'eval_loss')
            if 'loss' in data and 'accuracy' in data and 'eval_loss' not in data:
                metrics['epoch'].append(data.get('epoch', 0))
                metrics['loss'].append(data['loss'])
                metrics['accuracy'].append(data['accuracy'])
                metrics['lr'].append(data.get('learning_rate', 0))

                # SFT-specific metrics if available
                if 'sft_accuracy' in data:
                    metrics['sft_accuracy'].append(data['sft_accuracy'])
                if 'sft_p_correct_mean' in data:
                    metrics['sft_p_correct_mean'].append(data['sft_p_correct_mean'])

                # Rating-specific decomposed losses if available
                if 'cls_loss' in data:
                    metrics['cls_loss'].append(data['cls_loss'])
                if 'loss_bce' in data:
                    metrics['loss_bce'].append(data['loss_bce'])
                if 'loss_rating' in data:
                    metrics['loss_rating'].append(data['loss_rating'])

            # Eval metrics (has 'eval_loss' or 'eval_accuracy' with 'eval_runtime')
            elif 'eval_accuracy' in data and 'eval_runtime' in data:
                metrics['eval_epoch'].append(data.get('epoch', 0))
                metrics['eval_loss'].append(data.get('eval_loss', 0))
                metrics['eval_accuracy'].append(data.get('eval_accuracy', 0))
                metrics['eval_f1'].append(data.get('eval_f1', 0))

    # Create step index
    metrics['steps'] = list(range(len(metrics['loss'])))

    return metrics


def parse_trainer_log_jsonl(log_file: Path) -> dict:
    """Parse trainer_log.jsonl file for loss values.

    Note: This file typically doesn't have accuracy, but has loss per step.
    """
    metrics = {
        'steps': [],
        'loss': [],
        'epoch': [],
        'lr': [],
    }

    if not log_file.exists():
        print(f"Warning: {log_file} does not exist")
        return metrics

    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'current_steps' in data and 'loss' in data:
                    metrics['steps'].append(data['current_steps'])
                    metrics['loss'].append(data['loss'])
                    metrics['epoch'].append(data.get('epoch', 0))
                    metrics['lr'].append(data.get('lr', 0))
            except json.JSONDecodeError:
                continue

    return metrics


def find_log_files(base_log_dir: Path, job_pattern: str) -> dict:
    """Find log files matching job pattern.

    Returns dict mapping array_task_id to file path.
    Handles both array jobs (jobid_taskid.out) and single jobs (jobid.out).
    """
    files = {}
    # Try array job pattern first
    for f in base_log_dir.glob(f"{job_pattern}_*.out"):
        # Extract task id from filename like 4450300_0.out
        match = re.search(r'_(\d+)\.out$', f.name)
        if match:
            task_id = int(match.group(1))
            files[task_id] = f

    # If no array files found, try single job pattern
    if not files:
        single_file = base_log_dir / f"{job_pattern}.out"
        if single_file.exists():
            files[0] = single_file  # Use task_id 0 for single jobs

    return files


def identify_experiment_jobs(log_dir: Path) -> tuple:
    """Identify which job IDs correspond to text and vision experiments.

    Returns (text_job_id, vision_job_id) as strings.
    Prefers the most recent (highest numbered) job IDs.
    """
    text_jobs = []
    vision_jobs = []

    for f in sorted(log_dir.glob("*.out")):
        # Read first 20 lines to identify model
        with open(f, 'r') as fp:
            header = ''.join([fp.readline() for _ in range(20)])

        if 'Qwen2.5-7B-Instruct' in header and 'VL' not in header:
            # Text model
            match = re.search(r'/(\d+)_\d+\.out$', str(f))
            if match:
                text_jobs.append(match.group(1))
        elif 'Qwen2.5-VL-7B-Instruct' in header:
            # Vision model
            match = re.search(r'/(\d+)_\d+\.out$', str(f))
            if match:
                vision_jobs.append(match.group(1))

    # Return the most recent (highest numbered) job IDs
    text_job = max(set(text_jobs), key=int) if text_jobs else None
    vision_job = max(set(vision_jobs), key=int) if vision_jobs else None

    return text_job, vision_job




def create_plots(
    text_metrics: dict,
    vision_metrics: dict,
    output_path: Path,
):
    """Create 2x4 plot grid.

    Args:
        text_metrics: Dict mapping lr_config -> metrics dict
        vision_metrics: Dict mapping lr_config -> metrics dict
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Color scheme for different LR configs
    colors = {
        'lr_2e5': '#2ecc71',            # Green
        'lr_2e6': '#3498db',            # Blue
        'lr_2e5_backbone_2e6': '#e74c3c',  # Red
        'lr_2.5e6_bs32': '#9b59b6',     # Purple
        'lr_rating': '#f39c12',         # Orange
    }

    labels = {
        'lr_2e5': 'LR=2e-5 (uniform)',
        'lr_2e6': 'LR=2e-6 (uniform)',
        'lr_2e5_backbone_2e6': 'LR=2e-5 head, 2e-6 backbone',
        'lr_2.5e6_bs32': 'LR=2.5e-6, batch=32',
        'lr_rating': 'CLS+Rating',
    }

    # Helper function to plot one row
    def plot_row(ax_loss, ax_acc, ax_eval_loss, ax_eval_acc, metrics_dict, row_title):
        # Column 0: Training Loss
        for lr_config, metrics in metrics_dict.items():
            if len(metrics['loss']) > 0 and len(metrics['epoch']) > 0:
                color = colors.get(lr_config, 'gray')
                label = labels.get(lr_config, lr_config)

                # For rating jobs with decomposed losses, plot 3 lines
                if lr_config == 'lr_rating' and len(metrics['cls_loss']) > 0 and len(metrics['loss_rating']) > 0:
                    # Combined loss (solid line)
                    ax_loss.plot(metrics['epoch'], metrics['loss'],
                           color=color, label=label,
                           linewidth=1.5, alpha=0.9)
                    # CLS loss component (dashed line)
                    ax_loss.plot(metrics['epoch'], metrics['cls_loss'],
                           color=color, label=f'{label} (CLS)',
                           linewidth=1, linestyle='--', alpha=0.7)
                    # Rating loss component (dotted line)
                    ax_loss.plot(metrics['epoch'], metrics['loss_rating'],
                           color=color, label=f'{label} (Rating)',
                           linewidth=1, linestyle=':', alpha=0.7)
                else:
                    # Regular single-line plot
                    ax_loss.plot(metrics['epoch'], metrics['loss'],
                           color=color, label=label,
                           linewidth=1, alpha=0.8)

        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'{row_title} - Training Loss')
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)

        # Column 1: Training Accuracy
        for lr_config, metrics in metrics_dict.items():
            if len(metrics['accuracy']) > 0 and len(metrics['epoch']) > 0:
                ax_acc.plot(metrics['epoch'], metrics['accuracy'],
                       color=colors.get(lr_config, 'gray'),
                       label=labels.get(lr_config, lr_config),
                       linewidth=1, alpha=0.8)

        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title(f'{row_title} - Training Accuracy')
        ax_acc.legend(fontsize=8)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim(0.4, 1.0)

        # Column 2: Eval Loss
        for lr_config, metrics in metrics_dict.items():
            if len(metrics['eval_epoch']) > 0 and len(metrics['eval_loss']) > 0:
                epochs = metrics['eval_epoch']
                eval_loss = metrics['eval_loss']
                ax_eval_loss.plot(epochs, eval_loss,
                       color=colors.get(lr_config, 'gray'),
                       label=labels.get(lr_config, lr_config),
                       linewidth=2, marker='o', markersize=8)

        ax_eval_loss.set_xlabel('Epoch')
        ax_eval_loss.set_ylabel('Eval Loss')
        ax_eval_loss.set_title(f'{row_title} - Eval Loss')
        ax_eval_loss.legend(fontsize=8)
        ax_eval_loss.grid(True, alpha=0.3)

        # Column 3: Eval Accuracy
        for lr_config, metrics in metrics_dict.items():
            if len(metrics['eval_epoch']) > 0 and len(metrics['eval_accuracy']) > 0:
                epochs = metrics['eval_epoch']
                eval_acc = metrics['eval_accuracy']
                ax_eval_acc.plot(epochs, eval_acc,
                       color=colors.get(lr_config, 'gray'),
                       label=labels.get(lr_config, lr_config),
                       linewidth=2, marker='o', markersize=8)

        ax_eval_acc.set_xlabel('Epoch')
        ax_eval_acc.set_ylabel('Eval Accuracy')
        ax_eval_acc.set_title(f'{row_title} - Eval Accuracy')
        ax_eval_acc.legend(fontsize=8)
        ax_eval_acc.grid(True, alpha=0.3)
        ax_eval_acc.set_ylim(0.0, 1.0)

    # Row 0: Text experiments
    plot_row(axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3], text_metrics, 'Text Model')

    # Row 1: Vision experiments
    plot_row(axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3], vision_metrics, 'Vision Model')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def create_batch_comparison_plots(
    text_bs16_metrics: dict,
    text_bs32_metrics: dict,
    vision_bs16_metrics: dict,
    vision_bs32_metrics: dict,
    output_path: Path,
):
    """Create 2x2 plot comparing batch 16 vs batch 32.

    Args:
        text_bs16_metrics: Metrics for text model with batch=16 (lr_2e6 config)
        text_bs32_metrics: Metrics for text model with batch=32
        vision_bs16_metrics: Metrics for vision model with batch=16 (lr_2e6 config)
        vision_bs32_metrics: Metrics for vision model with batch=32
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Color scheme for batch sizes
    colors = {
        'bs16': '#3498db',   # Blue
        'bs32': '#e74c3c',   # Red
    }

    def plot_comparison(ax_loss, ax_acc, bs16_metrics, bs32_metrics, row_title):
        """Plot loss and accuracy comparison for one model type."""
        # Training Loss
        if bs16_metrics and len(bs16_metrics.get('loss', [])) > 0:
            epochs = bs16_metrics.get('epoch', list(range(len(bs16_metrics['loss']))))
            ax_loss.plot(epochs, bs16_metrics['loss'],
                        color=colors['bs16'], label='Batch=16, LR=2e-6',
                        linewidth=1.5, alpha=0.8)
        if bs32_metrics and len(bs32_metrics.get('loss', [])) > 0:
            epochs = bs32_metrics.get('epoch', list(range(len(bs32_metrics['loss']))))
            ax_loss.plot(epochs, bs32_metrics['loss'],
                        color=colors['bs32'], label='Batch=32, LR=2.5e-6',
                        linewidth=1.5, alpha=0.8)

        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'{row_title} - Training Loss')
        ax_loss.legend(fontsize=10)
        ax_loss.grid(True, alpha=0.3)

        # Training Accuracy
        if bs16_metrics and len(bs16_metrics.get('accuracy', [])) > 0:
            epochs = bs16_metrics.get('epoch', list(range(len(bs16_metrics['accuracy']))))
            ax_acc.plot(epochs, bs16_metrics['accuracy'],
                       color=colors['bs16'], label='Batch=16, LR=2e-6',
                       linewidth=1.5, alpha=0.8)
        if bs32_metrics and len(bs32_metrics.get('accuracy', [])) > 0:
            epochs = bs32_metrics.get('epoch', list(range(len(bs32_metrics['accuracy']))))
            ax_acc.plot(epochs, bs32_metrics['accuracy'],
                       color=colors['bs32'], label='Batch=32, LR=2.5e-6',
                       linewidth=1.5, alpha=0.8)

        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title(f'{row_title} - Training Accuracy')
        ax_acc.legend(fontsize=10)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim(0.4, 1.0)

    # Row 0: Text model comparison
    plot_comparison(axes[0, 0], axes[0, 1], text_bs16_metrics, text_bs32_metrics, 'Text Model')

    # Row 1: Vision model comparison
    plot_comparison(axes[1, 0], axes[1, 1], vision_bs16_metrics, vision_bs32_metrics, 'Vision Model')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved batch comparison plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze and plot LR experiment training curves')
    parser.add_argument('--log_dir', type=str, default='logs/lr_experiment_v7',
                       help='Directory containing log files')
    parser.add_argument('--saves_dir', type=str, default='saves/lr_experiment_v7',
                       help='Directory containing trainer_log.jsonl files')
    parser.add_argument('--output', type=str, default='figures/lr_experiment_trainagreeing_curves.png',
                       help='Output figure path')
    parser.add_argument('--text_job', type=str, default=None,
                       help='Text experiment job ID (auto-detected if not specified)')
    parser.add_argument('--vision_job', type=str, default=None,
                       help='Vision experiment job ID (auto-detected if not specified)')
    parser.add_argument('--text_bs32_job', type=str, default=None,
                       help='Text batch=32 experiment job ID')
    parser.add_argument('--vision_bs32_job', type=str, default=None,
                       help='Vision batch=32 experiment job ID')
    parser.add_argument('--text_rating_job', type=str, default='4523117',
                       help='Text CLS+Rating experiment job ID')
    parser.add_argument('--vision_rating_job', type=str, default='4523571',
                       help='Vision CLS+Rating experiment job ID')
    parser.add_argument('--batch_comparison', action='store_true',
                       help='Create separate batch size comparison plot')
    parser.add_argument('--batch_comparison_output', type=str,
                       default='figures/batch_size_comparison.png',
                       help='Output path for batch comparison plot')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    saves_dir = Path(args.saves_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # LR configs mapping: task_id -> lr_config_name
    lr_configs = {
        0: 'lr_2e5',
        1: 'lr_2e6',
        2: 'lr_2e5_backbone_2e6',
    }

    # Identify job IDs
    if args.text_job and args.vision_job:
        text_job = args.text_job
        vision_job = args.vision_job
    else:
        print("Auto-detecting experiment job IDs...")
        text_job, vision_job = identify_experiment_jobs(log_dir)

    print(f"Text experiment job: {text_job}")
    print(f"Vision experiment job: {vision_job}")

    # Parse text experiment logs
    text_metrics = {}
    if text_job:
        text_log_files = find_log_files(log_dir, text_job)
        print(f"Found text log files: {text_log_files}")
        for task_id, log_file in text_log_files.items():
            lr_config = lr_configs.get(task_id, f'task_{task_id}')
            print(f"  Parsing {log_file.name} -> {lr_config}")
            text_metrics[lr_config] = parse_training_log_from_output(log_file)
            m = text_metrics[lr_config]
            print(f"    Found {len(m['loss'])} training points, {len(m['eval_loss'])} eval points")
            if m['sft_accuracy']:
                print(f"    SFT metrics available: sft_accuracy, sft_p_correct_mean")

    # Parse text batch=32 experiment
    if args.text_bs32_job:
        text_bs32_files = find_log_files(log_dir, args.text_bs32_job)
        print(f"Found text batch=32 log files: {text_bs32_files}")
        for task_id, log_file in text_bs32_files.items():
            lr_config = 'lr_2.5e6_bs32'
            print(f"  Parsing {log_file.name} -> {lr_config}")
            text_metrics[lr_config] = parse_training_log_from_output(log_file)
            m = text_metrics[lr_config]
            print(f"    Found {len(m['loss'])} training points, {len(m['eval_loss'])} eval points")

    # Parse vision experiment logs
    vision_metrics = {}
    if vision_job:
        vision_log_files = find_log_files(log_dir, vision_job)
        print(f"Found vision log files: {vision_log_files}")
        for task_id, log_file in vision_log_files.items():
            lr_config = lr_configs.get(task_id, f'task_{task_id}')
            print(f"  Parsing {log_file.name} -> {lr_config}")
            vision_metrics[lr_config] = parse_training_log_from_output(log_file)
            m = vision_metrics[lr_config]
            print(f"    Found {len(m['loss'])} training points, {len(m['eval_loss'])} eval points")
            if m['sft_accuracy']:
                print(f"    SFT metrics available: sft_accuracy, sft_p_correct_mean")

    # Parse vision batch=32 experiment
    if args.vision_bs32_job:
        vision_bs32_files = find_log_files(log_dir, args.vision_bs32_job)
        print(f"Found vision batch=32 log files: {vision_bs32_files}")
        for task_id, log_file in vision_bs32_files.items():
            lr_config = 'lr_2.5e6_bs32'
            print(f"  Parsing {log_file.name} -> {lr_config}")
            vision_metrics[lr_config] = parse_training_log_from_output(log_file)
            m = vision_metrics[lr_config]
            print(f"    Found {len(m['loss'])} training points, {len(m['eval_loss'])} eval points")

    # Parse text rating experiment
    if args.text_rating_job:
        text_rating_files = find_log_files(log_dir, args.text_rating_job)
        print(f"Found text rating log files: {text_rating_files}")
        for task_id, log_file in text_rating_files.items():
            lr_config = 'lr_rating'
            print(f"  Parsing {log_file.name} -> {lr_config}")
            text_metrics[lr_config] = parse_training_log_from_output(log_file)
            m = text_metrics[lr_config]
            print(f"    Found {len(m['loss'])} training points, {len(m['eval_loss'])} eval points")

    # Parse vision rating experiment
    if args.vision_rating_job:
        vision_rating_files = find_log_files(log_dir, args.vision_rating_job)
        print(f"Found vision rating log files: {vision_rating_files}")
        for task_id, log_file in vision_rating_files.items():
            lr_config = 'lr_rating'
            print(f"  Parsing {log_file.name} -> {lr_config}")
            vision_metrics[lr_config] = parse_training_log_from_output(log_file)
            m = vision_metrics[lr_config]
            print(f"    Found {len(m['loss'])} training points, {len(m['eval_loss'])} eval points")

    # Create plots
    if text_metrics or vision_metrics:
        create_plots(text_metrics, vision_metrics, output_path)
    else:
        print("No metrics found to plot!")

    # Create batch comparison plot if requested
    if args.batch_comparison:
        # Use lr_2e6 as the batch=16 baseline (closest LR to batch=32's 2.5e-6)
        text_bs16 = text_metrics.get('lr_2e6', {})
        text_bs32 = text_metrics.get('lr_2.5e6_bs32', {})
        vision_bs16 = vision_metrics.get('lr_2e6', {})
        vision_bs32 = vision_metrics.get('lr_2.5e6_bs32', {})

        if text_bs16 or text_bs32 or vision_bs16 or vision_bs32:
            batch_output = Path(args.batch_comparison_output)
            batch_output.parent.mkdir(parents=True, exist_ok=True)
            create_batch_comparison_plots(
                text_bs16, text_bs32,
                vision_bs16, vision_bs32,
                batch_output
            )
        else:
            print("No batch comparison data found!")


if __name__ == '__main__':
    main()
