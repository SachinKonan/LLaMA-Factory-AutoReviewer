#!/usr/bin/env python3
"""
Create comprehensive 4x4 grid plot for original trainagreeing experiments.

Layout:
- Row 0: Text SFT (job 4524593)
- Row 1: Vision SFT (job 4524596)
- Row 2: Text CLS (jobs 4524591 CLS, 4524592 CLS+Rating)
- Row 3: Vision CLS (jobs 4524594 CLS, 4524595 CLS+Rating)

Columns:
- Column 0: Training Loss
- Column 1: Training Accuracy
- Column 2: Test/Eval Loss
- Column 3: Test/Eval Accuracy
"""

import json
import ast
import re
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def parse_trainer_log(log_file: Path) -> dict:
    """Parse trainer_log.jsonl file for training metrics.

    Returns dict with lists of:
    - epoch, loss, accuracy (cls_accuracy or sft_accuracy)
    - cls_loss, loss_bce, loss_rating (for rating jobs)
    """
    metrics = {
        'epoch': [],
        'loss': [],
        'accuracy': [],
        'cls_loss': [],
        'loss_bce': [],
        'loss_rating': [],
    }

    if not log_file.exists():
        print(f"Warning: {log_file} does not exist")
        return metrics

    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Skip lines without loss (like initial eval lines)
                if 'loss' not in data:
                    continue

                metrics['epoch'].append(data.get('epoch', 0))
                metrics['loss'].append(data['loss'])

                # Get accuracy (try both cls_accuracy and sft_accuracy)
                accuracy = data.get('cls_accuracy') or data.get('sft_accuracy', 0)
                metrics['accuracy'].append(accuracy)

                # Rating-specific decomposed losses
                if 'cls_loss' in data:
                    metrics['cls_loss'].append(data['cls_loss'])
                if 'loss_bce' in data:
                    metrics['loss_bce'].append(data['loss_bce'])
                if 'loss_rating' in data:
                    metrics['loss_rating'].append(data['loss_rating'])

            except json.JSONDecodeError:
                continue

    return metrics


def parse_output_log_for_eval(log_file: Path) -> dict:
    """Parse .out log file for evaluation metrics at epoch boundaries.

    Returns dict with lists of:
    - eval_epoch, eval_accuracy, eval_f1

    Note: eval_loss is not logged in these .out files
    """
    metrics = {
        'eval_epoch': [],
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

            # Eval metrics (has 'eval_accuracy')
            if 'eval_accuracy' in data and 'eval_runtime' in data:
                metrics['eval_epoch'].append(data.get('epoch', 0))
                metrics['eval_accuracy'].append(data.get('eval_accuracy', 0))
                metrics['eval_f1'].append(data.get('eval_f1', 0))

    return metrics


def create_comprehensive_plot(
    text_sft_train: dict,
    text_sft_eval: dict,
    vision_sft_train: dict,
    vision_sft_eval: dict,
    text_cls_train: dict,
    text_cls_eval: dict,
    text_cls_rating_train: dict,
    text_cls_rating_eval: dict,
    vision_cls_train: dict,
    vision_cls_eval: dict,
    vision_cls_rating_train: dict,
    vision_cls_rating_eval: dict,
    output_path: Path,
):
    """Create 4x3 comprehensive grid plot.

    Args:
        *_train: Training metrics dicts
        *_eval: Eval metrics dicts
        output_path: Path to save figure

    Note: Only 3 columns (train loss, train acc, eval acc) because eval_loss
    is not logged by LLaMA-Factory for these runs.
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 18))

    # Color scheme
    colors = {
        'sft': '#2ecc71',          # Green for SFT
        'cls': '#3498db',          # Blue for CLS only
        'cls_rating': '#e74c3c',   # Red for CLS+Rating
    }

    def plot_sft_row(row_idx, train_metrics, eval_metrics, row_title):
        """Plot SFT row (single job per modality)."""

        # Column 0: Training Loss
        if train_metrics and len(train_metrics['epoch']) > 0:
            axes[row_idx, 0].plot(train_metrics['epoch'], train_metrics['loss'],
                                 color=colors['sft'], label='SFT',
                                 linewidth=1.5, alpha=0.8)
        axes[row_idx, 0].set_xlabel('Epoch', fontsize=11)
        axes[row_idx, 0].set_ylabel('Loss', fontsize=11)
        axes[row_idx, 0].set_title(f'{row_title} SFT - Training Loss', fontsize=12, fontweight='bold')
        axes[row_idx, 0].legend(fontsize=9)
        axes[row_idx, 0].grid(True, alpha=0.3)

        # Column 1: Training Accuracy
        if train_metrics and len(train_metrics['epoch']) > 0:
            axes[row_idx, 1].plot(train_metrics['epoch'], train_metrics['accuracy'],
                                 color=colors['sft'], label='SFT',
                                 linewidth=1.5, alpha=0.8)
        axes[row_idx, 1].set_xlabel('Epoch', fontsize=11)
        axes[row_idx, 1].set_ylabel('Accuracy', fontsize=11)
        axes[row_idx, 1].set_title(f'{row_title} SFT - Training Accuracy', fontsize=12, fontweight='bold')
        axes[row_idx, 1].legend(fontsize=9)
        axes[row_idx, 1].grid(True, alpha=0.3)
        axes[row_idx, 1].set_ylim(0.4, 1.0)

        # Column 2: Eval Accuracy
        if eval_metrics and len(eval_metrics['eval_epoch']) > 0:
            axes[row_idx, 2].plot(eval_metrics['eval_epoch'], eval_metrics['eval_accuracy'],
                                 color=colors['sft'], label='SFT',
                                 linewidth=2, marker='o', markersize=6)
        axes[row_idx, 2].set_xlabel('Epoch', fontsize=11)
        axes[row_idx, 2].set_ylabel('Eval Accuracy', fontsize=11)
        axes[row_idx, 2].set_title(f'{row_title} SFT - Eval Accuracy', fontsize=12, fontweight='bold')
        axes[row_idx, 2].legend(fontsize=9)
        axes[row_idx, 2].grid(True, alpha=0.3)
        axes[row_idx, 2].set_ylim(0.4, 1.0)

    def plot_cls_row(row_idx, cls_train, cls_eval, cls_rating_train, cls_rating_eval, row_title):
        """Plot CLS row (comparing CLS vs CLS+Rating)."""

        # Column 0: Training Loss (with decomposed losses for rating)
        if cls_train and len(cls_train['epoch']) > 0:
            axes[row_idx, 0].plot(cls_train['epoch'], cls_train['loss'],
                                 color=colors['cls'], label='CLS',
                                 linewidth=1.5, alpha=0.8)

        if cls_rating_train and len(cls_rating_train['epoch']) > 0:
            # Combined loss (solid)
            axes[row_idx, 0].plot(cls_rating_train['epoch'], cls_rating_train['loss'],
                                 color=colors['cls_rating'], label='CLS+Rating',
                                 linewidth=1.5, alpha=0.9)
            # CLS loss component (dashed)
            if cls_rating_train['cls_loss']:
                axes[row_idx, 0].plot(cls_rating_train['epoch'], cls_rating_train['cls_loss'],
                                     color=colors['cls_rating'], label='CLS+Rating (CLS)',
                                     linewidth=1, linestyle='--', alpha=0.7)
            # Rating loss component (dotted)
            if cls_rating_train['loss_rating']:
                axes[row_idx, 0].plot(cls_rating_train['epoch'], cls_rating_train['loss_rating'],
                                     color=colors['cls_rating'], label='CLS+Rating (Rating)',
                                     linewidth=1, linestyle=':', alpha=0.7)

        axes[row_idx, 0].set_xlabel('Epoch', fontsize=11)
        axes[row_idx, 0].set_ylabel('Loss', fontsize=11)
        axes[row_idx, 0].set_title(f'{row_title} CLS - Training Loss', fontsize=12, fontweight='bold')
        axes[row_idx, 0].legend(fontsize=8)
        axes[row_idx, 0].grid(True, alpha=0.3)

        # Column 1: Training Accuracy
        if cls_train and len(cls_train['epoch']) > 0:
            axes[row_idx, 1].plot(cls_train['epoch'], cls_train['accuracy'],
                                 color=colors['cls'], label='CLS',
                                 linewidth=1.5, alpha=0.8)
        if cls_rating_train and len(cls_rating_train['epoch']) > 0:
            axes[row_idx, 1].plot(cls_rating_train['epoch'], cls_rating_train['accuracy'],
                                 color=colors['cls_rating'], label='CLS+Rating',
                                 linewidth=1.5, alpha=0.8)

        axes[row_idx, 1].set_xlabel('Epoch', fontsize=11)
        axes[row_idx, 1].set_ylabel('CLS Accuracy', fontsize=11)
        axes[row_idx, 1].set_title(f'{row_title} CLS - Training Accuracy', fontsize=12, fontweight='bold')
        axes[row_idx, 1].legend(fontsize=9)
        axes[row_idx, 1].grid(True, alpha=0.3)
        axes[row_idx, 1].set_ylim(0.4, 1.0)

        # Column 2: Eval Accuracy
        if cls_eval and len(cls_eval['eval_epoch']) > 0:
            axes[row_idx, 2].plot(cls_eval['eval_epoch'], cls_eval['eval_accuracy'],
                                 color=colors['cls'], label='CLS',
                                 linewidth=2, marker='o', markersize=6)
        if cls_rating_eval and len(cls_rating_eval['eval_epoch']) > 0:
            axes[row_idx, 2].plot(cls_rating_eval['eval_epoch'], cls_rating_eval['eval_accuracy'],
                                 color=colors['cls_rating'], label='CLS+Rating',
                                 linewidth=2, marker='o', markersize=6)

        axes[row_idx, 2].set_xlabel('Epoch', fontsize=11)
        axes[row_idx, 2].set_ylabel('Eval Accuracy', fontsize=11)
        axes[row_idx, 2].set_title(f'{row_title} CLS - Eval Accuracy', fontsize=12, fontweight='bold')
        axes[row_idx, 2].legend(fontsize=9)
        axes[row_idx, 2].grid(True, alpha=0.3)
        axes[row_idx, 2].set_ylim(0.4, 1.0)

    # Row 0: Text SFT
    plot_sft_row(0, text_sft_train, text_sft_eval, 'Text')

    # Row 1: Vision SFT
    plot_sft_row(1, vision_sft_train, vision_sft_eval, 'Vision')

    # Row 2: Text CLS
    plot_cls_row(2, text_cls_train, text_cls_eval,
                 text_cls_rating_train, text_cls_rating_eval, 'Text')

    # Row 3: Vision CLS
    plot_cls_row(3, vision_cls_train, vision_cls_eval,
                 vision_cls_rating_train, vision_cls_rating_eval, 'Vision')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create comprehensive 4x4 plot for original trainagreeing experiments'
    )
    parser.add_argument('--saves_dir', type=str, default='saves/lr_experiment_v7',
                       help='Directory containing save directories with trainer_log.jsonl')
    parser.add_argument('--logs_dir', type=str, default='logs/lr_experiment_v7',
                       help='Directory containing .out log files')
    parser.add_argument('--output', type=str,
                       default='figures/original_trainagreeing_comprehensive.png',
                       help='Output figure path')
    args = parser.parse_args()

    saves_dir = Path(args.saves_dir)
    logs_dir = Path(args.logs_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Job directory mappings
    jobs = {
        'text_sft': {
            'save_dir': 'trainagreeing_original_no2024_text_sft',
            'log_file': '4524593.out',
        },
        'vision_sft': {
            'save_dir': 'trainagreeing_original_no2024_vision_sft',
            'log_file': '4524596.out',
        },
        'text_cls': {
            'save_dir': 'trainagreeing_original_no2024_text_cls',
            'log_file': '4524591.out',
        },
        'text_cls_rating': {
            'save_dir': 'trainagreeing_original_no2024_text_cls_rating',
            'log_file': '4524592.out',
        },
        'vision_cls': {
            'save_dir': 'trainagreeing_original_no2024_vision_cls',
            'log_file': '4524594.out',
        },
        'vision_cls_rating': {
            'save_dir': 'trainagreeing_original_no2024_vision_cls_rating',
            'log_file': '4524595.out',
        },
    }

    # Parse all training and eval metrics
    print("Parsing training and eval logs...")
    train_metrics = {}
    eval_metrics = {}

    for job_name, job_info in jobs.items():
        # Parse training metrics from trainer_log.jsonl
        trainer_log = saves_dir / job_info['save_dir'] / 'trainer_log.jsonl'
        print(f"  {job_name} training: {trainer_log}")
        train_metrics[job_name] = parse_trainer_log(trainer_log)
        print(f"    Found {len(train_metrics[job_name]['loss'])} training points")

        # Parse eval metrics from .out file
        out_log = logs_dir / job_info['log_file']
        print(f"  {job_name} eval: {out_log}")
        eval_metrics[job_name] = parse_output_log_for_eval(out_log)
        print(f"    Found {len(eval_metrics[job_name]['eval_loss'])} eval points")

    # Create comprehensive plot
    create_comprehensive_plot(
        text_sft_train=train_metrics['text_sft'],
        text_sft_eval=eval_metrics['text_sft'],
        vision_sft_train=train_metrics['vision_sft'],
        vision_sft_eval=eval_metrics['vision_sft'],
        text_cls_train=train_metrics['text_cls'],
        text_cls_eval=eval_metrics['text_cls'],
        text_cls_rating_train=train_metrics['text_cls_rating'],
        text_cls_rating_eval=eval_metrics['text_cls_rating'],
        vision_cls_train=train_metrics['vision_cls'],
        vision_cls_eval=eval_metrics['vision_cls'],
        vision_cls_rating_train=train_metrics['vision_cls_rating'],
        vision_cls_rating_eval=eval_metrics['vision_cls_rating'],
        output_path=output_path,
    )


if __name__ == '__main__':
    main()
