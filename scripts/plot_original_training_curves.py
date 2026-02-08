#!/usr/bin/env python3
"""
Plot training curves for the original trainagreeing runs (CLS only).

This script creates a 2x3 grid showing:
- Row 1: Text models
- Row 2: Vision models
- Column 1: Training Loss
- Column 2: Training Accuracy
- Column 3: Rating Loss (for CLS+Rating models)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def parse_trainer_log(log_file: Path) -> dict:
    """Parse trainer_log.jsonl file.

    Returns dict with lists of metrics:
    - epoch, loss, cls_accuracy, cls_loss for all runs
    - loss_bce, loss_rating for CLS+Rating runs
    """
    metrics = {
        'epoch': [],
        'loss': [],
        'cls_accuracy': [],
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
                metrics['cls_accuracy'].append(data.get('cls_accuracy', 0))
                metrics['cls_loss'].append(data.get('cls_loss', 0))

                # Rating-specific metrics
                if 'loss_bce' in data:
                    metrics['loss_bce'].append(data['loss_bce'])
                if 'loss_rating' in data:
                    metrics['loss_rating'].append(data['loss_rating'])

            except json.JSONDecodeError:
                continue

    return metrics


def create_plot(
    text_cls_metrics: dict,
    text_cls_rating_metrics: dict,
    vision_cls_metrics: dict,
    vision_cls_rating_metrics: dict,
    output_path: Path,
):
    """Create 2x3 plot grid for original trainagreeing runs.

    Args:
        text_cls_metrics: Metrics for text CLS model
        text_cls_rating_metrics: Metrics for text CLS+Rating model
        vision_cls_metrics: Metrics for vision CLS model
        vision_cls_rating_metrics: Metrics for vision CLS+Rating model
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Color scheme
    colors = {
        'cls': '#3498db',          # Blue for CLS only
        'cls_rating': '#e74c3c',   # Red for CLS+Rating
    }

    def plot_row(ax_loss, ax_acc, ax_rating_loss, cls_metrics, cls_rating_metrics, row_title):
        """Plot one row (either text or vision)."""

        # Column 0: Training Loss
        if cls_metrics and len(cls_metrics['epoch']) > 0:
            ax_loss.plot(cls_metrics['epoch'], cls_metrics['loss'],
                        color=colors['cls'], label='CLS only',
                        linewidth=1.5, alpha=0.8)
        if cls_rating_metrics and len(cls_rating_metrics['epoch']) > 0:
            ax_loss.plot(cls_rating_metrics['epoch'], cls_rating_metrics['loss'],
                        color=colors['cls_rating'], label='CLS+Rating',
                        linewidth=1.5, alpha=0.8)

        ax_loss.set_xlabel('Epoch', fontsize=11)
        ax_loss.set_ylabel('Loss', fontsize=11)
        ax_loss.set_title(f'{row_title} - Training Loss', fontsize=12, fontweight='bold')
        ax_loss.legend(fontsize=10)
        ax_loss.grid(True, alpha=0.3)

        # Column 1: Training Accuracy
        if cls_metrics and len(cls_metrics['epoch']) > 0:
            ax_acc.plot(cls_metrics['epoch'], cls_metrics['cls_accuracy'],
                       color=colors['cls'], label='CLS only',
                       linewidth=1.5, alpha=0.8)
        if cls_rating_metrics and len(cls_rating_metrics['epoch']) > 0:
            ax_acc.plot(cls_rating_metrics['epoch'], cls_rating_metrics['cls_accuracy'],
                       color=colors['cls_rating'], label='CLS+Rating',
                       linewidth=1.5, alpha=0.8)

        ax_acc.set_xlabel('Epoch', fontsize=11)
        ax_acc.set_ylabel('CLS Accuracy', fontsize=11)
        ax_acc.set_title(f'{row_title} - Training Accuracy', fontsize=12, fontweight='bold')
        ax_acc.legend(fontsize=10)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim(0.4, 1.0)

        # Column 2: Rating Loss (only for CLS+Rating models)
        if cls_rating_metrics and len(cls_rating_metrics['epoch']) > 0 and cls_rating_metrics['loss_rating']:
            ax_rating_loss.plot(cls_rating_metrics['epoch'], cls_rating_metrics['loss_rating'],
                               color=colors['cls_rating'], label='Rating Loss',
                               linewidth=1.5, alpha=0.8)
            ax_rating_loss.set_xlabel('Epoch', fontsize=11)
            ax_rating_loss.set_ylabel('Rating Loss', fontsize=11)
            ax_rating_loss.set_title(f'{row_title} - Rating Loss', fontsize=12, fontweight='bold')
            ax_rating_loss.legend(fontsize=10)
            ax_rating_loss.grid(True, alpha=0.3)
        else:
            # Leave empty for CLS-only models
            ax_rating_loss.text(0.5, 0.5, 'N/A for CLS-only',
                              ha='center', va='center',
                              fontsize=14, color='gray',
                              transform=ax_rating_loss.transAxes)
            ax_rating_loss.set_title(f'{row_title} - Rating Loss', fontsize=12, fontweight='bold')
            ax_rating_loss.set_xticks([])
            ax_rating_loss.set_yticks([])

    # Row 0: Text models
    plot_row(axes[0, 0], axes[0, 1], axes[0, 2],
             text_cls_metrics, text_cls_rating_metrics, 'Text Model')

    # Row 1: Vision models
    plot_row(axes[1, 0], axes[1, 1], axes[1, 2],
             vision_cls_metrics, vision_cls_rating_metrics, 'Vision Model')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot training curves for original trainagreeing runs'
    )
    parser.add_argument('--saves_dir', type=str, default='saves/lr_experiment_v7',
                       help='Directory containing save directories')
    parser.add_argument('--output', type=str,
                       default='figures/original_trainagreeing_training_curves.png',
                       help='Output figure path')
    args = parser.parse_args()

    saves_dir = Path(args.saves_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Job directory mappings
    jobs = {
        'text_cls': 'trainagreeing_original_no2024_text_cls',
        'text_cls_rating': 'trainagreeing_original_no2024_text_cls_rating',
        'vision_cls': 'trainagreeing_original_no2024_vision_cls',
        'vision_cls_rating': 'trainagreeing_original_no2024_vision_cls_rating',
    }

    # Parse all logs
    print("Parsing trainer logs...")
    metrics = {}
    for job_name, job_dir in jobs.items():
        log_file = saves_dir / job_dir / 'trainer_log.jsonl'
        print(f"  {job_name}: {log_file}")
        metrics[job_name] = parse_trainer_log(log_file)
        m = metrics[job_name]
        print(f"    Found {len(m['loss'])} training points")
        if m['loss_rating']:
            print(f"    Found {len(m['loss_rating'])} rating loss points")

    # Create plot
    create_plot(
        text_cls_metrics=metrics['text_cls'],
        text_cls_rating_metrics=metrics['text_cls_rating'],
        vision_cls_metrics=metrics['vision_cls'],
        vision_cls_rating_metrics=metrics['vision_cls_rating'],
        output_path=output_path,
    )


if __name__ == '__main__':
    main()
