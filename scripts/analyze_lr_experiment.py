#!/usr/bin/env python3
"""
Analyze learning rate experiment results.

This script compares the performance of different learning rate configurations
for both text and vision models on the trainagreeing dataset.

Usage:
    python scripts/analyze_lr_experiment.py
    python scripts/analyze_lr_experiment.py --output results/lr_experiment_analysis.csv
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def load_results(results_dir: Path) -> Dict[str, Dict]:
    """Load prediction results from all experiment directories."""
    results = {}

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return results

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        predict_file = exp_dir / "predict_results.json"

        if predict_file.exists():
            try:
                with open(predict_file, 'r') as f:
                    results[exp_name] = json.load(f)
                print(f"Loaded: {exp_name}")
            except Exception as e:
                print(f"Error loading {exp_name}: {e}")
        else:
            print(f"No results yet: {exp_name}")

    return results


def parse_experiment_name(exp_name: str) -> Dict[str, str]:
    """Parse experiment name to extract model type and LR config."""
    parts = exp_name.split('_')

    if exp_name.startswith('text_'):
        model_type = 'Text'
        lr_config = '_'.join(parts[2:])  # After 'text_trainagreeing_'
    elif exp_name.startswith('vision_'):
        model_type = 'Vision'
        lr_config = '_'.join(parts[2:])  # After 'vision_trainagreeing_'
    else:
        model_type = 'Unknown'
        lr_config = exp_name

    # Format LR config for display
    if lr_config == 'lr_2e5':
        lr_display = 'LR=2e-5 (uniform)'
    elif lr_config == 'lr_2e6':
        lr_display = 'LR=2e-6 (uniform)'
    elif lr_config == 'lr_2e5_backbone_2e6':
        lr_display = 'LR=2e-5 head, 2e-6 backbone'
    else:
        lr_display = lr_config

    return {
        'model_type': model_type,
        'lr_config': lr_config,
        'lr_display': lr_display
    }


def extract_metrics(results: Dict) -> Dict[str, float]:
    """Extract key metrics from prediction results."""
    metrics = {}

    # Standard metrics
    for key in ['predict_accuracy', 'predict_f1', 'predict_precision',
                'predict_recall', 'predict_loss']:
        if key in results:
            metrics[key.replace('predict_', '')] = results[key]

    # Runtime metrics
    if 'predict_runtime' in results:
        metrics['runtime'] = results['predict_runtime']
    if 'predict_samples_per_second' in results:
        metrics['samples_per_sec'] = results['predict_samples_per_second']

    return metrics


def print_summary_table(results: Dict[str, Dict]):
    """Print a formatted summary table."""
    if not results:
        print("\nNo results available yet.")
        return

    # Group by model type
    text_results = {k: v for k, v in results.items() if k.startswith('text_')}
    vision_results = {k: v for k, v in results.items() if k.startswith('vision_')}

    print("\n" + "="*80)
    print("LEARNING RATE EXPERIMENT RESULTS")
    print("="*80)

    for model_name, model_results in [("TEXT MODEL", text_results),
                                       ("VISION MODEL", vision_results)]:
        if not model_results:
            continue

        print(f"\n{model_name}")
        print("-" * 80)
        print(f"{'Configuration':<35} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 80)

        for exp_name in sorted(model_results.keys()):
            parsed = parse_experiment_name(exp_name)
            metrics = extract_metrics(model_results[exp_name])

            acc = metrics.get('accuracy', 0) * 100
            f1 = metrics.get('f1', 0) * 100
            prec = metrics.get('precision', 0) * 100
            rec = metrics.get('recall', 0) * 100

            print(f"{parsed['lr_display']:<35} {acc:>9.2f}% {f1:>9.2f}% {prec:>9.2f}% {rec:>9.2f}%")

    print("\n" + "="*80)


def save_csv(results: Dict[str, Dict], output_file: Path):
    """Save results to CSV file."""
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model Type', 'LR Config', 'LR Display', 'Accuracy', 'F1',
            'Precision', 'Recall', 'Loss', 'Runtime', 'Samples/sec'
        ])

        for exp_name, exp_results in sorted(results.items()):
            parsed = parse_experiment_name(exp_name)
            metrics = extract_metrics(exp_results)

            writer.writerow([
                parsed['model_type'],
                parsed['lr_config'],
                parsed['lr_display'],
                metrics.get('accuracy', ''),
                metrics.get('f1', ''),
                metrics.get('precision', ''),
                metrics.get('recall', ''),
                metrics.get('loss', ''),
                metrics.get('runtime', ''),
                metrics.get('samples_per_sec', '')
            ])

    print(f"\nResults saved to: {output_file}")


def find_best_config(results: Dict[str, Dict]) -> Dict[str, tuple]:
    """Find best configuration for each model type."""
    best = {'text': None, 'vision': None}

    for exp_name, exp_results in results.items():
        parsed = parse_experiment_name(exp_name)
        metrics = extract_metrics(exp_results)

        model_key = parsed['model_type'].lower()
        if model_key not in best:
            continue

        acc = metrics.get('accuracy', 0)

        if best[model_key] is None or acc > best[model_key][1]:
            best[model_key] = (parsed['lr_display'], acc, metrics.get('f1', 0))

    return best


def main():
    parser = argparse.ArgumentParser(description='Analyze LR experiment results')
    parser.add_argument('--results-dir', type=Path,
                        default=Path('results/lr_experiment_v7'),
                        help='Directory containing experiment results')
    parser.add_argument('--output', type=Path,
                        help='Output CSV file path')
    args = parser.parse_args()

    # Load results
    results = load_results(args.results_dir)

    if not results:
        print("\nNo results found. Experiments may still be running.")
        print("Check with: squeue -u $USER")
        return

    # Print summary table
    print_summary_table(results)

    # Find best configurations
    best = find_best_config(results)
    print("\nBEST CONFIGURATIONS:")
    print("-" * 80)
    for model_type, result in best.items():
        if result:
            lr_display, acc, f1 = result
            print(f"{model_type.upper():>8}: {lr_display:<35} (Acc: {acc*100:.2f}%, F1: {f1*100:.2f}%)")

    # Save to CSV if requested
    if args.output:
        save_csv(results, args.output)

    # Additional analysis
    print("\n" + "="*80)
    print("EXPERIMENT STATUS")
    print("="*80)
    print(f"Completed experiments: {len(results)}/6")

    expected = [
        'text_trainagreeing_lr_2e5',
        'text_trainagreeing_lr_2e6',
        'text_trainagreeing_lr_2e5_backbone_2e6',
        'vision_trainagreeing_lr_2e5',
        'vision_trainagreeing_lr_2e6',
        'vision_trainagreeing_lr_2e5_backbone_2e6',
    ]

    missing = [exp for exp in expected if exp not in results]
    if missing:
        print("\nPending experiments:")
        for exp in missing:
            print(f"  - {exp}")
    else:
        print("\nAll experiments completed!")

    print("="*80 + "\n")


if __name__ == '__main__':
    main()
