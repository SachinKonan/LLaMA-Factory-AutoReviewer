#!/usr/bin/env python3
"""
Generate comparison plots for inference scaling experiments.

Creates:
1. Subplot grid: columns = modalities, rows = prompt techniques
2. Subplot grid: each subplot = modality, bars = prompt techniques

Also recalculates calibration thresholds to achieve ~50% acceptance rate.

Usage:
    python inference_scaling/scripts/plot_metrics.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Configuration
RESULTS_DIR = Path("inference_scaling/results")
METRICS_FILE = Path("inference_scaling/metrics/all_metrics.json")
OUTPUT_DIR = Path("inference_scaling/metrics")

MODALITIES = ["clean", "clean_images", "vision"]
MODALITY_LABELS = {"clean": "Text Only", "clean_images": "Text + Images", "vision": "Vision Only"}

PROMPT_TECHNIQUES = ["original", "new", "new_fewshot_single", "new_fewshot_majority",
                     "new_fewshot_calibrated", "new_fewshot_meta"]
TECHNIQUE_LABELS = {
    "original": "Original",
    "new": "New (Single)",
    "new_fewshot_single": "Few-shot (Single)",
    "new_fewshot_majority": "Few-shot (Majority)",
    "new_fewshot_calibrated": "Few-shot (Calibrated)",
    "new_fewshot_calibrated_high": "Few-shot (Calib-High)",
    "new_fewshot_calibrated_low": "Few-shot (Calib-Low)",
    "new_fewshot_meta": "Few-shot (Meta)",
}


def parse_json_decision(text: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Extract decision and full review from JSON output."""
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            review = json.loads(json_match.group(1))
            decision = review.get("decision", "").lower()
            if decision in ["accept", "reject"]:
                return decision.capitalize(), review
            return None, review
        except json.JSONDecodeError:
            pass

    try:
        json_start = text.rfind('{')
        json_end = text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = text[json_start:json_end]
            review = json.loads(json_str)
            decision = review.get("decision", "").lower()
            if decision in ["accept", "reject"]:
                return decision.capitalize(), review
            return None, review
    except json.JSONDecodeError:
        pass

    return None, None


def parse_boxed_decision(text: str) -> Optional[str]:
    """Extract decision from \\boxed{Accept} or \\boxed{Reject} format."""
    match = re.search(r'\\boxed\{(Accept|Reject)\}', text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None


def extract_overall_scores(predictions_path: Path) -> List[Tuple[Optional[int], str]]:
    """
    Extract overall scores and ground truth from predictions file.

    Returns:
        List of (overall_score or None, ground_truth) tuples for each sample
    """
    results = []

    with open(predictions_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            predictions = data.get("predict", [])
            label = data.get("label", "")

            # Extract ground truth
            gt = parse_boxed_decision(label)
            if gt is None:
                gt, _ = parse_json_decision(label)
            if gt is None:
                if "accept" in label.lower():
                    gt = "Accept"
                elif "reject" in label.lower():
                    gt = "Reject"

            # Get all overall scores from predictions
            if isinstance(predictions, str):
                predictions = [predictions]

            scores = []
            for pred in predictions:
                _, review = parse_json_decision(pred)
                if review and "overall" in review:
                    try:
                        scores.append(int(review["overall"]))
                    except (ValueError, TypeError):
                        pass

            # Use mean score if multiple
            if scores:
                mean_score = sum(scores) / len(scores)
                results.append((mean_score, gt))
            else:
                results.append((None, gt))

    return results


def find_optimal_threshold(scores_and_labels: List[Tuple[Optional[float], str]],
                           target_accept_rate: float = 0.5) -> Tuple[int, int, float, float]:
    """
    Find thresholds that give acceptance rates closest to target (above and below).

    Returns:
        (threshold_high, threshold_low, accept_rate_high, accept_rate_low)
        threshold_high gives accept_rate >= target
        threshold_low gives accept_rate < target
    """
    valid_samples = [(s, gt) for s, gt in scores_and_labels if s is not None and gt is not None]

    if not valid_samples:
        return 6, 6, 0.5, 0.5

    total = len(valid_samples)

    best_high = (6, 1.0)  # (threshold, accept_rate) for >= target
    best_low = (6, 0.0)   # (threshold, accept_rate) for < target

    for threshold in range(1, 11):
        accept_count = sum(1 for s, _ in valid_samples if s >= threshold)
        accept_rate = accept_count / total

        if accept_rate >= target_accept_rate:
            if abs(accept_rate - target_accept_rate) < abs(best_high[1] - target_accept_rate):
                best_high = (threshold, accept_rate)
        else:
            if abs(accept_rate - target_accept_rate) < abs(best_low[1] - target_accept_rate):
                best_low = (threshold, accept_rate)

    return best_high[0], best_low[0], best_high[1], best_low[1]


def compute_calibrated_accuracy(scores_and_labels: List[Tuple[Optional[float], str]],
                                threshold: int) -> Dict:
    """Compute accuracy metrics for a given threshold."""
    valid_samples = [(s, gt) for s, gt in scores_and_labels if s is not None and gt is not None]

    if not valid_samples:
        return {"total": 0, "accuracy": 0, "accept_rate": 0}

    total = len(valid_samples)
    tp = sum(1 for s, gt in valid_samples if s >= threshold and gt == "Accept")
    tn = sum(1 for s, gt in valid_samples if s < threshold and gt == "Reject")
    fp = sum(1 for s, gt in valid_samples if s >= threshold and gt == "Reject")
    fn = sum(1 for s, gt in valid_samples if s < threshold and gt == "Accept")

    correct = tp + tn
    accept_count = tp + fp

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "accept_rate": accept_count / total if total > 0 else 0,
        "accept_recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "reject_recall": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "threshold": threshold
    }


def load_existing_metrics() -> Dict:
    """Load existing metrics from JSON file."""
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {}


def get_accuracy_for_config(metrics: Dict, modality: str, technique: str) -> Optional[float]:
    """Get accuracy for a specific modality/technique combination."""

    # Map technique to metrics key
    key_mappings = {
        "original": f"{modality}/original/single",
        "new": f"{modality}/new/single",
        "new_fewshot_single": f"{modality}/new_fewshot/single",
        "new_fewshot_majority": f"{modality}/new_fewshot/majority",
        "new_fewshot_calibrated": f"{modality}/new_fewshot/calibrated",
        "new_fewshot_meta": f"{modality}/new_fewshot/metareview",
    }

    key = key_mappings.get(technique)
    if key and key in metrics:
        return metrics[key].get("accuracy")
    return None


def recalculate_calibrated_metrics():
    """Recalculate calibration with optimal thresholds for ~50% acceptance rate."""
    results = {}

    for modality in MODALITIES:
        # Check for new_fewshot predictions
        predictions_path = RESULTS_DIR / modality / "new_fewshot" / "predictions.jsonl"
        if not predictions_path.exists():
            predictions_path = RESULTS_DIR / modality / "new" / "predictions.jsonl"

        if not predictions_path.exists():
            continue

        scores_and_labels = extract_overall_scores(predictions_path)

        # Find optimal thresholds
        thresh_high, thresh_low, rate_high, rate_low = find_optimal_threshold(scores_and_labels)

        # Compute metrics for both thresholds
        metrics_high = compute_calibrated_accuracy(scores_and_labels, thresh_high)
        metrics_low = compute_calibrated_accuracy(scores_and_labels, thresh_low)

        results[modality] = {
            "calibrated_high": {
                **metrics_high,
                "description": f"Threshold {thresh_high} -> {rate_high:.1%} accept rate"
            },
            "calibrated_low": {
                **metrics_low,
                "description": f"Threshold {thresh_low} -> {rate_low:.1%} accept rate"
            }
        }

        print(f"\n{modality}:")
        print(f"  High threshold: {thresh_high} -> {rate_high:.1%} accept, accuracy={metrics_high['accuracy']:.3f}")
        print(f"  Low threshold:  {thresh_low} -> {rate_low:.1%} accept, accuracy={metrics_low['accuracy']:.3f}")

    return results


def plot_modality_columns(metrics: Dict, calibrated: Dict, output_path: Path):
    """
    Create subplot grid: columns = modalities, each subplot shows prompt techniques.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    techniques = ["original", "new", "new_fewshot_majority", "new_fewshot_calibrated", "new_fewshot_meta"]
    technique_display = ["Original", "New", "Majority", "Calibrated", "Meta-review"]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    for idx, modality in enumerate(MODALITIES):
        ax = axes[idx]

        accuracies = []
        labels = []

        for tech, display in zip(techniques, technique_display):
            acc = get_accuracy_for_config(metrics, modality, tech)

            # For calibrated, use recalculated value if available
            if tech == "new_fewshot_calibrated" and modality in calibrated:
                # Use the one closest to 50%
                high_acc = calibrated[modality]["calibrated_high"]["accuracy"]
                low_acc = calibrated[modality]["calibrated_low"]["accuracy"]
                high_rate = calibrated[modality]["calibrated_high"]["accept_rate"]
                low_rate = calibrated[modality]["calibrated_low"]["accept_rate"]

                # Choose the one with accept rate closest to 50%
                if abs(high_rate - 0.5) < abs(low_rate - 0.5):
                    acc = high_acc
                else:
                    acc = low_acc

            if acc is not None:
                accuracies.append(acc)
                labels.append(display)

        if accuracies:
            x = np.arange(len(labels))
            bars = ax.bar(x, accuracies, color=colors[:len(labels)], edgecolor='black', linewidth=0.5)

            # Add value labels
            for bar, acc in zip(bars, accuracies):
                ax.annotate(f'{acc:.2f}', xy=(bar.get_x() + bar.get_width()/2, acc),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_title(MODALITY_LABELS.get(modality, modality), fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.grid(axis='y', alpha=0.3)

        if idx == 0:
            ax.set_ylabel('Accuracy', fontsize=11)

    plt.suptitle('Accuracy by Modality and Prompting Technique', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_technique_subplots(metrics: Dict, calibrated: Dict, output_path: Path):
    """
    Create subplot grid: each subplot = prompting technique, bars = modalities.
    """
    techniques = ["original", "new", "new_fewshot_majority", "new_fewshot_calibrated", "new_fewshot_meta"]
    technique_titles = ["Original Prompt", "New Prompt", "Few-shot (Majority)",
                        "Few-shot (Calibrated)", "Few-shot (Meta-review)"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    colors = {'clean': '#3498db', 'clean_images': '#2ecc71', 'vision': '#e74c3c'}
    modality_display = ['Text Only', 'Text+Images', 'Vision Only']

    for idx, (tech, title) in enumerate(zip(techniques, technique_titles)):
        ax = axes[idx]

        accuracies = []
        for modality in MODALITIES:
            acc = get_accuracy_for_config(metrics, modality, tech)

            # For calibrated, use recalculated value
            if tech == "new_fewshot_calibrated" and modality in calibrated:
                high_acc = calibrated[modality]["calibrated_high"]["accuracy"]
                low_acc = calibrated[modality]["calibrated_low"]["accuracy"]
                high_rate = calibrated[modality]["calibrated_high"]["accept_rate"]
                low_rate = calibrated[modality]["calibrated_low"]["accept_rate"]

                if abs(high_rate - 0.5) < abs(low_rate - 0.5):
                    acc = high_acc
                else:
                    acc = low_acc

            accuracies.append(acc if acc is not None else 0)

        x = np.arange(len(MODALITIES))
        bar_colors = [colors[m] for m in MODALITIES]
        bars = ax.bar(x, accuracies, color=bar_colors, edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.annotate(f'{acc:.2f}', xy=(bar.get_x() + bar.get_width()/2, acc),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(modality_display, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('Accuracy', fontsize=10)

    # Hide the 6th subplot
    axes[5].axis('off')

    # Add legend to the empty subplot area
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=colors[m], edgecolor='black',
                                      label=MODALITY_LABELS[m]) for m in MODALITIES]
    axes[5].legend(handles=legend_elements, loc='center', fontsize=11, frameon=True)
    axes[5].set_title('Legend', fontsize=11, fontweight='bold')

    plt.suptitle('Accuracy by Prompting Technique and Modality', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_calibration_comparison(metrics: Dict, calibrated: Dict, output_path: Path):
    """
    Plot showing both high and low calibration thresholds for comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, modality in enumerate(MODALITIES):
        ax = axes[idx]

        if modality not in calibrated:
            ax.text(0.5, 0.5, 'No calibration data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(MODALITY_LABELS.get(modality, modality))
            continue

        cal_data = calibrated[modality]

        labels = ['Calib-High', 'Calib-Low']
        accuracies = [cal_data['calibrated_high']['accuracy'], cal_data['calibrated_low']['accuracy']]
        accept_rates = [cal_data['calibrated_high']['accept_rate'], cal_data['calibrated_low']['accept_rate']]
        thresholds = [cal_data['calibrated_high']['threshold'], cal_data['calibrated_low']['threshold']]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db', edgecolor='black')
        bars2 = ax.bar(x + width/2, accept_rates, width, label='Accept Rate', color='#e74c3c', edgecolor='black')

        # Add value labels
        for bar, val in zip(bars1, accuracies):
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, accept_rates):
            ax.annotate(f'{val:.0%}', xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{l}\n(T={t})' for l, t in zip(labels, thresholds)], fontsize=10)
        ax.set_title(MODALITY_LABELS.get(modality, modality), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        if idx == 0:
            ax.set_ylabel('Value', fontsize=11)

    plt.suptitle('Calibration Threshold Comparison (T = threshold for Accept)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comprehensive_comparison(metrics: Dict, output_path: Path):
    """
    Create comprehensive comparison: modalities as columns, techniques as bars.
    Excludes calibrated since score clustering makes it meaningless.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    # Only include meaningful techniques
    techniques = ["original", "new", "new_fewshot_single", "new_fewshot_majority", "new_fewshot_meta"]
    technique_display = ["Original", "New", "Single", "Majority", "Meta-review"]
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6']

    for idx, modality in enumerate(MODALITIES):
        ax = axes[idx]

        accuracies = []
        labels = []
        bar_colors = []

        for tech, display, color in zip(techniques, technique_display, colors):
            acc = get_accuracy_for_config(metrics, modality, tech)
            if acc is not None and acc > 0.1:  # Filter out missing/invalid data
                accuracies.append(acc)
                labels.append(display)
                bar_colors.append(color)

        if accuracies:
            x = np.arange(len(labels))
            bars = ax.bar(x, accuracies, color=bar_colors, edgecolor='black', linewidth=0.5, width=0.7)

            # Add value labels
            for bar, acc in zip(bars, accuracies):
                ax.annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width()/2, acc),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.set_title(MODALITY_LABELS.get(modality, modality), fontsize=13, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Random (50%)')
            ax.grid(axis='y', alpha=0.3)

        if idx == 0:
            ax.set_ylabel('Accuracy', fontsize=12)

    plt.suptitle('Accuracy Comparison: Modalities × Prompting Techniques\n(Qwen2.5-VL-7B)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_technique_focused(metrics: Dict, output_path: Path):
    """
    Create subplots for each technique showing performance across modalities.
    """
    techniques = [
        ("original", "Original Prompt"),
        ("new", "New Prompt (JSON)"),
        ("new_fewshot_single", "Few-shot: Single Response"),
        ("new_fewshot_majority", "Few-shot: Majority Vote (5)"),
        ("new_fewshot_meta", "Few-shot: Meta-review (5→1)")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = {'clean': '#3498db', 'clean_images': '#2ecc71', 'vision': '#e74c3c'}
    modality_display = ['Text Only', 'Text+Images', 'Vision Only']

    for idx, (tech, title) in enumerate(techniques):
        ax = axes[idx]

        accuracies = []
        valid_modalities = []
        bar_colors = []

        for modality, display in zip(MODALITIES, modality_display):
            acc = get_accuracy_for_config(metrics, modality, tech)
            if acc is not None and acc > 0.1:  # Filter invalid
                accuracies.append(acc)
                valid_modalities.append(display)
                bar_colors.append(colors[modality])

        if accuracies:
            x = np.arange(len(valid_modalities))
            bars = ax.bar(x, accuracies, color=bar_colors, edgecolor='black', linewidth=0.5, width=0.6)

            for bar, acc in zip(bars, accuracies):
                ax.annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width()/2, acc),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(valid_modalities, fontsize=10)
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.grid(axis='y', alpha=0.3)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=10)

    # Use last subplot for legend
    axes[5].axis('off')
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=colors[m], edgecolor='black',
                                      label=MODALITY_LABELS[m]) for m in MODALITIES]
    axes[5].legend(handles=legend_elements, loc='center', fontsize=12, frameon=True, title='Modality')

    plt.suptitle('Performance by Prompting Technique\n(Red dashed line = random baseline at 50%)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def load_json_metrics() -> Dict:
    """Load JSON validity metrics."""
    json_metrics_path = OUTPUT_DIR / "json_metrics.json"
    if json_metrics_path.exists():
        with open(json_metrics_path, "r") as f:
            return json.load(f)
    return {}


def plot_validity_comparison(json_metrics: Dict, output_path: Path):
    """
    Plot format validity rates by modality and prompt variant.
    Original uses boxed format, new/new_fewshot use JSON format.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Organize data by modality and prompt
    modalities = ["clean", "clean_images", "vision"]
    prompts = ["original", "new", "new_fewshot"]
    prompt_labels = {"original": "Original (boxed)", "new": "New (JSON)", "new_fewshot": "Few-shot (JSON)"}

    x = np.arange(len(modalities))
    width = 0.25

    colors = ['#95a5a6', '#3498db', '#e74c3c']

    for i, prompt in enumerate(prompts):
        validity_rates = []
        for modality in modalities:
            key = f"{modality}/{prompt}"
            if key in json_metrics:
                validity_rates.append(json_metrics[key].get("json_validity_rate", 0))
            else:
                validity_rates.append(0)

        offset = (i - 1) * width  # Center the 3 bars
        bars = ax.bar(x + offset, validity_rates, width, label=prompt_labels[prompt],
                     color=colors[i], edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, rate in zip(bars, validity_rates):
            if rate > 0:
                ax.annotate(f'{rate:.1%}', xy=(bar.get_x() + bar.get_width()/2, rate),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Modality', fontsize=12)
    ax.set_ylabel('Format Validity Rate', fontsize=12)
    ax.set_title('Format Validity Rate by Modality and Prompt Variant\n(Original=boxed, New/Few-shot=JSON)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODALITY_LABELS.get(m, m) for m in modalities], fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_validity_single_subplot(json_metrics: Dict, output_path: Path):
    """
    Plot format validity rates as a single bar chart with modality groupings.
    Shows all modality/prompt combinations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    prompts = ["original", "new", "new_fewshot"]
    prompt_labels = ["Original\n(boxed)", "New\n(JSON)", "Few-shot\n(JSON)"]
    colors = ['#95a5a6', '#3498db', '#e74c3c']

    for idx, modality in enumerate(MODALITIES):
        ax = axes[idx]

        validity_rates = []
        labels = []

        for prompt, label in zip(prompts, prompt_labels):
            key = f"{modality}/{prompt}"
            if key in json_metrics:
                rate = json_metrics[key].get("json_validity_rate", 0)
                n = json_metrics[key].get("total_predictions", 0)
                validity_rates.append(rate)
                labels.append(f"{label}\n(n={n})")
            else:
                validity_rates.append(0)
                labels.append(f"{label}\n(n=0)")

        x = np.arange(len(labels))
        bars = ax.bar(x, validity_rates, color=colors, edgecolor='black', linewidth=0.5)

        for bar, rate in zip(bars, validity_rates):
            if rate > 0:
                ax.annotate(f'{rate:.1%}', xy=(bar.get_x() + bar.get_width()/2, rate),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(MODALITY_LABELS.get(modality, modality), fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        if idx == 0:
            ax.set_ylabel('Format Validity Rate', fontsize=11)

    plt.suptitle('Format Validity Rate: Boxed vs JSON Output\n'
                 '(Few-shot generates 5 responses per sample)',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def extract_voting_distributions(predictions_path: Path) -> List[float]:
    """
    Extract voting winning percentages from few-shot predictions.

    For each sample with 5 predictions, calculate the winning percentage:
    - 5-0 = 100% (5/5)
    - 4-1 = 80% (4/5)
    - 3-2 = 60% (3/5)

    Returns:
        List of winning percentages (0.6, 0.8, or 1.0)
    """
    winning_percentages = []

    with open(predictions_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            predictions = data.get("predict", [])
            if isinstance(predictions, str):
                predictions = [predictions]

            if len(predictions) < 2:
                continue  # Need multiple predictions for voting

            # Extract decisions from each prediction
            decisions = []
            for pred in predictions:
                # Try JSON format first
                decision, review = parse_json_decision(pred)
                if decision is None:
                    # Try boxed format
                    decision = parse_boxed_decision(pred)
                if decision:
                    decisions.append(decision)

            if len(decisions) >= 2:
                # Count votes
                accept_count = sum(1 for d in decisions if d == "Accept")
                reject_count = len(decisions) - accept_count

                # Winning percentage
                majority_count = max(accept_count, reject_count)
                winning_pct = majority_count / len(decisions)
                winning_percentages.append(winning_pct)

    return winning_percentages


def plot_voting_distribution(output_path: Path):
    """
    Plot density/histogram of voting distributions for few-shot predictions.
    X-axis is "winning percentage" (60%, 80%, 100% for 3-2, 4-1, 5-0).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    colors = {'clean': '#3498db', 'clean_images': '#2ecc71', 'vision': '#e74c3c'}

    for idx, modality in enumerate(MODALITIES):
        ax = axes[idx]

        predictions_path = RESULTS_DIR / modality / "new_fewshot" / "predictions.jsonl"
        if not predictions_path.exists():
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(MODALITY_LABELS.get(modality, modality), fontsize=12, fontweight='bold')
            continue

        winning_pcts = extract_voting_distributions(predictions_path)

        if not winning_pcts:
            ax.text(0.5, 0.5, 'No valid votes', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(MODALITY_LABELS.get(modality, modality), fontsize=12, fontweight='bold')
            continue

        # Count occurrences for each category
        from collections import Counter
        pct_counts = Counter()
        for pct in winning_pcts:
            # Round to nearest expected value
            if pct <= 0.65:
                pct_counts[0.6] += 1  # 3-2
            elif pct <= 0.85:
                pct_counts[0.8] += 1  # 4-1
            else:
                pct_counts[1.0] += 1  # 5-0

        total = len(winning_pcts)
        categories = [0.6, 0.8, 1.0]
        labels = ['60%\n(3-2)', '80%\n(4-1)', '100%\n(5-0)']
        heights = [pct_counts.get(c, 0) / total for c in categories]

        bars = ax.bar(range(len(categories)), heights, color=colors[modality],
                     edgecolor='black', linewidth=0.5, width=0.7)

        # Add value labels
        for bar, h, count in zip(bars, heights, [pct_counts.get(c, 0) for c in categories]):
            ax.annotate(f'{h:.1%}\n(n={count})', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10)

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(f'{MODALITY_LABELS.get(modality, modality)}\n(n={total} samples)',
                    fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Proportion of Samples' if idx == 0 else '', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Voting Agreement Distribution in Few-shot (5 generations)\n'
                 'Higher winning % = stronger consensus',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_by_year_subplots(metrics: Dict, output_path: Path):
    """
    Plot accuracy by year with separate subplots for each prompting technique.
    Each subplot shows all modalities for one technique.
    """
    # Define techniques to plot
    techniques = [
        ("original/single", "Original"),
        ("new/single", "New (JSON)"),
        ("new_fewshot/single", "Few-shot (Single)"),
        ("new_fewshot/majority", "Few-shot (Majority)"),
        ("new_fewshot/metareview", "Few-shot (Meta-review)"),
    ]

    colors = {'clean': '#3498db', 'clean_images': '#2ecc71', 'vision': '#e74c3c'}
    markers = {'clean': 'o', 'clean_images': 's', 'vision': '^'}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (tech_key, tech_name) in enumerate(techniques):
        ax = axes[idx]

        has_data = False
        for modality in MODALITIES:
            config_key = f"{modality}/{tech_key}"
            if config_key not in metrics:
                continue

            m = metrics[config_key]
            if "by_year" not in m:
                continue

            years = sorted([y for y in m["by_year"].keys() if y != "unknown"])
            if not years:
                continue

            accuracies = [m["by_year"][y]["accuracy"] for y in years]

            ax.plot(years, accuracies, marker=markers[modality], markersize=8,
                   linewidth=2, color=colors[modality],
                   label=MODALITY_LABELS.get(modality, modality))
            has_data = True

        if has_data:
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel('Accuracy', fontsize=10)
            ax.set_title(tech_name, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.grid(alpha=0.3)
            ax.legend(loc='best', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(tech_name, fontsize=12, fontweight='bold')

    # Use last subplot for legend
    axes[5].axis('off')
    legend_elements = [plt.Line2D([0], [0], marker=markers[m], color=colors[m],
                                   markersize=10, linewidth=2,
                                   label=MODALITY_LABELS[m]) for m in MODALITIES]
    axes[5].legend(handles=legend_elements, loc='center', fontsize=12,
                   frameon=True, title='Modality', title_fontsize=12)

    plt.suptitle('Accuracy by Year for Each Prompting Technique\n'
                 '(Red dashed line = 50% random baseline, 2025 = out-of-distribution)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 70)
    print("Generating Inference Scaling Plots")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing metrics
    metrics = load_existing_metrics()
    if not metrics:
        print("Warning: No existing metrics found. Run compute_metrics.py first.")
        return

    # Print summary of available data
    print("\nAvailable metrics:")
    for key in sorted(metrics.keys()):
        m = metrics[key]
        print(f"  {key}: acc={m.get('accuracy', 0):.3f}, n={m.get('total', 0)}")

    # Generate plots (skip calibration analysis since scores are clustered at 7)
    print("\nGenerating plots...")
    print("(Note: Calibration skipped - model gives score 7 for ~98% of samples)")

    # Plot 1: Comprehensive comparison with modalities as columns
    plot_comprehensive_comparison(metrics, OUTPUT_DIR / "accuracy_by_modality.png")

    # Plot 2: Technique-focused subplots
    plot_technique_focused(metrics, OUTPUT_DIR / "accuracy_by_technique.png")

    # Plot 3: Modality columns (original version)
    plot_modality_columns(metrics, {}, OUTPUT_DIR / "accuracy_modality_columns.png")

    # Plot 4: Technique subplots (original version)
    plot_technique_subplots(metrics, {}, OUTPUT_DIR / "accuracy_technique_subplots.png")

    # Load JSON validity metrics and plot
    json_metrics = load_json_metrics()
    if json_metrics:
        print("\nJSON Validity Rates:")
        for key in sorted(json_metrics.keys()):
            m = json_metrics[key]
            print(f"  {key}: {m.get('json_validity_rate', 0):.1%} ({m.get('total_predictions', 0)} predictions)")

        # Plot 5: JSON validity comparison
        plot_validity_comparison(json_metrics, OUTPUT_DIR / "validity_comparison.png")

        # Plot 6: JSON validity by modality
        plot_validity_single_subplot(json_metrics, OUTPUT_DIR / "validity_by_modality.png")
    else:
        print("\nNo JSON validity metrics found. Run compute_metrics.py first.")

    # Plot 7: Voting distribution for few-shot predictions
    print("\nGenerating voting distribution plot...")
    plot_voting_distribution(OUTPUT_DIR / "voting_distribution.png")

    # Plot 8: Accuracy by year with technique subplots
    print("Generating accuracy by year subplots...")
    plot_accuracy_by_year_subplots(metrics, OUTPUT_DIR / "accuracy_by_year_techniques.png")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
