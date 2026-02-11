"""Shared analysis utilities for all experiments."""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Data Loading
# ============================================================================

def load_results(path: str) -> List[Dict]:
    """Load results from a JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results


def load_predictions(path: str) -> List[Dict]:
    """Load predictions from a JSONL file."""
    return load_results(path)


def load_dataset(path: str) -> List[Dict]:
    """Load a dataset from a data.json file."""
    json_path = os.path.join(path, "data.json") if os.path.isdir(path) else path
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Metric Computation
# ============================================================================

def compute_accuracy(results: List[Dict]) -> float:
    """Compute overall accuracy from results."""
    valid = [r for r in results if r.get("correct") is not None]
    if not valid:
        return 0.0
    return sum(1 for r in valid if r["correct"]) / len(valid)


def compute_acceptance_rate(results: List[Dict]) -> float:
    """Compute acceptance rate from results."""
    valid = [r for r in results if r.get("prediction") is not None]
    if not valid:
        return 0.0
    return sum(1 for r in valid if r["prediction"] == "Accept") / len(valid)


def compute_recall(results: List[Dict], label: str) -> float:
    """Compute recall for a specific class (Accept or Reject)."""
    class_samples = [r for r in results if r.get("ground_truth") == label]
    if not class_samples:
        return 0.0
    return sum(1 for r in class_samples if r["prediction"] == label) / len(class_samples)


def compute_precision(results: List[Dict], label: str) -> float:
    """Compute precision for a specific class."""
    predicted = [r for r in results if r.get("prediction") == label]
    if not predicted:
        return 0.0
    return sum(1 for r in predicted if r["ground_truth"] == label) / len(predicted)


def compute_confusion_matrix(results: List[Dict]) -> Dict[str, int]:
    """Compute confusion matrix counts."""
    valid = [r for r in results if r.get("prediction") and r.get("ground_truth")]
    return {
        "TP": sum(1 for r in valid if r["prediction"] == "Accept" and r["ground_truth"] == "Accept"),
        "TN": sum(1 for r in valid if r["prediction"] == "Reject" and r["ground_truth"] == "Reject"),
        "FP": sum(1 for r in valid if r["prediction"] == "Accept" and r["ground_truth"] == "Reject"),
        "FN": sum(1 for r in valid if r["prediction"] == "Reject" and r["ground_truth"] == "Accept"),
    }


def compute_metrics_summary(results: List[Dict]) -> Dict:
    """Compute a full metrics summary."""
    cm = compute_confusion_matrix(results)
    total = sum(cm.values())
    return {
        "total": total,
        "accuracy": compute_accuracy(results),
        "acceptance_rate": compute_acceptance_rate(results),
        "accept_recall": compute_recall(results, "Accept"),
        "reject_recall": compute_recall(results, "Reject"),
        "accept_precision": compute_precision(results, "Accept"),
        "reject_precision": compute_precision(results, "Reject"),
        "confusion_matrix": cm,
    }


def compute_by_year(results: List[Dict], metadata: Optional[List[Dict]] = None) -> Dict[int, Dict]:
    """Compute metrics broken down by year.

    If metadata is provided, it should align with results (same indices).
    Otherwise, results must contain a 'year' field.
    """
    by_year = {}
    for i, r in enumerate(results):
        year = r.get("year")
        if year is None and metadata and i < len(metadata):
            year = metadata[i].get("_metadata", {}).get("year")
        if year is None:
            continue
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(r)

    return {year: compute_metrics_summary(rs) for year, rs in sorted(by_year.items())}


# ============================================================================
# Plotting
# ============================================================================

def setup_plot_style():
    """Set up consistent plot styling."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })


def plot_confusion_matrix(cm: Dict[str, int], title: str, save_path: str):
    """Plot a confusion matrix heatmap."""
    setup_plot_style()
    matrix = np.array([[cm["TP"], cm["FN"]], [cm["FP"], cm["TN"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")

    labels = ["Accept", "Reject"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=16)

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_bar_comparison(data: Dict[str, float], title: str, ylabel: str, save_path: str,
                        ylim: Optional[Tuple[float, float]] = None, colors=None):
    """Plot a bar chart comparing values."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(data.keys())
    values = list(data.values())
    x = np.arange(len(names))

    if colors is None:
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}" if val <= 1 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_line(data: Dict[str, List[float]], x_labels: List[str],
              title: str, ylabel: str, save_path: str,
              ylim: Optional[Tuple[float, float]] = None):
    """Plot line chart with multiple series."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set1(np.linspace(0, 1, len(data)))
    for (name, values), color in zip(data.items(), colors):
        ax.plot(x_labels, values, marker="o", label=name, color=color, linewidth=2)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def save_metrics_json(metrics: Dict, save_path: str):
    """Save metrics to a JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {save_path}")
