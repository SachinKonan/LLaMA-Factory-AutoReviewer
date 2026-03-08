#!/usr/bin/env python3
"""
Heatmap: model confidence vs reviewer rating (pct_rating), colored by accuracy.

Shows how accuracy varies across the confidence × rating space, with
isoclines and marginal distributions. Uses best optim_search_2026 checkpoints.

Usage:
    python scripts/plot_confidence_rating_heatmap.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Best optim_search_2026 checkpoints (text: bz32 epoch 2, vision: bz16 epoch 2)
RUNS = {
    "Text (bz32, ep2)": {
        "pred_path": PROJECT_ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/finetuned-ckpt-1322.jsonl",
        "dataset": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered",
    },
    "Vision (bz16, ep2)": {
        "pred_path": PROJECT_ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl",
        "dataset": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480",
    },
}

DECISION_TOKEN_IDX = 5
DPI = 150

# Bins
N_CONF_BINS = 10
N_RATING_BINS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_prediction(text: str) -> str:
    text_lower = text.lower().strip()
    if "\\boxed{accept}" in text_lower or "boxed{accept}" in text_lower:
        return "accept"
    if "\\boxed{reject}" in text_lower or "boxed{reject}" in text_lower:
        return "reject"
    if "accept" in text_lower:
        return "accept"
    if "reject" in text_lower:
        return "reject"
    return "unknown"


def load_data(pred_path, dataset_name):
    """Load predictions + metadata, return arrays of confidence, correctness, pct_rating, year."""
    # Load predictions
    preds_data = []
    with open(pred_path) as f:
        for line in f:
            preds_data.append(json.loads(line))

    # Load test metadata
    test_path = PROJECT_ROOT / "data" / f"{dataset_name}_test" / "data.json"
    with open(test_path) as f:
        test_data = json.load(f)

    confidences = []
    correct_arr = []
    ratings = []
    years = []
    pred_labels = []
    true_labels = []

    for i, rec in enumerate(preds_data):
        if i >= len(test_data):
            break
        meta = test_data[i].get("_metadata", {})
        pct_rating = meta.get("pct_rating")
        year = meta.get("year")
        if pct_rating is None:
            continue

        pred = extract_prediction(rec["predict"])
        label = extract_prediction(rec["label"])
        logprobs = rec["token_logprobs"]
        conf = math.exp(logprobs[DECISION_TOKEN_IDX])

        confidences.append(conf)
        correct_arr.append(pred == label and pred != "unknown")
        ratings.append(pct_rating)
        years.append(year)
        pred_labels.append(pred)
        true_labels.append(label)

    return (
        np.array(confidences),
        np.array(correct_arr),
        np.array(ratings),
        np.array(years),
        pred_labels,
        true_labels,
    )


def make_heatmap_data(confidences, correct, ratings, n_conf_bins, n_rating_bins):
    """Bin data into confidence x rating grid, return accuracy and count matrices."""
    conf_edges = np.linspace(0.5, 1.0, n_conf_bins + 1)
    rat_edges = np.linspace(0.0, 1.0, n_rating_bins + 1)

    acc_grid = np.full((n_rating_bins, n_conf_bins), np.nan)
    count_grid = np.zeros((n_rating_bins, n_conf_bins), dtype=int)

    for ri in range(n_rating_bins):
        for ci in range(n_conf_bins):
            if ci < n_conf_bins - 1:
                conf_mask = (confidences >= conf_edges[ci]) & (confidences < conf_edges[ci + 1])
            else:
                conf_mask = (confidences >= conf_edges[ci]) & (confidences <= conf_edges[ci + 1])

            if ri < n_rating_bins - 1:
                rat_mask = (ratings >= rat_edges[ri]) & (ratings < rat_edges[ri + 1])
            else:
                rat_mask = (ratings >= rat_edges[ri]) & (ratings <= rat_edges[ri + 1])

            mask = conf_mask & rat_mask
            n = mask.sum()
            count_grid[ri, ci] = n
            if n >= 3:  # Minimum samples for meaningful accuracy
                acc_grid[ri, ci] = correct[mask].mean() * 100

    return acc_grid, count_grid, conf_edges, rat_edges


def plot_single_heatmap(ax, acc_grid, count_grid, conf_edges, rat_edges, title):
    """Plot a single confidence x rating heatmap with isoclines."""
    # Mask cells with too few samples
    masked_acc = np.ma.masked_invalid(acc_grid)

    im = ax.pcolormesh(
        conf_edges, rat_edges, masked_acc,
        cmap="RdYlGn", vmin=40, vmax=95,
        shading="flat",
    )

    # Add count annotations
    conf_centers = (conf_edges[:-1] + conf_edges[1:]) / 2
    rat_centers = (rat_edges[:-1] + rat_edges[1:]) / 2
    for ri in range(len(rat_centers)):
        for ci in range(len(conf_centers)):
            n = count_grid[ri, ci]
            if n > 0:
                acc_val = acc_grid[ri, ci]
                # Text color based on background
                if np.isnan(acc_val):
                    txt = f"n={n}"
                    color = "gray"
                else:
                    txt = f"{acc_val:.0f}%\nn={n}"
                    color = "white" if acc_val < 55 or acc_val > 85 else "black"
                ax.text(conf_centers[ci], rat_centers[ri], txt,
                        ha="center", va="center", fontsize=6.5, color=color, fontweight="bold")

    # Add isoclines (contour lines at accuracy levels)
    if not np.all(np.isnan(acc_grid)):
        # Interpolate NaN for smooth contours
        acc_for_contour = acc_grid.copy()
        # Replace NaN with nearest neighbor for contour purposes
        from scipy.ndimage import generic_filter
        def nanmean_filter(x):
            valid = x[~np.isnan(x)]
            return np.mean(valid) if len(valid) > 0 else np.nan
        acc_filled = generic_filter(acc_for_contour, nanmean_filter, size=3)

        try:
            cs = ax.contour(
                conf_centers, rat_centers, acc_filled,
                levels=[50, 60, 70, 80, 90],
                colors="black", linewidths=0.8, linestyles="--", alpha=0.5,
            )
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f%%")
        except Exception:
            pass  # Skip if contour fails

    ax.set_xlabel("Model Confidence (P(chosen token))", fontsize=11)
    ax.set_ylabel("Reviewer Rating Percentile (pct_rating)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.0, 1.0)

    return im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading data ...")

all_data = {}
for name, cfg in RUNS.items():
    conf, corr, rat, yr, pl, tl = load_data(cfg["pred_path"], cfg["dataset"])
    all_data[name] = {
        "conf": conf, "correct": corr, "rating": rat, "year": yr,
        "pred_labels": pl, "true_labels": tl,
    }
    print(f"  {name}: {len(conf)} samples, acc={corr.mean():.4f}")

# ---------------------------------------------------------------------------
# Figure 1: Side-by-side heatmaps (text vs vision)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 7))
gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.25)

for idx, (name, data) in enumerate(all_data.items()):
    ax = fig.add_subplot(gs[0, idx])
    acc_grid, count_grid, conf_edges, rat_edges = make_heatmap_data(
        data["conf"], data["correct"], data["rating"], N_CONF_BINS, N_RATING_BINS
    )
    im = plot_single_heatmap(ax, acc_grid, count_grid, conf_edges, rat_edges, name)

# Shared colorbar
cax = fig.add_subplot(gs[0, 2])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Accuracy (%)", fontsize=11)

fig.suptitle("Accuracy by Model Confidence × Reviewer Rating", fontsize=15, fontweight="bold", y=1.02)
out1 = FIGURES_DIR / "confidence_rating_heatmap.png"
fig.savefig(out1, dpi=DPI, bbox_inches="tight")
print(f"Saved {out1}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Combined heatmap with marginal distributions
# ---------------------------------------------------------------------------
for name, data in all_data.items():
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(
        3, 3, width_ratios=[0.15, 1, 0.05], height_ratios=[0.15, 1, 0.001],
        wspace=0.05, hspace=0.05,
    )

    # Main heatmap
    ax_main = fig.add_subplot(gs[1, 1])
    acc_grid, count_grid, conf_edges, rat_edges = make_heatmap_data(
        data["conf"], data["correct"], data["rating"], N_CONF_BINS, N_RATING_BINS
    )
    im = plot_single_heatmap(ax_main, acc_grid, count_grid, conf_edges, rat_edges, "")

    # Top marginal: confidence distribution + accuracy
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    conf_centers = (conf_edges[:-1] + conf_edges[1:]) / 2
    conf_counts = np.array([((data["conf"] >= conf_edges[i]) & (data["conf"] < conf_edges[i + 1])).sum()
                             for i in range(N_CONF_BINS)])
    conf_accs = np.array([data["correct"][(data["conf"] >= conf_edges[i]) & (data["conf"] < conf_edges[i + 1])].mean() * 100
                           if conf_counts[i] > 0 else np.nan
                           for i in range(N_CONF_BINS)])

    ax_top.bar(conf_centers, conf_counts, width=0.045, alpha=0.6, color="#1f77b4", label="Count")
    ax_top_acc = ax_top.twinx()
    ax_top_acc.plot(conf_centers, conf_accs, "ro-", markersize=4, linewidth=1.5, label="Accuracy")
    ax_top_acc.set_ylim(40, 100)
    ax_top_acc.set_ylabel("Acc %", fontsize=8, color="red")
    ax_top_acc.tick_params(labelsize=7, colors="red")
    ax_top.set_ylabel("Count", fontsize=8)
    ax_top.tick_params(labelsize=7)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_title(f"{name}: Confidence × Rating → Accuracy", fontsize=13, fontweight="bold")

    # Left marginal: rating distribution + accuracy
    ax_left = fig.add_subplot(gs[1, 0], sharey=ax_main)
    rat_centers = (rat_edges[:-1] + rat_edges[1:]) / 2
    rat_counts = np.array([((data["rating"] >= rat_edges[i]) & (data["rating"] < rat_edges[i + 1])).sum()
                            for i in range(N_RATING_BINS)])
    rat_accs = np.array([data["correct"][(data["rating"] >= rat_edges[i]) & (data["rating"] < rat_edges[i + 1])].mean() * 100
                          if rat_counts[i] > 0 else np.nan
                          for i in range(N_RATING_BINS)])

    ax_left.barh(rat_centers, rat_counts, height=0.09, alpha=0.6, color="#ff7f0e", label="Count")
    ax_left_acc = ax_left.twiny()
    ax_left_acc.plot(rat_accs, rat_centers, "go-", markersize=4, linewidth=1.5, label="Accuracy")
    ax_left_acc.set_xlim(40, 100)
    ax_left_acc.set_xlabel("Acc %", fontsize=8, color="green")
    ax_left_acc.tick_params(labelsize=7, colors="green")
    ax_left.set_xlabel("Count", fontsize=8)
    ax_left.tick_params(labelsize=7)
    ax_left.invert_xaxis()
    plt.setp(ax_left.get_yticklabels(), visible=False)

    # Colorbar
    cax = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Accuracy (%)", fontsize=10)

    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    out = FIGURES_DIR / f"confidence_rating_heatmap_{safe_name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: Agreement/disagreement analysis
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (name, data) in enumerate(all_data.items()):
    ax = axes[idx]
    conf = data["conf"]
    rating = data["rating"]
    correct = data["correct"]

    # Quadrants based on confidence and rating
    high_conf = conf >= 0.7
    high_rating = rating >= 0.5  # Above median rating → likely accept

    # Model predicts accept when confidence is high AND prediction is accept
    pred_accept = np.array([p == "accept" for p in data["pred_labels"]])
    rating_says_accept = high_rating

    # Agreement: both say accept or both say reject
    agree = pred_accept == rating_says_accept

    # Scatter plot colored by correctness
    scatter_correct = ax.scatter(
        conf[correct], rating[correct],
        c="green", alpha=0.15, s=10, label=f"Correct ({correct.sum()})", rasterized=True,
    )
    scatter_wrong = ax.scatter(
        conf[~correct], rating[~correct],
        c="red", alpha=0.25, s=10, label=f"Wrong ({(~correct).sum()})", rasterized=True,
    )

    # Quadrant labels
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0.7, color="gray", linestyle="--", alpha=0.5)

    # Compute accuracy in each quadrant
    quadrants = [
        ("Low C / Low R", ~high_conf & ~high_rating),
        ("Low C / High R", ~high_conf & high_rating),
        ("High C / Low R", high_conf & ~high_rating),
        ("High C / High R", high_conf & high_rating),
    ]
    positions = [(0.55, 0.25), (0.55, 0.75), (0.85, 0.25), (0.85, 0.75)]

    for (qname, qmask), (px, py) in zip(quadrants, positions):
        n = qmask.sum()
        if n > 0:
            qacc = correct[qmask].mean() * 100
            ax.text(px, py, f"{qacc:.0f}%\nn={n}", ha="center", va="center",
                    fontsize=9, fontweight="bold", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Model Confidence", fontsize=11)
    ax.set_ylabel("Reviewer Rating Percentile", fontsize=11)
    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)

fig.suptitle("Model Confidence vs Reviewer Rating: Correct/Incorrect", fontsize=14, fontweight="bold")
plt.tight_layout()
out3 = FIGURES_DIR / "confidence_rating_scatter.png"
fig.savefig(out3, dpi=DPI, bbox_inches="tight")
print(f"Saved {out3}")
plt.close(fig)

print("Done.")
