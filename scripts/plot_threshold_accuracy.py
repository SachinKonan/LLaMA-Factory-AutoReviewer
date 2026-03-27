so i think the highest confidence #!/usr/bin/env python3
"""
Plot: pct_rating threshold vs accuracy, per year.
Shows what accuracy you'd get if you classified papers as accept/reject
based solely on whether pct_rating >= threshold.
Overlays the model's actual accuracy as horizontal lines.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset with 2026
DATASET = "2020_2026_balanced_vision_binary_noreviews_v8"

# Model accuracy per year (from report.md, bz16_lr1e-6_vision ckpt-2648)
MODEL_ACC = {
    2020: 71.7, 2021: 69.5, 2022: 66.5, 2023: 67.9, 2025: 70.9, 2026: 66.5,
}


def load_data():
    path = BASE / "data" / f"{DATASET}_test" / "data.json"
    with open(path) as f:
        data = json.load(f)

    by_year = defaultdict(list)
    for item in data:
        meta = item["_metadata"]
        year = meta["year"]
        pct = meta.get("pct_rating")
        if pct is None:
            continue

        # Get label from conversation
        label = None
        for c in item["conversations"]:
            if c["from"] == "gpt":
                text = c["value"].lower()
                if "accept" in text:
                    label = 1
                elif "reject" in text:
                    label = 0
                break
        if label is not None:
            by_year[year].append((pct, label))

    return by_year


def threshold_accuracy_curve(samples, thresholds):
    """For each threshold T, predict accept if pct_rating >= T, reject otherwise.
    Return accuracy at each threshold."""
    pcts = np.array([s[0] for s in samples])
    labels = np.array([s[1] for s in samples])
    accs = []
    for t in thresholds:
        preds = (pcts >= t).astype(int)
        acc = np.mean(preds == labels) * 100
        accs.append(acc)
    return accs


def main():
    by_year = load_data()
    thresholds = np.linspace(0, 1, 200)

    # Years to plot (skip 2024 since not in MODEL_ACC)
    plot_years = [2020, 2021, 2022, 2023, 2025, 2026]

    # Color scheme: 2026 stands out
    year_colors = {
        2020: "#8da0cb",
        2021: "#66c2a5",
        2022: "#a6d854",
        2023: "#ffd92f",
        2025: "#fc8d62",
        2026: "#e5394b",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for year in plot_years:
        if year not in by_year:
            continue
        samples = by_year[year]
        accs = threshold_accuracy_curve(samples, thresholds)
        best_idx = np.argmax(accs)
        best_t = thresholds[best_idx]
        best_acc = accs[best_idx]

        lw = 3 if year == 2026 else 1.8
        alpha = 1.0 if year == 2026 else 0.75
        ax.plot(thresholds, accs, color=year_colors[year], linewidth=lw, alpha=alpha,
                label=f"{year} (best={best_acc:.1f}% @ t={best_t:.2f}, n={len(samples)})")

        # Mark the best threshold
        ax.plot(best_t, best_acc, "o", color=year_colors[year], markersize=7, zorder=5)

    # Add model accuracy horizontal lines
    for year in plot_years:
        if year in MODEL_ACC:
            ax.axhline(y=MODEL_ACC[year], color=year_colors[year], linestyle="--",
                       alpha=0.4, linewidth=1)

    # Add a combined "model range" band
    model_vals = [MODEL_ACC[y] for y in plot_years if y in MODEL_ACC]
    ax.axhspan(min(model_vals), max(model_vals), alpha=0.08, color="blue",
               label=f"Model accuracy range ({min(model_vals):.1f}-{max(model_vals):.1f}%)")

    ax.set_xlabel("pct_rating Threshold", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Rating Threshold Classifier vs Model Accuracy by Year", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(45, 100)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)

    # Annotate
    ax.text(0.5, 47, "← predict all Accept          predict all Reject →",
            ha="center", fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    out = OUT_DIR / "report_threshold_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
