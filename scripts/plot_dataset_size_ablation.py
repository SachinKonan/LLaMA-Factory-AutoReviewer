#!/usr/bin/env python3
"""
Plot dataset size ablation: table figure and train-size-vs-accuracy chart.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
LABELSIZE = 13
TITLESIZE = 14
LEGENDSIZE = 9
TICKSIZE = 10

# ── Data ───────────────────────────────────────────────────────────────────
# From modality_v7 metrics_table.csv (v7 splits, accuracy on v7 test)
# Each entry: (dataset_variant, modality, train_size, accuracy)
records = [
    # balanced (2020-2025)
    ("2020-2025 Balanced", "Text",        12745, 66.2),
    ("2020-2025 Balanced", "Text+Images", 12451, 67.0),
    ("2020-2025 Balanced", "Vision",      12765, 69.8),
    # balanced_2017_2025
    ("2017-2025 Balanced", "Text",        14577, 65.7),
    ("2017-2025 Balanced", "Text+Images", 14211, 67.0),
    ("2017-2025 Balanced", "Vision",      14594, 69.5),
    # balanced_2024_2025
    ("2024-2025 Balanced", "Text",         3889, 63.8),
    ("2024-2025 Balanced", "Text+Images",  3772, 63.5),
    ("2024-2025 Balanced", "Vision",       3893, 70.9),
    # balanced_trainagreeing
    ("2020-2025 Trainagreeing", "Text",        8296, 66.9),
    ("2020-2025 Trainagreeing", "Text+Images", 8093, 68.2),
    ("2020-2025 Trainagreeing", "Vision",      8315, 70.4),
]

# v8 dataset sizes (for reference table; not all have accuracy results yet)
v8_sizes = {
    "2020_2025_clean":              {"train": 17099, "val": 1495, "test": 3016},
    "2020_2025_vision":             {"train": 17128, "val": 1496, "test": 3018},
    "2020_2025_trainagreeing_clean": {"train": 11132, "val": 1495, "test": 3016},
    "2020_2025_trainagreeing_vision":{"train": 11154, "val": 1496, "test": 3018},
    "2020_2026_clean":              {"train": 25503, "val": 1495, "test": 3016},
    "2020_2026_vision":             {"train": 27266, "val": 1594, "test": 3228},
    "2017_2026_clean":              {"train": 27223, "val": 1593, "test": 3226},
    "2017_2026_vision":             {"train": 27266, "val": 1594, "test": 3228},
}

# ── Derived structures for the table ──────────────────────────────────────
dataset_variants = [
    "2020-2025 Balanced",
    "2017-2025 Balanced",
    "2024-2025 Balanced",
    "2020-2025 Trainagreeing",
]
modalities = ["Text", "Text+Images", "Vision"]

# Build lookup: (variant, modality) -> (train, acc)
lookup = {}
for variant, modality, train_size, acc in records:
    lookup[(variant, modality)] = (train_size, acc)


# ══════════════════════════════════════════════════════════════════════════
# Plot 1: Table figure
# ══════════════════════════════════════════════════════════════════════════
def plot_table():
    col_labels = ["Dataset Variant", "Modality", "Train", "Val", "Test", "Best Acc (%)"]

    # For val/test we use approximate values consistent with v7 splits
    # (v7 had 85/5/10 splits, so val ~ train*5/85, test ~ train*10/85)
    rows = []
    for variant in dataset_variants:
        best_acc_in_row = max(
            lookup[(variant, m)][1] for m in modalities if (variant, m) in lookup
        )
        for modality in modalities:
            train_size, acc = lookup[(variant, modality)]
            # Approximate val/test from the 85/5/10 split ratio
            val_size = int(round(train_size * 5 / 85))
            test_size = int(round(train_size * 10 / 85))
            is_best = abs(acc - best_acc_in_row) < 0.01
            rows.append((variant, modality, train_size, val_size, test_size, acc, is_best))

    cell_text = []
    cell_colours = []
    for variant, modality, train_sz, val_sz, test_sz, acc, is_best in rows:
        cell_text.append([
            variant, modality,
            f"{train_sz:,}", f"{val_sz:,}", f"{test_sz:,}",
            f"{acc:.1f}"
        ])
        if is_best:
            cell_colours.append(["#ffffff"] * 5 + ["#c6efce"])  # green highlight
        else:
            cell_colours.append(["#ffffff"] * 6)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_title("Dataset Size Ablation", fontsize=TITLESIZE, fontweight="bold", pad=20)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colours,
        colColours=["#d9e2f3"] * len(col_labels),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(TICKSIZE)
    table.scale(1.0, 1.5)

    # Bold header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=TICKSIZE + 1)

    fig.savefig(FIGURES_DIR / "dataset_sizes.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'dataset_sizes.png'}")


# ══════════════════════════════════════════════════════════════════════════
# Plot 2: Train size vs accuracy scatter/line
# ══════════════════════════════════════════════════════════════════════════
def plot_train_size_vs_accuracy():
    modality_colors = {
        "Text": "#1f77b4",        # blue
        "Text+Images": "#2ca02c", # green
        "Vision": "#ff7f0e",      # orange
    }
    variant_markers = {
        "2020-2025 Balanced": "o",
        "2017-2025 Balanced": "o",
        "2024-2025 Balanced": "o",
        "2020-2025 Trainagreeing": "^",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    all_x, all_y = [], []

    for variant, modality, train_size, acc in records:
        color = modality_colors[modality]
        marker = variant_markers[variant]
        ax.scatter(
            train_size, acc,
            c=color, marker=marker, s=90, edgecolors="black", linewidths=0.5, zorder=5,
        )
        # Label: abbreviated variant name
        short = variant.replace("Balanced", "Bal").replace("Trainagreeing", "TA")
        ax.annotate(
            short,
            (train_size, acc),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            color=color,
            alpha=0.85,
        )
        all_x.append(train_size)
        all_y.append(acc)

    # Trend line (linear fit)
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    z = np.polyfit(all_x, all_y, 1)
    p = np.poly1d(z)
    x_range = np.linspace(all_x.min() - 500, all_x.max() + 500, 200)
    ax.plot(x_range, p(x_range), "--", color="gray", alpha=0.5, linewidth=1.2, label="Trend (linear fit)")

    # Legend: modality colours
    for modality, color in modality_colors.items():
        ax.scatter([], [], c=color, marker="o", s=60, edgecolors="black", linewidths=0.5, label=modality)
    # Legend: marker shapes
    ax.scatter([], [], c="gray", marker="o", s=60, edgecolors="black", linewidths=0.5, label="Balanced")
    ax.scatter([], [], c="gray", marker="^", s=60, edgecolors="black", linewidths=0.5, label="Trainagreeing")

    ax.set_xlabel("Training Set Size", fontsize=LABELSIZE)
    ax.set_ylabel("Test Accuracy (%)", fontsize=LABELSIZE)
    ax.set_title("Training Set Size vs. Test Accuracy", fontsize=TITLESIZE, fontweight="bold")
    ax.tick_params(labelsize=TICKSIZE)
    ax.legend(fontsize=LEGENDSIZE, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.savefig(FIGURES_DIR / "train_size_vs_accuracy.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'train_size_vs_accuracy.png'}")


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    plot_table()
    plot_train_size_vs_accuracy()
    print("Done.")
