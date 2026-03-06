#!/usr/bin/env python3
"""Bar charts comparing accuracy across year ranges and modalities."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
CSV_PATH = PROJECT_ROOT / "results" / "summarized_investigation" / "modality_v7" / "metrics_table.csv"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABELSIZE = 13
TITLESIZE = 14
LEGENDSIZE = 9
TICKSIZE = 10
DPI = 150

MODALITY_RENAME = {"clean": "Text", "images": "Text+Images", "vision": "Vision"}
MODALITY_ORDER = ["Text", "Text+Images", "Vision"]

YEAR_RANGE_MAP = {
    "balanced": "2020-2025",
    "balanced_2017_2025": "2017-2025",
    "balanced_2024_2025": "2024-2025",
}
YEAR_RANGE_ORDER = ["2020-2025", "2017-2025", "2024-2025"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df["modality_label"] = df["modality"].map(MODALITY_RENAME)

# ---------------------------------------------------------------------------
# Plot 1 – Accuracy by Year Range and Modality (balanced variants only)
# ---------------------------------------------------------------------------
balanced_groups = set(YEAR_RANGE_MAP.keys())
df_bal = df[df["dataset_group"].isin(balanced_groups)].copy()
df_bal["year_range"] = df_bal["dataset_group"].map(YEAR_RANGE_MAP)

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(YEAR_RANGE_ORDER))
n_mod = len(MODALITY_ORDER)
width = 0.22

for i, mod in enumerate(MODALITY_ORDER):
    subset = df_bal[df_bal["modality_label"] == mod].set_index("year_range")
    vals = [subset.loc[yr, "accuracy"] * 100 if yr in subset.index else 0
            for yr in YEAR_RANGE_ORDER]
    bars = ax.bar(x + i * width, vals, width, label=mod)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=TICKSIZE)

ax.set_xticks(x + width)
ax.set_xticklabels(YEAR_RANGE_ORDER, fontsize=TICKSIZE)
ax.set_ylabel("Accuracy (%)", fontsize=LABELSIZE)
ax.set_xlabel("Year Range", fontsize=LABELSIZE)
ax.set_title("Accuracy by Year Range and Modality", fontsize=TITLESIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.tick_params(axis="y", labelsize=TICKSIZE)

fig.savefig(FIG_DIR / "year_range_ablation.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR / 'year_range_ablation.png'}")

# ---------------------------------------------------------------------------
# Plot 2 – Balanced vs Trainagreeing
# ---------------------------------------------------------------------------
df_cmp = df[df["dataset_group"].isin(["balanced", "balanced_trainagreeing"])].copy()
df_cmp["group_label"] = df_cmp["dataset_group"].map(
    {"balanced": "Balanced", "balanced_trainagreeing": "Trainagreeing"}
)
GROUP_ORDER = ["Balanced", "Trainagreeing"]

fig, ax = plt.subplots(figsize=(7, 5))

x = np.arange(len(MODALITY_ORDER))
width = 0.30

for i, grp in enumerate(GROUP_ORDER):
    subset = df_cmp[df_cmp["group_label"] == grp].set_index("modality_label")
    vals = [subset.loc[mod, "accuracy"] * 100 if mod in subset.index else 0
            for mod in MODALITY_ORDER]
    bars = ax.bar(x + i * width, vals, width, label=grp)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=TICKSIZE)

ax.set_xticks(x + width / 2)
ax.set_xticklabels(MODALITY_ORDER, fontsize=TICKSIZE)
ax.set_ylabel("Accuracy (%)", fontsize=LABELSIZE)
ax.set_xlabel("Modality", fontsize=LABELSIZE)
ax.set_title("Balanced vs Trainagreeing Dataset", fontsize=TITLESIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.tick_params(axis="y", labelsize=TICKSIZE)

fig.savefig(FIG_DIR / "year_range_ablation_trainagreeing.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR / 'year_range_ablation_trainagreeing.png'}")
