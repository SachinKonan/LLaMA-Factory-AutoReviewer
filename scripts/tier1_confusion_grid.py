#!/usr/bin/env python3
"""3x3 grid of confusion matrices: rows = train ratio, cols = test ratio.

Uses epoch-2 checkpoints everywhere. Filtered to 2025+2026.
Cells without inference results show a "pending" placeholder.

Usage:
  python scripts/tier1_confusion_grid.py --modality text
  python scripts/tier1_confusion_grid.py --modality vision
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# -- style guide (matches the user's paper plotting conventions) ------------
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "helvetica",
})
labelsize = 20
titlesize = 20
legendsize = 20
ticksize = 16

# Palette: use the user's blue as the cell fill, white annotations.
BLUE = "#6098FF"
CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "paper_blue", ["#D7E3FF", BLUE, "#1A3E8A"]
)

# -- data plumbing ----------------------------------------------------------
ROOT = Path("results/final_sweep_v7_datasweepv3/optim_search_2026")
DATA = Path("data")

EPOCH2 = {
    ("text", "50_50"): 1322, ("text", "40_60"): 1322, ("text", "30_70"): 1322,
    ("vision", "50_50"): 2648, ("vision", "40_60"): 2642, ("vision", "30_70"): 2642,
}
SHORT = {
    ("text", "50_50"): "bz32_lr1e-6_text",
    ("text", "40_60"): "bz32_lr1e-6_text_40_60",
    ("text", "30_70"): "bz32_lr1e-6_text_30_70",
    ("vision", "50_50"): "bz16_lr1e-6_vision",
    ("vision", "40_60"): "bz16_lr1e-6_vision_40_60",
    ("vision", "30_70"): "bz16_lr1e-6_vision_30_70",
}
TEST_DATA = {
    ("text", "50_50"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
    ("text", "40_60"): DATA / "iclr_2020_2023_2025_2026_40_60_original_text_v7_filtered_test/data.json",
    ("text", "30_70"): DATA / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_test/data.json",
    ("vision", "50_50"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json",
    ("vision", "40_60"): DATA / "iclr_2020_2023_2025_2026_40_60_original_vision_v7_filtered_test/data.json",
    ("vision", "30_70"): DATA / "iclr_2020_2023_2025_2026_30_70_original_vision_v7_filtered_test/data.json",
}

RATIOS = ["50_50", "40_60", "30_70"]
YEARS = {2025, 2026}


def jsonl_for(modality, train, test):
    short = SHORT[(modality, train)]
    step = EPOCH2[(modality, train)]
    if train == "50_50":
        if test == "50_50":
            return ROOT / short / f"finetuned-ckpt-{step}.jsonl"
        return ROOT / "ratio_crossval_clean" / short / f"test_{test}" / f"finetuned-ckpt-{step}.jsonl"
    tag = "balanced" if test == "50_50" else test
    return ROOT / "ratio_sweep" / short / tag / f"finetuned-ckpt-{step}.jsonl"


def confusion(jsonl_path, data_path):
    if not jsonl_path.exists():
        return None
    entries = json.loads(data_path.read_text())
    keep = {i for i, e in enumerate(entries) if (e.get("_metadata") or {}).get("year") in YEARS}
    tn = fp = fn = tp = 0
    with jsonl_path.open() as fh:
        for i, line in enumerate(fh):
            if i not in keep:
                continue
            r = json.loads(line)
            g = 1 if "Accept" in r.get("label", "") else 0 if "Reject" in r.get("label", "") else None
            p = 1 if "Accept" in r.get("predict", "") else 0 if "Reject" in r.get("predict", "") else None
            if g is None or p is None:
                continue
            if g == 1 and p == 1:
                tp += 1
            elif g == 0 and p == 0:
                tn += 1
            elif g == 0 and p == 1:
                fp += 1
            elif g == 1 and p == 0:
                fn += 1
    return tn, fp, fn, tp


# -- plotting ---------------------------------------------------------------

def plot_cell(ax, result):
    if result is None:
        ax.text(0.5, 0.5, "pending", ha="center", va="center",
                fontsize=labelsize, color="#888888", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linestyle("--"); spine.set_alpha(0.4)
        return

    tn, fp, fn, tp = result
    cm = np.array([[tn, fp], [fn, tp]])
    n = cm.sum()
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.where(row_sums == 0, 1, row_sums)

    # Shift vmin so even 0% is a readable medium-blue -> white text always works.
    ax.imshow(cm_norm, cmap=CMAP, vmin=-0.10, vmax=1.0, aspect="equal")
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            frac = cm_norm[i, j]
            ax.text(j, i, f"{count}\n({frac*100:.1f}\\%)",
                    ha="center", va="center", fontsize=ticksize,
                    color="white", fontweight="bold")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Reject", "Accept"], fontsize=ticksize)
    ax.set_yticklabels(["Reject", "Accept"], fontsize=ticksize)
    ax.tick_params(axis="both", which="both", length=0)

    acc = (tp + tn) / n * 100 if n else 0
    ax.text(0.5, -0.40, f"acc={acc:.1f}\\%",
            ha="center", va="top", fontsize=ticksize, color="#333",
            transform=ax.transAxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["text", "vision"], default="text")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    mod = args.modality
    out = Path(args.output or f"tmp_latex_dir/figures/confusion_grid_{mod}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.5))
    fig.suptitle(
        f"\\textbf{{{mod.capitalize()}}} --- confusion matrices (2025+2026)",
        fontsize=titlesize + 2, y=0.98,
    )

    # Leave room on left for row labels + inside gaps so nothing collides.
    plt.subplots_adjust(left=0.22, right=0.97, top=0.92, bottom=0.05,
                        hspace=0.55, wspace=0.40)

    for r_idx, train in enumerate(RATIOS):
        for c_idx, test in enumerate(RATIOS):
            ax = axes[r_idx, c_idx]

            if r_idx == 0:
                ax.set_title(f"test {test.replace('_','/')}",
                             fontsize=labelsize, pad=8)

            # Predicted label on x axis of bottom-row cells.
            if r_idx == 2:
                ax.set_xlabel("Predicted", fontsize=labelsize, labelpad=8)
            # Ground-truth on y axis of left-column cells.
            if c_idx == 0:
                ax.set_ylabel("Ground Truth", fontsize=labelsize, labelpad=10)

            result = confusion(jsonl_for(mod, train, test),
                               TEST_DATA[(mod, test)])
            plot_cell(ax, result)

    # Row labels "train XX/YY" placed in the far-left margin, well clear of
    # the leftmost cell's y-axis tick labels.
    for r_idx, train in enumerate(RATIOS):
        ax = axes[r_idx, 0]
        # Centered on the row at a fixed figure-x coordinate on the left margin.
        bbox = ax.get_position()
        fig.text(0.03, (bbox.y0 + bbox.y1) / 2.0,
                 f"\\textbf{{train {train.replace('_','/')}}}",
                 fontsize=labelsize + 1, ha="center", va="center",
                 rotation=90)

    plt.savefig(out, dpi=200, bbox_inches="tight")
    png = out.with_suffix(".png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out}")
    print(f"saved: {png}")


if __name__ == "__main__":
    main()
