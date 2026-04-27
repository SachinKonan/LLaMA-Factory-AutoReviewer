#!/usr/bin/env python3
"""
3x2 figure per modality: ratio-sweep training curves.

Rows = training ratio (50/50, 40/60, 30/70).
Cols = (1) training loss vs epoch, (2) test accuracy on 2025+2026 across the
       three test ratios (50/50, 40/60, 30/70).

50/50 row in col 2 has a single point per non-50/50 test ratio (only the best
checkpoint was cross-evaluated on the off-ratio test sets — see
`ratio_crossval_clean/`). The 40/60 and 30/70 rows have all checkpoints
evaluated on all three test ratios.

Train accuracy column intentionally omitted — the 14 train-acc inference jobs
are still queued. Will regenerate this script's output once train-ckpt JSONs
land. For now: just train loss + test acc.

Outputs:
    tmp_latex_dir/figures/ratio_train_curves_text.pdf / .png
    tmp_latex_dir/figures/ratio_train_curves_vision.pdf / .png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style (matches existing tmp_latex_dir scripts)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})

labelsize = 20
titlesize = 20
legendsize = 14
ticksize = 16

LINEWIDTH = 2.4
MARKERSIZE = 9

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_ROOT = ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
SAVES_ROOT = ROOT / "saves/final_sweep_v7_datasweepv3/optim_search_2026"
DATA_ROOT = ROOT / "data"

# Color per test ratio (matches tier1_auc_bars.py palette)
COLOR_PER_TEST = {
    "50_50": "#6098FF",
    "40_60": "#77B25D",
    "30_70": "#B28CFF",
}
TEST_LABEL = {"50_50": "Test 50/50", "40_60": "Test 40/60", "30_70": "Test 30/70"}

EVAL_YEARS = {2025, 2026}

# Per-modality config: train ratio -> (short_name, ckpts, save_dir)
TEXT_RUNS = [
    ("50_50", "bz32_lr1e-6_text",         [661, 1322, 1983, 2644]),
    ("40_60", "bz32_lr1e-6_text_40_60",   [661, 1322, 1983]),
    ("30_70", "bz32_lr1e-6_text_30_70",   [661, 1322, 1983]),
]
VISION_RUNS = [
    ("50_50", "bz16_lr1e-6_vision",         [1324, 2648, 3972, 5296]),
    ("40_60", "bz16_lr1e-6_vision_40_60",   [1321, 2642, 3963, 5284]),
    ("30_70", "bz16_lr1e-6_vision_30_70",   [1321, 2642, 3963, 5284]),
]

TEXT_TEST_DATA = {
    "50_50": DATA_ROOT / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
    "40_60": DATA_ROOT / "iclr_2020_2023_2025_2026_40_60_original_text_v7_filtered_test/data.json",
    "30_70": DATA_ROOT / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_test/data.json",
}
VISION_TEST_DATA = {
    "50_50": DATA_ROOT / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json",
    "40_60": DATA_ROOT / "iclr_2020_2023_2025_2026_40_60_original_vision_v7_filtered_test/data.json",
    "30_70": DATA_ROOT / "iclr_2020_2023_2025_2026_30_70_original_vision_v7_filtered_test/data.json",
}

# Save dirs differ for ratio_sweep vs the 50/50 baseline
def save_dir(modality: str, train_ratio: str, short: str) -> Path:
    if train_ratio == "50_50":
        return SAVES_ROOT / short
    return SAVES_ROOT / "ratio_sweep" / short


def jsonl_for(modality: str, train_ratio: str, short: str, test_ratio: str, ckpt: int) -> Path | None:
    """Return path to the inference jsonl for (train, test) at this ckpt, or None."""
    if train_ratio == "50_50":
        if test_ratio == "50_50":
            return RESULTS_ROOT / short / f"finetuned-ckpt-{ckpt}.jsonl"
        # cross-eval lives in ratio_crossval_clean
        return RESULTS_ROOT / "ratio_crossval_clean" / short / f"test_{test_ratio}" / f"finetuned-ckpt-{ckpt}.jsonl"
    # 40/60 or 30/70 train: cross-eval subdirs under ratio_sweep
    tag = "balanced" if test_ratio == "50_50" else test_ratio
    return RESULTS_ROOT / "ratio_sweep" / short / tag / f"finetuned-ckpt-{ckpt}.jsonl"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_train_loss(save_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (steps, epochs, losses)."""
    log = save_path / "trainer_log.jsonl"
    steps, epochs, losses = [], [], []
    with log.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "loss" not in r:
                continue
            steps.append(r["current_steps"])
            epochs.append(r.get("epoch", 0.0))
            losses.append(r["loss"])
    return np.array(steps), np.array(epochs), np.array(losses)


def extract_pred(text: str) -> str:
    t = text.lower().strip()
    if "\\boxed{accept}" in t or "boxed{accept}" in t:
        return "accept"
    if "\\boxed{reject}" in t or "boxed{reject}" in t:
        return "reject"
    if "accept" in t:
        return "accept"
    if "reject" in t:
        return "reject"
    return "unknown"


def acc_2526(jsonl: Path, test_data_path: Path) -> tuple[float | None, int]:
    if not jsonl.exists() or not test_data_path.exists():
        return None, 0
    test_meta = json.loads(test_data_path.read_text())
    ok = miss = 0
    with jsonl.open() as f:
        for i, line in enumerate(f):
            if i >= len(test_meta):
                break
            yr = (test_meta[i].get("_metadata") or {}).get("year")
            if yr not in EVAL_YEARS:
                continue
            r = json.loads(line)
            p = extract_pred(r.get("predict", ""))
            g = extract_pred(r.get("label", ""))
            if p == "unknown" or g == "unknown":
                miss += 1
                continue
            if p == g:
                ok += 1
            else:
                miss += 1
    n = ok + miss
    if n == 0:
        return None, 0
    return ok / n * 100.0, n


def epoch_for_step(epochs: np.ndarray, steps: np.ndarray, target_step: int) -> float:
    """Return the trainer-logged epoch closest to target_step. Falls back to step/N approx."""
    if len(steps) == 0:
        return 0.0
    idx = int(np.argmin(np.abs(steps - target_step)))
    return float(epochs[idx])


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def smooth(y: np.ndarray, window: int = 10) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def panel_train_loss(ax, steps, epochs, losses, ratio_label):
    color = COLOR_PER_TEST[ratio_label.replace("/", "_")]
    ax.plot(epochs, losses, color=color, linewidth=0.8, alpha=0.25, zorder=1)
    if len(losses) >= 10:
        ys = smooth(losses, 10)
        # Align xs
        offset = (len(losses) - len(ys)) // 2
        xs = epochs[offset:offset + len(ys)]
        ax.plot(xs, ys, color=color, linewidth=LINEWIDTH, label="loss (smoothed)", zorder=3)
    else:
        ax.plot(epochs, losses, color=color, linewidth=LINEWIDTH, zorder=3)

    ax.set_xlabel("Epoch", fontsize=labelsize - 2)
    ax.set_ylabel("Train Loss", fontsize=labelsize)
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


def panel_test_acc(ax, modality, train_ratio, short, ckpts, save_path,
                   test_datasets):
    steps, epochs, losses = load_train_loss(save_path)

    # Plot one line per test ratio
    all_acc = []
    for test_ratio in ("50_50", "40_60", "30_70"):
        xs, ys, ns = [], [], []
        for ckpt in ckpts:
            jsonl = jsonl_for(modality, train_ratio, short, test_ratio, ckpt)
            if jsonl is None:
                continue
            acc, n = acc_2526(jsonl, test_datasets[test_ratio])
            if acc is None:
                continue
            ep = epoch_for_step(epochs, steps, ckpt)
            xs.append(ep); ys.append(acc); ns.append(n)
        if not xs:
            continue
        all_acc.extend(ys)
        color = COLOR_PER_TEST[test_ratio]
        if len(xs) > 1:
            ax.plot(xs, ys, "-o", color=color, linewidth=LINEWIDTH,
                    markersize=MARKERSIZE, label=TEST_LABEL[test_ratio], zorder=3)
        else:
            # single point — render as a star marker
            ax.plot(xs, ys, "*", color=color, markersize=MARKERSIZE + 6,
                    markeredgecolor="black", markeredgewidth=0.8,
                    label=TEST_LABEL[test_ratio] + " (best ckpt)", zorder=4, linestyle="None")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        fontsize=ticksize - 4, color=color)

    ax.set_xlabel("Epoch", fontsize=labelsize - 2)
    ax.set_ylabel("Test Acc 25/26 (%)", fontsize=labelsize)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    if all_acc:
        lo, hi = min(all_acc), max(all_acc)
        span = max(hi - lo, 2.0)
        ax.set_ylim(lo - 0.2 * span, hi + 0.45 * span)
    ax.legend(fontsize=legendsize, loc="lower right", framealpha=0.95, ncol=1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_figure(modality: str, runs, test_data: dict, out_basename: str):
    nrows = len(runs)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.5 * nrows))

    print(f"\n=== {modality.upper()} ===")
    for row, (ratio_label, short, ckpts) in enumerate(runs):
        ratio_disp = ratio_label.replace("_", "/")
        save_path = save_dir(modality, ratio_label, short)
        steps, epochs, losses = load_train_loss(save_path)
        n_log = len(losses)
        print(f"Train {ratio_disp}: {short}  log entries={n_log}  final epoch={epochs[-1] if n_log else 0:.2f}")

        # Col 1: train loss
        panel_train_loss(axes[row, 0], steps, epochs, losses, ratio_label)
        axes[row, 0].set_title(f"Train Loss — {ratio_disp}", fontsize=titlesize - 4, pad=8)

        # Col 2: test acc on 25/26 across 3 test ratios
        panel_test_acc(axes[row, 1], modality, ratio_label, short, ckpts, save_path, test_data)
        axes[row, 1].set_title(f"Test Acc 25/26 — {ratio_disp}", fontsize=titlesize - 4, pad=8)

    fig.suptitle(f"Ratio Sweep — {modality.title()}", fontsize=titlesize + 2, y=1.005, fontweight="bold")
    plt.tight_layout()

    out = OUTPUT_DIR / out_basename
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved: {out}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["text", "vision", "both"], default="both")
    args = ap.parse_args()

    if args.modality in ("text", "both"):
        make_figure("text", TEXT_RUNS, TEXT_TEST_DATA, "ratio_train_curves_text")
    if args.modality in ("vision", "both"):
        make_figure("vision", VISION_RUNS, VISION_TEST_DATA, "ratio_train_curves_vision")


if __name__ == "__main__":
    main()
