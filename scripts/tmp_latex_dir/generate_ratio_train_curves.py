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

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

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


def _score_logodds(row: dict) -> float | None:
    """Log-odds for Accept (matches tier1_auc_bars.py logic).

    Preferred: lp_accept - lp_reject at the decision step where the chosen
    token's logprob == max(lp_accept, lp_reject).
    Fallback: derive from min(token_logprobs) and the chosen class.
    """
    predict = row.get("predict", "")
    if "\\boxed{Accept}" in predict:
        chosen = 1
    elif "\\boxed{Reject}" in predict:
        chosen = 0
    else:
        return None

    la = row.get("logprob_accept") or []
    lr = row.get("logprob_reject") or []
    tl = row.get("token_logprobs") or []

    for k in range(min(len(tl), len(la), len(lr))):
        if la[k] is not None and lr[k] is not None and tl[k] is not None:
            a, r = la[k], lr[k]
            if math.isclose(tl[k], max(a, r), abs_tol=1e-3):
                return a - r

    # Fallback
    if not tl:
        return 6.0 if chosen == 1 else -6.0
    valid_lp = [x for x in tl if x is not None]
    if not valid_lp:
        return 6.0 if chosen == 1 else -6.0
    min_lp = min(valid_lp)
    p_chosen = max(min(math.exp(min_lp), 1 - 1e-6), 1e-6)
    p_accept = p_chosen if chosen == 1 else 1.0 - p_chosen
    p_accept = max(min(p_accept, 1 - 1e-6), 1e-6)
    score = math.log(p_accept / (1.0 - p_accept))
    if chosen == 1:
        score = max(score, 1e-3)
    else:
        score = min(score, -1e-3)
    return score


def metrics_2526(jsonl: Path, test_data_path: Path) -> dict:
    """Compute accuracy, accept recall, reject recall, AUC on 25/26 subset.

    Returns dict with keys: acc, accr, rejr, auc, n. Values may be None
    if computation isn't possible. acc/accr/rejr are percentages.
    """
    out = {"acc": None, "accr": None, "rejr": None, "auc": None, "n": 0}
    if not jsonl.exists() or not test_data_path.exists():
        return out
    test_meta = json.loads(test_data_path.read_text())

    tp = tn = fp = fn = 0
    y_true: list[int] = []
    y_score: list[float] = []
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
                continue
            if p == "accept" and g == "accept":
                tp += 1
            elif p == "reject" and g == "reject":
                tn += 1
            elif p == "accept" and g == "reject":
                fp += 1
            elif p == "reject" and g == "accept":
                fn += 1
            sc = _score_logodds(r)
            if sc is not None:
                y_true.append(1 if g == "accept" else 0)
                y_score.append(sc)

    n = tp + tn + fp + fn
    if n == 0:
        return out
    out["n"] = n
    out["acc"] = (tp + tn) / n * 100.0
    out["accr"] = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else None
    out["rejr"] = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else None
    if roc_auc_score is not None and len(set(y_true)) > 1:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["auc"] = None
    return out


def acc_2526(jsonl: Path, test_data_path: Path) -> tuple[float | None, int]:
    """Backward-compat thin wrapper."""
    m = metrics_2526(jsonl, test_data_path)
    return m["acc"], m["n"]


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


def _collect_metrics(modality, train_ratio, short, ckpts, test_data, save_path):
    """For each test ratio, collect (epochs, [metrics dict per ckpt])."""
    steps, epochs, _ = load_train_loss(save_path)
    out: dict[str, dict] = {}
    for test_ratio in ("50_50", "40_60", "30_70"):
        xs, metrics = [], []
        for ckpt in ckpts:
            jsonl = jsonl_for(modality, train_ratio, short, test_ratio, ckpt)
            if jsonl is None:
                continue
            m = metrics_2526(jsonl, test_data[test_ratio])
            if m["n"] == 0:
                continue
            xs.append(epoch_for_step(epochs, steps, ckpt))
            metrics.append(m)
        out[test_ratio] = {"xs": xs, "metrics": metrics}
    return out


def panel_test_acc(ax, modality, train_ratio, short, ckpts, save_path,
                   test_datasets):
    metric_data = _collect_metrics(modality, train_ratio, short, ckpts, test_datasets, save_path)

    all_acc = []
    for test_ratio in ("50_50", "40_60", "30_70"):
        xs = metric_data[test_ratio]["xs"]
        ys = [m["acc"] for m in metric_data[test_ratio]["metrics"]]
        if not xs:
            continue
        all_acc.extend(ys)
        color = COLOR_PER_TEST[test_ratio]
        if len(xs) > 1:
            ax.plot(xs, ys, "-o", color=color, linewidth=LINEWIDTH,
                    markersize=MARKERSIZE, label=TEST_LABEL[test_ratio], zorder=3)
        else:
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
        ax.set_ylim(lo - 0.15 * span, hi + 0.25 * span)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


# ---------------------------------------------------------------------------
# New columns: recall mini-stack + AUC
# ---------------------------------------------------------------------------

ACCR_COLOR = "#2CA02C"   # green for accept recall
REJR_COLOR = "#D62728"   # red for reject recall


def panel_recall_minis(fig, outer_spec, modality, train_ratio, short, ckpts,
                       save_path, test_datasets):
    """Inside one outer cell, draw 3 vertically-stacked mini panels (one per
    test ratio), each plotting accept_recall + reject_recall vs epoch."""
    metric_data = _collect_metrics(modality, train_ratio, short, ckpts, test_datasets, save_path)

    inner = outer_spec.subgridspec(3, 1, hspace=0.05)
    sub_axes = []
    all_vals: list[float] = []
    for sub_row, test_ratio in enumerate(("50_50", "40_60", "30_70")):
        ax = fig.add_subplot(inner[sub_row])
        sub_axes.append(ax)
        xs = metric_data[test_ratio]["xs"]
        accrs = [m["accr"] for m in metric_data[test_ratio]["metrics"] if m["accr"] is not None]
        rejrs = [m["rejr"] for m in metric_data[test_ratio]["metrics"] if m["rejr"] is not None]
        valid_xs = [x for x, m in zip(xs, metric_data[test_ratio]["metrics"]) if m["accr"] is not None]

        if valid_xs:
            all_vals.extend(accrs); all_vals.extend(rejrs)
            if len(valid_xs) > 1:
                ax.plot(valid_xs, accrs, "-s", color=ACCR_COLOR,
                        linewidth=LINEWIDTH - 0.5, markersize=MARKERSIZE - 3,
                        label="Acc Rec", zorder=3)
                ax.plot(valid_xs, rejrs, "-^", color=REJR_COLOR,
                        linewidth=LINEWIDTH - 0.5, markersize=MARKERSIZE - 3,
                        label="Rej Rec", zorder=3)
            else:
                ax.plot(valid_xs, accrs, "s", color=ACCR_COLOR,
                        markersize=MARKERSIZE - 1, markeredgecolor="black",
                        markeredgewidth=0.5, label="Acc Rec", zorder=4)
                ax.plot(valid_xs, rejrs, "^", color=REJR_COLOR,
                        markersize=MARKERSIZE - 1, markeredgecolor="black",
                        markeredgewidth=0.5, label="Rej Rec", zorder=4)

        # Mini-panel cosmetics
        ax.tick_params(axis="both", labelsize=ticksize - 6)
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)
        # Tag with test ratio (small inset text, top-left)
        ax.text(0.02, 0.92, TEST_LABEL[test_ratio], transform=ax.transAxes,
                fontsize=ticksize - 4, fontweight="bold",
                color=COLOR_PER_TEST[test_ratio],
                ha="left", va="top",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
        # Hide x-tick labels except on bottom
        if sub_row < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Epoch", fontsize=labelsize - 4)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

    # Shared y-range across the three minis (so visual height differences = real differences)
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        span = max(hi - lo, 5.0)
        ylim = (lo - 0.15 * span, hi + 0.25 * span)
        for ax in sub_axes:
            ax.set_ylim(*ylim)

    # Single shared y-label on the top mini (no per-panel legend; column legend at top)
    sub_axes[0].set_ylabel("Recall (%)", fontsize=labelsize - 4)
    return sub_axes


def panel_auc(ax, modality, train_ratio, short, ckpts, save_path, test_datasets):
    """Single panel: AUC vs epoch with one line per test ratio."""
    metric_data = _collect_metrics(modality, train_ratio, short, ckpts, test_datasets, save_path)

    all_aucs: list[float] = []
    for test_ratio in ("50_50", "40_60", "30_70"):
        xs = metric_data[test_ratio]["xs"]
        aucs = [m["auc"] for m in metric_data[test_ratio]["metrics"]]
        valid = [(x, a) for x, a in zip(xs, aucs) if a is not None]
        if not valid:
            continue
        vx, vy = zip(*valid)
        all_aucs.extend(vy)
        color = COLOR_PER_TEST[test_ratio]
        if len(vx) > 1:
            ax.plot(vx, vy, "-o", color=color, linewidth=LINEWIDTH,
                    markersize=MARKERSIZE, label=TEST_LABEL[test_ratio], zorder=3)
        else:
            ax.plot(vx, vy, "*", color=color, markersize=MARKERSIZE + 6,
                    markeredgecolor="black", markeredgewidth=0.8,
                    label=TEST_LABEL[test_ratio] + " (best ckpt)", zorder=4, linestyle="None")
        for x, y in zip(vx, vy):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        fontsize=ticksize - 4, color=color)

    ax.set_xlabel("Epoch", fontsize=labelsize - 2)
    ax.set_ylabel("ROC-AUC (25/26)", fontsize=labelsize)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    if all_aucs:
        lo, hi = min(all_aucs), max(all_aucs)
        span = max(hi - lo, 0.04)
        ax.set_ylim(lo - 0.15 * span, hi + 0.25 * span)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_figure(modality: str, runs, test_data: dict, out_basename: str):
    nrows = len(runs)
    # 4 cols: train loss | test acc | recall mini-stack | AUC
    fig = plt.figure(figsize=(26, 5.5 * nrows))
    outer = fig.add_gridspec(
        nrows, 4,
        width_ratios=[1.0, 1.1, 1.0, 1.1],
        wspace=0.32, hspace=0.55,
    )

    print(f"\n=== {modality.upper()} ===")
    for row, (ratio_label, short, ckpts) in enumerate(runs):
        ratio_disp = ratio_label.replace("_", "/")
        save_path = save_dir(modality, ratio_label, short)
        steps, epochs, losses = load_train_loss(save_path)
        n_log = len(losses)
        print(f"Train {ratio_disp}: {short}  log entries={n_log}  final epoch={epochs[-1] if n_log else 0:.2f}")

        # Col 0: train loss
        ax_loss = fig.add_subplot(outer[row, 0])
        panel_train_loss(ax_loss, steps, epochs, losses, ratio_label)
        ax_loss.set_title(f"Train Loss — {ratio_disp}", fontsize=titlesize - 4, pad=8)

        # Col 1: test accuracy on 25/26
        ax_acc = fig.add_subplot(outer[row, 1])
        panel_test_acc(ax_acc, modality, ratio_label, short, ckpts, save_path, test_data)
        ax_acc.set_title(f"Test Acc 25/26 — {ratio_disp}", fontsize=titlesize - 4, pad=8)

        # Col 2: recall mini-stack (3 sub-rows)
        sub_axes = panel_recall_minis(fig, outer[row, 2], modality, ratio_label,
                                       short, ckpts, save_path, test_data)
        # Title above the top mini
        sub_axes[0].set_title(f"Acc/Rej Recall 25/26 — {ratio_disp}",
                               fontsize=titlesize - 4, pad=8)

        # Col 3: AUC
        ax_auc = fig.add_subplot(outer[row, 3])
        panel_auc(ax_auc, modality, ratio_label, short, ckpts, save_path, test_data)
        ax_auc.set_title(f"ROC-AUC 25/26 — {ratio_disp}",
                         fontsize=titlesize - 4, pad=8)

    fig.suptitle(f"Ratio Sweep — {modality.title()}",
                 fontsize=titlesize + 4, y=0.995, fontweight="bold")

    # Column-level legends placed above row-0 panels.
    from matplotlib.lines import Line2D

    # Test-ratio legend (used for cols 1 (test acc) and 3 (AUC))
    test_handles = [
        Line2D([0], [0], color=COLOR_PER_TEST["50_50"], linewidth=LINEWIDTH,
               marker="o", markersize=MARKERSIZE - 1, label="Test 50/50"),
        Line2D([0], [0], color=COLOR_PER_TEST["40_60"], linewidth=LINEWIDTH,
               marker="o", markersize=MARKERSIZE - 1, label="Test 40/60"),
        Line2D([0], [0], color=COLOR_PER_TEST["30_70"], linewidth=LINEWIDTH,
               marker="o", markersize=MARKERSIZE - 1, label="Test 30/70"),
        Line2D([0], [0], color="gray", marker="*", markersize=MARKERSIZE + 4,
               linestyle="None", markeredgecolor="black", markeredgewidth=0.8,
               label="Best ckpt only"),
    ]
    # Recall legend (col 2)
    recall_handles = [
        Line2D([0], [0], color=ACCR_COLOR, linewidth=LINEWIDTH - 0.5,
               marker="s", markersize=MARKERSIZE - 3, label="Accept Recall"),
        Line2D([0], [0], color=REJR_COLOR, linewidth=LINEWIDTH - 0.5,
               marker="^", markersize=MARKERSIZE - 3, label="Reject Recall"),
    ]

    # Column-center x-positions in figure coordinates.
    # Width ratios are [1.0, 1.1, 1.0, 1.1] = total 4.2.
    # Approx panel centers given matplotlib's default left/right padding.
    # Using bbox_to_anchor in figure coordinates.
    legend_y = 0.965  # just below the suptitle
    col_centers = [0.155, 0.395, 0.625, 0.855]
    legend_kwargs = dict(
        loc="upper center",
        ncol=4,
        fontsize=legendsize - 1,
        framealpha=0.95,
        handletextpad=0.5,
        columnspacing=1.0,
        borderaxespad=0.2,
    )

    # Col 1 (Test Acc): test-ratio legend
    fig.legend(handles=test_handles,
               bbox_to_anchor=(col_centers[1], legend_y),
               bbox_transform=fig.transFigure, **legend_kwargs)
    # Col 2 (Recall mini-stack): recall legend
    fig.legend(handles=recall_handles,
               bbox_to_anchor=(col_centers[2], legend_y),
               bbox_transform=fig.transFigure,
               **{**legend_kwargs, "ncol": 2})
    # Col 3 (AUC): test-ratio legend (same as col 1)
    fig.legend(handles=test_handles,
               bbox_to_anchor=(col_centers[3], legend_y),
               bbox_transform=fig.transFigure, **legend_kwargs)

    # Make room above subplots for the legends
    fig.subplots_adjust(top=0.91)

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
