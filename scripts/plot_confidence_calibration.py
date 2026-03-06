#!/usr/bin/env python3
"""
Confidence / calibration analysis plots using prediction files with token_logprobs.

Creates:
  - figures/confidence_calibration.png  (2x2 panel)
  - figures/coverage_vs_accuracy.png    (standalone)
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

VISION_PATH = (
    PROJECT_ROOT
    / "results/final_sweep_v7_datasweepv3/wd_sweep_expdecay"
    / "bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-3990.jsonl"
)
TEXT_PATH = (
    PROJECT_ROOT
    / "results/final_sweep_v7_datasweepv3/wd_sweep_expdecay"
    / "bz16_lr1e-6_wd0.001_text/finetuned-ckpt-1594.jsonl"
)

DECISION_TOKEN_IDX = 5

LABELSIZE = 13
TITLESIZE = 14
LEGENDSIZE = 9
TICKSIZE = 10
DPI = 150

COLOR_TEXT = "#1f77b4"
COLOR_VISION = "#ff7f0e"

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


def load_predictions(path: Path):
    """Return arrays: confidences, correctness (bool), predictions."""
    confidences = []
    correct = []
    preds = []
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            pred = extract_prediction(rec["predict"])
            label = extract_prediction(rec["label"])
            logprobs = rec["token_logprobs"]
            conf = math.exp(logprobs[DECISION_TOKEN_IDX])
            confidences.append(conf)
            correct.append(pred == label and pred != "unknown")
            preds.append(pred)
    return np.array(confidences), np.array(correct), preds


def calibration_curve(confidences, correct, n_bins=10, low=0.5, high=1.0):
    """Bin by confidence and return (mean_conf, accuracy, count) per bin."""
    bin_edges = np.linspace(low, high, n_bins + 1)
    mean_confs, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        n = mask.sum()
        if n == 0:
            mean_confs.append((lo + hi) / 2)
            accs.append(np.nan)
            counts.append(0)
        else:
            mean_confs.append(confidences[mask].mean())
            accs.append(correct[mask].mean())
            counts.append(int(n))
    return np.array(mean_confs), np.array(accs), np.array(counts)


def coverage_accuracy_sweep(confidences, correct, thresholds):
    """For each threshold return (accuracy, coverage)."""
    accs, covs = [], []
    n = len(confidences)
    for t in thresholds:
        mask = confidences >= t
        cov = mask.sum() / n
        acc = correct[mask].mean() if mask.sum() > 0 else np.nan
        accs.append(acc)
        covs.append(cov)
    return np.array(accs), np.array(covs)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading predictions ...")
conf_vis, corr_vis, preds_vis = load_predictions(VISION_PATH)
conf_txt, corr_txt, preds_txt = load_predictions(TEXT_PATH)
print(f"  Vision: {len(conf_vis)} samples, acc={corr_vis.mean():.4f}")
print(f"  Text:   {len(conf_txt)} samples, acc={corr_txt.mean():.4f}")

# ---------------------------------------------------------------------------
# Build 2x2 figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ---- Subplot 1: Reliability diagram (top-left) --------------------------
ax = axes[0, 0]
n_bins = 10
mc_txt, acc_txt, cnt_txt = calibration_curve(conf_txt, corr_txt, n_bins=n_bins)
mc_vis, acc_vis, cnt_vis = calibration_curve(conf_vis, corr_vis, n_bins=n_bins)

ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1, label="Perfect calibration")
ax.plot(mc_txt, acc_txt, "o-", color=COLOR_TEXT, linewidth=1.5, markersize=5, label="Text")
ax.plot(mc_vis, acc_vis, "s-", color=COLOR_VISION, linewidth=1.5, markersize=5, label="Vision")

# Annotate bin counts
for x, y, c in zip(mc_txt, acc_txt, cnt_txt):
    if not np.isnan(y) and c > 0:
        ax.annotate(str(c), (x, y), textcoords="offset points", xytext=(-8, 8),
                    fontsize=7, color=COLOR_TEXT, alpha=0.8)
for x, y, c in zip(mc_vis, acc_vis, cnt_vis):
    if not np.isnan(y) and c > 0:
        ax.annotate(str(c), (x, y), textcoords="offset points", xytext=(4, -12),
                    fontsize=7, color=COLOR_VISION, alpha=0.8)

ax.set_xlabel("Mean Predicted Confidence", fontsize=LABELSIZE)
ax.set_ylabel("Actual Accuracy", fontsize=LABELSIZE)
ax.set_title("Reliability Diagram", fontsize=TITLESIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0.3, 1.05)
ax.tick_params(labelsize=TICKSIZE)
ax.grid(True, alpha=0.3)

# ---- Subplot 2: Coverage vs Accuracy tradeoff (top-right) ---------------
ax = axes[0, 1]
thresholds = np.linspace(0.5, 1.0, 100)
acc_t, cov_t = coverage_accuracy_sweep(conf_txt, corr_txt, thresholds)
acc_v, cov_v = coverage_accuracy_sweep(conf_vis, corr_vis, thresholds)

l1, = ax.plot(thresholds, acc_t, "-", color=COLOR_TEXT, linewidth=1.5, label="Text Accuracy")
l2, = ax.plot(thresholds, acc_v, "-", color=COLOR_VISION, linewidth=1.5, label="Vision Accuracy")
ax.set_xlabel("Confidence Threshold", fontsize=LABELSIZE)
ax.set_ylabel("Accuracy (above threshold)", fontsize=LABELSIZE)
ax.tick_params(labelsize=TICKSIZE)

ax2 = ax.twinx()
l3, = ax2.plot(thresholds, cov_t, "--", color=COLOR_TEXT, linewidth=1.2, alpha=0.7, label="Text Coverage")
l4, = ax2.plot(thresholds, cov_v, "--", color=COLOR_VISION, linewidth=1.2, alpha=0.7, label="Vision Coverage")
ax2.set_ylabel("Coverage (fraction above threshold)", fontsize=LABELSIZE)
ax2.tick_params(labelsize=TICKSIZE)

lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, fontsize=LEGENDSIZE, loc="center left")
ax.set_title("Coverage vs Accuracy Tradeoff", fontsize=TITLESIZE)
ax.set_xlim(0.5, 1.0)
ax.grid(True, alpha=0.3)

# ---- Subplot 3: Confidence distribution correct vs incorrect (bot-left) --
ax = axes[1, 0]
bins_hist = np.linspace(0.4, 1.0, 40)

ax.hist(conf_txt[corr_txt], bins=bins_hist, alpha=0.45, color=COLOR_TEXT, label="Text Correct", density=True)
ax.hist(conf_txt[~corr_txt], bins=bins_hist, alpha=0.45, color=COLOR_TEXT, label="Text Incorrect",
        density=True, hatch="//", edgecolor=COLOR_TEXT, linewidth=0.5)
ax.hist(conf_vis[corr_vis], bins=bins_hist, alpha=0.35, color=COLOR_VISION, label="Vision Correct", density=True)
ax.hist(conf_vis[~corr_vis], bins=bins_hist, alpha=0.35, color=COLOR_VISION, label="Vision Incorrect",
        density=True, hatch="\\\\", edgecolor=COLOR_VISION, linewidth=0.5)

ax.set_xlabel("Confidence", fontsize=LABELSIZE)
ax.set_ylabel("Density", fontsize=LABELSIZE)
ax.set_title("Confidence Distribution (Correct vs Incorrect)", fontsize=TITLESIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.tick_params(labelsize=TICKSIZE)
ax.grid(True, alpha=0.3)

# ---- Subplot 4: Coverage vs Accuracy parametric curve (bot-right) --------
ax = axes[1, 1]
# Sweep threshold from 1.0 down to 0.5 so coverage goes from ~0 to 1
thresholds_sweep = np.linspace(1.0, 0.5, 200)
acc_t_s, cov_t_s = coverage_accuracy_sweep(conf_txt, corr_txt, thresholds_sweep)
acc_v_s, cov_v_s = coverage_accuracy_sweep(conf_vis, corr_vis, thresholds_sweep)

ax.plot(cov_t_s, acc_t_s, "-", color=COLOR_TEXT, linewidth=1.5, label="Text")
ax.plot(cov_v_s, acc_v_s, "-", color=COLOR_VISION, linewidth=1.5, label="Vision")

# Annotate key coverage points
for target_cov in [0.50, 0.75, 0.90]:
    for cov_arr, acc_arr, color, name, offset_y in [
        (cov_t_s, acc_t_s, COLOR_TEXT, "T", 8),
        (cov_v_s, acc_v_s, COLOR_VISION, "V", -14),
    ]:
        idx = np.argmin(np.abs(cov_arr - target_cov))
        if not np.isnan(acc_arr[idx]):
            ax.plot(cov_arr[idx], acc_arr[idx], "o", color=color, markersize=5)
            ax.annotate(
                f"{name} {target_cov:.0%}: {acc_arr[idx]:.1%}",
                (cov_arr[idx], acc_arr[idx]),
                textcoords="offset points",
                xytext=(6, offset_y),
                fontsize=7,
                color=color,
            )

ax.set_xlabel("Coverage (fraction of predictions kept)", fontsize=LABELSIZE)
ax.set_ylabel("Accuracy on Kept Predictions", fontsize=LABELSIZE)
ax.set_title("Coverage vs Accuracy Curve", fontsize=TITLESIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.set_xlim(-0.02, 1.05)
ax.tick_params(labelsize=TICKSIZE)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "confidence_calibration.png", dpi=DPI, bbox_inches="tight")
print(f"Saved {FIGURES_DIR / 'confidence_calibration.png'}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Standalone coverage vs accuracy figure
# ---------------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(8, 6))

ax.plot(cov_t_s, acc_t_s, "-", color=COLOR_TEXT, linewidth=2, label="Text")
ax.plot(cov_v_s, acc_v_s, "-", color=COLOR_VISION, linewidth=2, label="Vision")

for target_cov in [0.50, 0.75, 0.90]:
    for cov_arr, acc_arr, color, name, offset_y in [
        (cov_t_s, acc_t_s, COLOR_TEXT, "Text", 10),
        (cov_v_s, acc_v_s, COLOR_VISION, "Vision", -16),
    ]:
        idx = np.argmin(np.abs(cov_arr - target_cov))
        if not np.isnan(acc_arr[idx]):
            ax.plot(cov_arr[idx], acc_arr[idx], "o", color=color, markersize=7)
            ax.annotate(
                f"{name} @{target_cov:.0%}: {acc_arr[idx]:.1%}",
                (cov_arr[idx], acc_arr[idx]),
                textcoords="offset points",
                xytext=(8, offset_y),
                fontsize=9,
                color=color,
            )

ax.set_xlabel("Coverage (fraction of predictions kept)", fontsize=LABELSIZE + 1)
ax.set_ylabel("Accuracy on Kept Predictions", fontsize=LABELSIZE + 1)
ax.set_title("Coverage vs Accuracy Tradeoff", fontsize=TITLESIZE + 1)
ax.legend(fontsize=LEGENDSIZE + 2)
ax.set_xlim(-0.02, 1.05)
ax.tick_params(labelsize=TICKSIZE + 1)
ax.grid(True, alpha=0.3)

fig2.savefig(FIGURES_DIR / "coverage_vs_accuracy.png", dpi=DPI, bbox_inches="tight")
print(f"Saved {FIGURES_DIR / 'coverage_vs_accuracy.png'}")
plt.close(fig2)

print("Done.")
