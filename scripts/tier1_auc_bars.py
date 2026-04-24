#!/usr/bin/env python3
"""Roll up confusion matrices into AUC: 3x1 bar charts per modality.

Layout: one figure per modality, 3 rows (one per train ratio), each row a
bar chart of ROC-AUC across the 3 test ratios (50/50, 40/60, 30/70).

Score per sample (log-odds for "Accept"):
  - preferred: lp_accept - lp_reject at the decision step when both are
    present in the inference jsonl;
  - fallback: derived from the chosen token + exp(min(token_logprobs)) so
    that if the model said Accept we use p(accept) and if it said Reject we
    use 1 - p(reject). Score = logit(p_accept).

Filtered to 2025+2026. Epoch-2 checkpoints throughout. Cells without
inference results render as greyed-out bars.

Usage:
  python scripts/tier1_auc_bars.py --modality text
  python scripts/tier1_auc_bars.py --modality vision
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
except ImportError as e:
    raise SystemExit("sklearn required: source .venv/bin/activate") from e

# -- paper style ------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "helvetica",
})
labelsize = 20
titlesize = 20
legendsize = 20
ticksize = 16

# Palette (reuses colors the user shared; one per test ratio).
COLOR_PER_TEST = {
    "50_50": "#6098FF",  # blue
    "40_60": "#77B25D",  # green
    "30_70": "#B28CFF",  # purple
}
PENDING_FACE = "#E8E8E8"
PENDING_EDGE = "#888888"

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
_EPS = 1e-6


def _clip(p: float) -> float:
    return min(max(p, _EPS), 1.0 - _EPS)


def score_from_row(row: dict) -> float | None:
    """Log-odds for Accept, or None if unparseable."""
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

    # Preferred: true log-odds from decision step where Accept/Reject emitted.
    for k in range(min(len(tl), len(la), len(lr))):
        if la[k] is not None and lr[k] is not None:
            a, r = la[k], lr[k]
            if math.isclose(tl[k], max(a, r), abs_tol=1e-3):
                return a - r

    # Fallback: p_chosen = exp(min(token_logprobs)) at the most uncertain step.
    if not tl:
        return 6.0 if chosen == 1 else -6.0
    min_lp = min(tl)
    p_chosen = _clip(math.exp(min_lp))
    p_accept = p_chosen if chosen == 1 else 1.0 - p_chosen
    p_accept = _clip(p_accept)
    score = math.log(p_accept / (1.0 - p_accept))
    # Don't let the approximation contradict the greedy choice.
    if chosen == 1 and score <= 0:
        return _EPS
    if chosen == 0 and score >= 0:
        return -_EPS
    return score


def jsonl_for(modality, train, test):
    short = SHORT[(modality, train)]
    step = EPOCH2[(modality, train)]
    if train == "50_50":
        if test == "50_50":
            return ROOT / short / f"finetuned-ckpt-{step}.jsonl"
        return ROOT / "ratio_crossval_clean" / short / f"test_{test}" / f"finetuned-ckpt-{step}.jsonl"
    tag = "balanced" if test == "50_50" else test
    return ROOT / "ratio_sweep" / short / tag / f"finetuned-ckpt-{step}.jsonl"


def auc_for(modality, train, test) -> tuple[float | None, int]:
    jp = jsonl_for(modality, train, test)
    if not jp.exists():
        return None, 0
    entries = json.loads(TEST_DATA[(modality, test)].read_text())
    keep = {i for i, e in enumerate(entries) if (e.get("_metadata") or {}).get("year") in YEARS}
    y, s = [], []
    with jp.open() as fh:
        for i, line in enumerate(fh):
            if i not in keep:
                continue
            r = json.loads(line)
            gold = 1 if "Accept" in r.get("label", "") else 0 if "Reject" in r.get("label", "") else None
            if gold is None:
                continue
            sc = score_from_row(r)
            if sc is None:
                continue
            y.append(gold); s.append(sc)
    if not y:
        return None, 0
    try:
        auc = roc_auc_score(y, s)
    except ValueError:
        auc = None
    return auc, len(y)


# -- plotting ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["text", "vision"], default="text")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    mod = args.modality
    out = Path(args.output or f"tmp_latex_dir/figures/auc_bars_{mod}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Collect 3x3 AUC grid
    grid = {}
    for tr in RATIOS:
        for ts in RATIOS:
            grid[(tr, ts)] = auc_for(mod, tr, ts)

    # 3x1 layout (rows = train ratio), each row is a bar chart across test ratios.
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle(
        f"\\textbf{{{mod.capitalize()}}} --- ROC-AUC (2025+2026)",
        fontsize=titlesize + 2, y=0.995,
    )
    plt.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.08, hspace=0.80)

    for r_idx, train in enumerate(RATIOS):
        ax = axes[r_idx]
        xs = np.arange(len(RATIOS))
        labels = [f"test {t.replace('_','/')}" for t in RATIOS]
        for i, ts in enumerate(RATIOS):
            auc, n = grid[(train, ts)]
            if auc is None:
                ax.bar(i, 0.5, color=PENDING_FACE, edgecolor=PENDING_EDGE,
                       linestyle="--", linewidth=1.5, hatch="//", alpha=0.7)
                ax.text(i, 0.52, "pending", ha="center", va="bottom",
                        fontsize=ticksize, color="#666")
            else:
                ax.bar(i, auc, color=COLOR_PER_TEST[ts],
                       edgecolor="black", linewidth=1.2, alpha=0.95)
                ax.text(i, auc + 0.01, f"{auc:.3f}", ha="center", va="bottom",
                        fontsize=ticksize, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=ticksize)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("ROC-AUC", fontsize=labelsize)
        ax.tick_params(axis="y", labelsize=ticksize)
        ax.set_title(f"\\textbf{{train {train.replace('_','/')}}}",
                     fontsize=labelsize, pad=10)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
        ax.set_axisbelow(True)
        for spine_name in ("top", "right"):
            ax.spines[spine_name].set_visible(False)

    plt.savefig(out, dpi=200, bbox_inches="tight")
    png = out.with_suffix(".png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out}")
    print(f"saved: {png}")


if __name__ == "__main__":
    main()
