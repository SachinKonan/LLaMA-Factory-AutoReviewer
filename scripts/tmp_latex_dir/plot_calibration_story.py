#!/usr/bin/env python3
"""
Two figures showing the calibration / cross-distribution / bias story:

  fig_iclr_crosseval.pdf : 3-panel — raw ACC, calibrated ACC (Δ-colored), AUC.
                           Each cell shows [25 / 26] year-split numbers.
  fig_calibration_bias.pdf : 2-panel — arxiv per-venue dumbbells (raw→calib),
                             pred-A% vs gold-A% bias scatter.

Source data: ICLR + arxiv y24up text cross-eval at ckpt 1322.
Computation logic mirrors scripts/iclr_arxiv_calibrated_eval.py.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})
LSIZE, TSIZE, TICK = 18, 18, 14

ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
RES  = ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
DATA = ROOT / "data"
OUT  = ROOT / "tmp_latex_dir/figures"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = 1322

# ---------- ICLR config ----------
ICLR_TEST = {
    "50_50": DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
    "40_60": DATA / "iclr_2020_2023_2025_2026_40_60_original_text_v7_filtered_test/data.json",
    "30_70": DATA / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_test/data.json",
}
ICLR_VAL = {
    "50_50": DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation/data.json",
    "40_60": DATA / "iclr_2020_2023_2025_2026_40_60_original_text_v7_filtered_validation_DERIVED/data.json",
    "30_70": DATA / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_validation_DERIVED/data.json",
}
ICLR_SHORTS = {"50_50": "bz32_lr1e-6_text",
               "40_60": "bz32_lr1e-6_text_40_60",
               "30_70": "bz32_lr1e-6_text_30_70"}
RATIOS = ["50_50", "40_60", "30_70"]
LBL = {"50_50": "50/50", "40_60": "40/60", "30_70": "30/70"}

ARXIV_RES = ROOT / "results/cross_conference_arxiv_y24up"
ARXIV_TEST_PATH = DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test/data.json"
ARXIV_VAL_PATH  = DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation/data.json"
ACL_FAMILY = {"acl", "emnlp", "naacl"}

RATIO_COLOR = {"50_50": "#6098FF", "40_60": "#FECC81", "30_70": "#7FB069"}

# ---------- helpers ----------
def extract_pred(t):
    s = (t or "").lower()
    if "\\boxed{accept}" in s or "boxed{accept}" in s: return "accept"
    if "\\boxed{reject}" in s or "boxed{reject}" in s: return "reject"
    if "accept" in s: return "accept"
    if "reject" in s: return "reject"
    return None

def score_logodds(row):
    p = row.get("predict","")
    chosen = 1 if "Accept" in p else (0 if "Reject" in p else None)
    if chosen is None: return None
    la = row.get("logprob_accept") or []
    lr = row.get("logprob_reject") or []
    tl = row.get("token_logprobs") or []
    for k in range(min(len(tl), len(la), len(lr))):
        if la[k] is not None and lr[k] is not None and tl[k] is not None:
            if math.isclose(tl[k], max(la[k], lr[k]), abs_tol=1e-3):
                return la[k] - lr[k]
    return 6.0 if chosen == 1 else -6.0

def auc(scores, labels):
    if not scores or len(set(labels)) < 2: return None
    pairs = sorted(zip(scores, labels))
    n = len(pairs); npos = sum(labels); nneg = n - npos
    rs = sum(i+1 for i,(s,l) in enumerate(pairs) if l == 1)
    return (rs - npos*(npos+1)/2) / (npos*nneg)

def best_tau(pairs):
    if not pairs: return 0.0
    pairs = sorted(pairs)
    n = len(pairs); npos = sum(g for _,g in pairs)
    correct = npos
    best_acc, best_tau = correct/n, -math.inf
    for s,g in pairs:
        correct += (1 if g == 0 else -1)
        if correct/n > best_acc:
            best_acc, best_tau = correct/n, s
    return best_tau

def load_pairs(jsonl_path, meta_path, year_min=None, with_meta=False):
    if not jsonl_path.exists() or not meta_path.exists(): return []
    meta = json.loads(meta_path.read_text())
    out = []
    with jsonl_path.open() as f:
        for i, line in enumerate(f):
            if i >= len(meta): break
            yr = (meta[i].get("_metadata") or {}).get("year")
            try: yr = int(yr) if yr is not None else None
            except: pass
            if year_min is not None and (yr is None or yr < year_min): continue
            r = json.loads(line)
            g = extract_pred(r.get("label",""))
            s = score_logodds(r)
            if g is None or s is None: continue
            gold = 1 if g == "accept" else 0
            if with_meta:
                m = meta[i].get("_metadata") or {}
                v = (m.get("pl_venue") or m.get("venue") or "?").lower()
                if v in ACL_FAMILY: v = "acl_family"
                out.append((yr, gold, s, v))
            else:
                out.append((yr, gold, s))
    return out

# ---------- ICLR cell metrics (matches eval script) ----------
def iclr_test_path(train, test):
    short = ICLR_SHORTS[train]
    if train == "50_50":
        if test == "50_50":
            return RES / short / f"finetuned-ckpt-{CKPT}.jsonl"
        return RES / "ratio_crossval_clean" / short / f"test_{test}" / f"finetuned-ckpt-{CKPT}.jsonl"
    tag = "balanced" if test == "50_50" else test
    return RES / "ratio_sweep" / short / tag / f"finetuned-ckpt-{CKPT}.jsonl"

def iclr_val_path(train, val):
    short = ICLR_SHORTS[train]
    if train == "50_50" and val == "50_50":
        return RES / short / f"validation-ckpt-{CKPT}.jsonl"
    val_tag = "balanced" if val == "50_50" else val
    return ROOT / "results/iclr_val_calib" / short / f"val_{val_tag}" / f"finetuned-ckpt-{CKPT}.jsonl"

def iclr_cell(train, test):
    test_pairs = load_pairs(iclr_test_path(train, test), ICLR_TEST[test], year_min=2025)
    val_pairs  = load_pairs(iclr_val_path(train, test),  ICLR_VAL[test],  year_min=2025)
    if not val_pairs:
        val_pairs = load_pairs(iclr_val_path(train, test), ICLR_VAL[test], year_min=None)
    tau = best_tau([(s,g) for _,g,s in val_pairs])
    out = {"tau": tau}
    for yr in (2025, 2026):
        sub = [(g,s) for y,g,s in test_pairs if y == yr]
        n = len(sub)
        if n == 0:
            out[yr] = dict(acc_raw=None, acc_cal=None, auc=None, n=0,
                           pred_a=None, gold_a=None)
            continue
        acc_raw = sum(1 for g,s in sub if (s>0)==(g==1))/n*100
        acc_cal = sum(1 for g,s in sub if (s>tau)==(g==1))/n*100
        scores = [s for g,s in sub]; gold = [g for g,s in sub]
        au = auc(scores, gold)
        pred_a = sum(1 for g,s in sub if s>0)/n*100  # raw predicted accept rate
        gold_a = sum(g for g,s in sub)/n*100
        out[yr] = dict(acc_raw=acc_raw, acc_cal=acc_cal, auc=au, n=n,
                       pred_a=pred_a, gold_a=gold_a)
    return out


# =======================================================
# FIGURE 1: ICLR 3x3 — Raw ACC | Calibrated ACC | AUC
# =======================================================
def fig1_iclr_crosseval():
    cells = {(t, e): iclr_cell(t, e) for t in RATIOS for e in RATIOS}

    # Build matrices for color encoding
    raw_avg = np.zeros((3, 3))
    cal_avg = np.zeros((3, 3))
    auc_avg = np.zeros((3, 3))
    for i, t in enumerate(RATIOS):
        for j, e in enumerate(RATIOS):
            c = cells[(t, e)]
            raw_avg[i, j] = (c[2025]["acc_raw"] + c[2026]["acc_raw"]) / 2
            cal_avg[i, j] = (c[2025]["acc_cal"] + c[2026]["acc_cal"]) / 2
            auc_avg[i, j] = (c[2025]["auc"]    + c[2026]["auc"])    / 2
    delta = cal_avg - raw_avg

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2),
                              gridspec_kw={"wspace": 0.35})

    def draw_heatmap(ax, mat, vmin, vmax, cmap, title, cell_text_fn, cbar_label):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(3)); ax.set_xticklabels([LBL[r] for r in RATIOS], fontsize=TICK)
        ax.set_yticks(range(3)); ax.set_yticklabels([LBL[r] for r in RATIOS], fontsize=TICK)
        ax.set_xlabel("Test ratio", fontsize=LSIZE - 2)
        ax.set_ylabel("Train ratio", fontsize=LSIZE - 2)
        ax.set_title(title, fontsize=TSIZE - 2, pad=10)
        for i, t in enumerate(RATIOS):
            for j, e in enumerate(RATIOS):
                ax.text(j, i, cell_text_fn(cells[(t, e)]),
                        ha="center", va="center", fontsize=TICK,
                        color="black", linespacing=1.0)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label, fontsize=TICK - 1)
        cb.ax.tick_params(labelsize=TICK - 2)

    # Panel 1: Raw ACC
    draw_heatmap(
        axes[0], raw_avg,
        vmin=raw_avg.min() - 1, vmax=raw_avg.max() + 1, cmap="YlGnBu",
        title="Raw ACC (no calibration)",
        cell_text_fn=lambda c: f"{c[2025]['acc_raw']:.1f}\n{c[2026]['acc_raw']:.1f}",
        cbar_label="Avg ACC (25/26)",
    )

    # Panel 2: Calibrated ACC, color = delta from raw
    dmax = max(abs(delta.min()), abs(delta.max()))
    draw_heatmap(
        axes[1], delta,
        vmin=-dmax, vmax=dmax, cmap="RdBu_r",
        title="Calibrated ACC (Δ from raw)",
        cell_text_fn=lambda c: f"{c[2025]['acc_cal']:.1f}\n{c[2026]['acc_cal']:.1f}",
        cbar_label="Δ ACC (calib − raw)",
    )

    # Panel 3: AUC
    draw_heatmap(
        axes[2], auc_avg,
        vmin=auc_avg.min() - 0.01, vmax=auc_avg.max() + 0.01, cmap="viridis",
        title="ROC-AUC",
        cell_text_fn=lambda c: f"{c[2025]['auc']:.3f}\n{c[2026]['auc']:.3f}",
        cbar_label="Avg AUC (25/26)",
    )

    fig.suptitle(
        "ICLR 3×3 cross-eval (text, ckpt 1322, 25/26 in each cell as 25 over 26)",
        fontsize=TSIZE, y=1.02, fontweight="bold")
    out_pdf = OUT / "fig_iclr_crosseval.pdf"
    out_png = OUT / "fig_iclr_crosseval.png"
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"saved {out_pdf}, {out_png}")
    return cells


# =======================================================
# FIGURE 2: Arxiv calibration dumbbells | ICLR bias scatter
# =======================================================
def fig2_calibration_and_bias(iclr_cells):
    fig = plt.figure(figsize=(20, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.28)

    # ---- Panel 1: Arxiv per-venue dumbbells ----
    ax1 = fig.add_subplot(gs[0, 0])

    arxiv_data = {}
    for ratio in RATIOS:
        short = ICLR_SHORTS[ratio]
        val_pairs  = load_pairs(ARXIV_RES / short / "arxiv_val"  / f"finetuned-ckpt-{CKPT}.jsonl",
                                ARXIV_VAL_PATH, with_meta=True)
        test_pairs = load_pairs(ARXIV_RES / short / "arxiv_eval" / f"finetuned-ckpt-{CKPT}.jsonl",
                                ARXIV_TEST_PATH, with_meta=True)
        val_by_v  = defaultdict(list)
        test_by_v = defaultdict(list)
        for _,g,s,v in val_pairs:  val_by_v[v].append((s,g))
        for _,g,s,v in test_pairs: test_by_v[v].append((s,g))
        global_tau = best_tau([(s,g) for _,g,s,_ in val_pairs])
        thr = {}
        for v, pairs in val_by_v.items():
            thr[v] = best_tau(pairs) if len(pairs) >= 10 else global_tau
        for v in test_by_v:
            thr.setdefault(v, global_tau)
        arxiv_data[ratio] = {"test_by_v": test_by_v, "thr": thr}

    # Sort venues by total test count (desc) so big ones are at top
    venues = sorted({v for r in RATIOS for v in arxiv_data[r]["test_by_v"]},
                    key=lambda v: -sum(len(arxiv_data[r]["test_by_v"].get(v, [])) for r in RATIOS))

    n_v = len(venues)
    y_positions = np.arange(n_v)
    bar_offsets = {"50_50": -0.25, "40_60": 0.0, "30_70": +0.25}

    for ratio in RATIOS:
        d = arxiv_data[ratio]
        col = RATIO_COLOR[ratio]
        offs = bar_offsets[ratio]
        for i, v in enumerate(venues):
            tp = d["test_by_v"].get(v, [])
            if not tp: continue
            n = len(tp)
            tau = d["thr"][v]
            raw = sum(1 for s,g in tp if (s>0)==(g==1)) / n * 100
            cal = sum(1 for s,g in tp if (s>tau)==(g==1)) / n * 100
            y = i + offs
            ax1.plot([raw, cal], [y, y], color=col, linewidth=2.5, alpha=0.55, zorder=2)
            ax1.scatter([raw], [y], color=col, marker="o", s=70, zorder=3,
                        edgecolor="black", linewidth=0.5,
                        label=f"{LBL[ratio]} raw" if i == 0 else None)
            ax1.scatter([cal], [y], color=col, marker="D", s=70, zorder=4,
                        edgecolor="black", linewidth=0.5,
                        label=f"{LBL[ratio]} calib" if i == 0 else None)

    ax1.set_yticks(range(n_v)); ax1.set_yticklabels(venues, fontsize=TICK)
    ax1.invert_yaxis()
    ax1.set_xlabel("Accuracy (%)", fontsize=LSIZE - 2)
    ax1.set_xlim(20, 105)
    ax1.axvline(x=50, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    ax1.set_title("Arxiv per-venue: raw → calibrated accuracy",
                  fontsize=TSIZE - 2, pad=10)
    ax1.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax1.legend(fontsize=TICK - 1, loc="lower right", ncol=3,
               framealpha=0.95, columnspacing=1.0)

    # ---- Panel 2: Bias scatter (pred-A% vs gold-A%) ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot([0, 100], [0, 100], "--", color="gray", alpha=0.6, linewidth=1, label="no bias (y=x)")

    for t in RATIOS:
        for e in RATIOS:
            c = iclr_cells[(t, e)]
            for yr, marker in [(2025, "o"), (2026, "s")]:
                row = c[yr]
                if row["pred_a"] is None: continue
                ax2.scatter([row["gold_a"]], [row["pred_a"]],
                            color=RATIO_COLOR[t], marker=marker, s=110,
                            edgecolor="black", linewidth=0.6, zorder=3)

    # Legend (per train ratio + year markers)
    for t in RATIOS:
        ax2.scatter([], [], color=RATIO_COLOR[t], s=110, edgecolor="black",
                    linewidth=0.6, label=f"Train {LBL[t]}")
    ax2.scatter([], [], color="gray", marker="o", s=110, edgecolor="black",
                linewidth=0.6, label="2025")
    ax2.scatter([], [], color="gray", marker="s", s=110, edgecolor="black",
                linewidth=0.6, label="2026")

    ax2.set_xlabel("Gold accept rate (%)", fontsize=LSIZE - 2)
    ax2.set_ylabel("Predicted accept rate (%, raw τ=0)", fontsize=LSIZE - 2)
    ax2.set_xlim(20, 80); ax2.set_ylim(0, 100)
    ax2.set_title("ICLR bias direction (raw decoding)",
                  fontsize=TSIZE - 2, pad=10)
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(fontsize=TICK - 1, loc="upper left", framealpha=0.95)

    # Annotate "above y=x → over-accepts; below → over-rejects"
    ax2.text(0.95, 0.05, "above line: over-accepts\nbelow line: over-rejects",
             transform=ax2.transAxes, fontsize=TICK - 2, ha="right", va="bottom",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

    fig.suptitle("Calibration convergence (left) + Bias direction (right)",
                 fontsize=TSIZE, y=1.02, fontweight="bold")

    out_pdf = OUT / "fig_calibration_bias.pdf"
    out_png = OUT / "fig_calibration_bias.png"
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"saved {out_pdf}, {out_png}")


def main():
    cells = fig1_iclr_crosseval()
    fig2_calibration_and_bias(cells)


if __name__ == "__main__":
    main()
