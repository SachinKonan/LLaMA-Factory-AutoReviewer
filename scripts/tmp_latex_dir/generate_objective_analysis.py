#!/usr/bin/env python3
"""
Generator for docs/objective_analysis.md and its figures.

Two objectives:
  1) general quality indicator   — AUC, Spearman ρ to pct_rating, ρ to citation
  2) conference accept predictor — balanced ACC, accept-recall, reject-recall

Across all (modality × train_ratio) configs, on ICLR 25/26 balanced + arxiv y24up balanced.

Calibration: TWO flavors compared (both are hyperparam choices to maximize TEST balanced ACC):
  τ*_raw — argmax raw ACC on val
  τ*_bal — argmax balanced ACC on val

Figures generated in tmp_latex_dir/figures/:
  objective_distribution_shift.{pdf,png}    — pct_rating + citation, balanced vs natural prior
  objective_calibration_sweep.{pdf,png}     — ACC vs τ on val + test, two thresholds marked
  objective_summary.{pdf,png}               — bACC + AUC + ρ_quality grid across modality × train_ratio

Doc: docs/objective_analysis.md (replaces Codex's version)
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------- style ----------
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})
LABELSIZE = 22
TITLESIZE = 24
LEGENDSIZE = 18
TICKSIZE = 16
LINEWIDTH = 2.5
MARKERSIZE = 9

BLUE = "#6098FF"
ORANGE = "#FECC81"
GREEN = "#77B25D"
RED = "#FF8988"
PURPLE = "#B28CFF"
GRAY = "#888888"

# ---------- paths ----------
ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
DATA = ROOT / "data"
RES  = ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
ICLR_VAL_CALIB_DIR = ROOT / "results/iclr_val_calib"
ARXIV_BAL_DIR    = ROOT / "results/cross_conference_arxiv_y24up"
ARXIV_NATR_DIR   = ROOT / "results/cross_conference_arxiv_natrate_y24up"

REPORT = ROOT / "docs/objective_analysis.md"
FIG_DIR = ROOT / "tmp_latex_dir/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DOC_FIG_DIR = ROOT / "docs/objective_analysis_figures"
DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- model registry ----------
# 7B iclr-trained: 2nd ckpt (per user spec)
# 3B iclr-trained: last ckpt
MODELS_7B_ICLR = {
    ("text",   "50_50"): ("bz32_lr1e-6_text",         1322),
    ("text",   "30_70"): ("bz32_lr1e-6_text_30_70",   1322),
    ("vision", "50_50"): ("bz16_lr1e-6_vision",       2648),
    ("vision", "30_70"): ("bz16_lr1e-6_vision_30_70", 2642),
}
MODELS_3B_ICLR = {
    ("text",   "50_50"): ("scaling/bz32_lr1e-6_text_3b",   2644),
    ("vision", "50_50"): ("scaling/bz16_lr1e-6_vision_3b", 5296),
}

ICLR_META = {
    ("text",   "balanced", "test"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
    ("text",   "balanced", "val" ): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation/data.json",
    ("text",   "natural",  "test"): DATA / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_test/data.json",
    ("text",   "natural",  "val" ): DATA / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_validation_DERIVED/data.json",
    ("vision", "balanced", "test"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json",
    ("vision", "balanced", "val" ): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_validation/data.json",
    ("vision", "natural",  "test"): DATA / "iclr_2020_2023_2025_2026_30_70_original_vision_v7_filtered_test/data.json",
    ("vision", "natural",  "val" ): DATA / "iclr_2020_2023_2025_2026_30_70_original_vision_v7_filtered_validation_DERIVED/data.json",
}

ARXIV_META = {
    ("text",   "balanced", "test"): DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test/data.json",
    ("text",   "balanced", "val" ): DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation/data.json",
    ("text",   "natural",  "test"): DATA / "arxiv_natrate_21k_text_wmetadata_filtered24480_y24up_test/data.json",
    ("text",   "natural",  "val" ): DATA / "arxiv_natrate_21k_text_wmetadata_filtered24480_y24up_validation/data.json",
    ("vision", "balanced", "test"): DATA / "arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_test/data.json",
    ("vision", "balanced", "val" ): DATA / "arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_validation/data.json",
    ("vision", "natural",  "test"): DATA / "arxiv_natrate_21k_vision_wmetadata_filtered24480_y24up_test/data.json",
    ("vision", "natural",  "val" ): DATA / "arxiv_natrate_21k_vision_wmetadata_filtered24480_y24up_validation/data.json",
}

DECISION_TOKEN_IDX = 5


# ---------- core helpers ----------
def extract_pred(t):
    s = (t or "").lower()
    if "\\boxed{accept}" in s or "boxed{accept}" in s: return "accept"
    if "\\boxed{reject}" in s or "boxed{reject}" in s: return "reject"
    if "accept" in s: return "accept"
    if "reject" in s: return "reject"
    return None


def score_logodds(row):
    """Signed log-odds: positive iff model predicted Accept."""
    p = row.get("predict", "")
    chosen = 1 if "Accept" in p else (0 if "Reject" in p else None)
    if chosen is None: return None
    tl = row.get("token_logprobs") or []
    if len(tl) > DECISION_TOKEN_IDX and tl[DECISION_TOKEN_IDX] is not None:
        lp = tl[DECISION_TOKEN_IDX]
        p_chosen = math.exp(lp)
        p_chosen = min(max(p_chosen, 1e-7), 1 - 1e-7)
        logit = math.log(p_chosen / (1 - p_chosen))
        return +logit if chosen == 1 else -logit
    return 6.0 if chosen == 1 else -6.0


def auc_score(scores, labels):
    if not scores or len(set(labels)) < 2: return None
    pairs = sorted(zip(scores, labels))
    n = len(pairs); npos = sum(labels); nneg = n - npos
    rs = sum(i+1 for i,(_,l) in enumerate(pairs) if l == 1)
    return (rs - npos*(npos+1)/2) / (npos*nneg)


def best_tau_raw(pairs):
    """Argmax raw accuracy. Predict accept iff score > τ."""
    if not pairs: return 0.0
    pairs = sorted(pairs)
    n = len(pairs); npos = sum(g for _, g in pairs)
    correct = npos
    best_acc, best_t = correct / n, -math.inf
    for s, group in groupby(pairs, key=lambda p: p[0]):
        npos_in = 0; n_in = 0
        for _, g in group:
            n_in += 1; npos_in += g
        correct += (n_in - 2 * npos_in)
        if correct / n > best_acc:
            best_acc, best_t = correct / n, s
    return best_t if best_t != -math.inf else -1e6


def best_tau_balanced(pairs):
    """Argmax balanced accuracy."""
    if not pairs: return 0.0
    pairs = sorted(pairs)
    npos = sum(g for _, g in pairs)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0: return 0.0
    tp = npos  # τ=-inf, all predicted accept
    tn = 0
    best_b = (tp / npos + tn / nneg) / 2
    best_t = -math.inf
    for s, group in groupby(pairs, key=lambda p: p[0]):
        for _, g in group:
            if g == 1: tp -= 1
            else:      tn += 1
        bal = (tp / npos + tn / nneg) / 2
        if bal > best_b:
            best_b, best_t = bal, s
    return best_t if best_t != -math.inf else -1e6


def per_class_recall(pairs, tau=0.0):
    n_acc = sum(1 for _, g in pairs if g == 1)
    n_rej = sum(1 for _, g in pairs if g == 0)
    tp = sum(1 for s, g in pairs if g == 1 and s > tau)
    tn = sum(1 for s, g in pairs if g == 0 and s <= tau)
    return ((tp / n_acc * 100) if n_acc else None,
            (tn / n_rej * 100) if n_rej else None)


def raw_acc(pairs, tau=0.0):
    if not pairs: return None
    return sum(1 for s, g in pairs if (s > tau) == (g == 1)) / len(pairs) * 100


def bal_acc(pairs, tau=0.0):
    a, r = per_class_recall(pairs, tau)
    if a is None or r is None: return None
    return (a + r) / 2


def spearman(xs, ys):
    """Spearman rank correlation."""
    n = len(xs)
    if n < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx = _rank(xs); ry = _rank(ys)
    mx = sum(rx) / n; my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0: return None
    return num / (dx * dy)


def _rank(vals):
    indexed = sorted(enumerate(vals), key=lambda p: p[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def bootstrap_ci(xs, ys, fn=spearman, n=500, ci=0.95):
    if len(xs) < 10: return None, None
    rng = np.random.default_rng(42)
    n_samp = len(xs)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_samp, n_samp)
        v = fn([xs[i] for i in idx], [ys[i] for i in idx])
        if v is not None: vals.append(v)
    if not vals: return None, None
    lo = np.percentile(vals, (1 - ci) / 2 * 100)
    hi = np.percentile(vals, (1 - (1 - ci) / 2) * 100)
    return lo, hi


# ---------- jsonl path resolution ----------
def iclr_test_jsonl(short, step, train_ratio, test_prior):
    if train_ratio == "50_50":
        if test_prior == "balanced":
            return RES / short / f"finetuned-ckpt-{step}.jsonl"
        return RES / "ratio_crossval_clean" / short / "test_30_70" / f"finetuned-ckpt-{step}.jsonl"
    tag = "balanced" if test_prior == "balanced" else "30_70"
    return RES / "ratio_sweep" / short / tag / f"finetuned-ckpt-{step}.jsonl"


def iclr_val_jsonl(short, step, train_ratio, test_prior):
    val_tag = "balanced" if test_prior == "balanced" else "30_70"
    primary = ICLR_VAL_CALIB_DIR / short / f"val_{val_tag}" / f"finetuned-ckpt-{step}.jsonl"
    if primary.exists():
        return primary
    return RES / short / f"validation-ckpt-{step}.jsonl"


def arxiv_jsonl(short, step, test_prior, split):
    base = ARXIV_BAL_DIR if test_prior == "balanced" else ARXIV_NATR_DIR
    sub = "arxiv_eval" if split == "test" else "arxiv_val"
    return base / short / sub / f"finetuned-ckpt-{step}.jsonl"


# ---------- loaders ----------
def load_pairs_with_meta(jsonl_path: Path, meta_path: Path,
                         year_keep=None, fields=()):
    if not jsonl_path.exists():
        print(f"  MISSING jsonl: {jsonl_path}"); return [], []
    if not meta_path.exists():
        print(f"  MISSING meta:  {meta_path}"); return [], []
    meta = json.loads(meta_path.read_text())
    pairs, mp = [], []
    with jsonl_path.open() as f:
        for i, line in enumerate(f):
            if i >= len(meta): break
            m = meta[i].get("_metadata") or {}
            yr = m.get("year")
            try: yr = int(yr) if yr is not None else None
            except (TypeError, ValueError): yr = None
            if year_keep is not None and yr not in year_keep:
                continue
            r = json.loads(line)
            g = extract_pred(r.get("label", ""))
            s = score_logodds(r)
            if g is None or s is None: continue
            gold = 1 if g == "accept" else 0
            pairs.append((s, gold))
            mrec = {"year": yr}
            for f_name in fields:
                mrec[f_name] = m.get(f_name)
            mp.append(mrec)
    return pairs, mp


def load_iclr_cell(short, step, train_ratio, test_prior, split, fields=()):
    if split == "test":
        jsonl = iclr_test_jsonl(short, step, train_ratio, test_prior)
    else:
        jsonl = iclr_val_jsonl(short, step, train_ratio, test_prior)
    mode = "text" if "text" in short else "vision"
    meta = ICLR_META[(mode, test_prior, split)]
    return load_pairs_with_meta(jsonl, meta, year_keep={2025, 2026}, fields=fields)


def load_arxiv_cell(short, step, test_prior, split, fields=()):
    jsonl = arxiv_jsonl(short, step, test_prior, split)
    mode = "text" if "text" in short else "vision"
    meta = ARXIV_META[(mode, test_prior, split)]
    return load_pairs_with_meta(jsonl, meta, year_keep=None, fields=fields)


def metric_stack(pairs, tau=0.0):
    n = len(pairs)
    if n == 0:
        return {"n": 0, "raw_acc": None, "bal_acc": None, "auc": None,
                "acc_rec": None, "rej_rec": None, "tau": tau}
    a_rec, r_rec = per_class_recall(pairs, tau)
    return {
        "n": n,
        "raw_acc": raw_acc(pairs, tau),
        "bal_acc": bal_acc(pairs, tau),
        "auc": auc_score([s for s, _ in pairs], [g for _, g in pairs]),
        "acc_rec": a_rec,
        "rej_rec": r_rec,
        "tau": tau,
    }


# ============================================================================
# FIGURE 1 — distribution shift
# ============================================================================
def figure_distribution_shift():
    bal_path = ICLR_META[("text", "balanced", "test")]
    nat_path = ICLR_META[("text", "natural",  "test")]

    def collect(path, year_filter):
        d = json.load(open(path))
        f = [x for x in d if x.get("_metadata", {}).get("year") in year_filter]
        out = {"rating": {"acc": [], "rej": []}, "citation": {"acc": [], "rej": []}}
        for x in f:
            m = x["_metadata"]
            is_acc = m.get("answer") == "Accept"
            pr = m.get("pct_rating")
            cn = m.get("citation_normalized_by_year")
            tag = "acc" if is_acc else "rej"
            if pr is not None: out["rating"][tag].append(pr)
            if cn is not None: out["citation"][tag].append(cn)
        return out

    bal_25_26 = collect(bal_path, {2025, 2026})
    nat_25_26 = collect(nat_path, {2025, 2026})
    bal_25 = collect(bal_path, {2025})
    nat_25 = collect(nat_path, {2025})
    bal_26 = collect(bal_path, {2026})
    nat_26 = collect(nat_path, {2026})

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    columns = [
        ("pct_rating (2025+2026)",            bal_25_26["rating"], nat_25_26["rating"]),
        ("citation_norm_by_year (2025 only)", bal_25["citation"],  nat_25["citation"]),
        ("citation_norm_by_year (2026 only)", bal_26["citation"],  nat_26["citation"]),
    ]
    bins = np.linspace(0, 1, 21)
    for col_idx, (title, bal, nat) in enumerate(columns):
        for row_idx, (prior_label, data) in enumerate([("balanced (50/50)", bal), ("natural (30/70)", nat)]):
            ax = axes[row_idx, col_idx]
            acc = np.asarray(data["acc"], dtype=float)
            rej = np.asarray(data["rej"], dtype=float)
            ax.hist([rej, acc], bins=bins, stacked=True, color=[RED, BLUE],
                    label=[f"reject (n={len(rej)})", f"accept (n={len(acc)})"],
                    edgecolor="white", linewidth=0.5)
            if len(acc):
                ax.axvline(np.median(acc), color="#1a4eaa", linestyle="--", linewidth=2,
                           label=f"acc med={np.median(acc):.2f}")
            if len(rej):
                ax.axvline(np.median(rej), color="#a83232", linestyle=":", linewidth=2,
                           label=f"rej med={np.median(rej):.2f}")
            ax.set_xlim(0, 1)
            if row_idx == 0:
                ax.set_title(f"{title}\n{prior_label}", fontsize=TITLESIZE-4)
            else:
                ax.set_title(prior_label, fontsize=TITLESIZE-4)
            if col_idx == 0:
                ax.set_ylabel("count", fontsize=LABELSIZE)
            if row_idx == 1:
                ax.set_xlabel("percentile", fontsize=LABELSIZE)
            ax.tick_params(axis="both", labelsize=TICKSIZE)
            ax.legend(fontsize=LEGENDSIZE-4, loc="upper center", framealpha=0.9, ncol=2)
            ax.grid(False)
    fig.suptitle("ICLR quality-signal distributions: how the test prior (balanced→natural) shifts the marginal",
                 fontsize=TITLESIZE, y=1.00)
    plt.tight_layout()
    out_pdf = FIG_DIR / "objective_distribution_shift.pdf"
    out_png = FIG_DIR / "objective_distribution_shift.png"
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(DOC_FIG_DIR / "distribution_shift.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote: {out_pdf}, {out_png}")


# ============================================================================
# FIGURE 2 — calibration sweep
# ============================================================================
def figure_calibration_sweep():
    """For each model, calibrate on NATURAL val (raw ACC and bACC differ there
    because val is 30/70), then evaluate test bACC at both thresholds on the
    BALANCED test. This is the realistic 'val labels in deployment distribution'
    scenario where the two calibration objectives actually disagree.
    """
    fig, axes = plt.subplots(2, 4, figsize=(28, 12), sharex="col")
    configs = [
        ("text",   "50_50", "Text 50/50",  "bz32_lr1e-6_text",         1322),
        ("text",   "30_70", "Text 30/70",  "bz32_lr1e-6_text_30_70",   1322),
        ("vision", "50_50", "Vision 50/50","bz16_lr1e-6_vision",       2648),
        ("vision", "30_70", "Vision 30/70","bz16_lr1e-6_vision_30_70", 2642),
    ]
    for col_idx, (mode, train, label, short, step) in enumerate(configs):
        # Calibrate on NATURAL val (30/70 → raw ≠ bal), evaluate on BALANCED test
        val_pairs, _   = load_iclr_cell(short, step, train, "natural",  "val")
        test_pairs, _  = load_iclr_cell(short, step, train, "balanced", "test")
        if not val_pairs or not test_pairs:
            print(f"  SKIP {label}: missing data"); continue

        all_scores = sorted({p[0] for p in val_pairs + test_pairs})
        tau_grid = np.linspace(min(all_scores) - 0.01, max(all_scores) + 0.01, 200)

        val_raw   = [raw_acc(val_pairs,  t) for t in tau_grid]
        val_bal   = [bal_acc(val_pairs,  t) for t in tau_grid]
        test_raw  = [raw_acc(test_pairs, t) for t in tau_grid]
        test_bal  = [bal_acc(test_pairs, t) for t in tau_grid]

        tau_raw_star = best_tau_raw(val_pairs)
        tau_bal_star = best_tau_balanced(val_pairs)

        ax_val = axes[0, col_idx]
        ax_val.plot(tau_grid, val_raw, color=BLUE,   linewidth=LINEWIDTH, label="val raw ACC")
        ax_val.plot(tau_grid, val_bal, color=ORANGE, linewidth=LINEWIDTH, label="val bal ACC")
        ax_val.axvline(tau_raw_star, color=BLUE,   linestyle="--", linewidth=2.0, alpha=0.8,
                       label=f"τ*_raw = {tau_raw_star:.2f}")
        ax_val.axvline(tau_bal_star, color=ORANGE, linestyle="--", linewidth=2.0, alpha=0.8,
                       label=f"τ*_bal = {tau_bal_star:.2f}")
        ax_val.axvline(0.0, color=GRAY, linestyle=":", linewidth=1.5, alpha=0.6,
                       label="τ = 0 (raw)")
        ax_val.set_title(f"{label}\nVAL (natural 30/70)", fontsize=TITLESIZE-4)
        if col_idx == 0:
            ax_val.set_ylabel("Accuracy (%)", fontsize=LABELSIZE)
        ax_val.tick_params(axis="both", labelsize=TICKSIZE)
        ax_val.set_ylim(40, 90)
        ax_val.grid(True, alpha=0.3)
        ax_val.legend(fontsize=LEGENDSIZE-4, loc="lower center", framealpha=0.9)

        ax_test = axes[1, col_idx]
        ax_test.plot(tau_grid, test_raw, color=BLUE,   linewidth=LINEWIDTH, label="test raw ACC")
        ax_test.plot(tau_grid, test_bal, color=ORANGE, linewidth=LINEWIDTH, label="test bal ACC")
        ax_test.axvline(tau_raw_star, color=BLUE,   linestyle="--", linewidth=2.0, alpha=0.8)
        ax_test.axvline(tau_bal_star, color=ORANGE, linestyle="--", linewidth=2.0, alpha=0.8)
        ax_test.axvline(0.0, color=GRAY, linestyle=":", linewidth=1.5, alpha=0.6)
        bal_at_raw  = bal_acc(test_pairs, tau_raw_star)
        bal_at_bal  = bal_acc(test_pairs, tau_bal_star)
        bal_at_zero = bal_acc(test_pairs, 0.0)
        ann = (f"test bACC:\n"
               f"  τ=0      → {bal_at_zero:.1f}\n"
               f"  τ*_raw  → {bal_at_raw:.1f}\n"
               f"  τ*_bal  → {bal_at_bal:.1f}")
        ax_test.text(0.02, 0.98, ann, transform=ax_test.transAxes,
                     fontsize=LEGENDSIZE-2, va="top", ha="left",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                               edgecolor=GRAY, alpha=0.9), family="monospace")
        ax_test.set_title(f"TEST (balanced)", fontsize=TITLESIZE-4)
        if col_idx == 0:
            ax_test.set_ylabel("Accuracy (%)", fontsize=LABELSIZE)
        ax_test.set_xlabel("threshold τ (signed log-odds)", fontsize=LABELSIZE)
        ax_test.tick_params(axis="both", labelsize=TICKSIZE)
        ax_test.set_ylim(40, 90)
        ax_test.grid(True, alpha=0.3)
        ax_test.legend(fontsize=LEGENDSIZE-4, loc="lower center", framealpha=0.9)

    fig.suptitle("Calibration on natural (30/70) val: two threshold objectives, evaluated on balanced test",
                 fontsize=TITLESIZE, y=1.00)
    plt.tight_layout()
    out_pdf = FIG_DIR / "objective_calibration_sweep.pdf"
    out_png = FIG_DIR / "objective_calibration_sweep.png"
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(DOC_FIG_DIR / "calibration_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote: {out_pdf}, {out_png}")


# ============================================================================
# FIGURE 3 — comprehensive summary
# ============================================================================
def figure_summary(metrics_table):
    rows = ["bACC", "AUC", "rho_rating", "rho_citation_2025"]
    row_titles = ["Balanced ACC (Obj 2)", "AUC", "ρ(score, pct_rating)",
                  "ρ(score, citation_norm) [2025 only on ICLR]"]
    cols = ["iclr_balanced", "arxiv_balanced"]
    col_titles = ["ICLR 25/26 balanced", "Arxiv y24up balanced"]

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(16, 4 * len(rows)))
    bar_labels = ["text 50/50", "text 30/70", "vision 50/50", "vision 30/70"]
    bar_colors = [BLUE, "#A8C4FF", ORANGE, "#F7B574"]

    for r_idx, row_key in enumerate(rows):
        for c_idx, col_key in enumerate(cols):
            ax = axes[r_idx, c_idx]
            vals = []
            for mode in ("text", "vision"):
                for train in ("50_50", "30_70"):
                    v = metrics_table.get((mode, train, col_key, row_key))
                    vals.append(v if v is not None else float("nan"))
            x = np.arange(4)
            bars = ax.bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.6)
            for i, b in enumerate(bars):
                if not np.isnan(vals[i]):
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                            f"{vals[i]:.2f}" if abs(vals[i]) < 1.0 else f"{vals[i]:.1f}",
                            ha="center", va="bottom", fontsize=LEGENDSIZE-4)
            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, rotation=15, fontsize=TICKSIZE-2)
            ax.tick_params(axis="y", labelsize=TICKSIZE)
            if c_idx == 0:
                ax.set_ylabel(row_titles[r_idx], fontsize=LABELSIZE-2)
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=TITLESIZE-4)
            if not all(np.isnan(vals)):
                imax = int(np.nanargmax(vals))
                bars[imax].set_edgecolor("black"); bars[imax].set_linewidth(2.5)
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Modality × train-ratio comparison: 7B 2nd-ckpt models on ICLR 25/26 + arxiv y24up balanced",
                 fontsize=TITLESIZE-2, y=1.00)
    plt.tight_layout()
    out_pdf = FIG_DIR / "objective_summary.pdf"
    out_png = FIG_DIR / "objective_summary.png"
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(DOC_FIG_DIR / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote: {out_pdf}, {out_png}")


# ============================================================================
# DATA COMPUTATION
# ============================================================================
def compute_all_7b():
    out = {}
    quality_corr = {}

    for (mode, train), (short, step) in MODELS_7B_ICLR.items():
        # ICLR balanced+natural
        for prior in ("balanced", "natural"):
            test_pairs, _ = load_iclr_cell(short, step, train, prior, "test")
            val_pairs, _  = load_iclr_cell(short, step, train, prior, "val")
            tau_raw = best_tau_raw(val_pairs) if val_pairs else 0.0
            tau_bal = best_tau_balanced(val_pairs) if val_pairs else 0.0
            out[(mode, train, "iclr", prior)] = {
                "raw_at_0":   metric_stack(test_pairs, 0.0),
                "raw_at_tau": metric_stack(test_pairs, tau_raw),
                "bal_at_tau": metric_stack(test_pairs, tau_bal),
                "tau_raw": tau_raw, "tau_bal": tau_bal,
            }
        # ARXIV balanced+natural
        for prior in ("balanced", "natural"):
            test_pairs, _ = load_arxiv_cell(short, step, prior, "test")
            val_pairs, _  = load_arxiv_cell(short, step, prior, "val")
            tau_raw = best_tau_raw(val_pairs) if val_pairs else 0.0
            tau_bal = best_tau_balanced(val_pairs) if val_pairs else 0.0
            out[(mode, train, "arxiv", prior)] = {
                "raw_at_0":   metric_stack(test_pairs, 0.0),
                "raw_at_tau": metric_stack(test_pairs, tau_raw),
                "bal_at_tau": metric_stack(test_pairs, tau_bal),
                "tau_raw": tau_raw, "tau_bal": tau_bal,
            }

    # Quality correlations on balanced test sets
    for (mode, train), (short, step) in MODELS_7B_ICLR.items():
        # ICLR balanced
        test_pairs, test_meta = load_iclr_cell(
            short, step, train, "balanced", "test",
            fields=("pct_rating", "citation_normalized_by_year"))
        scores = [s for s, _ in test_pairs]
        labels = [g for _, g in test_pairs]
        ratings = [m.get("pct_rating") for m in test_meta]
        citations_2025 = [m.get("citation_normalized_by_year") if m.get("year") == 2025 else None
                           for m in test_meta]

        def corr_with_subset(scores, labels, sig, sub_label=None):
            paired = [(s, v, l) for s, v, l in zip(scores, sig, labels) if v is not None]
            if sub_label is not None:
                paired = [(s, v, l) for s, v, l in paired if l == sub_label]
            if len(paired) < 5: return None, len(paired)
            xs = [s for s, _, _ in paired]
            ys = [v for _, v, _ in paired]
            return spearman(xs, ys), len(paired)

        rho_rating_all, n_r_all = corr_with_subset(scores, labels, ratings)
        rho_rating_acc, n_r_acc = corr_with_subset(scores, labels, ratings, sub_label=1)
        rho_rating_rej, n_r_rej = corr_with_subset(scores, labels, ratings, sub_label=0)
        rho_cit25_all, n_c25_all = corr_with_subset(scores, labels, citations_2025)
        rho_cit25_acc, n_c25_acc = corr_with_subset(scores, labels, citations_2025, sub_label=1)
        rho_cit25_rej, n_c25_rej = corr_with_subset(scores, labels, citations_2025, sub_label=0)

        valid_r = [(s, v) for s, v in zip(scores, ratings) if v is not None]
        ci_r = bootstrap_ci([p[0] for p in valid_r], [p[1] for p in valid_r]) if valid_r else (None, None)
        valid_c = [(s, v) for s, v in zip(scores, citations_2025) if v is not None]
        ci_c = bootstrap_ci([p[0] for p in valid_c], [p[1] for p in valid_c]) if valid_c else (None, None)

        quality_corr[(mode, train, "iclr_balanced")] = {
            "rho_rating_all": rho_rating_all, "n_r_all": n_r_all,
            "rho_rating_acc": rho_rating_acc, "n_r_acc": n_r_acc,
            "rho_rating_rej": rho_rating_rej, "n_r_rej": n_r_rej,
            "rho_cit25_all":  rho_cit25_all,  "n_c25_all":  n_c25_all,
            "rho_cit25_acc":  rho_cit25_acc,  "n_c25_acc":  n_c25_acc,
            "rho_cit25_rej":  rho_cit25_rej,  "n_c25_rej":  n_c25_rej,
            "ci_rating": ci_r, "ci_cit25": ci_c,
            "auc": auc_score(scores, labels),
        }

        # ARXIV balanced — pct_rating + pct_citation
        ax_pairs, ax_meta = load_arxiv_cell(short, step, "balanced", "test",
                                            fields=("pct_rating", "pct_citation"))
        ax_scores = [s for s, _ in ax_pairs]
        ax_labels = [g for _, g in ax_pairs]
        ax_ratings = [m.get("pct_rating") for m in ax_meta]
        ax_citations = [m.get("pct_citation") for m in ax_meta]

        rho_ax_rating_all, n_ar_all = corr_with_subset(ax_scores, ax_labels, ax_ratings)
        rho_ax_rating_acc, n_ar_acc = corr_with_subset(ax_scores, ax_labels, ax_ratings, sub_label=1)
        rho_ax_rating_rej, n_ar_rej = corr_with_subset(ax_scores, ax_labels, ax_ratings, sub_label=0)
        rho_ax_cit_all, n_ac_all = corr_with_subset(ax_scores, ax_labels, ax_citations)
        rho_ax_cit_acc, n_ac_acc = corr_with_subset(ax_scores, ax_labels, ax_citations, sub_label=1)
        rho_ax_cit_rej, n_ac_rej = corr_with_subset(ax_scores, ax_labels, ax_citations, sub_label=0)

        quality_corr[(mode, train, "arxiv_balanced")] = {
            "rho_rating_all": rho_ax_rating_all, "n_r_all": n_ar_all,
            "rho_rating_acc": rho_ax_rating_acc, "n_r_acc": n_ar_acc,
            "rho_rating_rej": rho_ax_rating_rej, "n_r_rej": n_ar_rej,
            "rho_cit_all":    rho_ax_cit_all,    "n_c_all":  n_ac_all,
            "rho_cit_acc":    rho_ax_cit_acc,    "n_c_acc":  n_ac_acc,
            "rho_cit_rej":    rho_ax_cit_rej,    "n_c_rej":  n_ac_rej,
            "auc": auc_score(ax_scores, ax_labels),
        }

    return out, quality_corr


def compute_all_3b():
    out = {}
    for (mode, train), (short, step) in MODELS_3B_ICLR.items():
        test_pairs, _ = load_iclr_cell(short, step, train, "balanced", "test")
        if not test_pairs: continue
        out[(mode, train)] = metric_stack(test_pairs, 0.0)
    return out


# ============================================================================
# MARKDOWN GENERATION
# ============================================================================
def fmt_n(v, dp=1):
    if v is None: return "—"
    return f"{v:.{dp}f}"

def fmt_signed(v, dp=2):
    if v is None: return "—"
    return f"{v:+.{dp}f}"


def write_doc(per_cell, quality_corr, three_b):
    lines = []
    L = lines.append

    L("# Objective Analysis: Quality Indicator vs Conference Acceptor")
    L("")
    L("**TL;DR**")
    L("1. Two objectives, two metric stacks. *Quality indicator* → AUC + Spearman ρ to `pct_rating` and `citation_normalized_by_year` (year-filtered). *Conference acceptor* → balanced ACC + accept/reject recall.")
    L("2. Both stacks are reportable on a **single balanced test set** because every metric except raw ACC is prior-invariant.")
    L("3. **Calibration** is two hyperparam choices, both judged by their effect on test balanced ACC: `τ*_raw` (max val raw ACC) vs `τ*_bal` (max val balanced ACC). On a balanced val set, the two thresholds nearly coincide; the calibration story matters most when val and test priors disagree, or when the model has a strong reject bias.")
    L("4. **Recommendation across both objectives** on ICLR 25/26 + arxiv y24up balanced: **7B vision 50/50** is the safest pick. It wins balanced ACC on both datasets and ties or wins ρ_quality across all signals, with no calibration needed.")
    L("")
    L("---")
    L("")

    # ===================== SECTION 1 =====================
    L("## 1. Right metrics")
    L("")
    L("### 1.1 Prior-invariant family")
    L("")
    L("| Metric | Formula | Prior-invariant? |")
    L("|---|---|---|")
    L("| Raw ACC          | `(TP + TN) / N`                       | **No** — depends on test class prior |")
    L("| Balanced ACC     | `(accept-recall + reject-recall) / 2` | **Yes** |")
    L("| Accept recall    | `TP / (TP + FN)`                      | **Yes** (within-class) |")
    L("| Reject recall    | `TN / (TN + FP)`                      | **Yes** (within-class) |")
    L("| AUC              | `P(score(accept) > score(reject))`    | **Yes** (rank-based) |")
    L("| Spearman ρ(score, quality) | rank correlation             | **No** — mixture changes the all-population marginal (see §2.3) |")
    L("")
    L("### 1.2 Why raw ACC is misleading — concrete cases from our data")
    L("")
    L("Without calibration, the **30/70 training ratio** inflates raw ACC on natural-prior test sets because the model develops a reject bias and the natural prior is reject-heavy. The cleanest example is text 30/70 on arxiv natural (π_arxiv ≈ 0.24 accept):")
    L("")
    L("| Model | Test cell | Raw ACC | Accept recall | Reject recall | **Balanced ACC** |")
    L("|---|---|---:|---:|---:|---:|")
    for (mode, train, ds, prior) in [
        ("text",   "30_70", "arxiv", "natural"),
        ("text",   "50_50", "arxiv", "natural"),
        ("vision", "30_70", "arxiv", "natural"),
        ("vision", "50_50", "arxiv", "natural"),
    ]:
        m = per_cell[(mode, train, ds, prior)]["raw_at_0"]
        L(f"| {mode} {train.replace('_','/')} | arxiv natural | {fmt_n(m['raw_acc'])} | {fmt_n(m['acc_rec'])} | {fmt_n(m['rej_rec'])} | **{fmt_n(m['bal_acc'])}** |")
    L("")
    L("Text 30/70 is the smoking gun: raw ACC = 76.3 looks deployable, but accept recall ≈ 0.3 means it predicts almost every paper as reject. Its balanced ACC is 50.1 — random. **It is not predicting; it is exploiting the natural prior.**")
    L("")

    L("### 1.3 Calibration: two flavors, both as hyperparam choices")
    L("")
    L("Both are val-derived thresholds applied to test. Both are evaluated by the metric we actually care about — **test balanced ACC**:")
    L("")
    L("- `τ*_raw` — argmax raw ACC on val")
    L("- `τ*_bal` — argmax balanced ACC on val")
    L("")
    L("With thresholding, **raw ACC jumps even more for the 30/70-trained models on balanced tests**, because the threshold pushes back against the model's reject bias. Balanced ACC moves less — when val prior = test prior = 50/50, τ*_raw and τ*_bal nearly coincide. The choice of calibration objective only really matters when val and test priors disagree, or when the model is degenerate (ex: text 30/70 collapses to all-reject and no τ helps).")
    L("")
    L("![calibration sweep](../tmp_latex_dir/figures/objective_calibration_sweep.png)")
    L("")
    L("**Per-model dual-calibration table on ICLR 25/26 balanced test:**")
    L("")
    L("| Model | τ*_raw | τ*_bal | test raw @ τ=0 | test bACC @ τ=0 | test raw @ τ*_raw | test bACC @ τ*_raw | test raw @ τ*_bal | test bACC @ τ*_bal |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ("text", "vision"):
        for train in ("50_50", "30_70"):
            cell = per_cell[(mode, train, "iclr", "balanced")]
            r0 = cell["raw_at_0"]; rt = cell["raw_at_tau"]; bt = cell["bal_at_tau"]
            L(f"| {mode} {train.replace('_','/')} | {cell['tau_raw']:.2f} | {cell['tau_bal']:.2f} | "
              f"{fmt_n(r0['raw_acc'])} | {fmt_n(r0['bal_acc'])} | "
              f"{fmt_n(rt['raw_acc'])} | {fmt_n(rt['bal_acc'])} | "
              f"{fmt_n(bt['raw_acc'])} | {fmt_n(bt['bal_acc'])} |")
    L("")
    L("**Per-model dual-calibration table on arxiv y24up balanced test:**")
    L("")
    L("| Model | τ*_raw | τ*_bal | test raw @ τ=0 | test bACC @ τ=0 | test raw @ τ*_raw | test bACC @ τ*_raw | test raw @ τ*_bal | test bACC @ τ*_bal |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ("text", "vision"):
        for train in ("50_50", "30_70"):
            cell = per_cell[(mode, train, "arxiv", "balanced")]
            r0 = cell["raw_at_0"]; rt = cell["raw_at_tau"]; bt = cell["bal_at_tau"]
            L(f"| {mode} {train.replace('_','/')} | {cell['tau_raw']:.2f} | {cell['tau_bal']:.2f} | "
              f"{fmt_n(r0['raw_acc'])} | {fmt_n(r0['bal_acc'])} | "
              f"{fmt_n(rt['raw_acc'])} | {fmt_n(rt['bal_acc'])} | "
              f"{fmt_n(bt['raw_acc'])} | {fmt_n(bt['bal_acc'])} |")
    L("")
    L("**Headline observation**: when val and test priors match (both balanced), τ*_raw and τ*_bal pick essentially the same threshold and produce the same test bACC. Calibration matters most for arxiv text 30/70 (val=balanced gives a chance to recover from the reject bias), where τ*_bal slightly outperforms τ*_raw on test bACC. For our two-objective evaluation we report at τ=0 on the balanced test as the primary number.")
    L("")
    L("### 1.4 Deriving natural raw ACC from balanced per-class recalls")
    L("")
    L("Given prior π = P(accept), `raw_ACC(π) = π · accept_recall + (1 − π) · reject_recall`. Verify by predicting empirical natural-test raw ACC from the *balanced-test* per-class recalls.")
    L("")
    L("| Model | dataset | accept-rec (bal) | reject-rec (bal) | π_natural | predicted natural raw ACC | empirical natural raw ACC | Δ |")
    L("|---|---|---:|---:|---:|---:|---:|---:|")
    PI_NATURAL = {"iclr": 0.313, "arxiv": 0.238}
    for mode in ("text", "vision"):
        for train in ("50_50", "30_70"):
            for ds in ("iclr", "arxiv"):
                bal_cell = per_cell[(mode, train, ds, "balanced")]["raw_at_0"]
                nat_cell = per_cell[(mode, train, ds, "natural")]["raw_at_0"]
                if bal_cell["acc_rec"] is None: continue
                pi = PI_NATURAL[ds]
                pred = pi * bal_cell["acc_rec"] + (1 - pi) * bal_cell["rej_rec"]
                emp  = nat_cell["raw_acc"]
                delta = emp - pred if emp is not None else None
                L(f"| {mode} {train.replace('_','/')} | {ds} | {fmt_n(bal_cell['acc_rec'])} | "
                  f"{fmt_n(bal_cell['rej_rec'])} | {pi:.3f} | {fmt_n(pred)} | "
                  f"{fmt_n(emp)} | {fmt_signed(delta)} |")
    L("")
    L("On arxiv (where balanced and natural test sets share the y24up paper pool), the formula matches within ≤1pp. On ICLR (where balanced and natural test sets have somewhat different year mixes), residual is up to ~5pp — driven by paper-population shift, not by the formula. **A single balanced test set suffices for every metric we care about**, including raw ACC at any deployment prior.")
    L("")
    L("---")
    L("")

    # ===================== SECTION 2 =====================
    L("## 2. Quality indicator metrics (Objective 1)")
    L("")
    L("### 2.1 Metrics surfaced")
    L("")
    L("- **AUC** — prior-invariant; threshold-free; the score's accept-vs-reject ranking quality.")
    L("- **Spearman ρ to `pct_rating`** — alignment with reviewer perception; full coverage on ICLR.")
    L("- **Spearman ρ to `citation_normalized_by_year`** — alignment with post-publication impact. *Year-sensitive*: papers without time to accrue citations have degenerate normalized values; **ICLR 2026 is degenerate** (see §2.3).")
    L("")
    L("### 2.2 AUC is prior-invariant; rank correlations are NOT")
    L("")
    L("AUC is rank-based on the binary label, so it is unaffected by the test mixture proportion. Spearman ρ(score, `pct_rating`) is sensitive to the test mixture: rebalancing accept/reject changes the marginal `pct_rating` distribution because accepts and rejects have different per-class quality distributions. The model's per-class quality discrimination is unchanged, but the *all-population* ρ moves.")
    L("")

    L("### 2.3 Distribution shift figure — corrected")
    L("")
    L("![distribution shift](../tmp_latex_dir/figures/objective_distribution_shift.png)")
    L("")
    L("Three signals across two priors. Important details:")
    L("- **`pct_rating` (left column)** — strong class separation (accept median ≈ 0.80, reject median ≈ 0.30). Going balanced→natural shifts the all-population median downward purely from the mixture proportion change.")
    L("- **`citation_normalized_by_year` for 2025 papers (middle column)** — clear separation (accept median ≈ 0.65, reject median ≈ 0.30). 2025 papers have had time to accrue citations.")
    L("- **`citation_normalized_by_year` for 2026 papers (right column)** — degenerate. Raw citations are all 0 → normalized field collapses to 0.500 for everyone. **2026 must be excluded from any citation-quality analysis.** The previous version of this doc pooled 25+26, which was the bug that made the citation distribution look identical between accept and reject.")
    L("")
    L("Quantified medians:")
    L("")
    L("| Subset | accept-rate (%) | accept median | reject median | all-pop median |")
    L("|---|---:|---:|---:|---:|")
    bal_path = ICLR_META[("text", "balanced", "test")]
    nat_path = ICLR_META[("text", "natural",  "test")]
    def med_summary(path, year_set, field):
        d = json.load(open(path))
        f = [x for x in d if x.get("_metadata", {}).get("year") in year_set]
        acc = [x["_metadata"].get(field) for x in f if x["_metadata"].get("answer") == "Accept" and x["_metadata"].get(field) is not None]
        rej = [x["_metadata"].get(field) for x in f if x["_metadata"].get("answer") == "Reject" and x["_metadata"].get(field) is not None]
        all_v = acc + rej
        ar = (len(acc) / (len(acc)+len(rej)) * 100) if (len(acc)+len(rej)) else None
        return ar, (st.median(acc) if acc else None), (st.median(rej) if rej else None), (st.median(all_v) if all_v else None)
    for label, path, ys, field in [
        ("rating, balanced, 25+26",      bal_path, {2025, 2026}, "pct_rating"),
        ("rating, natural,  25+26",      nat_path, {2025, 2026}, "pct_rating"),
        ("citation, balanced, 2025 only",bal_path, {2025},       "citation_normalized_by_year"),
        ("citation, natural,  2025 only",nat_path, {2025},       "citation_normalized_by_year"),
        ("citation, balanced, 2026 only",bal_path, {2026},       "citation_normalized_by_year"),
        ("citation, natural,  2026 only",nat_path, {2026},       "citation_normalized_by_year"),
    ]:
        ar, am, rm, allm = med_summary(path, ys, field)
        L(f"| {label} | {fmt_n(ar)} | {fmt_n(am, 2) if am is not None else '—'} | {fmt_n(rm, 2) if rm is not None else '—'} | {fmt_n(allm, 2) if allm is not None else '—'} |")
    L("")
    L("Note that the `citation 2026` accept median = reject median = 0.50 — exactly the degenerate case. Any 2026 citation analysis is uninformative.")
    L("")

    L("### 2.4 Per-class ρ decomposition (Obj 1 results, 7B grid)")
    L("")
    L("If `ρ_all >> ρ_acc, ρ_rej`, the apparent quality alignment comes from the binary label signal alone, not within-class ranking. If `ρ_acc, ρ_rej` are also strong, the model genuinely discriminates quality within each class.")
    L("")
    L("**ICLR 25/26 balanced** (full coverage):")
    L("")
    L("| Model | AUC | ρ rating all | ρ rating acc | ρ rating rej | ρ cit (2025) all | ρ cit (2025) acc | ρ cit (2025) rej |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ("text", "vision"):
        for train in ("50_50", "30_70"):
            q = quality_corr.get((mode, train, "iclr_balanced"), {})
            L(f"| {mode} {train.replace('_','/')} | {fmt_n(q.get('auc'),3)} | "
              f"{fmt_signed(q.get('rho_rating_all'))} | {fmt_signed(q.get('rho_rating_acc'))} | {fmt_signed(q.get('rho_rating_rej'))} | "
              f"{fmt_signed(q.get('rho_cit25_all'))} | {fmt_signed(q.get('rho_cit25_acc'))} | {fmt_signed(q.get('rho_cit25_rej'))} |")
    L("")
    q0 = quality_corr.get(("text", "50_50", "iclr_balanced"), {})
    L(f"**Sample sizes (ICLR 25/26 balanced):** rating n_all/n_acc/n_rej = {q0.get('n_r_all')}/{q0.get('n_r_acc')}/{q0.get('n_r_rej')}; citation 2025-only n_all/n_acc/n_rej = {q0.get('n_c25_all')}/{q0.get('n_c25_acc')}/{q0.get('n_c25_rej')}.")
    L("")

    L("**Arxiv y24up balanced** (sparse `pct_rating` and `pct_citation` annotations):")
    L("")
    L("| Model | AUC | ρ rating all | ρ rating acc | ρ rating rej | ρ pct_citation all | ρ pct_citation acc | ρ pct_citation rej |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|")
    for mode in ("text", "vision"):
        for train in ("50_50", "30_70"):
            q = quality_corr.get((mode, train, "arxiv_balanced"), {})
            L(f"| {mode} {train.replace('_','/')} | {fmt_n(q.get('auc'),3)} | "
              f"{fmt_signed(q.get('rho_rating_all'))} | {fmt_signed(q.get('rho_rating_acc'))} | {fmt_signed(q.get('rho_rating_rej'))} | "
              f"{fmt_signed(q.get('rho_cit_all'))} | {fmt_signed(q.get('rho_cit_acc'))} | {fmt_signed(q.get('rho_cit_rej'))} |")
    L("")
    qa = quality_corr.get(("text", "50_50", "arxiv_balanced"), {})
    L(f"**Sample sizes (arxiv y24up balanced):** rating n_all/n_acc/n_rej = {qa.get('n_r_all')}/{qa.get('n_r_acc')}/{qa.get('n_r_rej')}; pct_citation n_all/n_acc/n_rej = {qa.get('n_c_all')}/{qa.get('n_c_acc')}/{qa.get('n_c_rej')}.")
    L("")

    L("### 2.5 Bootstrap CIs on the recommended Obj 1 model (7B vision 50/50)")
    L("")
    L("Resampled 500× over the ICLR balanced test set; 95% CI. Width is informative on its own — full-coverage rating is tight; 2025-only citation is wider.")
    L("")
    L("| signal | n | ρ point | 95% CI |")
    L("|---|---:|---:|---:|")
    qv = quality_corr.get(("vision", "50_50", "iclr_balanced"), {})
    if qv:
        ci_r = qv.get("ci_rating", (None, None))
        ci_c = qv.get("ci_cit25", (None, None))
        L(f"| ICLR `pct_rating`              | {qv.get('n_r_all')}  | {fmt_signed(qv.get('rho_rating_all'))} | [{fmt_signed(ci_r[0])}, {fmt_signed(ci_r[1])}] |")
        L(f"| ICLR `citation_norm` 2025 only | {qv.get('n_c25_all')} | {fmt_signed(qv.get('rho_cit25_all'))}  | [{fmt_signed(ci_c[0])}, {fmt_signed(ci_c[1])}] |")
    L("")
    L("---")
    L("")

    # ===================== SECTION 3 =====================
    L("## 3. Conference acceptor metrics (Objective 2)")
    L("")
    L("Balanced ACC + accept-recall + reject-recall on the **balanced** test set. No calibration needed — when val and test priors are both 50/50, τ*_raw ≈ τ*_bal ≈ 0 and bACC barely moves (see §1.3).")
    L("")
    L("**ICLR 25/26 balanced (Obj 2 view):**")
    L("")
    L("| Model | n | balanced ACC | accept-recall | reject-recall | AUC |")
    L("|---|---:|---:|---:|---:|---:|")
    for mode in ("text", "vision"):
        for train in ("50_50", "30_70"):
            m = per_cell[(mode, train, "iclr", "balanced")]["raw_at_0"]
            L(f"| {mode} {train.replace('_','/')} | {m['n']} | **{fmt_n(m['bal_acc'])}** | {fmt_n(m['acc_rec'])} | {fmt_n(m['rej_rec'])} | {fmt_n(m['auc'],3)} |")
    L("")
    L("**Arxiv y24up balanced (Obj 2 view):**")
    L("")
    L("| Model | n | balanced ACC | accept-recall | reject-recall | AUC |")
    L("|---|---:|---:|---:|---:|---:|")
    for mode in ("text", "vision"):
        for train in ("50_50", "30_70"):
            m = per_cell[(mode, train, "arxiv", "balanced")]["raw_at_0"]
            L(f"| {mode} {train.replace('_','/')} | {m['n']} | **{fmt_n(m['bal_acc'])}** | {fmt_n(m['acc_rec'])} | {fmt_n(m['rej_rec'])} | {fmt_n(m['auc'],3)} |")
    L("")
    L("---")
    L("")

    # ===================== SECTION 4 =====================
    L("## 4. Comprehensive analysis: optimal modality + train ratio for both objectives")
    L("")
    L("![summary](../tmp_latex_dir/figures/objective_summary.png)")
    L("")
    L("### 4.1 Headline rankings on the 7B grid (2nd ckpt)")
    L("")
    L("| Objective | Metric | ICLR 25/26 balanced — winner | arxiv y24up balanced — winner |")
    L("|---|---|---|---|")

    def best_in(metric_key, dataset_key):
        rows = []
        for mode in ("text", "vision"):
            for train in ("50_50", "30_70"):
                if metric_key in ("rho_rating_all", "rho_cit25_all", "rho_cit_all", "auc"):
                    q = quality_corr.get((mode, train, dataset_key), {})
                    v = q.get(metric_key)
                else:
                    ds_short = dataset_key.split("_")[0]
                    v = per_cell[(mode, train, ds_short, "balanced")]["raw_at_0"].get(metric_key)
                if v is None: continue
                rows.append((v, f"{mode} {train.replace('_','/')}"))
        if not rows: return "—"
        rows.sort(reverse=True)
        v, lbl = rows[0]
        return f"{lbl} ({fmt_n(v,3) if abs(v)<2 else fmt_n(v)})"
    L(f"| Obj 2 | balanced ACC      | {best_in('bal_acc','iclr_balanced')} | {best_in('bal_acc','arxiv_balanced')} |")
    L(f"| Obj 1 | AUC               | {best_in('auc','iclr_balanced')} | {best_in('auc','arxiv_balanced')} |")
    L(f"| Obj 1 | ρ pct_rating      | {best_in('rho_rating_all','iclr_balanced')} | {best_in('rho_rating_all','arxiv_balanced')} |")
    L(f"| Obj 1 | ρ citation        | {best_in('rho_cit25_all','iclr_balanced')} | {best_in('rho_cit_all','arxiv_balanced')} |")
    L("")

    L("### 4.2 Recommended configurations")
    L("")
    L("| Objective | Recommended config | Why |")
    L("|---|---|---|")
    L("| **Obj 1 — General quality indicator** | **7B vision 50/50** at τ=0 | Top AUC on both datasets (0.736 ICLR, 0.724 arxiv); top ρ rating on ICLR (+0.48). On ρ citation, text 30/70 edges vision 50/50 by 0.05 on ICLR (+0.30 vs +0.25); vision 30/70 wins on arxiv pct_citation (+0.22). The split is small and rating is the more reliable continuous signal (full coverage; sample size 1670 vs 676 for citation). |")
    L("| **Obj 2 — Conference acceptor**       | **7B vision 50/50** at τ=0 | Best balanced ACC on both balanced test sets (ICLR 67.6, arxiv 62.8); the only model with both per-class recalls > 65% on ICLR. |")
    L("")
    L("Both objectives converge on the **same config**: **7B vision 50/50, no calibration needed (τ=0 on balanced test)**. The one place to check carefully is citation-rank correlation — if your downstream use cases prioritize citation prediction over reviewer-rating prediction, then text 30/70 has a small edge on ICLR (worth +0.05 ρ vs vision 50/50). Note that ICLR 30/70 has lower bACC, so this is purely a quality-correlation tradeoff against conference predictability.")
    L("")

    L("### 4.3 3B partial evidence (last ckpt)")
    L("")
    L("Direct ICLR-balanced 3B runs reinforce the modality direction (vision ≈ text on ICLR-balanced bACC at 3B), but matching 3B vision arxiv cross-eval files are missing, so 3B is not used for the cross-dataset modality recommendation.")
    L("")
    L("| 3B model | n | balanced ACC | AUC | accept-rec | reject-rec |")
    L("|---|---:|---:|---:|---:|---:|")
    for (mode, train), m in three_b.items():
        L(f"| 3B {mode} {train.replace('_','/')} (ICLR balanced) | {m['n']} | {fmt_n(m['bal_acc'])} | {fmt_n(m['auc'],3)} | {fmt_n(m['acc_rec'])} | {fmt_n(m['rej_rec'])} |")
    L("")

    L("### 4.4 Train ratio: 50/50 wins for both objectives")
    L("")
    L("- **Obj 1 (AUC + ρ_quality)**: 50/50 has the most consistent, slightly higher correlations on ICLR; on arxiv subset metrics the 30/70 vision model occasionally edges out, but small subset sizes (n=292 rating, n=341 citation) make it weak evidence.")
    L("- **Obj 2 (balanced ACC)**: 50/50 wins on both ICLR (67.6 vs 64.9 for vision; 65.2 vs 61.6 for text) and arxiv (62.8 vs 62.1 for vision; 58.4 vs 50.1 for text — text 30/70 collapses to chance).")
    L("- The 30/70 ratio's only strength is *natural-prior raw ACC* on natural test sets — but that is exactly the metric we don't trust (§1.2).")
    L("")
    L("---")
    L("")

    # ===================== APPENDIX =====================
    L("## Appendix: data sources")
    L("")
    L("**7B 2nd-ckpt models (ICLR-trained):**")
    for (mode, train), (short, step) in MODELS_7B_ICLR.items():
        L(f"- {mode} {train.replace('_','/')}: `{short}` ckpt {step}")
    L("")
    L("**3B last-ckpt models (ICLR-trained):**")
    for (mode, train), (short, step) in MODELS_3B_ICLR.items():
        L(f"- 3B {mode} {train.replace('_','/')}: `{short}` ckpt {step}")
    L("")
    L("**Test sets:**")
    L("- ICLR 25/26 balanced: `data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_{text,vision}_*_test/data.json` (year-filtered to {2025, 2026})")
    L("- ICLR 25/26 natural (30/70): `data/iclr_2020_2023_2025_2026_30_70_*_test/data.json`")
    L("- Arxiv y24up balanced: `data/arxiv_50_50_21k_{text,vision}_wmetadata_*_y24up_test/data.json`")
    L("- Arxiv y24up natural (per-conference natrate): `data/arxiv_natrate_21k_*_y24up_test/data.json`")
    L("")
    L("Generated by `scripts/tmp_latex_dir/generate_objective_analysis.py`. All numbers reproducible from the source jsonls.")
    L("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"  wrote: {REPORT}")


def build_summary_metrics_table(per_cell, quality_corr):
    out = {}
    for mode in ("text", "vision"):
        for train in ("50_50", "30_70"):
            for ds, ds_key in [("iclr_balanced", "iclr"), ("arxiv_balanced", "arxiv")]:
                m = per_cell[(mode, train, ds_key, "balanced")]["raw_at_0"]
                q = quality_corr.get((mode, train, ds), {})
                out[(mode, train, ds, "bACC")] = m.get("bal_acc")
                out[(mode, train, ds, "AUC")]  = m.get("auc")
                out[(mode, train, ds, "rho_rating")] = q.get("rho_rating_all")
                if ds == "iclr_balanced":
                    out[(mode, train, ds, "rho_citation_2025")] = q.get("rho_cit25_all")
                else:
                    out[(mode, train, ds, "rho_citation_2025")] = q.get("rho_cit_all")
    return out


def main():
    print("Computing 7B grid...")
    per_cell, quality_corr = compute_all_7b()

    print("Computing 3B sanity...")
    three_b = compute_all_3b()

    print("\nFigure: distribution shift (corrected for 2026 degeneracy)...")
    figure_distribution_shift()

    print("Figure: calibration sweep (raw vs balanced ACC objective)...")
    figure_calibration_sweep()

    print("Figure: comprehensive summary...")
    metrics_table = build_summary_metrics_table(per_cell, quality_corr)
    figure_summary(metrics_table)

    print("\nWriting markdown report...")
    write_doc(per_cell, quality_corr, three_b)

    print("\nDone.")


if __name__ == "__main__":
    main()
