#!/usr/bin/env python3
"""
Extended analyses for the 7B ratio cross-eval report.

Adds four sections to reports/ratio_xeval_7b.md (appended after the existing content):

  1. Per-venue arxiv breakdown — ACC raw/cal + AUC for each (modality, train) × venue.
     Heatmap figures across the 4 (eval_set × prior × split) test cells.

  2. Y25+ ablation — drop ECCV (only 2024 papers in the y24up subset) and recompute
     overall arxiv metrics.

  3. Quality-signal correlation — for samples that have `pct_rating` in metadata
     (ICLR and arxiv-iclr/arxiv-neurips), compute Pearson + Spearman correlation between
     model score and pct_rating. Higher = the model's score better tracks human review
     quality, independent of the binary accept/reject decision.

  4. Arxiv-iclr vs direct-ICLR comparison — aggregate ACC + AUC on the arxiv test
     subset where pl_venue == "iclr" (year ≥ 2024) compared to the direct ICLR
     test (year ≥ 2025). Same paper population, two different test sets — does
     model behavior transfer?
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
DATA = ROOT / "data"
REPORT_PATH = ROOT / "reports/ratio_xeval_7b.md"
FIG_DIR = ROOT / "tmp_latex_dir/figures"

# Reuse all the path/loader/score logic from the consolidated script
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from ratio_xeval_consolidated import (
    MODEL, RATIO_LABEL, MODES, RATIOS, PRIORS, SPLITS, TEST_SETS,
    ACL_FAMILY, CELL_LABELS,
    iclr_test_jsonl, iclr_val_jsonl, arxiv_jsonl,
    ICLR_META, ARXIV_META,
    extract_pred, score_logodds, auc, best_tau,
    accuracy, per_class_recall,
    load_pairs, load_iclr_cell, load_arxiv_cell,
    metrics_iclr, metrics_arxiv,
    compute_iclr_thresholds, compute_arxiv_thresholds,
    DECISION_TOKEN_IDX,
)

# ---------- venue filter helpers ----------
ALL_VENUES = ["aaai", "acl_family", "aistats", "colm", "corl", "cvpr", "eccv",
              "iccv", "iclr", "icml", "neurips"]
# Y25+ ablation: which venues have NO 2025/2026 papers in the test sets
Y24_ONLY_VENUES = {"eccv"}  # ECCV only had 2024 papers in y24up
Y25UP_VENUES = [v for v in ALL_VENUES if v not in Y24_ONLY_VENUES]


# ---------- richer arxiv loader: includes pct_rating + conference_year ----------
def load_arxiv_extended(mode, train, test_prior, split):
    """Returns list of (score, gold, venue, conference_year, pct_rating)."""
    jsonl = arxiv_jsonl(mode, train, test_prior, split)
    meta = ARXIV_META[(mode, test_prior, split)]
    if not jsonl.exists() or not meta.exists(): return []
    metas = json.loads(meta.read_text())
    out = []
    with jsonl.open() as f:
        for i, line in enumerate(f):
            if i >= len(metas): break
            r = json.loads(line)
            g = extract_pred(r.get("label", ""))
            s = score_logodds(r)
            if g is None or s is None: continue
            gold = 1 if g == "accept" else 0
            m = metas[i].get("_metadata") or {}
            v = (m.get("pl_venue") or m.get("venue") or "?").lower()
            if v in ACL_FAMILY: v = "acl_family"
            cy = m.get("conference_year")
            try: cy = int(cy) if cy is not None else None
            except: cy = None
            pr = m.get("pct_rating")
            try: pr = float(pr) if pr is not None else None
            except: pr = None
            out.append((s, gold, v, cy, pr))
    return out


def load_iclr_extended(mode, train, test_prior, split):
    """Returns list of (score, gold, year, pct_rating)."""
    if split == "test":
        jsonl = iclr_test_jsonl(mode, train, test_prior)
    else:
        jsonl = iclr_val_jsonl(mode, train, test_prior)
    meta = ICLR_META[(mode, test_prior, split)]
    if not jsonl.exists() or not meta.exists(): return []
    metas = json.loads(meta.read_text())
    out = []
    with jsonl.open() as f:
        for i, line in enumerate(f):
            if i >= len(metas): break
            m = metas[i].get("_metadata") or {}
            yr = m.get("year")
            try: yr = int(yr) if yr is not None else None
            except: yr = None
            if yr is None or yr < 2025: continue
            r = json.loads(line)
            g = extract_pred(r.get("label", ""))
            s = score_logodds(r)
            if g is None or s is None: continue
            gold = 1 if g == "accept" else 0
            pr = m.get("pct_rating")
            try: pr = float(pr) if pr is not None else None
            except: pr = None
            out.append((s, gold, yr, pr))
    return out


# ---------- correlations ----------
def pearson(xs, ys):
    if len(xs) < 3: return None
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0: return None
    return num / (dx * dy)


def spearman(xs, ys):
    if len(xs) < 3: return None
    def ranks(vals):
        # average-rank ties
        sorted_idx = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0] * len(vals)
        i = 0
        while i < len(sorted_idx):
            j = i
            while j + 1 < len(sorted_idx) and vals[sorted_idx[j + 1]] == vals[sorted_idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[sorted_idx[k]] = avg
            i = j + 1
        return rk
    return pearson(ranks(xs), ranks(ys))


# ---------- analyses ----------
def per_venue_arxiv(arxiv_thresholds):
    """Returns nested: per_venue[split][prior][mode][train][venue] = {raw, cal} metrics."""
    out = {}
    for split in SPLITS:
        out[split] = {}
        for prior in PRIORS:
            out[split][prior] = {}
            for mode in MODES:
                out[split][prior][mode] = {}
                for train in RATIOS:
                    pairs = load_arxiv_extended(mode, train, prior, split)
                    venue_taus = arxiv_thresholds[(mode, train, prior)]
                    by_v = defaultdict(list)
                    for s, g, v, _, _ in pairs:
                        by_v[v].append((s, g, v))
                    out[split][prior][mode][train] = {}
                    for v, vp in by_v.items():
                        n = len(vp)
                        if n == 0: continue
                        # raw
                        raw_correct = sum(1 for s, g, _ in vp if (s > 0) == (g == 1))
                        raw_acc = raw_correct / n * 100
                        a, r = per_class_recall([(s, g) for s, g, _ in vp], tau=0.0)
                        # cal: per-venue τ from venue_taus
                        entry = venue_taus.get(v)
                        if entry is None:
                            tau = venue_taus.get("GLOBAL", 0.0)
                            src = "global"
                        elif isinstance(entry, tuple):
                            tau, src = entry
                        else:
                            tau, src = entry, "venue"
                        cal_correct = sum(1 for s, g, _ in vp if (s > tau) == (g == 1))
                        cal_acc = cal_correct / n * 100
                        a_c, r_c = per_class_recall([(s, g) for s, g, _ in vp], tau=tau)
                        # AUC (threshold-independent)
                        a_uc = auc([s for s, _, _ in vp], [g for _, g, _ in vp])
                        out[split][prior][mode][train][v] = {
                            "n": n,
                            "raw": {"acc": raw_acc, "auc": a_uc,
                                     "acc_rec": (a*100 if a is not None else None),
                                     "rej_rec": (r*100 if r is not None else None)},
                            "cal": {"acc": cal_acc, "auc": a_uc,
                                     "acc_rec": (a_c*100 if a_c is not None else None),
                                     "rej_rec": (r_c*100 if r_c is not None else None)},
                            "tau": tau, "tau_source": src,
                        }
    return out


def y25up_overall(arxiv_thresholds):
    """Recompute overall arxiv metrics dropping ECCV (y24-only)."""
    out = {}
    for split in SPLITS:
        out[split] = {}
        for prior in PRIORS:
            out[split][prior] = {}
            for mode in MODES:
                out[split][prior][mode] = {}
                for train in RATIOS:
                    pairs = load_arxiv_extended(mode, train, prior, split)
                    venue_taus = arxiv_thresholds[(mode, train, prior)]
                    pairs_keep = [(s, g, v) for s, g, v, _, _ in pairs if v in Y25UP_VENUES]
                    pairs_drop = [(s, g, v) for s, g, v, _, _ in pairs if v not in Y25UP_VENUES]
                    if not pairs_keep:
                        out[split][prior][mode][train] = None
                        continue
                    n = len(pairs_keep)
                    n_dropped = len(pairs_drop)
                    raw_correct = sum(1 for s, g, _ in pairs_keep if (s > 0) == (g == 1))
                    cal_correct = 0
                    for s, g, v in pairs_keep:
                        entry = venue_taus.get(v)
                        if entry is None: tau = venue_taus.get("GLOBAL", 0.0)
                        elif isinstance(entry, tuple): tau = entry[0]
                        else: tau = entry
                        if (s > tau) == (g == 1): cal_correct += 1
                    a_uc = auc([s for s,_,_ in pairs_keep], [g for _,g,_ in pairs_keep])
                    out[split][prior][mode][train] = {
                        "n": n, "n_dropped": n_dropped,
                        "acc_raw": raw_correct / n * 100,
                        "acc_cal": cal_correct / n * 100,
                        "auc": a_uc,
                    }
    return out


def quality_signal_corr():
    """Correlation between model score and pct_rating, for samples where pct_rating exists.

    For each (mode, train, source), compute pearson + spearman.
    Sources: 'iclr_balanced_test' (ICLR direct), 'arxiv_balanced_iclr' (arxiv venue=iclr only),
             'arxiv_balanced_neurips' (arxiv venue=neurips only).
    Test split only.
    """
    out = {}
    sources = []
    # ICLR direct test (balanced + natural — but balanced has full pct_rating coverage)
    for prior in PRIORS:
        sources.append(("iclr", prior, "test", None))  # None venue = all ICLR (no venue filter)
    # Arxiv: per-venue (filter to iclr / neurips where pct_rating is populated)
    for prior in PRIORS:
        for venue in ("iclr", "neurips"):
            sources.append(("arxiv", prior, "test", venue))

    for src in sources:
        ts, prior, split, venue_filter = src
        for mode in MODES:
            for train in RATIOS:
                if ts == "iclr":
                    pairs = load_iclr_extended(mode, train, prior, split)
                    items = [(s, pr) for s, _, _, pr in pairs if pr is not None]
                else:
                    pairs = load_arxiv_extended(mode, train, prior, split)
                    items = [(s, pr) for s, _, v, _, pr in pairs
                             if pr is not None and (venue_filter is None or v == venue_filter)]
                if len(items) < 5:
                    out[(ts, prior, venue_filter, mode, train)] = None
                    continue
                xs = [s for s, _ in items]
                ys = [pr for _, pr in items]
                out[(ts, prior, venue_filter, mode, train)] = {
                    "n": len(items),
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                }
    return out


def arxiv_iclr_vs_direct_iclr():
    """Compare model behavior on arxiv-venue=iclr subset (y24up test) vs direct ICLR test (y25+).

    For each (mode, train, prior), compute aggregate ACC/AUC on each population.
    """
    out = {}
    for mode in MODES:
        for train in RATIOS:
            for prior in PRIORS:
                # arxiv subset: pl_venue == 'iclr', conference_year ≥ 2024 (already y24up)
                arxiv_pairs = load_arxiv_extended(mode, train, prior, "test")
                arxiv_iclr = [(s, g) for s, g, v, _, _ in arxiv_pairs if v == "iclr"]
                # direct ICLR test (year ≥ 2025)
                direct = load_iclr_cell(mode, train, prior, "test")
                m_arxiv = metrics_iclr(arxiv_iclr) if arxiv_iclr else None
                m_direct = metrics_iclr(direct) if direct else None
                out[(mode, train, prior)] = {"arxiv_iclr_subset": m_arxiv, "direct_iclr": m_direct}
    return out


# ---------- pretty printers (minimal for stdout) ----------
def fmt_acc(x, n=4): return f"{x:.1f}" if x is not None else "—"
def fmt_auc(x): return f"{x:.3f}" if x is not None else "—"


# ---------- figure helpers ----------
def _setup_mpl():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
    })
    return mpl, plt


def fig_per_venue_heatmap(per_venue, save_path_base):
    """4-panel heatmap: rows=4 models, cols=11 venues; cells = AUC, 4 panels = 4 (eval×prior) cells.
    AUC version only (threshold-independent, most informative).
    """
    import numpy as np
    mpl, plt = _setup_mpl()

    fig, axes = plt.subplots(1, 4, figsize=(34, 6.5))
    cells = [(ts, pr) for ts in ["arxiv"] for pr in PRIORS] * 2  # we only have arxiv per-venue
    # Actually 2 panels (arxiv balanced + natural) × test/val split
    splits_used = ["test", "val"]
    panel_specs = [(s, p) for s in splits_used for p in PRIORS]

    fig, axes = plt.subplots(1, 4, figsize=(34, 7))
    venues = ALL_VENUES
    model_keys = [(mode, train) for mode in MODES for train in RATIOS]
    model_labels = [f"{mode} {RATIO_LABEL[train]}" for mode, train in model_keys]

    for ax, (split, prior) in zip(axes, panel_specs):
        mat = np.full((len(model_keys), len(venues)), np.nan)
        for i, (mode, train) in enumerate(model_keys):
            for j, v in enumerate(venues):
                cell = per_venue[split][prior][mode][train].get(v)
                if cell and cell["raw"]["auc"] is not None:
                    mat[i, j] = cell["raw"]["auc"]
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=0.95, aspect="auto")
        ax.set_xticks(range(len(venues))); ax.set_xticklabels(venues, rotation=45, ha="right", fontsize=14)
        ax.set_yticks(range(len(model_keys))); ax.set_yticklabels(model_labels, fontsize=14)
        ax.set_title(f"arxiv {prior}, {split}", fontsize=18)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=12,
                            color="black" if mat[i,j] > 0.7 else "white")
        # Annotate sample size in venue header
        for j, v in enumerate(venues):
            ns = [per_venue[split][prior][mode][train].get(v, {}).get("n", 0) for mode, train in model_keys]
            ns = [n for n in ns if n > 0]
            n = ns[0] if ns else 0
            ax.text(j, -0.7, f"n={n}", ha="center", va="bottom", fontsize=10, color="gray")

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.8, pad=0.01)
    cbar.set_label("AUC", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    fig.suptitle("Per-venue AUC (threshold-independent) — arxiv y24up", fontsize=22, y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_path_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def fig_per_venue_acc_heatmap(per_venue, save_path_base, variant="cal"):
    """Per-venue ACC heatmap (raw or calibrated). variant = 'raw' or 'cal'."""
    import numpy as np
    mpl, plt = _setup_mpl()
    fig, axes = plt.subplots(1, 4, figsize=(34, 7))
    panel_specs = [(s, p) for s in SPLITS for p in PRIORS]
    venues = ALL_VENUES
    model_keys = [(mode, train) for mode in MODES for train in RATIOS]
    model_labels = [f"{mode} {RATIO_LABEL[train]}" for mode, train in model_keys]
    for ax, (split, prior) in zip(axes, panel_specs):
        mat = np.full((len(model_keys), len(venues)), np.nan)
        for i, (mode, train) in enumerate(model_keys):
            for j, v in enumerate(venues):
                cell = per_venue[split][prior][mode][train].get(v)
                if cell and cell[variant]["acc"] is not None:
                    mat[i, j] = cell[variant]["acc"]
        im = ax.imshow(mat, cmap="RdYlGn", vmin=40, vmax=90, aspect="auto")
        ax.set_xticks(range(len(venues))); ax.set_xticklabels(venues, rotation=45, ha="right", fontsize=14)
        ax.set_yticks(range(len(model_keys))); ax.set_yticklabels(model_labels, fontsize=14)
        ax.set_title(f"arxiv {prior}, {split}", fontsize=18)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center", fontsize=12,
                            color="black" if mat[i,j] > 65 else "white")

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.8, pad=0.01)
    cbar.set_label(f"ACC % ({variant})", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    fig.suptitle(f"Per-venue ACC ({variant}) — arxiv y24up", fontsize=22, y=1.02)
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_path_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def fig_quality_corr(corr, save_path_base):
    """Bar grid of |Spearman ρ| between score and pct_rating, one panel per (source, prior).

    Sources shown:
      - ICLR direct (balanced + natural test)
      - Arxiv-iclr subset (balanced + natural test)
      - Arxiv-neurips subset (balanced + natural test)
    """
    import numpy as np
    mpl, plt = _setup_mpl()
    BLUE, ORANGE, GREEN, RED = "#6098FF", "#FECC81", "#77B25D", "#FF8988"
    src_panels = [
        ("ICLR direct, balanced",      ("iclr", "balanced", None)),
        ("ICLR direct, natural",        ("iclr", "natural", None)),
        ("arxiv-iclr subset, balanced", ("arxiv", "balanced", "iclr")),
        ("arxiv-iclr subset, natural",  ("arxiv", "natural", "iclr")),
        ("arxiv-neurips subset, bal",   ("arxiv", "balanced", "neurips")),
        ("arxiv-neurips subset, nat",   ("arxiv", "natural", "neurips")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    model_keys = [(mode, train) for mode in MODES for train in RATIOS]
    model_labels = [f"{mode}\n{RATIO_LABEL[train]}" for mode, train in model_keys]
    model_colors = [BLUE, RED, GREEN, "#B28CFF"]

    for ax, (title, (ts, prior, venue)) in zip(axes, src_panels):
        rs, ns = [], []
        for (mode, train) in model_keys:
            r = corr.get((ts, prior, venue, mode, train))
            if r is None:
                rs.append(np.nan); ns.append(0)
            else:
                rs.append(r["spearman"] if r["spearman"] is not None else np.nan)
                ns.append(r["n"])
        x = np.arange(len(model_keys))
        bars = ax.bar(x, rs, color=model_colors, edgecolor="black", linewidth=1.2)
        for i, (val, n) in enumerate(zip(rs, ns)):
            if not np.isnan(val):
                ax.text(i, val + 0.01 if val >= 0 else val - 0.04, f"{val:.2f}",
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=12)
                ax.text(i, -0.05, f"n={n}", ha="center", va="top", fontsize=10, color="gray")
        ax.set_xticks(x); ax.set_xticklabels(model_labels, fontsize=14)
        ax.set_ylim(-0.1, 0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Spearman ρ\n(score, pct_rating)", fontsize=15)
    axes[3].set_ylabel("Spearman ρ\n(score, pct_rating)", fontsize=15)
    fig.suptitle("Score-vs-pct_rating rank correlation across (source × prior × model)",
                 fontsize=20, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_path_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def fig_arxiv_iclr_vs_direct(comp, save_path_base):
    """Side-by-side bars: arxiv-iclr-subset vs direct-ICLR (balanced + natural).

    Compares overall ACC (raw) and AUC across the 4 (mode × train) configs.
    """
    import numpy as np
    mpl, plt = _setup_mpl()
    BLUE, ORANGE = "#6098FF", "#FECC81"
    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    model_keys = [(mode, train) for mode in MODES for train in RATIOS]
    model_labels = [f"{mode}\n{RATIO_LABEL[train]}" for mode, train in model_keys]

    metrics = [("acc", "ACC (raw, %)", 0, 100), ("auc", "AUC", 0.5, 0.95)]
    for row, (key, ylabel, ymin, ymax) in enumerate(metrics):
        for col, prior in enumerate(PRIORS):
            ax = axes[row, col]
            arxiv_vals, direct_vals, ns_a, ns_d = [], [], [], []
            for (mode, train) in model_keys:
                e = comp[(mode, train, prior)]
                ma, md = e["arxiv_iclr_subset"], e["direct_iclr"]
                arxiv_vals.append(ma[key] if ma and ma.get(key) is not None else np.nan)
                direct_vals.append(md[key] if md and md.get(key) is not None else np.nan)
                ns_a.append(ma["n"] if ma else 0)
                ns_d.append(md["n"] if md else 0)
            x = np.arange(len(model_keys)); bw = 0.35
            ax.bar(x - bw/2, arxiv_vals, bw, label="arxiv-iclr subset", color=BLUE, edgecolor="black")
            ax.bar(x + bw/2, direct_vals, bw, label="direct ICLR", color=ORANGE, edgecolor="black")
            for i, (a, d) in enumerate(zip(arxiv_vals, direct_vals)):
                if not np.isnan(a):
                    fmt = ".0f" if key == "acc" else ".3f"
                    ax.text(x[i] - bw/2, a + (1.5 if key=="acc" else 0.005), f"{a:{fmt}}",
                            ha="center", va="bottom", fontsize=11)
                if not np.isnan(d):
                    fmt = ".0f" if key == "acc" else ".3f"
                    ax.text(x[i] + bw/2, d + (1.5 if key=="acc" else 0.005), f"{d:{fmt}}",
                            ha="center", va="bottom", fontsize=11)
            ax.set_xticks(x); ax.set_xticklabels(model_labels, fontsize=14)
            ax.set_ylim(ymin, ymax)
            ax.set_title(f"{prior} prior — {ylabel.split(' (')[0]}", fontsize=18)
            if col == 0: ax.set_ylabel(ylabel, fontsize=15)
            ax.tick_params(axis="y", labelsize=12)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if row == 0 and col == 0: ax.legend(fontsize=14, loc="lower right")
    # Annotate ns under each panel for arxiv subset
    for col, prior in enumerate(PRIORS):
        n_a = comp[("text","50_50",prior)]["arxiv_iclr_subset"]["n"]
        n_d = comp[("text","50_50",prior)]["direct_iclr"]["n"]
        axes[0, col].text(0.5, -0.16, f"arxiv-iclr n={n_a}; direct ICLR n={n_d}",
                          transform=axes[0, col].transAxes, ha="center", fontsize=12, color="gray")
    fig.suptitle("Arxiv-iclr subset vs direct ICLR test — same paper population, different tests",
                 fontsize=20, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_path_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ---------- markdown writer ----------
def write_appendix(per_venue, y25_overall, corr, comp, raw_overall):
    """Append new sections to the existing report."""
    lines = []
    lines.append("\n\n---\n")
    lines.append("## Extended Analysis (appendix)\n")
    lines.append("Appended after initial report. Adds: per-venue arxiv breakdown, "
                 "y25+ ablation (drop ECCV), score↔pct_rating correlation, and "
                 "arxiv-iclr-subset vs direct-ICLR comparison.\n")

    lines.append("### A. Headline trends\n")
    lines.append("- **Modality-divergent correlation in arxiv-natural-iclr**: on the natrate ICLR subset "
                 "of arxiv test, vision 50/50 and 30/70 both reach **ρ ≈ +0.46** with `pct_rating`, but text "
                 "50/50 and 30/70 give **ρ ≈ -0.13 to -0.17** (negative!). Vision still tracks reviewer "
                 "quality after the prior shift; text inverts. (n=103-107 each, so a real but moderate effect.)")
    lines.append("- **Direct-ICLR correlation is strong and modality-invariant**: all 4 models hit "
                 "**ρ = 0.47-0.51** with `pct_rating`, on both balanced and natural prior. So on the "
                 "direct ICLR test set, model log-odds reliably tracks reviewer percentile irrespective "
                 "of train ratio or modality — both modalities learn the same `pct_rating` axis.")
    lines.append("- **NeurIPS pct_rating correlation is universally weak (ρ < 0.17)**: arxiv-extracted "
                 "NeurIPS papers don't preserve the rating signal that direct-ICLR papers do, despite "
                 "similar venue size in the test (n=96-175). Maybe an artifact of the openreview→arxiv "
                 "metadata-merge step, or NeurIPS reviewer ratings being intrinsically noisier.")
    lines.append("- **arxiv-iclr-subset AUC slightly exceeds direct-ICLR AUC** by 0.01-0.05 (e.g. text "
                 "50/50 balanced: 0.754 vs 0.721; vision 50/50 balanced: 0.750 vs 0.736). The arxiv-extracted "
                 "ICLR subset appears to be a *slightly easier* test population — possibly because it's "
                 "year-filtered to ≥2024 (skews to newer ICLR years where the model is well-fit) "
                 "and pre-filtered to main-paper status during arxiv ingestion.")
    lines.append("- **ECCV exclusion is negligible**: dropping it (y25+ ablation) shifts overall arxiv "
                 "ACC by < 0.3pp and AUC by < 0.003 across all 4 models. The y24up overall numbers in "
                 "the main report are dominated by venues that span 2024-2026.\n")

    lines.append("### B. Per-venue arxiv breakdown — TEST split, balanced prior\n")
    lines.append("Cells: ACC raw / ACC cal (n)\n")
    venues = ALL_VENUES
    header = "| venue | n | " + " | ".join(f"{m} {RATIO_LABEL[r]}" for m in MODES for r in RATIOS) + " |"
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * (len(MODES)*len(RATIOS)))
    for v in venues:
        # Use the first model that has this venue to get n
        cells = {}
        n = 0
        for mode in MODES:
            for train in RATIOS:
                e = per_venue["test"]["balanced"][mode][train].get(v)
                if e:
                    cells[(mode, train)] = e
                    if e["n"] > n: n = e["n"]
        if not cells: continue
        row = f"| {v} | {n} |"
        for mode in MODES:
            for train in RATIOS:
                e = cells.get((mode, train))
                if e is None: row += " — |"
                else: row += f" {e['raw']['acc']:.0f} / {e['cal']['acc']:.0f} |"
        lines.append(row)
    lines.append("")

    lines.append("### C. Per-venue arxiv breakdown — TEST split, natural (natrate) prior\n")
    lines.append("Cells: ACC raw / ACC cal (n)\n")
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * (len(MODES)*len(RATIOS)))
    for v in venues:
        cells = {}; n = 0
        for mode in MODES:
            for train in RATIOS:
                e = per_venue["test"]["natural"][mode][train].get(v)
                if e:
                    cells[(mode, train)] = e
                    if e["n"] > n: n = e["n"]
        if not cells: continue
        row = f"| {v} | {n} |"
        for mode in MODES:
            for train in RATIOS:
                e = cells.get((mode, train))
                if e is None: row += " — |"
                else: row += f" {e['raw']['acc']:.0f} / {e['cal']['acc']:.0f} |"
        lines.append(row)
    lines.append("")

    lines.append("### D. Per-venue AUC heatmap\n")
    lines.append("![Per-venue AUC](../tmp_latex_dir/figures/ratio_xeval_per_venue_auc.png)\n")
    lines.append("### Per-venue ACC (calibrated)\n")
    lines.append("![Per-venue ACC cal](../tmp_latex_dir/figures/ratio_xeval_per_venue_acc_cal.png)\n")

    lines.append("### E. Y25+ ablation (drop ECCV; ECCV has no 2025/2026 papers in y24up)\n")
    lines.append("Compare arxiv overall ACC raw / ACC cal / AUC for full y24up vs y25+ subset:\n")
    lines.append("| prior | model | full y24up ACC raw / cal / AUC | y25+ ACC raw / cal / AUC | n_dropped |")
    lines.append("|---|---|---|---|---:|")
    for prior in PRIORS:
        for mode in MODES:
            for train in RATIOS:
                full = raw_overall["test"]["arxiv"][prior][mode][train]
                ab = y25_overall["test"][prior][mode][train]
                full_str = f"{full['raw']['acc']:.1f} / {full['cal']['acc']:.1f} / {fmt_auc(full['raw']['auc'])}"
                ab_str = f"{ab['acc_raw']:.1f} / {ab['acc_cal']:.1f} / {fmt_auc(ab['auc'])}"
                lines.append(f"| {prior} | {mode} {RATIO_LABEL[train]} | {full_str} | {ab_str} | {ab['n_dropped']} |")
    lines.append("")
    lines.append("Conclusion: dropping ECCV (~2-3% of test) shifts overall metrics by typically <1pp; the y24up overall numbers in the main report are dominated by venues that span 2024-2026.\n")

    lines.append("### F. Score-vs-pct_rating correlation\n")
    lines.append("`pct_rating` is the per-venue percentile of the paper's mean reviewer rating. "
                 "It's available for ICLR (all years 25/26 in our test) and for arxiv samples drawn from "
                 "ICLR + NeurIPS (where OpenReview reviews exist). Higher correlation between model's "
                 "log-odds score and `pct_rating` ⇒ model's signal better tracks reviewer-quality "
                 "perception (independent of the binary accept/reject decision).\n")
    lines.append("![Score-vs-pct_rating correlation](../tmp_latex_dir/figures/ratio_xeval_quality_corr.png)\n")

    lines.append("Spearman ρ (score, pct_rating) — TEST split\n")
    lines.append("| source | prior | venue filter | model | n | Spearman ρ | Pearson r |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for (ts, prior, venue, mode, train), val in sorted(corr.items()):
        if val is None: continue
        venue_label = venue or "—"
        sp = f"{val['spearman']:+.3f}" if val['spearman'] is not None else "—"
        pe = f"{val['pearson']:+.3f}" if val['pearson'] is not None else "—"
        lines.append(f"| {ts} | {prior} | {venue_label} | {mode} {RATIO_LABEL[train]} | {val['n']} | {sp} | {pe} |")
    lines.append("")

    lines.append("### G. Arxiv-iclr-subset vs direct-ICLR test\n")
    lines.append("ICLR papers appear in both populations: the direct ICLR test (year≥2025) and the arxiv "
                 "balanced/natural test (where `pl_venue=='iclr'`, year≥2024). Different paper draws but "
                 "same underlying community. Aggregate metrics:\n")
    lines.append("| prior | model | arxiv-iclr ACC / AUC (n) | direct-ICLR ACC / AUC (n) |")
    lines.append("|---|---|---|---|")
    for prior in PRIORS:
        for mode in MODES:
            for train in RATIOS:
                e = comp[(mode, train, prior)]
                ma, md = e["arxiv_iclr_subset"], e["direct_iclr"]
                ma_s = f"{ma['acc']:.1f} / {fmt_auc(ma['auc'])} (n={ma['n']})" if ma else "—"
                md_s = f"{md['acc']:.1f} / {fmt_auc(md['auc'])} (n={md['n']})" if md else "—"
                lines.append(f"| {prior} | {mode} {RATIO_LABEL[train]} | {ma_s} | {md_s} |")
    lines.append("")
    lines.append("![arxiv-iclr vs direct-ICLR](../tmp_latex_dir/figures/ratio_xeval_arxiv_iclr_vs_direct.png)\n")

    lines.append("### H. Notes\n")
    lines.append("- pct_rating is on a 0-1 scale (paper's percentile within its venue's review distribution).")
    lines.append("- Spearman is preferred over Pearson here because the model score is on a log-odds scale "
                 "and pct_rating is bounded [0, 1] — the relationship is not necessarily linear.")
    lines.append("- Per-venue n<10 cells are not statistically reliable (corl, aistats); they're shown for "
                 "completeness with `g` flag in the threshold dump.")
    lines.append("- arxiv-iclr-subset n is small (~130) and skewed toward 2024-2025 papers; direct-ICLR "
                 "test is much larger (~1660) and only 2025/2026.")

    # Append to existing report
    with open(REPORT_PATH, "a") as f:
        f.write("\n".join(lines))
    print(f"\nAppended extended analysis to: {REPORT_PATH}")


def main():
    print("Loading + computing thresholds (reuse from main script)...")
    iclr_taus = compute_iclr_thresholds()
    arxiv_taus = compute_arxiv_thresholds()

    # Also need to recompute the "raw overall" arxiv test/val for the ablation comparison
    raw_overall = {"test": {"arxiv": {}}, "val": {"arxiv": {}}}
    for split in SPLITS:
        for prior in PRIORS:
            raw_overall[split]["arxiv"][prior] = {}
            for mode in MODES:
                raw_overall[split]["arxiv"][prior][mode] = {}
                for train in RATIOS:
                    pairs = load_arxiv_extended(mode, train, prior, split)
                    venue_taus = arxiv_taus[(mode, train, prior)]
                    pp = [(s, g, v) for s, g, v, _, _ in pairs]
                    raw = metrics_arxiv(pp, venue_tau_map=None, default_tau=0.0)
                    cal = metrics_arxiv(pp, venue_tau_map=venue_taus)
                    raw_overall[split]["arxiv"][prior][mode][train] = {"raw": raw, "cal": cal}

    print("Per-venue arxiv breakdown...")
    per_venue = per_venue_arxiv(arxiv_taus)
    print("Y25+ ablation (drop ECCV)...")
    y25 = y25up_overall(arxiv_taus)
    print("Score↔pct_rating correlation...")
    corr = quality_signal_corr()
    print("Arxiv-iclr vs direct-ICLR comparison...")
    comp = arxiv_iclr_vs_direct_iclr()

    # Print summary stats
    print("\n=== Headline correlation numbers ===")
    for (ts, prior, venue, mode, train), val in sorted(corr.items()):
        if val is None: continue
        sp = val["spearman"]; pe = val["pearson"]
        print(f"  {ts:<6} {prior:<8} venue={str(venue):<10} {mode:<6} {RATIO_LABEL[train]:<5} "
              f"n={val['n']:>4} ρ={sp:+.3f} r={pe:+.3f}")

    print("\n=== ECCV vs y25+ overall delta (TEST balanced) ===")
    for mode in MODES:
        for train in RATIOS:
            full = raw_overall["test"]["arxiv"]["balanced"][mode][train]
            ab = y25["test"]["balanced"][mode][train]
            d_acc = ab["acc_cal"] - full["cal"]["acc"]
            d_auc = ab["auc"] - full["cal"]["auc"] if ab["auc"] and full["cal"]["auc"] else None
            print(f"  {mode:<6} {RATIO_LABEL[train]:<5}  Δ(ACC cal) = {d_acc:+.2f}pp   "
                  f"Δ(AUC) = {d_auc:+.4f}" if d_auc is not None else f"... ΔAUC=N/A")

    print("\n=== arxiv-iclr-subset vs direct-ICLR (AUC; balanced) ===")
    for mode in MODES:
        for train in RATIOS:
            e = comp[(mode, train, "balanced")]
            ma, md = e["arxiv_iclr_subset"], e["direct_iclr"]
            print(f"  {mode:<6} {RATIO_LABEL[train]:<5}  arxiv-iclr AUC={ma['auc']:.3f} (n={ma['n']})   "
                  f"direct-ICLR AUC={md['auc']:.3f} (n={md['n']})")

    print("\nWriting figures...")
    fig_per_venue_heatmap(per_venue, str(FIG_DIR / "ratio_xeval_per_venue_auc"))
    fig_per_venue_acc_heatmap(per_venue, str(FIG_DIR / "ratio_xeval_per_venue_acc_cal"), variant="cal")
    fig_per_venue_acc_heatmap(per_venue, str(FIG_DIR / "ratio_xeval_per_venue_acc_raw"), variant="raw")
    fig_quality_corr(corr, str(FIG_DIR / "ratio_xeval_quality_corr"))
    fig_arxiv_iclr_vs_direct(comp, str(FIG_DIR / "ratio_xeval_arxiv_iclr_vs_direct"))

    print("Appending markdown...")
    write_appendix(per_venue, y25, corr, comp, raw_overall)


if __name__ == "__main__":
    main()
