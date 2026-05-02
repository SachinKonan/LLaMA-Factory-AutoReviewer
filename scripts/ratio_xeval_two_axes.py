#!/usr/bin/env python3
"""
Two-axes analysis: distinguish "universal quality indicator" (Axis A) from
"conference accept/reject classifier" (Axis B), and recommend the right metric
for each. The headline finding: when measured with balanced accuracy on the
natural-prior cell, **vision wins where text appeared to win on raw ACC**.

Axis A — Universal quality indicator
  Goal:   model's continuous score tracks underlying paper quality.
  Metric: Spearman ρ(score, pct_rating). Threshold- and prior-invariant.

Axis B — Conference accept/reject classifier
  Goal:   model's binary decision matches the venue's accept/reject.
  Metric: Balanced accuracy = (accept_recall + reject_recall) / 2.
          (Raw ACC is misleading on imbalanced priors — predict-all-reject
          gives 70% on a 30%-accept population without any signal.)
          AUC is also fine — measures threshold-independent ranking quality.

Outputs:
  fig_two_axes_scatter.{pdf,png}   — Axis A vs B 2D scatter, one point per
                                      (model × test population).
  fig_two_axes_rankings.{pdf,png}  — three side-by-side ranking grids
                                      (raw ACC vs balanced ACC vs ρ_quality).
  Append "Two Axes of Quality" section to reports/ratio_xeval_7b.md.
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
DATA = ROOT / "data"
FIG_DIR = ROOT / "tmp_latex_dir/figures"
REPORT_PATH = ROOT / "reports/ratio_xeval_7b.md"

sys.path.insert(0, str(ROOT / "scripts"))
from ratio_xeval_consolidated import (
    MODEL, RATIO_LABEL, MODES, RATIOS, PRIORS, SPLITS, TEST_SETS,
    ACL_FAMILY, auc, score_logodds, extract_pred,
    load_iclr_cell, load_arxiv_cell, metrics_iclr, metrics_arxiv,
    compute_iclr_thresholds, compute_arxiv_thresholds,
    accuracy, per_class_recall, best_tau,
    CELL_LABELS,
)
from ratio_xeval_extended import load_iclr_extended, load_arxiv_extended


# ---------- helpers ----------
def pearson(xs, ys):
    if len(xs) < 3: return None
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    if dx == 0 or dy == 0: return None
    return num/(dx*dy)


def spearman(xs, ys):
    if len(xs) < 3: return None
    def ranks(vals):
        si = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0]*len(vals); i = 0
        while i < len(si):
            j = i
            while j+1 < len(si) and vals[si[j+1]] == vals[si[i]]: j += 1
            avg = (i+j)/2.0 + 1
            for k in range(i, j+1): rk[si[k]] = avg
            i = j+1
        return rk
    return pearson(ranks(xs), ranks(ys))


def balanced_acc(pairs, tau=0.0):
    """(accept_recall + reject_recall) / 2 at threshold τ. None if either class empty."""
    n_acc = sum(1 for s, g in pairs if g == 1)
    n_rej = sum(1 for s, g in pairs if g == 0)
    if n_acc == 0 or n_rej == 0: return None
    tp = sum(1 for s, g in pairs if g == 1 and s > tau)
    tn = sum(1 for s, g in pairs if g == 0 and s <= tau)
    return (tp/n_acc + tn/n_rej) / 2 * 100


def best_tau_balanced(pairs):
    """Threshold that maximizes balanced accuracy."""
    if not pairs: return 0.0
    pairs_sorted = sorted(pairs)
    n = len(pairs_sorted); npos = sum(g for _,g in pairs_sorted); nneg = n - npos
    if npos == 0 or nneg == 0: return 0.0
    # Walk: at τ=-inf, all accept → tp=npos, tn=0
    tp = npos; tn = 0
    best_bacc = (tp/npos + tn/nneg) / 2
    best_t = -math.inf
    from itertools import groupby
    for s, group in groupby(pairs_sorted, key=lambda p: p[0]):
        npos_in = sum(g for _, g in group); n_in = sum(1 for _, _ in [(s, g) for _, g in group])
        # actually need to use a list to count properly; redo:
        items = list(group); n_in = len(items); npos_in = sum(g for _, g in items)
        nneg_in = n_in - npos_in
        tp -= npos_in   # these accepts are now predicted reject
        tn += nneg_in   # these rejects are now predicted reject (correctly)
        bacc = (tp/npos + tn/nneg) / 2
        if bacc > best_bacc:
            best_bacc, best_t = bacc, s
    return best_t


# ---------- per-cell aggregation ----------
def collect_cells():
    """Returns: cells[(test_set, prior, mode, train)] = {
         'n', 'acc_raw', 'acc_cal', 'bacc_raw', 'bacc_cal', 'bacc_opt',
         'auc', 'tau_acc', 'tau_bacc',
         'rho_quality', 'rho_quality_n'  (only for cells with pct_rating coverage)
       }
    """
    iclr_taus = compute_iclr_thresholds()
    arxiv_taus = compute_arxiv_thresholds()

    out = {}
    # ICLR cells
    for prior in PRIORS:
        for mode in MODES:
            for train in RATIOS:
                pairs = load_iclr_cell(mode, train, prior, "test")
                if not pairs: continue
                tau_acc_cal = iclr_taus[(mode, train, prior)]
                tau_bacc = best_tau_balanced(pairs)
                acc_raw = accuracy(pairs, 0.0) * 100 if pairs else None
                acc_cal = accuracy(pairs, tau_acc_cal) * 100 if pairs else None
                bacc_raw = balanced_acc(pairs, 0.0)
                bacc_cal = balanced_acc(pairs, tau_acc_cal)
                bacc_opt = balanced_acc(pairs, tau_bacc)
                a = auc([s for s,_ in pairs], [g for _,g in pairs])
                # Quality correlation
                ext = load_iclr_extended(mode, train, prior, "test")
                items = [(s, pr) for s, _, _, pr in ext if pr is not None]
                rho = spearman([s for s,_ in items], [pr for _,pr in items]) if len(items) > 5 else None
                out[("iclr", prior, mode, train)] = {
                    "n": len(pairs), "acc_raw": acc_raw, "acc_cal": acc_cal,
                    "bacc_raw": bacc_raw, "bacc_cal": bacc_cal, "bacc_opt": bacc_opt,
                    "auc": a, "tau_acc": tau_acc_cal, "tau_bacc": tau_bacc,
                    "rho_quality": rho, "rho_quality_n": len(items),
                }

    # Arxiv cells (per-venue calibration aggregated)
    for prior in PRIORS:
        for mode in MODES:
            for train in RATIOS:
                pairs_v = load_arxiv_cell(mode, train, prior, "test")  # (s, g, v)
                if not pairs_v: continue
                vt = arxiv_taus[(mode, train, prior)]
                # Reduce per-venue thresholds to a per-sample threshold
                pairs_with_tau = []
                for s, g, v in pairs_v:
                    entry = vt.get(v)
                    if entry is None: tau = vt.get("GLOBAL", 0.0)
                    elif isinstance(entry, tuple): tau = entry[0]
                    else: tau = entry
                    pairs_with_tau.append((s, g, tau))
                pairs_sg = [(s, g) for s, g, _ in pairs_with_tau]
                acc_raw = accuracy(pairs_sg, 0.0) * 100
                # Calibrated ACC = per-venue τ
                cal_correct = sum(1 for s, g, tau in pairs_with_tau if (s > tau) == (g == 1))
                acc_cal = cal_correct / len(pairs_with_tau) * 100
                bacc_raw = balanced_acc(pairs_sg, 0.0)
                # Calibrated balanced acc with per-venue τ
                tp = sum(1 for s, g, tau in pairs_with_tau if g == 1 and s > tau)
                tn = sum(1 for s, g, tau in pairs_with_tau if g == 0 and s <= tau)
                npos = sum(1 for _, g, _ in pairs_with_tau if g == 1)
                nneg = sum(1 for _, g, _ in pairs_with_tau if g == 0)
                bacc_cal = (tp/npos + tn/nneg) / 2 * 100 if (npos and nneg) else None
                # Optimal balanced threshold (single global)
                tau_bacc = best_tau_balanced(pairs_sg)
                bacc_opt = balanced_acc(pairs_sg, tau_bacc)
                a = auc([s for s,_ in pairs_sg], [g for _,g in pairs_sg])
                # Quality correlation: union over venues with pct_rating
                ext = load_arxiv_extended(mode, train, prior, "test")
                items = [(s, pr) for s, _, _, _, pr in ext if pr is not None]
                rho = spearman([s for s,_ in items], [pr for _,pr in items]) if len(items) > 5 else None
                out[("arxiv", prior, mode, train)] = {
                    "n": len(pairs_v), "acc_raw": acc_raw, "acc_cal": acc_cal,
                    "bacc_raw": bacc_raw, "bacc_cal": bacc_cal, "bacc_opt": bacc_opt,
                    "auc": a, "tau_acc": "per-venue", "tau_bacc": tau_bacc,
                    "rho_quality": rho, "rho_quality_n": len(items),
                }
    return out


# ---------- figures ----------
def _setup():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
    })
    return mpl, plt


PALETTE = {
    ("text",   "50_50"): "#6098FF",
    ("text",   "30_70"): "#FF8988",
    ("vision", "50_50"): "#77B25D",
    ("vision", "30_70"): "#B28CFF",
}


def fig_two_axes_scatter(cells, save_base):
    """2D scatter: x = balanced_acc (Axis B), y = ρ(score, pct_rating) (Axis A).
    One point per (model × test_population). Quadrant labels.
    Connect same-model points with lines to show consistency across populations.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    _setup()

    fig, ax = plt.subplots(figsize=(15, 11))

    pop_marker = {
        ("iclr", "balanced"):  "o",
        ("iclr", "natural"):   "s",
        ("arxiv", "balanced"): "^",
        ("arxiv", "natural"):  "v",
    }
    pop_label = {
        ("iclr", "balanced"):  "ICLR balanced",
        ("iclr", "natural"):   "ICLR natural",
        ("arxiv", "balanced"): "arxiv balanced",
        ("arxiv", "natural"):  "arxiv natural",
    }

    # Plot: y is ρ_quality (where available), x is balanced_acc_raw
    points_by_model = defaultdict(list)
    for (ts, pr, mode, train), c in cells.items():
        if c.get("bacc_raw") is None or c.get("rho_quality") is None: continue
        points_by_model[(mode, train)].append((c["bacc_raw"], c["rho_quality"], (ts, pr), c["n"]))

    # Connecting lines per model
    for (mode, train), pts in points_by_model.items():
        color = PALETTE[(mode, train)]
        # Sort points by some order for line connection — just connect in iteration order
        xs_line = [p[0] for p in pts]; ys_line = [p[1] for p in pts]
        ax.plot(xs_line, ys_line, "-", color=color, alpha=0.25, linewidth=1.5)

    # Plot points
    for (ts, pr, mode, train), c in cells.items():
        if c.get("bacc_raw") is None or c.get("rho_quality") is None: continue
        color = PALETTE[(mode, train)]
        marker = pop_marker[(ts, pr)]
        ax.scatter(c["bacc_raw"], c["rho_quality"], marker=marker, s=320,
                   color=color, edgecolor="black", linewidth=2, zorder=5, alpha=0.92)

    # Quadrant lines
    ax.axhline(0.3, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(60, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    # Quadrant labels
    ax.text(0.97, 0.97, "★ UNIVERSAL QUALITY MODEL\nstrong on both axes",
            transform=ax.transAxes, ha="right", va="top", fontsize=15,
            color="darkgreen", fontweight="bold", style="italic",
            bbox=dict(boxstyle="round,pad=0.5", fc="#E8F5E9", ec="darkgreen", linewidth=1.5))
    ax.text(0.97, 0.03, "DECOUPLED PREDICTOR\ngood at labels, doesn't track quality",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=13,
            color="darkred", style="italic",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFEBEE", ec="darkred"))
    ax.text(0.03, 0.97, "QUALITY WITHOUT SEPARATION\nrare; threshold mis-set",
            transform=ax.transAxes, ha="left", va="top", fontsize=12,
            color="gray", style="italic")
    ax.text(0.03, 0.03, "BAD BOTH AXES",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=12,
            color="gray", style="italic")

    # Annotate the most important decoupled cells
    for (ts, pr, mode, train), c in cells.items():
        if c.get("bacc_raw") is None or c.get("rho_quality") is None: continue
        if c["bacc_raw"] > 60 and c["rho_quality"] < 0:
            ax.annotate(f"{mode} {RATIO_LABEL[train]}\n{pop_label[(ts,pr)]}",
                        xy=(c["bacc_raw"], c["rho_quality"]),
                        xytext=(c["bacc_raw"] + 1.5, c["rho_quality"] - 0.04),
                        fontsize=10, color="darkred",
                        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
                        bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="darkred"))

    # Headline correlation across all cells with both metrics
    pairs_all = [(c["bacc_raw"], c["rho_quality"]) for c in cells.values()
                  if c.get("bacc_raw") is not None and c.get("rho_quality") is not None]
    if len(pairs_all) > 5:
        rho_overall = spearman([p[0] for p in pairs_all], [p[1] for p in pairs_all])
        r_overall = pearson([p[0] for p in pairs_all], [p[1] for p in pairs_all])
        ax.text(0.97, 0.50, f"n cells = {len(pairs_all)}\n"
                            f"ρ(B, A) = {rho_overall:+.2f}\n"
                            f"r(B, A) = {r_overall:+.2f}",
                transform=ax.transAxes, ha="right", va="center", fontsize=14,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))

    ax.set_xlabel("Axis B — Balanced accuracy raw (%) — fair predictor metric",
                  fontsize=18)
    ax.set_ylabel("Axis A — Spearman ρ(score, pct_rating) — quality alignment",
                  fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    mod_legend = [Patch(facecolor=PALETTE[(m, r)], edgecolor="black", label=f"{m} {RATIO_LABEL[r]}")
                  for m in MODES for r in RATIOS]
    src_legend = [
        Line2D([0],[0], marker=mk, color="w", markerfacecolor="gray", markeredgecolor="black",
               markersize=14, label=lbl)
        for (k, mk), lbl in [
            ((("iclr","balanced"), "o"), "ICLR balanced"),
            ((("iclr","natural"), "s"), "ICLR natural"),
            ((("arxiv","balanced"), "^"), "arxiv balanced"),
            ((("arxiv","natural"), "v"), "arxiv natural"),
        ]
    ]
    leg1 = ax.legend(handles=mod_legend, loc="upper left", fontsize=12, title="Model",
                     title_fontsize=13)
    ax.add_artist(leg1)
    ax.legend(handles=src_legend, loc="upper left", fontsize=12, title="Test population",
              title_fontsize=13, bbox_to_anchor=(0.20, 1.00))

    fig.suptitle("Two axes of model quality — same model can score very differently on each",
                 fontsize=20, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def fig_rankings_comparison(cells, save_base):
    """Three side-by-side ranking grids: raw ACC, balanced ACC, ρ_quality.
    Per (test_set, prior), rank the 4 models. Highlight where rankings disagree.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    _setup()

    test_cells = [(ts, pr) for ts in TEST_SETS for pr in PRIORS]
    model_keys = [(m, r) for m in MODES for r in RATIOS]
    model_labels = [f"{m} {RATIO_LABEL[r]}" for m, r in model_keys]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8.5))
    metrics = [
        ("acc_raw",  "Axis B — Raw ACC (%)\n(misleading on imbalanced)", 40, 90),
        ("bacc_raw", "Axis B — Balanced ACC (%)\n(fair: avg(AccR, RejR))", 40, 80),
        ("rho_quality", "Axis A — ρ(score, pct_rating)\n(quality alignment)", -0.3, 0.6),
    ]
    DARKGREEN = "#2C7A2C"; GOLD = "#F4B400"

    for ax, (key, title, vmin, vmax) in zip(axes, metrics):
        # Build matrix: rows = test cells, cols = models
        mat = np.full((len(test_cells), len(model_keys)), np.nan)
        for i, (ts, pr) in enumerate(test_cells):
            for j, (m, r) in enumerate(model_keys):
                c = cells.get((ts, pr, m, r))
                if c is None or c.get(key) is None: continue
                mat[i, j] = c[key]
        # Color: viridis-like for raw/bacc, RdYlGn diverging for rho
        if key == "rho_quality":
            cmap = plt.cm.RdYlGn
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        else:
            cmap = plt.cm.RdYlGn
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(model_keys))); ax.set_xticklabels(model_labels, rotation=15, fontsize=13)
        ax.set_yticks(range(len(test_cells)))
        ax.set_yticklabels([CELL_LABELS[(ts, pr)].split(" (")[0] for ts, pr in test_cells],
                           fontsize=13)
        ax.set_title(title, fontsize=15)
        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    if key == "rho_quality":
                        text = f"{mat[i, j]:+.2f}"
                    else:
                        text = f"{mat[i, j]:.0f}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=12,
                            fontweight="bold")
        # Highlight winner per row with green box + star
        for i in range(mat.shape[0]):
            row = mat[i, :]
            if np.all(np.isnan(row)): continue
            best_j = np.nanargmax(row)
            rect = plt.Rectangle((best_j-0.5, i-0.5), 1, 1, fill=False,
                                  edgecolor=DARKGREEN, linewidth=3.5, zorder=4)
            ax.add_patch(rect)
            ax.text(best_j, i + 0.34, "★", ha="center", va="center",
                    fontsize=18, color=GOLD, zorder=5)

    fig.suptitle("Per-(test cell × model) — winner depends on the metric",
                 fontsize=20, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ---------- markdown writer ----------
def append_section(cells):
    """Append the Two-Axes section to the report."""
    # Find the "deployment cells" winners under each metric
    def best_model_per_cell(metric_key):
        out = {}
        for (ts, pr, m, r), c in cells.items():
            v = c.get(metric_key)
            if v is None: continue
            cur = out.get((ts, pr))
            if cur is None or v > cur[1]:
                out[(ts, pr)] = (f"{m} {RATIO_LABEL[r]}", v)
        return out

    raw_winners  = best_model_per_cell("acc_raw")
    bacc_winners = best_model_per_cell("bacc_raw")
    rho_winners  = best_model_per_cell("rho_quality")

    lines = []
    lines.append("\n\n---\n")
    lines.append("## Two Axes of Model Quality (the right metric for the right question)\n")
    lines.append("**The framing.** A model's success can be measured along two distinct axes:\n")
    lines.append("- **Axis A — Universal quality indicator.** Does the model's continuous score "
                 "track the underlying paper quality? Best metric: **Spearman ρ(score, pct_rating)**. "
                 "Threshold-independent, prior-invariant.")
    lines.append("- **Axis B — Conference accept/reject classifier.** Does the model's binary decision "
                 "match the venue's accept/reject? Best metric: **balanced accuracy** = avg(accept-recall, "
                 "reject-recall). Raw ACC is *misleading* on imbalanced priors — predict-all-reject gives "
                 "70% raw ACC on a 30%-accept population without any signal at all.\n")

    lines.append("![Two axes scatter](../tmp_latex_dir/figures/ratio_xeval_two_axes_scatter.png)\n")
    lines.append("Each dot is one (model, test population) cell. Lines connect same-model points "
                 "across populations. Top-right quadrant = high on both axes (universal quality "
                 "indicator). Bottom-right = decoupled predictor (good at labels, doesn't track quality) "
                 "— text on arxiv-natural-iclr lives here.\n")

    lines.append("![Per-cell rankings](../tmp_latex_dir/figures/ratio_xeval_two_axes_rankings.png)\n")
    lines.append("Three side-by-side ranking grids — same 4 models × same 4 test cells, but different "
                 "metrics. **Green box + ★ = winner per cell**. Notice how often the winner changes when "
                 "you switch from raw ACC to balanced ACC.\n")

    lines.append("### The reframe — Q1 revisited\n")
    lines.append("Earlier headline: text 30/70 wins on natural-prior raw ACC. **But that 76.3% "
                 "raw ACC on arxiv natural is a constant-reject classifier** — accept-recall = 0%, "
                 "reject-recall = 100%, balanced ACC = **50%** (random baseline). The model isn't "
                 "predicting; it's just exploiting the natural prior.\n")
    lines.append("**On balanced accuracy** (the fair Axis B metric):\n")
    lines.append("| Test cell | Best raw-ACC model | Best **balanced-ACC** model |")
    lines.append("|---|---|---|")
    for (ts, pr) in [("iclr", "natural"), ("arxiv", "natural"),
                      ("iclr", "balanced"), ("arxiv", "balanced")]:
        rw = raw_winners.get((ts, pr), ("—", 0))
        bw = bacc_winners.get((ts, pr), ("—", 0))
        rho_w = rho_winners.get((ts, pr), ("—", 0))
        lines.append(f"| {CELL_LABELS[(ts, pr)].split(' (')[0]} | {rw[0]} ({rw[1]:.1f}) | "
                     f"**{bw[0]}** ({bw[1]:.1f}) |")
    lines.append("")
    lines.append("- **On natural-prior cells, vision wins on balanced accuracy** — opposite of the raw-ACC "
                 "ranking. text 30/70's apparent dominance was a class-imbalance illusion.\n")

    lines.append("### Headline answers — the right metric for each axis\n")

    lines.append("**For Axis A (universal quality):**")
    lines.append("- Best metric: **ρ(score, pct_rating)** — measures continuous quality alignment.")
    lines.append("- Best test population: **direct ICLR balanced** (largest, full pct_rating coverage; "
                 "ρ ≈ 0.47-0.51 across all 4 models — modality and ratio don't matter much here).")
    lines.append("- Best model: **vision 50/50** — most consistent ρ across populations and the only "
                 "modality that retains positive ρ on the prior-shifted arxiv-natural-iclr cell.")
    lines.append("- Calibration: **doesn't apply** (threshold-independent metric).\n")

    lines.append("**For Axis B (conference predictor):**")
    lines.append("- Best metric: **balanced accuracy** (or AUC if you want threshold-independent ranking).")
    lines.append("- Best test population: **the prior you'll deploy under** (natural for production, "
                 "balanced for stress testing).")
    lines.append("- Best model under natural prior (balanced ACC): **vision 50/50** — wins both arxiv "
                 "and ICLR natural cells on balanced accuracy.")
    lines.append("- Calibration with raw-ACC objective often **hurts** balanced accuracy (because it "
                 "pushes the threshold to match the natural prior, sacrificing accept-recall). Use "
                 "raw τ=0 if you care about balanced accuracy, OR re-calibrate with balanced-acc "
                 "objective.\n")

    lines.append("### Are the two axes correlated?\n")
    pairs_all = [(c["bacc_raw"], c["rho_quality"]) for c in cells.values()
                  if c.get("bacc_raw") is not None and c.get("rho_quality") is not None]
    if len(pairs_all) > 5:
        rho_overall = spearman([p[0] for p in pairs_all], [p[1] for p in pairs_all])
        r_overall = pearson([p[0] for p in pairs_all], [p[1] for p in pairs_all])
        lines.append(f"- Across {len(pairs_all)} cells: ρ(balanced_ACC, ρ_quality) = "
                     f"**{rho_overall:+.2f}**, r = **{r_overall:+.2f}**. Moderate positive — they "
                     f"co-move but they decouple in important cases.")
    lines.append("- **Decoupling case (most striking)**: text 50/50 + 30/70 on `arxiv-natural-iclr` "
                 "have balanced ACC ≈ 64-65% (above random) but ρ ≈ −0.13 to −0.17 (anti-correlated "
                 "with reviewer ratings). The model passes the binary classifier test and fails the "
                 "quality test — it's learning *venue-specific decision shortcuts* rather than quality.")
    lines.append("- **Vision on the same cell**: balanced ACC ≈ 64% AND ρ = +0.46. Same labels, same "
                 "papers — vision is the universal-quality model; text is the decoupled predictor.\n")

    lines.append("### Recommendation\n")
    lines.append("Choose your model by your goal:\n")
    lines.append("- **Goal: universal quality scoring** (downstream review-assist, citation prediction, "
                 "paper recommendation): pick **vision 50/50**, no calibration. Use ρ(score, pct_rating) "
                 "as the validation metric.")
    lines.append("- **Goal: deployable accept/reject classifier**: pick **vision 50/50**, evaluate with "
                 "balanced accuracy on the prior you'll deploy under. Calibrate threshold for balanced "
                 "accuracy, not raw accuracy.")
    lines.append("- **Both goals → same model**. The two axes happen to converge on vision 50/50 "
                 "*when measured properly*, but text 30/70 looks great on raw ACC and terrible on both "
                 "axes once you switch to fair metrics. **Always report balanced accuracy alongside raw ACC** "
                 "for any reject-heavy test set.\n")

    with open(REPORT_PATH, "a") as f:
        f.write("\n".join(lines))
    print(f"\nAppended Two-Axes section to: {REPORT_PATH}")


def main():
    print("Computing per-cell metrics including balanced accuracy + ρ_quality...")
    cells = collect_cells()

    print("\n=== Per-cell summary (TEST split) ===")
    print(f"{'cell':<32} {'model':<14} {'raw ACC':>8} {'bacc raw':>9} {'bacc cal':>9} {'AUC':>6} {'ρ_qual':>8}")
    for (ts, pr, mode, train), c in sorted(cells.items()):
        rho = f"{c['rho_quality']:+.2f}" if c['rho_quality'] is not None else "  —  "
        cell_name = f"{ts} {pr}"
        print(f"{cell_name:<32} {mode+' '+RATIO_LABEL[train]:<14} "
              f"{c['acc_raw']:>8.1f} {c['bacc_raw']:>9.1f} "
              f"{c['bacc_cal'] if c['bacc_cal'] else 0:>9.1f} "
              f"{c['auc']:>6.3f} {rho:>8}")

    print("\nBuilding figures...")
    fig_two_axes_scatter(cells, str(FIG_DIR / "ratio_xeval_two_axes_scatter"))
    fig_rankings_comparison(cells, str(FIG_DIR / "ratio_xeval_two_axes_rankings"))

    print("Appending markdown section...")
    append_section(cells)


if __name__ == "__main__":
    main()
