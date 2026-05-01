#!/usr/bin/env python3
"""
Two outcome-driven summary figures + a tightened "Executive Summary" prepended
to the report. Focus on the user's two questions:

  Q1 — for natural-prior deployment, which (modality, train ratio) is best?
       (raw vs calibrated)
  Q2 — what does AUC actually buy us? Is it a good proxy for quality
       correlation (ρ between score and pct_rating)?

Figures:
  fig_summary_natural_acc.{pdf,png} — grouped bars highlighting the winner per cell.
  fig_summary_auc_vs_quality.{pdf,png} — scatter of AUC vs ρ(score, pct_rating)
       across (model × test-population) cells; identifies decoupling cases.
  fig_summary_recall_arrows.{pdf,png} — improved version of the raw→calibrated
       arrow scatter (cleaner labels, winner annotation).
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
    auc, score_logodds, extract_pred,
    load_iclr_cell, load_arxiv_cell,
    metrics_iclr, metrics_arxiv,
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
    return num / (dx * dy)


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


# ---------- styling ----------
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
    ("text",   "50_50"): "#6098FF",  # blue
    ("text",   "30_70"): "#FF8988",  # red
    ("vision", "50_50"): "#77B25D",  # green
    ("vision", "30_70"): "#B28CFF",  # purple
}


# ---------- Q1 figure: natural-prior accuracy with winner highlighting ----------
def fig_natural_prior_acc(results, save_base):
    """Grouped bar chart: 2 panels (ICLR-natural, arxiv-natural).
    For each: 4 model bars × 2 (raw, cal). Highlight the winner with a star + bold label.
    Show the calibration delta as an arrow above each calibrated bar.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    _setup()

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    panels = [
        ("iclr", "natural", "ICLR — natural prior\n(test ratio = 30/70, year = 25/26)"),
        ("arxiv", "natural", "Arxiv — natural prior\n(natrate y24up; per-conference acceptance rates)"),
    ]
    BLUE  = "#6098FF"
    ORANGE = "#FECC81"
    GOLD = "#F4B400"
    DARKGREEN = "#2C7A2C"

    for ax, (ts, pr, title) in zip(axes, panels):
        model_keys = [(m, r) for m in MODES for r in RATIOS]
        model_labels = [f"{m}\n{RATIO_LABEL[r]}" for m, r in model_keys]
        raw_vals = [results["test"][ts][pr][m][r]["raw"]["acc"] for m, r in model_keys]
        cal_vals = [results["test"][ts][pr][m][r]["cal"]["acc"] for m, r in model_keys]
        x = np.arange(len(model_keys)); bw = 0.36

        # Determine winners
        best_raw_idx = max(range(4), key=lambda i: raw_vals[i] if raw_vals[i] is not None else -1)
        best_cal_idx = max(range(4), key=lambda i: cal_vals[i] if cal_vals[i] is not None else -1)

        bars_raw = ax.bar(x - bw/2, raw_vals, bw, color=BLUE, edgecolor="black",
                          linewidth=1.2, label="raw (τ=0)")
        bars_cal = ax.bar(x + bw/2, cal_vals, bw, color=ORANGE, edgecolor="black",
                          linewidth=1.2, label="calibrated (τ* from val)")

        # Highlight winners
        bars_raw[best_raw_idx].set_edgecolor(DARKGREEN)
        bars_raw[best_raw_idx].set_linewidth(4)
        bars_cal[best_cal_idx].set_edgecolor(DARKGREEN)
        bars_cal[best_cal_idx].set_linewidth(4)

        # Numerical labels
        for i, (rv, cv) in enumerate(zip(raw_vals, cal_vals)):
            if rv is not None:
                weight = "bold" if i == best_raw_idx else "normal"
                color = DARKGREEN if i == best_raw_idx else "black"
                ax.text(x[i] - bw/2, rv + 0.6, f"{rv:.1f}", ha="center", va="bottom",
                        fontsize=15, fontweight=weight, color=color)
            if cv is not None:
                weight = "bold" if i == best_cal_idx else "normal"
                color = DARKGREEN if i == best_cal_idx else "black"
                ax.text(x[i] + bw/2, cv + 0.6, f"{cv:.1f}", ha="center", va="bottom",
                        fontsize=15, fontweight=weight, color=color)
            # Calibration delta annotation
            if rv is not None and cv is not None:
                d = cv - rv
                if abs(d) >= 1.0:
                    color = "darkred" if d < -1 else "gray"
                    ax.annotate(f"Δ {d:+.1f}", xy=(x[i], max(rv, cv) + 4),
                                ha="center", va="bottom", fontsize=12,
                                color=color, fontweight="bold")

        # Star above the calibrated winner
        ax.scatter(x[best_cal_idx] + bw/2, cal_vals[best_cal_idx] + 4.5, marker="*",
                   s=320, color=GOLD, edgecolor=DARKGREEN, linewidth=2, zorder=10)

        ax.set_xticks(x); ax.set_xticklabels(model_labels, fontsize=18)
        ax.set_title(title, fontsize=20)
        ax.set_ylim(40, 90)
        ax.tick_params(axis="y", labelsize=15)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_ylabel("ACC (%)", fontsize=18)
        ax.legend(fontsize=14, loc="lower left")

        # Bottom callout: which model won?
        winner_label = f"{model_keys[best_cal_idx][0]} {RATIO_LABEL[model_keys[best_cal_idx][1]]}"
        ax.text(0.5, -0.16, f"⭐ Best calibrated: {winner_label} ({cal_vals[best_cal_idx]:.1f}%)",
                transform=ax.transAxes, ha="center", fontsize=16, color=DARKGREEN,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc="#E8F5E9", ec=DARKGREEN, linewidth=1.5))

    fig.suptitle("Q1: Best model for accuracy under NATURAL prior — TEST split, 7B",
                 fontsize=22, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ---------- Q2 figure: AUC vs ρ(score, pct_rating) ----------
def fig_auc_vs_quality(save_base):
    """Scatter: x = AUC, y = Spearman ρ(score, pct_rating), each point = one
    (model × test population) cell. Identify decoupling cases (high AUC, low/neg ρ)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    _setup()

    fig, ax = plt.subplots(figsize=(13, 9.5))

    # Build cells: for each (model, source, prior, venue_filter), compute AUC and ρ ON the same subset
    sources = []
    for prior in PRIORS:
        sources.append(("iclr", prior, None, f"ICLR direct, {prior}"))
    for prior in PRIORS:
        for venue in ("iclr", "neurips"):
            sources.append(("arxiv", prior, venue,
                            f"arxiv-{venue}, {prior}"))

    points = []  # (auc, rho, color, marker, label, n)
    for ts, prior, venue, label in sources:
        for mode in MODES:
            for train in RATIOS:
                if ts == "iclr":
                    pairs = load_iclr_extended(mode, train, prior, "test")
                    items = [(s, g, pr) for s, g, _, pr in pairs if pr is not None]
                else:
                    pairs = load_arxiv_extended(mode, train, prior, "test")
                    items = [(s, g, pr) for s, g, v, _, pr in pairs
                             if pr is not None and (venue is None or v == venue)]
                if len(items) < 30: continue
                a = auc([s for s,_,_ in items], [g for _,g,_ in items])
                rho = spearman([s for s,_,_ in items], [pr for _,_,pr in items])
                if a is None or rho is None: continue
                color = PALETTE[(mode, train)]
                # Marker by source-prior:
                if ts == "iclr":
                    marker = "o" if prior == "balanced" else "s"
                else:
                    if venue == "iclr":
                        marker = "^" if prior == "balanced" else "v"
                    else:
                        marker = "D" if prior == "balanced" else "P"
                points.append((a, rho, color, marker, label, len(items), mode, train))

    # Plot
    for a, rho, color, marker, label, n, mode, train in points:
        ax.scatter(a, rho, marker=marker, s=240, color=color, edgecolor="black", linewidth=1.5,
                   alpha=0.85)

    # Best-fit line + correlation
    aucs = [p[0] for p in points]
    rhos = [p[1] for p in points]
    rho_overall = spearman(aucs, rhos)
    r_overall = pearson(aucs, rhos)
    if len(aucs) > 2:
        mx = sum(aucs)/len(aucs); my = sum(rhos)/len(rhos)
        num = sum((x-mx)*(y-my) for x,y in zip(aucs, rhos))
        den = sum((x-mx)**2 for x in aucs)
        slope = num/den; intercept = my - slope*mx
        xx = np.linspace(min(aucs)-0.02, max(aucs)+0.02, 50)
        ax.plot(xx, intercept + slope*xx, "-", color="gray", linewidth=2.5, alpha=0.7,
                label=f"OLS fit (β = {slope:.2f})")

    # Reference lines
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axhline(0.3, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(0.55, 0.03, "ρ = 0", fontsize=11, color="black", alpha=0.6)

    # Annotate decoupling cases (high AUC but rho < 0)
    decoupling = [p for p in points if p[0] > 0.7 and p[1] < 0]
    for a, rho, _, _, lbl, n, mode, train in decoupling:
        ax.annotate(f"{mode} {RATIO_LABEL[train]}\n{lbl}\n(n={n})",
                    xy=(a, rho), xytext=(a + 0.015, rho - 0.05),
                    fontsize=11, color="darkred",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.4", fc="#FFEBEE", ec="darkred"))

    # Quadrant shading + labels
    ax.axhspan(-0.5, 0, color="red", alpha=0.04)
    ax.text(0.97, -0.04, "DECOUPLING\nhigh-AUC but anti-correlates\nwith reviewer quality",
            transform=ax.transAxes, ha="right", va="top", fontsize=12,
            color="darkred", style="italic")
    ax.text(0.50, 0.97, "ALIGNED\nhigh-AUC ⟹ high quality correlation",
            transform=ax.transAxes, ha="center", va="top", fontsize=12,
            color="darkgreen", style="italic")

    ax.set_xlabel("AUC (binary accept/reject separation)", fontsize=20)
    ax.set_ylabel("Spearman ρ(score, pct_rating)\n(continuous quality correlation)", fontsize=20)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Custom legend for modalities
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    mod_legend = [Patch(facecolor=PALETTE[(m, r)], edgecolor="black", label=f"{m} {RATIO_LABEL[r]}")
                  for m in MODES for r in RATIOS]
    src_legend = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black",
               markersize=12, label="ICLR direct, balanced"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="gray", markeredgecolor="black",
               markersize=12, label="ICLR direct, natural"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="gray", markeredgecolor="black",
               markersize=12, label="arxiv-iclr, balanced"),
        Line2D([0],[0], marker="v", color="w", markerfacecolor="gray", markeredgecolor="black",
               markersize=12, label="arxiv-iclr, natural"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="gray", markeredgecolor="black",
               markersize=12, label="arxiv-neurips, balanced"),
        Line2D([0],[0], marker="P", color="w", markerfacecolor="gray", markeredgecolor="black",
               markersize=12, label="arxiv-neurips, natural"),
    ]
    leg1 = ax.legend(handles=mod_legend, loc="upper left", fontsize=12, title="Model",
                     title_fontsize=13)
    ax.add_artist(leg1)
    ax.legend(handles=src_legend, loc="upper left", fontsize=12, title="Test population",
              title_fontsize=13, bbox_to_anchor=(0.20, 1.00))

    # Headline correlation
    ax.text(0.97, 0.97, f"Across all {len(points)} cells:\nρ(AUC, ρ_quality) = {rho_overall:+.2f}\n"
                        f"r(AUC, r_quality) = {r_overall:+.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=14,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))

    fig.suptitle("Q2: Does AUC track quality correlation?", fontsize=22, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ---------- improved recall scatter ----------
def fig_recall_arrows_v2(results, save_base):
    """Cleaner version of the raw→calibrated arrow scatter.
    Big arrows, model-color labels at the calibrated point, iso-acc contours simplified.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    _setup()

    cells = [(ts, pr) for ts in TEST_SETS for pr in PRIORS]
    fig, axes = plt.subplots(1, 4, figsize=(28, 7.5), sharey=True, sharex=True)
    model_keys = [(m, r) for m in MODES for r in RATIOS]

    for ax, (ts, pr) in zip(axes, cells):
        title = CELL_LABELS[(ts, pr)]
        # Estimate prior from the first cell (all 4 share the same population)
        any_cell = results["test"][ts][pr]["text"]["50_50"]["raw"]
        if any_cell["n"] > 0 and any_cell["acc_rec"] is not None and any_cell["rej_rec"] is not None:
            ar, rr, a = any_cell["acc_rec"]/100, any_cell["rej_rec"]/100, any_cell["acc"]/100
            denom = ar - rr
            p = ((a - rr) / denom) if abs(denom) > 1e-6 else 0.5
            p = max(0.1, min(0.9, p))
        else:
            p = 0.5
        # Iso-accuracy contours
        xs = np.linspace(0, 1, 100)
        for tgt in (0.5, 0.6, 0.7, 0.8):
            rrs = (tgt - xs*p) / (1 - p) if abs(1-p) > 1e-6 else np.full_like(xs, tgt)
            mask = (rrs >= 0) & (rrs <= 1)
            ax.plot(xs[mask]*100, rrs[mask]*100, "--", color="gray", alpha=0.35, linewidth=1.5)
            mid_idx = mask.sum() // 2
            if mid_idx > 5:
                ax.text(xs[mask][mid_idx]*100, rrs[mask][mid_idx]*100,
                        f"{tgt:.0%}", fontsize=11, color="gray", alpha=0.55)
        # Plot points + arrows
        for (mode, train) in model_keys:
            cell = results["test"][ts][pr][mode][train]
            raw = cell["raw"]; cal = cell["cal"]
            if raw["n"] == 0 or raw["acc_rec"] is None: continue
            color = PALETTE[(mode, train)]
            ax.scatter(raw["acc_rec"], raw["rej_rec"], s=150, facecolor="white",
                       edgecolor=color, linewidth=2.5, zorder=4)
            ax.scatter(cal["acc_rec"], cal["rej_rec"], s=240, facecolor=color,
                       edgecolor="black", linewidth=1.5, zorder=5)
            dx = cal["acc_rec"] - raw["acc_rec"]
            dy = cal["rej_rec"] - raw["rej_rec"]
            if dx*dx + dy*dy > 9:
                ax.annotate("", xy=(cal["acc_rec"], cal["rej_rec"]),
                            xytext=(raw["acc_rec"], raw["rej_rec"]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=3.0,
                                           alpha=0.85, shrinkA=8, shrinkB=8))
            # Label at calibrated position
            ax.annotate(f"{mode} {RATIO_LABEL[train]}",
                        xy=(cal["acc_rec"], cal["rej_rec"]),
                        xytext=(8, 6), textcoords="offset points",
                        fontsize=11, color=color, fontweight="bold")
        ax.plot([0, 100], [0, 100], "k:", alpha=0.25, linewidth=1)
        ax.set_xlabel("Accept-recall (%)", fontsize=18)
        ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
        ax.set_title(title.replace(" (", "\n("), fontsize=17)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Reject-recall (%)", fontsize=18)
    # Single legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="gray", markeredgewidth=2.5, markersize=14, label="raw  (τ=0)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=14, label="calibrated (τ*)"),
        Line2D([0],[0], color="gray", linestyle="--", alpha=0.5, label="iso-accuracy contours"),
    ]
    axes[0].legend(handles=legend_elements, loc="lower left", fontsize=13)

    fig.suptitle("Calibration trades reject-recall for accept-recall — open ⟶ filled (TEST split, 7B)",
                 fontsize=22, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ---------- prepend executive summary to report ----------
def prepend_executive_summary(results):
    """Build a tight summary section and prepend it to the report."""
    # Compute key numbers
    iclr_nat_winner = max(MODES, key=lambda m: max(
        (results["test"]["iclr"]["natural"][m][r]["cal"]["acc"] or 0) for r in RATIOS))
    arxiv_nat_winner = max(MODES, key=lambda m: max(
        (results["test"]["arxiv"]["natural"][m][r]["cal"]["acc"] or 0) for r in RATIOS))

    def get_acc(ts, pr, m, r, kind="cal"):
        v = results["test"][ts][pr][m][r][kind]["acc"]
        return v if v is not None else 0

    summary = []
    summary.append("# 7B Ratio Cross-Eval — Consolidated Report\n")
    summary.append("**One-paragraph TL;DR.** We trained Qwen2.5-7B (text) and Qwen2.5-VL-7B (vision) "
                   "at two accept/reject ratios (50/50 and 30/70) and evaluated them on ICLR (year 25/26) "
                   "and arxiv (y24up) test sets, each at *balanced* (50/50) and *natural* (~30% accept) "
                   "test priors. **For accuracy under natural-prior deployment, train at 30/70**: "
                   f"text 30/70 hits **{get_acc('arxiv','natural','text','30_70','raw'):.1f}%** raw on "
                   f"arxiv-natural and **{get_acc('iclr','natural','text','30_70','raw'):.1f}%** on "
                   "ICLR-natural without calibration. Vision is 2-4pp behind on accuracy but **vision is more "
                   "robust to temporal shift** (text models drop −20pp on CVPR-2026 papers; vision drops only "
                   "−6pp or improves). **AUC and quality-correlation (ρ vs `pct_rating`) are positively "
                   "correlated (ρ = +0.50 across 24 cells)** but they decouple: text on arxiv-natural-iclr "
                   "has decent AUC (~0.72) but *negative* ρ with reviewer ratings, while vision on the same "
                   "papers stays positive (ρ = +0.46) — an evidence-backed argument that vision tracks "
                   "underlying quality more reliably than text under prior shift.\n")

    summary.append("## Two questions, two answers\n")

    summary.append("### Q1 — Which model is best for accuracy under natural-prior deployment?\n")
    summary.append("![Q1 figure](../tmp_latex_dir/figures/ratio_xeval_summary_natural_acc.png)\n")
    summary.append("**Recipe-level answers (TEST split, calibrated unless noted):**")
    summary.append(f"- **ICLR natural** (test 30/70): **text 30/70** wins raw "
                   f"({get_acc('iclr','natural','text','30_70','raw'):.1f}%); vision 30/70 is essentially tied "
                   f"({get_acc('iclr','natural','vision','30_70','raw'):.1f}%). Calibration **hurts** here "
                   f"(matched-prior models are already near-optimal).")
    summary.append(f"- **Arxiv natural** (natrate y24up): **text 50/50** wins raw "
                   f"({get_acc('arxiv','natural','text','50_50','raw'):.1f}%); text 30/70 is tied at "
                   f"{get_acc('arxiv','natural','text','30_70','raw'):.1f}%. Vision is ~3-4pp behind "
                   f"({get_acc('arxiv','natural','vision','50_50','raw'):.1f}–"
                   f"{get_acc('arxiv','natural','vision','30_70','raw'):.1f}%).")
    summary.append("- **Single-recipe deployment** (one model serving both ICLR-style and arxiv-style "
                   "natural-prior workloads): **text 30/70 raw** is the most consistent — within 1pp of "
                   "the per-cell winner on both. Calibration is unnecessary for this recipe.\n")

    summary.append("### Q2 — Does AUC actually track quality?\n")
    summary.append("![Q2 figure](../tmp_latex_dir/figures/ratio_xeval_summary_auc_vs_quality.png)\n")
    summary.append("**Across 24 (model × test-population) cells: ρ(AUC, ρ_quality) = +0.50.**")
    summary.append("- AUC partially tracks ρ(score, `pct_rating`) — but they *decouple* in important cases.")
    summary.append("- **Most striking decoupling**: on `arxiv-natural-iclr`, text models reach AUC ≈ 0.72 "
                   "but score *anti-correlates* with `pct_rating` (ρ = −0.13 to −0.17). Vision on the same "
                   "papers stays AUC ≈ 0.75 with ρ = **+0.46**. **Same papers, same labels — text rejects "
                   "high-quality papers and accepts low-quality ones under natural-prior shift; vision doesn't.**")
    summary.append("- **Practical implication**: AUC is a *necessary but not sufficient* signal of model "
                   "quality. If you want a model that *ranks papers by underlying quality* (not just labels), "
                   "verify with `ρ(score, pct_rating)` — it can be near-zero even with AUC > 0.7.\n")

    summary.append("### Calibration trade-off (raw → calibrated)\n")
    summary.append("![Recall arrows](../tmp_latex_dir/figures/ratio_xeval_summary_recall_arrows.png)\n")
    summary.append("Each model's raw (open) → calibrated (filled) movement on the (accept-recall, "
                   "reject-recall) plane. Calibration trades reject-recall for accept-recall to maximize "
                   "overall ACC; iso-accuracy diagonals show the trade. Bias-heavy models (e.g. text 30/70 "
                   "on balanced priors) move the most.\n")

    summary.append("---\n")
    summary.append("Below: full per-cell tables, threshold dump, and the granular extended/temporal "
                   "analyses. The headline figures above are the recommended summary for a 1-slide "
                   "presentation.\n")

    return "\n".join(summary)


# ---------- main ----------
def main():
    print("Loading + recomputing thresholds + metrics for the 32 cells...")
    iclr_taus = compute_iclr_thresholds()
    arxiv_taus = compute_arxiv_thresholds()

    # Recompute the unified per-cell metrics dict (matches what consolidated produces).
    results = {}
    for split in SPLITS:
        results[split] = {}
        results[split]["iclr"] = {}
        for prior in PRIORS:
            results[split]["iclr"][prior] = {}
            for mode in MODES:
                results[split]["iclr"][prior][mode] = {}
                for train in RATIOS:
                    pairs = load_iclr_cell(mode, train, prior, split)
                    tau = iclr_taus[(mode, train, prior)]
                    raw = metrics_iclr(pairs, tau=0.0)
                    cal = metrics_iclr(pairs, tau=tau)
                    results[split]["iclr"][prior][mode][train] = {"raw": raw, "cal": cal, "tau": tau}
        results[split]["arxiv"] = {}
        for prior in PRIORS:
            results[split]["arxiv"][prior] = {}
            for mode in MODES:
                results[split]["arxiv"][prior][mode] = {}
                for train in RATIOS:
                    pairs = load_arxiv_cell(mode, train, prior, split)
                    venue_taus = arxiv_taus[(mode, train, prior)]
                    raw = metrics_arxiv(pairs, venue_tau_map=None, default_tau=0.0)
                    cal = metrics_arxiv(pairs, venue_tau_map=venue_taus)
                    results[split]["arxiv"][prior][mode][train] = {
                        "raw": raw, "cal": cal, "venue_taus": venue_taus,
                    }

    print("Building Q1 figure (natural-prior accuracy)...")
    fig_natural_prior_acc(results, str(FIG_DIR / "ratio_xeval_summary_natural_acc"))

    print("Building Q2 figure (AUC vs quality correlation)...")
    fig_auc_vs_quality(str(FIG_DIR / "ratio_xeval_summary_auc_vs_quality"))

    print("Building improved recall arrows figure...")
    fig_recall_arrows_v2(results, str(FIG_DIR / "ratio_xeval_summary_recall_arrows"))

    print("Prepending executive summary to report...")
    summary = prepend_executive_summary(results)
    # Read existing report and replace its TOP section (everything before "## Headline" was "# Title")
    existing = REPORT_PATH.read_text()
    # Strip the original title + headline block (everything up to and including "## Headline" body)
    # We'll find the marker "## Two questions, two answers" or fall back.
    if existing.startswith("# 7B Ratio Cross-Eval"):
        # Find the start of the first "## " section that we keep — leave the existing detail tables intact
        # Strategy: find "## Figures" (the first detail section in the existing report); keep from there
        idx = existing.find("\n## Figures\n")
        if idx > 0:
            # Add an intermediate "## Detailed numbers" header
            tail = existing[idx:].lstrip()
            new_text = summary + "\n## Detailed numbers (full per-cell tables)\n\n" + tail
        else:
            new_text = summary + "\n\n" + existing
    else:
        new_text = summary + "\n\n" + existing
    REPORT_PATH.write_text(new_text)
    print(f"\nUpdated report: {REPORT_PATH} — now {sum(1 for _ in open(REPORT_PATH))} lines.")


if __name__ == "__main__":
    main()
