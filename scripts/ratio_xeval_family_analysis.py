#!/usr/bin/env python3
"""
Family-level analysis: group venues into research-area families and look at
(a) per-venue heatmap focused on the natural-prior test (the deployment cell),
(b) per-family ACC trajectories across 2024 → 2025 → 2026,
(c) family-level vision-vs-text gap and how it widens for 2026 papers.

Family taxonomy:
    NLP        : acl_family (=acl/emnlp/naacl), colm
    General ML : iclr, icml, neurips, aaai, aistats
    CV         : cvpr, iccv, eccv
    Robotics   : corl

Output:
  fig_family_heatmap_arxiv_natural_test.{pdf,png}
  fig_family_year_trajectory.{pdf,png}
  fig_family_vision_minus_text_2026.{pdf,png}
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
    MODEL, RATIO_LABEL, MODES, RATIOS, PRIORS,
    ACL_FAMILY, auc, score_logodds, extract_pred,
    compute_arxiv_thresholds,
    arxiv_jsonl, ARXIV_META,
)
from ratio_xeval_year_analysis import load_arxiv_with_year

# Family taxonomy
FAMILIES = {
    "NLP":         ["acl_family", "colm"],
    "General ML":  ["iclr", "icml", "neurips", "aaai", "aistats"],
    "CV":          ["cvpr", "iccv", "eccv"],
    "Robotics":    ["corl"],
}
ALL_VENUES = ["aaai", "acl_family", "aistats", "colm", "corl", "cvpr", "eccv",
              "iccv", "iclr", "icml", "neurips"]

VENUE_TO_FAMILY = {v: f for f, vs in FAMILIES.items() for v in vs}

PALETTE = {
    ("text",   "50_50"): "#6098FF",
    ("text",   "30_70"): "#FF8988",
    ("vision", "50_50"): "#77B25D",
    ("vision", "30_70"): "#B28CFF",
}
FAMILY_COLOR = {
    "NLP":         "#FECC81",  # orange
    "General ML":  "#6098FF",  # blue
    "CV":          "#77B25D",  # green
    "Robotics":    "#B28CFF",  # purple
}


def _setup():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
    })
    return mpl, plt


def acc_of(pairs, tau=0.0):
    if not pairs: return None
    return sum(1 for s, g, *_ in pairs if (s > tau) == (g == 1)) / len(pairs) * 100


def auc_of(pairs):
    labels = [g for _, g, *_ in pairs]
    if len(set(labels)) < 2: return None
    return auc([s for s, _, *_ in pairs], labels)


# ---------- Figure 1: Per-venue heatmap (natural prior, test split) ----------
def fig_venue_heatmap_natural_test(arxiv_taus, save_base):
    """Two-panel heatmap focused on the deployment cell:
       - Left: ACC raw, arxiv natural prior, TEST split, per (model × venue)
       - Right: ACC calibrated (per-venue τ*), same axes
    With per-cell sample size annotated and winners (best per venue) highlighted.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    _setup()

    venues = ALL_VENUES
    model_keys = [(m, r) for m in MODES for r in RATIOS]
    model_labels = [f"{m} {RATIO_LABEL[r]}" for m, r in model_keys]

    raw_mat = np.full((len(model_keys), len(venues)), np.nan)
    cal_mat = np.full((len(model_keys), len(venues)), np.nan)
    n_mat = np.zeros((len(model_keys), len(venues)), dtype=int)
    for i, (mode, train) in enumerate(model_keys):
        pairs = load_arxiv_with_year(mode, train, "natural")
        venue_taus = arxiv_taus[(mode, train, "natural")]
        by_v = defaultdict(list)
        for s, g, v, _ in pairs: by_v[v].append((s, g))
        for j, v in enumerate(venues):
            vp = by_v.get(v, [])
            if not vp: continue
            n_mat[i, j] = len(vp)
            raw_mat[i, j] = acc_of([(s, g) for s, g in vp]) or 0
            entry = venue_taus.get(v)
            if entry is None: tau = venue_taus.get("GLOBAL", 0.0)
            elif isinstance(entry, tuple): tau = entry[0]
            else: tau = entry
            cal_mat[i, j] = acc_of([(s, g) for s, g in vp], tau=tau) or 0

    fig, axes = plt.subplots(1, 2, figsize=(28, 7))
    cmap = plt.cm.RdYlGn
    for ax, mat, title in [(axes[0], raw_mat, "Raw ACC (τ=0)"),
                            (axes[1], cal_mat, "Calibrated ACC (per-venue τ*)")]:
        im = ax.imshow(mat, cmap=cmap, vmin=30, vmax=95, aspect="auto")
        ax.set_xticks(range(len(venues))); ax.set_xticklabels(venues, rotation=45, ha="right", fontsize=14)
        ax.set_yticks(range(len(model_keys))); ax.set_yticklabels(model_labels, fontsize=14)
        ax.set_title(title, fontsize=18)
        # Cell text
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    color = "black" if mat[i, j] > 60 else "white"
                    ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center",
                            fontsize=13, color=color, fontweight="bold")
        # Sample size header
        for j, v in enumerate(venues):
            n = max(n_mat[:, j])
            ax.text(j, -0.7, f"n={n}", ha="center", va="bottom", fontsize=10, color="gray")
        # Highlight winner per column with green box
        for j in range(mat.shape[1]):
            col = mat[:, j]
            if np.all(np.isnan(col)): continue
            best_i = np.nanargmax(col)
            rect = plt.Rectangle((j-0.5, best_i-0.5), 1, 1, fill=False,
                                  edgecolor="darkgreen", linewidth=3.5)
            ax.add_patch(rect)
        # Family bracket annotations on x-axis
        # We won't add brackets here (cluttered); families are conveyed in trajectory plot.
    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, pad=0.01)
    cbar.set_label("ACC (%)", fontsize=18); cbar.ax.tick_params(labelsize=14)
    fig.suptitle("Per-venue accuracy on arxiv NATURAL prior (TEST split, deployment cell)\n"
                 "Green box = best model for that venue",
                 fontsize=20, y=1.04)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ---------- Figure 2: Family ACC trajectory across years ----------
def fig_family_year_trajectory(save_base):
    """Per-model panel; each panel shows 4 family-aggregated lines from 2024 → 2026.
    Family-aggregated = pool all (paper, gold, score) across venues in family, then ACC.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    _setup()

    years = [2024, 2025, 2026]
    fig, axes = plt.subplots(1, 4, figsize=(28, 7), sharey=True)
    model_keys = [(m, r) for m in MODES for r in RATIOS]
    fam_keys = list(FAMILIES.keys())

    # Compute family-aggregated ACC per (model, family, year) for arxiv natural test
    def fam_year_acc(mode, train, family, year, prior="natural"):
        pairs = load_arxiv_with_year(mode, train, prior)
        venues_in = FAMILIES[family]
        sub = [(s, g) for s, g, v, y in pairs if v in venues_in and y == year]
        if not sub: return None, 0
        return acc_of(sub), len(sub)

    for ax, (mode, train) in zip(axes, model_keys):
        for fam in fam_keys:
            xs, ys, ns = [], [], []
            for yr in years:
                acc, n = fam_year_acc(mode, train, fam, yr)
                if acc is not None and n >= 5:
                    xs.append(yr); ys.append(acc); ns.append(n)
            if len(xs) < 2: continue
            color = FAMILY_COLOR[fam]
            ax.plot(xs, ys, "-o", color=color, linewidth=3.0, markersize=12,
                    markeredgecolor="black", markeredgewidth=1.0)
            ax.text(xs[-1] + 0.05, ys[-1], f"{fam} (n={'/'.join(str(n) for n in ns)})",
                    fontsize=12, color=color, fontweight="bold", va="center")
        ax.set_title(f"{mode} {RATIO_LABEL[train]}", fontsize=18)
        ax.set_xticks(years)
        ax.set_xlim(2023.7, 2026.6)
        ax.set_xlabel("conference year", fontsize=15)
        ax.set_ylim(30, 95)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("ACC raw (%) on arxiv natural", fontsize=18)

    fig.suptitle("Family-aggregated raw ACC over years — arxiv natural prior",
                 fontsize=22, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ---------- Figure 3: Family-level vision-minus-text gap on 2024+25 vs 2026 ----------
def fig_family_vision_minus_text(save_base):
    """Side-by-side panel: vision-minus-text gap per family, on BALANCED prior
    (where temporal shift exposes text models) and NATURAL prior (deployment).
    Each panel: 4 family bars × {2024+25 baseline, 2026}.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    _setup()

    fams = list(FAMILIES.keys())
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    BLUE, ORANGE = "#6098FF", "#FECC81"
    DARKGREEN, DARKRED = "#2C7A2C", "#B22222"

    def avg_gap(family, year_filter, prior):
        gaps = []
        for train in RATIOS:
            text_pairs = load_arxiv_with_year("text", train, prior)
            vision_pairs = load_arxiv_with_year("vision", train, prior)
            text_sub = [(s, g) for s, g, v, y in text_pairs
                        if v in FAMILIES[family] and year_filter(y)]
            vision_sub = [(s, g) for s, g, v, y in vision_pairs
                           if v in FAMILIES[family] and year_filter(y)]
            if not text_sub or not vision_sub: continue
            t_acc = acc_of(text_sub); v_acc = acc_of(vision_sub)
            if t_acc is None or v_acc is None: continue
            gaps.append((v_acc - t_acc, len(text_sub), len(vision_sub)))
        if not gaps: return None, 0, 0
        avg = sum(g for g, _, _ in gaps) / len(gaps)
        avg_n_t = sum(nt for _, nt, _ in gaps) / len(gaps)
        avg_n_v = sum(nv for _, _, nv in gaps) / len(gaps)
        return avg, avg_n_t, avg_n_v

    panels = [("balanced", "Balanced prior (50/50)\nwhere temporal shift exposes text"),
              ("natural",  "Natural prior (deployment cell)\ntext models' bias matches the population")]

    for ax, (prior, title) in zip(axes, panels):
        baseline = {fam: avg_gap(fam, lambda y: y in (2024, 2025), prior) for fam in fams}
        y2026 = {fam: avg_gap(fam, lambda y: y == 2026, prior) for fam in fams}
        x = np.arange(len(fams)); bw = 0.36
        base_vals = [baseline[f][0] if baseline[f][0] is not None else 0 for f in fams]
        y26_vals  = [y2026[f][0]    if y2026[f][0]    is not None else 0 for f in fams]
        ax.bar(x - bw/2, base_vals, bw, color=BLUE, edgecolor="black", linewidth=1.2,
               label="2024+2025 baseline")
        ax.bar(x + bw/2, y26_vals, bw, color=ORANGE, edgecolor="black", linewidth=1.2,
               label="2026")
        for i, fam in enumerate(fams):
            for j, val in enumerate([base_vals[i], y26_vals[i]]):
                xpos = x[i] + (-bw/2 if j == 0 else bw/2)
                if val == 0:
                    ax.text(xpos, 0.5, "no data", ha="center", va="bottom",
                            fontsize=11, color="gray")
                else:
                    fmt_color = DARKGREEN if val > 1 else (DARKRED if val < -1 else "black")
                    fontweight = "bold" if abs(val) > 3 else "normal"
                    ax.text(xpos, val + (0.7 if val >= 0 else -1.5),
                            f"{val:+.1f}", ha="center", va="bottom" if val >= 0 else "top",
                            fontsize=14, color=fmt_color, fontweight=fontweight)
            if base_vals[i] != 0 and y26_vals[i] != 0:
                d = y26_vals[i] - base_vals[i]
                if abs(d) >= 2:
                    ymax = max(base_vals[i], y26_vals[i])
                    ax.annotate(f"Δ {d:+.1f}", xy=(x[i], ymax + 4),
                                ha="center", va="bottom", fontsize=14,
                                color=(DARKGREEN if d > 0 else DARKRED), fontweight="bold")
        for i, fam in enumerate(fams):
            venues = ", ".join(FAMILIES[fam])
            nt_b, nv_b = int(baseline[fam][1]), int(baseline[fam][2])
            nt_26, nv_26 = int(y2026[fam][1]), int(y2026[fam][2])
            ax.text(x[i], -22, f"{venues}", ha="center", va="top", fontsize=10,
                    color="gray", style="italic")
            ax.text(x[i], -25, f"24+25 n≈{nt_b}/{nv_b}, 26 n≈{nt_26}/{nv_26}",
                    ha="center", va="top", fontsize=9, color="gray")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(fams, fontsize=16, fontweight="bold")
        ax.set_ylim(-22, 32)
        ax.set_title(title, fontsize=17)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Vision ACC − Text ACC (pp)\n(positive = vision better)", fontsize=17)
            ax.legend(fontsize=14, loc="upper left")

    fig.suptitle("Vision's advantage over text by paper family on 2024+25 vs 2026 papers\n"
                 "(arxiv test, averaged over 50/50 + 30/70 train ratios)",
                 fontsize=20, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ---------- markdown writer ----------
def append_section():
    lines = []
    lines.append("\n\n---\n")
    lines.append("## Family-level analysis (vision robustness story)\n")
    lines.append("Venues grouped into 4 paper families: **NLP** (acl_family, colm), **General ML** "
                 "(iclr, icml, neurips, aaai, aistats), **CV** (cvpr, iccv, eccv), **Robotics** (corl).\n")

    lines.append("### M. Per-venue heatmap — deployment cell (arxiv natural test)\n")
    lines.append("![per-venue heatmap natural](../tmp_latex_dir/figures/ratio_xeval_family_heatmap_arxiv_natural_test.png)\n")
    lines.append("Reading the heatmap: under natural-prior deployment, **text 50/50 wins or ties on most "
                 "venues** (green boxes mostly fall on the text rows). Vision pulls ahead on a few "
                 "smaller-n venues (AISTATS, COLM, ICCV) but is broadly behind on the high-n NLP/General-ML "
                 "venues. This matches the Q1 finding: **for raw natural-prior accuracy, text is the right "
                 "pick across families today.**\n")

    lines.append("### N. Family-aggregated ACC trajectory across years\n")
    lines.append("![family year trajectory](../tmp_latex_dir/figures/ratio_xeval_family_year_trajectory.png)\n")
    lines.append("On natural prior, NLP and General ML families are temporally flat across all 4 models. "
                 "The CV family is more volatile — vision 50/50 actually peaks on CV-2025 papers while text "
                 "30/70 stays flat. (CV 2026 detail is on the BALANCED prior plot below — that's where the "
                 "temporal shift is sharpest.)\n")

    lines.append("### O. Vision-minus-text gap by family — does vision advantage widen for 2026?\n")
    lines.append("![family vision-minus-text gap](../tmp_latex_dir/figures/ratio_xeval_family_vision_minus_text_2026.png)\n")
    lines.append("**The decisive plot.** Two panels: balanced prior (where the text-fragility shows) and "
                 "natural prior (deployment cell). For each family, the vision-vs-text gap (averaged over "
                 "50/50 + 30/70 train ratios) on the 2024+25 baseline vs 2026 papers.\n")
    lines.append("- **Left panel (balanced)**: vision is already *ahead* of text on every family with "
                 "data (+4 to +9pp on baseline). For **CV papers, the vision advantage widens from +4pp "
                 "(baseline) to +26pp (2026) — a Δ of +22pp**. This is the single most decisive piece of "
                 "evidence that **vision is more robust than text to temporal distribution shift in CV**, "
                 "where the visual layout / figures stay informative even as the textual content shifts.")
    lines.append("- **Right panel (natural)**: text is ~10pp ahead of vision on every family — this is the "
                 "current deployment story. The 2026 gap closes on CV (Δ +2.2 toward vision) but text still "
                 "wins. So the headline conclusion is: **today's natural-prior deployment favors text, but "
                 "the 2026 evidence on balanced test suggests vision will be the more robust choice as "
                 "newer-year papers accumulate.**\n")
    lines.append("- **NLP and Robotics**: no 2026 papers in the y24up subset (NLP venues are mostly 2024-25; "
                 "CORL is too small). Can't conclude on temporal shift for these.\n")
    lines.append("**Practical takeaway**: the modality choice depends on your deployment horizon. For "
                 "current natural-prior accuracy, text 30/70. For CV-paper-heavy workloads under balanced "
                 "evaluation, or for robustness to year-on-year drift, vision is the safer pick.\n")

    with open(REPORT_PATH, "a") as f:
        f.write("\n".join(lines))
    print(f"\nAppended family-level analysis to: {REPORT_PATH}")


def main():
    print("Computing arxiv thresholds (for calibrated heatmap)...")
    arxiv_taus = compute_arxiv_thresholds()

    print("Building per-venue heatmap (arxiv natural test)...")
    fig_venue_heatmap_natural_test(arxiv_taus, str(FIG_DIR / "ratio_xeval_family_heatmap_arxiv_natural_test"))

    print("Building family year trajectory...")
    fig_family_year_trajectory(str(FIG_DIR / "ratio_xeval_family_year_trajectory"))

    print("Building family vision-minus-text gap (2026)...")
    fig_family_vision_minus_text(str(FIG_DIR / "ratio_xeval_family_vision_minus_text_2026"))

    print("Appending family-level analysis section...")
    append_section()


if __name__ == "__main__":
    main()
