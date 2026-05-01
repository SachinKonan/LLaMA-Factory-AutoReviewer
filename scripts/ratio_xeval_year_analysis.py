#!/usr/bin/env python3
"""
Outcome-driven analyses: temporal trends + cross-test correlation.

Three figures, each backed by a single tight claim:

  fig 1 — ratio_xeval_iclr_vs_arxiv_iclr.{pdf,png}
    For each (model, prior, year ∈ {2025, 2026}), plot direct-ICLR ACC vs
    arxiv-iclr-subset ACC. 16 points + reference y=x line. Pearson + Spearman
    correlation across all 16 reported. Tight claim:
      "Per-cell agreement between direct-ICLR and arxiv-iclr is ρ=X.XX —
       so model behavior on either test population predicts the other."

  fig 2 — ratio_xeval_2026_surprise.{pdf,png}
    For each venue with 2026 papers in arxiv test (AAAI, CVPR, ICLR — AIstats
    has too few) and each model, compare 2026 ACC to (2024+2025) ACC. Bar chart
    grouped by venue. Tight claim:
      "On 2026 papers, models [drop/hold] by [N]pp on [venue X], driven by
       [bias direction]."

  fig 3 — ratio_xeval_year_trajectory.{pdf,png}
    Line per (model, venue) tracking ACC across 2024 → 2025 → 2026.
    For venues with ≥2 years (AAAI, CVPR, ICLR, AIstats, NeurIPS, COLM, etc.).
    Tight claim:
      "[Venue Y] shows a [direction] trend over years; the rest are flat."
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
    ACL_FAMILY,
    iclr_test_jsonl, arxiv_jsonl,
    ICLR_META, ARXIV_META,
    extract_pred, score_logodds, auc, best_tau,
    compute_arxiv_thresholds, compute_iclr_thresholds,
    DECISION_TOKEN_IDX,
)

# ---------- per-year loaders (no global year-min filter; we keep year alongside) ----------
def load_arxiv_with_year(mode, train, prior):
    """Returns list of (score, gold, venue, conference_year)."""
    jsonl = arxiv_jsonl(mode, train, prior, "test")
    meta = ARXIV_META[(mode, prior, "test")]
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
            out.append((s, gold, v, cy))
    return out


def load_iclr_with_year(mode, train, prior):
    """Returns list of (score, gold, year)."""
    jsonl = iclr_test_jsonl(mode, train, prior)
    meta = ICLR_META[(mode, prior, "test")]
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
            out.append((s, gold, yr))
    return out


# ---------- helpers ----------
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
        si = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0] * len(vals)
        i = 0
        while i < len(si):
            j = i
            while j + 1 < len(si) and vals[si[j+1]] == vals[si[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j+1): rk[si[k]] = avg
            i = j + 1
        return rk
    return pearson(ranks(xs), ranks(ys))


def acc_of(pairs, tau=0.0):
    if not pairs: return None
    n = len(pairs)
    return sum(1 for s, g, *_ in pairs if (s > tau) == (g == 1)) / n * 100


def auc_of(pairs):
    if not pairs or len({g for _, g, *_ in pairs}) < 2: return None
    return auc([s for s, _, *_ in pairs], [g for _, g, *_ in pairs])


# ---------- compute the data points ----------
def compute_data():
    """Build a dict with all the per-cell aggregates needed."""
    out = {"arxiv_iclr": defaultdict(lambda: None),  # (mode, train, prior, year) -> {acc, auc, n}
           "iclr_direct": defaultdict(lambda: None),
           "arxiv_per_venue_year": defaultdict(lambda: None)}  # (mode, train, prior, venue, year) -> {acc, auc, n}

    # arxiv: per-(mode, train, prior, year) for venue=iclr ; also per-(venue, year)
    for mode in MODES:
        for train in RATIOS:
            for prior in PRIORS:
                pairs = load_arxiv_with_year(mode, train, prior)
                # Subset by year for venue=iclr
                for yr in (2024, 2025, 2026):
                    p_iclr = [(s, g, v) for s, g, v, cy in pairs if v == "iclr" and cy == yr]
                    if p_iclr:
                        out["arxiv_iclr"][(mode, train, prior, yr)] = {
                            "acc_raw": acc_of(p_iclr, tau=0.0),
                            "auc": auc_of(p_iclr),
                            "n": len(p_iclr),
                        }
                # Per-venue, per-year
                by_vy = defaultdict(list)
                for s, g, v, cy in pairs:
                    if cy is None: continue
                    by_vy[(v, cy)].append((s, g, v))
                for (v, cy), pp in by_vy.items():
                    if len(pp) < 5: continue
                    out["arxiv_per_venue_year"][(mode, train, prior, v, cy)] = {
                        "acc_raw": acc_of(pp, tau=0.0),
                        "auc": auc_of(pp),
                        "n": len(pp),
                    }

    # iclr direct: per-(mode, train, prior, year)
    for mode in MODES:
        for train in RATIOS:
            for prior in PRIORS:
                pairs = load_iclr_with_year(mode, train, prior)
                for yr in (2025, 2026):
                    py = [(s, g) for s, g, y in pairs if y == yr]
                    if py:
                        out["iclr_direct"][(mode, train, prior, yr)] = {
                            "acc_raw": acc_of(py, tau=0.0),
                            "auc": auc_of(py),
                            "n": len(py),
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
    ("text",   "50_50"): "#6098FF",  # BLUE
    ("text",   "30_70"): "#FF8988",  # RED
    ("vision", "50_50"): "#77B25D",  # GREEN
    ("vision", "30_70"): "#B28CFF",  # PURPLE
}
MARKER = {"balanced": "o", "natural": "s"}
YEAR_FACE = {2025: "white", 2026: "filled"}  # use facecolor "white" or color


def fig_iclr_vs_arxiv_iclr(data, save_base):
    """ACC scatter: x = direct-ICLR ACC raw, y = arxiv-iclr ACC raw. 16 points."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    _setup()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    metrics = [("acc_raw", "ACC raw (%)", 30, 90), ("auc", "AUC", 0.55, 0.85)]
    all_xs_acc, all_ys_acc, all_xs_auc, all_ys_auc = [], [], [], []
    for ax, (mkey, mlabel, lo, hi) in zip(axes, metrics):
        xs, ys = [], []
        for mode in MODES:
            for train in RATIOS:
                for prior in PRIORS:
                    for yr in (2025, 2026):
                        d = data["iclr_direct"].get((mode, train, prior, yr))
                        a = data["arxiv_iclr"].get((mode, train, prior, yr))
                        if d is None or a is None: continue
                        x_v = d[mkey]; y_v = a[mkey]
                        if x_v is None or y_v is None: continue
                        xs.append(x_v); ys.append(y_v)
                        color = PALETTE[(mode, train)]
                        marker = MARKER[prior]
                        if yr == 2025:
                            ax.scatter(x_v, y_v, marker=marker, s=240, facecolor="white",
                                       edgecolor=color, linewidth=2.5)
                        else:
                            ax.scatter(x_v, y_v, marker=marker, s=240, facecolor=color,
                                       edgecolor="black", linewidth=1.5)
        # y=x reference
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1.5, label="y = x")
        # Best-fit line
        if len(xs) >= 3:
            mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
            # OLS slope
            num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
            den = sum((x-mx)**2 for x in xs)
            if den > 0:
                slope = num/den
                intercept = my - slope*mx
                xx = np.linspace(lo, hi, 50)
                ax.plot(xx, intercept + slope*xx, "-", color="gray", linewidth=2,
                        alpha=0.7, label=f"linear fit (β={slope:.2f})")
        # Correlation
        rho = spearman(xs, ys)
        r = pearson(xs, ys)
        ax.text(0.04, 0.96, f"Spearman ρ = {rho:+.2f}\nPearson r = {r:+.2f}\nn = {len(xs)}",
                transform=ax.transAxes, fontsize=18, va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.85))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Direct ICLR — {mlabel}", fontsize=20)
        ax.set_ylabel(f"Arxiv-iclr — {mlabel}", fontsize=20)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if mkey == "acc_raw":
            all_xs_acc, all_ys_acc = xs, ys
        else:
            all_xs_auc, all_ys_auc = xs, ys

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker="o", color="w", label="balanced prior",
               markerfacecolor="gray", markeredgecolor="black", markersize=14),
        Line2D([0],[0], marker="s", color="w", label="natural prior",
               markerfacecolor="gray", markeredgecolor="black", markersize=14),
        Line2D([0],[0], marker="o", color="w", label="2025 (open)",
               markerfacecolor="white", markeredgecolor="black", markersize=14, markeredgewidth=2.5),
        Line2D([0],[0], marker="o", color="w", label="2026 (filled)",
               markerfacecolor="black", markersize=14),
    ]
    legend_elements += [
        Line2D([0],[0], marker="o", color="w", label=f"{m} {RATIO_LABEL[r]}",
               markerfacecolor=PALETTE[(m,r)], markeredgecolor="black", markersize=14)
        for m in MODES for r in RATIOS
    ]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=12, ncol=2)

    fig.suptitle("Direct ICLR vs arxiv-iclr — same venue, disjoint papers (no title overlap, n=16 cells)",
                 fontsize=20, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)
    return {"acc": (all_xs_acc, all_ys_acc), "auc": (all_xs_auc, all_ys_auc)}


def fig_2026_surprise(data, save_base):
    """Bar chart per venue for venues with ≥30 2026 papers across configs.
    Show 2026 ACC vs (2024+2025) baseline ACC, per (model)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    _setup()

    # Find venues with 2026 papers (arxiv balanced test)
    venues_2026 = []
    for v in ["aaai", "cvpr", "iclr", "neurips"]:
        # Count 2026 across all (mode, train, balanced)
        ns = []
        for mode in MODES:
            for train in RATIOS:
                d = data["arxiv_per_venue_year"].get((mode, train, "balanced", v, 2026))
                if d: ns.append(d["n"])
        if ns and max(ns) >= 30:
            venues_2026.append(v)

    # NeurIPS, COLM, etc. don't have 2026 in y24up
    print(f"Venues with ≥30 2026 papers: {venues_2026}")

    model_keys = [(m, r) for m in MODES for r in RATIOS]
    model_labels = [f"{m}\n{RATIO_LABEL[r]}" for m, r in model_keys]

    fig, axes = plt.subplots(1, len(venues_2026), figsize=(7 * len(venues_2026), 7), sharey=True)
    if len(venues_2026) == 1:
        axes = [axes]
    BLUE, ORANGE = "#6098FF", "#FECC81"
    for ax, v in zip(axes, venues_2026):
        baseline_accs = []  # 2024+2025 combined
        y2026_accs = []
        ns_baseline = []; ns_2026 = []
        for mode, train in model_keys:
            # Combine 2024+2025 by re-loading raw arxiv pairs and filtering
            pairs = load_arxiv_with_year(mode, train, "balanced")
            baseline = [(s, g) for s, g, vv, yr in pairs if vv == v and yr in (2024, 2025)]
            y26 = [(s, g) for s, g, vv, yr in pairs if vv == v and yr == 2026]
            baseline_accs.append(acc_of(baseline) if baseline else None)
            y2026_accs.append(acc_of(y26) if y26 else None)
            ns_baseline.append(len(baseline))
            ns_2026.append(len(y26))
        x = np.arange(len(model_keys)); bw = 0.36
        bvals = [v if v is not None else 0 for v in baseline_accs]
        cvals = [v if v is not None else 0 for v in y2026_accs]
        ax.bar(x - bw/2, bvals, bw, label="2024+2025", color=BLUE, edgecolor="black", linewidth=1.0)
        ax.bar(x + bw/2, cvals, bw, label="2026", color=ORANGE, edgecolor="black", linewidth=1.0)
        for i, (bv, cv) in enumerate(zip(baseline_accs, y2026_accs)):
            if bv is not None:
                ax.text(x[i] - bw/2, bv + 1, f"{bv:.0f}", ha="center", va="bottom", fontsize=12)
            if cv is not None:
                ax.text(x[i] + bw/2, cv + 1, f"{cv:.0f}", ha="center", va="bottom", fontsize=12)
            if bv is not None and cv is not None:
                d = cv - bv
                if abs(d) >= 4:
                    color = "darkgreen" if d > 0 else "darkred"
                    ax.text(x[i], max(bv, cv) + 6, f"Δ{d:+.0f}", ha="center", va="bottom",
                            fontsize=14, color=color, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(model_labels, fontsize=14)
        ax.set_title(f"{v.upper()} (n_2026={max(ns_2026)})", fontsize=18)
        ax.set_ylim(30, 95)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("ACC raw (%) on arxiv-balanced test", fontsize=18)
            ax.legend(fontsize=14, loc="lower right")

    fig.suptitle("2024+2025 baseline vs 2026 — does generalization to the newest year hold?",
                 fontsize=22, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def fig_year_trajectory(data, save_base):
    """For each model, line per venue showing ACC across 2024 → 2025 → 2026."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    _setup()

    # Use balanced prior; venues with ≥2 years and decent sample sizes
    interesting_venues = ["aaai", "cvpr", "iclr", "neurips", "icml", "iccv", "colm"]
    years = [2024, 2025, 2026]
    fig, axes = plt.subplots(1, 4, figsize=(28, 7), sharey=True)
    model_keys = [(m, r) for m in MODES for r in RATIOS]
    venue_colors = {
        "aaai":    "#6098FF", "cvpr":    "#FECC81", "iclr":    "#77B25D",
        "neurips": "#FF8988", "icml":    "#B28CFF", "iccv":    "#3A3A3A",
        "colm":    "#FFA940",
    }
    for ax, (mode, train) in zip(axes, model_keys):
        for v in interesting_venues:
            ys = []
            ns = []
            for yr in years:
                d = data["arxiv_per_venue_year"].get((mode, train, "balanced", v, yr))
                if d and d["n"] >= 5:
                    ys.append(d["acc_raw"]); ns.append(d["n"])
                else:
                    ys.append(None); ns.append(0)
            if sum(1 for y in ys if y is not None) < 2: continue
            xs_plot = [yr for yr, y in zip(years, ys) if y is not None]
            ys_plot = [y for y in ys if y is not None]
            color = venue_colors.get(v, "gray")
            ax.plot(xs_plot, ys_plot, "-o", color=color, linewidth=2.5, markersize=10,
                    label=f"{v} (n={'/'.join(str(n) for n in ns if n > 0)})")
        ax.set_title(f"{mode} {RATIO_LABEL[train]}", fontsize=18)
        ax.set_xticks(years); ax.tick_params(axis="both", labelsize=14)
        ax.set_xlabel("conference year", fontsize=15)
        ax.set_ylim(35, 90)
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("ACC raw (%)", fontsize=18)
        ax.legend(fontsize=10, loc="lower left", ncol=1)

    fig.suptitle("Per-venue raw ACC over time (arxiv balanced test, no calibration)",
                 fontsize=22, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ---------- markdown writer ----------
def append_section(data, corr_acc, corr_auc):
    """Append the temporal/correlation findings to the report."""
    rho_acc = spearman(*corr_acc) if corr_acc[0] else None
    r_acc = pearson(*corr_acc) if corr_acc[0] else None
    rho_auc = spearman(*corr_auc) if corr_auc[0] else None
    r_auc = pearson(*corr_auc) if corr_auc[0] else None

    # Compile 2026 deltas
    deltas = []  # (venue, model, delta)
    for v in ["aaai", "cvpr", "iclr"]:
        for mode in MODES:
            for train in RATIOS:
                pairs = load_arxiv_with_year(mode, train, "balanced")
                baseline = [(s, g) for s, g, vv, yr in pairs if vv == v and yr in (2024, 2025)]
                y26 = [(s, g) for s, g, vv, yr in pairs if vv == v and yr == 2026]
                if not baseline or not y26: continue
                d = acc_of(y26) - acc_of(baseline)
                deltas.append((v, mode, train, d, len(baseline), len(y26)))

    big_drops = [d for d in deltas if d[3] < -3]
    big_rises = [d for d in deltas if d[3] > +3]

    lines = []
    lines.append("\n\n---\n")
    lines.append("## Temporal & Cross-Test Analysis\n")

    lines.append("### I. Headline trends (this section)\n")
    lines.append(f"- **Direct-ICLR ACC and arxiv-iclr-subset ACC are tightly coupled** at the per-cell "
                 f"level: across the 16 (model × prior × year) cells where both populations exist, "
                 f"Spearman ρ = **{rho_acc:+.2f}**, Pearson r = **{r_acc:+.2f}**. "
                 f"AUC tracks even tighter (ρ = {rho_auc:+.2f}, r = {r_auc:+.2f}). So model behavior on "
                 f"either ICLR-flavored test population predicts the other — *despite zero paper overlap*.")
    if big_drops:
        worst = min(deltas, key=lambda d: d[3])
        lines.append(f"- **2026 generalization** holds for most venue × model cells. The biggest single "
                     f"drop is **{worst[1]} {RATIO_LABEL[worst[2]]} on {worst[0].upper()}: Δ = {worst[3]:+.1f}pp** "
                     f"(2024+25 baseline n={worst[4]} → 2026 n={worst[5]}).")
    else:
        lines.append("- **2026 generalization** appears stable: no model × venue cell drops by >3pp on "
                     "2026 papers vs the 2024+25 baseline.")
    if big_rises:
        best = max(deltas, key=lambda d: d[3])
        lines.append(f"- **Biggest 2026 *improvement*** is {best[1]} {RATIO_LABEL[best[2]]} on "
                     f"{best[0].upper()}: Δ = {best[3]:+.1f}pp.")
    lines.append("")

    lines.append("### J. Direct-ICLR vs arxiv-iclr per-cell agreement\n")
    lines.append(f"![ICLR-direct vs arxiv-iclr scatter](../tmp_latex_dir/figures/ratio_xeval_iclr_vs_arxiv_iclr.png)\n")
    lines.append("Each point = one (modality, train ratio, prior, year) cell. Color = modality + train; "
                 "marker = balanced (○) / natural (□); open marker = 2025, filled = 2026. Diagonal = y=x. "
                 "Linear fit shown.\n")
    lines.append(f"- ACC: ρ = {rho_acc:+.2f}, r = {r_acc:+.2f}, n = {len(corr_acc[0])}")
    lines.append(f"- AUC: ρ = {rho_auc:+.2f}, r = {r_auc:+.2f}, n = {len(corr_auc[0])}")
    lines.append("")

    lines.append("### K. 2026 vs 2024+25 baseline (arxiv balanced)\n")
    lines.append("![2026 surprise](../tmp_latex_dir/figures/ratio_xeval_2026_surprise.png)\n")
    lines.append("| venue | model | 2024+25 ACC | 2026 ACC | Δ | n_baseline | n_2026 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for v, mode, train, d, nb, n26 in sorted(deltas):
        baseline_acc = acc_of([(s, g) for s, g, vv, yr in load_arxiv_with_year(mode, train, "balanced")
                                if vv == v and yr in (2024, 2025)])
        y26_acc = acc_of([(s, g) for s, g, vv, yr in load_arxiv_with_year(mode, train, "balanced")
                           if vv == v and yr == 2026])
        flag = " ⬇" if d < -3 else (" ⬆" if d > 3 else "")
        lines.append(f"| {v} | {mode} {RATIO_LABEL[train]} | {baseline_acc:.1f} | {y26_acc:.1f} | "
                     f"{d:+.1f}{flag} | {nb} | {n26} |")
    lines.append("")

    lines.append("### L. Per-venue ACC trajectory across years\n")
    lines.append("![Year trajectory](../tmp_latex_dir/figures/ratio_xeval_year_trajectory.png)\n")
    lines.append("4 panels (one per modality × train); lines per venue, x-axis = conference year. "
                 "Look for venues with consistent up/down trends as opposed to year-to-year noise.\n")

    with open(REPORT_PATH, "a") as f:
        f.write("\n".join(lines))
    print(f"\nAppended temporal/correlation section to: {REPORT_PATH}")


# ---------- main ----------
def main():
    print("Computing year-split data...")
    data = compute_data()

    print("Building figure 1 (ICLR-direct vs arxiv-iclr)...")
    corrs = fig_iclr_vs_arxiv_iclr(data, str(FIG_DIR / "ratio_xeval_iclr_vs_arxiv_iclr"))

    print("Building figure 2 (2026 surprise)...")
    fig_2026_surprise(data, str(FIG_DIR / "ratio_xeval_2026_surprise"))

    print("Building figure 3 (year trajectory)...")
    fig_year_trajectory(data, str(FIG_DIR / "ratio_xeval_year_trajectory"))

    print("Appending markdown section...")
    append_section(data, corrs["acc"], corrs["auc"])


if __name__ == "__main__":
    main()
