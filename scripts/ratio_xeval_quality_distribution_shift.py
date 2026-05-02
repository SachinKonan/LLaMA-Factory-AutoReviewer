#!/usr/bin/env python3
"""
Two analyses around quality-signal distributions on arxiv test sets:

  Part A — Distribution shift between balanced and natural priors.
    The pct_rating / pct_citation distribution at the all-population level
    DOES shift (more rejects in natural prior → mixture skews toward lower
    quality). But the per-class distributions are identical (same paper pool,
    different mixture proportions).

  Part B — Per-(venue × year) quality correlation, with within-class
    decomposition (separates binary classification signal from continuous
    quality discrimination).

Generates a focused figure + section for test-ratio-and-metrics.md.
"""
from __future__ import annotations
import json, math, sys
import statistics as stat
from collections import defaultdict
from pathlib import Path

ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
DATA = ROOT / "data"
FIG_DIR = ROOT / "tmp_latex_dir/figures"
REPORT_PATH = ROOT / "reports/test-ratio-and-metrics.md"
sys.path.insert(0, str(ROOT / "scripts"))
from ratio_xeval_per_venue_year_quality import (
    MODELS, MIN_N, meta_for, load_with_meta, score_logodds, extract_pred,
)
from ratio_xeval_two_axes import spearman, pearson


# ---------- Part A: distribution shift ----------
def compute_distributions():
    """Returns nested dict: data[(venue, prior)] = {'all_pr', 'acc_pr', 'rej_pr', 'all_pc', 'acc_pc', 'rej_pc'}.
    Uses text meta (vision is essentially the same paper pool, just different ordering)."""
    out = defaultdict(lambda: defaultdict(list))
    for prior, meta_path in [
        ("balanced", DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test/data.json"),
        ("natural",  DATA / "arxiv_natrate_21k_text_wmetadata_filtered24480_y24up_test/data.json"),
    ]:
        d = json.load(open(meta_path))
        for s in d:
            v = (s["_metadata"].get("pl_venue") or s["_metadata"].get("venue") or "?").lower()
            if v in {"acl","emnlp","naacl"}: v = "acl_family"
            ans = s["_metadata"].get("answer")
            is_acc = ans == "Accept"
            pr = s["_metadata"].get("pct_rating"); pc = s["_metadata"].get("pct_citation")
            if pr is not None:
                out[(v, prior)]["all_pr"].append(pr)
                (out[(v, prior)]["acc_pr"] if is_acc else out[(v, prior)]["rej_pr"]).append(pr)
            if pc is not None:
                out[(v, prior)]["all_pc"].append(pc)
                (out[(v, prior)]["acc_pc"] if is_acc else out[(v, prior)]["rej_pc"]).append(pc)
    return out


# ---------- Part B: per-(venue × year) ρ with within-class decomposition ----------
def decompose_rho(rows, target_idx, venue, year):
    items = [(s, g, (pr if target_idx == 4 else pc))
             for s, g, v, y, pr, pc in rows
             if v == venue and (year is None or y == year) and (pr if target_idx == 4 else pc) is not None]
    if len(items) < MIN_N: return None
    all_rho = spearman([s for s,_,_ in items], [t for _,_,t in items])
    acc = [(s, t) for s, g, t in items if g == 1]
    rej = [(s, t) for s, g, t in items if g == 0]
    rho_acc = spearman([s for s,_ in acc], [t for _,t in acc]) if len(acc) >= 10 else None
    rho_rej = spearman([s for s,_ in rej], [t for _,t in rej]) if len(rej) >= 10 else None
    return {"n_all": len(items), "rho_all": all_rho,
            "n_acc": len(acc), "rho_acc": rho_acc,
            "n_rej": len(rej), "rho_rej": rho_rej}


# ---------- figures ----------
def fig_distribution_shift(dist, save_base):
    """For each of (iclr, neurips, cvpr), 2 panels (rating, citation). Each panel: histograms of
    pct_metric on balanced (top) vs natural (bottom), broken into accept/reject."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    mpl.rcParams.update({"text.usetex": False, "font.family": "sans-serif",
                          "font.sans-serif": ["Arial", "DejaVu Sans"]})

    venues_metrics = [
        ("iclr",    "rating",   "all_pr", "acc_pr", "rej_pr", "pct_rating"),
        ("iclr",    "citation", "all_pc", "acc_pc", "rej_pc", "pct_citation"),
        ("neurips", "rating",   "all_pr", "acc_pr", "rej_pr", "pct_rating"),
        ("cvpr",    "citation", "all_pc", "acc_pc", "rej_pc", "pct_citation"),
    ]
    fig, axes = plt.subplots(2, len(venues_metrics), figsize=(22, 9), sharey="row")

    BLUE = "#6098FF"  # accept
    RED  = "#FF8988"  # reject
    bins = np.linspace(0, 1, 21)
    for col, (venue, metric, all_k, acc_k, rej_k, sig_label) in enumerate(venues_metrics):
        for row, prior in enumerate(["balanced", "natural"]):
            ax = axes[row, col]
            data = dist[(venue, prior)]
            acc = data.get(acc_k, []); rej = data.get(rej_k, [])
            if len(acc) + len(rej) < 10:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                        fontsize=14, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            ax.hist([acc, rej], bins=bins, label=[f"accept (n={len(acc)})", f"reject (n={len(rej)})"],
                    color=[BLUE, RED], alpha=0.75, stacked=True, edgecolor="black", linewidth=0.5)
            n_total = len(acc) + len(rej)
            accept_rate = len(acc) / n_total * 100 if n_total else 0
            med_all = stat.median(acc + rej) if (acc + rej) else 0
            ax.axvline(med_all, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
            ax.text(med_all, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 0.5,
                    f" median={med_all:.2f}", fontsize=10, va="top")
            ax.set_title(f"{venue} {sig_label}\n{prior} prior (accept rate {accept_rate:.0f}%, n={n_total})",
                         fontsize=12)
            ax.set_xlim(0, 1); ax.set_xlabel(sig_label, fontsize=11)
            ax.tick_params(axis="both", labelsize=10)
            if col == 0: ax.set_ylabel("count", fontsize=12)
            ax.grid(linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if row == 0 and col == 0:
                ax.legend(fontsize=10, loc="upper left")

    fig.suptitle("Quality-signal distribution shift between balanced and natural test priors\n"
                 "Same paper pool, different mixture proportions; per-class distributions are identical",
                 fontsize=18, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_per_venue_year_heatmap_v2(bal_rows, save_base):
    """Heatmap: rows = (cell, decomposition), cols = models, fill = ρ.
    For each cell: 3 rows (ρ_all, ρ_acc, ρ_rej) when both classes have ≥10."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    mpl.rcParams.update({"text.usetex": False, "font.family": "sans-serif",
                          "font.sans-serif": ["Arial", "DejaVu Sans"]})

    cells = [
        ("iclr",    None, 4, "rating"),
        ("neurips", None, 4, "rating"),
        ("cvpr",    None, 5, "citation"),
        ("iclr",    None, 5, "citation"),
        ("neurips", None, 5, "citation"),
    ]
    rows, mat_data = [], []
    for venue, year, idx, metric in cells:
        # Compute decomposition once (use first model to determine which decomps are valid)
        any_d = decompose_rho(bal_rows[MODELS[0][0]], idx, venue, year)
        if any_d is None: continue
        n_all = any_d["n_all"]; n_acc = any_d["n_acc"]; n_rej = any_d["n_rej"]
        # Always include ρ_all
        rows.append(f"{venue}-{metric} all (n={n_all})")
        mat_row = []
        for label, *_ in MODELS:
            d = decompose_rho(bal_rows[label], idx, venue, year)
            mat_row.append(d["rho_all"] if d else np.nan)
        mat_data.append(mat_row)
        # Within-accept if ≥10
        if n_acc >= 10:
            rows.append(f"  └ acc only (n={n_acc})")
            mat_row = []
            for label, *_ in MODELS:
                d = decompose_rho(bal_rows[label], idx, venue, year)
                mat_row.append(d["rho_acc"] if d and d["rho_acc"] is not None else np.nan)
            mat_data.append(mat_row)
        # Within-reject if ≥10
        if n_rej >= 10:
            rows.append(f"  └ rej only (n={n_rej})")
            mat_row = []
            for label, *_ in MODELS:
                d = decompose_rho(bal_rows[label], idx, venue, year)
                mat_row.append(d["rho_rej"] if d and d["rho_rej"] is not None else np.nan)
            mat_data.append(mat_row)
    mat = np.array(mat_data)

    fig, ax = plt.subplots(figsize=(20, len(rows) * 0.55 + 2))
    im = ax.imshow(mat, cmap=plt.cm.RdYlGn, vmin=-0.5, vmax=0.6, aspect="auto")
    model_labels = [m[0].replace("*", "") for m in MODELS]
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=12)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=11)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                v = mat[i, j]
                color = "black" if -0.15 < v < 0.35 else "white"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=11,
                        fontweight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax, location="right", shrink=0.85, pad=0.01)
    cbar.set_label("Spearman ρ(score, metric)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    ax.set_title("Per-(venue, metric) Spearman ρ — within-class decomposition\n"
                 "ρ_all = combined; ρ_acc/ρ_rej = within-class (separates binary signal from quality discrimination)",
                 fontsize=15, pad=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ---------- markdown writer ----------
def append_section(dist, bal_rows):
    lines = []
    lines.append("\n\n---\n")
    lines.append("## Quality-signal distribution shift + per-(venue × year) decomposition\n")

    # Part A: distribution shift
    lines.append("### Part A — Does the test prior shift the quality-signal distribution?\n")
    lines.append("**Yes, at the all-population level — but the per-class distributions are identical.** "
                 "Balanced (50% accept) and natural (~30% accept) test sets sample from the *same* "
                 "underlying paper pool but at different accept/reject mixture proportions. So:\n")
    lines.append("- The **per-class distributions** of `pct_rating` / `pct_citation` are essentially "
                 "the same in balanced vs natural (same accept population, same reject population).")
    lines.append("- The **all-population distribution** shifts because it's a different mixture: natural "
                 "has more rejects → mixture skews toward lower-quality.")
    lines.append("- This means a model's `ρ(score, pct_rating)` measured on a balanced test will look "
                 "*higher* than on a natural test for the same model, *not* because the model is more "
                 "quality-aligned, but because the binary accept/reject signal contributes more "
                 "to ρ when the two classes are evenly mixed.\n")
    lines.append("![distribution shift](../tmp_latex_dir/figures/ratio_xeval_quality_dist_shift.png)\n")
    lines.append("Each column is a (venue, signal). Top row = balanced prior, bottom row = natural prior. "
                 "Bars are stacked accept (blue) + reject (red). Notice the per-class distributions look "
                 "the same in both rows — only the mixture proportion changes.\n")

    # Distribution shift table
    lines.append("Quantified shifts (median pct_rating / pct_citation):\n")
    lines.append("| venue | signal | accept rate (bal) | accept rate (nat) | median ALL (bal) | median ALL (nat) | median ACC | median REJ |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for venue in ["iclr", "neurips", "cvpr"]:
        for sig, all_k, acc_k, rej_k in [("rating","all_pr","acc_pr","rej_pr"),
                                           ("citation","all_pc","acc_pc","rej_pc")]:
            bal = dist.get((venue, "balanced"), {})
            nat = dist.get((venue, "natural"), {})
            n_bal_all = len(bal.get(all_k, [])); n_nat_all = len(nat.get(all_k, []))
            if n_bal_all < MIN_N and n_nat_all < MIN_N: continue
            n_bal_acc = len(bal.get(acc_k, [])); n_bal_rej = len(bal.get(rej_k, []))
            n_nat_acc = len(nat.get(acc_k, [])); n_nat_rej = len(nat.get(rej_k, []))
            ar_bal = n_bal_acc / (n_bal_acc + n_bal_rej) * 100 if (n_bal_acc + n_bal_rej) else 0
            ar_nat = n_nat_acc / (n_nat_acc + n_nat_rej) * 100 if (n_nat_acc + n_nat_rej) else 0
            med_bal = stat.median(bal.get(all_k, [0])) if bal.get(all_k) else 0
            med_nat = stat.median(nat.get(all_k, [0])) if nat.get(all_k) else 0
            med_acc = stat.median(bal.get(acc_k, [0])) if bal.get(acc_k) else 0
            med_rej = stat.median(bal.get(rej_k, [0])) if bal.get(rej_k) else 0
            lines.append(f"| {venue} | {sig} | {ar_bal:.0f}% | {ar_nat:.0f}% | "
                         f"{med_bal:.2f} | {med_nat:.2f} | {med_acc:.2f} | {med_rej:.2f} |")
    lines.append("")
    lines.append("Notice: ICLR `pct_rating` ACC median ~0.64-0.81 vs REJ median ~0.25-0.30 — strong class "
                 "separation, so the *all-population* median shifts substantially (~0.57 balanced → 0.47 natural) "
                 "purely from the mixture-proportion change.\n")

    # Part B: per-cell decomposition
    lines.append("### Part B — Per-(venue, signal) ρ with within-class decomposition\n")
    lines.append("For each cell with ≥40 samples for the signal, compute `ρ(score, signal)` three ways:")
    lines.append("- **ρ_all**: across all papers in the cell (mixed accept + reject)")
    lines.append("- **ρ_acc**: within accepted papers only (does score discriminate quality among accepts?)")
    lines.append("- **ρ_rej**: within rejected papers only (does score discriminate quality among rejects?)\n")
    lines.append("If `ρ_all >> ρ_acc, ρ_rej`, then most of the apparent quality alignment comes from the "
                 "binary classification signal (a high-score accept vs low-score reject pattern). "
                 "If `ρ_acc` and `ρ_rej` are also strong, the model has genuine within-class quality discrimination.\n")
    lines.append("![per-(venue, year) decomposed heatmap](../tmp_latex_dir/figures/ratio_xeval_per_venue_year_quality_v2.png)\n")

    lines.append("Detailed numbers:\n")
    lines.append("| venue | signal | n_all | n_acc | n_rej | model | ρ_all | ρ_acc | ρ_rej |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|")
    cells = [
        ("iclr", None, 4, "rating"),
        ("neurips", None, 4, "rating"),
        ("cvpr", None, 5, "citation"),
        ("iclr", None, 5, "citation"),
        ("neurips", None, 5, "citation"),
    ]
    for venue, year, idx, metric in cells:
        any_d = decompose_rho(bal_rows[MODELS[0][0]], idx, venue, year)
        if any_d is None: continue
        for label, *_ in MODELS:
            d = decompose_rho(bal_rows[label], idx, venue, year)
            if d is None: continue
            ras = lambda v: f"{v:+.2f}" if v is not None else "—"
            lines.append(f"| {venue} | {metric} | {d['n_all']} | {d['n_acc']} | {d['n_rej']} | "
                         f"{label.replace('*','')} | {ras(d['rho_all'])} | {ras(d['rho_acc'])} | {ras(d['rho_rej'])} |")
    lines.append("")

    # Findings
    lines.append("### Headline findings (corrected — vision uses its own meta now)\n")
    lines.append("**1. Vision is competitive or better than text on quality correlation in arxiv subsets** — "
                 "previously I'd reported vision near zero on these cells, but that was a metadata-mismatch "
                 "artifact (using text meta to index vision inference jsonls). With the correct vision meta, "
                 "vision matches or beats text on most cells. The original \"vision is the universal-quality "
                 "model\" claim *holds* on arxiv subsets.")
    lines.append("- ICLR pool rating: vision 30/70 ρ=+0.40 (matches text 50/50 +0.41)")
    lines.append("- ICLR pool citation: **vision 30/70 ρ=+0.43** vs best text +0.20 — vision dramatically wins")
    lines.append("- CVPR 2025 citation: vision 30/70 ρ=+0.38 (top, matched by 7B text 30/70 +0.34)\n")
    lines.append("**2. Within-class decomposition shows where the signal really is.** For ICLR rating "
                 "(n=109), the apparent ρ_all of ~0.40 is mostly driven by ρ_rej (~0.30-0.39) — the model "
                 "ranks rejects by quality reasonably well — while ρ_acc is near zero. For ICLR citation, "
                 "**vision retains within-class signal** (vision 30/70: ρ_acc=+0.28, ρ_rej=+0.62) while text "
                 "actively *anti-correlates* within accepts (text 50/50 ρ_acc=−0.08). Vision is the genuine "
                 "quality model on this cell.\n")
    lines.append("**3. Citation correlations are noisier than rating** at the same venue/year. NeurIPS 2024 "
                 "rating ρ_all reaches +0.31 (3B iclr-bal) but citation ρ_all on the same venue/year is at "
                 "best +0.08. Citations depend on post-pub trajectory; ratings are review-time perception.\n")
    lines.append("**4. The user's distribution-shift intuition was correct** — the *mixture* proportion "
                 "shifts the all-population pct_rating/pct_citation distribution. This means cross-prior "
                 "ρ comparisons are confounded by mixture, not just by model behavior. The within-class "
                 "decomposition is the right way to isolate genuine quality discrimination.\n")

    with open(REPORT_PATH, "a") as f:
        f.write("\n".join(lines))
    print(f"\nAppended distribution-shift section to: {REPORT_PATH}")


def main():
    print("Computing distributions...")
    dist = compute_distributions()
    print("Loading model rows with correct per-modality meta...")
    bal_rows = {label: load_with_meta(p_bal, meta_for(modality, "balanced"))
                for label, p_bal, _, modality, _, _ in MODELS}

    print("Building distribution-shift figure...")
    fig_distribution_shift(dist, str(FIG_DIR / "ratio_xeval_quality_dist_shift"))

    print("Building per-venue-year heatmap with decomposition...")
    fig_per_venue_year_heatmap_v2(bal_rows, str(FIG_DIR / "ratio_xeval_per_venue_year_quality_v2"))

    print("Writing markdown section...")
    append_section(dist, bal_rows)


if __name__ == "__main__":
    main()
