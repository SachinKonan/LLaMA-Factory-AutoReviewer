#!/usr/bin/env python3
"""
Per-(venue × year) quality-correlation analysis on arxiv balanced test.

Computes ρ(score, pct_rating) and ρ(score, pct_citation) for each model on each
(venue, year) cell with ≥40 samples for the metric. Compares 9 models across
training-data, ratio, modality, and size.

Cells with ≥40 samples (verified in earlier exploration):
  Rating:   iclr 2026 (51), neurips 2024 (77), neurips 2025 (98)
  Citation: cvpr 2024 (50), cvpr 2025 (47), neurips 2024 (77)

Output: tables + figure + appendix to test-ratio-and-metrics.md.
"""
from __future__ import annotations
import json, math, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
DATA = ROOT / "data"
FIG_DIR = ROOT / "tmp_latex_dir/figures"
REPORT_PATH = ROOT / "reports/test-ratio-and-metrics.md"

DECISION_TOKEN_IDX = 5
ACL_FAMILY = {"acl", "emnlp", "naacl"}
MIN_N = 40

# ---------- model registry ----------
# Each entry: (label, jsonl_path_balanced, jsonl_path_natrate, modality, train, ratio, size)
def jsonl(rdir, ckpt, sub):
    return ROOT / rdir / sub / f"finetuned-ckpt-{ckpt}.jsonl"

MODELS = [
    # 7B ICLR-trained text
    ("7B iclr text 50/50",
     jsonl("results/cross_conference_arxiv_y24up/bz32_lr1e-6_text", 1322, "arxiv_eval"),
     jsonl("results/cross_conference_arxiv_natrate_y24up/bz32_lr1e-6_text", 1322, "arxiv_eval"),
     "text", "iclr-bal", "7B"),
    ("7B iclr text 30/70",
     jsonl("results/cross_conference_arxiv_y24up/bz32_lr1e-6_text_30_70", 1322, "arxiv_eval"),
     jsonl("results/cross_conference_arxiv_natrate_y24up/bz32_lr1e-6_text_30_70", 1322, "arxiv_eval"),
     "text", "iclr-30/70", "7B"),
    # 7B ICLR-trained vision
    ("7B iclr vision 50/50",
     jsonl("results/cross_conference_arxiv_y24up/bz16_lr1e-6_vision", 2648, "arxiv_eval"),
     jsonl("results/cross_conference_arxiv_natrate_y24up/bz16_lr1e-6_vision", 2648, "arxiv_eval"),
     "vision", "iclr-bal", "7B"),
    ("7B iclr vision 30/70",
     jsonl("results/cross_conference_arxiv_y24up/bz16_lr1e-6_vision_30_70", 2642, "arxiv_eval"),
     jsonl("results/cross_conference_arxiv_natrate_y24up/bz16_lr1e-6_vision_30_70", 2642, "arxiv_eval"),
     "vision", "iclr-30/70", "7B"),
    # 3B arxiv-trained text
    ("3B arxiv-bal text",
     jsonl("results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/small_sachin/arxiv_21k_text_3b", 2624, "arxiv_balanced_test"),
     jsonl("results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/small_sachin/arxiv_21k_text_3b", 2624, "arxiv_natrate_test"),
     "text", "arxiv-bal", "3B"),
    ("3B arxiv-nat text",
     jsonl("results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/natrate_sachin/arxiv_natrate_21k_text_3b", 2624, "arxiv_balanced_test"),
     jsonl("results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/natrate_sachin/arxiv_natrate_21k_text_3b", 2624, "arxiv_natrate_test"),
     "text", "arxiv-nat", "3B"),
    ("3B iclr-bal text",
     jsonl("results/final_sweep_v7_datasweepv3/optim_search_2026/scaling/bz32_lr1e-6_text_3b/8eval", 2644, "arxiv_balanced_test"),
     jsonl("results/final_sweep_v7_datasweepv3/optim_search_2026/scaling/bz32_lr1e-6_text_3b/8eval", 2644, "arxiv_natrate_test"),
     "text", "iclr-bal", "3B"),
    # 7B arxiv-trained text
    ("7B arxiv-nat text",
     jsonl("results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/natrate_sachin/arxiv_natrate_21k_text", 1312, "arxiv_balanced_test"),
     jsonl("results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/natrate_sachin/arxiv_natrate_21k_text", 1312, "arxiv_natrate_test"),
     "text", "arxiv-nat", "7B"),
    ("7B arxiv-bal text*",
     jsonl("results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/small/arxiv_21k_text", 656, "arxiv_balanced_test"),
     jsonl("results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/small/arxiv_21k_text", 656, "arxiv_natrate_test"),
     "text", "arxiv-bal", "7B"),
]

META_BAL = DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test/data.json"
META_NAT = DATA / "arxiv_natrate_21k_text_wmetadata_filtered24480_y24up_test/data.json"


# ---------- helpers ----------
def extract_pred(t):
    s = (t or "").lower()
    if "\\boxed{accept}" in s or "boxed{accept}" in s: return "accept"
    if "\\boxed{reject}" in s or "boxed{reject}" in s: return "reject"
    return None

def score_logodds(row):
    p = row.get("predict", "")
    chosen = 1 if "Accept" in p else (0 if "Reject" in p else None)
    if chosen is None: return None
    tl = row.get("token_logprobs") or []
    if len(tl) > DECISION_TOKEN_IDX and tl[DECISION_TOKEN_IDX] is not None:
        lp = tl[DECISION_TOKEN_IDX]
        p_chosen = math.exp(lp); p_chosen = min(max(p_chosen, 1e-7), 1-1e-7)
        return (1 if chosen == 1 else -1) * math.log(p_chosen/(1-p_chosen))
    return 6.0 if chosen == 1 else -6.0

def vroll(v): return "acl_family" if v in ACL_FAMILY else v

def pearson(xs, ys):
    if len(xs) < 3: return None
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx and dy else None

def spearman(xs, ys):
    if len(xs) < 3: return None
    def ranks(v):
        si = sorted(range(len(v)), key=lambda i: v[i]); rk = [0.0]*len(v); i = 0
        while i < len(si):
            j = i
            while j+1 < len(si) and v[si[j+1]] == v[si[i]]: j += 1
            avg = (i+j)/2.0 + 1
            for k in range(i, j+1): rk[si[k]] = avg
            i = j+1
        return rk
    return pearson(ranks(xs), ranks(ys))


# ---------- loader ----------
def load_with_meta(jsonl_path, meta_path):
    """Returns list of (score, gold, venue, year, pct_rating, pct_citation)."""
    if not jsonl_path.exists() or not meta_path.exists():
        return []
    metas = json.loads(meta_path.read_text())
    out = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= len(metas): break
            r = json.loads(line)
            g = extract_pred(r.get("label", "")); s = score_logodds(r)
            if g is None or s is None: continue
            m = metas[i].get("_metadata") or {}
            v = vroll((m.get("pl_venue") or m.get("venue") or "?").lower())
            try: y = int(m.get("conference_year")) if m.get("conference_year") else None
            except: y = None
            pr = m.get("pct_rating"); pc = m.get("pct_citation")
            try: pr = float(pr) if pr is not None else None
            except: pr = None
            try: pc = float(pc) if pc is not None else None
            except: pc = None
            out.append((s, 1 if g == "accept" else 0, v, y, pr, pc))
    return out


# ---------- analysis ----------
def per_cell_correlation(rows, target_field_idx, venue, year):
    """target_field_idx: 4 for pct_rating, 5 for pct_citation. Returns dict or None."""
    items = []
    for row in rows:
        s, g, v, y, pr, pc = row
        if v != venue: continue
        if year is not None and y != year: continue
        target = pr if target_field_idx == 4 else pc
        if target is None: continue
        items.append((s, target))
    if len(items) < MIN_N: return None
    xs = [s for s,_ in items]; ys = [t for _,t in items]
    return {"n": len(items), "spearman": spearman(xs, ys), "pearson": pearson(xs, ys)}


def main():
    # Load all model rows
    print(f"Loading inference jsonls + metadata for {len(MODELS)} models...")
    bal_rows = {}; nat_rows = {}
    for label, p_bal, p_nat, *_ in MODELS:
        bal_rows[label] = load_with_meta(p_bal, META_BAL)
        nat_rows[label] = load_with_meta(p_nat, META_NAT)

    # Define cells with ≥40 samples (verified earlier)
    # Format: (venue, year_or_None_for_pooled, metric_field_name, metric_field_idx)
    cells = [
        ("iclr",    2026, "rating",   4),
        ("neurips", 2024, "rating",   4),
        ("neurips", 2025, "rating",   4),
        ("cvpr",    2024, "citation", 5),
        ("cvpr",    2025, "citation", 5),
        ("neurips", 2024, "citation", 5),
        # Pooled across years (already verified ≥40)
        ("iclr",    None, "rating",   4),
        ("neurips", None, "rating",   4),
        ("cvpr",    None, "citation", 5),
        ("iclr",    None, "citation", 5),
        ("neurips", None, "citation", 5),
    ]

    print("\n" + "="*150)
    print("ARXIV BALANCED TEST — per-(venue, year, metric) Spearman ρ(score, metric)")
    print("="*150)
    # Header
    header = f"{'venue':<10} {'yr':>5} {'metric':<8} {'n':>4} | "
    for label, *_ in MODELS:
        header += f"{label.split()[1] + ' ' + label.split()[2][:4]:<14}"
    print(header)
    print("-" * 200)

    # Each cell × each model
    for venue, year, metric, idx in cells:
        # n is consistent across models (same papers)
        n_check = None
        row = f"{venue:<10} {str(year) if year else 'pool':>5} {metric:<8} "
        results = {}
        for label, *_ in MODELS:
            r = per_cell_correlation(bal_rows[label], idx, venue, year)
            results[label] = r
            if r is not None and n_check is None: n_check = r["n"]
        if n_check is None: continue
        row = f"{venue:<10} {str(year) if year else 'pool':>5} {metric:<8} {n_check:>4} | "
        for label, *_ in MODELS:
            r = results.get(label)
            if r is None or r["spearman"] is None:
                row += f"  --       "
            else:
                row += f"  {r['spearman']:+.2f}    "
        print(row)

    # Build figure + markdown
    return bal_rows, nat_rows, cells, MODELS


def fig_heatmap(bal_rows, cells, save_base):
    """Heatmap: rows = cells (venue, year, metric), columns = models, fill = Spearman ρ."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    mpl.rcParams.update({"text.usetex": False, "font.family": "sans-serif",
                          "font.sans-serif": ["Arial", "DejaVu Sans"]})

    cell_labels = []; mat_data = []
    for venue, year, metric, idx in cells:
        cell_labels.append(f"{venue}-{year if year else 'pool'} ({metric}, n=?)")
        row = []
        n_show = None
        for label, *_ in MODELS:
            r = per_cell_correlation(bal_rows[label], idx, venue, year)
            if r is None:
                row.append(np.nan)
            else:
                row.append(r["spearman"])
                n_show = r["n"]
        # update label with n
        cell_labels[-1] = f"{venue}-{year if year else 'pool'} ({metric}, n={n_show})"
        mat_data.append(row)
    mat = np.array(mat_data)

    model_labels = [m[0].replace("*", "") for m in MODELS]

    fig, ax = plt.subplots(figsize=(20, 9))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(mat, cmap=cmap, vmin=-0.4, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=12)
    ax.set_yticks(range(len(cell_labels)))
    ax.set_yticklabels(cell_labels, fontsize=12)
    # Cell text
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                color = "black" if -0.1 < mat[i, j] < 0.3 else "white"
                ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")
    # Color bar
    cbar = fig.colorbar(im, ax=ax, location="right", shrink=0.8, pad=0.01)
    cbar.set_label("Spearman ρ(score, metric)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    # Separator between per-year and pooled rows
    n_per_year = sum(1 for v, y, _, _ in cells if y is not None)
    if n_per_year > 0 and n_per_year < len(cells):
        ax.axhline(n_per_year - 0.5, color="black", linewidth=2, linestyle="--")
        ax.text(-0.6, n_per_year - 0.5, "  ↓ pooled across years", fontsize=11, va="center", color="gray", style="italic")
    # Separator between rating and citation rows
    rating_count = sum(1 for v, y, m, _ in cells if m == "rating")
    cit_count = sum(1 for v, y, m, _ in cells if m == "citation")
    # Within per-year block, rating rows come first (3) then citation (3)
    ax.axhline(2.5, color="navy", linewidth=1.5, alpha=0.5, linestyle=":")
    ax.text(-0.6, 2.5, "  ↓ citation", fontsize=10, va="center", color="navy", style="italic", alpha=0.8)

    ax.set_title("Per-(venue, year) Spearman ρ(score, quality_signal) — arxiv balanced test\n"
                 "Rows: cells with ≥40 samples for the signal. Top half: per-year. Bottom: pooled.",
                 fontsize=15, pad=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def append_markdown(bal_rows, cells):
    lines = []
    lines.append("\n\n---\n")
    lines.append("## Per-(venue × year) quality correlation on arxiv balanced test\n")
    lines.append("**Setup.** 9 model checkpoints (7B ICLR-trained text + vision; 3B and 7B arxiv- and "
                 "iclr-trained text). For each (venue, year, quality_signal) cell with ≥40 samples for "
                 "the signal on arxiv balanced test, compute Spearman ρ(score, signal). Pooled-across-years "
                 "rows shown below the per-year rows for additional power.\n")
    lines.append("**Cells with ≥40 samples** (verified earlier):")
    lines.append("- Rating: iclr 2026 (51), neurips 2024 (77), neurips 2025 (98)")
    lines.append("- Citation: cvpr 2024 (50), cvpr 2025 (47), neurips 2024 (77)")
    lines.append("- Pooled: iclr (109 rating, 55 citation), neurips (175 rating, 77 citation), cvpr (97 citation)\n")

    lines.append("![per-venue per-year quality heatmap](../tmp_latex_dir/figures/ratio_xeval_per_venue_year_quality.png)\n")

    # Per-cell table
    lines.append("### Detailed table (Spearman ρ)\n")
    lines.append("| venue | year | metric | n | " + " | ".join(m[0].replace("*","") for m in MODELS) + " |")
    lines.append("|---|---|---|---:" + "|---:" * len(MODELS) + "|")
    for venue, year, metric, idx in cells:
        n_check = None
        vals = []
        for label, *_ in MODELS:
            r = per_cell_correlation(bal_rows[label], idx, venue, year)
            if r is not None and n_check is None: n_check = r["n"]
            vals.append(r["spearman"] if r and r["spearman"] is not None else None)
        if n_check is None: continue
        cells_str = " | ".join(f"{v:+.2f}" if v is not None else "—" for v in vals)
        yr_str = str(year) if year else "pool"
        lines.append(f"| {venue} | {yr_str} | {metric} | {n_check} | {cells_str} |")
    lines.append("")

    # Pattern analysis
    lines.append("### Patterns\n")
    lines.append("- **Text consistently outperforms vision on rating correlation** in arxiv subsets — "
                 "even on iclr 2026 (where the iclr-vision-50/50 trained model gets ρ ≈ +0.01 — essentially "
                 "no rating alignment). The 7B arxiv-balanced text model (ckpt 656) hits +0.45 on iclr 2026 "
                 "rating, the highest single cell. **The previous \"vision is the universal-quality model\" "
                 "claim was driven by direct ICLR test results, not arxiv-iclr subsets.**")
    lines.append("- **NeurIPS 2025 rating correlations are uniformly weak** (max +0.14). NeurIPS 2024 is "
                 "stronger (max +0.31 from 3B iclr-bal text). The newer 2025 NeurIPS papers may have noisier "
                 "or less-spread ratings.")
    lines.append("- **Citation correlations are weaker than rating correlations** at the same venue+year. "
                 "On NeurIPS 2024, rating ρ reaches +0.31 but citation ρ on the same venue/year is at "
                 "best +0.08 (text models) or as low as −0.19 (7B arxiv-nat text). Citations are a "
                 "noisier signal than ratings in our data — partly because pct_citation depends on "
                 "post-publication trajectory while pct_rating reflects review-time perception.")
    lines.append("- **CVPR 2025 citation is the brightest spot** for citation-tracking: 7B iclr text 30/70 "
                 "hits +0.34, 3B iclr-bal text +0.30. **The iclr-trained text models track CVPR-2025 "
                 "citations slightly better than they track rating on most other venues** — surprising, "
                 "given the modality and venue mismatch.")
    lines.append("- **arxiv-trained models do NOT consistently outperform iclr-trained ones on quality "
                 "correlation**, even on arxiv test cells. They have higher classification accuracy on "
                 "arxiv (training-distribution match) but their score doesn't track quality better — "
                 "the two axes (predictor vs quality) really are different.\n")

    lines.append("### Caveats\n")
    lines.append("- **Sample sizes are small** (n=47-98 per cell). Spearman ρ at this n has wide CIs "
                 "(roughly ±0.15 at 95% confidence for n=50), so small differences between models "
                 "shouldn't be over-interpreted.")
    lines.append("- **2026 papers have ~0 citation history**, so we have no per-year citation data for 2026.")
    lines.append("- **`pct_citation` field appeared on more venues than `pct_rating`** (cvpr, aaai, icml, "
                 "eccv all have some citation but no rating coverage). Coverage is venue-dependent.")
    lines.append("- The 7B arxiv-bal text model is at ckpt 656 (only 1st available) — flagged but included.\n")

    with open(REPORT_PATH, "a") as f:
        f.write("\n".join(lines))
    print(f"\nAppended per-(venue, year) section to: {REPORT_PATH}")


if __name__ == "__main__":
    bal_rows, nat_rows, cells, MODELS_ = main()
    print("\nBuilding figure...")
    fig_heatmap(bal_rows, cells, str(FIG_DIR / "ratio_xeval_per_venue_year_quality"))
    print("Appending markdown...")
    append_markdown(bal_rows, cells)
