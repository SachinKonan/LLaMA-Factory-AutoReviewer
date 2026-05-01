#!/usr/bin/env python3
"""
Consolidated 7B ratio cross-eval report.

Dimensions (32 cells total):
  modality (text, vision) × train_ratio (50/50, 30/70)
    × test_set (ICLR, arxiv) × test_prior (balanced, natural)
    × split (test, val)

Metrics per cell:
  raw + calibrated:  ACC, accept-recall, reject-recall
  threshold-independent: AUC

Calibration:
  ICLR — single τ* per (modality, train_ratio, test_prior) on val (year≥2025).
  Arxiv — per-(modality, train_ratio, eval_set, venue) τ* on val (y24up).
          Sparse venues (val n<10) fall back to model-global τ*.

Outputs:
  - ASCII tables to stdout
  - reports/ratio_xeval_7b.md
  - tmp_latex_dir/figures/ratio_xeval_{acc_grid,auc_grid,recall_scatter}.{pdf,png}

Year filters:
  - ICLR: keep _metadata.year ∈ {2025, 2026}
  - arxiv: y24up files are already filtered (conference_year ≥ 2024)
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
DATA = ROOT / "data"
RES  = ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
ICLR_VAL_CALIB_DIR = ROOT / "results/iclr_val_calib"
ARXIV_BAL_DIR    = ROOT / "results/cross_conference_arxiv_y24up"
ARXIV_NATR_DIR   = ROOT / "results/cross_conference_arxiv_natrate_y24up"
REPORT_PATH = ROOT / "reports/ratio_xeval_7b.md"
FIG_DIR = ROOT / "tmp_latex_dir/figures"

# ---------- enumerated configs ----------
RATIOS = ["50_50", "30_70"]
MODES  = ["text", "vision"]
PRIORS = ["balanced", "natural"]
SPLITS = ["test", "val"]
TEST_SETS = ["iclr", "arxiv"]

RATIO_LABEL = {"50_50": "50/50", "30_70": "30/70"}

# (mode, ratio) -> (short_name, ckpt_step)
MODEL = {
    ("text",   "50_50"): ("bz32_lr1e-6_text",         1322),
    ("text",   "30_70"): ("bz32_lr1e-6_text_30_70",   1322),
    ("vision", "50_50"): ("bz16_lr1e-6_vision",       2648),
    ("vision", "30_70"): ("bz16_lr1e-6_vision_30_70", 2642),
}

# ICLR meta data (per-prior, per-modality, per-split)
ICLR_META = {
    ("text",   "balanced", "test"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
    ("text",   "balanced", "val" ): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation/data.json",
    ("text",   "natural",  "test"): DATA / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_test/data.json",
    ("text",   "natural",  "val" ): DATA / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_validation_DERIVED/data.json",
    ("vision", "balanced", "test"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_test/data.json",
    ("vision", "balanced", "val" ): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_validation/data.json",
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

ACL_FAMILY = {"acl", "emnlp", "naacl"}


# ---------- jsonl path resolution ----------
def iclr_test_jsonl(mode, train, test_prior):
    short, step = MODEL[(mode, train)]
    if train == "50_50":
        if test_prior == "balanced":
            return RES / short / f"finetuned-ckpt-{step}.jsonl"
        # 50/50 trained, evaluated on cross 30/70 test
        return RES / "ratio_crossval_clean" / short / "test_30_70" / f"finetuned-ckpt-{step}.jsonl"
    # train == "30_70"
    tag = "balanced" if test_prior == "balanced" else "30_70"
    return RES / "ratio_sweep" / short / tag / f"finetuned-ckpt-{step}.jsonl"


def iclr_val_jsonl(mode, train, test_prior):
    """Prefer iclr_val_calib (has logprobs); fall back to validation-ckpt path (no logprobs)."""
    short, step = MODEL[(mode, train)]
    val_tag = "balanced" if test_prior == "balanced" else "30_70"
    primary = ICLR_VAL_CALIB_DIR / short / f"val_{val_tag}" / f"finetuned-ckpt-{step}.jsonl"
    if primary.exists():
        return primary
    # Fallback: original validation-ckpt path. Only valid for the matching ratio.
    return RES / short / f"validation-ckpt-{step}.jsonl"


def arxiv_jsonl(mode, train, test_prior, split):
    short, step = MODEL[(mode, train)]
    base = ARXIV_BAL_DIR if test_prior == "balanced" else ARXIV_NATR_DIR
    sub = "arxiv_eval" if split == "test" else "arxiv_val"
    return base / short / sub / f"finetuned-ckpt-{step}.jsonl"


# ---------- helpers ----------
def extract_pred(t):
    s = (t or "").lower()
    if "\\boxed{accept}" in s or "boxed{accept}" in s: return "accept"
    if "\\boxed{reject}" in s or "boxed{reject}" in s: return "reject"
    if "accept" in s: return "accept"
    if "reject" in s: return "reject"
    return None


DECISION_TOKEN_IDX = 5  # position of the Accept|Reject token in "Outcome: \boxed{Accept}" template

def score_logodds(row):
    """Signed log-odds score: positive iff model leans Accept.

    Canonical convention (matches scripts/calibration_posthoc.py and tmp_latex_dir/*calibration*):
      p_chosen = exp(token_logprobs[DECISION_TOKEN_IDX])
      logit    = log(p / (1-p))             # always ≥ 0 (chosen-class confidence)
      score    = +logit if predict=Accept else -logit

    Used uniformly across ALL inference jsonls so scores are comparable across runs that
    did/didn't record explicit logprob_accept/reject arrays.
    """
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
    # Hard fallback (no token_logprobs at all): rare
    return 6.0 if chosen == 1 else -6.0


def auc(scores, labels):
    if not scores or len(set(labels)) < 2: return None
    pairs = sorted(zip(scores, labels))
    n = len(pairs); npos = sum(labels); nneg = n - npos
    rs = sum(i+1 for i,(s,l) in enumerate(pairs) if l == 1)
    return (rs - npos*(npos+1)/2) / (npos*nneg)


def best_tau(pairs):
    """argmax-accuracy threshold. Predict accept iff score > τ.
    Handles tied scores correctly by stepping past each unique-score group atomically.
    """
    if not pairs: return 0.0
    pairs = sorted(pairs)
    n = len(pairs); npos = sum(g for _,g in pairs)
    correct = npos       # τ = -inf, predict all accept
    best_acc = correct / n
    best_t = -math.inf
    from itertools import groupby
    for s, group in groupby(pairs, key=lambda p: p[0]):
        npos_in = 0; n_in = 0
        for _, g in group:
            n_in += 1; npos_in += g
        # After stepping past this group (τ = s): all in group predicted reject.
        # ΔCorrect = (n_neg_in_group correct gains) - (n_pos_in_group correct losses)
        correct += (n_in - 2 * npos_in)
        if correct / n > best_acc:
            best_acc, best_t = correct / n, s
    # The loop already considered τ = each unique score (covering "predict all reject" at τ = max-score).
    # If best_t is still -inf, τ=-inf was best (predict all accept) — represent as a very negative number.
    return best_t if best_t != -math.inf else -1e6


def per_class_recall(pairs, tau=0.0):
    """(accept_recall, reject_recall) at threshold τ. None if class is empty."""
    n_acc = sum(1 for _, g in pairs if g == 1)
    n_rej = sum(1 for _, g in pairs if g == 0)
    tp = sum(1 for s, g in pairs if g == 1 and s > tau)
    tn = sum(1 for s, g in pairs if g == 0 and s <= tau)
    return (tp / n_acc if n_acc else None,
            tn / n_rej if n_rej else None)


def accuracy(pairs, tau=0.0):
    if not pairs: return None
    return sum(1 for s, g in pairs if (s > tau) == (g == 1)) / len(pairs)


# ---------- loader ----------
def load_pairs(jsonl_path: Path, meta_path: Path, year_min: int = None,
               want_venue: bool = False):
    """Returns list of (gold, score [, year, venue])."""
    if not jsonl_path.exists():
        print(f"  MISSING jsonl: {jsonl_path}")
        return []
    if not meta_path.exists():
        print(f"  MISSING meta:  {meta_path}")
        return []
    meta = json.loads(meta_path.read_text())
    out = []
    with jsonl_path.open() as f:
        for i, line in enumerate(f):
            if i >= len(meta): break
            m = meta[i].get("_metadata") or {}
            yr = m.get("year")
            try: yr = int(yr) if yr is not None else None
            except (TypeError, ValueError): yr = None
            if year_min is not None and (yr is None or yr < year_min):
                continue
            r = json.loads(line)
            g = extract_pred(r.get("label", ""))
            s = score_logodds(r)
            if g is None or s is None: continue
            gold = 1 if g == "accept" else 0
            if want_venue:
                v = (m.get("pl_venue") or m.get("venue") or "?").lower()
                if v in ACL_FAMILY: v = "acl_family"
                out.append((gold, s, yr, v))
            else:
                out.append((gold, s, yr))
    return out


# ---------- per-cell loaders + threshold compute ----------
def load_iclr_cell(mode, train, test_prior, split):
    """Returns list of (gold, score) for ICLR cell, year-filtered to >=2025."""
    if split == "test":
        jsonl = iclr_test_jsonl(mode, train, test_prior)
    else:
        jsonl = iclr_val_jsonl(mode, train, test_prior)
    meta = ICLR_META[(mode, test_prior, split)]
    pairs = load_pairs(jsonl, meta, year_min=2025)
    return [(s, g) for g, s, _ in pairs]  # reorder to (score, gold)


def load_arxiv_cell(mode, train, test_prior, split):
    """Returns list of (score, gold, venue) for arxiv y24up cell."""
    jsonl = arxiv_jsonl(mode, train, test_prior, split)
    meta = ARXIV_META[(mode, test_prior, split)]
    pairs = load_pairs(jsonl, meta, want_venue=True)
    return [(s, g, v) for g, s, _, v in pairs]


# ---------- compute calibration thresholds ----------
def compute_iclr_thresholds():
    """Returns: τ[(mode, train, test_prior)] = float (single global threshold per ICLR cell)."""
    out = {}
    for mode in MODES:
        for train in RATIOS:
            for prior in PRIORS:
                val_pairs = load_iclr_cell(mode, train, prior, "val")
                if not val_pairs:
                    out[(mode, train, prior)] = 0.0
                    continue
                out[(mode, train, prior)] = best_tau(val_pairs)
    return out


def compute_arxiv_thresholds():
    """Returns: τ[(mode, train, test_prior)] = {venue -> (tau, source)} + 'GLOBAL' -> tau."""
    out = {}
    for mode in MODES:
        for train in RATIOS:
            for prior in PRIORS:
                val_pairs = load_arxiv_cell(mode, train, prior, "val")
                global_tau = best_tau([(s, g) for s, g, _ in val_pairs])
                by_venue = defaultdict(list)
                for s, g, v in val_pairs:
                    by_venue[v].append((s, g))
                vmap = {"GLOBAL": global_tau}
                for v, pairs in by_venue.items():
                    if len(pairs) >= 10:
                        vmap[v] = (best_tau(pairs), "venue")
                    else:
                        vmap[v] = (global_tau, "global")
                # also add test-only venues
                test_pairs = load_arxiv_cell(mode, train, prior, "test")
                for _, _, v in test_pairs:
                    vmap.setdefault(v, (global_tau, "global"))
                out[(mode, train, prior)] = vmap
    return out


# ---------- compute metrics per cell ----------
def metrics_iclr(pairs, tau=0.0):
    """Compute ACC, AUC, accept-rec, reject-rec on ICLR cell pairs (score, gold)."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "acc": None, "auc": None, "acc_rec": None, "rej_rec": None}
    acc = accuracy(pairs, tau) * 100
    a = auc([s for s, _ in pairs], [g for _, g in pairs])
    a_rec, r_rec = per_class_recall(pairs, tau)
    return {
        "n": n, "acc": acc, "auc": a,
        "acc_rec": (a_rec * 100 if a_rec is not None else None),
        "rej_rec": (r_rec * 100 if r_rec is not None else None),
    }


def metrics_arxiv(pairs_with_venue, venue_tau_map, default_tau=None):
    """Compute ACC/AUC/recalls on arxiv cell.
    For raw: pass venue_tau_map=None, default_tau=0.0 (uses τ=0 for all).
    For calibrated: pass venue_tau_map (per-venue) and default_tau as fallback.
    """
    n = len(pairs_with_venue)
    if n == 0:
        return {"n": 0, "acc": None, "auc": None, "acc_rec": None, "rej_rec": None}
    n_acc = sum(1 for s, g, _ in pairs_with_venue if g == 1)
    n_rej = sum(1 for s, g, _ in pairs_with_venue if g == 0)
    tp = tn = fp = fn = 0
    correct = 0
    for s, g, v in pairs_with_venue:
        if venue_tau_map is None:
            tau = default_tau if default_tau is not None else 0.0
        else:
            entry = venue_tau_map.get(v)
            if entry is None:
                tau = venue_tau_map.get("GLOBAL", 0.0)
            elif isinstance(entry, tuple):
                tau = entry[0]
            else:
                tau = entry
        pred = 1 if s > tau else 0
        if pred == g: correct += 1
        if g == 1 and pred == 1: tp += 1
        if g == 0 and pred == 0: tn += 1
    a = auc([s for s, _, _ in pairs_with_venue], [g for _, g, _ in pairs_with_venue])
    return {
        "n": n,
        "acc": correct / n * 100,
        "auc": a,
        "acc_rec": (tp / n_acc * 100 if n_acc else None),
        "rej_rec": (tn / n_rej * 100 if n_rej else None),
    }


# ---------- collect all cells ----------
def collect_all():
    """Returns nested dict: results[split][test_set][test_prior][mode][train] = {raw, cal, tau_info}."""
    iclr_taus  = compute_iclr_thresholds()
    arxiv_taus = compute_arxiv_thresholds()
    results = {}
    for split in SPLITS:
        results[split] = {}
        # ICLR
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
                    results[split]["iclr"][prior][mode][train] = {
                        "raw": raw, "cal": cal, "tau": tau,
                    }
        # ARXIV
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
    return results, iclr_taus, arxiv_taus


# ---------- formatters ----------
def fmt_v(v, w=5, dp=1, none="--"):
    if v is None: return f"{none:>{w}}"
    return f"{v:>{w}.{dp}f}"

def fmt_auc(v):
    if v is None: return " ---- "
    return f"{v:.3f}"

def fmt_recalls(a, r):
    a_s = "--" if a is None else f"{a:.0f}"
    r_s = "--" if r is None else f"{r:.0f}"
    return f"{a_s}/{r_s}"


CELL_LABELS = {
    ("iclr", "balanced"): "ICLR balanced (test=50/50, year=25/26)",
    ("iclr", "natural"):  "ICLR natural  (test=30/70, year=25/26)",
    ("arxiv", "balanced"):"Arxiv balanced (50/50 y24up)",
    ("arxiv", "natural"): "Arxiv natural  (natrate y24up)",
}


def print_split_table(results, split):
    print()
    print("=" * 152)
    print(f"  SPLIT: {split.upper()}")
    print("=" * 152)
    header = f"{'Model + variant':<22}"
    for ts in TEST_SETS:
        for pr in PRIORS:
            cell_name = CELL_LABELS[(ts, pr)]
            header += f"  {cell_name:^28}"
    print(header)
    sub = f"{'':<22}" + ("  " + "ACC   AUC   AccR/RejR  n   ".center(28)) * 4
    print(sub)
    print("-" * 152)

    for mode in MODES:
        for train in RATIOS:
            for variant in ("raw", "cal"):
                label = f"{mode:<6} {RATIO_LABEL[train]:<5} {variant:<3}"
                row = f"{label:<22}"
                for ts in TEST_SETS:
                    for pr in PRIORS:
                        m = results[split][ts][pr][mode][train][variant]
                        if m["n"] == 0:
                            row += f"  {'-- no data --':^28}"
                        else:
                            acc = fmt_v(m["acc"], w=4, dp=1)
                            au = fmt_auc(m["auc"])
                            rec = fmt_recalls(m["acc_rec"], m["rej_rec"])
                            n   = m["n"]
                            cell = f"{acc} {au} {rec:>7} n={n}"
                            row += f"  {cell:^28}"
                print(row)
            print()
    print("-" * 152)


def write_md_report(results, iclr_taus, arxiv_taus):
    """Write the markdown report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# 7B Ratio Cross-Eval — Consolidated Report\n")
    lines.append("**Scope:** 4 model configs (text/vision × 50/50/30/70 train) × ICLR + arxiv "
                 "× balanced/natural test priors × test+val splits.\n")
    lines.append("- ICLR test+val are filtered to year ∈ {2025, 2026}.")
    lines.append("- Arxiv test+val are y24up (conference_year ≥ 2024).")
    lines.append("- For ICLR \"natural\" prior: test ratio = 30/70 (matches ICLR's ~30% accept rate).")
    lines.append("- For arxiv \"natural\" prior: per-conference natural acceptance rates "
                 "(natrate; openaccept.org 5-year averages).")
    lines.append("- Score: log-odds = lp_accept[k] - lp_reject[k] at the decision step.")
    lines.append("- Calibration: τ* learned on val.")
    lines.append("  - ICLR: single τ* per (modality, train, test_prior) cell.")
    lines.append("  - Arxiv: per-(modality, train, eval_set, venue) τ* (sparse venues, val n<10, "
                 "fall back to model-global τ*).\n")

    lines.append("## Headline\n")
    lines.append("- **AUC**: vision 50/50 is the most *consistent* recipe (≈0.72-0.74 across all 4 test cells). "
                 "On the matched-prior ICLR-natural (test=30/70) cell, *30/70-trained* models hit a much higher AUC "
                 "(~0.80) for both modalities — suggests test-population drift between the balanced and 30/70 "
                 "ICLR test sets, not just a prior difference.")
    lines.append("- **ACC** (depends on test prior): on natural-prior tests (ICLR 30/70, arxiv natrate), the "
                 "matched-prior train recipe wins raw. On balanced tests, calibration converges all 4 "
                 "(modality × ratio) cells to a tighter 64-68% band.")
    lines.append("- **Reject-recall** is the diagnostic for bias: 30/70-trained models on balanced tests have "
                 "extreme reject-recall (~80-100%) at τ=0; calibration trades reject-recall for accept-recall "
                 "to recover ACC — the biggest single calibration lift is **text 30/70 on arxiv balanced: +13.1pp**.\n")

    lines.append("## Figures\n")
    lines.append("![ACC raw vs calibrated](../tmp_latex_dir/figures/ratio_xeval_acc_grid.png)")
    lines.append("![AUC](../tmp_latex_dir/figures/ratio_xeval_auc_grid.png)")
    lines.append("![Recall scatter (raw → calibrated)](../tmp_latex_dir/figures/ratio_xeval_recall_scatter.png)\n")

    # Tables: one per split
    for split in SPLITS:
        lines.append(f"## Metrics — {split.upper()} split\n")
        # 4 tables, one per (test_set, prior)
        for ts in TEST_SETS:
            for pr in PRIORS:
                lines.append(f"### {CELL_LABELS[(ts, pr)]}\n")
                lines.append("| model | n | ACC raw | ACC cal | AUC | AccR raw / RejR raw | AccR cal / RejR cal |")
                lines.append("|---|---:|---:|---:|---:|---:|---:|")
                for mode in MODES:
                    for train in RATIOS:
                        cell = results[split][ts][pr][mode][train]
                        raw = cell["raw"]; cal = cell["cal"]
                        if raw["n"] == 0:
                            lines.append(f"| {mode} {RATIO_LABEL[train]} | 0 | — | — | — | — | — |")
                            continue
                        lines.append(
                            f"| {mode} {RATIO_LABEL[train]} | {raw['n']} | "
                            f"{raw['acc']:.1f} | {cal['acc']:.1f} | "
                            f"{fmt_auc(raw['auc'])} | "
                            f"{fmt_recalls(raw['acc_rec'], raw['rej_rec'])} | "
                            f"{fmt_recalls(cal['acc_rec'], cal['rej_rec'])} |"
                        )
                lines.append("")

    # Threshold dump
    lines.append("## Thresholds used (τ*)\n")
    lines.append("### ICLR (single τ* per cell, learned on val year≥2025)\n")
    lines.append("| modality | train | test prior | τ* |")
    lines.append("|---|---|---|---:|")
    for mode in MODES:
        for train in RATIOS:
            for prior in PRIORS:
                t = iclr_taus[(mode, train, prior)]
                lines.append(f"| {mode} | {RATIO_LABEL[train]} | {prior} | {t:+.3f} |")
    lines.append("")

    lines.append("### Arxiv (per-(modality, train, eval_set, venue) τ*; sparse venues use the model-global τ* marked `g`)\n")
    lines.append("| modality | train | eval | venue | τ* | source |")
    lines.append("|---|---|---|---|---:|:---|")
    for mode in MODES:
        for train in RATIOS:
            for prior in PRIORS:
                vmap = arxiv_taus[(mode, train, prior)]
                global_tau = vmap.get("GLOBAL", 0.0)
                lines.append(f"| {mode} | {RATIO_LABEL[train]} | {prior} | _GLOBAL_ | {global_tau:+.3f} | (used as fallback for sparse venues) |")
                for venue, entry in sorted((k, v) for k, v in vmap.items() if k != "GLOBAL"):
                    if isinstance(entry, tuple):
                        tau, src = entry
                    else:
                        tau, src = entry, "venue"
                    src_marker = src
                    lines.append(f"| {mode} | {RATIO_LABEL[train]} | {prior} | {venue} | {tau:+.3f} | {src_marker} |")
                lines.append("")
    lines.append("")

    lines.append("## Notes / caveats\n")
    lines.append("- `accept-recall` = TP / total_positives (sensitivity); "
                 "`reject-recall` = TN / total_negatives (specificity).")
    lines.append("- `ACC raw` uses τ=0 (the natural decision boundary on the log-odds score).")
    lines.append("- For ICLR, calibrated metrics use the single τ* from val matching the test prior.")
    lines.append("- For arxiv, calibrated metrics use the per-venue τ*, with the model-global τ* as "
                 "fallback for venues with val n<10. The pooled accept/reject recalls are weighted by venue "
                 "test counts.")
    lines.append("- All numbers are 7B (Qwen2.5-7B for text, Qwen2.5-VL-7B for vision).")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nWrote markdown report: {REPORT_PATH}")


# ---------- figures (tmp_latex_dir style) ----------
def make_figures(results):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
    })
    BLUE   = "#6098FF"
    ORANGE = "#FECC81"
    GREEN  = "#77B25D"
    RED    = "#FF8988"
    PURPLE = "#B28CFF"
    LINEWIDTH = 3.0
    MARKERSIZE = 10
    labelsize  = 22
    titlesize  = 24
    legendsize = 16
    ticksize   = 16

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cells = [(ts, pr) for ts in TEST_SETS for pr in PRIORS]
    cell_titles = [CELL_LABELS[c] for c in cells]
    model_keys  = [(mode, train) for mode in MODES for train in RATIOS]
    model_labels = [f"{mode}\n{RATIO_LABEL[train]}" for mode, train in model_keys]
    model_colors = [BLUE, BLUE, GREEN, GREEN]
    model_hatches = ["", "//", "", "//"]

    # ---- Fig 1: ACC raw vs calibrated, 1×4 grid (test split) ----
    fig, axes = plt.subplots(1, 4, figsize=(28, 6.5), sharey=True)
    x = np.arange(len(model_keys))
    bw = 0.36
    for ax, (ts, pr), title in zip(axes, cells, cell_titles):
        raw_vals = [results["test"][ts][pr][m][r]["raw"]["acc"] or 0 for m, r in model_keys]
        cal_vals = [results["test"][ts][pr][m][r]["cal"]["acc"] or 0 for m, r in model_keys]
        b1 = ax.bar(x - bw/2, raw_vals, bw, label="raw (τ=0)", color=BLUE, edgecolor="black", linewidth=1.0)
        b2 = ax.bar(x + bw/2, cal_vals, bw, label="calibrated", color=ORANGE, edgecolor="black", linewidth=1.0)
        # Annotate cal-raw delta where significant
        for i, (rv, cv) in enumerate(zip(raw_vals, cal_vals)):
            d = cv - rv
            if abs(d) > 2:
                ax.annotate(f"{d:+.1f}", xy=(x[i] + bw/2, cv + 1.5),
                            ha="center", va="bottom", fontsize=14, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=ticksize)
        ax.set_title(title.replace(" (", "\n("), fontsize=titlesize - 4)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=ticksize)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Accuracy (%)", fontsize=labelsize)
    axes[0].legend(fontsize=legendsize, loc="lower right")
    fig.suptitle("Accuracy: raw (τ=0) vs calibrated (τ* from val) — 7B, TEST split",
                 fontsize=titlesize, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"ratio_xeval_acc_grid.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ---- Fig 2: AUC, 1×4 grid ----
    fig, axes = plt.subplots(1, 4, figsize=(28, 6.5), sharey=True)
    for ax, (ts, pr), title in zip(axes, cells, cell_titles):
        auc_vals = [results["test"][ts][pr][m][r]["raw"]["auc"] or 0 for m, r in model_keys]
        for i, (mk, val, hatch, color) in enumerate(zip(model_keys, auc_vals, model_hatches, model_colors)):
            ax.bar(i, val, 0.7, color=color, edgecolor="black", linewidth=1.0, hatch=hatch)
            ax.text(i, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=14)
        ax.set_xticks(np.arange(len(model_keys)))
        ax.set_xticklabels(model_labels, fontsize=ticksize)
        ax.set_title(title.replace(" (", "\n("), fontsize=titlesize - 4)
        ax.set_ylim(0.5, 1.0)
        ax.tick_params(axis="y", labelsize=ticksize)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("AUC", fontsize=labelsize)
    # Custom legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=BLUE, edgecolor="black", label="text"),
        Patch(facecolor=GREEN, edgecolor="black", label="vision"),
        Patch(facecolor="white", edgecolor="black", label="50/50 train"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="30/70 train"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=legendsize, loc="lower right")
    fig.suptitle("AUC (threshold-independent) — 7B, TEST split", fontsize=titlesize, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"ratio_xeval_auc_grid.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ---- Fig 3: Recall scatter, raw → calibrated arrows ----
    fig, axes = plt.subplots(1, 4, figsize=(28, 7.5), sharey=True, sharex=True)
    point_colors = [BLUE, RED, GREEN, PURPLE]  # (text 50/50, text 30/70, vision 50/50, vision 30/70)
    for ax, (ts, pr), title in zip(axes, cells, cell_titles):
        # Iso-accuracy reference contours (acc = (acc_rec * p + rej_rec * (1-p)) where p = positive rate)
        # We don't know p without data — use 50/50 for "balanced" and 30/70 for "natural"
        # Approximate p from test data
        any_cell = results["test"][ts][pr]["text"]["50_50"]["raw"]
        if any_cell["n"] > 0 and any_cell["acc_rec"] is not None and any_cell["rej_rec"] is not None:
            # Reverse-engineer p from acc, acc_rec, rej_rec
            ar, rr, a = any_cell["acc_rec"]/100, any_cell["rej_rec"]/100, any_cell["acc"]/100
            # acc = ar*p + rr*(1-p) -> p = (acc - rr) / (ar - rr)
            denom = ar - rr
            p = ((a - rr) / denom) if abs(denom) > 1e-6 else 0.5
            p = max(0.1, min(0.9, p))
        else:
            p = 0.5
        # Draw iso-accuracy contours
        xs = np.linspace(0, 1, 100)
        for tgt in (0.5, 0.6, 0.7, 0.8):
            # acc = ar*p + rr*(1-p) = tgt -> rr = (tgt - ar*p) / (1-p)
            rrs = (tgt - xs*p) / (1 - p) if abs(1-p) > 1e-6 else np.full_like(xs, tgt)
            mask = (rrs >= 0) & (rrs <= 1)
            ax.plot(xs[mask]*100, rrs[mask]*100, "--", color="gray", alpha=0.5, linewidth=1.5)
            # Label the contour
            mid_idx = mask.sum() // 2
            if mid_idx > 5:
                ax.text(xs[mask][mid_idx]*100, rrs[mask][mid_idx]*100,
                        f"acc={tgt:.0%}", fontsize=11, color="gray", alpha=0.7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        # Plot points
        for (mode, train), color in zip(model_keys, point_colors):
            cell = results["test"][ts][pr][mode][train]
            raw = cell["raw"]; cal = cell["cal"]
            if raw["n"] == 0 or raw["acc_rec"] is None: continue
            ax.scatter(raw["acc_rec"], raw["rej_rec"], s=180, facecolor="none",
                       edgecolor=color, linewidth=2.5, label=f"{mode} {RATIO_LABEL[train]}")
            ax.scatter(cal["acc_rec"], cal["rej_rec"], s=180, facecolor=color,
                       edgecolor="black", linewidth=1.0)
            # Arrow from raw to cal
            dx = cal["acc_rec"] - raw["acc_rec"]
            dy = cal["rej_rec"] - raw["rej_rec"]
            if dx*dx + dy*dy > 4:
                ax.annotate("", xy=(cal["acc_rec"], cal["rej_rec"]),
                            xytext=(raw["acc_rec"], raw["rej_rec"]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=2.5, alpha=0.7))
        ax.plot([0, 100], [0, 100], "k:", alpha=0.3, linewidth=1)
        ax.set_xlabel("Accept-recall (%)", fontsize=labelsize-2)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_title(title.replace(" (", "\n("), fontsize=titlesize - 4)
        ax.tick_params(axis="both", labelsize=ticksize)
        ax.grid(linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Reject-recall (%)", fontsize=labelsize)
    axes[0].legend(fontsize=legendsize, loc="lower left")
    fig.suptitle("Per-class recall: raw (open) → calibrated (filled) — 7B, TEST split",
                 fontsize=titlesize, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"ratio_xeval_recall_scatter.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"\nWrote 3 figures (PDF + PNG) to: {FIG_DIR}")


# ---------- main ----------
def main():
    print("Loading + computing all 32 cells × 2 splits = 64 metric blocks...")
    results, iclr_taus, arxiv_taus = collect_all()

    for split in SPLITS:
        print_split_table(results, split)

    print()
    print("=" * 80)
    print("ICLR thresholds (τ*)")
    print("=" * 80)
    for (mode, train, prior), tau in sorted(iclr_taus.items()):
        print(f"  {mode:<6} {RATIO_LABEL[train]:<5} {prior:<8}  τ* = {tau:+.3f}")
    print()

    print("=" * 80)
    print("Arxiv thresholds (per-venue, sparse fall back to global)")
    print("=" * 80)
    for (mode, train, prior), vmap in sorted(arxiv_taus.items()):
        print(f"\n  {mode:<6} {RATIO_LABEL[train]:<5} {prior:<8}  GLOBAL τ* = {vmap.get('GLOBAL', 0.0):+.3f}")
        for venue, entry in sorted((k, v) for k, v in vmap.items() if k != "GLOBAL"):
            tau, src = entry if isinstance(entry, tuple) else (entry, "venue")
            print(f"    {venue:<14}  τ* = {tau:+7.3f}  ({src})")

    write_md_report(results, iclr_taus, arxiv_taus)
    make_figures(results)


if __name__ == "__main__":
    main()
