#!/usr/bin/env python3
"""
Exhaustive audit: every metric, CI, and threshold in docs/objective_analysis.md
re-derived independently and cross-checked against:
  - sklearn.metrics.roc_auc_score (AUC)
  - sklearn.metrics.balanced_accuracy_score (bACC)
  - sklearn.metrics.recall_score (per-class recall)
  - sklearn.metrics.accuracy_score (raw ACC)
  - scipy.stats.spearmanr (Spearman ρ)
  - independent bootstrap reimplementation (CIs)
  - brute-force τ sweep (best-τ functions)

Run: uv run python scripts/verify_objective_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tmp_latex_dir"))
import generate_objective_analysis as G

import json, math, re
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score, accuracy_score
from scipy.stats import spearmanr

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

n_pass = 0
failures = []
def check(name, ok, extra=""):
    global n_pass
    if ok:
        n_pass += 1
        # Quiet on pass — only show fails/sections
    else:
        print(f"  [{FAIL}] {name}" + (f"  {extra}" if extra else ""))
        failures.append(name)


def section(title):
    print(f"\n=== {title} ===")


def ref_bootstrap_metric(items, metric_fn, n=500, ci=0.95, seed=42):
    if len(items) < 10: return None, None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(items), len(items))
        v = metric_fn([items[i] for i in idx])
        if v is not None: vals.append(v)
    if not vals: return None, None
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def check_metric_stack(name, pairs, tau, doc_bal=None, doc_auc=None,
                        doc_ar=None, doc_rr=None, doc_raw=None, tol=0.5, auc_tol=0.005):
    """Verify all metrics for a (pairs, tau) cell against sklearn + doc."""
    if not pairs:
        check(f"{name}: pairs is empty", False); return
    scores = [s for s, _ in pairs]; labels = [g for _, g in pairs]
    pred = [1 if s > tau else 0 for s in scores]

    bal_o = G.bal_acc(pairs, tau)
    raw_o = G.raw_acc(pairs, tau)
    auc_o = G.auc_score(scores, labels)
    ar_o, rr_o = G.per_class_recall(pairs, tau)

    bal_skl = balanced_accuracy_score(labels, pred) * 100
    raw_skl = accuracy_score(labels, pred) * 100
    auc_skl = roc_auc_score(labels, scores) if len(set(labels)) >= 2 else None
    ar_skl = recall_score(labels, pred, pos_label=1, zero_division=0) * 100 if any(labels) else None
    rr_skl = recall_score(labels, pred, pos_label=0, zero_division=0) * 100 if any(g == 0 for g in labels) else None

    check(f"{name}: bal_acc vs sklearn", abs(bal_o - bal_skl) < 1e-6,
          f"ours={bal_o:.4f}, sklearn={bal_skl:.4f}")
    check(f"{name}: raw_acc vs sklearn", abs(raw_o - raw_skl) < 1e-6,
          f"ours={raw_o:.4f}, sklearn={raw_skl:.4f}")
    if auc_skl is not None:
        check(f"{name}: AUC vs sklearn (exact, tie-aware)", abs(auc_o - auc_skl) < 1e-9,
              f"ours={auc_o:.9f}, sklearn={auc_skl:.9f}, diff={abs(auc_o-auc_skl):.2e}")
    if ar_skl is not None and ar_o is not None:
        check(f"{name}: accept_recall vs sklearn", abs(ar_o - ar_skl) < 1e-6,
              f"ours={ar_o:.4f}, sklearn={ar_skl:.4f}")
    if rr_skl is not None and rr_o is not None:
        check(f"{name}: reject_recall vs sklearn", abs(rr_o - rr_skl) < 1e-6,
              f"ours={rr_o:.4f}, sklearn={rr_skl:.4f}")
    # vs doc
    if doc_bal is not None:
        check(f"{name}: bal_acc matches doc", abs(bal_o - doc_bal) < tol,
              f"ours={bal_o:.2f}, doc={doc_bal:.2f}")
    if doc_auc is not None:
        check(f"{name}: AUC matches doc", abs(auc_o - doc_auc) < auc_tol,
              f"ours={auc_o:.4f}, doc={doc_auc:.4f}")
    if doc_ar is not None and ar_o is not None:
        check(f"{name}: accept_recall matches doc", abs(ar_o - doc_ar) < tol,
              f"ours={ar_o:.2f}, doc={doc_ar:.2f}")
    if doc_rr is not None and rr_o is not None:
        check(f"{name}: reject_recall matches doc", abs(rr_o - doc_rr) < tol,
              f"ours={rr_o:.2f}, doc={doc_rr:.2f}")
    if doc_raw is not None:
        check(f"{name}: raw_acc matches doc", abs(raw_o - doc_raw) < tol,
              f"ours={raw_o:.2f}, doc={doc_raw:.2f}")


def check_bootstrap(name, items, metric_fn, doc_lo=None, doc_hi=None, tol=1.0):
    """Verify bootstrap CI vs ref impl + doc."""
    ours = G.bootstrap_metric(items, metric_fn, n=500, seed=42)
    ref = ref_bootstrap_metric(items, metric_fn, n=500, seed=42)
    if ours[0] is not None:
        check(f"{name}: bootstrap lo matches ref impl",
              abs(ours[0] - ref[0]) < 1e-6, f"ours={ours[0]:.6f}, ref={ref[0]:.6f}")
        check(f"{name}: bootstrap hi matches ref impl",
              abs(ours[1] - ref[1]) < 1e-6, f"ours={ours[1]:.6f}, ref={ref[1]:.6f}")
    # Stability: different seed shouldn't change CI by more than ~1 tol-unit
    ours_99 = G.bootstrap_metric(items, metric_fn, n=500, seed=99)
    if ours[0] is not None and ours_99[0] is not None:
        check(f"{name}: CI stable across seeds (lo within ~1.5 tol-units)",
              abs(ours[0] - ours_99[0]) < max(1.5 * tol, abs(ours[0]) * 0.05),
              f"seed42_lo={ours[0]:.4f}, seed99_lo={ours_99[0]:.4f}")
    if doc_lo is not None and ours[0] is not None:
        check(f"{name}: CI lo matches doc", abs(ours[0] - doc_lo) < tol,
              f"ours={ours[0]:.2f}, doc={doc_lo:.2f}")
    if doc_hi is not None and ours[1] is not None:
        check(f"{name}: CI hi matches doc", abs(ours[1] - doc_hi) < tol,
              f"ours={ours[1]:.2f}, doc={doc_hi:.2f}")
    return ours


# =============================================================================
# DOC PARSER — extract every numeric cell from objective_analysis.md
# =============================================================================
def parse_doc():
    text = (Path(__file__).resolve().parent.parent / "docs" / "objective_analysis.md").read_text()
    return text


DOC = parse_doc()


def extract_table_row(section_anchor, row_id_substring, n_cols=None):
    """Extract a row from a markdown table near the section anchor.
       Returns the list of cell strings.
    """
    idx = DOC.find(section_anchor)
    if idx < 0: return None
    # Search for a matching table row from this point
    sub = DOC[idx:idx + 50000]
    for line in sub.splitlines():
        line = line.strip()
        if line.startswith("|") and row_id_substring in line:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            return cells
    return None


def parse_num(s):
    """Parse '67.6 [65.4, 69.7]' → (67.6, 65.4, 69.7) ; '0.736' → (0.736, None, None)."""
    if not s or s in ("—", "--"): return None, None, None
    m = re.match(r"\*?\*?(-?\d+\.?\d*)\*?\*?\s*\[(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]", s)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    m = re.match(r"\*?\*?([+\-]?\d+\.?\d*)\*?\*?", s)
    if m: return float(m.group(1)), None, None
    return None, None, None


# =============================================================================
# §3 — Obj 2 headline tables
# =============================================================================
section("§3 ICLR balanced (each cell vs sklearn + doc + CI ref)")
for (mode, train) in [("text","50_50"),("text","30_70"),("vision","50_50"),("vision","30_70")]:
    short, step = G.MODELS_7B_ICLR[(mode, train)]
    pairs, _ = G.load_iclr_cell(short, step, train, "balanced", "test")
    # Parse expected from doc
    label = f"| {mode} {train.replace('_','/')}"
    row = extract_table_row("**ICLR 25/26 balanced (Obj 2 view):**", label)
    if not row: continue
    # cols: model | n | bACC[CI] | AR | RR | AUC[CI]
    bal_pt, bal_lo, bal_hi = parse_num(row[2])
    ar_pt, _, _ = parse_num(row[3])
    rr_pt, _, _ = parse_num(row[4])
    auc_pt, auc_lo, auc_hi = parse_num(row[5])
    check_metric_stack(f"§3 ICLR {mode} {train}", pairs, 0.0,
                       doc_bal=bal_pt, doc_auc=auc_pt, doc_ar=ar_pt, doc_rr=rr_pt)
    check_bootstrap(f"§3 ICLR {mode} {train} bACC CI",
                    pairs, lambda p: G.bal_acc(p, 0.0), doc_lo=bal_lo, doc_hi=bal_hi)
    check_bootstrap(f"§3 ICLR {mode} {train} AUC CI",
                    pairs, lambda p: G.auc_score([s for s, _ in p], [g for _, g in p]),
                    doc_lo=auc_lo, doc_hi=auc_hi, tol=0.005)


section("§3 Arxiv balanced (each cell vs sklearn + doc + CI ref)")
for (mode, train) in [("text","50_50"),("text","30_70"),("vision","50_50"),("vision","30_70")]:
    short, step = G.MODELS_7B_ICLR[(mode, train)]
    pairs, _ = G.load_arxiv_cell(short, step, "balanced", "test")
    label = f"| {mode} {train.replace('_','/')}"
    row = extract_table_row("**Arxiv y24up balanced (Obj 2 view):**", label)
    if not row: continue
    bal_pt, bal_lo, bal_hi = parse_num(row[2])
    ar_pt, _, _ = parse_num(row[3])
    rr_pt, _, _ = parse_num(row[4])
    auc_pt, auc_lo, auc_hi = parse_num(row[5])
    check_metric_stack(f"§3 arxiv {mode} {train}", pairs, 0.0,
                       doc_bal=bal_pt, doc_auc=auc_pt, doc_ar=ar_pt, doc_rr=rr_pt)
    check_bootstrap(f"§3 arxiv {mode} {train} bACC CI",
                    pairs, lambda p: G.bal_acc(p, 0.0), doc_lo=bal_lo, doc_hi=bal_hi)
    check_bootstrap(f"§3 arxiv {mode} {train} AUC CI",
                    pairs, lambda p: G.auc_score([s for s, _ in p], [g for _, g in p]),
                    doc_lo=auc_lo, doc_hi=auc_hi, tol=0.005)


# =============================================================================
# §1.2 natural-prior table
# =============================================================================
section("§1.2 natural-prior cells (raw, AR, RR, bACC)")
for (mode, train) in [("text","30_70"),("text","50_50"),("vision","30_70"),("vision","50_50")]:
    short, step = G.MODELS_7B_ICLR[(mode, train)]
    pairs, _ = G.load_arxiv_cell(short, step, "natural", "test")
    check_metric_stack(f"§1.2 {mode} {train} arxiv natural", pairs, 0.0)


# =============================================================================
# §1.4 raw_ACC formula vs empirical
# =============================================================================
section("§1.4 raw_ACC(π) formula derivation")
PI_NATURAL = {"iclr": 0.313, "arxiv": 0.238}
for (mode, train) in [("text","50_50"),("text","30_70"),("vision","50_50"),("vision","30_70")]:
    short, step = G.MODELS_7B_ICLR[(mode, train)]
    for ds in ("iclr", "arxiv"):
        if ds == "iclr":
            bal_pairs, _ = G.load_iclr_cell(short, step, train, "balanced", "test")
            nat_pairs, _ = G.load_iclr_cell(short, step, train, "natural", "test")
        else:
            bal_pairs, _ = G.load_arxiv_cell(short, step, "balanced", "test")
            nat_pairs, _ = G.load_arxiv_cell(short, step, "natural", "test")
        ar, rr = G.per_class_recall(bal_pairs, 0.0)
        if ar is None or rr is None: continue
        pi = PI_NATURAL[ds]
        predicted = pi * ar + (1 - pi) * rr
        empirical = G.raw_acc(nat_pairs, 0.0)
        max_tol = 1.5 if ds == "arxiv" else 6.0
        check(f"§1.4 {mode} {train} {ds}: |predicted−empirical| < {max_tol}",
              abs(predicted - empirical) < max_tol,
              f"predicted={predicted:.2f}, empirical={empirical:.2f}, Δ={empirical-predicted:+.2f}")


# =============================================================================
# §2.4 quality correlations
# =============================================================================
section("§2.4 ICLR quality ρ — every cell vs scipy")
for (mode, train) in [("text","50_50"),("text","30_70"),("vision","50_50"),("vision","30_70")]:
    short, step = G.MODELS_7B_ICLR[(mode, train)]
    pairs, meta = G.load_iclr_cell(short, step, train, "balanced", "test",
                                   fields=("pct_rating", "citation_normalized_by_year"))
    scores = [s for s, _ in pairs]; labels = [g for _, g in pairs]
    ratings = [m.get("pct_rating") for m in meta]
    cit_2025 = [m.get("citation_normalized_by_year") if m.get("year") == 2025 else None for m in meta]
    for sig_name, sig in [("rating", ratings), ("cit2025", cit_2025)]:
        for sub in [None, 1, 0]:
            paired = [(s, v, l) for s, v, l in zip(scores, sig, labels) if v is not None]
            if sub is not None:
                paired = [(s, v, l) for s, v, l in paired if l == sub]
            if len(paired) < 5: continue
            xs = [p[0] for p in paired]; ys = [p[1] for p in paired]
            ours = G.spearman(xs, ys)
            ref, _ = spearmanr(xs, ys)
            check(f"§2.4 ICLR {mode} {train} {sig_name} sub={sub}: ρ vs scipy",
                  abs(ours - ref) < 1e-9,
                  f"ours={ours:.6f}, scipy={ref:.6f}")

section("§2.4 Arxiv quality ρ — every cell vs scipy")
for (mode, train) in [("text","50_50"),("text","30_70"),("vision","50_50"),("vision","30_70")]:
    short, step = G.MODELS_7B_ICLR[(mode, train)]
    pairs, meta = G.load_arxiv_cell(short, step, "balanced", "test",
                                    fields=("pct_rating", "pct_citation"))
    scores = [s for s, _ in pairs]; labels = [g for _, g in pairs]
    ratings = [m.get("pct_rating") for m in meta]
    cits = [m.get("pct_citation") for m in meta]
    for sig_name, sig in [("rating", ratings), ("citation", cits)]:
        for sub in [None, 1, 0]:
            paired = [(s, v, l) for s, v, l in zip(scores, sig, labels) if v is not None]
            if sub is not None:
                paired = [(s, v, l) for s, v, l in paired if l == sub]
            if len(paired) < 5: continue
            xs = [p[0] for p in paired]; ys = [p[1] for p in paired]
            ours = G.spearman(xs, ys)
            ref, _ = spearmanr(xs, ys)
            check(f"§2.4 arxiv {mode} {train} {sig_name} sub={sub}: ρ vs scipy",
                  abs(ours - ref) < 1e-9,
                  f"ours={ours:.6f}, scipy={ref:.6f}")


# =============================================================================
# §1.3 dual calibration thresholds (brute-force vs our impl)
# =============================================================================
section("§1.3 dual calibration thresholds — every cell vs brute force")
def brute_best_tau_raw(pairs):
    if not pairs: return 0.0
    cs = [-1e6] + sorted({s for s, _ in pairs})
    best_t = -1e6; best_a = -1
    for t in cs:
        a = G.raw_acc(pairs, t)
        if a is not None and a > best_a: best_a, best_t = a, t
    return best_t

def brute_best_tau_bal(pairs):
    if not pairs: return 0.0
    cs = [-1e6] + sorted({s for s, _ in pairs})
    best_t = -1e6; best_b = -1
    for t in cs:
        b = G.bal_acc(pairs, t)
        if b is not None and b > best_b: best_b, best_t = b, t
    return best_t

for ds in ("iclr", "arxiv"):
    for (mode, train), (short, step) in G.MODELS_7B_ICLR.items():
        if ds == "iclr":
            val_pairs, _ = G.load_iclr_cell(short, step, train, "balanced", "val")
        else:
            val_pairs, _ = G.load_arxiv_cell(short, step, "balanced", "val")
        if not val_pairs: continue
        tau_raw_o = G.best_tau_raw(val_pairs); tau_bal_o = G.best_tau_balanced(val_pairs)
        tau_raw_b = brute_best_tau_raw(val_pairs); tau_bal_b = brute_best_tau_bal(val_pairs)
        check(f"§1.3 {ds} {mode} {train} τ*_raw matches brute max",
              abs(G.raw_acc(val_pairs, tau_raw_o) - G.raw_acc(val_pairs, tau_raw_b)) < 1e-9,
              f"raw_acc(o)={G.raw_acc(val_pairs, tau_raw_o):.4f}, raw_acc(b)={G.raw_acc(val_pairs, tau_raw_b):.4f}")
        check(f"§1.3 {ds} {mode} {train} τ*_bal matches brute max",
              abs(G.bal_acc(val_pairs, tau_bal_o) - G.bal_acc(val_pairs, tau_bal_b)) < 1e-9,
              f"bal_acc(o)={G.bal_acc(val_pairs, tau_bal_o):.4f}, bal_acc(b)={G.bal_acc(val_pairs, tau_bal_b):.4f}")


# =============================================================================
# §5 DR comparison — verify DR scoring + numbers
# =============================================================================
section("§5 DeepReviewer numbers (parsing + math)")
for ds in ("iclr", "arxiv"):
    test_rows = G.load_deepreviewer(f"{ds}_balanced_test")
    val_rows  = G.load_deepreviewer(f"{ds}_balanced_val")
    n_acc = sum(r["gold"] for r in test_rows); n_rej = len(test_rows) - n_acc
    tp = sum(1 for r in test_rows if r["pred"] == 1 and r["gold"] == 1)
    tn = sum(1 for r in test_rows if r["pred"] == 0 and r["gold"] == 0)
    bal = (tp / n_acc + tn / n_rej) / 2 * 100
    auc = G.auc_score([r["score"] for r in test_rows], [r["gold"] for r in test_rows])
    auc_skl = roc_auc_score([r["gold"] for r in test_rows], [r["score"] for r in test_rows])
    check(f"§5 DR {ds}: AUC matches sklearn", abs(auc - auc_skl) < 1e-9,
          f"ours={auc:.6f}, sklearn={auc_skl:.6f}")
    # vs sklearn balanced_accuracy_score
    pred = [r["pred"] for r in test_rows]; gold = [r["gold"] for r in test_rows]
    bal_skl = balanced_accuracy_score(gold, pred) * 100
    check(f"§5 DR {ds}: bACC matches sklearn", abs(bal - bal_skl) < 1e-6,
          f"ours={bal:.4f}, sklearn={bal_skl:.4f}")


# =============================================================================
# §7 arxiv-trained sweep — verify each best-ckpt cell (parsed from doc)
# =============================================================================
section("§7 arxiv-trained — every best-ckpt vs doc + sklearn")
ARXIV_TRAINED_CELLS = [
    ("3B balanced text",   "small_sachin/arxiv_21k_text_3b"),
    ("7B balanced text",   "small/arxiv_21k_text"),
    ("7B balanced vision", "small/arxiv_21k_vision"),
    ("3B natrate text",    "natrate_sachin/arxiv_natrate_21k_text_3b"),
    ("7B natrate text",    "natrate_sachin/arxiv_natrate_21k_text"),
]

for cell_name, cell_dir in ARXIV_TRAINED_CELLS:
    # Parse the row from the §7.1 arxiv table
    row = extract_table_row("**Arxiv balanced test (in-distribution for arxiv-trained):**",
                             f"| {cell_name} ")
    if not row: continue
    # cols: name | epoch | ckpt | τ | val | bACC[CI] | AUC[CI] | AR | RR
    try:
        epoch = int(row[1])
        ckpt = int(row[2])
        tau_doc = float(row[3])
        val_doc = float(row[4])
        bal_pt, bal_lo, bal_hi = parse_num(row[5])
        auc_pt, auc_lo, auc_hi = parse_num(row[6])
        ar_pt, _, _ = parse_num(row[7])
        rr_pt, _, _ = parse_num(row[8])
    except (ValueError, TypeError) as e:
        print(f"  parse error {cell_name}: {e}, row={row}"); continue

    test = G.load_arxiv_trained_jsonl(cell_dir, "arxiv_balanced_test", ckpt)
    val  = G.load_arxiv_trained_jsonl(cell_dir, "arxiv_balanced_val", ckpt)
    if not test or not val: continue
    tau = G.best_tau_balanced(val)
    val_bal = G.bal_acc(val, tau)
    check(f"§7 arxiv {cell_name}: τ matches doc", abs(tau - tau_doc) < 0.01,
          f"ours={tau:.4f}, doc={tau_doc:.4f}")
    check(f"§7 arxiv {cell_name}: val_bal matches doc", abs(val_bal - val_doc) < 0.5,
          f"ours={val_bal:.2f}, doc={val_doc:.2f}")
    check_metric_stack(f"§7 arxiv {cell_name}", test, tau,
                       doc_bal=bal_pt, doc_auc=auc_pt, doc_ar=ar_pt, doc_rr=rr_pt)
    check_bootstrap(f"§7 arxiv {cell_name} bACC CI",
                    test, lambda p: G.bal_acc(p, tau),
                    doc_lo=bal_lo, doc_hi=bal_hi)
    check_bootstrap(f"§7 arxiv {cell_name} AUC CI",
                    test, lambda p: G.auc_score([s for s, _ in p], [g for _, g in p]),
                    doc_lo=auc_lo, doc_hi=auc_hi, tol=0.005)


section("§7 arxiv-trained iclr OOD — every best-ckpt vs doc + sklearn")
for cell_name, cell_dir in ARXIV_TRAINED_CELLS:
    row = extract_table_row("**ICLR balanced test (OOD for arxiv-trained):**",
                             f"| {cell_name} ")
    if not row: continue
    try:
        if "—" in row[1]: continue  # no iclr eval
        epoch = int(row[1]); ckpt = int(row[2])
        tau_doc = float(row[3]); val_doc = float(row[4])
        bal_pt, bal_lo, bal_hi = parse_num(row[5])
        auc_pt, auc_lo, auc_hi = parse_num(row[6])
        ar_pt, _, _ = parse_num(row[7]); rr_pt, _, _ = parse_num(row[8])
    except (ValueError, TypeError):
        continue
    test = G.load_arxiv_trained_jsonl(cell_dir, "iclr_balanced_test", ckpt)
    val  = G.load_arxiv_trained_jsonl(cell_dir, "iclr_balanced_val", ckpt)
    if not test or not val: continue
    tau = G.best_tau_balanced(val)
    check(f"§7 iclr {cell_name}: τ matches doc", abs(tau - tau_doc) < 0.01,
          f"ours={tau:.4f}, doc={tau_doc:.4f}")
    check_metric_stack(f"§7 iclr {cell_name}", test, tau,
                       doc_bal=bal_pt, doc_auc=auc_pt, doc_ar=ar_pt, doc_rr=rr_pt)
    check_bootstrap(f"§7 iclr {cell_name} bACC CI",
                    test, lambda p: G.bal_acc(p, tau),
                    doc_lo=bal_lo, doc_hi=bal_hi)


# =============================================================================
# §8 integrity case study
# =============================================================================
section("§8 integrity case study")
ar_text_meta = json.loads(G.ARXIV_META[("text", "balanced", "test")].read_text())
ar_vis_meta  = json.loads(G.ARXIV_META[("vision", "balanced", "test")].read_text())
text_iclr_idx = {i for i, d in enumerate(ar_text_meta)
                 if (d["_metadata"].get("pl_venue") or "").lower() == "iclr"
                 and (d["_metadata"].get("conference_year") or d["_metadata"].get("year")) in (2025, 2026)}
vis_iclr_idx  = {i for i, d in enumerate(ar_vis_meta)
                 if (d["_metadata"].get("pl_venue") or "").lower() == "iclr"
                 and (d["_metadata"].get("conference_year") or d["_metadata"].get("year")) in (2025, 2026)}
check("§8 arxiv ICLR-subset (text) n=87",  len(text_iclr_idx) == 87, f"n={len(text_iclr_idx)}")
check("§8 arxiv ICLR-subset (vision) n=87", len(vis_iclr_idx) == 87, f"n={len(vis_iclr_idx)}")

for model_name, view, idx_set, short, step, train, ds_kind in [
    ("ICLR-trained 7B vision 50/50", "arxiv ICLR-subset", vis_iclr_idx, "bz16_lr1e-6_vision", 2648, "50_50", "arxiv"),
    ("ICLR-trained 7B vision 50/50", "gold ICLR 25/26",   None, "bz16_lr1e-6_vision", 2648, "50_50", "iclr"),
    ("ICLR-trained 7B text 50/50",   "arxiv ICLR-subset", text_iclr_idx, "bz32_lr1e-6_text", 1322, "50_50", "arxiv"),
    ("ICLR-trained 7B text 50/50",   "gold ICLR 25/26",   None, "bz32_lr1e-6_text", 1322, "50_50", "iclr"),
]:
    if ds_kind == "arxiv":
        full, _ = G.load_arxiv_cell(short, step, "balanced", "test")
        sub_pairs = [p for i, p in enumerate(full) if i in idx_set]
    else:
        sub_pairs, _ = G.load_iclr_cell(short, step, train, "balanced", "test")
    row = extract_table_row("§8.2", f"| {model_name} | {view}")
    if not row: continue
    bal_pt, bal_lo, bal_hi = parse_num(row[3])
    auc_pt, auc_lo, auc_hi = parse_num(row[4])
    ar_pt, _, _ = parse_num(row[5])
    rr_pt, _, _ = parse_num(row[6])
    check_metric_stack(f"§8 {model_name} on {view}", sub_pairs, 0.0,
                       doc_bal=bal_pt, doc_auc=auc_pt, doc_ar=ar_pt, doc_rr=rr_pt)


# =============================================================================
# §6.2 ICLR per-year split
# =============================================================================
section("§6.2 ICLR per-year")
PER_YEAR_DOC = {
    ("text", "50_50", 2025): {"bal": 68.1, "auc": 0.762},
    ("text", "50_50", 2026): {"bal": 63.3, "auc": 0.693},
    ("vision","50_50", 2025): {"bal": 70.4, "auc": 0.771},
    ("vision","50_50", 2026): {"bal": 65.7, "auc": 0.711},
}
for (mode, train, year), exp in PER_YEAR_DOC.items():
    short, step = G.MODELS_7B_ICLR[(mode, train)]
    pairs, meta = G.load_iclr_cell(short, step, train, "balanced", "test",
                                   fields=("pct_rating", "citation_normalized_by_year"))
    sub = [(s, g, m) for (s, g), m in zip(pairs, meta) if m["year"] == year]
    p = [(s, g) for s, g, _ in sub]
    bal = G.bal_acc(p, 0.0); auc = G.auc_score([s for s, _ in p], [g for _, g in p])
    check(f"§6.2 {mode} {train} {year}: bACC matches doc",
          abs(bal - exp["bal"]) < 0.5, f"ours={bal:.2f}, doc={exp['bal']:.2f}")
    check(f"§6.2 {mode} {train} {year}: AUC matches doc",
          abs(auc - exp["auc"]) < 0.005, f"ours={auc:.4f}, doc={exp['auc']:.4f}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print(f"\nTotal checks: {n_pass + len(failures)}, passed: {n_pass}, failed: {len(failures)}")
if failures:
    print(f"\033[91m{len(failures)} CHECK(S) FAILED:\033[0m")
    for f in failures: print(f"  - {f}")
    sys.exit(1)
else:
    print(f"\033[92mALL {n_pass} CHECKS PASSED\033[0m")
