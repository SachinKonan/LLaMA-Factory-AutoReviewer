#!/usr/bin/env python3
"""
Analyze Qwen 3.5-122B blind text review accuracy against ground truth.

Loads reviews from results/reviews/qwen_blind_text_*.jsonl, merges with
ground truth labels and paper metadata, then produces a comprehensive
markdown report at staging/results/qwen35_analysis.md.
"""

import json
import csv
import random
import re
from collections import defaultdict, Counter
from pathlib import Path
from itertools import combinations

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REVIEW_GLOB = "results/reviews/qwen_blind_text_*.jsonl"
GT_DIRS = [
    "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train",
    "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test",
    "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation",
]
METADATA_CSV = "data/massive_metadata_v7.csv"
OUTPUT = "staging/results/qwen35_analysis.md"

REVIEW_STRING_FIELDS = [
    "strengths", "weaknesses", "critical_disputes",
    "most_important_strength", "most_important_weakness",
    "relation_to_other_work", "paper_quality",
]
MIN_FIELD_LEN = 20
NUM_CANDIDATES = 3
SAMPLE_SIZE_DIVERSITY = 500
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_valid_review(entry: dict) -> bool:
    """Check if a review entry has valid structure and content."""
    review = entry.get("review")
    if not isinstance(review, dict):
        return False
    if not isinstance(review.get("accept"), bool):
        return False
    for field in REVIEW_STRING_FIELDS:
        val = review.get(field)
        if not isinstance(val, str) or len(val) < MIN_FIELD_LEN:
            return False
    return True


def majority_vote(accepts: list[bool]) -> bool:
    """Return True if >=2/3 accept."""
    return sum(accepts) >= 2


def bin_value(val, edges, labels):
    """Bin a numeric value given edges and labels."""
    for i, edge in enumerate(edges):
        if val <= edge:
            return labels[i]
    return labels[-1]


def jaccard_words(a: str, b: str) -> float:
    """Word-level Jaccard similarity."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    def ngrams(s):
        s = s.lower()
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    sa, sb = ngrams(a), ngrams(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def fmt_pct(num, denom):
    if denom == 0:
        return "N/A"
    return f"{num/denom*100:.1f}%"


def fmt_f1(prec, rec):
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = fmt_f1(prec, rec)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "n": total}


def confusion(preds, gts):
    """preds and gts are lists of bool (True=Accept)."""
    tp = fp = fn = tn = 0
    for p, g in zip(preds, gts):
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif not p and g:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


# ---------------------------------------------------------------------------
# Step 1: Load & merge
# ---------------------------------------------------------------------------
def load_reviews():
    """Load all reviews, return dict: submission_id -> list of (candidate_idx, review_dict, entry)."""
    reviews = defaultdict(list)
    root = Path(".")
    files = sorted(root.glob(REVIEW_GLOB))
    print(f"Found {len(files)} review shard files")
    total = 0
    valid = 0
    for f in files:
        with open(f) as fh:
            for line in fh:
                total += 1
                entry = json.loads(line)
                if is_valid_review(entry):
                    valid += 1
                    sid = entry["submission_id"]
                    reviews[sid].append(entry)
    print(f"Total review entries: {total}, valid: {valid}")
    return reviews


def load_ground_truth():
    """Load ground truth from data.json files. Return dict: submission_id -> metadata."""
    gt = {}
    for d in GT_DIRS:
        p = Path(d) / "data.json"
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping")
            continue
        with open(p) as f:
            data = json.load(f)
        for item in data:
            meta = item.get("_metadata", {})
            sid = meta.get("submission_id")
            if sid:
                gt[sid] = meta
    print(f"Ground truth papers: {len(gt)}")
    return gt


def load_metadata_csv():
    """Load paper features from massive_metadata_v7.csv."""
    meta = {}
    p = Path(METADATA_CSV)
    if not p.exists():
        print(f"  WARNING: {p} not found")
        return meta
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("submission_id")
            if sid:
                meta[sid] = row
    print(f"Metadata CSV papers: {len(meta)}")
    return meta


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Qwen 3.5 Blind Text Review Analysis")
    print("=" * 60)

    # --- Load data ---
    reviews_by_paper = load_reviews()

    # Filter to papers with exactly 3 valid candidates
    papers_3valid = {}
    for sid, entries in reviews_by_paper.items():
        candidates = sorted(entries, key=lambda e: e["candidate_idx"])
        # deduplicate by candidate_idx
        seen = set()
        unique = []
        for e in candidates:
            if e["candidate_idx"] not in seen:
                seen.add(e["candidate_idx"])
                unique.append(e)
        if len(unique) >= NUM_CANDIDATES:
            papers_3valid[sid] = unique[:NUM_CANDIDATES]
    print(f"Papers with {NUM_CANDIDATES}/3 valid reviews: {len(papers_3valid)}")

    gt = load_ground_truth()
    csv_meta = load_metadata_csv()

    # Merge
    merged = []
    for sid, candidates in papers_3valid.items():
        if sid not in gt:
            continue
        meta = gt[sid]
        csv_row = csv_meta.get(sid, {})
        accepts = [c["review"]["accept"] for c in candidates]
        pred_accept = majority_vote(accepts)
        gt_accept = meta["answer"] == "Accept"
        unanimous = all(a == accepts[0] for a in accepts)
        merged.append({
            "sid": sid,
            "candidates": candidates,
            "accepts": accepts,
            "pred_accept": pred_accept,
            "gt_accept": gt_accept,
            "unanimous": unanimous,
            "year": meta.get("year"),
            "pct_rating": meta.get("pct_rating"),
            "ratings": meta.get("ratings"),
            "decision": meta.get("decision"),
            "citation": meta.get("citation"),
            "citation_norm": meta.get("citation_normalized_by_year"),
            "num_text_tokens": _safe_int(csv_row.get("num_text_tokens")),
            "num_pages": _safe_int(csv_row.get("num_pages")),
            "num_figures": _safe_int(csv_row.get("num_figures")),
            "num_equations": _safe_int(csv_row.get("num_equations")),
            "num_refs": _safe_int(csv_row.get("number_of_cited_references")),
        })
    print(f"Merged papers (reviews + ground truth): {len(merged)}")

    # --- Build report ---
    report = []
    report.append("# Qwen 3.5-122B Blind Text Review Analysis\n")
    report.append(f"**Papers analyzed**: {len(merged)}  ")
    report.append(f"**Papers with 3/3 valid reviews**: {len(papers_3valid)}  ")
    report.append(f"**Ground truth available**: {len(gt)}  \n")

    # ===== Step 2: Overall accuracy =====
    preds = [m["pred_accept"] for m in merged]
    gts = [m["gt_accept"] for m in merged]
    tp, fp, fn, tn = confusion(preds, gts)
    m_all = compute_metrics(tp, fp, fn, tn)

    report.append("## Overall Accuracy\n")
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Accuracy | {m_all['acc']:.4f} ({m_all['acc']*100:.1f}%) |")
    report.append(f"| Precision (Accept) | {m_all['prec']:.4f} |")
    report.append(f"| Recall (Accept) | {m_all['rec']:.4f} |")
    report.append(f"| F1 (Accept) | {m_all['f1']:.4f} |")
    report.append(f"| Total | {m_all['n']} |\n")

    report.append("### Confusion Matrix\n")
    report.append("| | Pred Accept | Pred Reject |")
    report.append("|---|---|---|")
    report.append(f"| **GT Accept** | {tp} (TP) | {fn} (FN) |")
    report.append(f"| **GT Reject** | {fp} (FP) | {tn} (TN) |\n")

    # Unanimous vs split
    unan = [m for m in merged if m["unanimous"]]
    split = [m for m in merged if not m["unanimous"]]
    for label, subset in [("Unanimous (3/3)", unan), ("Split (2/1)", split)]:
        if not subset:
            continue
        sp, sg = [m["pred_accept"] for m in subset], [m["gt_accept"] for m in subset]
        stp, sfp, sfn, stn = confusion(sp, sg)
        sm = compute_metrics(stp, sfp, sfn, stn)
        report.append(f"**{label}**: n={sm['n']}, acc={sm['acc']:.4f}, prec={sm['prec']:.4f}, rec={sm['rec']:.4f}, F1={sm['f1']:.4f}  ")
    report.append("")

    # Per-candidate correctness: how many papers have at least 1 correct candidate?
    at_least_1_correct = 0
    all_3_correct = 0
    correct_count_dist = Counter()  # 0, 1, 2, 3
    for m in merged:
        n_correct = sum(1 for a in m["accepts"] if a == m["gt_accept"])
        correct_count_dist[n_correct] += 1
        if n_correct >= 1:
            at_least_1_correct += 1
        if n_correct == 3:
            all_3_correct += 1

    report.append("### Individual Candidate Correctness\n")
    report.append("How many of the 3 candidates individually give the correct accept/reject answer?\n")
    report.append("| # Correct Candidates | Count | % |")
    report.append("|---------------------|-------|---|")
    for k in [0, 1, 2, 3]:
        report.append(f"| {k}/3 | {correct_count_dist[k]} | {fmt_pct(correct_count_dist[k], len(merged))} |")
    report.append(f"| **At least 1 correct** | **{at_least_1_correct}** | **{fmt_pct(at_least_1_correct, len(merged))}** |")
    report.append(f"| **All 3 correct** | **{all_3_correct}** | **{fmt_pct(all_3_correct, len(merged))}** |")
    report.append("")

    # Break down by GT label
    gt_accept_papers = [m for m in merged if m["gt_accept"]]
    gt_reject_papers = [m for m in merged if not m["gt_accept"]]
    atleast1_accept = sum(1 for m in gt_accept_papers if any(a for a in m["accepts"]))
    atleast1_reject = sum(1 for m in gt_reject_papers if any(not a for a in m["accepts"]))
    report.append(f"**GT=Accept papers with ≥1 candidate saying Accept**: {atleast1_accept}/{len(gt_accept_papers)} ({fmt_pct(atleast1_accept, len(gt_accept_papers))})  ")
    report.append(f"**GT=Reject papers with ≥1 candidate saying Reject**: {atleast1_reject}/{len(gt_reject_papers)} ({fmt_pct(atleast1_reject, len(gt_reject_papers))})  ")
    report.append("")

    # Prediction distribution
    pred_accept_count = sum(preds)
    pred_reject_count = len(preds) - pred_accept_count
    gt_accept_count = sum(gts)
    gt_reject_count = len(gts) - gt_accept_count
    report.append("### Prediction Distribution\n")
    report.append(f"| | Accept | Reject | Rate |")
    report.append(f"|---|---|---|---|")
    report.append(f"| **Predicted** | {pred_accept_count} | {pred_reject_count} | {fmt_pct(pred_accept_count, len(preds))} accept |")
    report.append(f"| **Ground Truth** | {gt_accept_count} | {gt_reject_count} | {fmt_pct(gt_accept_count, len(gts))} accept |\n")

    # Individual candidate accept rate
    all_individual = []
    for m in merged:
        all_individual.extend(m["accepts"])
    indiv_accept_rate = sum(all_individual) / len(all_individual) if all_individual else 0
    report.append(f"**Individual candidate accept rate**: {indiv_accept_rate:.4f} ({indiv_accept_rate*100:.1f}%)  ")
    report.append(f"**Majority vote accept rate**: {fmt_pct(pred_accept_count, len(preds))}  \n")

    # ===== Step 3: Accuracy by year =====
    report.append("## Accuracy by Year\n")
    by_year = defaultdict(list)
    for m in merged:
        by_year[m["year"]].append(m)

    report.append("| Year | N | Accuracy | Prec | Rec | F1 | Pred Accept% | GT Accept% |")
    report.append("|------|---|----------|------|-----|----|----|---|")
    for yr in sorted(by_year.keys()):
        subset = by_year[yr]
        sp = [m["pred_accept"] for m in subset]
        sg = [m["gt_accept"] for m in subset]
        stp, sfp, sfn, stn = confusion(sp, sg)
        sm = compute_metrics(stp, sfp, sfn, stn)
        pa = fmt_pct(sum(sp), len(sp))
        ga = fmt_pct(sum(sg), len(sg))
        flag = " **" if yr == 2026 else ""
        report.append(f"| {yr}{flag} | {sm['n']} | {sm['acc']:.4f} | {sm['prec']:.4f} | {sm['rec']:.4f} | {sm['f1']:.4f} | {pa} | {ga} |")
    report.append("")

    # ===== Step 4: Accuracy by paper features =====
    report.append("## Accuracy by Paper Features\n")

    # By pct_rating
    report.append("### By Reviewer Rating (pct_rating)\n")
    rating_bins = [
        (0.0, 0.3, "Strong Reject (0-0.3)"),
        (0.3, 0.45, "Lean Reject (0.3-0.45)"),
        (0.45, 0.55, "Borderline (0.45-0.55)"),
        (0.55, 0.7, "Lean Accept (0.55-0.7)"),
        (0.7, 1.01, "Strong Accept (0.7+)"),
    ]
    report.append("| Rating Bin | N | Accuracy | Prec | Rec | F1 | Pred Accept% | GT Accept% |")
    report.append("|------------|---|----------|------|-----|----|----|---|")
    for lo, hi, label in rating_bins:
        subset = [m for m in merged if m["pct_rating"] is not None and lo <= float(m["pct_rating"]) < hi]
        if not subset:
            continue
        sp = [m["pred_accept"] for m in subset]
        sg = [m["gt_accept"] for m in subset]
        stp, sfp, sfn, stn = confusion(sp, sg)
        sm = compute_metrics(stp, sfp, sfn, stn)
        report.append(f"| {label} | {sm['n']} | {sm['acc']:.4f} | {sm['prec']:.4f} | {sm['rec']:.4f} | {sm['f1']:.4f} | {fmt_pct(sum(sp), len(sp))} | {fmt_pct(sum(sg), len(sg))} |")
    report.append("")

    # Generic feature breakdown helper
    def feature_breakdown(feature_name, field_key, bin_edges, bin_labels):
        report.append(f"### By {feature_name}\n")
        report.append(f"| Bin | N | Accuracy | Pred Accept% | GT Accept% |")
        report.append(f"|-----|---|----------|----|----|")
        for i, label in enumerate(bin_labels):
            lo = 0 if i == 0 else bin_edges[i-1]
            hi = bin_edges[i] if i < len(bin_edges) else float('inf')
            subset = [m for m in merged if m[field_key] is not None and lo <= m[field_key] < hi]
            if not subset:
                # try with <= for last bin
                if i == len(bin_labels) - 1:
                    subset = [m for m in merged if m[field_key] is not None and lo <= m[field_key]]
                if not subset:
                    continue
            sp = [m["pred_accept"] for m in subset]
            sg = [m["gt_accept"] for m in subset]
            stp, sfp, sfn, stn = confusion(sp, sg)
            sm = compute_metrics(stp, sfp, sfn, stn)
            report.append(f"| {label} | {sm['n']} | {sm['acc']:.4f} | {fmt_pct(sum(sp), len(sp))} | {fmt_pct(sum(sg), len(sg))} |")
        report.append("")

    # Compute quartiles for token count
    token_vals = sorted([m["num_text_tokens"] for m in merged if m["num_text_tokens"] is not None])
    if token_vals:
        q1 = token_vals[len(token_vals)//4]
        q2 = token_vals[len(token_vals)//2]
        q3 = token_vals[3*len(token_vals)//4]
        feature_breakdown("Text Token Count", "num_text_tokens",
                          [q1, q2, q3],
                          [f"Q1 (≤{q1})", f"Q2 ({q1}-{q2})", f"Q3 ({q2}-{q3})", f"Q4 (>{q3})"])

    feature_breakdown("Number of Pages", "num_pages",
                      [6, 9, 12, 20],
                      ["≤6", "7-9", "10-12", "13-20", ">20"])

    feature_breakdown("Number of Figures", "num_figures",
                      [2, 5, 10, 20],
                      ["0-2", "3-5", "6-10", "11-20", ">20"])

    feature_breakdown("Number of Equations", "num_equations",
                      [5, 15, 30, 60],
                      ["0-5", "6-15", "16-30", "31-60", ">60"])

    feature_breakdown("Number of References", "num_refs",
                      [20, 35, 50, 70],
                      ["≤20", "21-35", "36-50", "51-70", ">70"])

    # By citation_normalized_by_year
    report.append("### By Citation Impact (citation_normalized_by_year)\n")
    cite_vals = sorted([float(m["citation_norm"]) for m in merged
                        if m["citation_norm"] is not None and m["citation_norm"] != ""])
    if cite_vals:
        cq1 = cite_vals[len(cite_vals)//4]
        cq2 = cite_vals[len(cite_vals)//2]
        cq3 = cite_vals[3*len(cite_vals)//4]
        report.append(f"| Quartile | N | Accuracy | Pred Accept% | GT Accept% |")
        report.append(f"|----------|---|----------|----|----|")
        for label, lo, hi in [
            (f"Q1 (≤{cq1:.1f})", -1e9, cq1),
            (f"Q2 ({cq1:.1f}-{cq2:.1f})", cq1, cq2),
            (f"Q3 ({cq2:.1f}-{cq3:.1f})", cq2, cq3),
            (f"Q4 (>{cq3:.1f})", cq3, 1e9),
        ]:
            subset = [m for m in merged if m["citation_norm"] is not None
                      and m["citation_norm"] != "" and lo <= float(m["citation_norm"]) < hi]
            if not subset:
                continue
            sp = [m["pred_accept"] for m in subset]
            sg = [m["gt_accept"] for m in subset]
            stp, sfp, sfn, stn = confusion(sp, sg)
            sm = compute_metrics(stp, sfp, sfn, stn)
            report.append(f"| {label} | {sm['n']} | {sm['acc']:.4f} | {fmt_pct(sum(sp), len(sp))} | {fmt_pct(sum(sg), len(sg))} |")
        report.append("")

    # ===== Step 5: Error analysis =====
    report.append("## Error Analysis\n")

    false_pos = [m for m in merged if m["pred_accept"] and not m["gt_accept"]]
    false_neg = [m for m in merged if not m["pred_accept"] and m["gt_accept"]]
    true_pos = [m for m in merged if m["pred_accept"] and m["gt_accept"]]
    true_neg = [m for m in merged if not m["pred_accept"] and not m["gt_accept"]]

    report.append(f"| Category | Count |")
    report.append(f"|----------|-------|")
    report.append(f"| True Positives (correct accept) | {len(true_pos)} |")
    report.append(f"| True Negatives (correct reject) | {len(true_neg)} |")
    report.append(f"| False Positives (predicted accept, actually reject) | {len(false_pos)} |")
    report.append(f"| False Negatives (predicted reject, actually accept) | {len(false_neg)} |\n")

    # Error characteristics
    def avg_feature(subset, key):
        vals = [m[key] for m in subset if m[key] is not None]
        if not vals:
            return "N/A"
        return f"{sum(vals)/len(vals):.1f}"

    def avg_rating(subset):
        vals = [float(m["pct_rating"]) for m in subset if m["pct_rating"] is not None]
        if not vals:
            return "N/A"
        return f"{sum(vals)/len(vals):.3f}"

    report.append("### Error vs Correct Characteristics\n")
    report.append("| Feature | True Pos | True Neg | False Pos | False Neg |")
    report.append("|---------|----------|----------|-----------|-----------|")
    report.append(f"| Count | {len(true_pos)} | {len(true_neg)} | {len(false_pos)} | {len(false_neg)} |")
    report.append(f"| Avg pct_rating | {avg_rating(true_pos)} | {avg_rating(true_neg)} | {avg_rating(false_pos)} | {avg_rating(false_neg)} |")
    report.append(f"| Avg tokens | {avg_feature(true_pos, 'num_text_tokens')} | {avg_feature(true_neg, 'num_text_tokens')} | {avg_feature(false_pos, 'num_text_tokens')} | {avg_feature(false_neg, 'num_text_tokens')} |")
    report.append(f"| Avg pages | {avg_feature(true_pos, 'num_pages')} | {avg_feature(true_neg, 'num_pages')} | {avg_feature(false_pos, 'num_pages')} | {avg_feature(false_neg, 'num_pages')} |")
    report.append(f"| Avg figures | {avg_feature(true_pos, 'num_figures')} | {avg_feature(true_neg, 'num_figures')} | {avg_feature(false_pos, 'num_figures')} | {avg_feature(false_neg, 'num_figures')} |")
    report.append(f"| Avg equations | {avg_feature(true_pos, 'num_equations')} | {avg_feature(true_neg, 'num_equations')} | {avg_feature(false_pos, 'num_equations')} | {avg_feature(false_neg, 'num_equations')} |")
    report.append(f"| Avg refs | {avg_feature(true_pos, 'num_refs')} | {avg_feature(true_neg, 'num_refs')} | {avg_feature(false_pos, 'num_refs')} | {avg_feature(false_neg, 'num_refs')} |")
    report.append("")

    # Borderline concentration
    borderline = [m for m in merged if m["pct_rating"] is not None and 0.4 <= float(m["pct_rating"]) <= 0.6]
    non_borderline = [m for m in merged if m["pct_rating"] is not None and (float(m["pct_rating"]) < 0.4 or float(m["pct_rating"]) > 0.6)]
    if borderline:
        bp, bg = [m["pred_accept"] for m in borderline], [m["gt_accept"] for m in borderline]
        btp, bfp, bfn, btn = confusion(bp, bg)
        bm = compute_metrics(btp, bfp, bfn, btn)
        report.append(f"**Borderline papers (pct_rating 0.4-0.6)**: n={bm['n']}, accuracy={bm['acc']:.4f}  ")
    if non_borderline:
        np_, ng_ = [m["pred_accept"] for m in non_borderline], [m["gt_accept"] for m in non_borderline]
        ntp, nfp, nfn, ntn = confusion(np_, ng_)
        nm = compute_metrics(ntp, nfp, nfn, ntn)
        report.append(f"**Non-borderline papers**: n={nm['n']}, accuracy={nm['acc']:.4f}  ")
    report.append("")

    # Year-specific errors
    report.append("### Errors by Year\n")
    report.append("| Year | FP | FN | FP Rate | FN Rate |")
    report.append("|------|----|----|---------|---------|")
    for yr in sorted(by_year.keys()):
        subset = by_year[yr]
        yr_fp = sum(1 for m in subset if m["pred_accept"] and not m["gt_accept"])
        yr_fn = sum(1 for m in subset if not m["pred_accept"] and m["gt_accept"])
        yr_reject = sum(1 for m in subset if not m["gt_accept"])
        yr_accept = sum(1 for m in subset if m["gt_accept"])
        report.append(f"| {yr} | {yr_fp} | {yr_fn} | {fmt_pct(yr_fp, yr_reject)} | {fmt_pct(yr_fn, yr_accept)} |")
    report.append("")

    # ===== Step 6: Tone & content analysis =====
    report.append("## Tone & Content Analysis\n")

    # Rejection rate
    report.append("### Prediction Skew\n")
    report.append(f"- **Individual candidate reject rate**: {(1-indiv_accept_rate)*100:.1f}%  ")
    report.append(f"- **Majority vote reject rate**: {fmt_pct(pred_reject_count, len(preds))}  ")
    report.append(f"- **Dataset base rate (reject)**: {fmt_pct(gt_reject_count, len(gts))}  \n")

    # Field length stats
    report.append("### Field Length Statistics (characters)\n")
    field_lengths = defaultdict(list)
    for m in merged:
        for c in m["candidates"]:
            rev = c["review"]
            for field in REVIEW_STRING_FIELDS:
                field_lengths[field].append(len(rev.get(field, "")))

    report.append("| Field | Mean | Median | Min | Max |")
    report.append("|-------|------|--------|-----|-----|")
    for field in REVIEW_STRING_FIELDS:
        vals = sorted(field_lengths[field])
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        median = vals[len(vals)//2]
        report.append(f"| {field} | {mean:.0f} | {median} | {vals[0]} | {vals[-1]} |")
    report.append("")

    # Tone: field lengths for accepted vs rejected predictions
    report.append("### Field Lengths: Accept vs Reject Predictions\n")
    report.append("| Field | Pred Accept (mean) | Pred Reject (mean) | Difference |")
    report.append("|-------|-------|-------|------|")
    for field in REVIEW_STRING_FIELDS:
        acc_lens = []
        rej_lens = []
        for m in merged:
            for c in m["candidates"]:
                if c["review"]["accept"]:
                    acc_lens.append(len(c["review"].get(field, "")))
                else:
                    rej_lens.append(len(c["review"].get(field, "")))
        a_mean = sum(acc_lens)/len(acc_lens) if acc_lens else 0
        r_mean = sum(rej_lens)/len(rej_lens) if rej_lens else 0
        diff = a_mean - r_mean
        report.append(f"| {field} | {a_mean:.0f} | {r_mean:.0f} | {diff:+.0f} |")
    report.append("")

    # ===== Step 7: Multi-generation diversity =====
    report.append("## Multi-Generation Diversity Analysis\n")

    # Verdict agreement
    unanimous_count = sum(1 for m in merged if m["unanimous"])
    split_count = len(merged) - unanimous_count
    all_accept = sum(1 for m in merged if all(a for a in m["accepts"]))
    all_reject = sum(1 for m in merged if all(not a for a in m["accepts"]))
    two_one = split_count

    report.append("### Verdict Agreement\n")
    report.append(f"| Pattern | Count | % |")
    report.append(f"|---------|-------|---|")
    report.append(f"| All 3 agree | {unanimous_count} | {fmt_pct(unanimous_count, len(merged))} |")
    report.append(f"| — All accept | {all_accept} | {fmt_pct(all_accept, len(merged))} |")
    report.append(f"| — All reject | {all_reject} | {fmt_pct(all_reject, len(merged))} |")
    report.append(f"| 2-1 split | {two_one} | {fmt_pct(two_one, len(merged))} |")
    report.append("")

    # Content similarity on sample
    sample_ids = random.sample(list(range(len(merged))), min(SAMPLE_SIZE_DIVERSITY, len(merged)))
    sample = [merged[i] for i in sample_ids]

    jaccard_strengths = []
    jaccard_weaknesses = []
    ngram_strengths = []
    ngram_weaknesses = []

    for m in sample:
        cands = m["candidates"]
        for i, j in combinations(range(NUM_CANDIDATES), 2):
            s_i = cands[i]["review"]["strengths"]
            s_j = cands[j]["review"]["strengths"]
            w_i = cands[i]["review"]["weaknesses"]
            w_j = cands[j]["review"]["weaknesses"]
            jaccard_strengths.append(jaccard_words(s_i, s_j))
            jaccard_weaknesses.append(jaccard_words(w_i, w_j))
            ngram_strengths.append(ngram_overlap(s_i, s_j))
            ngram_weaknesses.append(ngram_overlap(w_i, w_j))

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    report.append(f"### Content Similarity (sample of {len(sample)} papers, pairwise)\n")
    report.append("| Metric | Strengths | Weaknesses |")
    report.append("|--------|-----------|------------|")
    report.append(f"| Word Jaccard (mean) | {avg(jaccard_strengths):.4f} | {avg(jaccard_weaknesses):.4f} |")
    report.append(f"| 3-gram overlap (mean) | {avg(ngram_strengths):.4f} | {avg(ngram_weaknesses):.4f} |")
    report.append("")

    # ===== Step 7 (cont): Concrete examples =====
    report.append("### Concrete Examples\n")

    def format_review_example(m, label):
        lines = []
        lines.append(f"#### {label}\n")
        lines.append(f"**Paper**: `{m['sid']}` (year={m['year']}, pct_rating={m['pct_rating']})  ")
        lines.append(f"**Ground truth**: {'Accept' if m['gt_accept'] else 'Reject'}  ")
        lines.append(f"**Majority prediction**: {'Accept' if m['pred_accept'] else 'Reject'}  ")
        lines.append(f"**Individual votes**: {['Accept' if a else 'Reject' for a in m['accepts']]}  \n")
        for idx, c in enumerate(m["candidates"]):
            rev = c["review"]
            lines.append(f"<details><summary>Candidate {idx} ({'Accept' if rev['accept'] else 'Reject'})</summary>\n")
            for field in REVIEW_STRING_FIELDS + ["accept"]:
                val = rev.get(field, "")
                if field == "accept":
                    lines.append(f"**{field}**: {val}\n")
                else:
                    # Truncate to 500 chars for readability
                    text = str(val)[:500]
                    if len(str(val)) > 500:
                        text += "..."
                    lines.append(f"**{field}**:\n{text}\n")
            lines.append("</details>\n")
        return "\n".join(lines)

    # Pick examples
    examples = []

    # 1. Same paper 3 candidates side-by-side (pick a split decision)
    split_examples = [m for m in merged if not m["unanimous"]]
    if split_examples:
        ex = random.choice(split_examples)
        examples.append(format_review_example(ex, "Example 1: Split Decision (3 candidates side-by-side)"))

    # 2. Correct accept
    if true_pos:
        ex = random.choice(true_pos)
        examples.append(format_review_example(ex, "Example 2: Correct Accept Prediction"))

    # 3. Correct reject
    if true_neg:
        ex = random.choice(true_neg)
        examples.append(format_review_example(ex, "Example 3: Correct Reject Prediction"))

    # 4. False positive
    if false_pos:
        ex = random.choice(false_pos)
        examples.append(format_review_example(ex, "Example 4: False Positive (predicted Accept, actually Reject)"))

    # 5. False negative
    if false_neg:
        ex = random.choice(false_neg)
        examples.append(format_review_example(ex, "Example 5: False Negative (predicted Reject, actually Accept)"))

    for ex_text in examples:
        report.append(ex_text)

    # ===== Step 8: Key findings =====
    report.append("## Key Findings\n")
    findings = []
    findings.append(f"1. **Overall accuracy**: {m_all['acc']*100:.1f}% across {m_all['n']} papers")
    findings.append(f"2. **Massive rejection bias**: {(1-indiv_accept_rate)*100:.1f}% of individual candidates predict reject vs {fmt_pct(gt_reject_count, len(gts))} actual reject rate")
    findings.append(f"3. **Unanimous agreement**: {fmt_pct(unanimous_count, len(merged))} of papers have all 3 candidates agree")
    if borderline:
        findings.append(f"4. **Borderline difficulty**: Accuracy on borderline papers (pct_rating 0.4-0.6) is {bm['acc']*100:.1f}% vs {nm['acc']*100:.1f}% on non-borderline")
    if by_year.get(2026):
        yr26 = by_year[2026]
        sp26 = [m["pred_accept"] for m in yr26]
        sg26 = [m["gt_accept"] for m in yr26]
        tp26, fp26, fn26, tn26 = confusion(sp26, sg26)
        m26 = compute_metrics(tp26, fp26, fn26, tn26)
        findings.append(f"5. **2026 (unseen year)**: Accuracy {m26['acc']*100:.1f}% on {m26['n']} papers")
    findings.append(f"6. **Content similarity**: Pairwise word Jaccard for strengths={avg(jaccard_strengths):.3f}, weaknesses={avg(jaccard_weaknesses):.3f}")

    for f in findings:
        report.append(f)
    report.append("")

    # Write report
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write("\n".join(report))
    print(f"\nReport written to {OUTPUT}")
    print(f"Total papers in report: {len(merged)}")


def _safe_int(val):
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    main()
