#!/usr/bin/env python3
"""
Compute metrics for all alignment variants and print a comparison table.
Handles missing CSVs by recomputing from the jsonl match files.

Uses the same deduplication logic as grade_alignment.py:
  intersection_count = len(set(human_indices_matched))  — not len(all non-null matches)
"""
import json
import os
import csv

RESULTS_DIR = "alignment_analysis/results"


def compute_metrics(entry):
    matches = entry["matches"]
    n_human = len(entry["human_suggestions"])
    n_ai = len(entry["ai_suggestions"])
    # Deduplicate: count unique human indices matched (mirrors grade_alignment.py logic)
    matched_human_indices = set()
    matched_ai_count = 0
    for ai_pos, human_idx in matches.items():
        if human_idx is not None:
            matched_human_indices.add(human_idx)
            matched_ai_count += 1
    intersection_count = len(matched_human_indices)
    human_coverage = intersection_count / n_human if n_human > 0 else 0.0
    ai_coverage = matched_ai_count / n_ai if n_ai > 0 else 0.0
    unrelated_ai_count = n_ai - matched_ai_count
    return {
        "submission_id": entry["submission_id"],
        "intersection_count": intersection_count,
        "human_coverage": human_coverage,
        "ai_coverage": ai_coverage,
        "unrelated_ai_count": unrelated_ai_count,
        "n_human": n_human,
        "n_ai": n_ai,
    }


def load_and_compute(jsonl_path, csv_path):
    """Load matches, compute per-paper metrics, save CSV if missing, return summary."""
    if not os.path.exists(jsonl_path):
        return None, f"MISSING"

    all_metrics = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_metrics.append(compute_metrics(json.loads(line)))

    if not all_metrics:
        return None, f"EMPTY"

    # Save CSV if missing
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"  [saved] {csv_path}")

    n = len(all_metrics)
    summary = {
        "n": n,
        "avg_intersection": sum(m["intersection_count"] for m in all_metrics) / n,
        "avg_human_cov": sum(m["human_coverage"] for m in all_metrics) / n,
        "avg_ai_cov": sum(m["ai_coverage"] for m in all_metrics) / n,
        "avg_unrelated": sum(m["unrelated_ai_count"] for m in all_metrics) / n,
    }
    return summary, None


# File naming: the "original" variant was graded before the --ai_prompt_variant flag existed,
# so its files use "strict"/"lenient" infix (not "original_strict"/"original_lenient").
VARIANTS = [
    # (label, jsonl_infix, csv_infix)
    ("original strict  | conditioned  ", "conditioned_strict",           "conditioned_strict"),
    ("original strict  | unconditioned", "unconditioned_strict",         "unconditioned_strict"),
    ("original lenient | conditioned  ", "conditioned_lenient",          "conditioned_lenient"),
    ("original lenient | unconditioned", "unconditioned_lenient",        "unconditioned_lenient"),
    ("specific strict  | conditioned  ", "conditioned_specific_strict",  "conditioned_specific_strict"),
    ("specific strict  | unconditioned", "unconditioned_specific_strict","unconditioned_specific_strict"),
    ("specific lenient | conditioned  ", "conditioned_specific_lenient", "conditioned_specific_lenient"),
    ("specific lenient | unconditioned", "unconditioned_specific_lenient","unconditioned_specific_lenient"),
]


def run():
    print(f"\n{'Variant':<45} {'N':>5}  {'AvgInter':>9}  {'HumCov':>7}  {'AICov':>7}  {'Unrel':>6}")
    print("-" * 90)

    for label, jsonl_infix, csv_infix in VARIANTS:
        jsonl_path = os.path.join(RESULTS_DIR, f"alignment_matches_{jsonl_infix}.jsonl")
        csv_path   = os.path.join(RESULTS_DIR, f"alignment_metrics_{csv_infix}.csv")
        summary, err = load_and_compute(jsonl_path, csv_path)
        if err:
            print(f"{label:<45}  {err}")
        else:
            print(
                f"{label:<45} {summary['n']:>5}  "
                f"{summary['avg_intersection']:>9.3f}  "
                f"{summary['avg_human_cov']:>7.3f}  "
                f"{summary['avg_ai_cov']:>7.3f}  "
                f"{summary['avg_unrelated']:>6.3f}"
            )

    print()


if __name__ == "__main__":
    run()
