#!/usr/bin/env python3
"""
Analyze final_sweep_v7_datasweepv3 results with per-year breakdown.

For these datasets, 2024 is already excluded from the test set, so there is
no "excluding 2024" table. Instead we show "excluding 2026" since 2026 data
may be less reliable.

Usage:
    python scripts/analyze_v7_ds3.py results/final_sweep_v7_datasweepv3/bal_no2024_clean/
    python scripts/analyze_v7_ds3.py --all  # Analyze all available results
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


# Dataset mappings: short_name -> dataset_name
DATASET_MAPPINGS = {
    # ── Text: no 2024 ──
    "bal_no2024_orig_text": "iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered",
    "bal_no2024_clean": "iclr_2020_2023_2025_85_5_10_balanced_pdftitleabs_clean_v7_filtered",
    "ta_no2024_orig_text": "iclr_2020_2023_2025_85_5_10_trainagreeing_original_text_v7_filtered",
    "ta_no2024_clean": "iclr_2020_2023_2025_85_5_10_trainagreeing_pdftitleabs_clean_v7_filtered",
    # ── Text: no 2024 + 2026 ──
    "bal_no2024_w2026_orig_text": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_v7_filtered",
    "bal_no2024_w2026_clean": "iclr_2020_2023_2025_2026_85_5_10_balanced_pdftitleabs_clean_v7_filtered",
    "ta_no2024_w2026_orig_text": "iclr_2020_2023_2025_2026_85_5_10_trainagreeing_original_text_v7_filtered",
    "ta_no2024_w2026_clean": "iclr_2020_2023_2025_2026_85_5_10_trainagreeing_pdftitleabs_clean_v7_filtered",
    # ── Text: with 2024 (corrected) ──
    "bal_w2024_orig_text": "iclr_2020_2025_85_5_10_balanced_original_text_corrected_v7_filtered",
    "bal_w2024_clean": "iclr_2020_2025_85_5_10_balanced_pdftitleabs_clean_v7_filtered",
    "ta_w2024_orig_text": "iclr_2020_2025_85_5_10_trainagreeing_original_text_corrected_v7_filtered",
    "ta_w2024_clean": "iclr_2020_2025_85_5_10_trainagreeing_pdftitleabs_clean_v7_filtered",
    # ── Vision: no 2024 ──
    "bal_no2024_orig_vision": "iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered",
    "bal_no2024_pdftitleabs_vision": "iclr_2020_2023_2025_85_5_10_balanced_pdftitleabs_vision_v7_filtered",
    "ta_no2024_orig_vision": "iclr_2020_2023_2025_85_5_10_trainagreeing_original_vision_v7_filtered",
    "ta_no2024_pdftitleabs_vision": "iclr_2020_2023_2025_85_5_10_trainagreeing_pdftitleabs_vision_v7_filtered",
    # ── Vision: no 2024 + 2026 ──
    "bal_no2024_w2026_orig_vision": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_v7_filtered",
    "bal_no2024_w2026_pdftitleabs_vision": "iclr_2020_2023_2025_2026_85_5_10_balanced_pdftitleabs_vision_v7_filtered",
    "ta_no2024_w2026_orig_vision": "iclr_2020_2023_2025_2026_85_5_10_trainagreeing_original_vision_v7_filtered",
    "ta_no2024_w2026_pdftitleabs_vision": "iclr_2020_2023_2025_2026_85_5_10_trainagreeing_pdftitleabs_vision_v7_filtered",
    # ── Vision: with 2024 (corrected) ──
    "bal_w2024_orig_vision": "iclr_2020_2025_85_5_10_balanced_original_vision_corrected_v7_filtered",
    "bal_w2024_pdftitleabs_vision": "iclr_2020_2025_85_5_10_balanced_pdftitleabs_vision_v7_filtered",
    "ta_w2024_orig_vision": "iclr_2020_2025_85_5_10_trainagreeing_original_vision_corrected_v7_filtered",
    "ta_w2024_pdftitleabs_vision": "iclr_2020_2025_85_5_10_trainagreeing_pdftitleabs_vision_v7_filtered",
    # ── Optim search: text (all use same dataset) ──
    "bz16_lr0.5e-6_text": "iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered",
    "bz16_lr1e-6_text": "iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered",
    "bz32_lr1e-6_text": "iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered",
    "bz32_lr2e-6_text": "iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered",
    "bz64_lr2e-6_text": "iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered",
    "bz64_lr4e-6_text": "iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered",
    # ── Optim search: vision (all use same dataset) ──
    "bz16_lr1e-6_vision": "iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered",
    "bz16_lr2e-6_vision": "iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered",
    "bz32_lr2e-6_vision": "iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered",
    "bz32_lr4e-6_vision": "iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered",
    "bz64_lr4e-6_vision": "iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered",
    "bz64_lr5.5e-6_vision": "iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered",
}


def _wd_variant_dataset(variant_name: str) -> str:
    """Infer dataset for a WD sweep variant from its name suffix."""
    if variant_name.endswith("_text"):
        return "iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered"
    return "iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered"


def _optim_no2024_dataset(variant_name: str) -> str:
    """Dataset for optim search (no-2024) variants."""
    if variant_name.endswith("_text"):
        return "iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered"
    return "iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered"


def _optim_2026_labelfix_dataset(variant_name: str) -> str:
    """Dataset for optim_search_2026 (labelfix) variants."""
    if variant_name.endswith("_text"):
        return "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
    return "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480"


def _optim_2020_2025_origin_dataset(variant_name: str) -> str:
    """Dataset for optim_2020_2025_origin (corrected, with 2024) variants."""
    if variant_name.endswith("_text"):
        return "iclr_2020_2025_85_5_10_balanced_original_text_corrected_v7_filtered"
    return "iclr_2020_2025_85_5_10_balanced_original_vision_corrected_v7_filtered"


def _extras_dataset(variant_name: str) -> str:
    """Dataset for optim_search_2026_with_extras variants.

    Maps variant+modality to the correct extras-appended dataset.
    Parent dir name encodes the extras type (paper_stats/qwen_reviews/gemini_reviews).
    """
    suffix_map = {
        "paper_stats": "paperstats",
        "qwen_reviews": "qwenreviews",
        "gemini_reviews": "geminireviews",
    }
    # This resolver is called per-variant inside a subdir like
    # optim_search_2026_with_extras/paper_stats/bz32_lr1e-6_text
    # The variant_name is "bz32_lr1e-6_text", but we need the extras type
    # from the parent. Since SUBDIR_CONFIGS uses the full subdir path,
    # we handle all three suffixes via separate config entries.
    # The actual suffix is injected by the lambda in SUBDIR_CONFIGS.
    raise NotImplementedError("Use _extras_dataset_factory instead")


def _extras_dataset_factory(extras_suffix: str):
    """Return a resolver for extras datasets with the given suffix."""
    def resolver(variant_name: str) -> str:
        if variant_name.endswith("_text"):
            return f"iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_{extras_suffix}"
        return f"iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_{extras_suffix}"
    return resolver


def _trainval_dataset(variant_name: str) -> str:
    """Dataset for optim_search_trainval variants (eval on 2026 labelfix test)."""
    if variant_name.endswith("_text") or "_text_" in variant_name:
        return "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
    return "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480"


def _year_conditioned_dataset(variant_name: str) -> str:
    """Dataset for year_conditioned variants (multi-conference yearcond test)."""
    if "vision" in variant_name:
        return "all_conferences_vision_v7_extra_yearcond"
    return "all_conferences_text_v7_extra_yearcond"


def _gemini_no2026_dataset(variant_name: str) -> str:
    """Dataset for gemini_no2026 variants (multi-conference test)."""
    return "all_conferences_text_v7_extra"


def _pretrain_then_sft_dataset(variant_name: str) -> str:
    """Dataset for pretrain_then_sft variants (multi-conference text test).

    New runs (base_direct_text, base_iclr_only_*) use balanced_nips_icml_iclr_text_eval;
    old runs use all_conferences_text_v7_extra.
    """
    if variant_name in ("base_direct_text",) or variant_name.startswith("base_iclr_only"):
        return "balanced_nips_icml_iclr_text_eval"
    return "all_conferences_text_v7_extra"


def _trainagreeing_dataset(variant_name: str) -> str:
    """Dataset for trainagreeing variants (2026 labelfix vs no-2026)."""
    has_2026 = "2026" in variant_name and "no2026" not in variant_name
    is_text = "_text" in variant_name
    if has_2026:
        if is_text:
            return "iclr_2020_2023_2025_2026_85_5_10_trainagreeing_original_text_labelfix_v7_filtered"
        return "iclr_2020_2023_2025_2026_85_5_10_trainagreeing_original_vision_labelfix_v7_filtered"
    else:
        if is_text:
            return "iclr_2020_2023_2025_85_5_10_trainagreeing_original_text_v7_filtered"
        return "iclr_2020_2023_2025_85_5_10_trainagreeing_original_vision_v7_filtered"


# Subdirectories to scan and their dataset resolver functions.
# Each entry is (subdir_name, dataset_resolver_fn).
# The resolver takes a variant name and returns the dataset name.
SUBDIR_CONFIGS = [
    ("wd_sweep", _wd_variant_dataset),
    ("wd_sweep_expdecay", _wd_variant_dataset),
    ("wd_sweep_2epoch", _wd_variant_dataset),
    ("wd_sweep_2epoch_3epochexp", _wd_variant_dataset),
    ("wd_sweep_lrdrop", _wd_variant_dataset),
    ("optim_search", _optim_no2024_dataset),
    ("optim_search_6epochs", _optim_no2024_dataset),
    ("optim_search_2026", _optim_2026_labelfix_dataset),
    ("optim_2020_2025_origin", _optim_2020_2025_origin_dataset),
    ("trainagreeing", _trainagreeing_dataset),
    ("optim_search_trainval", _trainval_dataset),
    ("optim_search_2026_with_extras/paper_stats", _extras_dataset_factory("paperstats")),
    ("optim_search_2026_with_extras/paper_stats_full", _extras_dataset_factory("paperstats_full")),
    ("optim_search_2026_with_extras/qwen_reviews", _extras_dataset_factory("qwenreviews")),
    ("optim_search_2026_with_extras/gemini_reviews", _extras_dataset_factory("geminireviews")),
    ("optim_search_2026_with_extras/qwen_reviews_x2", _extras_dataset_factory("qwenreviews_x2")),
    ("optim_search_2026_with_extras/gemini_reviews_x2", _extras_dataset_factory("geminireviews_x2")),
    ("optim_search_2026_with_extras/gemini_reviews_x3", _extras_dataset_factory("geminireviews_x3")),
    ("year_conditioned", _year_conditioned_dataset),
    ("gemini_no2026", _gemini_no2026_dataset),
    ("data_cleaning", lambda v: (
        "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
        if v.startswith("text_") else
        "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480"
    )),
    ("scaling", _optim_2026_labelfix_dataset),
]

# Separate results dirs not under final_sweep_v7_datasweepv3
EXTRA_RESULT_CONFIGS = [
    ("results/pretrain_then_sft", _pretrain_then_sft_dataset),
]

DATA_DIR = Path("data")


def load_predictions(pred_file: Path) -> list[dict]:
    """Load predictions from jsonl file."""
    predictions = []
    with open(pred_file) as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def load_test_dataset(dataset_name: str) -> list[dict]:
    """Load test dataset with metadata."""
    path = DATA_DIR / f"{dataset_name}_test" / "data.json"
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")

    with open(path) as f:
        return json.load(f)


def extract_prediction(text: str) -> str:
    """Extract Accept/Reject from model prediction.

    Handles accept/reject, yes/no, and Y/N formats.
    """
    text_lower = text.lower().strip()

    # Single letter format (Y/N)
    if text_lower == "y":
        return "accept"
    if text_lower == "n":
        return "reject"

    # Look for boxed answers (accept/reject and yes/no)
    if "\\boxed{accept}" in text_lower or "boxed{accept}" in text_lower:
        return "accept"
    if "\\boxed{reject}" in text_lower or "boxed{reject}" in text_lower:
        return "reject"
    if "\\boxed{yes}" in text_lower or "boxed{yes}" in text_lower or "\\boxed{y}" in text_lower:
        return "accept"
    if "\\boxed{no}" in text_lower or "boxed{no}" in text_lower or "\\boxed{n}" in text_lower:
        return "reject"

    # Fallback: look for accept/reject/yes/no keywords
    yes_pos = text_lower.rfind("yes")
    no_pos = text_lower.rfind("no")
    accept_pos = text_lower.rfind("accept")
    reject_pos = text_lower.rfind("reject")

    # Find the last occurring keyword
    positions = [
        (yes_pos, "accept"),
        (no_pos, "reject"),
        (accept_pos, "accept"),
        (reject_pos, "reject"),
    ]
    # Filter out -1 (not found)
    positions = [(pos, label) for pos, label in positions if pos != -1]

    if positions:
        # Return the label of the last occurring keyword
        positions.sort(key=lambda x: x[0])
        return positions[-1][1]

    return "unknown"


def extract_label(text: str) -> str:
    """Extract Accept/Reject from ground truth label.

    Handles accept/reject, yes/no, and Y/N formats.
    """
    text_lower = text.lower().strip()

    # Single letter format (Y/N)
    if text_lower == "y":
        return "accept"
    if text_lower == "n":
        return "reject"

    # Full word formats
    if "accept" in text_lower or "yes" == text_lower:
        return "accept"
    if "reject" in text_lower or "no" == text_lower:
        return "reject"
    return "unknown"


def compute_stats(predictions: list[dict], test_data: list[dict]) -> tuple[dict, dict]:
    """Compute per-year statistics from SFT predictions.

    If the test set has multiple entries per submission_id (e.g. x2/x3
    datasets with different reviews per paper), predictions are grouped
    by submission_id and majority-voted before scoring.

    Returns (year_stats, conf_year_stats) where:
      year_stats = {year: {tp, tn, fp, fn, total}}
      conf_year_stats = {conference: {year: {tp, tn, fp, fn, total}}}
    """
    if len(predictions) != len(test_data):
        print(f"Warning: predictions ({len(predictions)}) != test_data ({len(test_data)})")
        min_len = min(len(predictions), len(test_data))
        predictions = predictions[:min_len]
        test_data = test_data[:min_len]

    # Detect multi-answer datasets by checking for duplicate submission_ids
    sid_counts: dict[str, int] = defaultdict(int)
    for item in test_data:
        sid = item.get("_metadata", {}).get("submission_id")
        if sid:
            sid_counts[sid] += 1
    max_copies = max(sid_counts.values()) if sid_counts else 1
    has_multi = max_copies > 1

    if has_multi:
        n_unique = len(sid_counts)
        n_multi = sum(1 for c in sid_counts.values() if c > 1)
        print(f"  Multi-answer dataset detected: {max_copies} copies/paper, "
              f"{n_unique} unique papers ({n_multi} with {max_copies} copies)")
        return _compute_stats_majority_vote(predictions, test_data)

    return _compute_stats_simple(predictions, test_data)


def _update_stats(stats: dict, pred_label: str, true_label: str):
    """Update a stats dict with a single prediction."""
    stats["total"] += 1
    if true_label == "accept":
        if pred_label == "accept":
            stats["tp"] += 1
        else:
            stats["fn"] += 1
    else:
        if pred_label == "reject":
            stats["tn"] += 1
        else:
            stats["fp"] += 1


def _new_stats():
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0}


def _compute_stats_simple(predictions: list[dict], test_data: list[dict]) -> tuple[dict, dict]:
    """Standard per-entry scoring (1 prediction per paper).

    Returns (year_stats, conf_year_stats) where:
      year_stats = {year: {tp, tn, fp, fn, total}}
      conf_year_stats = {conference: {year: {tp, tn, fp, fn, total}}}
    """
    year_stats = defaultdict(_new_stats)
    conf_year_stats = defaultdict(lambda: defaultdict(_new_stats))

    for i, pred in enumerate(predictions):
        metadata = test_data[i].get("_metadata", {})
        year = metadata.get("year")
        if year is None:
            continue

        conference = metadata.get("conference", "unknown").lower()
        pred_label = extract_prediction(pred.get("predict", ""))
        true_label = extract_label(pred.get("label", ""))

        if pred_label == "unknown" or true_label == "unknown":
            continue

        _update_stats(year_stats[year], pred_label, true_label)
        _update_stats(conf_year_stats[conference][year], pred_label, true_label)

    return dict(year_stats), {c: dict(v) for c, v in conf_year_stats.items()}


def _compute_stats_majority_vote(predictions: list[dict], test_data: list[dict]) -> tuple[dict, dict]:
    """Group predictions by submission_id and majority-vote before scoring."""
    # Collect all predictions per submission_id
    paper_preds: dict[str, dict] = {}  # sid -> {year, conference, true_label, pred_labels: []}
    for i, pred in enumerate(predictions):
        metadata = test_data[i].get("_metadata", {})
        sid = metadata.get("submission_id")
        year = metadata.get("year")
        if sid is None or year is None:
            continue

        conference = metadata.get("conference", "unknown").lower()
        pred_label = extract_prediction(pred.get("predict", ""))
        true_label = extract_label(pred.get("label", ""))

        if pred_label == "unknown" or true_label == "unknown":
            continue

        if sid not in paper_preds:
            paper_preds[sid] = {"year": year, "conference": conference, "true_label": true_label, "pred_labels": []}
        paper_preds[sid]["pred_labels"].append(pred_label)

    # Majority vote and score
    year_stats = defaultdict(_new_stats)
    conf_year_stats = defaultdict(lambda: defaultdict(_new_stats))

    for sid, info in paper_preds.items():
        year = info["year"]
        conference = info["conference"]
        true_label = info["true_label"]
        votes = info["pred_labels"]

        # Majority vote (accept wins ties)
        accept_count = sum(1 for v in votes if v == "accept")
        pred_label = "accept" if accept_count > len(votes) / 2 else "reject"

        _update_stats(year_stats[year], pred_label, true_label)
        _update_stats(conf_year_stats[conference][year], pred_label, true_label)

    return dict(year_stats), {c: dict(v) for c, v in conf_year_stats.items()}


def calc_metrics(stats: dict) -> dict:
    """Calculate accuracy, accept recall, reject recall from stats."""
    tp = stats.get("tp", 0)
    tn = stats.get("tn", 0)
    fp = stats.get("fp", 0)
    fn = stats.get("fn", 0)
    total = stats.get("total", 0)

    if total == 0:
        return {"accuracy": None, "accept_recall": None, "reject_recall": None, "n": 0}

    accuracy = (tp + tn) / total * 100 if total > 0 else None
    accept_recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else None
    reject_recall = tn / (tn + fp) * 100 if (tn + fp) > 0 else None

    return {
        "accuracy": accuracy,
        "accept_recall": accept_recall,
        "reject_recall": reject_recall,
        "n": total,
    }


def aggregate_stats(year_stats: dict, years: list[int]) -> dict:
    """Aggregate stats across specified years."""
    agg = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0}
    for year in years:
        if year in year_stats:
            stats = year_stats[year]
            agg["tp"] += stats["tp"]
            agg["tn"] += stats["tn"]
            agg["fp"] += stats["fp"]
            agg["fn"] += stats["fn"]
            agg["total"] += stats["total"]
    return agg


def format_combined_value(values: list[Optional[float]], fmt: str = ".1f") -> str:
    """Format multiple values as val1/val2/val3."""
    formatted = []
    for v in values:
        if v is None:
            formatted.append("-")
        else:
            formatted.append(f"{v:{fmt}}")
    return "/".join(formatted)


def get_ckpt_short_name(filename: str) -> str:
    """Extract short checkpoint name from filename."""
    match = re.search(r"ckpt-(\d+)", filename)
    if match:
        return match.group(1)
    return filename.replace(".jsonl", "")


def print_combined_per_year_table(all_year_stats: list[dict], ckpt_names: list[str]):
    """Print combined per-year metrics table."""
    print("\n--- Per-Year Metrics ---")
    print(f"Checkpoints: {' / '.join(ckpt_names)}")

    # Determine which years have data
    all_years = set()
    for year_stats in all_year_stats:
        all_years.update(year_stats.keys())
    years = sorted(all_years)

    if not years:
        print("No data available.")
        return

    # Compute metrics for each checkpoint and year
    all_metrics = []
    for year_stats in all_year_stats:
        metrics_by_year = {}
        for y in years:
            if y in year_stats:
                metrics_by_year[y] = calc_metrics(year_stats[y])
            else:
                metrics_by_year[y] = calc_metrics({})
        all_metrics.append(metrics_by_year)

    # Calculate column width
    num_ckpts = len(ckpt_names)
    col_width = max(8, num_ckpts * 5 + 2)

    # Header
    year_cols = [str(y) for y in years]
    header = f"{'Metric':<20}" + "".join(f"{y:>{col_width}}" for y in year_cols)
    print(header)
    print("-" * len(header))

    # Accuracy
    row = f"{'Accuracy (%)':<20}"
    for y in years:
        values = [m[y]["accuracy"] for m in all_metrics]
        row += f"{format_combined_value(values):>{col_width}}"
    print(row)

    # Accept Recall
    row = f"{'Accept Recall (%)':<20}"
    for y in years:
        values = [m[y]["accept_recall"] for m in all_metrics]
        row += f"{format_combined_value(values):>{col_width}}"
    print(row)

    # Reject Recall
    row = f"{'Reject Recall (%)':<20}"
    for y in years:
        values = [m[y]["reject_recall"] for m in all_metrics]
        row += f"{format_combined_value(values):>{col_width}}"
    print(row)

    # N
    row = f"{'N':<20}"
    for y in years:
        val = all_metrics[0][y]["n"]
        row += f"{val:>{col_width}}"
    print(row)


def print_combined_overall_table(all_year_stats: list[dict], ckpt_names: list[str]):
    """Print combined overall aggregated metrics."""
    print("\n--- Overall Metrics ---")
    print(f"Checkpoints: {' / '.join(ckpt_names)}")

    # Compute metrics for each checkpoint (all years)
    all_metrics = []
    for year_stats in all_year_stats:
        all_years = sorted(year_stats.keys())
        stats_all = aggregate_stats(year_stats, all_years)
        all_metrics.append(calc_metrics(stats_all))

    values = [m["accuracy"] for m in all_metrics]
    print(f"{'Accuracy (%):':<20}{format_combined_value(values)}")

    values = [m["accept_recall"] for m in all_metrics]
    print(f"{'Accept Recall (%):':<20}{format_combined_value(values)}")

    values = [m["reject_recall"] for m in all_metrics]
    print(f"{'Reject Recall (%):':<20}{format_combined_value(values)}")

    print(f"{'N:':<20}{all_metrics[0]['n']}")

    # Compute metrics excluding 2026
    all_metrics_no2026 = []
    for year_stats in all_year_stats:
        years_no2026 = [y for y in sorted(year_stats.keys()) if y != 2026]
        if years_no2026:
            stats_no2026 = aggregate_stats(year_stats, years_no2026)
            all_metrics_no2026.append(calc_metrics(stats_no2026))
        else:
            all_metrics_no2026.append(calc_metrics({}))

    if all_metrics_no2026 and all_metrics_no2026[0]['n'] > 0:
        print("\n--- Overall Metrics (excluding 2026) ---")
        values = [m["accuracy"] for m in all_metrics_no2026]
        print(f"{'Accuracy (%):':<20}{format_combined_value(values)}")

        values = [m["accept_recall"] for m in all_metrics_no2026]
        print(f"{'Accept Recall (%):':<20}{format_combined_value(values)}")

        values = [m["reject_recall"] for m in all_metrics_no2026]
        print(f"{'Reject Recall (%):':<20}{format_combined_value(values)}")

        print(f"{'N:':<20}{all_metrics_no2026[0]['n']}")

    # Compute metrics for only 2025 and 2026
    has_2026 = any(2026 in ys for ys in all_year_stats)
    if has_2026:
        all_metrics_2025_2026 = []
        for year_stats in all_year_stats:
            stats_2025_2026 = aggregate_stats(year_stats, [2025, 2026])
            all_metrics_2025_2026.append(calc_metrics(stats_2025_2026))

        print("\n--- Overall Metrics (only 2025 and 2026) ---")
        values = [m["accuracy"] for m in all_metrics_2025_2026]
        print(f"{'Accuracy (%):':<20}{format_combined_value(values)}")
        values = [m["accept_recall"] for m in all_metrics_2025_2026]
        print(f"{'Accept Recall (%):':<20}{format_combined_value(values)}")
        values = [m["reject_recall"] for m in all_metrics_2025_2026]
        print(f"{'Reject Recall (%):':<20}{format_combined_value(values)}")
        print(f"{'N:':<20}{all_metrics_2025_2026[0]['n']}")


def print_conference_breakdown(all_conf_year_stats: list[dict], ckpt_names: list[str]):
    """Print accuracy breakdown by conference, then ICLR 2025/2026 detail."""
    conferences = set()
    for cys in all_conf_year_stats:
        conferences.update(cys.keys())
    if len(conferences) <= 1:
        return  # No point in conference breakdown for single-conference data

    conferences = sorted(conferences)
    num_ckpts = len(ckpt_names)
    col_w = max(8, 7)

    # --- By Conference (all years) ---
    print(f"\n--- Accuracy by Conference ---")
    header = f"{'Conference':<12}{'N':>6}" + "".join(f"{'ep' + str(i+1):>{col_w}}" for i in range(num_ckpts))
    print(header)
    print("-" * len(header))

    for conf in conferences:
        # Aggregate all years for this conference
        all_years_for_conf = set()
        for cys in all_conf_year_stats:
            if conf in cys:
                all_years_for_conf.update(cys[conf].keys())
        all_years_list = sorted(all_years_for_conf)

        n = None
        vals = []
        for cys in all_conf_year_stats:
            if conf in cys:
                agg = aggregate_stats(cys[conf], all_years_list)
                m = calc_metrics(agg)
                vals.append(m["accuracy"])
                if n is None:
                    n = m["n"]
            else:
                vals.append(None)

        row = f"{conf:<12}{n or 0:>6}" + "".join(
            f"{(f'{v:.1f}%' if v is not None else '-'):>{col_w}}" for v in vals
        )
        print(row)

    # --- ICLR 2025/2026 detail ---
    if "iclr" in conferences:
        print(f"\n--- ICLR 2025/2026 Detail ---")
        header = f"{'Year':<12}{'N':>6}" + "".join(f"{'ep' + str(i+1):>{col_w}}" for i in range(num_ckpts))
        print(header)
        print("-" * len(header))

        for year in [2025, 2026]:
            n = None
            vals = []
            for cys in all_conf_year_stats:
                if "iclr" in cys and year in cys["iclr"]:
                    m = calc_metrics(cys["iclr"][year])
                    vals.append(m["accuracy"])
                    if n is None:
                        n = m["n"]
                else:
                    vals.append(None)
            row = f"{year:<12}{n or 0:>6}" + "".join(
                f"{(f'{v:.1f}%' if v is not None else '-'):>{col_w}}" for v in vals
            )
            print(row)

        # Combined 2025+2026
        n = None
        vals = []
        for cys in all_conf_year_stats:
            if "iclr" in cys:
                agg = aggregate_stats(cys["iclr"], [2025, 2026])
                m = calc_metrics(agg)
                vals.append(m["accuracy"])
                if n is None:
                    n = m["n"]
            else:
                vals.append(None)
        row = f"{'25+26':<12}{n or 0:>6}" + "".join(
            f"{(f'{v:.1f}%' if v is not None else '-'):>{col_w}}" for v in vals
        )
        print(row)

    # --- ICLR 2017-2019 detail (if present) ---
    if "iclr" in conferences:
        early_years = [2017, 2018, 2019]
        has_early = any(
            y in cys.get("iclr", {}) for cys in all_conf_year_stats for y in early_years
        )
        if has_early:
            print(f"\n--- ICLR 2017-2019 Detail ---")
            header = f"{'Year':<12}{'N':>6}" + "".join(f"{'ep' + str(i+1):>{col_w}}" for i in range(num_ckpts))
            print(header)
            print("-" * len(header))

            for year in early_years:
                n = None
                vals = []
                for cys in all_conf_year_stats:
                    if "iclr" in cys and year in cys["iclr"]:
                        m = calc_metrics(cys["iclr"][year])
                        vals.append(m["accuracy"])
                        if n is None:
                            n = m["n"]
                    else:
                        vals.append(None)
                row = f"{year:<12}{n or 0:>6}" + "".join(
                    f"{(f'{v:.1f}%' if v is not None else '-'):>{col_w}}" for v in vals
                )
                print(row)

            # Combined 2017-2019
            n = None
            vals = []
            for cys in all_conf_year_stats:
                if "iclr" in cys:
                    agg = aggregate_stats(cys["iclr"], early_years)
                    m = calc_metrics(agg)
                    vals.append(m["accuracy"])
                    if n is None:
                        n = m["n"]
                else:
                    vals.append(None)
            if n and n > 0:
                row = f"{'17-19':<12}{n:>6}" + "".join(
                    f"{(f'{v:.1f}%' if v is not None else '-'):>{col_w}}" for v in vals
                )
                print(row)

    # --- Non-ICLR conferences by year ---
    non_iclr = [c for c in conferences if c != "iclr"]
    if non_iclr:
        for conf in non_iclr:
            all_years_for_conf = set()
            for cys in all_conf_year_stats:
                if conf in cys:
                    all_years_for_conf.update(cys[conf].keys())
            if not all_years_for_conf:
                continue
            years_sorted = sorted(all_years_for_conf)

            print(f"\n--- {conf.upper()} by Year ---")
            header = f"{'Year':<12}{'N':>6}" + "".join(f"{'ep' + str(i+1):>{col_w}}" for i in range(num_ckpts))
            print(header)
            print("-" * len(header))

            for year in years_sorted:
                n = None
                vals = []
                for cys in all_conf_year_stats:
                    if conf in cys and year in cys[conf]:
                        m = calc_metrics(cys[conf][year])
                        vals.append(m["accuracy"])
                        if n is None:
                            n = m["n"]
                    else:
                        vals.append(None)
                row = f"{year:<12}{n or 0:>6}" + "".join(
                    f"{(f'{v:.1f}%' if v is not None else '-'):>{col_w}}" for v in vals
                )
                print(row)


def analyze_directory(result_dir: Path, data_root: Path, dataset_name: Optional[str] = None) -> tuple[list[dict], Optional[dict]]:
    """Analyze all checkpoints in a result directory with combined tables.

    Returns (csv_rows, best_2025_2026_info) where best_2025_2026_info is None
    when no 2026 data exists, or {"variant": str, "ckpt": str, "accuracy": float}.
    """
    short_name = result_dir.name
    csv_rows = []

    # Find all checkpoint/prediction files, sorted numerically
    def ckpt_sort_key(f):
        match = re.search(r"ckpt-(\d+)", f.name)
        return int(match.group(1)) if match else float('inf')

    ckpt_files = sorted(result_dir.glob("*.jsonl"), key=ckpt_sort_key)

    if not ckpt_files:
        print(f"No checkpoint files found in {result_dir}")
        return csv_rows, None

    # Resolve dataset name: explicit override > DATASET_MAPPINGS lookup
    if dataset_name is None:
        if short_name not in DATASET_MAPPINGS:
            print(f"Error: Unknown dataset '{short_name}'")
            print(f"  Extracted short_name: {short_name}")
            return csv_rows, None
        dataset_name = DATASET_MAPPINGS[short_name]
    data_dir = data_root / f"{dataset_name}_test"

    if not data_dir.exists():
        print(f"Error: Test data directory not found: {data_dir}")
        return csv_rows, None

    # Load test data once
    try:
        test_data = load_test_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return csv_rows, None

    # Load and compute stats for all checkpoints
    all_year_stats = []
    all_conf_year_stats = []
    ckpt_names = []
    for ckpt_file in ckpt_files:
        try:
            predictions = load_predictions(ckpt_file)
            year_stats, conf_year_stats = compute_stats(predictions, test_data)
            all_year_stats.append(year_stats)
            all_conf_year_stats.append(conf_year_stats)
            ckpt_names.append(get_ckpt_short_name(ckpt_file.name))
        except Exception as e:
            print(f"Error loading {ckpt_file}: {e}")
            continue

    if not all_year_stats:
        print(f"No valid checkpoints found in {result_dir}")
        return csv_rows, None

    # Print header
    print("=" * 80)
    print(f"{short_name}")
    print(f"Checkpoints: {', '.join([f.name for f in ckpt_files])}")
    print("=" * 80)

    # Print combined tables
    print_combined_per_year_table(all_year_stats, ckpt_names)
    print_combined_overall_table(all_year_stats, ckpt_names)

    # Print conference breakdown if multi-conference data
    print_conference_breakdown(all_conf_year_stats, ckpt_names)

    print()

    # Build CSV rows
    for ckpt_idx, (year_stats, ckpt_name) in enumerate(zip(all_year_stats, ckpt_names)):
        for year in sorted(year_stats.keys()):
            metrics = calc_metrics(year_stats[year])
            csv_rows.append({
                "variant": short_name,
                "checkpoint": ckpt_name,
                "epoch": ckpt_idx + 1,
                "year": year,
                "accuracy": metrics["accuracy"],
                "accept_recall": metrics["accept_recall"],
                "reject_recall": metrics["reject_recall"],
                "n": metrics["n"],
            })

    # Compute best 2025+2026 accuracy across checkpoints
    best_2025_2026 = None
    has_2026 = any(2026 in ys for ys in all_year_stats)
    if has_2026:
        for year_stats, ckpt_name in zip(all_year_stats, ckpt_names):
            stats = aggregate_stats(year_stats, [2025, 2026])
            m = calc_metrics(stats)
            if m["accuracy"] is not None:
                if best_2025_2026 is None or m["accuracy"] > best_2025_2026["accuracy"]:
                    best_2025_2026 = {"variant": short_name, "ckpt": ckpt_name, "accuracy": m["accuracy"]}

    return csv_rows, best_2025_2026


def write_csv(rows: list[dict], output_path: Path):
    """Write CSV rows to a file."""
    if not rows:
        return
    fieldnames = ["variant", "checkpoint", "epoch", "year", "accuracy", "accept_recall", "reject_recall", "n"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV: {output_path} ({len(rows)} rows)")


def analyze_all(data_root: Path, csv_dir: Optional[Path] = None):
    """Analyze all result directories in final_sweep_v7_datasweepv3."""
    results_dir = Path("results/final_sweep_v7_datasweepv3")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print(f"\nScanning: {results_dir}")

    # Analyze top-level short_name dirs (dataset sweep variants)
    toplevel_rows = []
    toplevel_best_infos = []
    for short_name in sorted(DATASET_MAPPINGS.keys()):
        result_dir = results_dir / short_name
        if result_dir.exists():
            rows, best_info = analyze_directory(result_dir, data_root)
            toplevel_rows.extend(rows)
            if best_info is not None:
                toplevel_best_infos.append(best_info)

    if toplevel_best_infos:
        overall_best = max(toplevel_best_infos, key=lambda x: x["accuracy"])
        print(f"\n--- Overall Metrics (only 2025 and 2026) --- <--------- MAX: "
              f"{overall_best['variant']} ckpt-{overall_best['ckpt']} = {overall_best['accuracy']:.1f}%")

    if csv_dir and toplevel_rows:
        write_csv(toplevel_rows, csv_dir / "dataset_sweep_metrics.csv")

    # Scan all configured subdirectories
    for subdir_name, dataset_resolver in SUBDIR_CONFIGS:
        subdir = results_dir / subdir_name
        if not subdir.exists():
            continue
        # Find all variant directories (skip files like .png, _plots)
        variant_dirs = sorted(
            d for d in subdir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
        if not variant_dirs:
            continue
        print(f"\nScanning: {subdir}")
        subdir_rows = []
        subdir_best_infos = []
        for variant_dir in variant_dirs:
            ds_name = dataset_resolver(variant_dir.name)
            rows, best_info = analyze_directory(variant_dir, data_root, dataset_name=ds_name)
            subdir_rows.extend(rows)
            if best_info is not None:
                subdir_best_infos.append(best_info)

        if subdir_best_infos:
            overall_best = max(subdir_best_infos, key=lambda x: x["accuracy"])
            print(f"\n--- Overall Metrics (only 2025 and 2026) --- <--------- MAX: "
                  f"{overall_best['variant']} ckpt-{overall_best['ckpt']} = {overall_best['accuracy']:.1f}%")

        if csv_dir and subdir_rows:
            write_csv(subdir_rows, csv_dir / f"{subdir_name}_metrics.csv")

    # Scan extra result directories (not under final_sweep_v7_datasweepv3)
    for extra_dir, dataset_resolver in EXTRA_RESULT_CONFIGS:
        extra_path = Path(extra_dir)
        if not extra_path.exists():
            continue
        variant_dirs = sorted(
            d for d in extra_path.iterdir()
            if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
        )
        if not variant_dirs:
            continue
        print(f"\nScanning: {extra_path}")
        extra_rows = []
        extra_best_infos = []
        for variant_dir in variant_dirs:
            ds_name = dataset_resolver(variant_dir.name)
            rows, best_info = analyze_directory(variant_dir, data_root, dataset_name=ds_name)
            extra_rows.extend(rows)
            if best_info is not None:
                extra_best_infos.append(best_info)

        if extra_best_infos:
            overall_best = max(extra_best_infos, key=lambda x: x["accuracy"])
            print(f"\n--- Overall Metrics (only 2025 and 2026) --- <--------- MAX: "
                  f"{overall_best['variant']} ckpt-{overall_best['ckpt']} = {overall_best['accuracy']:.1f}%")

        if csv_dir and extra_rows:
            safe_name = extra_dir.replace("/", "_")
            write_csv(extra_rows, csv_dir / f"{safe_name}_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description="Analyze final_sweep_v7_datasweepv3 results with per-year breakdown")
    parser.add_argument("path", nargs="*", type=Path, help="Result directory/directories")
    parser.add_argument("--all", action="store_true", help="Analyze all available results")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory for per-experiment CSV files with per-year metrics",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for test data (default: data)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override test dataset name (e.g. all_conferences_text_v7_extra)",
    )
    args = parser.parse_args()

    if args.all:
        analyze_all(args.data_root, csv_dir=args.csv)
    elif args.path:
        all_rows = []
        for p in args.path:
            if p.is_dir():
                ds_name = args.dataset
                if ds_name is None:
                    # Try to infer dataset from parent directory via SUBDIR_CONFIGS
                    for subdir_name, resolver in SUBDIR_CONFIGS:
                        subdir_parts = Path(subdir_name).parts
                        parent_parts = p.parent.parts
                        if len(parent_parts) >= len(subdir_parts) and parent_parts[-len(subdir_parts):] == subdir_parts:
                            ds_name = resolver(p.name)
                            break
                    # Also check EXTRA_RESULT_CONFIGS
                    if ds_name is None:
                        for extra_dir, resolver in EXTRA_RESULT_CONFIGS:
                            extra_parts = Path(extra_dir).parts
                            parent_parts = p.parent.parts
                            if len(parent_parts) >= len(extra_parts) and parent_parts[-len(extra_parts):] == extra_parts:
                                ds_name = resolver(p.name)
                                break
                rows, _ = analyze_directory(p, args.data_root, dataset_name=ds_name)
                all_rows.extend(rows)
            else:
                print(f"Error: {p} is not a directory")
        if args.csv and all_rows:
            write_csv(all_rows, args.csv / "manual_metrics.csv")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
