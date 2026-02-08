#!/usr/bin/env python3
"""
Analyze v7 sweep results with per-year breakdown.

Supports two model types:
1. SFT models: predictions with "predict" field (text generation)
2. CLS models: predictions with "logit"/"prob"/"pred" fields (binary classification)

Shows combined tables across all checkpoints:
1. Per-year metrics
2. Overall metrics

Each cell shows values from all checkpoints as: ckpt1/ckpt2/ckpt3...

Usage:
    python scripts/analyze_v7.py results/final_sweep_v7/balanced_clean/
    python scripts/analyze_v7.py results/lr_experiment_v7/text_trainagreeing_no2024_lr_1.75e6_bs16_3epoch/
    python scripts/analyze_v7.py --all  # Analyze all available results
    python scripts/analyze_v7.py --cls saves/lr_experiment_v7/text_*/  # Analyze CLS training curves
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


# Dataset mappings: short_name -> dataset_name
DATASET_MAPPINGS = {
    # Text-only
    "balanced_clean": "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7",
    "balanced_title_abstract": "iclr_2020_2025_85_5_10_split7_balanced_clean_title_abstract_binary_noreviews_v7",
    "balanced_trainagreeing": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7",
    "balanced_2024_2025": "iclr_2024_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7",
    "balanced_2017_2025": "iclr_2017_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7",
    # Vision
    "balanced_vision": "iclr_2020_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_trainagreeing_vision": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "balanced_2024_2025_vision": "iclr_2024_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_2017_2025_vision": "iclr_2017_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_no2024_vision": "iclr_2020_2023_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_trainagreeing_no2024_vision": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "balanced_trainagreeing_no2024_vision_lr1e-6": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "balanced_trainagreeing_no2024_vision_lr2e-6_10epoch": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "balanced_trainagreeing_no2024_vision_yesno_trainacc": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_yesno_noreviews_v7",
    # Images
    "balanced_clean_images": "iclr_2020_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7",
    "balanced_trainagreeing_images": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_images_binary_noreviews_v7",
    "balanced_2024_2025_images": "iclr_2024_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7",
    "balanced_2017_2025_images": "iclr_2017_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7",
    # Think tokens experiments (vision trainagreeing no2024)
    "trainagreeing_vision_think10_input": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "trainagreeing_vision_think10_label": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "trainagreeing_vision_think100_input": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "trainagreeing_vision_think100_label": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "trainagreeing_vision_think1000_input": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "trainagreeing_vision_think1000_label": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
}

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


def is_cls_prediction(pred: dict) -> bool:
    """Check if prediction is from CLS model (has logit/prob/pred fields)."""
    return "logit" in pred and "pred" in pred


def compute_stats(predictions: list[dict], test_data: list[dict] = None) -> dict:
    """Compute per-year statistics.

    Handles both SFT predictions (with "predict" text field) and
    CLS predictions (with "logit"/"prob"/"pred" fields).

    For CLS models, test_data can be None as metadata is embedded in predictions.
    """
    # Detect if this is CLS format
    is_cls = len(predictions) > 0 and is_cls_prediction(predictions[0])

    if not is_cls:
        # SFT format - need test_data for metadata
        if test_data is None:
            raise ValueError("test_data required for SFT predictions")
        if len(predictions) != len(test_data):
            print(f"Warning: predictions ({len(predictions)}) != test_data ({len(test_data)})")
            min_len = min(len(predictions), len(test_data))
            predictions = predictions[:min_len]
            test_data = test_data[:min_len]

    year_stats = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0})

    for i, pred in enumerate(predictions):
        # Get metadata - for CLS it's embedded, for SFT it's in test_data
        if is_cls:
            metadata = pred.get("_metadata", {})
        else:
            metadata = test_data[i].get("_metadata", {})

        year = metadata.get("year")
        if year is None:
            continue

        # Get prediction and label based on format
        if is_cls:
            # CLS format: pred is 0/1
            # Use metadata.answer for ground truth (more reliable than label field)
            pred_val = pred.get("pred", -1)
            if pred_val == -1:
                continue
            pred_label = "accept" if pred_val == 1 else "reject"

            # Get ground truth from metadata.answer (fallback to label field)
            answer = metadata.get("answer", "").lower()
            if "accept" in answer:
                true_label = "accept"
            elif "reject" in answer:
                true_label = "reject"
            else:
                # Fallback to label field if metadata not available
                label_val = pred.get("label", -1)
                if label_val == -1:
                    continue
                true_label = "accept" if label_val == 1 else "reject"
        else:
            # SFT format: extract from text
            pred_label = extract_prediction(pred.get("predict", ""))
            true_label = extract_label(pred.get("label", ""))

        if pred_label == "unknown" or true_label == "unknown":
            continue

        year_stats[year]["total"] += 1

        if true_label == "accept":
            if pred_label == "accept":
                year_stats[year]["tp"] += 1
            else:
                year_stats[year]["fn"] += 1
        else:
            if pred_label == "reject":
                year_stats[year]["tn"] += 1
            else:
                year_stats[year]["fp"] += 1

    return dict(year_stats)


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

    # Compute metrics excluding 2024
    all_metrics_no2024 = []
    for year_stats in all_year_stats:
        years_no2024 = [y for y in sorted(year_stats.keys()) if y != 2024]
        if years_no2024:
            stats_no2024 = aggregate_stats(year_stats, years_no2024)
            all_metrics_no2024.append(calc_metrics(stats_no2024))
        else:
            all_metrics_no2024.append(calc_metrics({}))

    if all_metrics_no2024 and all_metrics_no2024[0]['n'] > 0:
        print("\n--- Overall Metrics (excluding 2024) ---")
        values = [m["accuracy"] for m in all_metrics_no2024]
        print(f"{'Accuracy (%):':<20}{format_combined_value(values)}")

        values = [m["accept_recall"] for m in all_metrics_no2024]
        print(f"{'Accept Recall (%):':<20}{format_combined_value(values)}")

        values = [m["reject_recall"] for m in all_metrics_no2024]
        print(f"{'Reject Recall (%):':<20}{format_combined_value(values)}")

        print(f"{'N:':<20}{all_metrics_no2024[0]['n']}")


def load_trainer_state(save_dir: Path) -> Optional[dict]:
    """Load trainer_state.json from a saves directory."""
    state_file = save_dir / "trainer_state.json"
    if not state_file.exists():
        return None
    with open(state_file) as f:
        return json.load(f)


def extract_training_metrics(trainer_state: dict) -> dict:
    """Extract training and eval metrics from trainer_state log_history."""
    log_history = trainer_state.get("log_history", [])

    train_metrics = []
    eval_metrics = []

    for entry in log_history:
        step = entry.get("step", 0)
        epoch = entry.get("epoch", 0)

        # Training metrics (has "loss" but not "eval_loss")
        if "loss" in entry and "eval_loss" not in entry:
            train_metrics.append({
                "step": step,
                "epoch": epoch,
                "loss": entry.get("loss"),
                "cls_loss": entry.get("cls_loss"),
                "accuracy": entry.get("accuracy"),
                "loss_bce": entry.get("loss_bce"),
                "loss_rating": entry.get("loss_rating"),
            })

        # Eval metrics (has "eval_loss")
        if "eval_loss" in entry:
            eval_metrics.append({
                "step": step,
                "epoch": epoch,
                "eval_loss": entry.get("eval_loss"),
                "eval_cls_loss": entry.get("eval_cls_loss"),
                "eval_accuracy": entry.get("eval_accuracy"),
                "eval_loss_bce": entry.get("eval_loss_bce"),
                "eval_loss_rating": entry.get("eval_loss_rating"),
            })

    return {"train": train_metrics, "eval": eval_metrics}


def print_cls_training_summary(save_dir: Path):
    """Print training summary for a CLS model from trainer_state.json."""
    trainer_state = load_trainer_state(save_dir)
    if trainer_state is None:
        print(f"No trainer_state.json found in {save_dir}")
        return

    metrics = extract_training_metrics(trainer_state)
    train_metrics = metrics["train"]
    eval_metrics = metrics["eval"]

    print(f"\n{'=' * 80}")
    print(f"CLS Training Summary: {save_dir.name}")
    print(f"{'=' * 80}")

    # Training curve summary
    if train_metrics:
        print("\n--- Training Metrics ---")
        print(f"{'Step':>8} {'Epoch':>8} {'Loss':>10} {'Accuracy':>10} {'BCE':>10} {'Rating':>10}")
        print("-" * 66)
        # Show first, middle, and last few entries
        indices = list(range(min(3, len(train_metrics))))
        if len(train_metrics) > 6:
            indices += list(range(len(train_metrics) - 3, len(train_metrics)))
        else:
            indices = list(range(len(train_metrics)))

        for i in indices:
            m = train_metrics[i]
            loss = f"{m['loss']:.4f}" if m['loss'] else "-"
            acc = f"{m['accuracy']*100:.1f}%" if m['accuracy'] else "-"
            bce = f"{m['loss_bce']:.4f}" if m.get('loss_bce') else "-"
            rating = f"{m['loss_rating']:.4f}" if m.get('loss_rating') else "-"
            print(f"{m['step']:>8} {m['epoch']:>8.2f} {loss:>10} {acc:>10} {bce:>10} {rating:>10}")

    # Eval metrics summary
    if eval_metrics:
        print("\n--- Eval Metrics (per epoch) ---")
        print(f"{'Step':>8} {'Epoch':>8} {'Loss':>10} {'Accuracy':>10} {'BCE':>10} {'Rating':>10}")
        print("-" * 66)
        for m in eval_metrics:
            loss = f"{m['eval_loss']:.4f}" if m['eval_loss'] else "-"
            acc = f"{m['eval_accuracy']*100:.1f}%" if m.get('eval_accuracy') else "-"
            bce = f"{m['eval_loss_bce']:.4f}" if m.get('eval_loss_bce') else "-"
            rating = f"{m['eval_loss_rating']:.4f}" if m.get('eval_loss_rating') else "-"
            print(f"{m['step']:>8} {m['epoch']:>8.2f} {loss:>10} {acc:>10} {bce:>10} {rating:>10}")

        # Best eval accuracy
        best_eval = max(eval_metrics, key=lambda x: x.get('eval_accuracy', 0) or 0)
        if best_eval.get('eval_accuracy'):
            print(f"\nBest eval accuracy: {best_eval['eval_accuracy']*100:.2f}% at epoch {best_eval['epoch']:.1f}")


def analyze_cls_predictions(pred_file: Path) -> dict:
    """Analyze CLS predictions file and return per-year stats."""
    predictions = load_predictions(pred_file)
    if not predictions:
        return {}

    # CLS predictions have metadata embedded
    return compute_stats(predictions, test_data=None)


def analyze_directory(result_dir: Path, data_root: Path):
    """Analyze all checkpoints in a result directory with combined tables.

    Handles both SFT and CLS prediction formats automatically.
    """
    short_name = result_dir.name

    # Find all checkpoint/prediction files, sorted numerically
    def ckpt_sort_key(f):
        match = re.search(r"ckpt-(\d+)", f.name)
        return int(match.group(1)) if match else float('inf')

    ckpt_files = sorted(result_dir.glob("*.jsonl"), key=ckpt_sort_key)

    if not ckpt_files:
        print(f"No checkpoint files found in {result_dir}")
        return

    # Load first prediction to detect format
    try:
        first_preds = load_predictions(ckpt_files[0])
        is_cls = len(first_preds) > 0 and is_cls_prediction(first_preds[0])
    except Exception as e:
        print(f"Error loading {ckpt_files[0]}: {e}")
        return

    # For SFT, we need the dataset mapping and test data
    test_data = None
    if not is_cls:
        if short_name not in DATASET_MAPPINGS:
            print(f"Error: Unknown dataset '{short_name}'")
            print(f"  Extracted short_name: {short_name}")
            return

        dataset_name = DATASET_MAPPINGS[short_name]
        data_dir = data_root / f"{dataset_name}_test"

        if not data_dir.exists():
            print(f"Error: Test data directory not found: {data_dir}")
            return

        # Load test data once
        try:
            test_data = load_test_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading test data: {e}")
            return

    # Load and compute stats for all checkpoints
    all_year_stats = []
    ckpt_names = []
    for ckpt_file in ckpt_files:
        try:
            predictions = load_predictions(ckpt_file)
            year_stats = compute_stats(predictions, test_data)
            all_year_stats.append(year_stats)
            ckpt_names.append(get_ckpt_short_name(ckpt_file.name))
        except Exception as e:
            print(f"Error loading {ckpt_file}: {e}")
            continue

    if not all_year_stats:
        print(f"No valid checkpoints found in {result_dir}")
        return

    # Print header
    model_type = "CLS" if is_cls else "SFT"
    print("=" * 80)
    print(f"{short_name} ({model_type})")
    print(f"Checkpoints: {', '.join([f.name for f in ckpt_files])}")
    print("=" * 80)

    # Print combined tables
    print_combined_per_year_table(all_year_stats, ckpt_names)
    print_combined_overall_table(all_year_stats, ckpt_names)
    print()


def analyze_all(data_root: Path):
    """Analyze all result directories."""
    results_dirs = [
        Path("results/final_sweep_v7"),
        Path("results/final_sweep_v7_pli"),
    ]

    for results_dir in results_dirs:
        if not results_dir.exists():
            continue

        print(f"\nScanning: {results_dir}")

        for short_name in sorted(DATASET_MAPPINGS.keys()):
            result_dir = results_dir / short_name
            if result_dir.exists():
                analyze_directory(result_dir, data_root)


def analyze_cls_saves(save_dirs: list[Path]):
    """Analyze CLS training curves from saves directories."""
    for save_dir in save_dirs:
        if not save_dir.is_dir():
            print(f"Skipping {save_dir} (not a directory)")
            continue
        print_cls_training_summary(save_dir)


def analyze_lr_experiment_results(results_root: Path, data_root: Path):
    """Analyze all CLS experiment results in lr_experiment_v7 directory."""
    if not results_root.exists():
        print(f"Results directory not found: {results_root}")
        return

    # Corresponding saves directory
    saves_root = Path(str(results_root).replace("results/", "saves/"))

    print(f"\n{'=' * 80}")
    print(f"CLS Experiment Results: {results_root}")
    print(f"{'=' * 80}")

    # Find all experiment directories
    exp_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])

    for exp_dir in exp_dirs:
        pred_file = exp_dir / "generated_predictions.jsonl"
        if not pred_file.exists():
            continue

        try:
            predictions = load_predictions(pred_file)
            if not predictions or not is_cls_prediction(predictions[0]):
                continue

            year_stats = compute_stats(predictions, test_data=None)

            # Calculate overall metrics
            all_years = sorted(year_stats.keys())
            stats_all = aggregate_stats(year_stats, all_years)
            metrics = calc_metrics(stats_all)

            acc = f"{metrics['accuracy']:.1f}%" if metrics['accuracy'] else "-"
            ar = f"{metrics['accept_recall']:.1f}%" if metrics['accept_recall'] else "-"
            rr = f"{metrics['reject_recall']:.1f}%" if metrics['reject_recall'] else "-"
            print(f"\n{exp_dir.name}:")
            print(f"  Final Test - Accuracy: {acc}, Accept Recall: {ar}, Reject Recall: {rr}, N: {metrics['n']}")

            # Try to load per-checkpoint eval accuracies from trainer_state.json
            save_dir = saves_root / exp_dir.name
            trainer_state = load_trainer_state(save_dir)
            if trainer_state:
                eval_accs = []
                for entry in trainer_state.get("log_history", []):
                    if "eval_accuracy" in entry:
                        step = entry.get("step", 0)
                        epoch = entry.get("epoch", 0)
                        eval_acc = entry.get("eval_accuracy", 0)
                        eval_accs.append((step, epoch, eval_acc))

                if eval_accs:
                    ckpt_strs = [f"ckpt-{s}: {a*100:.1f}%" for s, e, a in eval_accs]
                    print(f"  Per-Checkpoint Eval: {', '.join(ckpt_strs)}")

        except Exception as e:
            print(f"Error analyzing {exp_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze v7 sweep results with per-year breakdown")
    parser.add_argument("path", nargs="*", type=Path, help="Result or saves directory/directories")
    parser.add_argument("--all", action="store_true", help="Analyze all available results")
    parser.add_argument("--cls", action="store_true", help="Analyze CLS training curves from saves directories")
    parser.add_argument("--lr-exp", action="store_true", help="Analyze lr_experiment_v7 CLS results")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for test data (default: data)",
    )
    args = parser.parse_args()

    if args.all:
        analyze_all(args.data_root)
    elif args.lr_exp:
        analyze_lr_experiment_results(Path("results/lr_experiment_v7"), args.data_root)
    elif args.cls and args.path:
        analyze_cls_saves(args.path)
    elif args.path:
        for p in args.path:
            if p.is_dir():
                # Check if it's a saves dir (has trainer_state.json) or results dir
                if (p / "trainer_state.json").exists():
                    print_cls_training_summary(p)
                else:
                    analyze_directory(p, args.data_root)
            else:
                print(f"Error: {p} is not a directory")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
