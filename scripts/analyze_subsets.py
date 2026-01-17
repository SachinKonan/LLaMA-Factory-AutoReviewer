#!/usr/bin/env python3
"""
Analyze model accuracy on dataset subsets.

This script computes accuracy metrics when filtering predictions by metadata,
leveraging the hierarchical relationship between v6 datasets.

Dataset relationships (test sets):
- original_2020-2025 → balanced_2020-2025: Filter by submission_id
- 2017-2025 → 2020-2025: Filter by year >= 2020
- nips_balanced → iclr only: Filter by conference == 'iclr'
- nips_balanced → nips only: Filter by conference == 'nips'
- nips_accepts → iclr only: Filter by conference == 'iclr'

Usage:
    python scripts/analyze_subsets.py --results-dir data_sweep_v2
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# Mapping from result directory short names to actual test dataset names
RESULT_TO_TEST_DATASET = {
    "iclr20_balanced_clean": "iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6_test",
    "iclr20_balanced_vision": "iclr_2020_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6_test",
    "iclr20_trainagreeing_clean": "iclr_2020_2025_85_5_10_split6_balanced_trainagreeing_clean_binary_noreviews_v6_test",
    "iclr20_trainagreeing_vision": "iclr_2020_2025_85_5_10_split6_balanced_trainagreeing_vision_binary_noreviews_v6_test",
    "iclr20_original_clean": "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_test",
    "iclr20_original_vision": "iclr_2020_2025_85_5_10_split6_original_vision_binary_noreviews_v6_test",
    "iclr17_balanced_clean": "iclr_2017_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6_test",
    "iclr17_balanced_vision": "iclr_2017_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6_test",
    "iclr_nips_balanced_clean": "iclr_nips_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6_test",
    "iclr_nips_balanced_vision": "iclr_nips_2020_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6_test",
    "iclr_nips_accepts_clean": "iclr_nips_2020_2025_85_5_10_split6_balanced_nips_accepts_clean_binary_noreviews_v6_test",
    "iclr_nips_accepts_vision": "iclr_nips_2020_2025_85_5_10_split6_balanced_nips_accepts_vision_binary_noreviews_v6_test",
}

# Reference datasets for filtering by submission_id
BALANCED_CLEAN_REF = "iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6_test"
BALANCED_VISION_REF = "iclr_2020_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6_test"


def get_dataset_size(dataset_name: str, data_dir: str = "data") -> int | None:
    """Get the number of samples in a dataset."""
    path = Path(data_dir) / dataset_name / "data.json"
    if not path.exists():
        return None
    with open(path) as f:
        return len(json.load(f))


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    if not text:
        return None
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def normalize_label(label: str | None) -> str | None:
    """Normalize label to 'accepted' or 'rejected'."""
    if label is None:
        return None
    label_lower = label.lower()
    if label_lower in ["accept", "accepted"]:
        return "accepted"
    if label_lower in ["reject", "rejected"]:
        return "rejected"
    return None


def compute_binary_metrics(y_true: list, y_pred: list) -> dict:
    """Compute binary classification metrics."""
    num_tps = sum(1 for t, p in zip(y_true, y_pred) if t == "accepted" and p == "accepted")
    num_fps = sum(1 for t, p in zip(y_true, y_pred) if t == "rejected" and p == "accepted")
    num_fns = sum(1 for t, p in zip(y_true, y_pred) if t == "accepted" and p == "rejected")
    num_tns = sum(1 for t, p in zip(y_true, y_pred) if t == "rejected" and p == "rejected")

    total = num_tps + num_fps + num_fns + num_tns
    accuracy = (num_tps + num_tns) / total if total > 0 else 0.0
    accept_recall = num_tps / (num_tps + num_fns) if (num_tps + num_fns) > 0 else 0.0
    reject_recall = num_tns / (num_tns + num_fps) if (num_tns + num_fps) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "accept_recall": accept_recall,
        "reject_recall": reject_recall,
        "num_tps": num_tps,
        "num_fps": num_fps,
        "num_fns": num_fns,
        "num_tns": num_tns,
    }


def load_predictions(path: Path) -> list[dict]:
    """Load predictions from jsonl file."""
    predictions = []
    with open(path) as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def load_test_data(dataset_name: str, data_dir: str = "data") -> list[dict]:
    """Load test data from data.json file."""
    path = Path(data_dir) / dataset_name / "data.json"
    if not path.exists():
        raise FileNotFoundError(f"Test data not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_submission_ids(dataset_name: str, data_dir: str = "data") -> set[str]:
    """Get set of submission_ids from a test dataset."""
    data = load_test_data(dataset_name, data_dir)
    return {d["_metadata"]["submission_id"] for d in data}


def join_predictions_with_metadata(
    predictions: list[dict], test_data: list[dict]
) -> list[dict]:
    """Join predictions with test data metadata by index."""
    if len(predictions) != len(test_data):
        print(f"  Warning: predictions ({len(predictions)}) != test_data ({len(test_data)})")

    joined = []
    for i, (pred, data) in enumerate(zip(predictions, test_data)):
        label = normalize_label(extract_boxed_answer(pred.get("label", "")))
        predicted = normalize_label(extract_boxed_answer(pred.get("predict", "")))
        metadata = data.get("_metadata", {})
        joined.append({
            "index": i,
            "label": label,
            "predicted": predicted,
            "submission_id": metadata.get("submission_id"),
            "year": metadata.get("year"),
            "conference": metadata.get("conference"),
        })
    return joined


def format_combined(metrics: dict) -> str:
    """Format combined metrics as acc/acc_rec/rej_rec."""
    return f"{metrics['accuracy']:.2f}/{metrics['accept_recall']:.2f}/{metrics['reject_recall']:.2f}"


def compute_dist_breakdown(valid_data: list[dict]) -> tuple[str, str, str]:
    """Compute in-distribution (year < 2025) vs out-of-distribution (year = 2025) breakdown.

    Returns: (combined, in_dist, ood) formatted strings
    Format: acc/accept_recall/reject_recall
    """
    in_dist = [d for d in valid_data if d["year"] is not None and d["year"] < 2025]
    ood = [d for d in valid_data if d["year"] is not None and d["year"] == 2025]

    # Combined metrics
    y_true_all = [d["label"] for d in valid_data]
    y_pred_all = [d["predicted"] for d in valid_data]
    m_all = compute_binary_metrics(y_true_all, y_pred_all)
    combined_str = format_combined(m_all)

    def format_metrics(data: list[dict]) -> str:
        if len(data) == 0:
            return "N/A"
        y_true = [d["label"] for d in data]
        y_pred = [d["predicted"] for d in data]
        m = compute_binary_metrics(y_true, y_pred)
        return f"{m['accuracy']:.2f}/{m['accept_recall']:.2f}/{m['reject_recall']:.2f}(n={len(data)})"

    return combined_str, format_metrics(in_dist), format_metrics(ood)


def analyze_subset(
    joined_data: list[dict],
    filter_fn: callable,
    subset_name: str,
) -> dict | None:
    """Analyze accuracy on a filtered subset."""
    filtered = [d for d in joined_data if filter_fn(d)]

    # Filter to valid predictions only
    valid = [d for d in filtered if d["label"] is not None and d["predicted"] is not None]

    if len(valid) == 0:
        return None

    y_true = [d["label"] for d in valid]
    y_pred = [d["predicted"] for d in valid]

    metrics = compute_binary_metrics(y_true, y_pred)

    # Compute in-dist vs ood breakdown
    combined, in_dist, ood = compute_dist_breakdown(valid)

    num_accepted = sum(1 for d in valid if d["label"] == "accepted")
    accept_rate = num_accepted / len(valid)

    return {
        "subset": subset_name,
        "size": len(valid),
        "test_accept_rate": f"{accept_rate:.2f}",
        "combined": combined,
        "in_dist": in_dist,
        "ood": ood,
    }


def get_subset_analyses(result_name: str, data_dir: str = "data") -> list[tuple[str, callable]]:
    """
    Get list of (subset_name, filter_fn) for a given result directory.

    Returns analyses that make sense for this result type.
    """
    analyses = []

    # Determine modality from result name
    is_vision = result_name.endswith("_vision")
    ref_balanced = BALANCED_VISION_REF if is_vision else BALANCED_CLEAN_REF

    if result_name.startswith("iclr20_original"):
        # original → balanced: filter by submission_id in balanced
        balanced_ids = get_submission_ids(ref_balanced, data_dir)
        analyses.append((
            "original → balanced",
            lambda d, ids=balanced_ids: d["submission_id"] in ids
        ))

    elif result_name.startswith("iclr17_balanced"):
        # 2017-2025 → 2020-2025: filter by year >= 2020
        analyses.append((
            "2017-2025 → 2020-2025",
            lambda d: d["year"] is not None and d["year"] >= 2020
        ))
        # Also show 2017-2019 subset
        analyses.append((
            "2017-2025 → 2017-2019",
            lambda d: d["year"] is not None and d["year"] < 2020
        ))

    elif result_name.startswith("iclr_nips_balanced"):
        # nips_balanced → iclr: filter by conference == 'iclr'
        analyses.append((
            "nips_balanced → iclr",
            lambda d: d["conference"] == "iclr"
        ))
        # nips_balanced → nips: filter by conference == 'nips'
        analyses.append((
            "nips_balanced → nips",
            lambda d: d["conference"] == "nips"
        ))

    elif result_name.startswith("iclr_nips_accepts"):
        # nips_accepts → iclr: filter by conference == 'iclr'
        analyses.append((
            "nips_accepts → iclr",
            lambda d: d["conference"] == "iclr"
        ))
        # nips_accepts → nips: filter by conference == 'nips'
        analyses.append((
            "nips_accepts → nips",
            lambda d: d["conference"] == "nips"
        ))

    return analyses


def analyze_result_dir(
    result_name: str,
    results_dir: Path,
    data_dir: str = "data",
    verbose: bool = False,
) -> list[dict]:
    """Analyze all relevant subsets for a result directory."""
    results = []

    # Get test dataset name
    test_dataset = RESULT_TO_TEST_DATASET.get(result_name)
    if test_dataset is None:
        print(f"  Skipping (unknown result directory): {result_name}")
        return results

    # Get train dataset size
    train_dataset = test_dataset.replace("_test", "_train")
    train_size = get_dataset_size(train_dataset, data_dir)

    # Load predictions
    pred_path = results_dir / result_name / "finetuned.jsonl"
    if not pred_path.exists():
        print(f"  Skipping (no predictions): {pred_path}")
        return results

    # Load test data
    try:
        test_data = load_test_data(test_dataset, data_dir)
    except FileNotFoundError as e:
        print(f"  {e}")
        return results

    predictions = load_predictions(pred_path)

    if verbose:
        print(f"  Loaded {len(predictions)} predictions, {len(test_data)} test samples, train_size={train_size}")

    # Join predictions with metadata
    joined = join_predictions_with_metadata(predictions, test_data)

    # First, compute full accuracy
    valid_full = [d for d in joined if d["label"] is not None and d["predicted"] is not None]

    if verbose:
        print(f"  Valid predictions: {len(valid_full)}")

    if len(valid_full) > 0:
        combined, in_dist, ood = compute_dist_breakdown(valid_full)
        num_accepted = sum(1 for d in valid_full if d["label"] == "accepted")
        accept_rate = num_accepted / len(valid_full)
        results.append({
            "result": result_name,
            "subset": "(full)",
            "train_size": train_size,
            "test_size": len(valid_full),
            "test_accept_rate": f"{accept_rate:.2f}",
            "combined": combined,
            "in_dist": in_dist,
            "ood": ood,
        })

    # Get subset analyses for this result type
    analyses = get_subset_analyses(result_name, data_dir)

    if verbose:
        print(f"  Subset analyses: {[a[0] for a in analyses]}")

    for subset_name, filter_fn in analyses:
        result = analyze_subset(joined, filter_fn, subset_name)
        if result is not None:
            results.append({
                "result": result_name,
                "subset": result["subset"],
                "train_size": train_size,
                "test_size": result["size"],
                "test_accept_rate": result["test_accept_rate"],
                "combined": result["combined"],
                "in_dist": result["in_dist"],
                "ood": result["ood"],
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model accuracy on dataset subsets"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Results directory name under results/ (e.g., data_sweep_v2)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output for debugging",
    )
    args = parser.parse_args()

    results_dir = Path("results") / args.results_dir

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    all_results = []

    # Process each result subdirectory
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue

        result_name = subdir.name
        print(f"Processing: {result_name}")

        results = analyze_result_dir(result_name, results_dir, args.data_dir, args.verbose)
        all_results.extend(results)

        if args.verbose:
            print(f"  -> Generated {len(results)} result rows")

    if not all_results:
        print("\nNo results found.")
        return

    print(f"\nTotal result rows: {len(all_results)}")

    # Create dataframe
    df = pd.DataFrame(all_results)

    # Reorder columns
    column_order = ["result", "subset", "train_size", "test_size", "test_accept_rate", "combined", "in_dist", "ood"]
    df = df[[c for c in column_order if c in df.columns]]

    # Ensure full display
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print("\n" + "=" * 120)
    print(" SUBSET ANALYSIS RESULTS (format: acc/accept_recall/reject_recall)")
    print("=" * 120)
    print(df.to_string(index=False))
    print()

    # Save CSV
    csv_path = results_dir / "subset_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
