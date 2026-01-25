#!/usr/bin/env python3
"""
Analyze multi-prediction results (predict is a list of predictions per sample).

Usage:
    python scripts/analyze_mult.py \
        --dir /path/to/results \
        --file-pattern "_ailab" \
        --force-result-type binary
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    if not text:
        return None
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip().lower()
    return None


def normalize_binary(answer: str | None) -> str | None:
    """Normalize to 'accepted' or 'rejected'."""
    if answer is None:
        return None
    answer = answer.lower()
    if answer in ["accept", "accepted"]:
        return "accepted"
    if answer in ["reject", "rejected"]:
        return "rejected"
    return None


def get_majority_vote(predictions: list[str], tie_breaker: str = "rejected") -> str | None:
    """
    Get majority vote from list of raw predictions.

    Args:
        predictions: List of raw prediction strings
        tie_breaker: Value to return on tie (default: 'rejected' for conservative choice)

    Returns:
        'accepted', 'rejected', or None if no valid predictions
    """
    votes = [normalize_binary(extract_boxed_answer(p)) for p in predictions]
    votes = [v for v in votes if v is not None]
    if not votes:
        return None
    counts = Counter(votes)
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]
    if len(winners) > 1:
        return tie_breaker  # Tie -> conservative (reject)
    return winners[0]


def compute_vote_agreement(predictions: list[str]) -> float | None:
    """Compute fraction of predictions agreeing with majority vote."""
    votes = [normalize_binary(extract_boxed_answer(p)) for p in predictions]
    votes = [v for v in votes if v is not None]
    if not votes:
        return None
    counts = Counter(votes)
    max_count = max(counts.values())
    return max_count / len(votes)


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
        "n": total,
    }


def format_metrics(data: list[dict]) -> str:
    """Format metrics as acc/accept_recall/reject_recall(n=samples)."""
    if len(data) == 0:
        return "N/A"
    y_true = [d["label"] for d in data]
    y_pred = [d["mv_pred"] for d in data]
    m = compute_binary_metrics(y_true, y_pred)
    return f"{m['accuracy']:.2f}/{m['accept_recall']:.2f}/{m['reject_recall']:.2f}(n={m['n']})"


def build_metadata_map(jsonl_files: list[Path]) -> tuple[dict[str, dict], list[dict]]:
    """Build a map from submission_id -> metadata and index-based list from files that have metadata."""
    metadata_map = {}
    metadata_list = []

    for path in jsonl_files:
        with open(path) as f:
            for idx, line in enumerate(f):
                sample = json.loads(line)
                metadata = sample.get("_metadata", {})
                submission_id = metadata.get("submission_id")
                if submission_id and metadata.get("year") is not None:
                    if submission_id not in metadata_map:
                        metadata_map[submission_id] = metadata
                    # Also store by index (first file with metadata wins)
                    if len(metadata_list) <= idx:
                        metadata_list.extend([None] * (idx + 1 - len(metadata_list)))
                    if metadata_list[idx] is None:
                        metadata_list[idx] = metadata
    return metadata_map, metadata_list


def analyze_mult_binary(
    jsonl_path: Path,
    metadata_map: dict[str, dict] | None = None,
    metadata_list: list[dict] | None = None,
) -> dict:
    """
    Analyze multi-prediction binary file.

    Args:
        jsonl_path: Path to jsonl file
        metadata_map: Optional map from submission_id -> metadata for fallback lookup
        metadata_list: Optional list of metadata by index for fallback lookup

    Returns dict with all metrics for this file.
    """
    with open(jsonl_path) as f:
        data = [json.loads(line) for line in f]

    n_samples = len(data)
    if n_samples == 0:
        return {
            "file": jsonl_path.name,
            "n_samples": 0,
            "n_preds": None,
            "vote_agr": None,
            "first": "N/A",
            "mv_combined": "N/A",
        }

    # Determine number of predictions per sample (from first sample)
    first_preds = data[0].get("predict", [])
    n_preds_per_sample = len(first_preds) if isinstance(first_preds, list) else 1

    # Build joined data with labels, predictions, and metadata
    joined_data = []
    vote_agreements = []

    for idx, sample in enumerate(data):
        label_raw = sample.get("label", "")
        label_norm = normalize_binary(extract_boxed_answer(label_raw))

        preds = sample.get("predict", [])
        if not isinstance(preds, list):
            preds = [preds] if preds else []

        # First prediction
        first_pred = normalize_binary(extract_boxed_answer(preds[0])) if preds else None

        # Majority vote
        mv_pred = get_majority_vote(preds)

        # Vote agreement
        va = compute_vote_agreement(preds)
        if va is not None:
            vote_agreements.append(va)

        # Get year from metadata (with fallback to metadata_map or metadata_list)
        metadata = sample.get("_metadata", {})
        submission_id = metadata.get("submission_id")
        year = metadata.get("year")

        # Fallback: try to get metadata from map using submission_id
        if year is None and metadata_map and submission_id:
            if submission_id in metadata_map:
                year = metadata_map[submission_id].get("year")

        # Fallback: try to get metadata from list using index
        if year is None and metadata_list and idx < len(metadata_list):
            if metadata_list[idx] is not None:
                year = metadata_list[idx].get("year")

        joined_data.append({
            "label": label_norm,
            "first_pred": first_pred,
            "mv_pred": mv_pred,
            "year": year,
        })

    vote_agreement = sum(vote_agreements) / len(vote_agreements) if vote_agreements else None

    # Filter to valid data (both label and mv_pred exist)
    valid_data = [d for d in joined_data if d["label"] is not None and d["mv_pred"] is not None]
    valid_first = [d for d in joined_data if d["label"] is not None and d["first_pred"] is not None]

    # First prediction metrics (compact format)
    if valid_first:
        y_true = [d["label"] for d in valid_first]
        y_pred = [d["first_pred"] for d in valid_first]
        m = compute_binary_metrics(y_true, y_pred)
        first_str = f"{m['accuracy']:.2f}/{m['accept_recall']:.2f}/{m['reject_recall']:.2f}(n={m['n']})"
    else:
        first_str = "N/A"

    # Majority vote combined metrics
    mv_combined_str = format_metrics(valid_data)

    # Year breakdown for majority vote
    year_metrics = {}
    for year in range(2020, 2026):
        year_data = [d for d in valid_data if d["year"] == year]
        col_name = f"y{year}"
        year_metrics[col_name] = format_metrics(year_data)

    result = {
        "file": jsonl_path.name,
        "n_samples": n_samples,
        "n_preds": n_preds_per_sample,
        "vote_agr": f"{vote_agreement:.2f}" if vote_agreement else None,
        "first": first_str,
        "mv_combined": mv_combined_str,
    }
    result.update(year_metrics)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-prediction results (predict is a list of predictions per sample)"
    )
    parser.add_argument(
        "--dir", type=str, required=True,
        help="Directory containing jsonl files"
    )
    parser.add_argument(
        "--file-pattern", type=str, default=None,
        help="Regex pattern to filter files (e.g., '_ailab' matches *_ailab.jsonl)"
    )
    parser.add_argument(
        "--force-result-type", type=str, required=True, choices=["binary"],
        help="Force all files as this type (currently only 'binary' supported)"
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Save results as CSV"
    )
    args = parser.parse_args()

    # Find files matching pattern
    results_dir = Path(args.dir)
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return

    pattern = re.compile(args.file_pattern) if args.file_pattern else None
    jsonl_files = [
        f for f in results_dir.glob("*.jsonl")
        if pattern is None or pattern.search(f.name)
    ]

    if not jsonl_files:
        print(f"No jsonl files found in {results_dir}" +
              (f" matching pattern '{args.file_pattern}'" if args.file_pattern else ""))
        return

    print(f"Found {len(jsonl_files)} file(s) to analyze")

    # Build metadata map from all files (for files missing metadata)
    print("  Building metadata map...")
    metadata_map, metadata_list = build_metadata_map(jsonl_files)
    print(f"  Found {len(metadata_map)} unique submission_ids, {len(metadata_list)} indexed entries")

    # Analyze each file
    results = []
    for path in sorted(jsonl_files):
        print(f"  Analyzing: {path.name}")
        result = analyze_mult_binary(path, metadata_map, metadata_list)
        results.append(result)

    # Create DataFrame and display
    df = pd.DataFrame(results)

    # Reorder columns for better display
    year_cols = [f"y{y}" for y in range(2020, 2026)]
    column_order = ["file", "n_samples", "n_preds", "vote_agr", "first", "mv_combined"] + year_cols
    df = df[[c for c in column_order if c in df.columns]]

    print("\n" + "=" * 160)
    print(" MULTI-PREDICTION BINARY RESULTS (format: acc/accept_recall/reject_recall(n))")
    print("=" * 160)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.to_string(index=False))
    print()

    if args.csv:
        csv_path = results_dir / "mult_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
