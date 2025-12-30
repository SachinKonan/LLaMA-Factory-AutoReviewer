#!/usr/bin/env python3
"""
Analyze prediction results from grid search experiments.

Usage:
    python scripts/analyze.py --runs "iclr_text_binary_20480_base,iclr_text_binary_20480_finetuned"
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def extract_binary_label(text: str) -> str | None:
    """Extract accept/reject from label text (handles both simple and structured formats).

    Formats supported:
    - Simple: "Accept", "Reject"
    - Structured: "...Final Decision: Accept" or "...Final Decision: Reject"
    """
    text = text.strip()

    # First try: "Final Decision: X" format (norm_reviews/reviews datasets)
    match = re.search(r"Final Decision:\s*(Accept|Reject)", text, re.IGNORECASE)
    if match:
        ans = match.group(1).lower()
        return "accepted" if ans == "accept" else "rejected"

    # Second try: simple format (base datasets)
    text_lower = text.lower()
    if text_lower in ["accept", "accepted"]:
        return "accepted"
    if text_lower in ["reject", "rejected"]:
        return "rejected"

    return None


def extract_binary_answer(text: str) -> str | None:
    """Extract accept/reject from prediction text (lowercase)."""
    # Matches: Accept, Accepted, Acceptance, Reject, Rejected, Rejection
    match = re.search(r"(Accept|Reject)(ed|ance|ion)?", text, re.IGNORECASE)
    if match:
        ans = match.group(1).lower()
        return "accepted" if ans == "accept" else "rejected"
    return None


def extract_multiclass_label(text: str) -> str | None:
    """Extract multiclass label (handles both simple and structured formats).

    Returns: rejected, poster, spotlight, oral (lowercase)
    """
    text = text.strip()

    # First try: "Final Decision: X" format (norm_reviews/reviews datasets)
    match = re.search(r"Final Decision:\s*(Reject|Poster|Spotlight|Oral)", text, re.IGNORECASE)
    if match:
        ans = match.group(1).lower()
        return "rejected" if ans == "reject" else ans

    # Second try: simple format (base datasets)
    text_lower = text.lower()
    if text_lower in ["reject", "rejected"]:
        return "rejected"
    if text_lower in ["poster", "spotlight", "oral"]:
        return text_lower

    return None


def extract_multiclass_answer(text: str) -> str | None:
    """Extract multiclass answer (lowercase): reject, poster, spotlight, oral."""
    # Matches: Accept/Accepted/Acceptance, Reject/Rejected/Rejection, Poster, Oral, Spotlight
    match = re.search(r"(Accept|Reject|Poster|Oral|Spotlight)(ed|ance|ion)?", text, re.IGNORECASE)
    if match:
        ans = match.group(1).lower()
        # Map accept -> poster (default accepted category)
        if ans == "accept":
            return "poster"
        elif ans == "reject":
            return "rejected"
        else:
            return ans  # poster, oral, spotlight
    return None


def multiclass_to_binary(label: str) -> str:
    """Map multiclass label to binary: poster/oral/spotlight -> accepted, rejected -> rejected."""
    if label in ["poster", "oral", "spotlight"]:
        return "accepted"
    return "rejected"


def extract_citation_answer(text: str) -> float | None:
    """Extract numeric citation value (format: 0.XX percentile)."""
    text = text.strip()

    # First try: if text starts with a number, use that (finetuned model)
    first_match = re.match(r"^(\d+\.?\d*)", text)
    if first_match:
        try:
            val = float(first_match.group())
            if 0 <= val <= 1:
                return val
        except ValueError:
            pass

    # Second try: find all percentile-like numbers (0.XX) and take the last one
    # This handles base model outputs like "percentile of 0.75"
    matches = re.findall(r"\b(0\.\d+)\b", text)
    if matches:
        try:
            return float(matches[-1])  # Take last match
        except ValueError:
            pass

    # Third try: find any decimal number between 0 and 1
    matches = re.findall(r"(\d+\.\d+)", text)
    for m in reversed(matches):  # Check from end
        try:
            val = float(m)
            if 0 <= val <= 1:
                return val
        except ValueError:
            pass

    return None


def load_jsonl(path: str) -> pd.DataFrame:
    """Load jsonl file as dataframe."""
    with open(path) as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def get_dataset_size(dataset_name: str, data_dir: str = "data") -> int | None:
    """Get dataset size from dataset_info.json."""
    info_path = Path(data_dir) / "dataset_info.json"
    if not info_path.exists():
        return None

    with open(info_path) as f:
        info = json.load(f)

    if dataset_name not in info:
        return None

    data_path = Path(data_dir) / info[dataset_name]["file_name"]
    if not data_path.exists():
        return None

    with open(data_path) as f:
        return len(json.load(f))


def get_dataset_base(run_name: str) -> str:
    """Extract dataset base name from run name (remove _base or _finetuned suffix)."""
    if run_name.endswith("_base"):
        return run_name[:-5]
    elif run_name.endswith("_finetuned"):
        return run_name[:-10]
    return run_name


def analyze_binary_results(
    run_names: list[str], results_dir: str = "results/grid_search", data_dir: str = "data"
) -> pd.DataFrame:
    """Analyze binary classification results."""
    results = []

    for run_name in run_names:
        path = Path(results_dir) / f"{run_name}.jsonl"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue

        df = load_jsonl(str(path))
        # Extract labels (handles both simple "Accept"/"Reject" and "Final Decision: X" formats)
        df["label_clean"] = df["label"].apply(extract_binary_label)
        df["pred_extracted"] = df["predict"].apply(extract_binary_answer)

        # Filter to rows where both label and prediction are extractable
        df_valid = df[df["pred_extracted"].notna() & df["label_clean"].notna()].copy()

        test_size = len(df)
        num_labels_extracted = df["label_clean"].notna().sum()
        test_num_extracted = len(df_valid)

        # Get train size
        dataset_base = get_dataset_base(run_name)
        train_size = get_dataset_size(f"{dataset_base}_train", data_dir)

        if test_num_extracted > 0:
            y_true = df_valid["label_clean"].tolist()
            y_pred = df_valid["pred_extracted"].tolist()

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, pos_label="accepted", zero_division=0)
            recall = recall_score(y_true, y_pred, pos_label="accepted", zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label="accepted", zero_division=0)

            # Confusion matrix components (positive = accepted)
            num_tps = sum(1 for t, p in zip(y_true, y_pred) if t == "accepted" and p == "accepted")
            num_fps = sum(1 for t, p in zip(y_true, y_pred) if t == "rejected" and p == "accepted")
            num_fns = sum(1 for t, p in zip(y_true, y_pred) if t == "accepted" and p == "rejected")
            num_tns = sum(1 for t, p in zip(y_true, y_pred) if t == "rejected" and p == "rejected")
        else:
            accuracy = precision = recall = f1 = None
            num_tps = num_fps = num_fns = num_tns = None

        results.append(
            {
                "run_name": run_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "num_tps": num_tps,
                "num_fps": num_fps,
                "num_fns": num_fns,
                "num_tns": num_tns,
                "train_size": train_size,
                "test_size": test_size,
                "test_num_extracted": test_num_extracted,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


def analyze_multiclass_results(
    run_names: list[str], results_dir: str = "results/grid_search", data_dir: str = "data"
) -> pd.DataFrame:
    """Analyze multiclass results. Accuracy uses 4 classes, precision/recall/f1 use binary mapping."""
    results = []

    for run_name in run_names:
        path = Path(results_dir) / f"{run_name}.jsonl"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue

        df = load_jsonl(str(path))
        # Extract labels (handles both simple and "Final Decision: X" formats)
        df["label_clean"] = df["label"].apply(extract_multiclass_label)
        df["pred_extracted"] = df["predict"].apply(extract_multiclass_answer)

        # Filter to rows where both label and prediction are extractable
        df_valid = df[df["pred_extracted"].notna() & df["label_clean"].notna()].copy()

        test_size = len(df)
        num_labels_extracted = df["label_clean"].notna().sum()
        test_num_extracted = len(df_valid)

        # Get train size
        dataset_base = get_dataset_base(run_name)
        train_size = get_dataset_size(f"{dataset_base}_train", data_dir)

        if test_num_extracted > 0:
            # Multiclass accuracy (4 classes: rejected, poster, spotlight, oral)
            y_true_multi = df_valid["label_clean"].tolist()
            y_pred_multi = df_valid["pred_extracted"].tolist()
            accuracy_multi = accuracy_score(y_true_multi, y_pred_multi)

            # Binary metrics: map to accepted/rejected
            y_true_binary = [multiclass_to_binary(y) for y in y_true_multi]
            y_pred_binary = [multiclass_to_binary(y) for y in y_pred_multi]

            accuracy_binary = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary, pos_label="accepted", zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, pos_label="accepted", zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, pos_label="accepted", zero_division=0)

            # Confusion matrix components (positive = accepted)
            num_tps = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == "accepted" and p == "accepted")
            num_fps = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == "rejected" and p == "accepted")
            num_fns = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == "accepted" and p == "rejected")
            num_tns = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == "rejected" and p == "rejected")
        else:
            accuracy_multi = accuracy_binary = precision = recall = f1 = None
            num_tps = num_fps = num_fns = num_tns = None

        results.append(
            {
                "run_name": run_name,
                "accuracy_4class": accuracy_multi,
                "accuracy_binary": accuracy_binary,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "num_tps": num_tps,
                "num_fps": num_fps,
                "num_fns": num_fns,
                "num_tns": num_tns,
                "train_size": train_size,
                "test_size": test_size,
                "test_num_extracted": test_num_extracted,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


def extract_citation_label(text: str) -> float | None:
    """Extract citation score from label (handles 'Citation Score: X.XX' format)."""
    text = text.strip()

    # First try: direct float conversion
    try:
        return float(text)
    except ValueError:
        pass

    # Second try: find "Citation Score: X.XX" pattern
    match = re.search(r"Citation Score:\s*(\d+\.?\d*)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Third try: find last 0.XX number
    matches = re.findall(r"\b(0\.\d+)\b", text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass

    return None


def analyze_citation_results(
    run_names: list[str], results_dir: str = "results/grid_search", data_dir: str = "data"
) -> pd.DataFrame:
    """Analyze citation prediction results (regression metrics)."""
    results = []

    for run_name in run_names:
        path = Path(results_dir) / f"{run_name}.jsonl"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue

        df = load_jsonl(str(path))
        df["label_clean"] = df["label"].apply(extract_citation_label)
        df["pred_extracted"] = df["predict"].apply(extract_citation_answer)

        # Filter to extractable predictions AND labels
        df_valid = df[df["pred_extracted"].notna() & df["label_clean"].notna()].copy()

        test_size = len(df)
        test_num_extracted = len(df_valid)

        # Get train size
        dataset_base = get_dataset_base(run_name)
        train_size = get_dataset_size(f"{dataset_base}_train", data_dir)

        if test_num_extracted > 0:
            y_true = df_valid["label_clean"]
            y_pred = df_valid["pred_extracted"]

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
        else:
            mae = mse = None

        results.append(
            {
                "run_name": run_name,
                "mae": mae,
                "mse": mse,
                "train_size": train_size,
                "test_size": test_size,
                "test_num_extracted": test_num_extracted,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


def print_results(df: pd.DataFrame, title: str) -> None:
    """Print results dataframe as formatted table, sorted by base/finetuned."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print("=" * 80)
    if df.empty:
        print("  No results found")
    else:
        # Add sort key: base comes before finetuned
        df = df.copy()
        df["_sort_key"] = df["run_name"].apply(
            lambda x: (0 if "_base" in x else 1, x)
        )
        df = df.sort_values("_sort_key").drop(columns=["_sort_key"])
        print(df.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze prediction results from grid search")
    parser.add_argument(
        "--runs", type=str, required=True, help="Comma-separated list of run names"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/grid_search", help="Results directory"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    run_names = [r.strip() for r in args.runs.split(",")]

    # Group by task type
    binary_runs = [r for r in run_names if "binary" in r]
    multiclass_runs = [r for r in run_names if "multiclass" in r]
    citation_runs = [r for r in run_names if "citation" in r]

    all_results = {}

    if binary_runs:
        binary_df = analyze_binary_results(binary_runs, args.results_dir, args.data_dir)
        print_results(binary_df, "BINARY CLASSIFICATION RESULTS")
        all_results["binary"] = binary_df

    if multiclass_runs:
        multiclass_df = analyze_multiclass_results(multiclass_runs, args.results_dir, args.data_dir)
        print_results(multiclass_df, "MULTICLASS RESULTS (mapped to Accept/Reject)")
        all_results["multiclass"] = multiclass_df

    if citation_runs:
        citation_df = analyze_citation_results(citation_runs, args.results_dir, args.data_dir)
        print_results(citation_df, "CITATION PREDICTION RESULTS")
        all_results["citation"] = citation_df

    return all_results


if __name__ == "__main__":
    main()
