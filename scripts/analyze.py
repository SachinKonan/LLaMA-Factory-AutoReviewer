#!/usr/bin/env python3
"""
Analyze prediction results from grid search experiments.

Usage:
    python scripts/analyze.py --dir grid_searchv2
    python scripts/analyze.py --dir grid_searchv2 --indicators binary,citation
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def extract_gemini_answer(text: str) -> str | None:
    """Extract answer from Gemini output (simple string matching)."""
    if not text:
        return None
    text_lower = text.lower()
    is_reject = "reject" in text_lower
    is_accept = "accept" in text_lower
    if is_accept and not is_reject:
        return "Accept"
    if is_reject and not is_accept:
        return "Reject"
    # If both or neither, return None
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


def discover_results(
    results_dir: Path, force_type: str | None = None
) -> dict[str, list[tuple[str, str, Path]]]:
    """
    Discover all result files in the directory.

    Returns dict mapping task_type -> list of (dataset_name, run_name, path).
    run_name is like "base" or "finetuned2871".

    Args:
        results_dir: Directory to scan for results
        force_type: If specified, treat all results as this type (binary/multiclass/citation)
    """
    results = {"binary": [], "multiclass": [], "citation": []}

    if not results_dir.exists():
        print(f"Warning: {results_dir} does not exist")
        return results

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue

        dataset_name = subdir.name

        # Determine task type from directory name (or use force_type)
        if force_type:
            task_type = force_type
        elif "binary" in dataset_name:
            task_type = "binary"
        elif "multiclass" in dataset_name:
            task_type = "multiclass"
        elif "citation" in dataset_name:
            task_type = "citation"
        else:
            print(f"Warning: Could not determine task type for {dataset_name}, skipping")
            continue

        # Find all jsonl files in this subdir
        for jsonl_file in sorted(subdir.glob("*.jsonl")):
            run_name = jsonl_file.stem  # "base" or "finetuned2871"
            results[task_type].append((dataset_name, run_name, jsonl_file))

    return results


def multiclass_to_binary(label: str) -> str:
    """Map multiclass label to binary: poster/oral/spotlight -> accepted, reject -> rejected."""
    label_lower = label.lower()
    if label_lower in ["poster", "oral", "spotlight"]:
        return "accepted"
    if label_lower in ["reject", "rejected"]:
        return "rejected"
    return label_lower


def compute_binary_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute binary classification metrics.

    Returns dict with: accuracy, accept_precision, reject_precision, accept_recall,
    reject_recall, accept_f1, reject_f1, num_tps, num_fps, num_fns, num_tns
    """
    # Confusion matrix components (positive = accepted)
    num_tps = sum(1 for t, p in zip(y_true, y_pred) if t == "accepted" and p == "accepted")
    num_fps = sum(1 for t, p in zip(y_true, y_pred) if t == "rejected" and p == "accepted")
    num_fns = sum(1 for t, p in zip(y_true, y_pred) if t == "accepted" and p == "rejected")
    num_tns = sum(1 for t, p in zip(y_true, y_pred) if t == "rejected" and p == "rejected")

    accuracy = accuracy_score(y_true, y_pred)

    # Accept metrics (positive class = accepted)
    accept_precision = num_tps / (num_tps + num_fps) if (num_tps + num_fps) > 0 else 0.0
    accept_recall = num_tps / (num_tps + num_fns) if (num_tps + num_fns) > 0 else 0.0
    accept_f1 = f1_score(y_true, y_pred, pos_label="accepted", zero_division=0)

    # Reject metrics (positive class = rejected, so TN/FN swap roles)
    reject_precision = num_tns / (num_tns + num_fns) if (num_tns + num_fns) > 0 else 0.0
    reject_recall = num_tns / (num_tns + num_fps) if (num_tns + num_fps) > 0 else 0.0
    reject_f1 = f1_score(y_true, y_pred, pos_label="rejected", zero_division=0)

    return {
        "accuracy": accuracy,
        "accept_precision": accept_precision,
        "reject_precision": reject_precision,
        "accept_recall": accept_recall,
        "reject_recall": reject_recall,
        "accept_f1": accept_f1,
        "reject_f1": reject_f1,
        "num_tps": num_tps,
        "num_fps": num_fps,
        "num_fns": num_fns,
        "num_tns": num_tns,
    }


def analyze_binary_results(
    result_files: list[tuple[str, str, Path]], data_dir: str = "data", is_gemini: bool = False
) -> pd.DataFrame:
    """Analyze binary classification results."""
    results = []

    # Choose extraction function based on is_gemini flag
    pred_extractor = extract_gemini_answer if is_gemini else extract_boxed_answer

    for dataset_name, run_name, path in result_files:
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue

        df = load_jsonl(str(path))
        df["label_clean"] = df["label"].apply(extract_boxed_answer)  # Labels always use boxed format
        df["pred_extracted"] = df["predict"].apply(pred_extractor)

        # Normalize to accepted/rejected
        df["label_clean"] = df["label_clean"].apply(
            lambda x: "accepted" if x and x.lower() in ["accept", "accepted"]
            else ("rejected" if x and x.lower() in ["reject", "rejected"] else None)
        )
        df["pred_extracted"] = df["pred_extracted"].apply(
            lambda x: "accepted" if x and x.lower() in ["accept", "accepted"]
            else ("rejected" if x and x.lower() in ["reject", "rejected"] else None)
        )

        # Filter to rows where both label and prediction are extractable
        df_valid = df[df["pred_extracted"].notna() & df["label_clean"].notna()].copy()

        test_size = len(df)

        # Get train size (dataset_name doesn't have _test suffix, so append _train)
        train_dataset_name = f"{dataset_name}_train"
        train_size = get_dataset_size(train_dataset_name, data_dir)

        if len(df_valid) > 0:
            y_true = df_valid["label_clean"].tolist()
            y_pred = df_valid["pred_extracted"].tolist()
            metrics = compute_binary_metrics(y_true, y_pred)

            # Calculate ground-truth acceptance rate and majority reject accuracy
            num_accepts = sum(1 for y in y_true if y == "accepted")
            num_rejects = sum(1 for y in y_true if y == "rejected")
            total = num_accepts + num_rejects
            acceptance_rate = num_accepts / total if total > 0 else None
            majority_reject_acc = 1 - acceptance_rate if acceptance_rate is not None else None
        else:
            metrics = {
                "accuracy": None,
                "accept_precision": None,
                "reject_precision": None,
                "accept_recall": None,
                "reject_recall": None,
                "accept_f1": None,
                "reject_f1": None,
                "num_tps": None,
                "num_fps": None,
                "num_fns": None,
                "num_tns": None,
            }
            acceptance_rate = None
            majority_reject_acc = None

        results.append(
            {
                "dataset": dataset_name,
                "run": run_name,
                **metrics,
                "train_size": train_size,
                "test_size": test_size,
                "acceptance_rate": acceptance_rate,
                "majority_reject_acc": majority_reject_acc,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


def analyze_multiclass_results(
    result_files: list[tuple[str, str, Path]], data_dir: str = "data"
) -> pd.DataFrame:
    """Analyze multiclass results. Accuracy uses 4 classes, then binary metrics after mapping."""
    results = []

    for dataset_name, run_name, path in result_files:
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue

        df = load_jsonl(str(path))
        df["label_clean"] = df["label"].apply(extract_boxed_answer)
        df["pred_extracted"] = df["predict"].apply(extract_boxed_answer)

        # Normalize multiclass labels
        valid_classes = ["reject", "rejected", "poster", "spotlight", "oral"]
        df["label_clean"] = df["label_clean"].apply(
            lambda x: x.lower() if x and x.lower() in valid_classes else None
        )
        df["pred_extracted"] = df["pred_extracted"].apply(
            lambda x: x.lower() if x and x.lower() in valid_classes else None
        )

        # Normalize "rejected" -> "reject" for consistency
        df["label_clean"] = df["label_clean"].apply(
            lambda x: "reject" if x == "rejected" else x
        )
        df["pred_extracted"] = df["pred_extracted"].apply(
            lambda x: "reject" if x == "rejected" else x
        )

        # Filter to rows where both label and prediction are extractable
        df_valid = df[df["pred_extracted"].notna() & df["label_clean"].notna()].copy()

        test_size = len(df)

        # Get train size (dataset_name doesn't have _test suffix, so append _train)
        train_dataset_name = f"{dataset_name}_train"
        train_size = get_dataset_size(train_dataset_name, data_dir)

        if len(df_valid) > 0:
            # Multiclass accuracy (4 classes: reject, poster, spotlight, oral)
            y_true_multi = df_valid["label_clean"].tolist()
            y_pred_multi = df_valid["pred_extracted"].tolist()
            accuracy_4class = accuracy_score(y_true_multi, y_pred_multi)

            # Binary metrics: map poster/spotlight/oral -> accepted
            y_true_binary = [multiclass_to_binary(y) for y in y_true_multi]
            y_pred_binary = [multiclass_to_binary(y) for y in y_pred_multi]
            metrics = compute_binary_metrics(y_true_binary, y_pred_binary)

            # Calculate ground-truth acceptance rate and majority reject accuracy
            num_accepts = sum(1 for y in y_true_binary if y == "accepted")
            num_rejects = sum(1 for y in y_true_binary if y == "rejected")
            total = num_accepts + num_rejects
            acceptance_rate = num_accepts / total if total > 0 else None
            majority_reject_acc = 1 - acceptance_rate if acceptance_rate is not None else None
        else:
            accuracy_4class = None
            metrics = {
                "accuracy": None,
                "accept_precision": None,
                "reject_precision": None,
                "accept_recall": None,
                "reject_recall": None,
                "accept_f1": None,
                "reject_f1": None,
                "num_tps": None,
                "num_fps": None,
                "num_fns": None,
                "num_tns": None,
            }
            acceptance_rate = None
            majority_reject_acc = None

        results.append(
            {
                "dataset": dataset_name,
                "run": run_name,
                "accuracy_4class": accuracy_4class,
                **metrics,
                "train_size": train_size,
                "test_size": test_size,
                "acceptance_rate": acceptance_rate,
                "majority_reject_acc": majority_reject_acc,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


def analyze_citation_results(
    result_files: list[tuple[str, str, Path]], data_dir: str = "data"
) -> pd.DataFrame:
    """Analyze citation prediction results (regression metrics)."""
    results = []

    for dataset_name, run_name, path in result_files:
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue

        df = load_jsonl(str(path))
        df["label_clean"] = df["label"].apply(extract_boxed_answer)
        df["pred_extracted"] = df["predict"].apply(extract_boxed_answer)

        # Convert to float
        def to_float(x):
            if x is None:
                return None
            try:
                val = float(x)
                if 0 <= val <= 1:
                    return val
            except ValueError:
                pass
            return None

        df["label_clean"] = df["label_clean"].apply(to_float)
        df["pred_extracted"] = df["pred_extracted"].apply(to_float)

        # Filter to extractable predictions AND labels
        df_valid = df[df["pred_extracted"].notna() & df["label_clean"].notna()].copy()

        test_size = len(df)

        # Get train size (dataset_name doesn't have _test suffix, so append _train)
        train_dataset_name = f"{dataset_name}_train"
        train_size = get_dataset_size(train_dataset_name, data_dir)

        if len(df_valid) > 0:
            y_true = df_valid["label_clean"].values
            y_pred = df_valid["pred_extracted"].values

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)

            # Compute correlation and p-value
            if len(y_true) > 2:
                correlation, p_value = stats.pearsonr(y_true, y_pred)
            else:
                correlation, p_value = None, None
        else:
            mae = mse = correlation = p_value = None

        results.append(
            {
                "dataset": dataset_name,
                "run": run_name,
                "mae": mae,
                "mse": mse,
                "correlation": correlation,
                "p_value": p_value,
                "train_size": train_size,
                "test_size": test_size,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    """Sort results by dataset name, then base before finetuned."""
    if df.empty:
        return df

    df = df.copy()
    # Sort key: (dataset, 0 if base else 1, run_name)
    df["_sort_key"] = df.apply(
        lambda row: (row["dataset"], 0 if row["run"] == "base" else 1, row["run"]),
        axis=1
    )
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"])
    return df


def print_results(df: pd.DataFrame, title: str) -> None:
    """Print results dataframe as formatted table."""
    print(f"\n{'=' * 120}")
    print(f" {title}")
    print("=" * 120)
    if df.empty:
        print("  No results found")
    else:
        df = sort_results(df)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze prediction results from grid search")
    parser.add_argument(
        "--dir", type=str, required=True, help="Directory name under results/ to scan (e.g., grid_searchv2)"
    )
    parser.add_argument(
        "--indicators", type=str, default=None,
        help="Comma-separated list of indicators to show (binary,multiclass,citation). Default: all"
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--csv", action="store_true",
        help="Save results as CSV files in results/{dir}/"
    )
    parser.add_argument(
        "--force-result-type", type=str, default=None,
        choices=["binary", "multiclass", "citation"],
        help="Force all results to be treated as this type (binary, multiclass, or citation)"
    )
    parser.add_argument(
        "--is-gemini", action="store_true",
        help="Use Gemini-style parsing (simple accept/reject string matching instead of \\boxed{})"
    )
    args = parser.parse_args()

    results_dir = Path("results") / args.dir

    # Parse indicators filter
    if args.indicators:
        indicators = [i.strip().lower() for i in args.indicators.split(",")]
    else:
        indicators = ["binary", "multiclass", "citation"]

    # Discover all results
    force_type = getattr(args, 'force_result_type', None)
    discovered = discover_results(results_dir, force_type=force_type)

    all_results = {}

    if "binary" in indicators and discovered["binary"]:
        binary_df = analyze_binary_results(discovered["binary"], args.data_dir, is_gemini=args.is_gemini)
        binary_df = sort_results(binary_df)
        print_results(binary_df, "BINARY CLASSIFICATION RESULTS")
        all_results["binary"] = binary_df
        if args.csv and not binary_df.empty:
            csv_path = results_dir / "binary_results.csv"
            binary_df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

    if "multiclass" in indicators and discovered["multiclass"]:
        multiclass_df = analyze_multiclass_results(discovered["multiclass"], args.data_dir)
        multiclass_df = sort_results(multiclass_df)
        print_results(multiclass_df, "DECISION (MULTICLASS) RESULTS")
        all_results["multiclass"] = multiclass_df
        if args.csv and not multiclass_df.empty:
            csv_path = results_dir / "multiclass_results.csv"
            multiclass_df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

    if "citation" in indicators and discovered["citation"]:
        citation_df = analyze_citation_results(discovered["citation"], args.data_dir)
        citation_df = sort_results(citation_df)
        print_results(citation_df, "CITATION PREDICTION RESULTS")
        all_results["citation"] = citation_df
        if args.csv and not citation_df.empty:
            csv_path = results_dir / "citation_results.csv"
            citation_df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

    if not any(not df.empty for df in all_results.values()):
        print(f"No results found in {results_dir}")

    return all_results


if __name__ == "__main__":
    main()
