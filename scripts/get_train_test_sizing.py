#!/usr/bin/env python3
"""
Get train/test sizes and acceptance rates for datasets matching a regex pattern.

Usage:
    python scripts/get_train_test_sizing.py --regex "balanced_deepreview.*binary"
    python scripts/get_train_test_sizing.py --regex "vision.*binary"
"""

import argparse
import json
import re
from pathlib import Path


def load_data_json(data_dir: Path, file_name: str) -> list:
    """Load a data.json file and return the list of entries."""
    path = data_dir / file_name
    if not path.exists():
        return []

    with open(path) as f:
        return json.load(f)


def count_acceptance(entries: list) -> tuple[int, int]:
    """
    Count Accept and Reject in the dataset.
    Returns (accept_count, reject_count)
    """
    accept_count = 0
    reject_count = 0

    for entry in entries:
        conversations = entry.get("conversations", [])
        for msg in conversations:
            if msg.get("from") == "assistant" or msg.get("from") == "gpt":
                value = msg.get("value", "")
                if r"\boxed{Accept}" in value or "\\boxed{Accept}" in value:
                    accept_count += 1
                elif r"\boxed{Reject}" in value or "\\boxed{Reject}" in value:
                    reject_count += 1

    return accept_count, reject_count


def main():
    parser = argparse.ArgumentParser(description="Get train/test sizes for datasets matching a regex pattern.")
    parser.add_argument(
        "--regex",
        type=str,
        required=True,
        help="Regex pattern to filter dataset names (e.g., 'balanced_deepreview.*binary')"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output as CSV format"
    )
    args = parser.parse_args()

    data_dir = Path("data")
    dataset_info_path = data_dir / "dataset_info.json"

    if not dataset_info_path.exists():
        print(f"Error: {dataset_info_path} not found")
        return

    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    # Compile regex pattern
    pattern = re.compile(args.regex)

    # Find matching datasets (without _train/_test suffix)
    base_datasets = set()
    for name in dataset_info.keys():
        if pattern.search(name):
            # Remove _train or _test suffix to get base name
            if name.endswith("_train"):
                base_name = name[:-6]
            elif name.endswith("_test"):
                base_name = name[:-5]
            else:
                continue  # Skip if not train/test
            base_datasets.add(base_name)

    if not base_datasets:
        print(f"No datasets found matching pattern: {args.regex}")
        return

    # Collect data for each base dataset
    results = []
    for base_name in sorted(base_datasets):
        train_name = f"{base_name}_train"
        test_name = f"{base_name}_test"

        train_info = dataset_info.get(train_name, {})
        test_info = dataset_info.get(test_name, {})

        train_file = train_info.get("file_name", "")
        test_file = test_info.get("file_name", "")

        # Load data
        train_data = load_data_json(data_dir, train_file) if train_file else []
        test_data = load_data_json(data_dir, test_file) if test_file else []

        train_size = len(train_data)
        test_size = len(test_data)

        # Count acceptance rate separately for train and test
        train_accept, train_reject = count_acceptance(train_data)
        test_accept, test_reject = count_acceptance(test_data)

        train_total = train_accept + train_reject
        test_total = test_accept + test_reject

        train_acceptance_pct = 100.0 * train_accept / train_total if train_total > 0 else None
        test_acceptance_pct = 100.0 * test_accept / test_total if test_total > 0 else None

        results.append({
            "dataset": base_name,
            "train_size": train_size,
            "test_size": test_size,
            "train_accept_pct": train_acceptance_pct,
            "test_accept_pct": test_acceptance_pct,
        })

    # Print results
    if args.csv:
        # CSV format
        print("dataset,train_size,test_size,train_accept_pct,test_accept_pct")
        for r in results:
            train_acc = f"{r['train_accept_pct']:.1f}" if r['train_accept_pct'] is not None else ""
            test_acc = f"{r['test_accept_pct']:.1f}" if r['test_accept_pct'] is not None else ""
            print(f"{r['dataset']},{r['train_size']},{r['test_size']},{train_acc},{test_acc}")
    else:
        # Table format
        print(f"\nDatasets matching: {args.regex}\n")
        print(f"{'Dataset':<70} {'Train':>8} {'Test':>8} {'Train Acc%':>12} {'Test Acc%':>12}")
        print("-" * 115)

        for r in results:
            train_acc = f"{r['train_accept_pct']:.1f}%" if r['train_accept_pct'] is not None else "N/A"
            test_acc = f"{r['test_accept_pct']:.1f}%" if r['test_accept_pct'] is not None else "N/A"
            print(f"{r['dataset']:<70} {r['train_size']:>8} {r['test_size']:>8} {train_acc:>12} {test_acc:>12}")

        # Print totals
        total_train = sum(r['train_size'] for r in results)
        total_test = sum(r['test_size'] for r in results)
        print("-" * 115)
        print(f"{'TOTAL':<70} {total_train:>8} {total_test:>8}")


if __name__ == "__main__":
    main()
