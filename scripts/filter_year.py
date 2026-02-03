#!/usr/bin/env python3
"""Filter dataset to exclude specific year(s) from the data."""

import argparse
import json
import os
from pathlib import Path


def filter_data_by_year(data: list, exclude_years: set) -> list:
    """Filter out entries where _metadata.year is in exclude_years."""
    return [entry for entry in data if entry.get("_metadata", {}).get("year") not in exclude_years]


def process_dataset(source_dir: Path, target_dir: Path, exclude_years: set) -> dict:
    """Process a single dataset directory.

    Returns dict with statistics about the filtering.
    """
    source_file = source_dir / "data.json"

    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    # Load data
    with open(source_file, "r") as f:
        data = json.load(f)

    original_count = len(data)

    # Get year distribution before filtering
    year_counts_before = {}
    for entry in data:
        year = entry.get("_metadata", {}).get("year", "unknown")
        year_counts_before[year] = year_counts_before.get(year, 0) + 1

    # Filter data
    filtered_data = filter_data_by_year(data, exclude_years)
    filtered_count = len(filtered_data)

    # Get year distribution after filtering
    year_counts_after = {}
    for entry in filtered_data:
        year = entry.get("_metadata", {}).get("year", "unknown")
        year_counts_after[year] = year_counts_after.get(year, 0) + 1

    # Create target directory and write data
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "data.json"

    with open(target_file, "w") as f:
        json.dump(filtered_data, f, indent=2)

    return {
        "source": str(source_dir),
        "target": str(target_dir),
        "original_count": original_count,
        "filtered_count": filtered_count,
        "removed_count": original_count - filtered_count,
        "years_before": year_counts_before,
        "years_after": year_counts_after,
    }


def main():
    parser = argparse.ArgumentParser(description="Filter dataset to exclude specific year(s)")
    parser.add_argument("--source", required=True, help="Source dataset base name (without _train/_test/_validation)")
    parser.add_argument("--target", required=True, help="Target dataset base name (without _train/_test/_validation)")
    parser.add_argument("--exclude-years", required=True, nargs="+", type=int, help="Year(s) to exclude")
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"], help="Splits to process")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    exclude_years = set(args.exclude_years)

    print(f"Filtering datasets to exclude year(s): {sorted(exclude_years)}")
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print()

    for split in args.splits:
        source_dir = data_dir / f"{args.source}_{split}"
        target_dir = data_dir / f"{args.target}_{split}"

        print(f"Processing {split}...")

        try:
            stats = process_dataset(source_dir, target_dir, exclude_years)
            print(f"  Original: {stats['original_count']} entries")
            print(f"  Filtered: {stats['filtered_count']} entries")
            print(f"  Removed:  {stats['removed_count']} entries")
            print(f"  Years before: {dict(sorted(stats['years_before'].items()))}")
            print(f"  Years after:  {dict(sorted(stats['years_after'].items()))}")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    print("Done!")


if __name__ == "__main__":
    main()
