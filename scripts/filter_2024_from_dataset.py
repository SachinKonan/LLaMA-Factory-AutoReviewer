#!/usr/bin/env python3
"""
Filter out 2024 samples from datasets to create no-2024 versions.

This script reads the source datasets (iclr_2020_2025_*) and creates
filtered versions (iclr_2020_2023_2025_*) that exclude all 2024 samples.
"""

import json
import os
from pathlib import Path


def filter_2024(src_dir: str, dst_dir: str) -> dict:
    """
    Filter out 2024 samples from a dataset.

    Args:
        src_dir: Source directory containing data.json
        dst_dir: Destination directory for filtered data.json

    Returns:
        dict with statistics about the filtering
    """
    os.makedirs(dst_dir, exist_ok=True)

    src_path = os.path.join(src_dir, "data.json")
    dst_path = os.path.join(dst_dir, "data.json")

    with open(src_path, 'r') as f:
        data = json.load(f)

    # Filter out 2024 samples
    filtered = [s for s in data if s['_metadata']['year'] != 2024]

    # Count by year for statistics
    year_counts_before = {}
    for s in data:
        year = s['_metadata']['year']
        year_counts_before[year] = year_counts_before.get(year, 0) + 1

    year_counts_after = {}
    for s in filtered:
        year = s['_metadata']['year']
        year_counts_after[year] = year_counts_after.get(year, 0) + 1

    with open(dst_path, 'w') as f:
        json.dump(filtered, f)

    removed = len(data) - len(filtered)
    print(f"{os.path.basename(src_dir)}: {len(data)} -> {len(filtered)} (removed {removed} samples from 2024)")
    print(f"  Year distribution before: {dict(sorted(year_counts_before.items()))}")
    print(f"  Year distribution after:  {dict(sorted(year_counts_after.items()))}")

    return {
        'src': src_dir,
        'dst': dst_dir,
        'before': len(data),
        'after': len(filtered),
        'removed': removed,
        'years_before': year_counts_before,
        'years_after': year_counts_after
    }


def main():
    base_dir = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data")

    # Define source -> destination mappings
    # We need to create clean text versions without 2024
    datasets_to_filter = [
        # trainagreeing_clean (text)
        ("iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7_train",
         "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7_train"),
        ("iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7_validation",
         "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7_validation"),
        ("iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7_test",
         "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7_test"),
        # balanced_clean (text)
        ("iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_train",
         "iclr_2020_2023_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_train"),
        ("iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_validation",
         "iclr_2020_2023_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_validation"),
        ("iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test",
         "iclr_2020_2023_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test"),
    ]

    print("=" * 60)
    print("Filtering 2024 samples from datasets")
    print("=" * 60)

    results = []
    for src_name, dst_name in datasets_to_filter:
        src_dir = base_dir / src_name
        dst_dir = base_dir / dst_name

        if not src_dir.exists():
            print(f"WARNING: Source directory does not exist: {src_dir}")
            continue

        if dst_dir.exists():
            print(f"SKIPPING: Destination already exists: {dst_dir}")
            continue

        result = filter_2024(str(src_dir), str(dst_dir))
        results.append(result)
        print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  {os.path.basename(r['dst'])}: {r['before']} -> {r['after']}")

    print("\nDone! Don't forget to add entries to dataset_info.json")


if __name__ == "__main__":
    main()
