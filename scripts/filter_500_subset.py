#!/usr/bin/env python3
"""
Filter out 2024 samples from datasets to create no-2024 versions.

This script reads the source datasets (iclr_2020_2025_*) and creates
filtered versions (iclr_2020_2023_2025_*) that exclude all 2024 samples.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import random


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


def filter_500(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    src_path = os.path.join(src_dir, "data.json")
    dst_path = os.path.join(dst_dir, "data.json")
    
    input_data = pd.read_json(src_path)
    result_data = pd.DataFrame(columns=input_data.columns)
    years = [2020, 2021, 2022, 2023, 2025] # TODO: 2026?
    year_count = defaultdict(int)
    # print(input_data.columns)
    # print(f"Metadata format: {input_data['_metadata'][0].keys()}")
    year_count = Counter([m['year'] for m in input_data['_metadata']])
    # 1. Get initial counts and denominator
    total_len = sum(year_count.values()) 

    # 2. Target 250 "pairs" instead of 500 single items
    target_pairs = 250 
    # Calculate proportions based on the 250 pairs
    proportions = {k: (v * target_pairs / total_len) for k, v in year_count.items()}
    # Floor the pairs
    pair_count = {k: int(p) for k, p in proportions.items()}
    # Add the shortfall pairs to the largest remainders
    shortfall = target_pairs - sum(pair_count.values())
    for k in sorted(proportions, key=lambda x: proportions[x] % 1, reverse=True)[:shortfall]:
        pair_count[k] += 1

    # 3. Multiply the final pairs by 2 to get your even counts
    year_count = {k: v * 2 for k, v in pair_count.items()}
    print(f"Year data: {year_count}")

    # Collect accepts & rejects for each year
    indices = []
    indices_by_year_label = defaultdict(lambda: defaultdict(list))
    for i, (meta, decision) in enumerate(zip(input_data["_metadata"], [m["decision"] for m in input_data["_metadata"]])):
        # Map decision to 0 (reject) or 1 (accept)
        label = 0 if decision == 'reject' else 1
        indices_by_year_label[meta["year"]][label].append(i)

    for year, count in year_count.items():
        half = count // 2
        for label in [0, 1]:
            pool = indices_by_year_label[year][label]
            if not pool:
                print(f"WARNING: No samples for year {year}, label {label}")
                continue
            indices.extend(random.sample(pool, min(len(pool), half)))

    result_data = input_data.loc[indices]
    result_data.to_json(dst_path, orient='records', indent=4)

    # TODO: what to return? probably have to write this.
    return result_data



def main():
    base_dir = Path("/scratch/gpfs/ZHUANGL/jl0796/shared/data")

    # Define source -> destination mappings
    # We need to create clean text versions without 2024
    datasets_to_filter = [
        # text
        ("iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered_test",
         "iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered_test_500"),
        # vision
        ("iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered_test",
         "iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered_test_500"),

        ("iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test",
        "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test_500"),                                                                                                                        
                                                                                                                                                                                  
        ("iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test",
        "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test_500")
    ]

    print("=" * 60)
    print("Creating 500-sample subsets of datasets")
    print("=" * 60)

    results = []
    for src_name, dst_name in datasets_to_filter:
        src_dir = base_dir / src_name
        dst_dir = base_dir / dst_name

        if not src_dir.exists():
            print(f"WARNING: Source directory does not exist: {src_dir}")
            continue

        # if dst_dir.exists():
        #     print(f"SKIPPING: Destination already exists: {dst_dir}")
        #     continue

        result = filter_500(str(src_dir), str(dst_dir))
        results.append(result)
        print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nDone! Don't forget to add entries to dataset_info.json")


if __name__ == "__main__":
    main()
