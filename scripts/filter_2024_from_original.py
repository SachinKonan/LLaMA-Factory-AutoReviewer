#!/usr/bin/env python3
"""
Filter 2024 samples from original datasets to create 2020_2023_2025 variants.

Usage:
    python scripts/filter_2024_from_original.py
"""

import json
from pathlib import Path


def filter_2024(input_file: Path, output_file: Path):
    """Remove all samples from 2024."""
    print(f"Processing {input_file}...")

    with open(input_file) as f:
        data = json.load(f)

    # Filter out 2024 samples based on _metadata
    filtered = [
        sample for sample in data
        if sample.get('_metadata', {}).get('year') != 2024
    ]

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write filtered data
    with open(output_file, 'w') as f:
        json.dump(filtered, f, indent=2)

    removed = len(data) - len(filtered)
    print(f"  {len(data):,} → {len(filtered):,} samples (removed {removed:,} from 2024)")
    return len(data), len(filtered), removed


def main():
    base_dir = Path("data")

    total_original = 0
    total_filtered = 0
    total_removed = 0

    print("=" * 70)
    print("Filtering 2024 from Original Trainagreeing Datasets")
    print("=" * 70)
    print()

    # Process all splits and modalities
    for split in ['train', 'validation', 'test']:
        for modality in ['text', 'vision']:
            input_name = f"iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_original_{modality}_binary_noreviews_v7_{split}"
            output_name = f"iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_original_{modality}_binary_noreviews_v7_{split}"

            input_path = base_dir / input_name / "data.json"
            output_path = base_dir / output_name / "data.json"

            if not input_path.exists():
                print(f"⚠️  SKIP: {input_path} does not exist")
                continue

            orig, filt, rem = filter_2024(input_path, output_path)
            total_original += orig
            total_filtered += filt
            total_removed += rem

    print()
    print("=" * 70)
    print("Summary:")
    print(f"  Total original:  {total_original:,} samples")
    print(f"  Total filtered:  {total_filtered:,} samples")
    print(f"  Total removed:   {total_removed:,} samples (2024)")
    print(f"  Removal rate:    {100*total_removed/total_original:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
