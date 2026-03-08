#!/usr/bin/env python3
"""
Stage 1: Generate training datasets with different accept/reject proportions.

Creates 4 datasets from the original v7 training data:
- 1:2 (50:50 - original distribution)
- 1:4 (25:75)
- 1:3 (33:67)
- 1:8 (12.5:87.5)

Usage:
    python stage1_generate_datasets.py
    python stage1_generate_datasets.py --debug  # Test on 100 samples
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Paths
ORIGINAL_TRAIN = "/scratch/gpfs/ZHUANGL/jl0796/shared/data/iclr_2020_2025_85_5_10_balanced_original_text_v7_filtered_train/data.json"
OUTPUT_DIR = Path("/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/2_11_26_training/weighted_loss_fn/data")

# Acceptance rates to test: accept_fraction = accept / (accept + reject)
# e.g. 1/2 means 50% accept, 1/3 means 33% accept, etc.
PROPORTIONS = {
    "1_2": (1, 1),  # 1/2 = 50% accept
    "1_3": (1, 2),  # 1/3 = 33% accept
    "1_4": (1, 3),  # 1/4 = 25% accept
    "1_8": (1, 7),  # 1/8 = 12.5% accept
}


def load_data(path: str) -> List[Dict]:
    """Load original training data."""
    with open(path, "r") as f:
        return json.load(f)


def split_by_label(data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split data into accepts and rejects."""
    accepts = []
    rejects = []

    for sample in data:
        # Check the assistant's response (last conversation turn)
        assistant_response = sample["conversations"][-1]["value"]
        if "Accept" in assistant_response:
            accepts.append(sample)
        elif "Reject" in assistant_response:
            rejects.append(sample)
        else:
            print(f"Warning: Sample has unclear label: {assistant_response[:100]}")

    return accepts, rejects


def create_dataset(
    accepts: List[Dict],
    rejects: List[Dict],
    accept_ratio: int,
    reject_ratio: int,
    total_samples: int,
    seed: int = 42,
) -> List[Dict]:
    """
    Create a dataset with a fixed total size and exact accept:reject proportion.

    Args:
        accepts: All accept samples
        rejects: All reject samples
        accept_ratio: Accept part of ratio (e.g., 1 in 1:4)
        reject_ratio: Reject part of ratio (e.g., 4 in 1:4)
        total_samples: Fixed total number of samples in every dataset
        seed: Random seed for sampling

    Returns:
        List of samples with the exact desired proportion
    """
    random.seed(seed)

    # Compute exact counts from the fixed total using integer arithmetic.
    # Round the accept count to the nearest integer that keeps the ratio closest,
    # then assign the remainder to rejects.
    total_ratio = accept_ratio + reject_ratio
    n_accepts = round(total_samples * accept_ratio / total_ratio)
    n_rejects = total_samples - n_accepts

    assert n_accepts > 0, f"Not enough total samples for ratio {accept_ratio}:{reject_ratio}"
    assert n_rejects > 0, f"Not enough total samples for ratio {accept_ratio}:{reject_ratio}"
    assert n_accepts <= len(accepts), f"Need {n_accepts} accepts but only have {len(accepts)}"
    assert n_rejects <= len(rejects), f"Need {n_rejects} rejects but only have {len(rejects)}"

    # Sample exact counts
    sampled_accepts = random.sample(accepts, n_accepts)
    sampled_rejects = random.sample(rejects, n_rejects)

    # Combine and shuffle
    combined = sampled_accepts + sampled_rejects
    random.shuffle(combined)

    print(f"  Target: {n_accepts} accepts, {n_rejects} rejects (ratio {accept_ratio}:{reject_ratio})")
    print(f"  Created dataset: {len(combined)} total, {n_accepts/len(combined)*100:.1f}% accept")

    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Test on 100 samples only")
    args = parser.parse_args()

    # Load original data
    print("Loading original training data...")
    data = load_data(ORIGINAL_TRAIN)
    print(f"Loaded {len(data)} samples")

    # Split by label
    accepts, rejects = split_by_label(data)
    print(f"Accepts: {len(accepts)} ({len(accepts)/len(data)*100:.1f}%)")
    print(f"Rejects: {len(rejects)} ({len(rejects)/len(data)*100:.1f}%)")

    # Fixed total = 50% of original data, same for every proportion
    total_samples = len(data) // 2

    # In debug mode, limit the pool and total
    if args.debug:
        random.seed(42)
        accepts = random.sample(accepts, min(50, len(accepts)))
        rejects = random.sample(rejects, min(50, len(rejects)))
        total_samples = 50
        print(f"Debug mode: limited to {len(accepts)} accepts, {len(rejects)} rejects")

    print(f"Fixed dataset size: {total_samples} samples per proportion")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate datasets for each proportion
    for prop_name, (accept_ratio, reject_ratio) in PROPORTIONS.items():
        print(f"\nGenerating dataset: {prop_name} ({accept_ratio}:{reject_ratio})")

        dataset = create_dataset(accepts, rejects, accept_ratio, reject_ratio, total_samples)

        # Create dataset directory
        dataset_dir = OUTPUT_DIR / f"iclr_weighted_loss_train_{prop_name}"
        dataset_dir.mkdir(exist_ok=True)

        # Write data
        output_path = dataset_dir / "data.json"
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"  Saved to: {output_path}")

    print("\nDataset generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
