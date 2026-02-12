#!/usr/bin/env python3
"""
Validate SFT and DPO Training Datasets.

Checks:
- Format correctness (conversations, chosen/rejected for DPO)
- Metadata completeness
- Score distributions
- Contrast requirements for DPO

Usage:
    python 2_8_26/training_checklist/scripts/validate_datasets.py \
        --data_dir 2_8_26/training_checklist/data \
        --stage sft  # or dpo or all
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def load_dataset(data_path: str) -> List[Dict]:
    """Load dataset from data.json file."""
    json_path = Path(data_path) / "data.json"
    if not json_path.exists():
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_sft_dataset(data_path: str, dataset_name: str) -> List[str]:
    """Validate SFT dataset format and content."""
    print(f"\nValidating {dataset_name}...")

    data = load_dataset(data_path)
    if not data:
        return [f"ERROR: No data found in {data_path}"]

    errors = []
    print(f"  Total samples: {len(data)}")

    # Format validation
    for i, sample in enumerate(data):
        if "conversations" not in sample:
            errors.append(f"Sample {i}: Missing 'conversations' key")
            continue

        convs = sample["conversations"]
        if len(convs) != 3:
            errors.append(f"Sample {i}: Expected 3 conversation turns, got {len(convs)}")

        # Check conversation format
        expected_roles = ["system", "human", "gpt"]
        for j, (conv, expected_role) in enumerate(zip(convs, expected_roles)):
            if conv.get("from") != expected_role:
                errors.append(f"Sample {i}, turn {j}: Expected role '{expected_role}', got '{conv.get('from')}'")

        # Metadata validation
        if "_metadata" not in sample:
            errors.append(f"Sample {i}: Missing '_metadata' key")
            continue

        meta = sample["_metadata"]
        required_meta = ["submission_id", "answer", "year", "checkmark_score"]
        for key in required_meta:
            if key not in meta:
                errors.append(f"Sample {i}: Missing metadata '{key}'")

    # Score distribution
    scores = [s["_metadata"]["checkmark_score"] for s in data if "_metadata" in s and "checkmark_score" in s["_metadata"]]
    if scores:
        print(f"  Checkmark score stats:")
        print(f"    Min: {min(scores):.3f}")
        print(f"    Max: {max(scores):.3f}")
        print(f"    Mean: {sum(scores)/len(scores):.3f}")

    # Decision distribution
    decisions = [s["_metadata"].get("answer") for s in data if "_metadata" in s]
    decision_counts = Counter(decisions)
    print(f"  Decision distribution:")
    for decision, count in decision_counts.items():
        print(f"    {decision}: {count} ({count/len(decisions)*100:.1f}%)")

    if errors:
        print(f"  ❌ Found {len(errors)} errors")
        for error in errors[:10]:  # Show first 10
            print(f"    - {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")
    else:
        print(f"  ✅ All checks passed")

    return errors


def validate_dpo_dataset(data_path: str, dataset_name: str) -> List[str]:
    """Validate DPO dataset format and preference pairs."""
    print(f"\nValidating {dataset_name}...")

    data = load_dataset(data_path)
    if not data:
        return [f"ERROR: No data found in {data_path}"]

    errors = []
    print(f"  Total samples: {len(data)}")

    # Format validation
    for i, sample in enumerate(data):
        if "conversations" not in sample:
            errors.append(f"Sample {i}: Missing 'conversations' key")

        if "chosen" not in sample:
            errors.append(f"Sample {i}: Missing 'chosen' key")

        if "rejected" not in sample:
            errors.append(f"Sample {i}: Missing 'rejected' key")

        # Metadata validation
        if "_metadata" not in sample:
            errors.append(f"Sample {i}: Missing '_metadata' key")
            continue

        meta = sample["_metadata"]

        # Check decision is Accept
        if meta.get("decision") != "Accept":
            errors.append(f"Sample {i}: Expected only accepted papers, got '{meta.get('decision')}'")

        # Check rating contrast
        if "rating_diff" in meta and meta["rating_diff"] < 2:
            errors.append(f"Sample {i}: Rating difference too small ({meta['rating_diff']})")

        # Check checkmark contrast
        if "checkmark_diff" in meta and meta["checkmark_diff"] < 2:
            errors.append(f"Sample {i}: Checkmark difference too small ({meta['checkmark_diff']})")

    # Contrast statistics
    rating_diffs = [s["_metadata"].get("rating_diff") for s in data if "_metadata" in s and "rating_diff" in s["_metadata"]]
    checkmark_diffs = [s["_metadata"].get("checkmark_diff") for s in data if "_metadata" in s and "checkmark_diff" in s["_metadata"]]

    if rating_diffs:
        print(f"  Rating difference stats:")
        print(f"    Min: {min(rating_diffs):.1f}")
        print(f"    Max: {max(rating_diffs):.1f}")
        print(f"    Mean: {sum(rating_diffs)/len(rating_diffs):.1f}")

    if checkmark_diffs:
        print(f"  Checkmark difference stats:")
        print(f"    Min: {min(checkmark_diffs)}")
        print(f"    Max: {max(checkmark_diffs)}")
        print(f"    Mean: {sum(checkmark_diffs)/len(checkmark_diffs):.1f}")

    if errors:
        print(f"  ❌ Found {len(errors)} errors")
        for error in errors[:10]:  # Show first 10
            print(f"    - {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")
    else:
        print(f"  ✅ All checks passed")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate training datasets")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing datasets",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["sft", "dpo", "all"],
        default="all",
        help="Which stage to validate",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)

    data_dir = Path(args.data_dir)
    all_errors = {}

    modalities = ["clean", "clean_images", "vision"]
    splits = ["train", "val"]

    # Validate SFT datasets
    if args.stage in ["sft", "all"]:
        print(f"\n{'='*80}")
        print("SFT DATASETS")
        print("=" * 80)

        for modality in modalities:
            for split in splits:
                dataset_name = f"sft_{modality}_{split}"
                data_path = data_dir / dataset_name
                errors = validate_sft_dataset(str(data_path), dataset_name)
                if errors:
                    all_errors[dataset_name] = errors

    # Validate DPO datasets
    if args.stage in ["dpo", "all"]:
        print(f"\n{'='*80}")
        print("DPO DATASETS")
        print("=" * 80)

        for modality in modalities:
            for split in splits:
                dataset_name = f"dpo_{modality}_{split}"
                data_path = data_dir / dataset_name
                errors = validate_dpo_dataset(str(data_path), dataset_name)
                if errors:
                    all_errors[dataset_name] = errors

    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if all_errors:
        print(f"\n❌ Found errors in {len(all_errors)} dataset(s):")
        for dataset_name, errors in all_errors.items():
            print(f"  - {dataset_name}: {len(errors)} errors")
    else:
        print(f"\n✅ All datasets passed validation!")

    return 0 if not all_errors else 1


if __name__ == "__main__":
    exit(main())
