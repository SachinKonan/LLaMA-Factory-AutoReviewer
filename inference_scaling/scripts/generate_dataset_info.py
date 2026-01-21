#!/usr/bin/env python3
"""
Generate dataset_info.json entries for the inference scaling datasets.

This script creates the necessary LlamaFactory dataset configuration entries
for all generated prompt variants.

Usage:
    python generate_dataset_info.py --output ./inference_scaling/data/dataset_info.json
"""

import argparse
import json
import os
from pathlib import Path


def generate_dataset_entry(dataset_name: str, has_images: bool = False) -> dict:
    """Generate a dataset_info.json entry for a dataset."""
    entry = {
        "file_name": f"{dataset_name}/data.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system"
        }
    }

    if has_images:
        entry["columns"]["images"] = "images"

    return entry


def generate_dataset_info(data_dir: str, output_path: str):
    """Generate dataset_info.json for all datasets in the data directory."""
    dataset_info = {}

    # Scan for dataset directories
    data_path = Path(data_dir)
    for dataset_dir in sorted(data_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        data_json = dataset_dir / "data.json"

        if not data_json.exists():
            print(f"Skipping {dataset_name}: no data.json found")
            continue

        # Determine if dataset has images based on name
        has_images = "images" in dataset_name or "vision" in dataset_name

        # For vision/clean_vision datasets, also check the data format
        if has_images:
            try:
                with open(data_json, "r") as f:
                    sample = json.load(f)[0]
                    if "images" not in sample:
                        has_images = False
            except (json.JSONDecodeError, IndexError, KeyError):
                pass

        entry = generate_dataset_entry(dataset_name, has_images)
        dataset_info[dataset_name] = entry
        print(f"Added: {dataset_name} (images={has_images})")

    # Save dataset_info.json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\nGenerated {len(dataset_info)} dataset entries")
    print(f"Saved to: {output_path}")

    return dataset_info


def main():
    parser = argparse.ArgumentParser(description="Generate dataset_info.json for inference scaling datasets")
    parser.add_argument("--data_dir", type=str,
                        default="./inference_scaling/data",
                        help="Directory containing generated datasets")
    parser.add_argument("--output", type=str,
                        default="./inference_scaling/data/dataset_info.json",
                        help="Output path for dataset_info.json")

    args = parser.parse_args()

    generate_dataset_info(args.data_dir, args.output)


if __name__ == "__main__":
    main()
