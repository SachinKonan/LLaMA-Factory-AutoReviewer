#!/usr/bin/env python3
"""
Create a critical analysis dataset by appending a critical analysis prompt prefix
to the human prompt in the base text dataset.

This dataset is limited to 10 samples for quick testing.
"""

import json
import os
from pathlib import Path

# Configuration
BASE_DATASET = "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_test"
OUTPUT_NAME = "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_critical_analysis_v6_test"
MAX_SAMPLES = 100

CRITICAL_ANALYSIS_INSERT = (
    ". A lot of papers tend to overclaim or their results aren't rigorous or there exist work "
    "that is highly similar, please look at these axes and critically analyze the paper."
)

# Text to find and modify
ORIGINAL_NOTE = "Note: ICLR generally has a ~30% acceptance rate"
MODIFIED_NOTE = ORIGINAL_NOTE + CRITICAL_ANALYSIS_INSERT

def main():
    # Paths
    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / "data"

    base_path = data_dir / BASE_DATASET / "data.json"
    output_dir = data_dir / OUTPUT_NAME
    output_path = output_dir / "data.json"
    dataset_info_path = data_dir / "dataset_info.json"

    # Load base dataset
    print(f"Loading base dataset from {base_path}")
    with open(base_path, "r") as f:
        data = json.load(f)

    print(f"Base dataset has {len(data)} samples")

    # Limit samples
    data = data[:MAX_SAMPLES]
    print(f"Limited to {len(data)} samples")

    # Process each sample - insert critical analysis note after acceptance rate note
    for sample in data:
        for conv in sample["conversations"]:
            if conv["from"] == "human":
                # Replace the original note with the modified note
                conv["value"] = conv["value"].replace(ORIGINAL_NOTE, MODIFIED_NOTE)
                break

    # Save output dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path}")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    # Update dataset_info.json
    print(f"Updating {dataset_info_path}")
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)

    dataset_info[OUTPUT_NAME] = {
        "file_name": f"{OUTPUT_NAME}/data.json",
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

    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("Done!")
    print(f"\nDataset created: {OUTPUT_NAME}")
    print(f"Samples: {len(data)}")

    # Show sample of modified prompt
    print("\n--- Sample modified prompt (first 500 chars) ---")
    for conv in data[0]["conversations"]:
        if conv["from"] == "human":
            print(conv["value"][:500])
            break


if __name__ == "__main__":
    main()
