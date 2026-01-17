#!/usr/bin/env python3
"""
Create new ShareGPT datasets using optimized prompts from TextGrad.

This script reads the original dataset, replaces the system prompt and human prefix
with the optimized versions, and saves as new datasets.
"""

import json
import shutil
from pathlib import Path


def load_best_prompts(prompts_file: str) -> dict:
    """Load optimized prompts from JSON file."""
    with open(prompts_file) as f:
        return json.load(f)


def create_optimized_dataset(
    source_dataset_dir: Path,
    target_dataset_dir: Path,
    system_prompt: str,
    human_prefix: str,
) -> int:
    """
    Create a new dataset with optimized prompts.

    Returns the number of samples processed.
    """
    source_data_file = source_dataset_dir / "data.json"
    target_dataset_dir.mkdir(parents=True, exist_ok=True)
    target_data_file = target_dataset_dir / "data.json"

    with open(source_data_file) as f:
        data = json.load(f)

    modified_data = []
    for item in data:
        conversations = item.get("conversations", [])
        new_conversations = []

        for msg in conversations:
            role = msg.get("from", "")
            value = msg.get("value", "")

            if role == "system":
                # Replace system prompt
                new_conversations.append({
                    "from": "system",
                    "value": system_prompt,
                })
            elif role == "human":
                # Replace human prefix, keep paper content
                # Split on "\n\n# " to find where paper content starts
                parts = value.split("\n\n# ", 1)
                if len(parts) == 2:
                    paper_content = "# " + parts[1]
                else:
                    # Fallback - keep original if can't split
                    paper_content = value

                new_human_message = human_prefix + "\n\n" + paper_content
                new_conversations.append({
                    "from": "human",
                    "value": new_human_message,
                })
            else:
                # Keep assistant response as-is
                new_conversations.append(msg)

        modified_data.append({"conversations": new_conversations})

    with open(target_data_file, "w") as f:
        json.dump(modified_data, f, indent=2)

    return len(modified_data)


def main():
    base_dir = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer")
    data_dir = base_dir / "data"
    results_dir = base_dir / "textgrad_experiments_v2" / "results"

    # Source datasets to convert
    source_datasets = [
        "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_validation",
        "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_test",
    ]

    # Best prompts files
    prompts_configs = [
        {
            "file": results_dir / "best_prompts_20260117_033408.json",
            "suffix": "textgrad_8b",  # 8B gradient model
        },
        {
            "file": results_dir / "best_prompts_20260117_033411.json",
            "suffix": "textgrad_14b",  # 14B gradient model
        },
        {
            "file": results_dir / "best_prompts_20260117_145512.json",
            "suffix": "textgrad_32b",  # 32B gradient model
        },
    ]

    # Process each prompt config
    for config in prompts_configs:
        prompts = load_best_prompts(config["file"])
        system_prompt = prompts["system_prompt"]
        human_prefix = prompts["human_prefix"]
        suffix = config["suffix"]

        print(f"\n{'='*60}")
        print(f"Creating datasets with {suffix} prompts")
        print(f"{'='*60}")
        print(f"System prompt: {system_prompt[:80]}...")
        print(f"Human prefix: {human_prefix[:80]}...")

        for source_name in source_datasets:
            source_dir = data_dir / source_name

            # Create new dataset name
            # e.g., iclr_..._validation -> iclr_..._validation_textgrad_8b
            target_name = f"{source_name}_{suffix}"
            target_dir = data_dir / target_name

            print(f"\n  {source_name}")
            print(f"  -> {target_name}")

            num_samples = create_optimized_dataset(
                source_dir, target_dir, system_prompt, human_prefix
            )
            print(f"     Processed {num_samples} samples")

    # Update dataset_info.json
    print(f"\n{'='*60}")
    print("Updating dataset_info.json")
    print(f"{'='*60}")

    dataset_info_path = data_dir / "dataset_info.json"
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    # Add new dataset entries
    for config in prompts_configs:
        suffix = config["suffix"]
        for source_name in source_datasets:
            target_name = f"{source_name}_{suffix}"
            target_name_test = f"{target_name}_test"  # For inference script compatibility

            # Add entry pointing to data.json
            dataset_info[target_name] = {
                "file_name": f"{target_name}/data.json",
                "formatting": "sharegpt",
                "columns": {"messages": "conversations"},
            }
            # Also add _test variant (inference script appends _test)
            dataset_info[target_name_test] = {
                "file_name": f"{target_name}/data.json",
                "formatting": "sharegpt",
                "columns": {"messages": "conversations"},
            }
            print(f"  Added: {target_name}")

    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\nDone! Created {len(prompts_configs) * len(source_datasets)} new datasets.")


if __name__ == "__main__":
    main()
