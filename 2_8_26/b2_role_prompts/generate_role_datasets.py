#!/usr/bin/env python3
"""
Generate role-based datasets for bias mitigation experiments.

Creates datasets with different reviewer personas:
- critical: Harsh reviewer, focuses on weaknesses
- enthusiastic: Optimistic reviewer, focuses on strengths
- standard: Baseline (no modifier)

Generates 9 datasets: 3 roles × 3 modalities (clean, clean_images, vision)

Usage:
    python 2_8_26/b2_role_prompts/generate_role_datasets.py
    python 2_8_26/b2_role_prompts/generate_role_datasets.py --limit 10
    python 2_8_26/b2_role_prompts/generate_role_datasets.py --roles critical enthusiastic
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.prompt_templates import build_system_prompt

# Configuration
BASE_DATA_DIR = Path("/n/fs/vision-mix/sk7524/LLaMA-Factory/data")
BASE_PREFIX = "iclr_2020_2025_85_5_10_split7_balanced"
ORIGINAL_PREFIX_END_MARKER = " - Note: ICLR generally has a ~30% acceptance rate\n\n"

ROLES = ["critical", "enthusiastic", "standard"]
MODALITIES = ["clean", "clean_images", "vision"]


def load_dataset(path):
    with open(path / "data.json", "r") as f:
        return json.load(f)


def save_dataset(data, output_path):
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "data.json"), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} samples to {output_path}")


def extract_paper_content(user_message):
    if ORIGINAL_PREFIX_END_MARKER in user_message:
        idx = user_message.find(ORIGINAL_PREFIX_END_MARKER)
        return user_message[idx + len(ORIGINAL_PREFIX_END_MARKER):].strip()
    return user_message


def clean_image_tags(text):
    return re.sub(r'<image>\s*', '', text)


def get_base_dataset_path(modality, split):
    if modality == "clean":
        name = f"{BASE_PREFIX}_clean_binary_noreviews_v7_{split}"
    elif modality == "clean_images":
        name = f"{BASE_PREFIX}_clean_images_binary_noreviews_v7_{split}"
    elif modality == "vision":
        name = f"{BASE_PREFIX}_vision_binary_noreviews_v7_{split}"
    else:
        raise ValueError(f"Unknown modality: {modality}")
    return BASE_DATA_DIR / name


def create_role_dataset(base_data, role, modality, output_format="json"):
    """Create a dataset with the given role modifier."""
    system_prompt = build_system_prompt(modifier=role, output_format=output_format)
    new_data = []

    for entry in base_data:
        new_entry = json.loads(json.dumps(entry))  # Deep copy

        # Update user message to just paper content
        for msg in new_entry["conversations"]:
            if msg["from"] == "human":
                paper_content = extract_paper_content(msg["value"])
                if modality == "clean":
                    paper_content = clean_image_tags(paper_content)
                msg["value"] = paper_content

        # Set system prompt with role modifier
        conversations = [msg for msg in new_entry["conversations"] if msg.get("from") != "system"]
        conversations.insert(0, {"from": "system", "value": system_prompt})
        new_entry["conversations"] = conversations

        new_data.append(new_entry)

    return new_data


def main():
    parser = argparse.ArgumentParser(description="Generate role-based datasets")
    parser.add_argument("--output_dir", type=str, default="./2_8_26/b2_role_prompts/data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--roles", nargs="+", default=ROLES, choices=ROLES)
    parser.add_argument("--modalities", nargs="+", default=MODALITIES, choices=MODALITIES)
    parser.add_argument("--output_format", type=str, default="json", choices=["boxed", "json"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_info = {}

    for modality in args.modalities:
        base_path = get_base_dataset_path(modality, args.split)
        if not base_path.exists():
            print(f"Warning: Skipping {modality}, dataset not found at {base_path}")
            continue

        print(f"\nLoading {modality} dataset...")
        data = load_dataset(base_path)

        if args.limit is not None:
            data = data[:args.limit]
            print(f"  Limited to {len(data)} samples")

        for role in args.roles:
            print(f"  Generating {role} dataset for {modality}...")
            role_data = create_role_dataset(data, role, modality, args.output_format)

            output_name = f"{BASE_PREFIX}_{modality}_v7_{args.split}_{role}"
            output_path = os.path.join(args.output_dir, output_name)
            save_dataset(role_data, output_path)

            # Determine if vision
            is_vision = modality in ["clean_images", "vision"]
            columns = {"messages": "conversations"}
            if is_vision:
                columns["images"] = "images"

            dataset_info[output_name] = {
                "file_name": f"{output_name}/data.json",
                "formatting": "sharegpt",
                "columns": columns,
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                    "system_tag": "system",
                },
            }

    # Save dataset_info.json
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    existing = {}
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            existing = json.load(f)
    existing.update(dataset_info)
    with open(info_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nUpdated {info_path}")

    print(f"\n{'='*60}")
    print(f"Generated {len(dataset_info)} role datasets:")
    for name in sorted(dataset_info.keys()):
        print(f"  - {name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
