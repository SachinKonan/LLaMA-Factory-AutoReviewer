#!/usr/bin/env python3
"""
Generate a dataset formatted for base (non-instruct) model inference.

Base models don't follow chat templates, so we reformat the ShareGPT conversation
into a single completion-style prompt. The model sees the prompt as a text continuation
task rather than a chat instruction.

Usage:
    python 2_8_26/h1_base_model/generate_dataset.py
    python 2_8_26/h1_base_model/generate_dataset.py --limit 10
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.prompt_templates import BASE_COMPLETION_SYSTEM_PROMPT

# Original prompt prefix marker (for extracting paper content)
ORIGINAL_PREFIX_END_MARKER = " - Note: ICLR generally has a ~30% acceptance rate\n\n"


def load_dataset(data_path):
    json_path = os.path.join(data_path, "data.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data, output_path):
    os.makedirs(output_path, exist_ok=True)
    json_path = os.path.join(output_path, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} samples to {json_path}")


def extract_paper_content(user_message):
    """Extract the paper content from the original user message."""
    if ORIGINAL_PREFIX_END_MARKER in user_message:
        idx = user_message.find(ORIGINAL_PREFIX_END_MARKER)
        return user_message[idx + len(ORIGINAL_PREFIX_END_MARKER):].strip()
    return user_message


def transform_for_base_model(sample):
    """Transform a ShareGPT sample into a completion-style format for base models.

    We keep the ShareGPT structure (system/human/gpt) since LLaMA Factory's
    `default` template will format it as a simple prompt without chat tokens.
    The system prompt is simplified and the user message is just the paper content.
    """
    new_sample = sample.copy()
    conversations = []

    for conv in sample["conversations"]:
        new_conv = conv.copy()
        if conv["from"] == "system":
            new_conv["value"] = BASE_COMPLETION_SYSTEM_PROMPT
        elif conv["from"] == "human":
            paper_content = extract_paper_content(conv["value"])
            new_conv["value"] = paper_content
        # Keep gpt response (ground truth label) unchanged
        conversations.append(new_conv)

    # Ensure system prompt exists
    if not any(c["from"] == "system" for c in conversations):
        conversations.insert(0, {"from": "system", "value": BASE_COMPLETION_SYSTEM_PROMPT})

    new_sample["conversations"] = conversations
    # Remove images — base model is text-only
    new_sample.pop("images", None)
    return new_sample


def main():
    parser = argparse.ArgumentParser(description="Generate base model dataset")
    parser.add_argument("--base_data_dir", type=str,
                        default="/n/fs/vision-mix/sk7524/LLaMA-Factory/data",
                        help="Base directory containing original datasets")
    parser.add_argument("--output_dir", type=str,
                        default="./2_8_26/h1_base_model/data",
                        help="Output directory")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    dataset_name = "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7"
    full_name = f"{dataset_name}_{args.split}"
    input_path = os.path.join(args.base_data_dir, full_name)

    if not os.path.exists(input_path):
        print(f"Error: Dataset not found: {input_path}")
        return

    print(f"Processing {full_name}...")
    data = load_dataset(input_path)

    if args.limit is not None:
        data = data[:args.limit]
        print(f"Limited to {len(data)} samples")

    base_data = [transform_for_base_model(s) for s in data]

    output_name = f"{full_name}_base_model"
    output_path = os.path.join(args.output_dir, output_name)
    save_dataset(base_data, output_path)

    # Also generate dataset_info.json entry
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)

    info[output_name] = {
        "file_name": f"{output_name}/data.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Updated {info_path}")

    # Print sample
    if base_data:
        print(f"\nSample system prompt:\n  {base_data[0]['conversations'][0]['value'][:200]}...")
        print(f"\nSample user message (first 200 chars):\n  {base_data[0]['conversations'][1]['value'][:200]}...")


if __name__ == "__main__":
    main()
