#!/usr/bin/env python3
"""
Generate title-only datasets for contamination checking.

Gives the model only the paper title and asks it to generate the abstract.
If the model can reproduce the abstract from the title alone, the paper is
likely in its training data (data contamination). The generated abstract is
then compared to the real abstract via embedding similarity downstream.

Usage:
    python 2_8_26/generate_title_dataset.py
    python 2_8_26/generate_title_dataset.py --limit 10  # test with 10 samples
"""

import argparse
import json
import os
from typing import Dict, List, Optional


# Original prompt prefix used in base datasets (needed for parsing)
ORIGINAL_PREFIX_END_MARKER = " - Note: ICLR generally has a ~30% acceptance rate\n\n"

# Title-only prompts — ask model to generate the abstract
TITLE_SYSTEM_PROMPT = (
    "You are a knowledgeable AI research assistant. You will be given only the title of "
    "an academic paper. Your task is to generate the abstract for this paper."
)

TITLE_USER_TEMPLATE = (
    "Given the following paper title, generate the abstract for this paper. "
    "Write only the abstract text, nothing else.\n\n"
    "Title: {title}"
)


def load_dataset(data_path: str) -> List[Dict]:
    """Load a dataset from a data.json file."""
    json_path = os.path.join(data_path, "data.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data: List[Dict], output_path: str):
    """Save a dataset to a data.json file."""
    os.makedirs(output_path, exist_ok=True)
    json_path = os.path.join(output_path, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} samples to {json_path}")


def extract_paper_content(user_message: str) -> str:
    """Extract the paper content from the original user message."""
    if ORIGINAL_PREFIX_END_MARKER in user_message:
        idx = user_message.find(ORIGINAL_PREFIX_END_MARKER)
        return user_message[idx + len(ORIGINAL_PREFIX_END_MARKER):].strip()
    return user_message


def extract_title(paper_content: str) -> str:
    """Extract the paper title from the paper content.

    The title is the first markdown H1 header (line starting with '# ').
    """
    for line in paper_content.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    # Fallback: use the first non-empty line
    for line in paper_content.split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


def transform_to_title_only(sample: Dict) -> Dict:
    """Transform a sample to contain only the paper title."""
    new_sample = sample.copy()
    conversations = []

    for conv in sample["conversations"]:
        new_conv = conv.copy()
        if conv["from"] == "system":
            new_conv["value"] = TITLE_SYSTEM_PROMPT
        elif conv["from"] == "human":
            paper_content = extract_paper_content(conv["value"])
            title = extract_title(paper_content)
            new_conv["value"] = TITLE_USER_TEMPLATE.format(title=title)
        # Keep gpt response (ground truth label) unchanged
        conversations.append(new_conv)

    new_sample["conversations"] = conversations
    # Remove images since title-only has none
    new_sample.pop("images", None)
    return new_sample


def generate_title_dataset(
    base_data_dir: str,
    output_dir: str,
    dataset_name: str,
    split: str = "test",
    limit: Optional[int] = None,
):
    """Generate the title-only dataset.

    Args:
        base_data_dir: Directory containing base datasets
        output_dir: Directory to save the title-only dataset
        dataset_name: Base dataset name (without split suffix)
        split: Dataset split to process
        limit: If set, only process the first N samples
    """
    os.makedirs(output_dir, exist_ok=True)

    full_name = f"{dataset_name}_{split}"
    input_path = os.path.join(base_data_dir, full_name)

    if not os.path.exists(input_path):
        print(f"Error: Dataset not found: {input_path}")
        return

    print(f"Processing {full_name}...")
    data = load_dataset(input_path)

    if limit is not None:
        data = data[:limit]
        print(f"  Limited to {len(data)} samples")

    title_data = [transform_to_title_only(s) for s in data]

    # Verify titles were extracted
    empty_titles = sum(
        1 for s in title_data
        if "Title: \n" in s["conversations"][1]["value"]
    )
    if empty_titles > 0:
        print(f"  Warning: {empty_titles} samples had empty titles")

    output_path = os.path.join(output_dir, f"{full_name}_title_only")
    save_dataset(title_data, output_path)

    # Print a sample for verification
    if title_data:
        sample_user = title_data[0]["conversations"][1]["value"]
        print(f"\n  Sample user message:\n  {sample_user[:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Generate title-only datasets for contamination checking")
    parser.add_argument("--base_data_dir", type=str,
                        default="/n/fs/vision-mix/sk7524/LLaMA-Factory/data",
                        help="Base directory containing original datasets")
    parser.add_argument("--output_dir", type=str,
                        default="./inference_scaling/data",
                        help="Output directory for title-only dataset")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to process")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N samples (for testing)")

    args = parser.parse_args()

    # Clean text-only dataset (v7/split7)
    dataset_name = "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7"

    generate_title_dataset(
        base_data_dir=args.base_data_dir,
        output_dir=args.output_dir,
        dataset_name=dataset_name,
        split=args.split,
        limit=args.limit,
    )

    print("\nTitle-only dataset generation complete!")


if __name__ == "__main__":
    main()
