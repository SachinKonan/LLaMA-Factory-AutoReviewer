#!/usr/bin/env python3
"""
Convert boxed format dataset to Y/N format.

Converts:
- Prompt: "\\boxed{Accept} or \\boxed{Reject}" -> "Y (Accept) or N (Reject)"
- Response: "Outcome: \\boxed{Accept}" -> "Y"
- Response: "Outcome: \\boxed{Reject}" -> "N"
"""

import json
import os
import argparse
import re
from pathlib import Path


def convert_prompt(text: str) -> str:
    """Convert the prompt from boxed format to Y/N format."""
    # Replace the instruction about answer format
    text = text.replace(
        "Your answer will either be: \\boxed{Accept} or \\boxed{Reject}",
        "Your answer will either be: Y (Accept) or N (Reject)"
    )
    return text


def convert_response(text: str) -> str:
    """Convert the response from boxed format to Y/N format."""
    if "\\boxed{Accept}" in text:
        return "Y"
    elif "\\boxed{Reject}" in text:
        return "N"
    else:
        # If no boxed format found, return original
        print(f"Warning: No boxed format found in response: {text}")
        return text


def convert_sample(sample: dict) -> dict:
    """Convert a single sample from boxed to Y/N format."""
    new_sample = sample.copy()

    if "conversations" in new_sample:
        new_conversations = []
        for conv in new_sample["conversations"]:
            new_conv = conv.copy()
            if conv.get("from") == "human":
                new_conv["value"] = convert_prompt(conv["value"])
            elif conv.get("from") == "gpt":
                new_conv["value"] = convert_response(conv["value"])
            new_conversations.append(new_conv)
        new_sample["conversations"] = new_conversations

    return new_sample


def convert_dataset(input_path: str, output_path: str) -> None:
    """Convert an entire dataset file."""
    print(f"Reading from: {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Converting {len(data)} samples...")

    converted_data = []
    accept_count = 0
    reject_count = 0

    for sample in data:
        converted = convert_sample(sample)
        converted_data.append(converted)

        # Count Y/N for verification
        for conv in converted.get("conversations", []):
            if conv.get("from") == "gpt":
                if conv["value"] == "Y":
                    accept_count += 1
                elif conv["value"] == "N":
                    reject_count += 1

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Writing to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2)

    print(f"Converted {len(converted_data)} samples")
    print(f"  Y (Accept): {accept_count}")
    print(f"  N (Reject): {reject_count}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Convert boxed format to Y/N format")
    parser.add_argument(
        "--input_base",
        type=str,
        required=True,
        help="Base name of input dataset (without _train/_validation/_test suffix)"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        required=True,
        help="Base name of output dataset (without _train/_validation/_test suffix)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to convert"
    )

    args = parser.parse_args()

    for split in args.splits:
        input_path = os.path.join(
            args.data_dir,
            f"{args.input_base}_{split}",
            "data.json"
        )
        output_path = os.path.join(
            args.data_dir,
            f"{args.output_base}_{split}",
            "data.json"
        )

        if os.path.exists(input_path):
            convert_dataset(input_path, output_path)
        else:
            print(f"Warning: Input file not found: {input_path}")


if __name__ == "__main__":
    main()
