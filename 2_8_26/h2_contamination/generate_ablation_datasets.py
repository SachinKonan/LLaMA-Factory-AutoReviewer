#!/usr/bin/env python3
"""
Generate progressive content ablation datasets for contamination checking.

Creates 5 content levels from the same test set:
1. title_only: Just the paper title
2. title_abstract: Title + Abstract
3. title_intro: Title + Abstract + Introduction
4. title_conclusion: Title + Abstract + Conclusion
5. full_paper: Full paper (baseline)

Usage:
    python 2_8_26/h2_contamination/generate_ablation_datasets.py
    python 2_8_26/h2_contamination/generate_ablation_datasets.py --limit 10
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.prompt_templates import CONTENT_ABLATION_SYSTEM_PROMPT, TITLE_ONLY_SYSTEM_PROMPT, TITLE_ONLY_USER_TEMPLATE

# Original prompt prefix marker
ORIGINAL_PREFIX_END_MARKER = " - Note: ICLR generally has a ~30% acceptance rate\n\n"

# Section patterns (reused from ablations/scripts_v1/generate_datasets.py)
SECTION_KEYWORDS = {
    "abstract": ["ABSTRACT"],
    "introduction": ["INTRODUCTION"],
    "conclusion": ["CONCLUSION", "CONCLUSIONS", "CONCLUDING REMARKS", "SUMMARY AND CONCLUSION"],
}


def load_dataset(data_path):
    json_path = os.path.join(data_path, "data.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data, output_path):
    os.makedirs(output_path, exist_ok=True)
    json_path = os.path.join(output_path, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} samples to {json_path}")


def extract_paper_content(user_message):
    """Extract paper content from original user message."""
    if ORIGINAL_PREFIX_END_MARKER in user_message:
        idx = user_message.find(ORIGINAL_PREFIX_END_MARKER)
        return user_message[idx + len(ORIGINAL_PREFIX_END_MARKER):].strip()
    return user_message


def extract_title(paper_content):
    """Extract the title (first H1 header)."""
    for line in paper_content.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    # Fallback: first non-empty line
    for line in paper_content.split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


def find_section_bounds(lines, section_keywords):
    """Find the start and end line indices of a section.

    Returns (start_line, end_line) or (None, None) if not found.
    """
    start_line = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            header_text = stripped.lstrip("#").strip().upper()
            # Check if this header matches any keyword
            if any(kw in header_text for kw in section_keywords):
                start_line = i
                break

    if start_line is None:
        return None, None

    # Find end: next header at same or higher level
    header_level = len(lines[start_line].strip()) - len(lines[start_line].strip().lstrip("#"))
    end_line = len(lines)
    for i in range(start_line + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            if level <= header_level:
                end_line = i
                break

    return start_line, end_line


def extract_title_and_abstract(paper_content):
    """Extract title + abstract from paper."""
    lines = paper_content.split("\n")

    # Find abstract section
    abs_start, abs_end = find_section_bounds(lines, SECTION_KEYWORDS["abstract"])

    if abs_start is not None:
        # Title is everything before abstract
        title_part = "\n".join(lines[:abs_start]).rstrip()
        abstract_part = "\n".join(lines[abs_start:abs_end]).rstrip()
        return f"{title_part}\n\n{abstract_part}"

    # Fallback: find first numbered section header, everything before is title+abstract
    for i, line in enumerate(lines):
        if i == 0:
            continue
        stripped = line.strip()
        if stripped.startswith("#") and re.match(r'^#\s*\d+', stripped):
            return "\n".join(lines[:i]).rstrip()

    # Last fallback: first 50 lines
    return "\n".join(lines[:min(50, len(lines))]).rstrip()


def extract_section(paper_content, section_type):
    """Extract a specific section from the paper.

    Returns the section text, or None if not found.
    """
    lines = paper_content.split("\n")
    keywords = SECTION_KEYWORDS.get(section_type, [])
    if not keywords:
        return None

    start, end = find_section_bounds(lines, keywords)
    if start is None:
        return None

    return "\n".join(lines[start:end]).rstrip()


def create_content_level(sample, level):
    """Create a new sample at a specific content level.

    Levels:
    - title_only: Just the title (ask model to generate abstract)
    - title_abstract: Title + Abstract
    - title_intro: Title + Abstract + Introduction
    - title_conclusion: Title + Abstract + Conclusion
    - full_paper: Full paper content
    """
    new_sample = json.loads(json.dumps(sample))  # Deep copy

    # Extract full paper content
    user_msg = None
    for conv in sample["conversations"]:
        if conv["from"] == "human":
            user_msg = conv["value"]
            break
    if user_msg is None:
        return None

    paper_content = extract_paper_content(user_msg)

    if level == "title_only":
        title = extract_title(paper_content)
        if not title:
            return None
        # Title-only uses a different prompt (generate abstract)
        new_conversations = []
        for conv in new_sample["conversations"]:
            new_conv = conv.copy()
            if conv["from"] == "system":
                new_conv["value"] = TITLE_ONLY_SYSTEM_PROMPT
            elif conv["from"] == "human":
                new_conv["value"] = TITLE_ONLY_USER_TEMPLATE.format(title=title)
            new_conversations.append(new_conv)
        new_sample["conversations"] = new_conversations

    elif level == "title_abstract":
        content = extract_title_and_abstract(paper_content)
        for conv in new_sample["conversations"]:
            if conv["from"] == "system":
                conv["value"] = CONTENT_ABLATION_SYSTEM_PROMPT
            elif conv["from"] == "human":
                conv["value"] = content

    elif level == "title_intro":
        ta = extract_title_and_abstract(paper_content)
        intro = extract_section(paper_content, "introduction")
        content = ta
        if intro:
            content = f"{ta}\n\n{intro}"
        for conv in new_sample["conversations"]:
            if conv["from"] == "system":
                conv["value"] = CONTENT_ABLATION_SYSTEM_PROMPT
            elif conv["from"] == "human":
                conv["value"] = content

    elif level == "title_conclusion":
        ta = extract_title_and_abstract(paper_content)
        conclusion = extract_section(paper_content, "conclusion")
        content = ta
        if conclusion:
            content = f"{ta}\n\n{conclusion}"
        for conv in new_sample["conversations"]:
            if conv["from"] == "system":
                conv["value"] = CONTENT_ABLATION_SYSTEM_PROMPT
            elif conv["from"] == "human":
                conv["value"] = content

    elif level == "full_paper":
        for conv in new_sample["conversations"]:
            if conv["from"] == "system":
                conv["value"] = CONTENT_ABLATION_SYSTEM_PROMPT
            elif conv["from"] == "human":
                conv["value"] = paper_content

    # Remove images for all text-only ablations
    new_sample.pop("images", None)
    return new_sample


def main():
    parser = argparse.ArgumentParser(description="Generate contamination ablation datasets")
    parser.add_argument("--base_data_dir", type=str,
                        default="/n/fs/vision-mix/sk7524/LLaMA-Factory/data")
    parser.add_argument("--output_dir", type=str,
                        default="./2_8_26/h2_contamination/data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset_name = "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7"
    full_name = f"{dataset_name}_{args.split}"
    input_path = os.path.join(args.base_data_dir, full_name)

    if not os.path.exists(input_path):
        print(f"Error: Dataset not found: {input_path}")
        return

    print(f"Loading {full_name}...")
    data = load_dataset(input_path)

    if args.limit is not None:
        data = data[:args.limit]
        print(f"Limited to {len(data)} samples")

    levels = ["title_only", "title_abstract", "title_intro", "title_conclusion", "full_paper"]
    dataset_info = {}

    for level in levels:
        print(f"\nGenerating {level} dataset...")
        level_data = []
        skipped = 0

        for sample in data:
            result = create_content_level(sample, level)
            if result is not None:
                level_data.append(result)
            else:
                skipped += 1

        if skipped > 0:
            print(f"  Skipped {skipped} samples (missing content)")

        output_name = f"{full_name}_{level}"
        output_path = os.path.join(args.output_dir, output_name)
        save_dataset(level_data, output_path)

        # Register in dataset_info
        dataset_info[output_name] = {
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

        # Print sample
        if level_data:
            user_msg = [c["value"] for c in level_data[0]["conversations"] if c["from"] == "human"][0]
            print(f"  Sample (first 200 chars): {user_msg[:200]}...")

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

    # Summary
    print(f"\n{'='*60}")
    print(f"Generated {len(levels)} ablation datasets:")
    for level in levels:
        print(f"  - {full_name}_{level}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
