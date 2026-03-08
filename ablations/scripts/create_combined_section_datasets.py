#!/usr/bin/env python3
"""
Create combined section datasets:
1. Only: Related work + methodology + experimental results
2. Except: Everything but related work + methodology + experimental results

These are the "core technical content" sections.
"""

import json
import re
import os
from pathlib import Path

# Configuration
DATA_DIR = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/data")
BASE_PREFIX = "iclr_2020_2025_85_5_10_split6_original"

# Section patterns
SECTIONS = {
    "introduction": re.compile(r"^#\s*(1\s*)?INTRODUCTION", re.MULTILINE | re.IGNORECASE),
    "related_work": re.compile(r"^#\s*(2\s*)?RELATED WORK", re.MULTILINE | re.IGNORECASE),
    "methodology": re.compile(r"^#\s*(3\s*)?(METHOD|METHODOLOGY|PROPOSED METHOD)", re.MULTILINE | re.IGNORECASE),
    "experimental_results": re.compile(r"^#\s*(4\s*)?(EXPERIMENTS|RESULTS|EXPERIMENTAL RESULTS)", re.MULTILINE | re.IGNORECASE),
    "discussion": re.compile(r"^#\s*(5\s*)?(DISCUSSION|LIMITATIONS)", re.MULTILINE | re.IGNORECASE),
    "conclusion": re.compile(r"^#\s*(6\s*)?CONCLUSION", re.MULTILINE | re.IGNORECASE),
}

KEYWORDS = {
    "introduction": ["INTRODUCTION"],
    "related_work": ["RELATED WORK", "BACKGROUND"],
    "methodology": ["METHOD", "METHODOLOGY", "APPROACH"],
    "experimental_results": ["RESULTS", "EXPERIMENTS", "EVALUATION"],
    "discussion": ["DISCUSSION", "LIMITATIONS"],
    "conclusion": ["CONCLUSION"],
}

# Combined sections we want
CORE_TECHNICAL_SECTIONS = ["related_work", "methodology", "experimental_results"]


def load_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_dataset(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def parse_and_ablate_combined(text, filter_type, target_sections):
    """
    Parses the text and performs ablation for multiple combined sections.

    filter_type: 'only' (keep only these sections + title/abstract)
                 'except' (remove these sections)
    target_sections: list of section keys to include/exclude
    """
    lines = text.split('\n')

    # Find all headers
    headers = []
    for i, line in enumerate(lines):
        if line.strip().startswith('# '):
            headers.append((i, line.strip()))

    if not headers:
        return None

    # Map headers to section keys
    section_map = {}
    for i, (line_idx, line_content) in enumerate(headers):
        matched = None
        for key, pattern in SECTIONS.items():
            if pattern.match(line_content):
                matched = key
                break

        # Keyword fallback
        if not matched:
            upper_line = line_content.upper()
            for key, kws in KEYWORDS.items():
                if any(kw in upper_line for kw in kws):
                    matched = key
                    break

        if matched:
            section_map[i] = matched

    # Check if any target section is found
    found_targets = [i for i, sec in section_map.items() if sec in target_sections]
    if not found_targets:
        return None  # None of the target sections found

    filtered_lines = []

    # Always keep Title/Abstract (before first header)
    first_header_line = headers[0][0]
    filtered_lines.extend(lines[0:first_header_line])

    for i in range(len(headers)):
        current_header_line = headers[i][0]
        next_header_line = headers[i + 1][0] if i + 1 < len(headers) else len(lines)

        section_key = section_map.get(i)
        is_target = section_key in target_sections

        should_keep = False
        if filter_type == "only":
            if is_target:
                should_keep = True
        elif filter_type == "except":
            if not is_target:
                should_keep = True

        if should_keep:
            filtered_lines.extend(lines[current_header_line:next_header_line])

    return "\n".join(filtered_lines)


def generate_combined_section_datasets(modality, split):
    """Generate datasets with combined sections."""
    input_dir = DATA_DIR / f"{BASE_PREFIX}_{modality}_binary_noreviews_v6_{split}"
    input_file = input_dir / "data.json"

    if not input_file.exists():
        print(f"Skipping missing source: {input_file}")
        return

    print(f"Processing {modality} {split}...")
    data = load_dataset(input_file)

    # Two outputs: only_core_technical and no_core_technical
    only_data = []
    except_data = []

    for entry in data:
        human_msg = next((msg for msg in entry["conversations"] if msg["from"] == "human"), None)
        if not human_msg:
            continue

        original_text = human_msg["value"]

        # Only core technical sections
        only_text = parse_and_ablate_combined(original_text, "only", CORE_TECHNICAL_SECTIONS)
        if only_text:
            new_entry = json.loads(json.dumps(entry))
            for msg in new_entry["conversations"]:
                if msg["from"] == "human":
                    msg["value"] = only_text
            only_data.append(new_entry)

        # Except core technical sections
        except_text = parse_and_ablate_combined(original_text, "except", CORE_TECHNICAL_SECTIONS)
        if except_text:
            new_entry = json.loads(json.dumps(entry))
            for msg in new_entry["conversations"]:
                if msg["from"] == "human":
                    msg["value"] = except_text
            except_data.append(new_entry)

    # Save datasets
    if only_data:
        output_name = f"{BASE_PREFIX}_{modality}_binary_noreviews_only_core_technical_v6_{split}"
        output_path = DATA_DIR / output_name / "data.json"
        print(f"  Saving only_core_technical: {len(only_data)} entries")
        save_dataset(only_data, output_path)

    if except_data:
        output_name = f"{BASE_PREFIX}_{modality}_binary_noreviews_no_core_technical_v6_{split}"
        output_path = DATA_DIR / output_name / "data.json"
        print(f"  Saving no_core_technical: {len(except_data)} entries")
        save_dataset(except_data, output_path)


def update_dataset_info():
    """Add new datasets to dataset_info.json."""
    info_path = DATA_DIR / "dataset_info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)

    for item in os.listdir(DATA_DIR):
        if not item.startswith(BASE_PREFIX):
            continue
        if not os.path.isdir(DATA_DIR / item):
            continue
        if item in info:
            continue

        print(f"Registering new dataset: {item}")
        info[item] = {
            "file_name": f"{item}/data.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system"
            }
        }

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


def main():
    print("=== Creating Combined Section Datasets ===")
    print("Core technical sections: related_work + methodology + experimental_results\n")

    # Only text modalities (as per original spec)
    for modality in ["clean"]:
        for split in ["train", "validation", "test"]:
            generate_combined_section_datasets(modality, split)

    print("\n--- Updating dataset_info.json ---")
    update_dataset_info()
    print("Done!")


if __name__ == "__main__":
    main()
