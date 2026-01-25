#!/usr/bin/env python3
"""
Create section ablation datasets (v2) with improved regex patterns.

The v1 patterns were too restrictive - they only matched specific section numbers
(e.g., "# 3 METHOD" but not "# 4 METHOD"). The v2 patterns match any section number.

Creates datasets for:
- methodology (only/no)
- discussion (only/no)
- intro_discussion (only/no)
"""

import json
import re
import os
from pathlib import Path

# Configuration
DATA_DIR = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/data")
BASE_PREFIX = "iclr_2020_2025_85_5_10_split6_original"

# IMPROVED Section patterns - match ANY section number with \d+
SECTIONS_V2 = {
    "introduction": re.compile(r"^#\s*(\d+\s*)?INTRODUCTION", re.MULTILINE | re.IGNORECASE),
    "related_work": re.compile(r"^#\s*(\d+\s*)?RELATED WORK", re.MULTILINE | re.IGNORECASE),
    "methodology": re.compile(r"^#\s*(\d+\s*)?(METHOD|METHODOLOGY|METHODS|PROPOSED METHOD|APPROACH)", re.MULTILINE | re.IGNORECASE),
    "experimental_results": re.compile(r"^#\s*(\d+\s*)?(EXPERIMENTS|RESULTS|EXPERIMENTAL RESULTS|EXPERIMENT)", re.MULTILINE | re.IGNORECASE),
    "discussion": re.compile(r"^#\s*(\d+\s*)?(DISCUSSION|LIMITATIONS|CONCLUSION)", re.MULTILINE | re.IGNORECASE),
    "conclusion": re.compile(r"^#\s*(\d+\s*)?CONCLUSION", re.MULTILINE | re.IGNORECASE),
}

KEYWORDS_V2 = {
    "introduction": ["INTRODUCTION"],
    "related_work": ["RELATED WORK", "BACKGROUND"],
    "methodology": ["METHOD", "METHODOLOGY", "METHODS", "APPROACH"],
    "experimental_results": ["RESULTS", "EXPERIMENTS", "EVALUATION", "EXPERIMENT"],
    "discussion": ["DISCUSSION", "LIMITATIONS", "CONCLUSION"],
    "conclusion": ["CONCLUSION"],
}


def load_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_dataset(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def parse_and_ablate(text, filter_type, section_key):
    """
    Parses the text to identify sections and performs ablation.

    filter_type: 'only' (keep only this section + title/abstract)
                 'except' (remove this section)
    section_key: key in SECTIONS_V2 dict, or 'intro_discussion' for combined
    """
    lines = text.split('\n')

    # Find all headers
    headers = []
    for i, line in enumerate(lines):
        if line.strip().startswith('# '):
            headers.append((i, line.strip()))

    if not headers:
        return None

    # Map headers to section types
    section_map = {}
    for i, (line_idx, line_content) in enumerate(headers):
        matched = None

        # First try regex patterns
        for key, pattern in SECTIONS_V2.items():
            if pattern.match(line_content):
                matched = key
                break

        # Fallback to keyword matching
        if not matched:
            upper_line = line_content.upper()
            for key, kws in KEYWORDS_V2.items():
                if any(kw in upper_line for kw in kws):
                    matched = key
                    break

        if matched:
            section_map[i] = matched

    # Handle combined intro_discussion
    if section_key == "intro_discussion":
        target_sections = ["introduction", "discussion"]
    else:
        target_sections = [section_key]

    # Check if any target section exists
    found_targets = [i for i, sec in section_map.items() if sec in target_sections]
    if not found_targets:
        return None

    # Build filtered content
    filtered_lines = []

    # Always keep Title/Abstract (before first header)
    first_header_line = headers[0][0]
    filtered_lines.extend(lines[0:first_header_line])

    for i in range(len(headers)):
        current_header_line = headers[i][0]
        next_header_line = headers[i + 1][0] if i + 1 < len(headers) else len(lines)

        section = section_map.get(i)
        is_target = section in target_sections

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


def generate_section_datasets_v2(modality, split):
    """Generate v2 section ablation datasets with improved regex patterns."""
    input_dir = DATA_DIR / f"{BASE_PREFIX}_{modality}_binary_noreviews_v6_{split}"
    input_file = input_dir / "data.json"

    if not input_file.exists():
        print(f"Skipping missing source: {input_file}")
        return

    print(f"Processing {modality} {split}...")
    data = load_dataset(input_file)

    # Define tasks - focusing on methodology and discussion which had issues
    tasks = [
        {"key": "methodology", "type": "only", "suffix": "only_methodology_v2"},
        {"key": "methodology", "type": "except", "suffix": "no_methodology_v2"},
        {"key": "discussion", "type": "only", "suffix": "only_discussion_v2"},
        {"key": "discussion", "type": "except", "suffix": "no_discussion_v2"},
        {"key": "intro_discussion", "type": "only", "suffix": "only_intro_discussion_v2"},
        {"key": "intro_discussion", "type": "except", "suffix": "no_intro_discussion_v2"},
    ]

    # Prepare containers
    new_datasets = {t["suffix"]: [] for t in tasks}

    # Track stats by year
    stats = {t["suffix"]: {2020: 0, 2021: 0, 2022: 0, 2023: 0, 2024: 0, 2025: 0} for t in tasks}
    total_by_year = {2020: 0, 2021: 0, 2022: 0, 2023: 0, 2024: 0, 2025: 0}

    for entry in data:
        meta = entry.get("_metadata", {})
        year = meta.get("year", 2020)
        total_by_year[year] = total_by_year.get(year, 0) + 1

        human_msg = next((msg for msg in entry["conversations"] if msg["from"] == "human"), None)
        if not human_msg:
            continue

        original_text = human_msg["value"]

        for task in tasks:
            ablated_text = parse_and_ablate(original_text, task["type"], task["key"])

            if ablated_text:
                new_entry = json.loads(json.dumps(entry))
                for msg in new_entry["conversations"]:
                    if msg["from"] == "human":
                        msg["value"] = ablated_text
                new_datasets[task["suffix"]].append(new_entry)
                stats[task["suffix"]][year] = stats[task["suffix"]].get(year, 0) + 1

    # Save and report stats
    for suffix, entries in new_datasets.items():
        if not entries:
            print(f"  WARNING: No entries for {suffix}")
            continue

        output_name = f"{BASE_PREFIX}_{modality}_binary_noreviews_{suffix}_v6_{split}"
        output_path = DATA_DIR / output_name / "data.json"
        print(f"  Saving {suffix}: {len(entries)} entries")
        save_dataset(entries, output_path)

        # Print year breakdown
        print(f"    By year: ", end="")
        for year in [2020, 2021, 2022, 2023, 2024, 2025]:
            total = total_by_year.get(year, 0)
            matched = stats[suffix].get(year, 0)
            pct = 100 * matched / total if total > 0 else 0
            print(f"{year}:{matched}/{total}({pct:.0f}%) ", end="")
        print()


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
    print("=== Creating Section Ablation Datasets (v2 - Improved Regex) ===\n")
    print("Improvements:")
    print("  - methodology: Now matches any section number (# 2/3/4/... METHOD)")
    print("  - discussion: Now matches any section number (# 5/6/7/... DISCUSSION)")
    print("  - discussion: Now includes CONCLUSION sections")
    print("  - Added METHODS (plural) and APPROACH patterns")
    print()

    # Generate for clean modality (text-only)
    for modality in ["clean"]:
        for split in ["train", "validation", "test"]:
            generate_section_datasets_v2(modality, split)

    print("\n--- Updating dataset_info.json ---")
    update_dataset_info()
    print("\nDone!")


if __name__ == "__main__":
    main()
