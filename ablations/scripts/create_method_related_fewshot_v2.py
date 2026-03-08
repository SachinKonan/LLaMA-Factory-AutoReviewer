#!/usr/bin/env python3
"""
Create fewshot datasets for methodology_v2 and related_work_v2 sections.

Uses the improved v2 regex patterns and creates fewshot variants with:
- 2 accept, 0 reject
- 1 accept, 1 reject
- 0 accept, 2 reject
- 0 accept, 3 reject
- 0 accept, 4 reject
"""

import json
import re
import os
import random
from pathlib import Path

# Configuration
DATA_DIR = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/data")
BASE_PREFIX = "iclr_2020_2025_85_5_10_split6_original"

# Seed for reproducibility
random.seed(42)

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
    "conclusion": ["CONCLUSION"], # not really used
}

# Fewshot mixtures: (n_accept, n_reject)
FEWSHOT_MIXTURES = [
    (2, 0),  # accept-heavy
    (1, 1),  # balanced
    (0, 2),  # reject-heavy
    (0, 3),  # more reject
    (0, 4),  # most reject
]


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
    section_key: key in SECTIONS_V2 dict
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


def generate_section_datasets_v2(modality, split, section_key, suffix):
    """Generate v2 section ablation datasets for a specific section."""
    input_dir = DATA_DIR / f"{BASE_PREFIX}_{modality}_binary_noreviews_v6_{split}"
    input_file = input_dir / "data.json"

    if not input_file.exists():
        print(f"Skipping missing source: {input_file}")
        return []

    print(f"Processing {modality} {split} for {section_key}...")
    data = load_dataset(input_file)

    new_data = []

    for entry in data:
        human_msg = next((msg for msg in entry["conversations"] if msg["from"] == "human"), None)
        if not human_msg:
            continue

        original_text = human_msg["value"]
        ablated_text = parse_and_ablate(original_text, "only", section_key)

        if ablated_text:
            new_entry = json.loads(json.dumps(entry))
            for msg in new_entry["conversations"]:
                if msg["from"] == "human":
                    msg["value"] = ablated_text
            new_data.append(new_entry)

    if new_data:
        output_name = f"{BASE_PREFIX}_{modality}_binary_noreviews_{suffix}_v6_{split}"
        output_path = DATA_DIR / output_name / "data.json"
        print(f"  Saving {suffix}: {len(new_data)} entries")
        save_dataset(new_data, output_path)

    return new_data


def select_fewshot_examples(train_data, n_accept, n_reject):
    """Select fixed fewshot examples from training data."""
    accepts = [d for d in train_data if d['_metadata'].get('answer') == 'Accept']
    rejects = [d for d in train_data if d['_metadata'].get('answer') == 'Reject']

    def get_prompt_length(entry):
        for msg in entry['conversations']:
            if msg['from'] == 'human':
                return len(msg['value'])
        return 0

    def filter_moderate_length(entries, min_len=3000, max_len=10000):
        filtered = [e for e in entries if min_len <= get_prompt_length(e) <= max_len]
        return filtered if filtered else entries

    # Group by year for diversity
    accepts_by_year = {}
    for d in accepts:
        year = d['_metadata'].get('year', 2020)
        if year not in accepts_by_year:
            accepts_by_year[year] = []
        accepts_by_year[year].append(d)

    rejects_by_year = {}
    for d in rejects:
        year = d['_metadata'].get('year', 2020)
        if year not in rejects_by_year:
            rejects_by_year[year] = []
        rejects_by_year[year].append(d)

    selected_accepts = []
    selected_rejects = []

    # Select accepts
    years = sorted(accepts_by_year.keys())
    for year in years:
        if len(selected_accepts) >= n_accept:
            break
        candidates = filter_moderate_length(accepts_by_year[year])
        if candidates:
            selected_accepts.append(random.choice(candidates))

    while len(selected_accepts) < n_accept:
        all_accepts = filter_moderate_length(accepts)
        remaining = [a for a in all_accepts if a not in selected_accepts]
        if remaining:
            selected_accepts.append(random.choice(remaining))
        else:
            break

    # Select rejects
    years = sorted(rejects_by_year.keys())
    for year in years:
        if len(selected_rejects) >= n_reject:
            break
        candidates = filter_moderate_length(rejects_by_year[year])
        if candidates:
            selected_rejects.append(random.choice(candidates))

    while len(selected_rejects) < n_reject:
        all_rejects = filter_moderate_length(rejects)
        remaining = [r for r in all_rejects if r not in selected_rejects]
        if remaining:
            selected_rejects.append(random.choice(remaining))
        else:
            break

    return selected_accepts, selected_rejects


def format_fewshot_example(entry):
    """Format a single example for fewshot context."""
    human_msg = ""
    gpt_msg = ""

    for msg in entry['conversations']:
        if msg['from'] == 'human':
            human_msg = msg['value']
        elif msg['from'] == 'gpt':
            gpt_msg = msg['value']

    year = entry['_metadata'].get('year', 'Unknown')
    decision = entry['_metadata'].get('answer', 'Unknown')

    example = f"""
---
**Example Paper (ICLR {year}, {decision})**

{human_msg}

**Reviewer Decision:** {gpt_msg}
---
"""
    return example


def create_fewshot_prompt_prefix(accept_examples, reject_examples):
    """Create the fewshot prompt prefix from selected examples."""
    prefix = """Here are some example paper reviews to help guide your evaluation:

"""
    all_examples = []
    max_len = max(len(accept_examples), len(reject_examples)) if accept_examples or reject_examples else 0
    for i in range(max_len):
        if i < len(accept_examples):
            all_examples.append(accept_examples[i])
        if i < len(reject_examples):
            all_examples.append(reject_examples[i])

    for i, ex in enumerate(all_examples, 1):
        prefix += f"**Example {i}:**\n"
        prefix += format_fewshot_example(ex)
        prefix += "\n"

    prefix += """
Now, please evaluate the following paper:

"""
    return prefix


def create_fewshot_dataset(base_dataset_path, output_name, fewshot_prefix):
    """Create a fewshot dataset by prepending examples to each prompt."""
    data = load_dataset(base_dataset_path)

    new_data = []
    for entry in data:
        new_entry = json.loads(json.dumps(entry))
        for msg in new_entry['conversations']:
            if msg['from'] == 'human':
                msg['value'] = fewshot_prefix + msg['value']
        new_data.append(new_entry)

    output_path = DATA_DIR / output_name / "data.json"
    print(f"  Saving {output_name}: {len(new_data)} entries")
    save_dataset(new_data, output_path)
    return output_name


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


def create_fewshot_for_section(section_key, suffix):
    """Create fewshot datasets for a specific section."""
    print(f"\n{'='*60}")
    print(f"Creating Fewshot Datasets for {suffix} (v2)")
    print(f"{'='*60}")

    # Step 1: Generate base section datasets if needed
    print("\n--- Step 1: Creating Base Section Datasets ---")
    for modality in ["clean"]:
        for split in ["train", "validation", "test"]:
            output_name = f"{BASE_PREFIX}_{modality}_binary_noreviews_{suffix}_v6_{split}"
            output_path = DATA_DIR / output_name / "data.json"
            if not output_path.exists():
                generate_section_datasets_v2(modality, split, section_key, suffix)
            else:
                print(f"  {suffix} {split} already exists, skipping...")

    # Step 2: Create fewshot variants
    print("\n--- Step 2: Creating Fewshot Datasets ---")

    train_path = DATA_DIR / f"{BASE_PREFIX}_clean_binary_noreviews_{suffix}_v6_train" / "data.json"
    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        return

    train_data = load_dataset(train_path)
    print(f"Loaded {len(train_data)} training examples for fewshot selection\n")

    for n_accept, n_reject in FEWSHOT_MIXTURES:
        print(f"\n--- Creating fewshot_{n_accept}acc_{n_reject}rej ---")

        # Reset random seed for each mixture to ensure reproducibility
        random.seed(42 + n_accept * 10 + n_reject)

        accept_examples, reject_examples = select_fewshot_examples(train_data, n_accept, n_reject)
        print(f"  Selected {len(accept_examples)} accept, {len(reject_examples)} reject examples")

        for ex in accept_examples:
            year = ex['_metadata'].get('year')
            print(f"    Accept ({year}): {ex['_metadata'].get('submission_id')}")
        for ex in reject_examples:
            year = ex['_metadata'].get('year')
            print(f"    Reject ({year}): {ex['_metadata'].get('submission_id')}")

        fewshot_prefix = create_fewshot_prompt_prefix(accept_examples, reject_examples)

        test_base = DATA_DIR / f"{BASE_PREFIX}_clean_binary_noreviews_{suffix}_v6_test" / "data.json"
        output_name = f"{BASE_PREFIX}_clean_binary_noreviews_{suffix}_fewshot_{n_accept}acc_{n_reject}rej_v6_test"
        create_fewshot_dataset(test_base, output_name, fewshot_prefix)


def main():
    print("=" * 60)
    print("Creating Fewshot Datasets for Methodology v2 and Related Work v2")
    print("=" * 60)
    print("\nFewshot mixtures:")
    for n_accept, n_reject in FEWSHOT_MIXTURES:
        print(f"  - {n_accept} accept, {n_reject} reject")

    # Create fewshot datasets for methodology v2
    create_fewshot_for_section("methodology", "only_methodology_v2")

    # Create fewshot datasets for related work v2
    create_fewshot_for_section("related_work", "only_related_work_v2")

    print("\n--- Updating dataset_info.json ---")
    update_dataset_info()
    print("\nDone!")


if __name__ == "__main__":
    main()
