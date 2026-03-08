#!/usr/bin/env python3
"""
Create fewshot datasets with fixed examples.

Uses the core_technical (related_work + methodology + experimental_results) dataset
and adds fewshot examples with different accept/reject mixtures:
- 2 accept, 2 reject (balanced)
- 3 accept, 1 reject (accept-heavy)
- 1 accept, 3 reject (reject-heavy)

The fewshot examples are drawn from the training set and are FIXED across all test samples.
"""

import json
import os
import random
from pathlib import Path

# Configuration
DATA_DIR = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/data")
BASE_PREFIX = "iclr_2020_2025_85_5_10_split6_original"

# Seed for reproducibility
random.seed(42)

# Fewshot mixtures to create: (n_accept, n_reject)
FEWSHOT_MIXTURES = [
    (2, 2),  # balanced
    (3, 1),  # accept-heavy
    (1, 3),  # reject-heavy
]


def load_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_dataset(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def select_fewshot_examples(train_data, n_accept, n_reject):
    """
    Select fixed fewshot examples from training data.

    Returns examples with diverse years and moderate length.
    """
    accepts = [d for d in train_data if d['_metadata'].get('answer') == 'Accept']
    rejects = [d for d in train_data if d['_metadata'].get('answer') == 'Reject']

    # Sort by year to get diversity, then sample
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

    # Select examples trying to get diverse years
    # Filter for moderate length (not too short, not too long)
    def get_prompt_length(entry):
        for msg in entry['conversations']:
            if msg['from'] == 'human':
                return len(msg['value'])
        return 0

    def filter_moderate_length(entries, min_len=5000, max_len=15000):
        """Filter to entries with moderate prompt length."""
        filtered = [e for e in entries if min_len <= get_prompt_length(e) <= max_len]
        return filtered if filtered else entries  # Fallback to all if none match

    selected_accepts = []
    selected_rejects = []

    # Try to get one from each year for diversity
    years = sorted(accepts_by_year.keys())
    for year in years:
        if len(selected_accepts) >= n_accept:
            break
        candidates = filter_moderate_length(accepts_by_year[year])
        if candidates:
            selected_accepts.append(random.choice(candidates))

    # Fill remaining if needed
    while len(selected_accepts) < n_accept:
        all_accepts = filter_moderate_length(accepts)
        remaining = [a for a in all_accepts if a not in selected_accepts]
        if remaining:
            selected_accepts.append(random.choice(remaining))
        else:
            break

    # Same for rejects
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
            # Extract just the paper content (after the instruction)
            human_msg = msg['value']
        elif msg['from'] == 'gpt':
            gpt_msg = msg['value']

    # Extract paper title from the content
    lines = human_msg.split('\n')
    title = "Unknown Paper"
    for line in lines:
        if line.startswith('# ') and not any(x in line.upper() for x in ['ABSTRACT', 'INTRODUCTION', 'METHOD']):
            title = line.strip('# ').strip()
            break

    year = entry['_metadata'].get('year', 'Unknown')
    decision = entry['_metadata'].get('answer', 'Unknown')

    # Format as a condensed example
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
    # Interleave accepts and rejects for balance
    all_examples = []
    max_len = max(len(accept_examples), len(reject_examples))
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
        new_entry = json.loads(json.dumps(entry))  # Deep copy

        for msg in new_entry['conversations']:
            if msg['from'] == 'human':
                # Prepend fewshot examples to the prompt
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


def main():
    print("=== Creating Fewshot Datasets ===")
    print("Base dataset: only_core_technical (related_work + methodology + experimental_results)\n")

    # Load training data for selecting fewshot examples
    train_path = DATA_DIR / f"{BASE_PREFIX}_clean_binary_noreviews_only_core_technical_v6_train" / "data.json"
    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        print("Please run create_combined_section_datasets.py first.")
        return

    train_data = load_dataset(train_path)
    print(f"Loaded {len(train_data)} training examples for fewshot selection\n")

    # Process each mixture
    for n_accept, n_reject in FEWSHOT_MIXTURES:
        print(f"\n--- Creating fewshot_{n_accept}acc_{n_reject}rej ---")

        # Select fixed examples
        accept_examples, reject_examples = select_fewshot_examples(train_data, n_accept, n_reject)
        print(f"  Selected {len(accept_examples)} accept, {len(reject_examples)} reject examples")

        # Show selected example titles
        for ex in accept_examples:
            year = ex['_metadata'].get('year')
            print(f"    Accept ({year}): {ex['_metadata'].get('submission_id')}")
        for ex in reject_examples:
            year = ex['_metadata'].get('year')
            print(f"    Reject ({year}): {ex['_metadata'].get('submission_id')}")

        # Create fewshot prefix
        fewshot_prefix = create_fewshot_prompt_prefix(accept_examples, reject_examples)

        # Create test dataset with fewshot
        test_base = DATA_DIR / f"{BASE_PREFIX}_clean_binary_noreviews_only_core_technical_v6_test" / "data.json"
        output_name = f"{BASE_PREFIX}_clean_binary_noreviews_only_core_technical_fewshot_{n_accept}acc_{n_reject}rej_v6_test"
        create_fewshot_dataset(test_base, output_name, fewshot_prefix)

    print("\n--- Updating dataset_info.json ---")
    update_dataset_info()
    print("\nDone!")


if __name__ == "__main__":
    main()
