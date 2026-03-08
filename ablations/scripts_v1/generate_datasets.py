#!/usr/bin/env python3
"""
Generate ablation datasets for v1 experiments.

Creates 45 datasets total:
1. Critical/Less Critical Modifiers (12 datasets):
   - 2 modifiers (critical, less_critical) x 2 output formats (boxed, json) x 3 modalities
   - Uses same random 500 subset across all variants

2. Fewshot Full Paper (9 datasets):
   - 3 variations (2acc_0rej, 1acc_1rej, 0acc_2rej) x 3 modalities (clean, clean_images, vision)

3. Fewshot Paper Parts (24 datasets):
   - 4 parts (intro, related, method, results) x 3 variations x 2 modalities (no vision)
   - Each part includes title + abstract + the specific section

Usage:
    python generate_datasets.py
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configuration
DATA_DIR = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/data")
BASE_DATA_DIR = Path("/n/fs/vision-mix/sk7524/LLaMA-Factory/data")
BASE_PREFIX = "iclr_2020_2025_85_5_10_split7_balanced"
OUTPUT_PREFIX = "iclr_2020_2025_85_5_10_split7_ablation_v1"

# Seed for reproducibility
random.seed(42)

# ============================================================================
# Prompt Templates
# ============================================================================

# Base system prompt
SYSTEM_PROMPT_BASE = "You are an expert academic reviewer tasked with evaluating research papers."

# Task instruction (goes in system prompt)
TASK_INSTRUCTION = """
I am giving you a paper. I want to predict its acceptance outcome at ICLR.
Note: ICLR generally has a ~30% acceptance rate."""

# Output format instructions (go in system prompt)
BOXED_FORMAT_INSTRUCTION = """
Your answer must start with: \\boxed{Accept} or \\boxed{Reject}"""

JSON_FORMAT_INSTRUCTION = """
Respond with a reasoning trace starting with a strictly formatted JSON block.

Provide the review in a valid JSON object inside a markdown code block.

Use the following JSON schema:
```json
{
  "summary": "string",
  "strengths": "string",
  "weaknesses": "string",
  "score": integer,  // Score 1-10
  "decision": "Accept" OR "Reject"
}
```"""

# Critical modifier instructions (go in system prompt)
CRITICAL_MODIFIER = """
IMPORTANT: Be critical of claims and reject papers that lack substance. Look for methodological flaws, unsupported claims, and insufficient experimental validation. When in doubt, reject."""

LESS_CRITICAL_MODIFIER = """
IMPORTANT: Be less critical of claims and accept papers that are on the border. Focus on the potential contribution and novelty. Give papers the benefit of the doubt when the core idea is sound."""


def build_system_prompt(
    modifier: Optional[str] = None,
    output_format: str = "boxed",
) -> str:
    """Build a complete system prompt with all instructions.

    Args:
        modifier: None, "critical", or "less_critical"
        output_format: "boxed" or "json"

    Returns:
        Complete system prompt string
    """
    parts = [SYSTEM_PROMPT_BASE, TASK_INSTRUCTION]

    # Add modifier if specified
    if modifier == "critical":
        parts.append(CRITICAL_MODIFIER)
    elif modifier == "less_critical":
        parts.append(LESS_CRITICAL_MODIFIER)

    # Add output format instruction
    if output_format == "boxed":
        parts.append(BOXED_FORMAT_INSTRUCTION)
    else:
        parts.append(JSON_FORMAT_INSTRUCTION)

    return "\n".join(parts)


def set_system_prompt(conversations: List[Dict], system_prompt: str) -> List[Dict]:
    """Set or update the system prompt in a conversations list.

    Returns a new list with the system prompt set.
    """
    # Remove any existing system message
    new_convs = [msg for msg in conversations if msg.get("from") != "system"]

    # Add system prompt at the beginning
    new_convs.insert(0, {"from": "system", "value": system_prompt})

    return new_convs

# Section patterns for extraction
SECTIONS = {
    "abstract": re.compile(r'^a\s*b\s*s\s*t\s*r\s*a\s*c\s*t$', re.IGNORECASE),
    "introduction": re.compile(r"^#\s*(\d+\s*)?INTRODUCTION", re.MULTILINE | re.IGNORECASE),
    "related_work": re.compile(r"^#\s*(\d+\s*)?(RELATED WORK|BACKGROUND)", re.MULTILINE | re.IGNORECASE),
    "methodology": re.compile(r"^#\s*(\d+\s*)?(METHOD|METHODOLOGY|METHODS|PROPOSED METHOD|APPROACH)", re.MULTILINE | re.IGNORECASE),
    "experimental_results": re.compile(r"^#\s*(\d+\s*)?(EXPERIMENTS|RESULTS|EXPERIMENTAL RESULTS|EXPERIMENT|EVALUATION)", re.MULTILINE | re.IGNORECASE),
}

SECTION_KEYWORDS = {
    "abstract": ["ABSTRACT"],
    "introduction": ["INTRODUCTION"],
    "related_work": ["RELATED WORK", "BACKGROUND"],
    "methodology": ["METHOD", "METHODOLOGY", "METHODS", "APPROACH"],
    "experimental_results": ["RESULTS", "EXPERIMENTS", "EVALUATION", "EXPERIMENT"],
}


# ============================================================================
# Utility Functions
# ============================================================================

def load_dataset(path: Path) -> List[Dict]:
    """Load dataset from data.json file."""
    with open(path / "data.json", 'r') as f:
        return json.load(f)


def save_dataset(data: List[Dict], output_name: str):
    """Save dataset to data.json file."""
    output_path = DATA_DIR / output_name
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / "data.json", 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {output_name}: {len(data)} entries")


def get_base_dataset_path(modality: str, split: str) -> Path:
    """Get the base dataset path for a given modality and split."""
    if modality == "clean":
        name = f"{BASE_PREFIX}_clean_binary_noreviews_v7_{split}"
    elif modality == "clean_images":
        name = f"{BASE_PREFIX}_clean_images_binary_noreviews_v7_{split}"
    elif modality == "vision":
        name = f"{BASE_PREFIX}_vision_binary_noreviews_v7_{split}"
    else:
        raise ValueError(f"Unknown modality: {modality}")
    return BASE_DATA_DIR / name


def extract_paper_content(user_message: str) -> str:
    """Extract paper content from user message (remove prompt prefix)."""
    prefix_end = " - Note: ICLR generally has a ~30% acceptance rate\n\n"
    if prefix_end in user_message:
        idx = user_message.find(prefix_end)
        return user_message[idx + len(prefix_end):]
    return user_message


def count_image_tags(text: str) -> int:
    """Count number of <image> tags in text."""
    return len(re.findall(r'<image>', text))


def get_image_tag_positions(text: str) -> List[int]:
    """Get positions of all <image> tags in text."""
    return [m.start() for m in re.finditer(r'<image>', text)]


def clean_image_tags(text: str) -> str:
    """Remove all <image> tags and trailing whitespace from text."""
    return re.sub(r'<image>\s*', '', text)


# ============================================================================
# Shared Sample Selection
# ============================================================================

def get_sample_ids(sample_size: int = 500) -> List[str]:
    """Select sample IDs to use across all ablation datasets.

    Uses clean_images test dataset as the reference.
    Returns list of submission_ids.
    """
    base_path = get_base_dataset_path("clean_images", "test")
    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found at {base_path}")

    base_data = load_dataset(base_path)
    total_samples = len(base_data)

    if sample_size > total_samples:
        print(f"Warning: sample_size ({sample_size}) > total ({total_samples}), using all")
        sample_indices = list(range(total_samples))
    else:
        sample_indices = sorted(random.sample(range(total_samples), sample_size))

    sample_ids = [base_data[i]["_metadata"]["submission_id"] for i in sample_indices]
    print(f"Selected {len(sample_ids)} sample IDs (shared across all datasets)")
    return sample_ids


# ============================================================================
# Critical/Less Critical Dataset Generation (12 datasets)
# ============================================================================

def create_critical_datasets(sample_ids: List[str]):
    """Create critical/less critical modifier datasets.

    Creates 12 datasets:
    - 2 modifiers (critical, less_critical)
    - 2 output formats (boxed, json)
    - 3 modalities (clean, clean_images, vision)

    Uses the provided sample_ids subset.
    """
    print("\n=== Creating Critical/Less Critical Datasets ===")

    modalities = ["clean", "clean_images", "vision"]
    modifier_names = ["critical", "less_critical"]
    output_formats = ["boxed", "json"]

    sample_size = len(sample_ids)
    created_datasets = []

    for modality in modalities:
        base_path = get_base_dataset_path(modality, "test")
        if not base_path.exists():
            print(f"Warning: Skipping {modality}, base dataset not found")
            continue

        data = load_dataset(base_path)

        # Subset to selected indices
        subset_data = [data[i] for i in range(len(data)) if data[i]["_metadata"]["submission_id"] in sample_ids]
        assert len(subset_data) == sample_size

        for modifier_name in modifier_names:
            for output_format in output_formats:
                output_name = f"{OUTPUT_PREFIX}_{modifier_name}_{output_format}_{modality}_test"

                # Build system prompt with modifier and output format
                system_prompt = build_system_prompt(
                    modifier=modifier_name,
                    output_format=output_format
                )

                new_data = []
                for entry in subset_data:
                    new_entry = json.loads(json.dumps(entry))  # Deep copy

                    # Update user message to just contain paper content
                    for msg in new_entry["conversations"]:
                        if msg["from"] == "human":
                            paper_content = extract_paper_content(msg["value"])

                            if modality == "clean":
                                paper_content = clean_image_tags(paper_content)

                            msg["value"] = paper_content

                    # Set system prompt
                    new_entry["conversations"] = set_system_prompt(
                        new_entry["conversations"], system_prompt
                    )

                    new_data.append(new_entry)

                save_dataset(new_data, output_name)
                created_datasets.append(output_name)

    return created_datasets


# ============================================================================
# Fewshot Full Paper Dataset Generation (9 datasets)
# ============================================================================

def select_fewshot_examples(train_data: List[Dict], n_accept: int, n_reject: int) -> Tuple[List[Dict], List[Dict]]:
    """Select fewshot examples from training data with year diversity."""
    accepts = [d for d in train_data if d.get('_metadata', {}).get('answer') == 'Accept']
    rejects = [d for d in train_data if d.get('_metadata', {}).get('answer') == 'Reject']

    # Sort by year for diversity
    accepts_by_year = {}
    for d in accepts:
        year = d.get('_metadata', {}).get('year', 2020)
        accepts_by_year.setdefault(year, []).append(d)

    rejects_by_year = {}
    for d in rejects:
        year = d.get('_metadata', {}).get('year', 2020)
        rejects_by_year.setdefault(year, []).append(d)

    def get_prompt_length(entry):
        for msg in entry['conversations']:
            if msg['from'] == 'human':
                return len(msg['value'])
        return 0

    def filter_moderate_length(entries, min_len=5000, max_len=20000):
        filtered = [e for e in entries if min_len <= get_prompt_length(e) <= max_len]
        return filtered if filtered else entries

    selected_accepts = []
    selected_rejects = []

    # Select accepts with year diversity
    for year in sorted(accepts_by_year.keys()):
        if len(selected_accepts) >= n_accept:
            break
        candidates = filter_moderate_length(accepts_by_year[year])
        if candidates:
            selected_accepts.append(random.choice(candidates))

    while len(selected_accepts) < n_accept:
        remaining = [a for a in filter_moderate_length(accepts) if a not in selected_accepts]
        if remaining:
            selected_accepts.append(random.choice(remaining))
        else:
            break

    # Select rejects with year diversity
    for year in sorted(rejects_by_year.keys()):
        if len(selected_rejects) >= n_reject:
            break
        candidates = filter_moderate_length(rejects_by_year[year])
        if candidates:
            selected_rejects.append(random.choice(candidates))

    while len(selected_rejects) < n_reject:
        remaining = [r for r in filter_moderate_length(rejects) if r not in selected_rejects]
        if remaining:
            selected_rejects.append(random.choice(remaining))
        else:
            break

    return selected_accepts, selected_rejects


def format_fewshot_example(entry: Dict, include_images: bool = False) -> str:
    """Format a single entry as a fewshot example."""
    human_msg = ""
    gpt_msg = ""

    for msg in entry['conversations']:
        if msg['from'] == 'human':
            human_msg = extract_paper_content(msg['value'])
        elif msg['from'] == 'gpt':
            gpt_msg = msg['value']

    year = entry.get('_metadata', {}).get('year', 'Unknown')
    decision = entry.get('_metadata', {}).get('answer', 'Unknown')

    # For vision/clean_images, we might want to indicate where images were
    if not include_images:
        human_msg = clean_image_tags(human_msg)

    return f"""
---
**Example Paper (ICLR {year}, {decision})**

{human_msg}

**Reviewer Decision:** {gpt_msg}
---
"""


def create_fewshot_prefix(accept_examples: List[Dict], reject_examples: List[Dict],
                          include_images: bool = False) -> str:
    """Create fewshot prompt prefix from examples."""
    prefix = "Here are example paper reviews to help guide your evaluation:\n\n"

    # Interleave accepts and rejects
    all_examples = []
    max_len = max(len(accept_examples), len(reject_examples))
    for i in range(max_len):
        if i < len(accept_examples):
            all_examples.append(accept_examples[i])
        if i < len(reject_examples):
            all_examples.append(reject_examples[i])

    for i, ex in enumerate(all_examples, 1):
        prefix += f"**Example {i}:**\n"
        prefix += format_fewshot_example(ex, include_images)
        prefix += "\n"

    prefix += "\nNow, please evaluate the following paper:\n\n"
    return prefix


def create_fewshot_full_paper_datasets(sample_ids: List[str]):
    """Create fewshot datasets with full paper content.

    Creates 9 datasets:
    - 3 variations (2acc_0rej, 1acc_1rej, 0acc_2rej)
    - 3 modalities (clean, clean_images, vision)

    Uses the provided sample_ids subset for test data.
    """
    print("\n=== Creating Fewshot Full Paper Datasets ===")

    modalities = ["clean", "clean_images", "vision"]
    fewshot_mixtures = [
        (2, 0, "2acc_0rej"),
        (1, 1, "1acc_1rej"),
        (0, 2, "0acc_2rej"),
    ]

    # Build system prompt (boxed format for fewshot)
    system_prompt = build_system_prompt(modifier=None, output_format="boxed")

    sample_ids_set = set(sample_ids)
    created_datasets = []

    for modality in modalities:
        print(f"\nProcessing modality: {modality}")

        # Load train and test data
        train_path = get_base_dataset_path(modality, "train")
        test_path = get_base_dataset_path(modality, "test")

        if not train_path.exists() or not test_path.exists():
            print(f"Warning: Skipping {modality}, datasets not found")
            continue

        train_data = load_dataset(train_path)
        full_test_data = load_dataset(test_path)

        # Filter test data to sample_ids
        test_data = [d for d in full_test_data if d["_metadata"]["submission_id"] in sample_ids_set]
        print(f"  Filtered test data: {len(test_data)}/{len(full_test_data)} entries")

        include_images = modality in ["clean_images", "vision"]

        for n_accept, n_reject, mixture_name in fewshot_mixtures:
            print(f"  Creating fewshot_{mixture_name}...")

            # Select examples
            accept_examples, reject_examples = select_fewshot_examples(
                train_data, n_accept, n_reject
            )
            print(f"    Selected {len(accept_examples)} accept, {len(reject_examples)} reject")

            # Create fewshot prefix (examples go in user message)
            fewshot_prefix = create_fewshot_prefix(accept_examples, reject_examples, include_images)

            # Apply to test data
            new_data = []
            for entry in test_data:
                new_entry = json.loads(json.dumps(entry))

                # Update user message: fewshot prefix + paper content
                for msg in new_entry["conversations"]:
                    if msg["from"] == "human":
                        paper_content = extract_paper_content(msg["value"])
                        if modality == "clean":
                            paper_content = clean_image_tags(paper_content)
                        msg["value"] = fewshot_prefix + paper_content

                # Set system prompt
                new_entry["conversations"] = set_system_prompt(
                    new_entry["conversations"], system_prompt
                )

                # For clean_images and vision, prepend fewshot example images
                if include_images and "images" in new_entry:
                    fewshot_images = []
                    for ex in accept_examples + reject_examples:
                        if "images" in ex:
                            fewshot_images.extend(ex["images"])
                    new_entry["images"] = fewshot_images + new_entry.get("images", [])

                new_data.append(new_entry)

            output_name = f"{OUTPUT_PREFIX}_fewshot_fullpaper_{mixture_name}_{modality}_test"
            save_dataset(new_data, output_name)
            created_datasets.append(output_name)

    return created_datasets


# ============================================================================
# Section Extraction for Fewshot Paper Parts
# ============================================================================

def find_all_headers(text: str) -> List[Tuple[int, str, int]]:
    """Find all headers in text with line indices and positions."""
    headers = []
    lines = text.split('\n')
    char_pos = 0

    for i, line in enumerate(lines):
        if line.strip().startswith('# '):
            headers.append((i, line.strip(), char_pos))
        char_pos += len(line) + 1  # +1 for newline

    return headers


def identify_section(header_line: str) -> Optional[str]:
    """Identify which section type a header belongs to."""
    for section_key, pattern in SECTIONS.items():
        if pattern.match(header_line):
            return section_key

    # Fallback to keyword matching
    upper_line = header_line.upper()
    for section_key, keywords in SECTION_KEYWORDS.items():
        if any(kw in upper_line for kw in keywords):
            return section_key

    return None


def extract_title_abstract(text: str) -> Tuple[str, int]:
    """Extract title and abstract using extract_section_content.

    Every paper has an abstract. Title is everything before the abstract header.

    Returns the content (title + abstract) and the character position where it ends.
    """
    # Extract abstract section
    abstract_content, abstract_start, abstract_end = extract_section_content(text, "abstract")

    if abstract_content is not None and abstract_start >= 0:
        # Title is everything before the abstract header
        title = text[:abstract_start].rstrip()
        # Combine title and abstract
        title_abstract = title + "\n\n" + abstract_content
        return title_abstract, abstract_end

    # Fallback: if no abstract found, use old logic
    lines = text.split('\n')

    # Find first section header (not title)
    first_section_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith('# ') and i > 0:
            # Check if it looks like a section (has number or known keyword)
            upper = line.upper()
            if any(kw in upper for section_keywords in SECTION_KEYWORDS.values() for kw in section_keywords):
                first_section_line = i
                break
            if re.match(r'^#\s*\d+', line.strip()):
                first_section_line = i
                break

    if first_section_line is None:
        # No sections found, take first 50 lines as title/abstract
        first_section_line = min(50, len(lines))

    title_abstract = '\n'.join(lines[:first_section_line])
    end_pos = sum(len(line) + 1 for line in lines[:first_section_line])

    return title_abstract, end_pos


def extract_section_content(text: str, section_key: str) -> Tuple[Optional[str], int, int]:
    """Extract a specific section's content.

    Returns (content, start_char_pos, end_char_pos) or (None, -1, -1) if not found.
    """
    lines = text.split('\n')
    headers = find_all_headers(text)

    # Find the header for this section
    section_start_idx = None
    section_start_pos = None

    for line_idx, header_line, char_pos in headers:
        if identify_section(header_line) == section_key:
            section_start_idx = line_idx
            section_start_pos = char_pos
            break

    if section_start_idx is None:
        return None, -1, -1

    # Find end of section (next header or end of text)
    section_end_idx = len(lines)
    for line_idx, _, _ in headers:
        if line_idx > section_start_idx:
            section_end_idx = line_idx
            break

    section_content = '\n'.join(lines[section_start_idx:section_end_idx])
    end_pos = sum(len(line) + 1 for line in lines[:section_end_idx])

    return section_content, section_start_pos, end_pos


def create_part_content(text: str, section_key: str) -> Tuple[Optional[str], List[int]]:
    """Create content for a paper part (title+abstract+section).

    Returns (new_content, list of kept image indices) or (None, []) if section not found.
    """
    title_abstract, ta_end_pos = extract_title_abstract(text)
    section_content, sec_start, sec_end = extract_section_content(text, section_key)

    if section_content is None:
        return None, []

    new_content = title_abstract + "\n\n" + section_content

    # Figure out which images to keep
    # Count <image> tags in kept regions
    ta_image_positions = get_image_tag_positions(title_abstract)
    sec_image_positions = get_image_tag_positions(section_content)

    # Map back to original positions
    all_original_positions = get_image_tag_positions(text)
    kept_image_indices = []

    for i, pos in enumerate(all_original_positions):
        # Check if this image is in title/abstract
        if pos < ta_end_pos:
            kept_image_indices.append(i)
        # Check if this image is in the section
        elif sec_start <= pos < sec_end:
            kept_image_indices.append(i)

    return new_content, kept_image_indices


def create_fewshot_paper_parts_datasets(sample_ids: List[str]):
    """Create fewshot datasets with part-based examples but full paper test data.

    Creates 24 datasets:
    - 4 parts (intro, related, method, results) - fewshot examples use title+abstract+section
    - 3 variations (2acc_0rej, 1acc_1rej, 0acc_2rej)
    - 2 modalities (clean, clean_images) - no vision
    - Test data uses the whole paper content
    - Same fewshot examples (by submission_id) used across modalities
    - Uses the provided sample_ids subset for test data.
    """
    print("\n=== Creating Fewshot Paper Parts Datasets ===")

    sample_ids_set = set(sample_ids)
    modalities = ["clean", "clean_images"]
    parts = [
        ("intro", "introduction"),
        ("related", "related_work"),
        ("method", "methodology"),
        ("results", "experimental_results"),
    ]
    fewshot_mixtures = [
        (2, 0, "2acc_0rej"),
        (1, 1, "1acc_1rej"),
        (0, 2, "0acc_2rej"),
    ]

    created_datasets = []

    # Load clean_images train data first to select fewshot example IDs
    # (clean_images IDs are a superset of clean IDs)
    reference_train_path = get_base_dataset_path("clean_images", "train")
    if not reference_train_path.exists():
        print(f"Error: Reference train dataset not found at {reference_train_path}")
        return created_datasets

    reference_train_data = load_dataset(reference_train_path)
    print(f"Loaded reference train data from clean_images: {len(reference_train_data)} entries")

    # Helper to check if an example has a valid section
    def example_has_section(ex: Dict, section_key: str) -> bool:
        human_msg = [m["value"] for m in ex["conversations"] if m["from"] == "human"][0]
        paper_content = extract_paper_content(human_msg)
        part_content, _ = create_part_content(paper_content, section_key)
        return part_content is not None

    # Pre-select fewshot example IDs for each (part, mixture) combination
    # Key: (part_short, mixture_name) -> (accept_ids, reject_ids)
    fewshot_ids_cache: Dict[Tuple[str, str], Tuple[List[str], List[str]]] = {}

    for part_short, part_section in parts:
        for n_accept, n_reject, mixture_name in fewshot_mixtures:
            # Collect valid examples across multiple random tries
            max_retries = 50
            collected_accept_ids = set()
            collected_reject_ids = set()

            for attempt in range(max_retries):
                # Check if we have enough
                if len(collected_accept_ids) >= n_accept and len(collected_reject_ids) >= n_reject:
                    break

                accept_examples, reject_examples = select_fewshot_examples(
                    reference_train_data, n_accept, n_reject
                )

                # Filter to examples that have the required section and add to collection
                for ex in accept_examples:
                    if len(collected_accept_ids) < n_accept and example_has_section(ex, part_section):
                        collected_accept_ids.add(ex["_metadata"]["submission_id"])

                for ex in reject_examples:
                    if len(collected_reject_ids) < n_reject and example_has_section(ex, part_section):
                        collected_reject_ids.add(ex["_metadata"]["submission_id"])

                if attempt > 0 and attempt % 10 == 0:
                    print(f"    Attempt {attempt}: {part_short}/{mixture_name} - collected {len(collected_accept_ids)}/{n_accept} accepts, {len(collected_reject_ids)}/{n_reject} rejects with valid {part_section} section")

            # Convert to lists
            best_accept_ids = list(collected_accept_ids)[:n_accept]
            best_reject_ids = list(collected_reject_ids)[:n_reject]

            fewshot_ids_cache[(part_short, mixture_name)] = (best_accept_ids, best_reject_ids)

            if len(best_accept_ids) < n_accept or len(best_reject_ids) < n_reject:
                print(f"  Warning: {part_short}/{mixture_name} - only found {len(best_accept_ids)}/{n_accept} accepts, {len(best_reject_ids)}/{n_reject} rejects after {max_retries} attempts")
            else:
                print(f"  Selected fewshot IDs for {part_short}/{mixture_name}: {len(best_accept_ids)} accept, {len(best_reject_ids)} reject")

    # Now process each modality using the pre-selected IDs
    for modality in modalities:
        print(f"\nProcessing modality: {modality}")

        train_path = get_base_dataset_path(modality, "train")
        test_path = get_base_dataset_path(modality, "test")

        if not train_path.exists() or not test_path.exists():
            print(f"Warning: Skipping {modality}, datasets not found")
            continue

        train_data = load_dataset(train_path)
        full_test_data = load_dataset(test_path)

        # Filter test data to sample_ids
        test_data = [d for d in full_test_data if d["_metadata"]["submission_id"] in sample_ids_set]
        print(f"  Filtered test data: {len(test_data)}/{len(full_test_data)} entries")

        # Build lookup by submission_id for this modality's train data
        train_by_id = {d["_metadata"]["submission_id"]: d for d in train_data}

        include_images = modality == "clean_images"

        for part_short, part_section in parts:
            print(f"\n  Processing part: {part_short} ({part_section})")

            for n_accept, n_reject, mixture_name in fewshot_mixtures:
                # Get pre-selected fewshot IDs
                accept_ids, reject_ids = fewshot_ids_cache[(part_short, mixture_name)]

                # Look up examples in this modality's train data
                accept_examples = [train_by_id[sid] for sid in accept_ids if sid in train_by_id]
                reject_examples = [train_by_id[sid] for sid in reject_ids if sid in train_by_id]

                if len(accept_examples) != len(accept_ids):
                    print(f"    Warning: Only found {len(accept_examples)}/{len(accept_ids)} accept examples in {modality}")
                if len(reject_examples) != len(reject_ids):
                    print(f"    Warning: Only found {len(reject_examples)}/{len(reject_ids)} reject examples in {modality}")

                # Build system prompt (boxed format for fewshot)
                system_prompt = build_system_prompt(modifier=None, output_format="boxed")

                # Process fewshot examples - extract their parts
                processed_accept = []
                processed_reject = []

                for ex in accept_examples:
                    human_msg = [m["value"] for m in ex["conversations"] if m["from"] == "human"][0]
                    paper_content = extract_paper_content(human_msg)
                    part_content, kept_indices = create_part_content(paper_content, part_section)

                    if part_content:
                        new_ex = json.loads(json.dumps(ex))
                        if modality == "clean":
                            part_content = clean_image_tags(part_content)
                        for msg in new_ex["conversations"]:
                            if msg["from"] == "human":
                                msg["value"] = part_content
                        if include_images and "images" in new_ex:
                            new_ex["images"] = [new_ex["images"][i] for i in kept_indices if i < len(new_ex["images"])]
                        processed_accept.append(new_ex)

                for ex in reject_examples:
                    human_msg = [m["value"] for m in ex["conversations"] if m["from"] == "human"][0]
                    paper_content = extract_paper_content(human_msg)
                    part_content, kept_indices = create_part_content(paper_content, part_section)

                    if part_content:
                        new_ex = json.loads(json.dumps(ex))
                        if modality == "clean":
                            part_content = clean_image_tags(part_content)
                        for msg in new_ex["conversations"]:
                            if msg["from"] == "human":
                                msg["value"] = part_content
                        if include_images and "images" in new_ex:
                            new_ex["images"] = [new_ex["images"][i] for i in kept_indices if i < len(new_ex["images"])]
                        processed_reject.append(new_ex)

                if not processed_accept and n_accept > 0:
                    print(f"    Warning: No valid accept examples for {part_short}")
                if not processed_reject and n_reject > 0:
                    print(f"    Warning: No valid reject examples for {part_short}")

                # Create fewshot prefix from processed examples
                fewshot_prefix = create_fewshot_prefix(processed_accept, processed_reject, include_images)

                # Process test data - use whole paper with part-based fewshot prefix
                new_data = []

                for entry in test_data:
                    human_msg = [m["value"] for m in entry["conversations"] if m["from"] == "human"][0]
                    paper_content = extract_paper_content(human_msg)
                    new_entry = json.loads(json.dumps(entry))

                    if modality == "clean":
                        paper_content = clean_image_tags(paper_content)
                    for msg in new_entry["conversations"]:
                        if msg["from"] == "human":
                            msg["value"] = fewshot_prefix + paper_content

                    # Set system prompt
                    new_entry["conversations"] = set_system_prompt(
                        new_entry["conversations"], system_prompt
                    )

                    # Handle images for clean_images
                    if include_images and "images" in new_entry:
                        # Get fewshot images
                        fewshot_images = []
                        for ex in processed_accept + processed_reject:
                            if "images" in ex:
                                fewshot_images.extend(ex["images"])

                        # Get kept images from test entry
                        test_images = new_entry["images"]

                        new_entry["images"] = fewshot_images + test_images

                    new_data.append(new_entry)

                output_name = f"{OUTPUT_PREFIX}_fewshot_{part_short}_{mixture_name}_{modality}_test"
                save_dataset(new_data, output_name)
                created_datasets.append(output_name)

    return created_datasets


# ============================================================================
# Base Dataset Generation (shared 500 samples)
# ============================================================================

def create_base_datasets(sample_ids: List[str]):
    """Create base datasets with the shared 500 samples across all modalities.

    Creates 3 datasets:
    - base_clean_test: Text-only version
    - base_clean_images_test: Text + images version
    - base_vision_test: Images-only version

    These serve as reference datasets with the exact 500 samples used in all ablations.
    """
    print("\n=== Creating Base Datasets (Shared 500 Samples) ===")

    modalities = ["clean", "clean_images", "vision"]
    sample_ids_set = set(sample_ids)
    created_datasets = []

    # Build simple system prompt (no modifier, boxed format)
    system_prompt = build_system_prompt(modifier=None, output_format="boxed")

    for modality in modalities:
        base_path = get_base_dataset_path(modality, "test")
        if not base_path.exists():
            print(f"Warning: Skipping {modality}, base dataset not found at {base_path}")
            continue

        data = load_dataset(base_path)

        # Filter to selected sample IDs (maintaining order)
        subset_data = [d for d in data if d["_metadata"]["submission_id"] in sample_ids_set]

        print(f"  {modality}: {len(subset_data)}/{len(data)} samples selected")

        if len(subset_data) != len(sample_ids):
            print(f"    Warning: Expected {len(sample_ids)} samples, got {len(subset_data)}")

        # Process entries
        new_data = []
        for entry in subset_data:
            new_entry = json.loads(json.dumps(entry))  # Deep copy

            # Update user message to just contain paper content
            for msg in new_entry["conversations"]:
                if msg["from"] == "human":
                    paper_content = extract_paper_content(msg["value"])

                    if modality == "clean":
                        paper_content = clean_image_tags(paper_content)

                    msg["value"] = paper_content

            # Set system prompt
            new_entry["conversations"] = set_system_prompt(
                new_entry["conversations"], system_prompt
            )

            new_data.append(new_entry)

        output_name = f"{OUTPUT_PREFIX}_base_{modality}_test"
        save_dataset(new_data, output_name)
        created_datasets.append(output_name)

    return created_datasets


def verify_sample_ids(sample_ids: List[str], reference_dataset_name: str = None):
    """Verify that the generated sample IDs match an existing ablation dataset.

    Args:
        sample_ids: List of submission IDs generated by get_sample_ids()
        reference_dataset_name: Name of existing dataset to compare against
                               (default: critical_boxed_clean_test)

    Returns:
        True if IDs match, False otherwise
    """
    if reference_dataset_name is None:
        reference_dataset_name = f"{OUTPUT_PREFIX}_critical_boxed_clean_test"

    reference_path = DATA_DIR / reference_dataset_name / "data.json"

    if not reference_path.exists():
        print(f"Warning: Reference dataset not found at {reference_path}")
        return False

    with open(reference_path) as f:
        reference_data = json.load(f)

    reference_ids = [d["_metadata"]["submission_id"] for d in reference_data]

    # Compare
    sample_set = set(sample_ids)
    reference_set = set(reference_ids)

    if sample_set == reference_set:
        print(f"✓ Sample IDs match reference dataset ({len(sample_ids)} IDs)")
        return True
    else:
        missing_in_sample = reference_set - sample_set
        missing_in_ref = sample_set - reference_set

        print(f"✗ Sample IDs do NOT match reference dataset!")
        print(f"  Sample IDs: {len(sample_ids)}, Reference IDs: {len(reference_ids)}")
        print(f"  In reference but not in sample: {len(missing_in_sample)}")
        print(f"  In sample but not in reference: {len(missing_in_ref)}")

        if missing_in_sample:
            print(f"  Missing examples: {list(missing_in_sample)[:5]}...")
        if missing_in_ref:
            print(f"  Extra examples: {list(missing_in_ref)[:5]}...")

        return False


def export_sample_ids(sample_ids: List[str], output_path: Path = None):
    """Export sample IDs to a JSON file for reference.

    Args:
        sample_ids: List of submission IDs
        output_path: Path to output file (default: DATA_DIR/shared_sample_ids.json)
    """
    if output_path is None:
        output_path = DATA_DIR / "shared_sample_ids.json"

    with open(output_path, 'w') as f:
        json.dump({
            "description": "Shared sample IDs used across all ablation v1 datasets",
            "count": len(sample_ids),
            "seed": 42,
            "submission_ids": sample_ids,
        }, f, indent=2)

    print(f"Exported {len(sample_ids)} sample IDs to {output_path}")


def load_sample_ids(input_path: Path = None) -> List[str]:
    """Load sample IDs from a JSON file.

    Args:
        input_path: Path to input file (default: DATA_DIR/shared_sample_ids.json)

    Returns:
        List of submission IDs
    """
    if input_path is None:
        input_path = DATA_DIR / "shared_sample_ids.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Sample IDs file not found at {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    return data["submission_ids"]


# ============================================================================
# Dataset Info Update
# ============================================================================

def update_dataset_info(created_datasets: List[str]):
    """Add new datasets to dataset_info.json."""
    info_path = DATA_DIR / "dataset_info.json"

    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
    else:
        info = {}

    for dataset_name in created_datasets:
        if dataset_name not in info:
            # Determine if vision model needed
            is_vision = "vision" in dataset_name or "clean_images" in dataset_name

            columns = {"messages": "conversations"}
            if is_vision:
                columns["images"] = "images"

            info[dataset_name] = {
                "file_name": f"{dataset_name}/data.json",
                "formatting": "sharegpt",
                "columns": columns,
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                    "system_tag": "system"
                }
            }
            print(f"Registered: {dataset_name}")

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate ablation datasets v1")
    parser.add_argument("--base-only", action="store_true",
                       help="Only generate base datasets with shared 500 samples")
    parser.add_argument("--verify", action="store_true",
                       help="Verify sample IDs match existing ablation datasets")
    parser.add_argument("--export-ids", action="store_true",
                       help="Export sample IDs to JSON file")
    parser.add_argument("--use-existing-ids", action="store_true",
                       help="Use existing sample IDs from shared_sample_ids.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Ablation Dataset Generation v1")
    print("=" * 70)

    # Generate or load shared sample IDs
    if args.use_existing_ids:
        try:
            sample_ids = load_sample_ids()
            print(f"Loaded {len(sample_ids)} sample IDs from existing file")
        except FileNotFoundError:
            print("No existing sample IDs file found, generating new ones")
            sample_ids = get_sample_ids(sample_size=500)
    else:
        sample_ids = get_sample_ids(sample_size=500)

    # Verify against existing datasets if requested
    if args.verify:
        print("\n=== Verifying Sample IDs ===")
        verify_sample_ids(sample_ids)
        return

    # Export sample IDs if requested
    if args.export_ids:
        export_sample_ids(sample_ids)

    all_datasets = []

    if args.base_only:
        # Only generate base datasets
        base_datasets = create_base_datasets(sample_ids)
        all_datasets.extend(base_datasets)
    else:
        # Generate all datasets

        # 0. Base datasets (3 datasets)
        base_datasets = create_base_datasets(sample_ids)
        all_datasets.extend(base_datasets)

        # 1. Critical/Less Critical Modifiers (12 datasets)
        critical_datasets = create_critical_datasets(sample_ids)
        all_datasets.extend(critical_datasets)

        # 2. Fewshot Full Paper (9 datasets)
        fewshot_full_datasets = create_fewshot_full_paper_datasets(sample_ids)
        all_datasets.extend(fewshot_full_datasets)

        # 3. Fewshot Paper Parts (24 datasets)
        fewshot_parts_datasets = create_fewshot_paper_parts_datasets(sample_ids)
        all_datasets.extend(fewshot_parts_datasets)

    # Update dataset_info.json
    print("\n=== Updating dataset_info.json ===")
    update_dataset_info(all_datasets)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total datasets created: {len(all_datasets)}")
    if not args.base_only:
        print(f"  - Base: {len(base_datasets)}")
        print(f"  - Critical/Less Critical: {len(critical_datasets)}")
        print(f"  - Fewshot Full Paper: {len(fewshot_full_datasets)}")
        print(f"  - Fewshot Paper Parts: {len(fewshot_parts_datasets)}")
    print("\nDataset names:")
    for name in sorted(all_datasets):
        print(f"  {name}")


if __name__ == "__main__":
    main()
