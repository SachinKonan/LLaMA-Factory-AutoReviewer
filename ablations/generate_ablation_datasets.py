
import json
import os
import re
import argparse
from warnings import warn

# Configuration
DATA_DIR = "/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/data"
OUTPUT_DIR = "/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/data"

# Base Dataset Common Prefix
# Based on file listing: iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_train
BASE_PREFIX = "iclr_2020_2025_85_5_10_split6_original"

MODALITIES = ["clean", "clean+images", "vision"]
SPLITS = ["train", "validation", "test"]

# Section Regex Patterns (Case Insensitive)
# Adjust these based on common variations in ICLR papers
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
    "related_work": ["RELATED WORK"],
    "methodology": ["METHOD", "METHODOLOGY"],
    "experimental_results": ["RESULTS", "EXPERIMENTS", "EVALUATION"],
    "discussion": ["DISCUSSION"],
    "conclusion": ["CONCLUSION"],
}

def load_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_dataset(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def generate_counterfactual(base_name, modality, split):
    """
    Task 1: Filters for <2024 papers that were 'Accept', turns them into 'Reject'.
    Includes 2024/2025 papers as is.
    """
    input_dir = os.path.join(DATA_DIR, f"{BASE_PREFIX}_{modality}_binary_noreviews_v6_{split}")
    input_file = os.path.join(input_dir, "data.json")
    
    if not os.path.exists(input_file):
        print(f"Skipping missing source: {input_file}")
        return

    data = load_dataset(input_file)
    new_data = []

    # Valid accept decisions
    accept_decisions = ["accept", "oral", "poster", "spotlight", "top-5%"]

    for entry in data:
        meta = entry.get("_metadata", {})
        year = meta.get("year")
        decision = meta.get("decision", "").lower()
        
        # Check if it's an accept
        is_accept = any(d in decision for d in accept_decisions) or meta.get("answer") == "Accept"
        
        # Logic: 
        # 1. If < 2024 AND Accept -> Flip to Reject
        # 2. Else -> Keep as is
        
        if year is not None and year < 2024 and is_accept:
            # Modify: Turn into Reject
            new_entry = json.loads(json.dumps(entry)) # Deep copy
            new_entry["_metadata"]["decision"] = "reject"
            new_entry["_metadata"]["answer"] = "Reject"
            
            # Modify conversation output
            for msg in new_entry["conversations"]:
                if msg["from"] == "gpt":
                    msg["value"] = msg["value"].replace("Accept", "Reject")
                    # Also handle boxed format if needed? Usually just replace Accept
            
            new_data.append(new_entry)
        else:
            # Keep as is
            new_data.append(entry)

    if not new_data:
        print(f"Warning: No entries found for counterfactual in {base_name}")
        return

    output_name = f"{BASE_PREFIX}_{modality}_binary_noreviews_counterfactual_reject_v6_{split}"
    output_path = os.path.join(OUTPUT_DIR, output_name, "data.json")
    print(f"Saving {len(new_data)} entries to {output_path}")
    save_dataset(new_data, output_path)
    return output_name

def parse_and_ablate(text, filter_type, section_key):
    """
    Parses the text to identify sections and performs ablation.
    
    filter_type: 'only' (keep only this section + title/abstract)
                 'except' (remove this section)
    section_key: key in SECTIONS dict
    """
    
    # Simple markdown parser assumes sections start with # header
    # We locate all headers to find boundaries
    
    # 1. Identify all headers positions
    lines = text.split('\n')
    
    header_indices = []
    # Always assume start (0) is Title/Abstract until first header
    
    # Locate requested section
    target_section_start = -1
    target_section_end = -1
    
    # Very basic parsing: Find line numbers of all headers
    headers = []
    for i, line in enumerate(lines):
        if line.strip().startswith('# '):
            headers.append((i, line.strip()))
            
    # Map headers to our semantics
    section_map = {} # header_index -> section_key
    
    for i, (line_idx, line_content) in enumerate(headers):
        matched = None
        for key, pattern in SECTIONS.items():
            if pattern.match(line_content):
                matched = key
                break
        
        # Heuristic fallbacks if regex fails but simple keyword works (optional)
        if not matched: 
            upper_line = line_content.upper()
            for key, kws in KEYWORDS.items():
                if any(kw in upper_line for kw in kws):
                    matched = key
                    break
        
        if matched:
            section_map[i] = matched

    # Find target section bounds
    target_indices = [i for i, (idx, content) in enumerate(headers) if section_map.get(i) == section_key]
    
    # Special case for intro_discussion: intersection of both
    if section_key == "intro_discussion":
        intro_indices = [i for i, (idx, content) in enumerate(headers) if section_map.get(i) == 'introduction']
        disc_indices = [i for i, (idx, content) in enumerate(headers) if section_map.get(i) == 'discussion']
        
        if intro_indices and disc_indices:
            target_indices = intro_indices + disc_indices

    if not target_indices:
        return None # Target section not found, filter this paper out
    
    
    # Logics
    # Structure: [Title/Abs] [Header 0] ... [Header N]
    
    # Content blocks:
    # Block -1: Start to Header 0 (Title/Abstract) - ALWAYS KEEP
    # Block i: Header i to Header i+1 (or End)
    
    filtered_lines = []
    
    # Always keep Title/Abstract (0 to Header 0)
    if headers:
        first_header_line = headers[0][0]
        filtered_lines.extend(lines[0:first_header_line])
    else:
        # No headers found at all
        return None 
        
    for i in range(len(headers)):
        current_header_line = headers[i][0]
        next_header_line = headers[i+1][0] if i+1 < len(headers) else len(lines)
        
        is_target = False
        if section_key == "intro_discussion":
             if section_map.get(i) in ["introduction", "discussion"]:
                 is_target = True
        elif section_map.get(i) == section_key:
            is_target = True
            
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


def generate_section_ablation(base_name, modality, split):
    """
    Task 2: Section Ablations.
    """
    if modality == "vision":
        return # Task says "Use 2 modalities text, text+images"
        
    input_dir = os.path.join(DATA_DIR, f"{BASE_PREFIX}_{modality}_binary_noreviews_v6_{split}")
    input_file = os.path.join(input_dir, "data.json")

    if not os.path.exists(input_file):
        return

    print(f"Processing {base_name} ({modality}, {split})...")
    data = load_dataset(input_file)
    
    # Define tasks
    # key: section_key_for_parser
    # name: for filename
    single_sections = ["introduction", "related_work", "methodology", "experimental_results", "discussion"]
    
    tasks = []
    for sec in single_sections:
        tasks.append({"key": sec, "type": "only", "suffix": f"only_{sec}"})
        tasks.append({"key": sec, "type": "except", "suffix": f"no_{sec}"})
        
    # Intro+Discussion
    tasks.append({"key": "intro_discussion", "type": "only", "suffix": "only_intro_discussion"})
    tasks.append({"key": "intro_discussion", "type": "except", "suffix": "no_intro_discussion"})

    # Prepare containers for new datasets
    new_datasets = {t["suffix"]: [] for t in tasks}
    
    stats_headers_found = {k: 0 for k in SECTIONS.keys()}
    stats_total = 0
    
    for entry in data:
        stats_total += 1
        
        # Extract prompt
        human_msg = next((msg for msg in entry["conversations"] if msg["from"] == "human"), None)
        if not human_msg: continue
        
        original_text = human_msg["value"]
        
        # Check contained sections for stats
        # (Naive check just to see what exists)
        # Using the same mapping logic as the parser for consistency would be better, 
        # but let's just rely on the parser return being None to indicate missing.
        
        for task in tasks:
            ablated_text = parse_and_ablate(original_text, task["type"], task["key"])
            
            if ablated_text:
                new_entry = json.loads(json.dumps(entry))
                # Update text
                for msg in new_entry["conversations"]:
                    if msg["from"] == "human":
                        msg["value"] = ablated_text
                new_datasets[task["suffix"]].append(new_entry)

    # Save
    for suffix, entries in new_datasets.items():
        if not entries:
            continue
            
        output_name = f"{BASE_PREFIX}_{modality}_binary_noreviews_{suffix}_v6_{split}"
        output_path = os.path.join(OUTPUT_DIR, output_name, "data.json")
        print(f"  Saving {suffix}: {len(entries)} entries")
        save_dataset(entries, output_path)
        
    print(f"  Total processed: {stats_total}")


def update_dataset_info():
    """
    Scans data directory for new datasets and adds them to dataset_info.json
    """
    info_path = os.path.join(DATA_DIR, "dataset_info.json")
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    # Find all directories starting with BASE_PREFIX
    for item in os.listdir(DATA_DIR):
        if not item.startswith(BASE_PREFIX):
            continue
            
        if not os.path.isdir(os.path.join(DATA_DIR, item)):
            continue
            
        # Check if already in info
        if item in info:
            continue
            
        # Add new entry
        # Template copies structure from a known existing one, e.g., the base train one
        # "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_train": {
        #   "file_name": "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_train/data.json",
        #   "formatting": "sharegpt", ... }
        
        print(f"Registering new dataset: {item}")
        info[item] = {
            "file_name": f"{item}/data.json",
            "formatting": "sharegpt",
            "columns": {
              "messages": "conversations"
            },
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-counterfactual", action="store_true")
    parser.add_argument("--skip-sections", action="store_true")
    args = parser.parse_args()

    if not args.skip_counterfactual:
        print("--- Generating Counterfactual Datasets ---")
        for modality in MODALITIES:
            for split in SPLITS:
                generate_counterfactual(BASE_PREFIX, modality, split)

    if not args.skip_sections:
        print("\n--- Generating Section Ablation Datasets ---")
        # Only text modalities
        for modality in ["clean", "clean+images"]:
            for split in SPLITS:
                generate_section_ablation(BASE_PREFIX, modality, split)

    print("\n--- Updating dataset_info.json ---")
    update_dataset_info()
    print("Done.")

if __name__ == "__main__":
    main()
