#!/usr/bin/env python3
"""
Generate modified datasets for final inference scaling experiments.

Generates 8 single-prompt variants for both clean and vision modalities,
plus 1 dataset for Mimic 3 Reviewers (RL System Prompt).

Usage:
    python generate_datasets.py --base_data_dir /path/to/base/data --output_dir ./data
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "2_8_26"))
from shared.prompt_templates import build_system_prompt

FEWSHOT_EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "fewshot_examples.json")

def load_dataset(data_path: str) -> List[Dict]:
    json_path = os.path.join(data_path, "data.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_dataset(data: List[Dict], output_path: str):
    os.makedirs(output_path, exist_ok=True)
    json_path = os.path.join(output_path, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} samples to {json_path}")

def extract_paper_content(user_message: str) -> str:
    prefix_end_marker = " - Note: ICLR generally has a ~30% acceptance rate\n\n"
    if prefix_end_marker in user_message:
        idx = user_message.find(prefix_end_marker)
        return user_message[idx + len(prefix_end_marker):].strip()

    lines = user_message.split('\n')
    for i, line in enumerate(lines):
        if i > 0 and line.strip() == '' and i < len(lines) - 1:
            if lines[i-1].strip().endswith('rate'):
                return '\n'.join(lines[i+1:]).strip()
    return user_message

def format_fewshot_string(examples: dict, format_type: str = "1-1") -> str:
    if format_type == "1-1":
        return f"""Here are example reviews for reference:

Example 1 (Accept):
Paper: {examples['accept_paper']}
Review: {examples['accept_review']}

Example 2 (Reject):
Paper: {examples['reject_paper']}
Review: {examples['reject_review']}

"""
    elif format_type == "0-2":
        return f"""Here are example reviews for reference:

Example 1 (Reject):
Paper: {examples['reject_paper']}
Review: {examples['reject_review']}

Example 2 (Reject):
Paper: {examples['reject_paper_2']}
Review: {examples['reject_review_2']}

"""
    return ""

def transform_sample(sample: Dict, modifier: str, output_format: str, is_fewshot: bool, fewshot_string: Optional[str] = None, prompt_options: str = None) -> Dict:
    new_sample = sample.copy()
    conversations = []

    # Get the appropriate system prompt directly from templates
    system_prompt = build_system_prompt(modifier=modifier if modifier != "standard" else None, output_format=output_format, options=prompt_options)

    # Reconstruct user message using correct prompt instructions
    user_prefix = ""
    if output_format == "new" or output_format == "json":
        # Add a custom prefix to introduce the task if not using boxed
        user_prefix = "Review this paper and predict its ICLR acceptance. Note: ICLR has a ~30% acceptance rate.\n"
    else:
        # standard boxed
        user_prefix = "I am giving you a paper. I want to predict its acceptance outcome at ICLR.\n - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n - Note: ICLR generally has a ~30% acceptance rate\n\n"

    # Add fewshot examples to the prompt if needed
    fewshot_section = fewshot_string if is_fewshot and fewshot_string else ""

    for conv in sample["conversations"]:
        new_conv = conv.copy()
        if conv["from"] == "system":
            new_conv["value"] = system_prompt
        elif conv["from"] == "human":
            paper_content = extract_paper_content(conv["value"])
            if prompt_options == "SACHIN_RL":
                user_message = f"Here is the paper to evaluate:\n\n{paper_content}"
            else:
                user_message = f"{user_prefix}{fewshot_section}Here is the paper:\n\n{paper_content}"
            
            new_conv["value"] = user_message
        
        conversations.append(new_conv)
        
        # If there's no system prompt initially, we need to ensure it's there. The templates build_system_prompt returns the prompt text itself.
        # But wait, the standard datasets usually have a system prompt entry.

    # If the original didn't have a system prompt (some data might not), we should insert it.
    has_system = any(c["from"] == "system" for c in conversations)
    if not has_system:
        # Insert system prompt at the beginning
        conversations.insert(0, {"from": "system", "value": system_prompt})

    new_sample["conversations"] = conversations
    return new_sample

def load_fewshot_examples() -> Optional[dict]:
    if not os.path.exists(FEWSHOT_EXAMPLES_PATH):
        print(f"Warning: Few-shot examples file not found at {FEWSHOT_EXAMPLES_PATH}")
        return None
    with open(FEWSHOT_EXAMPLES_PATH, "r", encoding="utf-8") as f:
        examples = json.load(f)
    return examples

def generate_datasets(
    base_data_dir: str,
    output_dir: str,
    dataset_names: List[str],
    splits: List[str] = ["test"],
    limit: Optional[int] = None
):
    os.makedirs(output_dir, exist_ok=True)
    fewshot_string = load_fewshot_examples()

    dataset_info = {}

    for dataset_name in dataset_names:
        for split in splits:
            full_name = f"{dataset_name}_{split}"
            input_path = os.path.join(base_data_dir, full_name)

            if not os.path.exists(input_path):
                print(f"Warning: Dataset not found: {input_path}")
                continue

            print(f"\\nProcessing {full_name}...")
            data = load_dataset(input_path)
            if limit is not None:
                data = data[:limit]

            is_vision = "vision" in full_name
            columns = {"messages": "conversations"}
            if is_vision:
                columns["images"] = "images"

            # 1. 12 Single-prompt variants
            for modifier in ["standard", "critical"]:
                for fewshot_opt in ["nofewshot", "fewshot_1-1", "fewshot_0-2"]:
                    for format_opt in ["boxed", "json"]:
                        variant_name = f"{modifier}_{fewshot_opt}_{format_opt}"
                        is_fewshot = (fewshot_opt != "nofewshot")
                        
                        fewshot_string_variant = None
                        if is_fewshot and fewshot_string:
                            fs_type = fewshot_opt.split("_")[1] # 1-1 or 0-2
                            fewshot_string_variant = format_fewshot_string(fewshot_string, format_type=fs_type)
                        
                        variant_data = [transform_sample(s, modifier=modifier, output_format="boxed" if format_opt == "boxed" else "json", is_fewshot=is_fewshot, fewshot_string=fewshot_string_variant) for s in data]
                        
                        out_name = f"{full_name}_{variant_name}"
                        out_path = os.path.join(output_dir, out_name)
                        save_dataset(variant_data, out_path)
                        
                        dataset_info[out_name] = {
                            "file_name": f"{out_name}/data.json",
                            "formatting": "sharegpt",
                            "columns": columns,
                            "tags": {
                                "role_tag": "from",
                                "content_tag": "value",
                                "user_tag": "human",
                                "assistant_tag": "gpt",
                                "system_tag": "system",
                            },
                        }

            # 2. Mimic 3 Reviewers (RL System Prompt -> PDR)
            rl_data = [transform_sample(s, modifier="standard", output_format="boxed", is_fewshot=False, prompt_options="SACHIN_RL") for s in data]
            rl_out_name = f"{full_name}_pdr_boxed"
            rl_out_path = os.path.join(output_dir, rl_out_name)
            save_dataset(rl_data, rl_out_path)
            
            dataset_info[rl_out_name] = {
                "file_name": f"{rl_out_name}/data.json",
                "formatting": "sharegpt",
                "columns": columns,
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "human",
                    "assistant_tag": "gpt",
                    "system_tag": "system",
                },
            }

    # Save dataset_info.json
    info_path = os.path.join(output_dir, "dataset_info.json")
    existing = {}
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            existing = json.load(f)
    existing.update(dataset_info)
    with open(info_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\\nUpdated {info_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_dir", type=str, default="/scratch/gpfs/ZHUANGL/jl0796/shared/data")
    parser.add_argument("--output_dir", type=str, default="./final_inference_scaling/data")
    parser.add_argument("--splits", type=str, nargs="+", default=["test"])
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()

    dataset_names = [
        "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered",
        "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480",
    ]

    generate_datasets(
        base_data_dir=args.base_data_dir,
        output_dir=args.output_dir,
        dataset_names=dataset_names,
        splits=args.splits,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
