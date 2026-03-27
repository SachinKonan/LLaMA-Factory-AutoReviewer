#!/usr/bin/env python3
"""Strip '# REFERENCES' sections from NIPS and ICML/COLM text datasets.

NIPS 2023-2025 and ICML/COLM 2024-2025 text datasets accidentally include full
REFERENCES sections. ICLR data already has these stripped. This script creates
cleaned _noref versions with references removed.
"""

import json
import os
import re
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REFERENCE_PATTERN = re.compile(r"^# (?:REFERENCES|References)\s*$", re.MULTILINE)

# Source -> destination dataset name mapping
DATASETS = {
    "nips_2021_2025_original_text_v7_train": "nips_2021_2025_original_text_v7_noref_train",
    "nips_2021_2025_original_text_v7_validation": "nips_2021_2025_original_text_v7_noref_validation",
    "nips_2021_2025_original_text_v7_test": "nips_2021_2025_original_text_v7_noref_test",
    "icml_colm_2024_2025_clean_binary_train": "icml_colm_2024_2025_clean_binary_noref_train",
    "icml_colm_2024_2025_clean_binary_validation": "icml_colm_2024_2025_clean_binary_noref_validation",
    "icml_colm_2024_2025_clean_binary_test": "icml_colm_2024_2025_clean_binary_noref_test",
}

# Dir to delete
EVAL500_DIR = "nips_2021_2025_original_text_v7_eval500"


def strip_references(text: str) -> tuple[str, bool]:
    """Remove everything from '# REFERENCES' onward. Returns (new_text, was_stripped)."""
    match = REFERENCE_PATTERN.search(text)
    if match:
        return text[: match.start()].rstrip(), True
    return text, False


def process_dataset(src_name: str, dst_name: str) -> dict:
    """Process one dataset, return stats."""
    src_path = DATA_DIR / src_name / "data.json"
    dst_dir = DATA_DIR / dst_name
    dst_path = dst_dir / "data.json"

    with open(src_path) as f:
        data = json.load(f)

    total = len(data)
    stripped = 0
    total_chars_before = 0
    total_chars_after = 0

    for entry in data:
        for conv in entry["conversations"]:
            if conv["from"] == "human":
                text = conv["value"]
                total_chars_before += len(text)
                new_text, was_stripped = strip_references(text)
                if was_stripped:
                    conv["value"] = new_text
                    stripped += 1
                total_chars_after += len(conv["value"])

    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    avg_before = total_chars_before / total if total else 0
    avg_after = total_chars_after / total if total else 0

    return {
        "total": total,
        "stripped": stripped,
        "avg_chars_before": avg_before,
        "avg_chars_after": avg_after,
        "reduction_pct": (1 - avg_after / avg_before) * 100 if avg_before else 0,
    }


def update_dataset_info():
    """Add noref entries and remove eval500 entry from dataset_info.json."""
    info_path = DATA_DIR / "dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)

    # Remove eval500
    if EVAL500_DIR in info:
        del info[EVAL500_DIR]
        print(f"Removed '{EVAL500_DIR}' from dataset_info.json")

    # Add noref entries
    entry_template = {
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

    for dst_name in DATASETS.values():
        info[dst_name] = {
            "file_name": f"{dst_name}/data.json",
            **entry_template,
        }
        print(f"Added '{dst_name}' to dataset_info.json")

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")


def delete_eval500():
    """Delete the eval500 directory."""
    eval_dir = DATA_DIR / EVAL500_DIR
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
        print(f"Deleted directory: {eval_dir}")
    else:
        print(f"Directory not found (already deleted?): {eval_dir}")


def main():
    print("=" * 70)
    print("Strip References from NIPS and ICML/COLM Text Datasets")
    print("=" * 70)

    # Step 1: Delete eval500
    print("\n--- Deleting eval500 ---")
    delete_eval500()

    # Step 2: Process datasets
    print("\n--- Processing datasets ---")
    for src_name, dst_name in DATASETS.items():
        print(f"\n{src_name} -> {dst_name}")
        stats = process_dataset(src_name, dst_name)
        print(f"  Total entries: {stats['total']}")
        print(f"  Entries with refs stripped: {stats['stripped']}")
        print(f"  Avg human msg chars: {stats['avg_chars_before']:.0f} -> {stats['avg_chars_after']:.0f} ({stats['reduction_pct']:.1f}% reduction)")

    # Step 3: Update dataset_info.json
    print("\n--- Updating dataset_info.json ---")
    update_dataset_info()

    # Step 4: Verify no references remain
    print("\n--- Verification ---")
    all_clean = True
    for dst_name in DATASETS.values():
        dst_path = DATA_DIR / dst_name / "data.json"
        with open(dst_path) as f:
            data = json.load(f)
        for entry in data:
            for conv in entry["conversations"]:
                if conv["from"] == "human" and REFERENCE_PATTERN.search(conv["value"]):
                    print(f"  WARNING: References still found in {dst_name}!")
                    all_clean = False
                    break
    if all_clean:
        print("  All noref datasets verified clean — no '# REFERENCES' found.")

    print("\nDone!")


if __name__ == "__main__":
    main()
