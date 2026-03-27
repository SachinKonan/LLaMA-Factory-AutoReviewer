#!/usr/bin/env python3
"""Create balanced NeurIPS/ICML eval datasets by taking all rejects and
subsampling accepts to match, per venue/year. Since midtraining used no labels
and SFT used only ICLR data, all splits (train/val/test) can be used.

Outputs:
  data/balanced_nips_icml_eval/data.json          (text)
  data/balanced_nips_icml_eval_vision/data.json    (vision)
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
DATA_DIR = Path("data")

# Source datasets: (directory_name, ...) for each modality
TEXT_SOURCES = [
    "nips_2021_2025_original_text_v7_noref_train",
    "nips_2021_2025_original_text_v7_noref_validation",
    "nips_2021_2025_original_text_v7_noref_test",
    "icml_colm_2024_2025_clean_binary_noref_train",
    "icml_colm_2024_2025_clean_binary_noref_validation",
    "icml_colm_2024_2025_clean_binary_noref_test",
]

VISION_SOURCES = [
    "nips_2021_2025_original_vision_v7_train",
    "nips_2021_2025_original_vision_v7_validation",
    "nips_2021_2025_original_vision_v7_test",
    "icml_colm_2024_2025_vision_binary_train",
    "icml_colm_2024_2025_vision_binary_validation",
    "icml_colm_2024_2025_vision_binary_test",
]

# Only include venue/year combos that have rejects
# From dataMaster.json:
#   NeurIPS 2024: 196R text, 196R vision
#   NeurIPS 2025: 247R text, 249R vision
#   ICML 2025:    158R text, 162R vision
#   ICML 2024:    0R (skip)
#   COLM:         0R (skip)
VALID_VENUE_YEARS = {
    ("nips", 2024),
    ("nips", 2025),
    ("ICML", 2025),
}


def load_all_entries(source_dirs):
    """Load and deduplicate entries from multiple data.json files."""
    seen_ids = set()
    entries = []
    for dirname in source_dirs:
        fpath = DATA_DIR / dirname / "data.json"
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping")
            continue
        with open(fpath) as f:
            data = json.load(f)
        for entry in data:
            meta = entry.get("_metadata", {})
            sid = meta.get("submission_id", "")
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                entries.append(entry)
    return entries


def get_label(entry):
    """Extract accept/reject label from _metadata."""
    return entry.get("_metadata", {}).get("answer", "")


def get_venue_year(entry):
    """Extract (conference, year) from _metadata."""
    meta = entry.get("_metadata", {})
    return (meta.get("conference", ""), meta.get("year", 0))


def create_balanced_dataset(entries):
    """Create balanced dataset: for each valid venue/year, take all rejects
    and sample same number of accepts. Returns list of entries."""
    # Group by venue/year and label
    groups = defaultdict(lambda: {"Accept": [], "Reject": []})
    for entry in entries:
        vy = get_venue_year(entry)
        if vy not in VALID_VENUE_YEARS:
            continue
        label = get_label(entry)
        if label in ("Accept", "Reject"):
            groups[vy][label].append(entry)

    rng = random.Random(SEED)
    balanced = []

    for vy in sorted(groups.keys()):
        accepts = groups[vy]["Accept"]
        rejects = groups[vy]["Reject"]
        n_reject = len(rejects)
        n_accept = len(accepts)

        if n_reject == 0:
            print(f"  {vy}: 0 rejects, skipping")
            continue

        # Sample accepts to match rejects
        if n_accept >= n_reject:
            sampled_accepts = rng.sample(accepts, n_reject)
        else:
            print(f"  WARNING: {vy} has fewer accepts ({n_accept}) than rejects ({n_reject})")
            sampled_accepts = accepts

        balanced.extend(rejects)
        balanced.extend(sampled_accepts)
        print(f"  {vy[0]} {vy[1]}: {n_reject}R + {len(sampled_accepts)}A = {n_reject + len(sampled_accepts)}")

    # Shuffle the final dataset
    rng.shuffle(balanced)
    return balanced


def main():
    # --- Text dataset ---
    print("=== Text Dataset ===")
    print("Loading text entries...")
    text_entries = load_all_entries(TEXT_SOURCES)
    print(f"Total unique text entries: {len(text_entries)}")

    print("Creating balanced text dataset...")
    text_balanced = create_balanced_dataset(text_entries)
    print(f"Total balanced text samples: {len(text_balanced)}")

    out_dir = DATA_DIR / "balanced_nips_icml_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "data.json", "w") as f:
        json.dump(text_balanced, f, indent=2)
    print(f"Saved to {out_dir / 'data.json'}")

    # --- Vision dataset ---
    print("\n=== Vision Dataset ===")
    print("Loading vision entries...")
    vision_entries = load_all_entries(VISION_SOURCES)
    print(f"Total unique vision entries: {len(vision_entries)}")

    print("Creating balanced vision dataset...")
    vision_balanced = create_balanced_dataset(vision_entries)
    print(f"Total balanced vision samples: {len(vision_balanced)}")

    out_dir_v = DATA_DIR / "balanced_nips_icml_eval_vision"
    out_dir_v.mkdir(parents=True, exist_ok=True)
    with open(out_dir_v / "data.json", "w") as f:
        json.dump(vision_balanced, f, indent=2)
    print(f"Saved to {out_dir_v / 'data.json'}")

    # --- Verify ---
    print("\n=== Verification ===")
    for name, dataset in [("text", text_balanced), ("vision", vision_balanced)]:
        vy_counts = defaultdict(lambda: {"Accept": 0, "Reject": 0})
        for entry in dataset:
            vy = get_venue_year(entry)
            label = get_label(entry)
            vy_counts[vy][label] += 1
        print(f"\n{name} dataset ({len(dataset)} total):")
        for vy in sorted(vy_counts.keys()):
            c = vy_counts[vy]
            print(f"  {vy[0]} {vy[1]}: {c['Accept']}A / {c['Reject']}R")


if __name__ == "__main__":
    main()
