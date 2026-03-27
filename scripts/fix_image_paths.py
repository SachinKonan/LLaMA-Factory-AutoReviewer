#!/usr/bin/env python3
"""
Fix image paths in vision datasets: change data/images/ -> data/images_extra/
for NIPS 2021-2025, ICML/COLM 2024-2025, and ICLR 2017-2019.

Also warns if any submission_id doesn't have a folder in data/images_extra/.

Usage:
    uv run scripts/fix_image_paths.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_TO_FIX = [
    "nips_2021_2025_original_vision_v7_train",
    "nips_2021_2025_original_vision_v7_validation",
    "nips_2021_2025_original_vision_v7_test",
    "nips_2021_2025_original_vision_v7_eval500",
    "icml_colm_2024_2025_vision_binary_train",
    "icml_colm_2024_2025_vision_binary_validation",
    "icml_colm_2024_2025_vision_binary_test",
    "iclr_2017_2019_original_vision_v7_train",
    "iclr_2017_2019_original_vision_v7_validation",
    "iclr_2017_2019_original_vision_v7_test",
    "iclr_2017_2019_original_vision_v7_eval250_balanced",
]


def fix_dataset(ds_name: str):
    path = PROJECT_ROOT / "data" / ds_name / "data.json"
    if not path.exists():
        print(f"  SKIP (not found): {path}")
        return

    with open(path) as f:
        data = json.load(f)

    fixed_count = 0
    already_correct = 0
    missing_dirs = []

    for entry in data:
        sid = entry.get("_metadata", {}).get("submission_id", "unknown")
        images = entry.get("images", [])
        new_images = []

        for img in images:
            if img.startswith("data/images/") and not img.startswith("data/images_extra/"):
                new_img = img.replace("data/images/", "data/images_extra/", 1)
                new_images.append(new_img)
                fixed_count += 1
            else:
                new_images.append(img)
                if img.startswith("data/images_extra/"):
                    already_correct += 1

        entry["images"] = new_images

        # Check if submission_id folder exists in images_extra
        sid_dir = PROJECT_ROOT / "data" / "images_extra" / sid
        if not sid_dir.exists():
            missing_dirs.append(sid)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  {ds_name}: {len(data)} entries, {fixed_count} paths fixed, {already_correct} already correct")
    if missing_dirs:
        unique_missing = sorted(set(missing_dirs))
        print(f"    WARNING: {len(unique_missing)} submission_ids missing from data/images_extra/:")
        for sid in unique_missing[:10]:
            print(f"      - {sid}")
        if len(unique_missing) > 10:
            print(f"      ... and {len(unique_missing) - 10} more")


def main():
    print("Fixing image paths: data/images/ -> data/images_extra/\n")

    for ds_name in DATASETS_TO_FIX:
        fix_dataset(ds_name)

    print("\nDone!")


if __name__ == "__main__":
    main()
