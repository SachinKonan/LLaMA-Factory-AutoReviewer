#!/usr/bin/env python3
"""Copy 2026 labelfix original text+vision datasets to shared directory.

Copies:
- Text labelfix: 3 data.json (train/test/val)
- Vision labelfix: 6 data.json (3 regular + 3 filtered24480)
- Only images referenced by the vision datasets
- Updates destination dataset_info.json
"""

import json
import os
import shutil
import sys
from pathlib import Path

SRC = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data")
DST = Path("/scratch/gpfs/ZHUANGL/jl0796/shared/data")

TEXT_PREFIX = "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
VISION_PREFIX = "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480"

SPLITS = ["train", "test", "validation"]

DATASET_NAMES = (
    [f"{TEXT_PREFIX}_{s}" for s in SPLITS]
    + [f"{VISION_PREFIX}_{s}" for s in SPLITS]
)


def copy_data_jsons():
    """Copy all data.json files."""
    copied = 0
    for name in DATASET_NAMES:
        src_file = SRC / name / "data.json"
        dst_dir = DST / name
        dst_file = dst_dir / "data.json"

        if not src_file.exists():
            print(f"  WARNING: missing {src_file}")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        if dst_file.exists() and dst_file.stat().st_size == src_file.stat().st_size:
            print(f"  SKIP (same size): {name}/data.json")
        else:
            print(f"  COPY: {name}/data.json ({src_file.stat().st_size / 1e6:.1f} MB)")
            shutil.copy2(str(src_file), str(dst_file))
        copied += 1
    return copied


def collect_images():
    """Collect all unique image paths from vision datasets."""
    all_images = set()
    for split in SPLITS:
        path = SRC / f"{VISION_PREFIX}_{split}" / "data.json"
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            all_images.update(entry.get("images", []))
    return all_images


def copy_images(images):
    """Copy only referenced images to destination.

    Image paths in JSON look like: "data/images/{submission_id}/page_N.png"
    Source: SRC (repo/data) contains images/ subdirectory
    Dest: DST (shared/data) should also contain images/ subdirectory
    """
    total = len(images)
    copied = 0
    skipped = 0
    missing = 0

    for i, img_path in enumerate(sorted(images)):
        # img_path = "data/images/xxx/page.png"
        # strip "data/" prefix to get path relative to data dir
        rel = img_path.removeprefix("data/")
        src_file = SRC / rel
        dst_file = DST / rel

        if dst_file.exists():
            skipped += 1
        elif not src_file.exists():
            missing += 1
            if missing <= 5:
                print(f"  MISSING: {src_file}")
        else:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_file), str(dst_file))
            copied += 1

        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i+1}/{total} (copied={copied}, skipped={skipped}, missing={missing})")

    return copied, skipped, missing


def update_dataset_info():
    """Add entries to destination dataset_info.json."""
    src_info_path = SRC / "dataset_info.json"
    dst_info_path = DST / "dataset_info.json"

    with open(src_info_path) as f:
        src_info = json.load(f)

    if dst_info_path.exists():
        with open(dst_info_path) as f:
            dst_info = json.load(f)
    else:
        dst_info = {}

    added = 0
    for name in DATASET_NAMES:
        if name in src_info:
            dst_info[name] = src_info[name]
            added += 1
        else:
            print(f"  WARNING: {name} not in source dataset_info.json")

    with open(dst_info_path, "w") as f:
        json.dump(dst_info, f, indent=2)
        f.write("\n")

    return added


def main():
    print("=" * 60)
    print("Copying 2026 labelfix datasets to shared directory")
    print(f"  Source: {SRC}")
    print(f"  Dest:   {DST}")
    print("=" * 60)

    print("\n--- Step 1: Copy data.json files ---")
    n_jsons = copy_data_jsons()
    print(f"  Done: {n_jsons} data.json files")

    print("\n--- Step 2: Collect referenced images ---")
    images = collect_images()
    print(f"  Found {len(images)} unique images")

    print("\n--- Step 3: Copy images ---")
    copied, skipped, missing = copy_images(images)
    print(f"  Copied: {copied}, Skipped (existing): {skipped}, Missing: {missing}")

    print("\n--- Step 4: Update dataset_info.json ---")
    n_entries = update_dataset_info()
    print(f"  Added/updated {n_entries} entries")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
