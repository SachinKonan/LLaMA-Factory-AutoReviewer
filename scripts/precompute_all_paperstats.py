#!/usr/bin/env python3
"""Pre-compute all 187 features from logreg_baseline.py for every paper.

Iterates through source text datasets (train/val/test), extracts features
using extract_all_features(), and saves to data/all_paperstats_v7.json.

Usage:
    python scripts/precompute_all_paperstats.py
"""

import csv
import json
import sys
import os
from pathlib import Path

# Add project root so we can import from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logreg_baseline import extract_all_features

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "massive_metadata_v7.csv"
OUTPUT_PATH = DATA_DIR / "all_paperstats_v7.json"

# Use text dataset (not vision) since extract_all_features operates on text
TEXT_DATASET = "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
SPLITS = ["train", "test", "validation"]


def load_csv_stats():
    stats = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            sid = row.get("submission_id")
            if sid:
                stats[sid] = row
    return stats


def main():
    print("Loading CSV metadata...", flush=True)
    csv_stats = load_csv_stats()
    print(f"  Loaded {len(csv_stats)} rows from CSV", flush=True)

    all_features = {}
    total = 0
    skipped = 0

    for split in SPLITS:
        path = DATA_DIR / f"{TEXT_DATASET}_{split}" / "data.json"
        print(f"\nProcessing {split} from {path}...", flush=True)
        with open(path) as f:
            data = json.load(f)
        print(f"  {len(data)} entries", flush=True)

        for i, entry in enumerate(data):
            meta = entry.get("_metadata", {})
            sid = meta.get("submission_id")
            if not sid:
                skipped += 1
                continue
            if sid in all_features:
                continue  # already processed
            if sid not in csv_stats:
                skipped += 1
                continue

            # Extract text from human turn
            text = ""
            for conv in entry["conversations"]:
                if conv["from"] == "human":
                    text = conv["value"]
                    break

            if not text:
                skipped += 1
                continue

            feats = extract_all_features(text, meta, csv_stats[sid])
            # Convert numpy types to Python native
            all_features[sid] = {k: float(v) if hasattr(v, 'item') else v for k, v in feats.items()}
            total += 1

            if total % 500 == 0:
                print(f"  Extracted features for {total} papers...", flush=True)

    print(f"\nDone! Total: {total} papers, Skipped: {skipped}", flush=True)

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_features, f)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"Saved to {OUTPUT_PATH} ({size_mb:.1f} MB)", flush=True)

    # Print a sample
    sample_sid = next(iter(all_features))
    sample = all_features[sample_sid]
    print(f"\nSample entry ({sample_sid}): {len(sample)} features")
    for k, v in list(sample.items())[:10]:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
