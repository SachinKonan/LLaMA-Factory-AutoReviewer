#!/usr/bin/env python3
"""Count text tokens, vision tokens, and pages per entry in a dataset.

Outputs a CSV with per-entry stats for downstream plotting.

Usage:
    python scripts/count_token_stats.py --dataset nips_2021_2025_original_text_v7_noref
    python scripts/count_token_stats.py --dataset iclr_2017_2019_original_vision_v7 --config configs/final_sweep_v7_vision.yaml
"""

import argparse
import csv
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "token_stats"

IMAGE_DIRS = [
    DATA_DIR / "images",
    DATA_DIR / "images_extra",
]

# Default tokenizer (same vocab for Qwen2.5 and Qwen2.5-VL)
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ── Vision token calculation (from run_tokenization_and_count_overctx.py) ──

def smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    if max(height, width) / min(height, width) > 200:
        if width > height:
            width = height * 180
        else:
            height = width * 180
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def get_image_tokens(image_path, min_pixels=784, max_pixels=1003520):
    """Calculate image tokens from dimensions (fast, header-only)."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        rh, rw = smart_resize(height, width, factor=28, min_pixels=min_pixels, max_pixels=max_pixels)
        tokens = (rh // 14 * rw // 14) // 4  # merge_size=2, so //4
        tokens += 2  # vision_start + vision_end
        return tokens
    except Exception as e:
        print(f"Warning: {image_path}: {e}")
        return 0


# ── Page counting ──

def count_pages_from_disk(submission_id):
    """Count page images on disk for a submission."""
    for img_dir in IMAGE_DIRS:
        sub_dir = img_dir / submission_id
        if sub_dir.is_dir():
            return len([f for f in os.listdir(sub_dir) if f.startswith("page_") and f.endswith(".png")])
    return 0


# ── Worker functions ──

_tokenizer = None


def init_worker(model_name):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def process_entry(entry):
    """Process one dataset entry, return stats dict."""
    global _tokenizer

    meta = entry.get("_metadata", {})
    submission_id = meta.get("submission_id", "")
    year = meta.get("year", 0)
    conference = meta.get("conference", "unknown").lower()

    # Count text tokens (all conversation turns, excluding <image> tags)
    text_tokens = 0
    for conv in entry.get("conversations", []):
        text = conv.get("value", "").replace("<image>", "")
        text_tokens += len(_tokenizer.encode(text, add_special_tokens=False))

    # Count vision tokens and pages
    images = entry.get("images", [])
    if images:
        num_pages = len(images)
        vision_tokens = sum(get_image_tokens(img) for img in images)
    else:
        num_pages = count_pages_from_disk(submission_id)
        vision_tokens = 0

    return {
        "submission_id": submission_id,
        "year": year,
        "conference": conference,
        "num_pages": num_pages,
        "text_tokens": text_tokens,
        "vision_tokens": vision_tokens,
        "total_tokens": text_tokens + vision_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Count token stats per entry")
    parser.add_argument("--dataset", required=True, help="Dataset base name (without _train/_test/_validation)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Tokenizer model name")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--modality", default="auto", choices=["auto", "text", "vision"],
                        help="Dataset modality (auto-detected if not specified)")
    args = parser.parse_args()

    dataset_base = args.dataset.replace("_train", "").replace("_test", "").replace("_validation", "")

    # Load all splits
    all_data = []
    for split in ["train", "validation", "test"]:
        path = DATA_DIR / f"{dataset_base}_{split}" / "data.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            print(f"Loaded {split}: {len(data)} entries")
            all_data.extend(data)
        else:
            print(f"Skipping {split}: {path} not found")

    print(f"Total entries: {len(all_data)}")

    # Auto-detect modality
    modality = args.modality
    if modality == "auto":
        modality = "vision" if all_data and "images" in all_data[0] else "text"
    print(f"Modality: {modality}")

    # Process entries in parallel
    print(f"Counting tokens with {args.workers} workers...")
    with Pool(args.workers, initializer=init_worker, initargs=(args.model,)) as pool:
        results = list(tqdm(
            pool.imap(process_entry, all_data, chunksize=50),
            total=len(all_data),
            desc="Processing",
        ))

    # Add modality column
    for r in results:
        r["modality"] = modality

    # Save CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset_base}.csv"
    fieldnames = ["submission_id", "year", "conference", "modality", "num_pages", "text_tokens", "vision_tokens", "total_tokens"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {out_path}")

    # Print summary
    import numpy as np
    years = sorted(set(r["year"] for r in results))
    print(f"\n{'Year':<6} {'Count':>6} {'TextTok mean':>12} {'TextTok min':>11} {'TextTok max':>11} {'VisTok mean':>11} {'Pages mean':>10}")
    print("-" * 80)
    for year in years:
        yr_data = [r for r in results if r["year"] == year]
        tt = [r["text_tokens"] for r in yr_data]
        vt = [r["vision_tokens"] for r in yr_data]
        pg = [r["num_pages"] for r in yr_data]
        print(f"{year:<6} {len(yr_data):>6} {np.mean(tt):>12.0f} {min(tt):>11} {max(tt):>11} {np.mean(vt):>11.0f} {np.mean(pg):>10.1f}")


if __name__ == "__main__":
    main()
