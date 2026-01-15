#!/usr/bin/env python3
"""
Count tokens in a dataset using fast dimension-based calculation.

For Qwen2VL, image tokens are calculated directly from image dimensions
using the smart_resize formula, which is much faster than loading images.

Usage:
    uv run python scripts/run_tokenization_and_count_overctx.py \
        --config configs/sweep_final_clean_images.yaml

    # With filtering
    uv run python scripts/run_tokenization_and_count_overctx.py \
        --config configs/sweep_final_clean_images.yaml \
        --filter
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from functools import partial

import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class TokenCountConfig:
    """Config for token counting."""

    cutoff_len: int
    media_dir: str
    image_min_pixels: int
    image_max_pixels: int
    # Qwen2VL constants
    factor: int = 28  # patch_size * merge_size
    patch_size: int = 14
    merge_size: int = 2


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """Qwen2VL smart_resize - rescales image to meet constraints.

    1. Both dimensions divisible by 'factor' (28)
    2. Total pixels within [min_pixels, max_pixels]
    3. Aspect ratio maintained
    """
    if max(height, width) / min(height, width) > 200:
        # Clamp extreme aspect ratios
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


def get_image_tokens_fast(image_path: str, config: TokenCountConfig) -> int:
    """Calculate image tokens from dimensions only (fast, no pixel loading)."""
    try:
        # PIL.Image.open only reads headers, not pixel data
        with Image.open(image_path) as img:
            width, height = img.size

        # Apply smart_resize to get final dimensions
        resized_h, resized_w = smart_resize(
            height,
            width,
            factor=config.factor,
            min_pixels=config.image_min_pixels,
            max_pixels=config.image_max_pixels,
        )

        # Calculate grid dimensions
        grid_h = resized_h // config.patch_size
        grid_w = resized_w // config.patch_size
        grid_t = 1  # temporal dimension = 1 for static images

        # Token count = grid product / merge_length
        merge_length = config.merge_size**2  # = 4
        tokens = (grid_t * grid_h * grid_w) // merge_length

        # Add vision tokens (start/end): <|vision_start|> + image_tokens + <|vision_end|>
        tokens += 2

        return tokens
    except Exception as e:
        print(f"Warning: Failed to get image dimensions for {image_path}: {e}")
        return 0


def count_sample_tokens(
    sample: dict,
    tokenizer,
    config: TokenCountConfig,
) -> int:
    """Count total tokens for a sample (text + images)."""
    # Count text tokens
    text_tokens = 0
    for conv in sample.get("conversations", []):
        content = conv.get("value", "")
        # Remove <image> placeholders for text token counting
        content_no_images = content.replace("<image>", "")
        tokens = tokenizer.encode(content_no_images, add_special_tokens=False)
        text_tokens += len(tokens)

    # Count image tokens
    image_tokens = 0
    images = sample.get("images", [])
    for img_path in images:
        # Resolve relative paths - paths in dataset are relative to project root
        # e.g., "data/images/xxx/yyy.jpg"
        if not os.path.isabs(img_path):
            # Just use as relative path from current working directory
            pass
        image_tokens += get_image_tokens_fast(img_path, config)

    # Add special tokens overhead (BOS, chat template, etc.)
    special_tokens = 10

    return text_tokens + image_tokens + special_tokens


# Global variables for worker processes
_worker_tokenizer = None
_worker_config = None


def init_worker(tokenizer_name: str, config_dict: dict):
    """Initialize worker with tokenizer (called once per worker)."""
    global _worker_tokenizer, _worker_config
    _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    _worker_config = TokenCountConfig(**config_dict)


def process_sample_worker(sample: dict) -> int:
    """Process a single sample using pre-initialized tokenizer."""
    global _worker_tokenizer, _worker_config
    try:
        return count_sample_tokens(sample, _worker_tokenizer, _worker_config)
    except Exception as e:
        print(f"Warning: Failed to process sample: {e}")
        return _worker_config.cutoff_len + 1


def process_dataset_parallel(
    data: list[dict],
    tokenizer_name: str,
    config: TokenCountConfig,
    num_workers: int = 16,
    desc: str = "Processing",
) -> list[int]:
    """Process all samples in parallel and return token counts."""
    config_dict = {
        "cutoff_len": config.cutoff_len,
        "media_dir": config.media_dir,
        "image_min_pixels": config.image_min_pixels,
        "image_max_pixels": config.image_max_pixels,
    }

    with Pool(
        num_workers,
        initializer=init_worker,
        initargs=(tokenizer_name, config_dict),
    ) as pool:
        counts = list(
            tqdm(
                pool.imap(process_sample_worker, data, chunksize=100),
                total=len(data),
                desc=desc,
            )
        )

    return counts


def process_dataset_single(
    data: list[dict],
    tokenizer,
    config: TokenCountConfig,
    desc: str = "Processing",
) -> list[int]:
    """Process all samples single-threaded (for debugging)."""
    counts = []
    for sample in tqdm(data, desc=desc):
        try:
            count = count_sample_tokens(sample, tokenizer, config)
            counts.append(count)
        except Exception as e:
            print(f"Warning: Failed to process sample: {e}")
            counts.append(config.cutoff_len + 1)
    return counts


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in dataset (fast dimension-based calculation)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config file (e.g., configs/sweep_final_clean_images.yaml)",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Create filtered dataset excluding over-length samples",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (without _train/_test suffix). Required if config has 'placeholder'.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )
    parser.add_argument(
        "--single-thread",
        action="store_true",
        help="Use single-threaded processing (for debugging)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")

    # Extract parameters
    model_name = config["model_name_or_path"]
    cutoff_len = config["cutoff_len"]
    media_dir = config.get(
        "media_dir",
        "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/images",
    )
    image_min_pixels = config.get("image_min_pixels", 784)
    image_max_pixels = config.get("image_max_pixels", 156800)
    dataset_name = args.dataset or config.get("dataset", "placeholder")

    # If dataset is still placeholder, error out
    if dataset_name == "placeholder":
        print("Error: dataset is 'placeholder' in config and --dataset not provided")
        print("Usage: --dataset iclr_2020_2025_..._v3 (without _train/_test)")
        return

    # Remove _train/_test suffix if present
    dataset_base = dataset_name.replace("_train", "").replace("_test", "")

    print(f"\n{'='*60}")
    print(f"Fast Token Counting (dimension-based)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Cutoff: {cutoff_len} tokens")
    print(f"Image pixels: {image_min_pixels} - {image_max_pixels}")
    print(f"Dataset: {dataset_base}")
    print(f"Workers: {args.workers if not args.single_thread else 1}")
    print(f"{'='*60}\n")

    # Create token counting config
    token_config = TokenCountConfig(
        cutoff_len=cutoff_len,
        media_dir=media_dir,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load datasets
    train_path = f"data/{dataset_base}_train/data.json"
    test_path = f"data/{dataset_base}_test/data.json"

    print(f"Loading train data from: {train_path}")
    with open(train_path) as f:
        train_data = json.load(f)

    print(f"Loading test data from: {test_path}")
    with open(test_path) as f:
        test_data = json.load(f)

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print()

    # Count tokens
    if args.single_thread:
        print("Counting tokens (single-threaded)...")
        train_counts = process_dataset_single(
            train_data, tokenizer, token_config, "Train"
        )
        test_counts = process_dataset_single(
            test_data, tokenizer, token_config, "Test"
        )
    else:
        print("Counting tokens (parallel)...")
        train_counts = process_dataset_parallel(
            train_data, model_name, token_config, args.workers, "Train"
        )
        test_counts = process_dataset_parallel(
            test_data, model_name, token_config, args.workers, "Test"
        )

    # Calculate statistics
    train_over = sum(1 for c in train_counts if c > cutoff_len)
    test_over = sum(1 for c in test_counts if c > cutoff_len)

    train_max = max(train_counts) if train_counts else 0
    test_max = max(test_counts) if test_counts else 0
    train_mean = sum(train_counts) / len(train_counts) if train_counts else 0
    test_mean = sum(test_counts) / len(test_counts) if test_counts else 0

    # Print results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")

    print(f"\nTrain split:")
    print(f"  Total samples: {len(train_data)}")
    print(
        f"  Over cutoff ({cutoff_len}): {train_over} ({100*train_over/len(train_data):.1f}%)"
    )
    print(
        f"  Under cutoff: {len(train_data) - train_over} ({100*(len(train_data)-train_over)/len(train_data):.1f}%)"
    )
    print(f"  Max tokens: {train_max}")
    print(f"  Mean tokens: {train_mean:.0f}")

    print(f"\nTest split:")
    print(f"  Total samples: {len(test_data)}")
    print(
        f"  Over cutoff ({cutoff_len}): {test_over} ({100*test_over/len(test_data):.1f}%)"
    )
    print(
        f"  Under cutoff: {len(test_data) - test_over} ({100*(len(test_data)-test_over)/len(test_data):.1f}%)"
    )
    print(f"  Max tokens: {test_max}")
    print(f"  Mean tokens: {test_mean:.0f}")

    # Filter if requested
    if args.filter:
        print(f"\n{'='*60}")
        print("Creating filtered dataset")
        print(f"{'='*60}")

        train_filtered = [
            s for s, c in zip(train_data, train_counts) if c <= cutoff_len
        ]
        test_filtered = [s for s, c in zip(test_data, test_counts) if c <= cutoff_len]

        # Create directories
        filtered_name = f"{dataset_base}_filtered{cutoff_len}"
        train_dir = f"data/{filtered_name}_train"
        test_dir = f"data/{filtered_name}_test"

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Save filtered data
        with open(f"{train_dir}/data.json", "w") as f:
            json.dump(train_filtered, f)
        print(f"Saved: {train_dir}/data.json ({len(train_filtered)} samples)")

        with open(f"{test_dir}/data.json", "w") as f:
            json.dump(test_filtered, f)
        print(f"Saved: {test_dir}/data.json ({len(test_filtered)} samples)")

        # Update dataset_info.json
        dataset_info_path = "data/dataset_info.json"
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)

        # Copy original dataset entry for train and test
        for split in ["train", "test"]:
            orig_key = f"{dataset_base}_{split}"
            new_key = f"{filtered_name}_{split}"

            if orig_key in dataset_info:
                dataset_info[new_key] = dataset_info[orig_key].copy()
                dataset_info[new_key]["file_name"] = f"{filtered_name}_{split}/data.json"
            else:
                # Create new entry
                dataset_info[new_key] = {
                    "file_name": f"{filtered_name}_{split}/data.json",
                    "formatting": "sharegpt",
                    "columns": {"messages": "conversations", "images": "images"},
                    "tags": {
                        "role_tag": "from",
                        "content_tag": "value",
                        "user_tag": "human",
                        "assistant_tag": "gpt",
                        "system_tag": "system",
                    },
                }

        with open(dataset_info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)
        print(f"Updated: {dataset_info_path}")

        print(f"\nFiltered dataset created: {filtered_name}")
        print(f"  Train: {len(train_filtered)} samples")
        print(f"  Test: {len(test_filtered)} samples")


if __name__ == "__main__":
    main()
