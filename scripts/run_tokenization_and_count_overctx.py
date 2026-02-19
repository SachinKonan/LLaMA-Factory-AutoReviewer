#!/usr/bin/env python3
"""
Count tokens in a dataset.

Two modes:
  - Fast (default): dimension-based approximation for image tokens + flat overhead
  - Exact (--exact): uses LLaMA-Factory's actual preprocessing pipeline

Usage:
    python scripts/run_tokenization_and_count_overctx.py \
        --config configs/final_sweep_v7_vision.yaml \
        --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_v7_filtered \
        --cutoff_len 26480

    # Exact mode with filtering
    python scripts/run_tokenization_and_count_overctx.py \
        --config configs/final_sweep_v7_vision.yaml \
        --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_v7_filtered \
        --exact --cutoff_len 26480 --filter --filter_cutoff 24480
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from functools import partial

import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer


# ==============================================================================
# Fast (approximation) mode
# ==============================================================================

@dataclass
class TokenCountConfig:
    """Config for token counting."""
    cutoff_len: int
    media_dir: str
    image_min_pixels: int
    image_max_pixels: int
    factor: int = 28
    patch_size: int = 14
    merge_size: int = 2


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """Qwen2VL smart_resize."""
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


def get_image_tokens_fast(image_path: str, config: TokenCountConfig) -> int:
    """Calculate image tokens from dimensions only (fast, no pixel loading)."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size

        resized_h, resized_w = smart_resize(
            height, width,
            factor=config.factor,
            min_pixels=config.image_min_pixels,
            max_pixels=config.image_max_pixels,
        )

        grid_h = resized_h // config.patch_size
        grid_w = resized_w // config.patch_size
        merge_length = config.merge_size**2
        tokens = (grid_h * grid_w) // merge_length
        tokens += 2  # vision_start + vision_end
        return tokens
    except Exception as e:
        print(f"Warning: Failed to get image dimensions for {image_path}: {e}")
        return 0


def count_sample_tokens_fast(sample: dict, tokenizer, config: TokenCountConfig) -> int:
    """Count total tokens for a sample (fast approximation)."""
    text_tokens = 0
    for conv in sample.get("conversations", []):
        content = conv.get("value", "")
        content_no_images = content.replace("<image>", "")
        tokens = tokenizer.encode(content_no_images, add_special_tokens=False)
        text_tokens += len(tokens)

    image_tokens = 0
    for img_path in sample.get("images", []):
        image_tokens += get_image_tokens_fast(img_path, config)

    special_tokens = 10
    return text_tokens + image_tokens + special_tokens


# Fast mode workers
_fast_tokenizer = None
_fast_config = None


def init_fast_worker(tokenizer_name: str, config_dict: dict):
    global _fast_tokenizer, _fast_config
    _fast_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    _fast_config = TokenCountConfig(**config_dict)


def fast_worker(sample: dict) -> int:
    global _fast_tokenizer, _fast_config
    try:
        return count_sample_tokens_fast(sample, _fast_tokenizer, _fast_config)
    except Exception as e:
        print(f"Warning: Failed to process sample: {e}")
        return -1


# ==============================================================================
# Exact mode (LLaMA-Factory pipeline with fast image grid computation)
# ==============================================================================

def _simulate_plugin_preprocess(w: int, h: int, max_pixels: int, min_pixels: int) -> tuple[int, int]:
    """Simulate BasePlugin._preprocess_image + Qwen2VLPlugin._preprocess_image.

    Returns (width, height) after plugin preprocessing (before image_processor).
    """
    # BasePlugin: proportional resize if outside pixel range
    if w * h > max_pixels:
        factor = math.sqrt(max_pixels / (w * h))
        w, h = int(w * factor), int(h * factor)
    if w * h < min_pixels:
        factor = math.sqrt(min_pixels / (w * h))
        w, h = int(w * factor), int(h * factor)

    # Qwen2VLPlugin: ensure min dimension >= 28
    if min(w, h) < 28:
        w, h = max(w, 28), max(h, 28)
    # Qwen2VLPlugin: clamp extreme aspect ratios
    if w / max(h, 1) > 200:
        w = h * 180
    if h / max(w, 1) > 200:
        h = w * 180

    return w, h


def _fast_get_mm_inputs(plugin_self, images, videos, audios, processor):
    """Fast replacement for _get_mm_inputs that computes image_grid_thw
    from image dimensions only (no pixel loading).

    Simulates the exact two-step process:
    1. Plugin preprocessing (resize to processor's pixel range)
    2. Image processor's smart_resize (grid-snap to processor's own pixel range)
    """
    import torch

    image_processor = getattr(processor, "image_processor")
    mm_inputs = {}

    if len(images) != 0:
        # Processor-level pixel limits (patched by LLaMA-Factory)
        plugin_max_px = getattr(processor, "image_max_pixels", 768 * 768)
        plugin_min_px = getattr(processor, "image_min_pixels", 32 * 32)

        # Image processor's own smart_resize limits
        proc_min = getattr(image_processor, "min_pixels", 3136)
        proc_max = getattr(image_processor, "max_pixels", 12845056)

        image_grid_thw = []
        for image in images:
            # Get dimensions (header-only, no pixel loading)
            if isinstance(image, str):
                with Image.open(image) as img:
                    w, h = img.size
            else:
                w, h = image.size

            # Step 1: simulate plugin preprocessing
            w, h = _simulate_plugin_preprocess(w, h, plugin_max_px, plugin_min_px)

            # Step 2: image processor's smart_resize
            rh, rw = smart_resize(h, w, factor=28, min_pixels=proc_min, max_pixels=proc_max)

            grid_t = 1
            grid_h = rh // 14
            grid_w = rw // 14
            image_grid_thw.append([grid_t, grid_h, grid_w])

        mm_inputs["image_grid_thw"] = torch.tensor(image_grid_thw)

    return mm_inputs


def count_sample_tokens_exact(sample: dict, tokenizer, processor, template) -> int:
    """Count exact tokens using LLaMA-Factory's template encoding pipeline.

    Uses fast dimension-based image grid computation (monkey-patched _get_mm_inputs)
    with the real template encoding for exact text tokenization.
    Replicates SupervisedDatasetProcessor._encode_data_example without truncation.
    """
    conversations = sample.get("conversations", [])
    image_paths = sample.get("images", [])

    # Extract system message and convert sharegpt to prompt/response
    system = None
    messages = []
    for conv in conversations:
        role_tag = conv["from"]
        if role_tag in ("system",):
            system = conv["value"]
        elif role_tag in ("human", "user"):
            messages.append({"role": "user", "content": conv["value"]})
        else:  # gpt, assistant
            messages.append({"role": "assistant", "content": conv["value"]})

    if len(messages) < 2:
        # Need at least user + assistant
        return -1

    # Split: prompt = all but last (user turns), response = last (assistant)
    prompt = messages[:-1]
    response = messages[-1:]

    images = list(image_paths)

    # Step 1: process_messages with monkey-patched _get_mm_inputs
    processed = template.mm_plugin.process_messages(
        prompt + response, images, [], [], processor
    )

    # Step 2: process_token_ids (returns empty for Qwen2VL)
    input_ids, labels = template.mm_plugin.process_token_ids(
        [], [], images, [], [], tokenizer, processor
    )

    # Step 3: encode_multiturn — exact template tokenization
    # Use extracted system message, fallback to template default
    if system is None:
        system = template.default_system
    encoded_pairs = template.encode_multiturn(tokenizer, processed, system, None)

    # Step 4: count total (same logic as _encode_data_example, but NO truncation)
    total = len(input_ids)
    if template.efficient_eos:
        total += 1

    for source_ids, target_ids in encoded_pairs:
        total += len(source_ids) + len(target_ids)

    return total


# Exact mode workers
_exact_tokenizer = None
_exact_processor = None
_exact_template = None


def init_exact_worker(model_name: str, template_name: str, image_min_pixels: int, image_max_pixels: int):
    global _exact_tokenizer, _exact_processor, _exact_template

    # Add LLaMA-Factory src to path
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from transformers import AutoProcessor
    from llamafactory.data.template import TEMPLATES

    _exact_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _exact_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Patch processor with pixel limits (same as LLaMA-Factory's patch_processor)
    _exact_processor.image_max_pixels = image_max_pixels
    _exact_processor.image_min_pixels = image_min_pixels

    _exact_template = TEMPLATES[template_name]

    # Monkey-patch _get_mm_inputs to skip expensive pixel loading
    import types
    _exact_template.mm_plugin._get_mm_inputs = types.MethodType(_fast_get_mm_inputs, _exact_template.mm_plugin)


def exact_worker(sample: dict) -> int:
    global _exact_tokenizer, _exact_processor, _exact_template
    try:
        return count_sample_tokens_exact(sample, _exact_tokenizer, _exact_processor, _exact_template)
    except Exception as e:
        print(f"Warning: Failed to process sample: {e}")
        return -1


# ==============================================================================
# Common processing functions
# ==============================================================================

def process_dataset_parallel(
    data: list[dict],
    num_workers: int,
    init_fn,
    init_args: tuple,
    worker_fn,
    desc: str = "Processing",
) -> list[int]:
    """Process all samples in parallel."""
    with Pool(num_workers, initializer=init_fn, initargs=init_args) as pool:
        counts = list(
            tqdm(
                pool.imap(worker_fn, data, chunksize=50),
                total=len(data),
                desc=desc,
            )
        )
    return counts


def process_dataset_single(
    data: list[dict],
    count_fn,
    desc: str = "Processing",
) -> list[int]:
    """Process all samples single-threaded."""
    counts = []
    for sample in tqdm(data, desc=desc):
        try:
            counts.append(count_fn(sample))
        except Exception as e:
            print(f"Warning: Failed to process sample: {e}")
            counts.append(-1)
    return counts


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Count tokens in dataset")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--filter", action="store_true", help="Create filtered dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (without _train/_test)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers (default: 16)")
    parser.add_argument("--single-thread", action="store_true", help="Single-threaded (for debugging)")
    parser.add_argument("--cutoff_len", type=int, default=None, help="Override cutoff_len from config")
    parser.add_argument("--filter_cutoff", type=int, default=None,
                        help="Separate cutoff for --filter (default: same as cutoff_len)")
    parser.add_argument("--exact", action="store_true",
                        help="Use LLaMA-Factory's exact preprocessing pipeline (slower but precise)")
    parser.add_argument("--template", type=str, default=None,
                        help="Template name for --exact mode (default: from config or 'qwen2_vl')")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")

    # Extract parameters
    model_name = config["model_name_or_path"]
    cutoff_len = args.cutoff_len if args.cutoff_len is not None else config["cutoff_len"]
    filter_cutoff = args.filter_cutoff if args.filter_cutoff is not None else cutoff_len
    media_dir = config.get("media_dir", "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/images")
    image_min_pixels = config.get("image_min_pixels", 784)
    image_max_pixels = config.get("image_max_pixels", 156800)
    template_name = args.template or config.get("template", "qwen2_vl")
    dataset_name = args.dataset or config.get("dataset", "placeholder")

    if dataset_name == "placeholder":
        print("Error: dataset is 'placeholder' and --dataset not provided")
        return

    dataset_base = dataset_name.replace("_train", "").replace("_test", "").replace("_validation", "")

    mode = "EXACT (LLaMA-Factory pipeline)" if args.exact else "FAST (dimension-based approximation)"
    print(f"\n{'='*60}")
    print(f"Token Counting — {mode}")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Cutoff: {cutoff_len} tokens")
    if args.filter and filter_cutoff != cutoff_len:
        print(f"Filter cutoff: {filter_cutoff} tokens")
    print(f"Image pixels: {image_min_pixels} - {image_max_pixels}")
    print(f"Dataset: {dataset_base}")
    if args.exact:
        print(f"Template: {template_name}")
    print(f"Workers: {args.workers if not args.single_thread else 1}")
    print(f"{'='*60}\n")

    # Load datasets (train, test, and optionally validation)
    splits = {}
    for split_name in ["train", "test", "validation"]:
        path = f"data/{dataset_base}_{split_name}/data.json"
        if os.path.exists(path):
            print(f"Loading {split_name} data from: {path}")
            with open(path) as f:
                splits[split_name] = json.load(f)
            print(f"  {split_name} samples: {len(splits[split_name])}")
        elif split_name in ("train", "test"):
            print(f"ERROR: {path} not found")
            return

    print()

    # Count tokens for each split
    split_counts = {}
    for split_name, data in splits.items():
        if args.exact:
            # Exact mode: use LLaMA-Factory pipeline
            if args.single_thread:
                # Initialize once for single-thread
                import types
                src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from transformers import AutoProcessor
                from llamafactory.data.template import TEMPLATES

                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                processor.image_max_pixels = image_max_pixels
                processor.image_min_pixels = image_min_pixels
                template = TEMPLATES[template_name]
                # Monkey-patch to skip expensive pixel loading
                template.mm_plugin._get_mm_inputs = types.MethodType(
                    _fast_get_mm_inputs, template.mm_plugin)

                count_fn = partial(count_sample_tokens_exact,
                                   tokenizer=tokenizer, processor=processor, template=template)
                split_counts[split_name] = process_dataset_single(
                    data, count_fn, split_name.capitalize()
                )
            else:
                split_counts[split_name] = process_dataset_parallel(
                    data, args.workers,
                    init_exact_worker, (model_name, template_name, image_min_pixels, image_max_pixels),
                    exact_worker, split_name.capitalize(),
                )
        else:
            # Fast mode: dimension-based approximation
            token_config = TokenCountConfig(
                cutoff_len=cutoff_len, media_dir=media_dir,
                image_min_pixels=image_min_pixels, image_max_pixels=image_max_pixels,
            )
            config_dict = {
                "cutoff_len": cutoff_len, "media_dir": media_dir,
                "image_min_pixels": image_min_pixels, "image_max_pixels": image_max_pixels,
            }
            if args.single_thread:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                count_fn = partial(count_sample_tokens_fast, tokenizer=tokenizer, config=token_config)
                split_counts[split_name] = process_dataset_single(
                    data, count_fn, split_name.capitalize()
                )
            else:
                split_counts[split_name] = process_dataset_parallel(
                    data, args.workers,
                    init_fast_worker, (model_name, config_dict),
                    fast_worker, split_name.capitalize(),
                )

    # Print results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")

    for split_name in splits:
        counts = split_counts[split_name]
        data = splits[split_name]
        n = len(data)
        errors = sum(1 for c in counts if c < 0)
        valid_counts = [c for c in counts if c >= 0]
        over = sum(1 for c in valid_counts if c > cutoff_len)
        max_tok = max(valid_counts) if valid_counts else 0
        mean_tok = sum(valid_counts) / len(valid_counts) if valid_counts else 0

        print(f"\n{split_name.capitalize()} split:")
        print(f"  Total samples: {n}")
        if errors:
            print(f"  Errors: {errors}")
        print(f"  Over cutoff ({cutoff_len}): {over} ({100*over/n:.1f}%)")
        print(f"  Under/equal cutoff: {n - over - errors} ({100*(n-over-errors)/n:.1f}%)")
        print(f"  Max tokens: {max_tok}")
        print(f"  Mean tokens: {mean_tok:.0f}")

        if filter_cutoff != cutoff_len:
            over_filter = sum(1 for c in valid_counts if c > filter_cutoff)
            print(f"  Over filter_cutoff ({filter_cutoff}): {over_filter} ({100*over_filter/n:.1f}%)")

    # Filter if requested
    if args.filter:
        print(f"\n{'='*60}")
        print(f"Creating filtered dataset (cutoff={filter_cutoff})")
        print(f"{'='*60}")

        filtered_name = f"{dataset_base}_filtered{filter_cutoff}"

        dataset_info_path = "data/dataset_info.json"
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)

        for split_name in splits:
            data = splits[split_name]
            counts = split_counts[split_name]
            # Keep samples that are under/equal filter_cutoff AND not errors
            filtered = [s for s, c in zip(data, counts) if 0 <= c <= filter_cutoff]

            split_dir = f"data/{filtered_name}_{split_name}"
            os.makedirs(split_dir, exist_ok=True)

            with open(f"{split_dir}/data.json", "w") as f:
                json.dump(filtered, f)
            print(f"Saved: {split_dir}/data.json ({len(filtered)} samples, was {len(data)})")

            orig_key = f"{dataset_base}_{split_name}"
            new_key = f"{filtered_name}_{split_name}"

            if orig_key in dataset_info:
                dataset_info[new_key] = dataset_info[orig_key].copy()
                dataset_info[new_key]["file_name"] = f"{filtered_name}_{split_name}/data.json"
            else:
                dataset_info[new_key] = {
                    "file_name": f"{filtered_name}_{split_name}/data.json",
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

        print(f"\nFiltered dataset: {filtered_name}")
        for split_name in splits:
            counts = split_counts[split_name]
            n_filtered = sum(1 for c in counts if 0 <= c <= filter_cutoff)
            print(f"  {split_name}: {n_filtered} samples")


if __name__ == "__main__":
    main()
