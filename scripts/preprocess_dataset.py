#!/usr/bin/env python3
"""
Preprocess and cache a dataset without loading the full model.
This is useful for vision datasets that take a long time to tokenize.

Usage:
    python scripts/preprocess_dataset.py \
        --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
        --dataset iclr_..._train \
        --eval_dataset iclr_..._test \
        --template qwen2_vl \
        --tokenized_path data/tokenized_cache/my_dataset \
        --cutoff_len 24480 \
        --image_min_pixels 784 \
        --image_max_pixels 1003520 \
        --preprocessing_num_workers 8
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llamafactory.hparams import get_train_args
from llamafactory.data import get_template_and_fix_tokenizer, get_dataset
from llamafactory.model import load_tokenizer


def main():
    # Parse arguments (reuses existing arg parsing)
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()

    # Ensure tokenized_path is set
    if data_args.tokenized_path is None:
        print("ERROR: --tokenized_path is required for preprocessing")
        sys.exit(1)

    print(f"=== Dataset Preprocessing ===")
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Dataset: {data_args.dataset}")
    print(f"Eval Dataset: {data_args.eval_dataset}")
    print(f"Output: {data_args.tokenized_path}")
    print(f"Num workers: {data_args.preprocessing_num_workers}")
    print()

    # Load tokenizer and processor (CPU only, no model)
    print("Loading tokenizer and processor...")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor")

    # Get template
    print("Setting up template...")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Preprocess dataset - this will save to tokenized_path
    print(f"Preprocessing dataset (this may take a while for vision data)...")
    print(f"  Training data: {data_args.dataset}")
    print(f"  Eval data: {data_args.eval_dataset}")

    # should_save is a property based on process_index and should_save_on_each_node
    # For single process, it should already be True

    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="cls",  # or "sft" - doesn't matter much for preprocessing
        **tokenizer_module
    )

    print()
    print(f"=== Preprocessing Complete ===")
    print(f"Tokenized dataset saved to: {data_args.tokenized_path}")
    print()
    print("To use in training, add to your command:")
    print(f"    --tokenized_path {data_args.tokenized_path}")


if __name__ == "__main__":
    main()
