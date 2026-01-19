#!/usr/bin/env python3
"""
Extract accept/reject token probabilities from a trained checkpoint.

This script loads a model checkpoint and computes per-sample probabilities
for accept/reject decisions. Useful for:
- Post-hoc analysis of saved checkpoints
- Calibration analysis
- Confidence distribution visualization

Usage:
    python scripts/extract_token_probs.py \
        --checkpoint saves/qwen2.5vl-3b/full/sft_ds3/checkpoint-500 \
        --dataset iclr_2020_2025_85_5_10_split6_original_vision_binary_noreviews_v6_validation \
        --output results/probs_ckpt500.jsonl

Output format (JSONL):
    {"sample_id": 0, "p_accept": 0.72, "p_reject": 0.18, "gt": "accept", "pred": "accept", "correct": true}
"""

import argparse
import json
import os
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer


def get_token_ids(tokenizer, words: list[str]) -> list[int]:
    """Get token IDs for a list of words."""
    ids = set()
    for word in words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        ids.update(token_ids)
    return list(ids)


def find_answer_position(labels: torch.Tensor, accept_ids: list[int], reject_ids: list[int]) -> int:
    """Find the position where Accept/Reject token appears in labels.

    Args:
        labels: Label tensor for a single sample, shape (seq_len,)
        accept_ids: Token IDs for accept variants
        reject_ids: Token IDs for reject variants

    Returns:
        Position of the answer token, or -1 if not found
    """
    # Find valid (non-IGNORE_INDEX) positions from the end
    valid_mask = labels != IGNORE_INDEX
    valid_positions = torch.where(valid_mask)[0]

    if len(valid_positions) == 0:
        return -1

    # Search from the end for accept/reject tokens
    all_answer_ids = set(accept_ids + reject_ids)
    for pos in reversed(valid_positions.tolist()):
        if labels[pos].item() in all_answer_ids:
            return pos

    return -1


def extract_probs(
    model_name_or_path: str,
    dataset: str,
    dataset_dir: str = "data",
    template: str = "qwen2_vl",
    cutoff_len: int = 16384,
    max_samples: Optional[int] = None,
    output: str = "probs.jsonl",
    batch_size: int = 1,
):
    """Extract accept/reject probabilities from a checkpoint."""

    # Load model and tokenizer
    model_args, data_args, _, _ = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Get token IDs for accept/reject
    accept_ids = get_token_ids(tokenizer, ["Accept", "accept"])
    reject_ids = get_token_ids(tokenizer, ["Reject", "reject"])
    print(f"Accept token IDs: {accept_ids}")
    print(f"Reject token IDs: {reject_ids}")

    # Load dataset
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    train_dataset = dataset_module.get("train_dataset") or dataset_module.get("eval_dataset")

    if train_dataset is None:
        raise ValueError("No dataset found")

    print(f"Dataset size: {len(train_dataset)}")

    # Load model
    print(f"Loading model from {model_name_or_path}...")
    model = load_model(tokenizer, model_args, finetuning_args=None, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    # Process samples
    results = []
    from torch.utils.data import DataLoader

    # Simple collate function
    def collate_fn(batch):
        # batch is a list of dicts
        keys = batch[0].keys()
        collated = {}
        for key in keys:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                # Pad tensors
                max_len = max(v.size(0) for v in values)
                padded = []
                for v in values:
                    if v.size(0) < max_len:
                        if key == "labels":
                            pad_val = IGNORE_INDEX
                        else:
                            pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                        padding = torch.full((max_len - v.size(0),), pad_val, dtype=v.dtype)
                        v = torch.cat([v, padding])
                    padded.append(v)
                collated[key] = torch.stack(padded)
            else:
                collated[key] = values
        return collated

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    sample_id = 0
    for batch in tqdm(dataloader, desc="Extracting probabilities"):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        for i in range(logits.size(0)):
            sample_labels = labels[i] if labels is not None else None
            if sample_labels is None:
                sample_id += 1
                continue

            answer_pos = find_answer_position(sample_labels, accept_ids, reject_ids)
            if answer_pos < 0 or answer_pos == 0:
                sample_id += 1
                continue

            # Get logits at position before the answer token
            pred_logits = logits[i, answer_pos - 1, :]  # (vocab_size,)
            probs = F.softmax(pred_logits, dim=-1)

            # Sum probabilities for accept and reject token variants
            p_accept = sum(probs[tid].item() for tid in accept_ids if tid < probs.size(0))
            p_reject = sum(probs[tid].item() for tid in reject_ids if tid < probs.size(0))

            # Determine ground truth
            gt_token_id = sample_labels[answer_pos].item()
            if gt_token_id in accept_ids:
                gt = "accept"
            elif gt_token_id in reject_ids:
                gt = "reject"
            else:
                gt = "unknown"

            # Determine prediction
            pred = "accept" if p_accept > p_reject else "reject"
            correct = pred == gt

            result = {
                "sample_id": sample_id,
                "p_accept": round(p_accept, 6),
                "p_reject": round(p_reject, 6),
                "prob_mass": round(p_accept + p_reject, 6),
                "confidence": round(max(p_accept, p_reject), 6),
                "gt": gt,
                "pred": pred,
                "correct": correct,
            }
            results.append(result)
            sample_id += 1

    # Save results
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Print summary statistics
    if results:
        avg_p_accept = sum(r["p_accept"] for r in results) / len(results)
        avg_p_reject = sum(r["p_reject"] for r in results) / len(results)
        avg_prob_mass = sum(r["prob_mass"] for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        accuracy = sum(1 for r in results if r["correct"]) / len(results)

        print("\n" + "=" * 50)
        print("Summary Statistics")
        print("=" * 50)
        print(f"Total samples: {len(results)}")
        print(f"Average P(accept): {avg_p_accept:.4f}")
        print(f"Average P(reject): {avg_p_reject:.4f}")
        print(f"Average prob mass (P(accept)+P(reject)): {avg_prob_mass:.4f}")
        print(f"Average confidence: {avg_confidence:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Results saved to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Extract accept/reject token probabilities from a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (from dataset_info.json)")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Dataset directory")
    parser.add_argument("--template", type=str, default="qwen2_vl", help="Template name")
    parser.add_argument("--cutoff_len", type=int, default=16384, help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--output", type=str, default="probs.jsonl", help="Output file path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    extract_probs(
        model_name_or_path=args.checkpoint,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        template=args.template,
        cutoff_len=args.cutoff_len,
        max_samples=args.max_samples,
        output=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
