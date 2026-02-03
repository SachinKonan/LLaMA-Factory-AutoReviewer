# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Classification inference script using ModelForBinaryClassification.
Outputs: {"logit": ..., "prob": ..., "pred": ..., "label": ..., "_metadata": ...}

Usage:
    python scripts/cls_infer.py \
        --model_name_or_path saves/cls_model \
        --dataset iclr_2020_2025_test \
        --save_name results/cls/finetuned.jsonl
"""

import json
import os

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import FinetuningArguments


def cls_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 24480,
    max_samples: int | None = None,
    save_name: str = "predictions.jsonl",
    batch_size: int = 8,
    positive_token: str = "Accept",
    negative_token: str = "Reject",
    default_system: str | None = None,
):
    r"""Perform batch classification inference using a trained classification model.

    Usage: python cls_infer.py --model_name_or_path saves/cls_model --dataset test_dataset
    """
    # Map common template aliases to valid names
    template_aliases = {
        "qwen2.5": "qwen",
        "qwen2": "qwen",
    }
    if template in template_aliases:
        print(f"Mapping template '{template}' -> '{template_aliases[template]}'")
        template = template_aliases[template]

    model_args, data_args, finetuning_args, _ = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
        )
    )

    # Set stage for loading
    finetuning_args.stage = "cls"

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Load model with binary classification head
    print(f"Loading model from {model_name_or_path}...")
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_binary_cls=True)
    model.eval()
    device = next(model.parameters()).device

    # Load dataset
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "cls", **tokenizer_module)
    eval_dataset = dataset_module.get("eval_dataset") or dataset_module.get("train_dataset")

    if eval_dataset is None:
        raise ValueError("No dataset found. Please check dataset configuration.")

    print(f"Loaded {len(eval_dataset)} examples for inference.")

    # Create output directory if needed
    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)

    # Get metadata if available
    metadata_list = None
    if "_metadata" in eval_dataset.column_names:
        metadata_list = eval_dataset["_metadata"]

    # Custom collate function for inference
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for item in batch:
            input_ids = item["input_ids"]
            pad_len = max_len - len(input_ids)

            if tokenizer.padding_side == "right":
                input_ids_padded = input_ids + [tokenizer.pad_token_id] * pad_len
                attention_mask = [1] * len(input_ids) + [0] * pad_len
            else:
                input_ids_padded = [tokenizer.pad_token_id] * pad_len + input_ids
                attention_mask = [0] * pad_len + [1] * len(input_ids)

            input_ids_batch.append(input_ids_padded)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(item["labels"])

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": labels_batch,
        }

    dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    results = []
    sample_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            # Forward pass through our classifier
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            # Convert to predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            for i in range(len(preds)):
                result = {
                    "logit": round(logits[i].item(), 4),
                    "prob": round(probs[i].item(), 4),
                    "pred": preds[i].item(),
                    "label": int(labels[i]),
                }
                # Add metadata if available
                if metadata_list is not None and sample_idx < len(metadata_list):
                    result["_metadata"] = metadata_list[sample_idx]
                results.append(result)
                sample_idx += 1

    # Write results
    with open(save_name, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Compute and print metrics
    correct = sum(1 for r in results if r["pred"] == r["label"])
    accuracy = correct / len(results) if results else 0.0

    tp = sum(1 for r in results if r["pred"] == 1 and r["label"] == 1)
    fp = sum(1 for r in results if r["pred"] == 1 and r["label"] == 0)
    fn = sum(1 for r in results if r["pred"] == 0 and r["label"] == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("*" * 70)
    print(f"{len(results)} predictions saved to {save_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(cls_infer)
