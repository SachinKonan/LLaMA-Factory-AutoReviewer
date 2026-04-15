#!/usr/bin/env python3
"""
Step 2: Compute decision confidence from text model predictions.

Reads text_predictions.jsonl (output of vllm_infer.py), finds the Accept/Reject
decision token in each prediction, extracts its log-probability, and writes
confidence.jsonl.

Usage:
    python alignment_analysis/compute_decision_confidence.py \
        --input alignment_analysis/results/text_predictions.jsonl \
        --output alignment_analysis/results/confidence.jsonl \
        --model_name_or_path /scratch/gpfs/ZHUANGL/jl0796/shared/saves/best_2025_2026_text/checkpoint-1322
"""

import argparse
import json
import math
import re
import os

from transformers import AutoTokenizer


def find_decision_token_index(predict: str, token_logprobs: list, tokenizer) -> tuple[str, int, float]:
    """
    Find the Accept/Reject decision token and its logprob.

    Returns (decision, token_index, logprob).
    decision is "Accept" or "Reject", or None if not found.
    """
    # Search for Accept or Reject in the predicted text
    m = re.search(r"\b(Accept|Reject)\b", predict)
    if m is None:
        return None, None, None

    decision = m.group(1)
    decision_char_start = m.start(1)

    # Tokenize the prediction prefix up to the decision token to find its index
    prefix = predict[:decision_char_start]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    token_idx = len(prefix_ids)

    if token_idx >= len(token_logprobs):
        # Fallback: just use the first non-None logprob in the prediction
        for i, lp in enumerate(token_logprobs):
            if lp is not None:
                return decision, i, lp
        return decision, None, None

    lp = token_logprobs[token_idx]
    return decision, token_idx, lp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="alignment_analysis/results/text_predictions.jsonl")
    parser.add_argument("--output", default="alignment_analysis/results/confidence.jsonl")
    parser.add_argument(
        "--model_name_or_path",
        default="/scratch/gpfs/ZHUANGL/jl0796/shared/saves/best_2025_2026_text/checkpoint-1322",
    )
    parser.add_argument(
        "--dataset_json",
        default="/scratch/gpfs/ZHUANGL/jl0796/shared/data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
        help="Path to the original dataset JSON to extract submission_ids from."
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    submission_ids = []
    submission_years = []
    if args.dataset_json and os.path.exists(args.dataset_json):
        print(f"Loading metadata from {args.dataset_json}")
        with open(args.dataset_json, "r") as f:
            data = json.load(f)
            for entry in data:
                metadata = entry.get("_metadata", {})
                submission_ids.append(metadata.get("submission_id"))
                submission_years.append(metadata.get("year"))

    n_total = n_found = n_missing_id = 0

    with open(args.input) as f_in, open(args.output, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            n_total += 1

            submission_id = entry.get("submission_id")
            year = entry.get("year")
            
            if not submission_id and n_total - 1 < len(submission_ids):
                submission_id = submission_ids[n_total - 1]
            if not year and n_total - 1 < len(submission_years):
                year = submission_years[n_total - 1]
            
            if not submission_id:
                submission_id = f"sample_{n_total - 1:05d}"
                n_missing_id += 1

            predict = entry.get("predict", "")
            token_logprobs = entry.get("token_logprobs", [])
            label = entry.get("label", None)

            decision, tok_idx, logprob = find_decision_token_index(predict, token_logprobs, tokenizer)

            if decision is None:
                print(f"  WARNING: No Accept/Reject found in prediction for {submission_id}: {predict!r}")
                out = {
                    "submission_id": submission_id,
                    "decision": None,
                    "decision_logit": None,
                    "label": label,
                    "year": year,
                    "predict": predict,
                }
            else:
                n_found += 1
                prob = math.exp(logprob)
                accept_prob = prob if decision == "Accept" else 1.0 - prob
                
                out = {
                    "submission_id": submission_id,
                    "decision": decision,
                    "decision_logit": logprob,
                    "accept_probability": accept_prob,
                    "label": label,
                    "year": year,
                    "decision_token_idx": tok_idx,
                    "predict": predict,
                }

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Done. {n_total} entries processed, {n_found} with decision found.")
    if n_missing_id:
        print(f"  WARNING: {n_missing_id} entries had no submission_id — used sequential fallback.")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
