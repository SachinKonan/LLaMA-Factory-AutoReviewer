#!/bin/bash
# Resume blind text review generation — skips already completed, shards the rest
# Uses --partial_predictions to filter out completed submission_ids
#
# Usage:
#   bash scripts/sh/gemini_synthetic_reasoning/blind_text_batch_resume.sh [NUM_SHARDS]
#   Default: 10 shards

NUM_SHARDS=${1:-10}

PYTHONUNBUFFERED=1 uv run python scripts/gemini_batch_review.py \
    --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered \
    --mode blind \
    --model gemini-2.5-flash \
    --candidate_count 3 \
    --thinking_budget 256 \
    --temperature 1.0 \
    --partial_predictions data/gemini_text-predictions.jsonl \
    --partial_metadata results/reviews/blind_text_metadata.json \
    --num_shards "$NUM_SHARDS" \
    --submit_only \
    --output results/reviews/blind_text_resume.jsonl
