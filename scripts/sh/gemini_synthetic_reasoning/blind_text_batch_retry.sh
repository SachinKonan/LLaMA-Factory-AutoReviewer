#!/bin/bash
# Retry failed blind text reviews — 100 papers per batch job
# 13,053 failed papers → ~131 jobs, each finishing well under 24h
#
# Usage:
#   bash scripts/sh/gemini_synthetic_reasoning/blind_text_batch_retry.sh

PYTHONUNBUFFERED=1 uv run python scripts/gemini_batch_review.py \
    --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered \
    --mode blind \
    --model gemini-2.5-flash \
    --candidate_count 3 \
    --thinking_budget 256 \
    --temperature 1.0 \
    --retry_file results/reviews/gemini_blind_failed_sids.txt \
    --num_shards 131 \
    --submit_only \
    --output results/reviews/blind_text_retry.jsonl
