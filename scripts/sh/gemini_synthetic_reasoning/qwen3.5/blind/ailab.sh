#!/bin/bash
# Blind text review — Qwen3.5-122B-A10B — ailab partition (shards 0-14)
# Usage: sbatch scripts/sh/gemini_synthetic_reasoning/qwen3.5/blind/ailab.sh

#SBATCH --job-name=qwen-blind-ailab
#SBATCH --array=0-14
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=5
#SBATCH --output=logs/qwen_review/blind/%A_%a.out
#SBATCH --error=logs/qwen_review/blind/%A_%a.err

set -euo pipefail
mkdir -p logs/qwen_review/blind results/reviews
source /scratch/gpfs/ZHUANGL/sk7524/Sky-qwen3.5/.venv/bin/activate

MODEL_PATH="/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99"
DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
MODE="blind"
WORKERS=5

echo "=== Qwen Blind Text Review (ailab) ==="
echo "Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

export VLLM_ENGINE_READY_TIMEOUT_S=1200
nohup vllm serve "$MODEL_PATH" \
    -dp 2 --enable-expert-parallel --language-model-only \
    --reasoning-parser qwen3 --enable-prefix-caching \
    --max-model-len 20480 --max-num-batched-tokens 4096 \
    > "logs/qwen_review/${MODE}/vllm_shard${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

python scripts/qwen_batch_review.py \
    --dataset "$DATASET" --split train test --mode $MODE \
    --candidate_count 3 --api_base http://localhost:8000 \
    --workers $WORKERS --num_shards 15 --shard_subset "0:11700" \
    --shard_prefix ailab --wait_for_server 900 \
    --output "results/reviews/qwen_${MODE}_text.jsonl"

echo "End: $(date)"
