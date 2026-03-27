#!/bin/bash
# Blind text review generation via Qwen3.5-122B-A10B + Ray Data + vLLM
# Dataset: text (labelfix, filtered), train+test splits
# Model: Qwen3.5-122B-A10B, candidate_count=3, tensor_parallel_size=4
#
# Usage:
#   sbatch scripts/sh/gemini_synthetic_reasoning/qwen_blind_text_batch.sh

#SBATCH --job-name=qwen-review
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/qwen_review/%j.log

set -euo pipefail

# Create log directory if needed
mkdir -p logs/qwen_review

# Activate the venv with vLLM 0.16.1rc1 + Ray 2.54.0
source /scratch/gpfs/ZHUANGL/sk7524/Sky-qwen3.5/.venv/bin/activate

# Set the model path — UPDATE THIS to the actual location
MODEL_PATH="/path/to/Qwen3.5-122B-A10B"

echo "=== Qwen Blind Text Batch Review ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Model: $MODEL_PATH"
echo "Start: $(date)"
echo ""

python scripts/qwen_batch_review.py \
    --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered \
    --mode blind \
    --model "$MODEL_PATH" \
    --candidate_count 3 \
    --tensor_parallel_size 4 \
    --max_model_len 16384 \
    --temperature 1.0 \
    --output results/reviews/qwen_blind_text.jsonl

echo ""
echo "End: $(date)"
