#!/bin/bash
#SBATCH --job-name=test_train_eval
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=350G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/final_sweep_v7/%j.out
#SBATCH --error=logs/final_sweep_v7/%j.err

# Test script for eval_training_ckpt.py
# Tests the offline training metrics computation on a specific checkpoint

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

# Test configuration - modify these for your checkpoint
CHECKPOINT="/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/saves/final_sweep_v7/balanced_trainagreeing_no2024_vision_lr1e-6/checkpoint-4278"
DATASET="iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7"
RESULTS_DIR="results/final_sweep_v7/balanced_trainagreeing_no2024_vision_lr1e-6"

echo "=============================================="
echo "Test: Eval Training Checkpoint"
echo "=============================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET}_train"
echo "Results: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p "${RESULTS_DIR}" logs/final_sweep_v7

python scripts/eval_training_ckpt.py \
    --model_name_or_path "${CHECKPOINT}" \
    --dataset "${DATASET}_train" \
    --template qwen2_vl \
    --cutoff_len 24480 \
    --image_max_pixels 1003520 \
    --image_min_pixels 784 \
    --per_device_eval_batch_size 1 \
    --save_name "${RESULTS_DIR}/train-ckpt-4278-test.json" \
    --test_dataset "${DATASET}_test" \
    --max_samples 100

echo ""
echo "=============================================="
echo "Test Complete"
echo "=============================================="
