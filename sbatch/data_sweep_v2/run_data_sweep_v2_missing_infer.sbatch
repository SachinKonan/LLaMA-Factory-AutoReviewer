#!/bin/bash
#SBATCH --job-name=dsv2_infer
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-test
#SBATCH --array=0-1
#SBATCH --output=logs/data_sweep_v2/%A_%a_infer.out
#SBATCH --error=logs/data_sweep_v2/%A_%a_infer.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 2 configurations: indexes 3,5 (vision models that OOMed during inference)
# Format: "DATASET_BASE SHORT_NAME"
CONFIGS=(
    "iclr_2020_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6 iclr20_balanced"
    "iclr_2020_2025_85_5_10_split6_balanced_trainagreeing_vision_binary_noreviews_v6 iclr20_trainagreeing"
)

read DATASET SHORT_NAME <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODALITY="vision"
TEMPLATE="qwen2_vl"
IMAGE_MAX_PIXELS=1003520

MODEL_DIR="saves/data_sweep_v2/${SHORT_NAME}_${MODALITY}"
RESULTS_DIR="results/data_sweep_v2/${SHORT_NAME}_${MODALITY}"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}_${MODALITY}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p "${RESULTS_DIR}"

# Find the latest checkpoint
CKPT_DIR=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "Using checkpoint: ${CKPT_DIR}"

# Run inference on TEST set
python scripts/vllm_infer.py \
    --model_name_or_path "${CKPT_DIR}" \
    --dataset "${DATASET}_test" \
    --template "${TEMPLATE}" \
    --cutoff_len 24480 \
    --max_new_tokens 1280 \
    --image_min_pixels 784 \
    --image_max_pixels "${IMAGE_MAX_PIXELS}" \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}_${MODALITY}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="