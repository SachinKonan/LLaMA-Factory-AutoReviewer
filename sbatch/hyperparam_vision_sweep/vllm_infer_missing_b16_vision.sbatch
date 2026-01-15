#!/bin/bash
#SBATCH --job-name=infer_b16_vis
#SBATCH --time=09:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --array=0-1
#SBATCH --output=logs/hyperparam_vision_sweep/%A_%a_infer.out
#SBATCH --error=logs/hyperparam_vision_sweep/%A_%a_infer.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 2 missing vision batch16 models
CONFIGS=(
    "bs16_proj_frozen_vision"
    "bs16_proj_unfrozen_vision"
)

CONFIG_NAME="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL_DIR="saves/qwen2.5-vl-7b/full/hyperparam_vision_sweep/${CONFIG_NAME}"
RESULTS_DIR="results/hyperparam_vision_sweep/${CONFIG_NAME}"
DATASET="iclr_vision_binary_no_reviews_v3"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Config: ${CONFIG_NAME}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p "${RESULTS_DIR}"

CKPT_DIR=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | head -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "Using checkpoint: ${CKPT_DIR}"

python scripts/vllm_infer.py \
    --model_name_or_path "${CKPT_DIR}" \
    --dataset "${DATASET}_test" \
    --template qwen2_vl \
    --cutoff_len 20480 \
    --max_new_tokens 10240 \
    --image_min_pixels 784 \
    --image_max_pixels 1003520 \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${CONFIG_NAME}"
echo "Model: ${CKPT_DIR}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
