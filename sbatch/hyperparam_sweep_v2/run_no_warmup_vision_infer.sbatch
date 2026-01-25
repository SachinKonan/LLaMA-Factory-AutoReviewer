#!/bin/bash
#SBATCH --job-name=hp_nowarmup_infer
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --array=0-2
#SBATCH --output=logs/hyperparam_iclr20_balanced/%A_%a_infer.out
#SBATCH --error=logs/hyperparam_iclr20_balanced/%A_%a_infer.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 3 checkpoints to evaluate
CHECKPOINTS=(1072 2144 3216)
CKPT_NUM=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

DATASET="iclr_2020_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6"
SHORT_NAME="iclr20bal_lr2.0e-6_b16_e4_w0_vision"
TEMPLATE="qwen2_vl"
IMAGE_MAX_PIXELS=1003520

MODEL_DIR="saves/hyperparam_iclr20_balanced/${SHORT_NAME}"
CKPT_DIR="${MODEL_DIR}/checkpoint-${CKPT_NUM}"
RESULTS_DIR="results/hyperparam_iclr20_balanced/${SHORT_NAME}/checkpoint-${CKPT_NUM}"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "Checkpoint: ${CKPT_NUM}"
echo "Model Dir: ${CKPT_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p "${RESULTS_DIR}"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT_DIR}"
    exit 1
fi

echo "Using checkpoint: ${CKPT_DIR}"

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
echo "COMPLETED: ${SHORT_NAME} checkpoint-${CKPT_NUM}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
