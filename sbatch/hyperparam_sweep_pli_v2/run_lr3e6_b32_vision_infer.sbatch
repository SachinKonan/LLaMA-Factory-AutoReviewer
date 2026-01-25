#!/bin/bash
#SBATCH --job-name=hp_infer_b32_vision
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:2
#SBATCH --constraint="nomig&gpu80"
#SBATCH --output=logs/hyperparam_sweep_pli_v2/%A_lr3e6_b32_vision_infer.out
#SBATCH --error=logs/hyperparam_sweep_pli_v2/%A_lr3e6_b32_vision_infer.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Config: lr3.0e-6, batch32, vision (OOM rerun)
DATASET="iclr_2020_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6"
SHORT_NAME="lr3.0e-6_b32_vision"
TEMPLATE="qwen2_vl"
IMAGE_MAX_PIXELS=1003520

MODEL_DIR="saves/hyperparam_sweep_pli_v2/${SHORT_NAME}"
RESULTS_DIR="results/hyperparam_sweep_pli_v2/${SHORT_NAME}"

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
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
echo "COMPLETED: ${SHORT_NAME}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
