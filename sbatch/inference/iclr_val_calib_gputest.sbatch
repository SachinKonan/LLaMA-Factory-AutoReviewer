#!/bin/bash
#SBATCH --job-name=iclr_val_calib
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=logs/auto_inference/iclr_val_calib_%j.out
#SBATCH --error=logs/auto_inference/iclr_val_calib_%j.err

# Inference of an ICLR ratio-trained text checkpoint on a chosen val ratio.
# Used for per-(train_ratio, val_ratio) threshold calibration applied to the
# test cross-eval matrix.
#
# Required env:
#   CHECKPOINT_DIR  - path to checkpoint
#   CKPT_STEP       - step number
#   TEMPLATE        - "qwen"
#   SHORT_NAME      - bz32_lr1e-6_text / _40_60 / _30_70
#   VAL_RATIO       - "balanced" | "40_60" | "30_70"
#   VAL_DS          - dataset name (looked up in dataset_info.json)

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p logs/auto_inference

for v in CHECKPOINT_DIR CKPT_STEP TEMPLATE SHORT_NAME VAL_RATIO VAL_DS; do
    if [ -z "${!v}" ]; then echo "ERROR: $v not set"; exit 2; fi
done

CUTOFF_LEN=${CUTOFF_LEN:-24480}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1280}

OUT_BASE="results/iclr_val_calib/${SHORT_NAME}/val_${VAL_RATIO}"
mkdir -p "$OUT_BASE"
OUT="${OUT_BASE}/finetuned-ckpt-${CKPT_STEP}.jsonl"

echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Dataset: ${VAL_DS}"
echo "Output: ${OUT}"

# Vision template needs image params (no-op for text)
IMAGE_ARGS=""
if [ "$TEMPLATE" = "qwen2_vl" ]; then
    IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-784}
    IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-1003520}
    IMAGE_ARGS="--image_min_pixels ${IMAGE_MIN_PIXELS} --image_max_pixels ${IMAGE_MAX_PIXELS}"
fi

python scripts/vllm_infer.py \
    --model_name_or_path "${CHECKPOINT_DIR}" \
    --dataset "${VAL_DS}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --save_logprobs \
    ${IMAGE_ARGS} \
    --save_name "${OUT}"

echo "DONE"
