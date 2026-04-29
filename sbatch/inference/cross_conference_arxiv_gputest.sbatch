#!/bin/bash
#SBATCH --job-name=xconf_arxiv_y24up
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=logs/auto_inference/xconf_arxiv_y24up_%j.out
#SBATCH --error=logs/auto_inference/xconf_arxiv_y24up_%j.err

# Cross-conference arxiv inference on the y24up text subset (1,415 samples,
# conference_year >= 2024). Designed for the gputest partition for fast
# turnaround vs. the full filtered24480 PLI runs.
#
# Required env:
#   CHECKPOINT_DIR  - path to checkpoint
#   CKPT_STEP       - step number (used in output filenames)
#   TEMPLATE        - "qwen" (text); vision unsupported here (no images dataset yet)
#   SHORT_NAME      - short identifier for output dir layout

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p logs/auto_inference

for v in CHECKPOINT_DIR CKPT_STEP TEMPLATE SHORT_NAME; do
    if [ -z "${!v}" ]; then
        echo "ERROR: $v not set"
        exit 2
    fi
done

CUTOFF_LEN=${CUTOFF_LEN:-24480}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1280}
SPLIT=${SPLIT:-test}   # test | validation

# Pick dataset by template + split
if [ "$TEMPLATE" = "qwen2_vl" ]; then
    if [ "$SPLIT" = "validation" ]; then
        ARXIV_DS="arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_validation"
        OUT_SUBDIR="arxiv_val"
    else
        ARXIV_DS="arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_test"
        OUT_SUBDIR="arxiv_eval"
    fi
    IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-784}
    IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-1003520}
    IMAGE_ARGS="--image_min_pixels ${IMAGE_MIN_PIXELS} --image_max_pixels ${IMAGE_MAX_PIXELS}"
else
    if [ "$SPLIT" = "validation" ]; then
        ARXIV_DS="arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation"
        OUT_SUBDIR="arxiv_val"
    else
        ARXIV_DS="arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test"
        OUT_SUBDIR="arxiv_eval"
    fi
    IMAGE_ARGS=""
fi

OUT_BASE="results/cross_conference_arxiv_y24up/${SHORT_NAME}"
mkdir -p "${OUT_BASE}/${OUT_SUBDIR}"
ARXIV_OUT="${OUT_BASE}/${OUT_SUBDIR}/finetuned-ckpt-${CKPT_STEP}.jsonl"

echo "=============================================="
echo "Cross-Conference (arxiv y24up) Inference"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $(hostname)"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Step:       ${CKPT_STEP}"
echo "  Short:      ${SHORT_NAME}"
echo "  Template:   ${TEMPLATE}"
echo "  Dataset:    ${ARXIV_DS}"
echo "  Output:     ${ARXIV_OUT}"
echo "=============================================="

python scripts/vllm_infer.py \
    --model_name_or_path "${CHECKPOINT_DIR}" \
    --dataset "${ARXIV_DS}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --save_logprobs \
    ${IMAGE_ARGS} \
    --save_name "${ARXIV_OUT}"

echo ""
echo "=============================================="
echo "DONE: $SHORT_NAME ckpt-$CKPT_STEP"
echo "  -> ${ARXIV_OUT}"
echo "=============================================="
