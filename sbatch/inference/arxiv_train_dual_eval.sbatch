#!/bin/bash
#SBATCH --job-name=arxiv_train_dual_eval
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/auto_inference/arxiv_dual_%j.out
#SBATCH --error=logs/auto_inference/arxiv_dual_%j.err

# Dual-test inference for the arxiv_train pipeline. Loads a checkpoint and
# runs vLLM TWICE in the same allocation (once on arxiv_y24up, once on
# iclr_y25up), saving each to its own jsonl. Designed to be invoked by
# auto_infer_watcher.py via --inference_sbatch (single-test mode); the
# DATASET env var the watcher passes is ignored — we use the hardcoded
# pair below so we save one PLI submission per checkpoint vs two.
#
# Required env (set by auto_infer_watcher):
#   CHECKPOINT_DIR  - path to checkpoint
#   CKPT_STEP       - step number (used in output filenames)
#   TEMPLATE        - "qwen" or "qwen2_vl"
#   RESULTS_DIR     - base results dir; we write
#                       ${RESULTS_DIR}/arxiv_y24up/finetuned-ckpt-${STEP}.jsonl
#                       ${RESULTS_DIR}/iclr_y25up/finetuned-ckpt-${STEP}.jsonl

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p logs/auto_inference

for v in CHECKPOINT_DIR CKPT_STEP TEMPLATE RESULTS_DIR; do
    if [ -z "${!v}" ]; then
        echo "ERROR: $v not set"
        exit 2
    fi
done

CUTOFF_LEN=${CUTOFF_LEN:-24480}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1280}

# Hardcoded test pair, picked by template
if [ "$TEMPLATE" = "qwen2_vl" ]; then
    ARXIV_DS="arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_test"
    ICLR_DS="iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_y25up_test"
    IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-784}
    IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-1003520}
    IMAGE_ARGS="--image_min_pixels ${IMAGE_MIN_PIXELS} --image_max_pixels ${IMAGE_MAX_PIXELS}"
else
    ARXIV_DS="arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test"
    ICLR_DS="iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_test"
    IMAGE_ARGS=""
fi

mkdir -p "${RESULTS_DIR}/arxiv_y24up" "${RESULTS_DIR}/iclr_y25up"

ARXIV_OUT="${RESULTS_DIR}/arxiv_y24up/finetuned-ckpt-${CKPT_STEP}.jsonl"
ICLR_OUT="${RESULTS_DIR}/iclr_y25up/finetuned-ckpt-${CKPT_STEP}.jsonl"

echo "=============================================="
echo "Arxiv-train Dual Eval Inference"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $(hostname)"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Step:       ${CKPT_STEP}"
echo "  Template:   ${TEMPLATE}"
echo "  Arxiv ds:   ${ARXIV_DS}"
echo "  Iclr ds:    ${ICLR_DS}"
echo "  Image args: ${IMAGE_ARGS:-<none>}"
echo "=============================================="

echo ""
echo "=== STEP 1: arxiv_y24up -> ${ARXIV_OUT} ==="
echo ""

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
echo "=== STEP 2: iclr_y25up -> ${ICLR_OUT} ==="
echo ""

python scripts/vllm_infer.py \
    --model_name_or_path "${CHECKPOINT_DIR}" \
    --dataset "${ICLR_DS}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --save_logprobs \
    ${IMAGE_ARGS} \
    --save_name "${ICLR_OUT}"

echo ""
echo "=============================================="
echo "DONE: ckpt-${CKPT_STEP}"
echo "  ${ARXIV_OUT}"
echo "  ${ICLR_OUT}"
echo "=============================================="
