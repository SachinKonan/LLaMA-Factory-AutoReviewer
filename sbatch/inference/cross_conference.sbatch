#!/bin/bash
#SBATCH --job-name=xconf_infer
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/auto_inference/xconf_%j.out
#SBATCH --error=logs/auto_inference/xconf_%j.err

# Cross-conference inference: runs vLLM on nips_eval500 and icml_colm_eval350
# back-to-back in a single allocation (1 hour total budget).
#
# Required env:
#   CHECKPOINT_DIR  - path to checkpoint
#   CKPT_STEP       - step number (used in output filenames)
#   TEMPLATE        - "qwen" (text) or "qwen2_vl" (vision)
#   SHORT_NAME      - short identifier for output dir layout
#                     (e.g. "bz32_lr1e-6_text_40_60")
# Optional:
#   CUTOFF_LEN, MAX_NEW_TOKENS, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS

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

# Pick datasets per modality
if [ "$TEMPLATE" = "qwen2_vl" ]; then
    NIPS_DS="nips_2021_2025_original_vision_v7_eval500"
    ICML_DS="icml_colm_2024_2025_vision_binary_eval350"
    IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-784}
    IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-1003520}
    IMAGE_ARGS="--image_min_pixels ${IMAGE_MIN_PIXELS} --image_max_pixels ${IMAGE_MAX_PIXELS}"
else
    NIPS_DS="nips_2021_2025_original_text_v7_noref_eval500"
    ICML_DS="icml_colm_2024_2025_clean_binary_noref_eval350"
    IMAGE_ARGS=""
fi

OUT_BASE="results/cross_conference/${SHORT_NAME}"
mkdir -p "${OUT_BASE}/nips_eval" "${OUT_BASE}/icml_colm_eval"

echo "=============================================="
echo "Cross-Conference Inference"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Node:       $(hostname)"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Step:       ${CKPT_STEP}"
echo "  Short:      ${SHORT_NAME}"
echo "  Template:   ${TEMPLATE}"
echo "  NIPS ds:    ${NIPS_DS}"
echo "  ICML ds:    ${ICML_DS}"
echo "  Image args: ${IMAGE_ARGS:-<none>}"
echo "=============================================="

NIPS_OUT="${OUT_BASE}/nips_eval/finetuned-ckpt-${CKPT_STEP}.jsonl"
ICML_OUT="${OUT_BASE}/icml_colm_eval/finetuned-ckpt-${CKPT_STEP}.jsonl"

echo ""
echo "=== STEP 1: NIPS eval500 -> ${NIPS_OUT} ==="
echo ""

python scripts/vllm_infer.py \
    --model_name_or_path "${CHECKPOINT_DIR}" \
    --dataset "${NIPS_DS}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --save_logprobs \
    ${IMAGE_ARGS} \
    --save_name "${NIPS_OUT}"

echo ""
echo "=== STEP 2: ICML/COLM eval350 -> ${ICML_OUT} ==="
echo ""

python scripts/vllm_infer.py \
    --model_name_or_path "${CHECKPOINT_DIR}" \
    --dataset "${ICML_DS}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --save_logprobs \
    ${IMAGE_ARGS} \
    --save_name "${ICML_OUT}"

echo ""
echo "=============================================="
echo "DONE: $SHORT_NAME ckpt-$CKPT_STEP"
echo "  NIPS: ${NIPS_OUT}"
echo "  ICML: ${ICML_OUT}"
echo "=============================================="
