#!/bin/bash
#SBATCH --job-name=xconf_arxiv_natrate
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=logs/auto_inference/xconf_arxiv_natrate_%j.out
#SBATCH --error=logs/auto_inference/xconf_arxiv_natrate_%j.err

# Cross-conference inference on the arxiv NATRATE y24up text subset (~1.4k
# samples, conference_year >= 2024, with per-venue rates matching natural
# acceptance rates). Mirror of cross_conference_arxiv_gputest.sbatch but
# pointed at the natrate variant.
#
# Required env:
#   CHECKPOINT_DIR  - path to checkpoint
#   CKPT_STEP       - step number
#   TEMPLATE        - "qwen" (text); vision uses different sbatch
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

ARXIV_DS="arxiv_natrate_21k_text_wmetadata_filtered24480_y24up_test"
IMAGE_ARGS=""

OUT_BASE="results/cross_conference_arxiv_natrate_y24up/${SHORT_NAME}"
mkdir -p "${OUT_BASE}/arxiv_eval"

ARXIV_OUT="${OUT_BASE}/arxiv_eval/finetuned-ckpt-${CKPT_STEP}.jsonl"

echo "=============================================="
echo "Cross-Conference (arxiv natrate y24up) Inference"
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
