#!/bin/bash
#SBATCH --job-name=infer_ckpt
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/final_sweep_v7_infer/%j.out
#SBATCH --error=logs/final_sweep_v7_infer/%j.err

# Single-checkpoint inference job
# Called by checkpoint_watcher.py with environment variables:
#   CHECKPOINT_DIR - Path to checkpoint directory
#   DATASET - Dataset name (without _test suffix)
#   TEMPLATE - Template name (qwen or qwen2_vl)
#   SHORT_NAME - Short name for logging
#   CKPT_STEP - Checkpoint step number
#   RESULTS_DIR - Output directory for results
#   CUTOFF_LEN - Cutoff length
#   MAX_NEW_TOKENS - Max new tokens for generation
#   IMAGE_PARAMS - Image parameters (empty for text-only)

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Validate required environment variables
if [ -z "$CHECKPOINT_DIR" ] || [ -z "$DATASET" ] || [ -z "$TEMPLATE" ] || [ -z "$SHORT_NAME" ] || [ -z "$CKPT_STEP" ] || [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: Missing required environment variables"
    echo "  CHECKPOINT_DIR: ${CHECKPOINT_DIR:-<not set>}"
    echo "  DATASET: ${DATASET:-<not set>}"
    echo "  TEMPLATE: ${TEMPLATE:-<not set>}"
    echo "  SHORT_NAME: ${SHORT_NAME:-<not set>}"
    echo "  CKPT_STEP: ${CKPT_STEP:-<not set>}"
    echo "  RESULTS_DIR: ${RESULTS_DIR:-<not set>}"
    exit 1
fi

# Default values
CUTOFF_LEN=${CUTOFF_LEN:-24480}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1280}

echo "=============================================="
echo "Checkpoint Inference Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Dataset: ${DATASET}"
echo "Template: ${TEMPLATE}"
echo "Short Name: ${SHORT_NAME}"
echo "Step: ${CKPT_STEP}"
echo "Results Dir: ${RESULTS_DIR}"
echo "Cutoff Len: ${CUTOFF_LEN}"
echo "Max New Tokens: ${MAX_NEW_TOKENS}"
echo "Image Params: ${IMAGE_PARAMS:-<none>}"
echo "=============================================="

# Create results directory
mkdir -p logs/final_sweep_v7_infer "${RESULTS_DIR}"

# Output file name: finetuned-ckpt-{step}.jsonl
OUTPUT_FILE="${RESULTS_DIR}/finetuned-ckpt-${CKPT_STEP}.jsonl"

echo ""
echo "=== Running Inference ==="
echo "Output: ${OUTPUT_FILE}"
echo ""

# Run inference
# Note: IMAGE_PARAMS is intentionally unquoted to allow word splitting
python scripts/vllm_infer.py \
    --model_name_or_path "${CHECKPOINT_DIR}" \
    --dataset "${DATASET}_test" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    ${IMAGE_PARAMS} \
    --save_name "${OUTPUT_FILE}"

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME} checkpoint-${CKPT_STEP}"
echo "Output saved to: ${OUTPUT_FILE}"
echo "=============================================="
