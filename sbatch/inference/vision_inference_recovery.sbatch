#!/bin/bash
#SBATCH --job-name=recovery_infer
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/auto_inference/%j.out
#SBATCH --error=logs/auto_inference/%j.err

# Recovery inference job for vision models
# Accepts environment variables:
#   CHECKPOINT_DIR - Path to checkpoint directory
#   DATASET - Dataset name (with _test suffix for inference, _train for eval)
#   TEMPLATE - Template name (qwen2_vl)
#   CUTOFF_LEN - Cutoff length
#   MAX_NEW_TOKENS - Max new tokens for generation
#   IMAGE_PARAMS - Image parameters (--image_min_pixels X --image_max_pixels Y)
#   RESULTS_DIR - Output directory for results
#   CKPT_STEP - Checkpoint step number
#   OUTPUT_FILENAME - Output filename (e.g., finetuned.jsonl or finetuned-ckpt-1596.jsonl)

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Validate required environment variables
if [ -z "$CHECKPOINT_DIR" ] || [ -z "$DATASET" ] || [ -z "$TEMPLATE" ] || [ -z "$CKPT_STEP" ] || [ -z "$RESULTS_DIR" ] || [ -z "$OUTPUT_FILENAME" ]; then
    echo "ERROR: Missing required environment variables"
    echo "  CHECKPOINT_DIR: ${CHECKPOINT_DIR:-<not set>}"
    echo "  DATASET: ${DATASET:-<not set>}"
    echo "  TEMPLATE: ${TEMPLATE:-<not set>}"
    echo "  CKPT_STEP: ${CKPT_STEP:-<not set>}"
    echo "  RESULTS_DIR: ${RESULTS_DIR:-<not set>}"
    echo "  OUTPUT_FILENAME: ${OUTPUT_FILENAME:-<not set>}"
    exit 1
fi

# Default values
CUTOFF_LEN=${CUTOFF_LEN:-24480}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1280}

# Derive train dataset from test dataset
TRAIN_DATASET="${DATASET%_test}_train"

echo "=============================================="
echo "Recovery Inference Job (Vision)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Test Dataset: ${DATASET}"
echo "Train Dataset: ${TRAIN_DATASET}"
echo "Template: ${TEMPLATE}"
echo "Step: ${CKPT_STEP}"
echo "Results Dir: ${RESULTS_DIR}"
echo "Output Filename: ${OUTPUT_FILENAME}"
echo "Cutoff Len: ${CUTOFF_LEN}"
echo "Max New Tokens: ${MAX_NEW_TOKENS}"
echo "Image Params: ${IMAGE_PARAMS:-<none>}"
echo "=============================================="

# Create directories
mkdir -p logs/auto_inference "${RESULTS_DIR}"

# Output file paths
TRAIN_METRICS_FILE="${RESULTS_DIR}/train-ckpt-${CKPT_STEP}.json"
OUTPUT_FILE="${RESULTS_DIR}/${OUTPUT_FILENAME}"

# ============================================
# Step 1: Eval on training data for metrics
# ============================================
echo ""
echo "=== Step 1: Eval on training data ==="
echo "Output: ${TRAIN_METRICS_FILE}"
echo ""

# Parse IMAGE_PARAMS to extract individual values for eval script
IMAGE_MAX_PIXELS=""
IMAGE_MIN_PIXELS=""
if [ -n "$IMAGE_PARAMS" ]; then
    IMAGE_MAX_PIXELS=$(echo "$IMAGE_PARAMS" | grep -oP '(?<=--image_max_pixels )\d+' || echo "")
    IMAGE_MIN_PIXELS=$(echo "$IMAGE_PARAMS" | grep -oP '(?<=--image_min_pixels )\d+' || echo "")
fi

# Build eval command with optional image params
EVAL_CMD="python scripts/eval_training_ckpt.py \
    --model_name_or_path \"${CHECKPOINT_DIR}\" \
    --dataset \"${TRAIN_DATASET}\" \
    --template \"${TEMPLATE}\" \
    --cutoff_len ${CUTOFF_LEN} \
    --save_name \"${TRAIN_METRICS_FILE}\" \
    --sft_accuracy_format boxed \
    --sft_positive_token Accept \
    --sft_negative_token Reject \
    --per_device_eval_batch_size 2 \
    --max_samples 2000 \
    --test_dataset \"${DATASET}\""

if [ -n "$IMAGE_MAX_PIXELS" ]; then
    EVAL_CMD="$EVAL_CMD --image_max_pixels $IMAGE_MAX_PIXELS"
fi
if [ -n "$IMAGE_MIN_PIXELS" ]; then
    EVAL_CMD="$EVAL_CMD --image_min_pixels $IMAGE_MIN_PIXELS"
fi

echo "Running: $EVAL_CMD"
eval $EVAL_CMD

echo ""
echo "=== Step 1 Complete ==="
echo ""

# ============================================
# Step 2: vLLM inference on test data
# ============================================
echo ""
echo "=== Step 2: vLLM inference on test data ==="
echo "Output: ${OUTPUT_FILE}"
echo ""

python scripts/vllm_infer.py \
    --model_name_or_path "${CHECKPOINT_DIR}" \
    --dataset "${DATASET}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    ${IMAGE_PARAMS} \
    --temperature 0 \
    --save_logprobs \
    --save_name "${OUTPUT_FILE}"

echo ""
echo "=============================================="
echo "COMPLETED: checkpoint-${CKPT_STEP}"
echo "Train metrics saved to: ${TRAIN_METRICS_FILE}"
echo "Test predictions saved to: ${OUTPUT_FILE}"
echo "=============================================="
