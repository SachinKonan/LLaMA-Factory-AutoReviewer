#!/bin/bash
#SBATCH --job-name=grid_search_text
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --array=0-8
#SBATCH --output=logs/grid_search/%A_%a.out
#SBATCH --error=logs/grid_search/%A_%a.err

# Grid Search over 9 ICLR dataset configurations
# Each job runs: base model inference -> training -> fine-tuned model inference
#
# Usage:
#   sbatch sbatch/grid_search_text/grid_search_array.sbatch

set -e  # Exit on error

cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

export TRANSFORMERS_OFFLINE=1

# Dataset configurations: DATASET_BASE:CUTOFF_LEN (all 20480 context)
CONFIGS=(
    "iclr_text_binary_20480:20480"
    "iclr_text_binary_20480_reviews:20480"
    "iclr_text_binary_20480_norm_reviews:20480"
    "iclr_text_citation_20480:20480"
    "iclr_text_citation_20480_reviews:20480"
    "iclr_text_citation_20480_norm_reviews:20480"
    "iclr_text_multiclass_20480:20480"
    "iclr_text_multiclass_20480_reviews:20480"
    "iclr_text_multiclass_20480_norm_reviews:20480"
)

# Parse configuration for this array task
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
DATASET_BASE="${CONFIG%%:*}"
CUTOFF_LEN="${CONFIG##*:}"

# Derive dataset names
TRAIN_DATASET="${DATASET_BASE}_train"
TEST_DATASET="${DATASET_BASE}_test"

# Output paths
OUTPUT_DIR="saves/qwen2.5-7b/full/grid_search/${DATASET_BASE}"
RESULTS_DIR="results/grid_search"

# Unique port for FSDP (avoid conflicts when multiple jobs on same host)
MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

# Base model path
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID (Array Task: $SLURM_ARRAY_TASK_ID)"
echo "Running on host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Dataset: ${DATASET_BASE}"
echo "Train: ${TRAIN_DATASET}, Test: ${TEST_DATASET}"
echo "Cutoff length: ${CUTOFF_LEN}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Results dir: ${RESULTS_DIR}"
echo "Master port: ${MASTER_PORT}"
echo "=============================================="

# Create output directories
mkdir -p logs
mkdir -p "${RESULTS_DIR}"
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# STEP 1: Base Model Inference
# =============================================================================
echo ""
echo "[Step 1/3] Running vLLM inference on BASE model..."
echo "Model: ${BASE_MODEL}"
echo "Test dataset: ${TEST_DATASET}"

python scripts/vllm_infer.py \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset "${TEST_DATASET}" \
    --template qwen \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 10240 \
    --save_name "${RESULTS_DIR}/${DATASET_BASE}_base.jsonl"

echo "[Step 1/3] Base model inference complete."

# =============================================================================
# STEP 2: Fine-tuning with FSDP2
# =============================================================================
echo ""
echo "[Step 2/3] Starting fine-tuning with FSDP2..."
echo "Train dataset: ${TRAIN_DATASET}"
echo "Output dir: ${OUTPUT_DIR}"

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py configs/grid_search_base.yaml \
    dataset="${TRAIN_DATASET}" \
    eval_dataset="${TEST_DATASET}" \
    output_dir="${OUTPUT_DIR}" \
    cutoff_len="${CUTOFF_LEN}"

echo "[Step 2/3] Fine-tuning complete."

# =============================================================================
# STEP 3: Fine-tuned Model Inference
# =============================================================================
echo ""
echo "[Step 3/3] Running vLLM inference on FINE-TUNED model..."
echo "Model: ${OUTPUT_DIR}"
echo "Test dataset: ${TEST_DATASET}"

python scripts/vllm_infer.py \
    --model_name_or_path "${OUTPUT_DIR}" \
    --dataset "${TEST_DATASET}" \
    --template qwen \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 10240 \
    --save_name "${RESULTS_DIR}/${DATASET_BASE}_finetuned.jsonl"

echo "[Step 3/3] Fine-tuned model inference complete."

echo ""
echo "=============================================="
echo "All steps complete for ${DATASET_BASE}"
echo "Results:"
echo "  Base model: ${RESULTS_DIR}/${DATASET_BASE}_base.jsonl"
echo "  Fine-tuned: ${RESULTS_DIR}/${DATASET_BASE}_finetuned.jsonl"
echo "Model saved: ${OUTPUT_DIR}"
echo "=============================================="
