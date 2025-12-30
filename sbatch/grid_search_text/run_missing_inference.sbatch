#!/bin/bash
#SBATCH --job-name=missing_infer
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --array=0-5
#SBATCH --output=logs/grid_search/missing_infer_%A_%a.out
#SBATCH --error=logs/grid_search/missing_infer_%A_%a.err

# Run finetuned model inference for jobs that timed out
# 6 datasets that are missing _finetuned.jsonl

set -e

cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

export TRANSFORMERS_OFFLINE=1

# Datasets missing finetuned inference: DATASET_BASE:CHECKPOINT:CUTOFF_LEN
CONFIGS=(
    "iclr_text_binary_20480:checkpoint-1000:20480"
    "iclr_text_binary_20480_norm_reviews:checkpoint-1370:20480"
    "iclr_text_binary_20480_reviews:checkpoint-1415:20480"
    "iclr_text_multiclass_20480:checkpoint-1000:20480"
    "iclr_text_multiclass_20480_norm_reviews:checkpoint-1370:20480"
    "iclr_text_multiclass_20480_reviews:checkpoint-1415:20480"
)

# Parse configuration
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
DATASET_BASE=$(echo "$CONFIG" | cut -d: -f1)
CHECKPOINT=$(echo "$CONFIG" | cut -d: -f2)
CUTOFF_LEN=$(echo "$CONFIG" | cut -d: -f3)

# Paths
TEST_DATASET="${DATASET_BASE}_test"
MODEL_PATH="saves/qwen2.5-7b/full/grid_search/${DATASET_BASE}/${CHECKPOINT}"
RESULTS_DIR="results/grid_search"

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID (Array Task: $SLURM_ARRAY_TASK_ID)"
echo "Running on host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Dataset: ${DATASET_BASE}"
echo "Test dataset: ${TEST_DATASET}"
echo "Model path: ${MODEL_PATH}"
echo "Cutoff length: ${CUTOFF_LEN}"
echo "=============================================="

# Verify model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model path does not exist: ${MODEL_PATH}"
    exit 1
fi

mkdir -p logs/grid_search
mkdir -p "${RESULTS_DIR}"

echo ""
echo "Running vLLM inference on FINE-TUNED model..."

python scripts/vllm_infer.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${TEST_DATASET}" \
    --template qwen \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 10240 \
    --save_name "${RESULTS_DIR}/${DATASET_BASE}_finetuned.jsonl"

echo ""
echo "=============================================="
echo "Inference complete for ${DATASET_BASE}"
echo "Output: ${RESULTS_DIR}/${DATASET_BASE}_finetuned.jsonl"
echo "=============================================="