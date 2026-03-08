#!/bin/bash
#SBATCH --job-name=eval_ckpts
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_checkpoints/eval_%A_%a.out
#SBATCH --error=logs/eval_checkpoints/eval_%A_%a.err

set -e
cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Configuration
CHECKPOINT_BASE="saves/qwen2.5-3b/full/sft_ds3"
DATASET_BASE="iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3"
TEST_DATASET="${DATASET_BASE}_test"
RESULTS_DIR="results/checkpoint_eval/${DATASET_BASE}"

# Find all checkpoints and sort numerically
mapfile -t CHECKPOINTS < <(find "${CHECKPOINT_BASE}" -maxdepth 1 -type d -name "checkpoint-*" | sort -t'-' -k2 -n)

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints found in ${CHECKPOINT_BASE}"
    exit 1
fi

# Get checkpoint for this array task
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "Available checkpoints:"
    for i in "${!CHECKPOINTS[@]}"; do
        echo "  [$i] ${CHECKPOINTS[$i]}"
    done
    echo ""
    echo "Run with: sbatch --array=0-$((${#CHECKPOINTS[@]}-1)) sbatch/eval_checkpoints.sbatch"
    exit 0
fi

CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT_NAME=$(basename "${CHECKPOINT}")
OUTPUT_FILE="${RESULTS_DIR}/${CHECKPOINT_NAME}.jsonl"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${TEST_DATASET}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

mkdir -p "${RESULTS_DIR}"
mkdir -p logs/eval_checkpoints

# Run inference
python scripts/vllm_infer.py \
    --model_name_or_path "${CHECKPOINT}" \
    --dataset "${TEST_DATASET}" \
    --template qwen \
    --cutoff_len 16384 \
    --max_new_tokens 10240 \
    --save_name "${OUTPUT_FILE}"

echo "Inference complete for ${CHECKPOINT_NAME}!"

# Run analysis on all completed results so far
echo ""
echo "Running analysis..."
python scripts/analyze.py --dir "checkpoint_eval/${DATASET_BASE}" --indicators binary

echo "Done!"
