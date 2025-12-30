#!/bin/bash
#SBATCH --job-name=gsv2_ailab
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --array=0-8
#SBATCH --output=logs/grid_searchv2/%A_%a.out
#SBATCH --error=logs/grid_searchv2/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

CONFIGS=(
    "iclr_clean_binary_no_reviews_v2"
    "iclr_clean_binary_normalized_normalizedmeta_v2"
    "iclr_clean_binary_normalized_3normalizedreviews_and_normalizedmeta_v2"
    "iclr_clean_multiclass_no_reviews_v2"
    "iclr_clean_multiclass_normalized_normalizedmeta_v2"
    "iclr_clean_multiclass_normalized_3normalizedreviews_and_normalizedmeta_v2"
    "iclr_clean_citation_no_reviews_v2"
    "iclr_clean_citation_normalized_normalizedmeta_v2"
    "iclr_clean_citation_normalized_3normalizedreviews_and_normalizedmeta_v2"
)

DATASET_BASE="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
TRAIN_DATASET="${DATASET_BASE}_train"
TEST_DATASET="${DATASET_BASE}_test"
OUTPUT_DIR="saves/qwen2.5-7b/full/grid_searchv2/${DATASET_BASE}"
MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET_BASE}"
echo "Train: ${TRAIN_DATASET}, Test: ${TEST_DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

mkdir -p logs/grid_searchv2 "${OUTPUT_DIR}"

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py configs/grid_searchv2_base_batch16_ailab.yaml \
    dataset="${TRAIN_DATASET}" \
    eval_dataset="${TEST_DATASET}" \
    output_dir="${OUTPUT_DIR}"

echo "=============================================="
echo "Training complete for ${DATASET_BASE}"
echo "Model saved: ${OUTPUT_DIR}"
echo "=============================================="
