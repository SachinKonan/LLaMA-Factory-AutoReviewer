#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0
#SBATCH --output=logs/grid_searchv2/infer_base_%A_%a.out
#SBATCH --error=logs/grid_searchv2/infer_base_%A_%a.err

set -e
cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer # /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

CONFIGS=("iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3"
    # "iclr_clean-title+abstract_binary_no_reviews_v2"
    # "iclr_clean_binary_no_reviews_v2"
    # "iclr_clean_binary_normalized_3normalizedreviews_and_normalizedmeta_v2"
    # "iclr_clean_binary_normalized_normalizedmeta_v2"
    # "iclr_clean_citation_no_reviews_v2"
    # "iclr_clean_citation_normalized_normalizedmeta_v2"
    # "iclr_clean_multiclass_no_reviews_v2"
    # "iclr_clean_multiclass_normalized_3normalizedreviews_and_normalizedmeta_v2"
    # "iclr_clean_multiclass_normalized_normalizedmeta_v2"
)

DATASET_BASE="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
TEST_DATASET="${DATASET_BASE}_test"
BASE_MODEL="saves/qwen2.5-3b/full/sft_ds3/checkpoint-1509" #"Qwen/Qwen2.5-7B-Instruct"
RESULTS_DIR="results/grid_searchv2/${DATASET_BASE}"
OUTPUT_FILE="${RESULTS_DIR}/base.jsonl"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET_BASE}"
echo "Model: ${BASE_MODEL}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

mkdir -p "${RESULTS_DIR}"

python scripts/vllm_infer.py \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset "${TEST_DATASET}" \
    --template qwen \
    --cutoff_len 16384 \
    --max_new_tokens 10240 \
    --save_name "${OUTPUT_FILE}"

echo "Inference complete!"
