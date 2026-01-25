#!/bin/bash
#SBATCH --job-name=vllm_int_v2
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --array=0-19
#SBATCH --output=logs/grid_searchv2/infer_int_%A_%a.out
#SBATCH --error=logs/grid_searchv2/infer_int_%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Each entry: "dataset_name|checkpoint_number"
CONFIGS=(
    # iclr_clean_binary_no_reviews_v2: 500, 1000, 1500, 2000, 2500
    "iclr_clean_binary_no_reviews_v2|500"
    "iclr_clean_binary_no_reviews_v2|1000"
    "iclr_clean_binary_no_reviews_v2|1500"
    "iclr_clean_binary_no_reviews_v2|2000"
    "iclr_clean_binary_no_reviews_v2|2500"
    # iclr_clean_binary_normalized_3normalizedreviews_and_normalizedmeta_v2: 500, 1000, 1500, 2000
    "iclr_clean_binary_normalized_3normalizedreviews_and_normalizedmeta_v2|500"
    "iclr_clean_binary_normalized_3normalizedreviews_and_normalizedmeta_v2|1000"
    "iclr_clean_binary_normalized_3normalizedreviews_and_normalizedmeta_v2|1500"
    "iclr_clean_binary_normalized_3normalizedreviews_and_normalizedmeta_v2|2000"
    # iclr_clean_binary_normalized_normalizedmeta_v2: 500, 1000, 1500, 2000, 2500
    "iclr_clean_binary_normalized_normalizedmeta_v2|500"
    "iclr_clean_binary_normalized_normalizedmeta_v2|1000"
    "iclr_clean_binary_normalized_normalizedmeta_v2|1500"
    "iclr_clean_binary_normalized_normalizedmeta_v2|2000"
    "iclr_clean_binary_normalized_normalizedmeta_v2|2500"
    # iclr_clean_citation_no_reviews_v2: 500, 1000, 1500
    "iclr_clean_citation_no_reviews_v2|500"
    "iclr_clean_citation_no_reviews_v2|1000"
    "iclr_clean_citation_no_reviews_v2|1500"
    # iclr_clean_citation_normalized_normalizedmeta_v2: 500, 1000, 1500
    "iclr_clean_citation_normalized_normalizedmeta_v2|500"
    "iclr_clean_citation_normalized_normalizedmeta_v2|1000"
    "iclr_clean_citation_normalized_normalizedmeta_v2|1500"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
DATASET_BASE=$(echo "$CONFIG" | cut -d'|' -f1)
CKPT_NUM=$(echo "$CONFIG" | cut -d'|' -f2)

TEST_DATASET="${DATASET_BASE}_test"
MODEL_PATH="saves/qwen2.5-7b/full/grid_searchv2/${DATASET_BASE}/checkpoint-${CKPT_NUM}"
RESULTS_DIR="results/grid_searchv2/${DATASET_BASE}"
OUTPUT_FILE="${RESULTS_DIR}/finetuned${CKPT_NUM}.jsonl"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET_BASE}"
echo "Checkpoint: ${CKPT_NUM}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

mkdir -p "${RESULTS_DIR}"

python scripts/vllm_infer.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${TEST_DATASET}" \
    --template qwen \
    --cutoff_len 20480 \
    --max_new_tokens 10240 \
    --save_name "${OUTPUT_FILE}"

echo "Inference complete!"
