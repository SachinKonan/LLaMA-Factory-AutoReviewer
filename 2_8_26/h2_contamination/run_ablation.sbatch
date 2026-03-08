#!/bin/bash
#SBATCH --job-name=h2_contamination
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=2_8_26/logs/h2_contamination/infer_%A_%a.out
#SBATCH --error=2_8_26/logs/h2_contamination/infer_%A_%a.err
#SBATCH --comment="H2: Contamination progressive ablation"

# ============================================================================
# H2: Progressive Content Ablation
# ============================================================================
# Tests accuracy at 5 content levels to detect data contamination.
# If the model can predict acceptance from title alone, the paper is
# likely memorized from pretraining data.
#
# Array jobs:
#   0: title_only        (minimal context — contamination signal)
#   1: title_abstract    (standard abstract-level)
#   2: title_intro       (title + abstract + introduction)
#   3: title_conclusion  (title + abstract + conclusion)
#   4: full_paper        (full paper baseline)
#
# Usage:
#   # Generate datasets first
#   python 2_8_26/h2_contamination/generate_ablation_datasets.py
#
#   # Run all levels
#   sbatch 2_8_26/h2_contamination/run_ablation.sbatch
#
#   # Run single level (e.g., title_only)
#   sbatch --array=0 2_8_26/h2_contamination/run_ablation.sbatch
# ============================================================================

set -e

PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
source .vllm/bin/activate

mkdir -p 2_8_26/logs/h2_contamination
mkdir -p 2_8_26/h2_contamination/results

# Data
DATA_DIR="${PROJECT_DIR}/2_8_26/h2_contamination/data"
BASE_DATASET="iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test"

# Content levels
LEVELS=(
    "title_only"
    "title_abstract"
    "title_intro"
    "title_conclusion"
    "full_paper"
)

LEVEL="${LEVELS[$SLURM_ARRAY_TASK_ID]}"
DATASET="${BASE_DATASET}_${LEVEL}"

# Model
MODEL="Qwen/Qwen2.5-7B-Instruct"
TEMPLATE="qwen"

# Adjust cutoff length based on content level
case $LEVEL in
    title_only)       CUTOFF_LEN=4096;  MAX_NEW_TOKENS=512 ;;
    title_abstract)   CUTOFF_LEN=8192;  MAX_NEW_TOKENS=2048 ;;
    title_intro)      CUTOFF_LEN=16384; MAX_NEW_TOKENS=4096 ;;
    title_conclusion) CUTOFF_LEN=16384; MAX_NEW_TOKENS=4096 ;;
    full_paper)       CUTOFF_LEN=20480; MAX_NEW_TOKENS=10240 ;;
esac

# YaRN rope scaling for extended context
ROPE_SCALING='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'

# Output
OUTPUT_DIR="2_8_26/h2_contamination/results/${LEVEL}"
OUTPUT_FILE="${OUTPUT_DIR}/predictions.jsonl"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "H2: Content Ablation - ${LEVEL}"
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Cutoff: ${CUTOFF_LEN}, Max New: ${MAX_NEW_TOKENS}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

# Check dataset
if [ ! -d "${DATA_DIR}/${DATASET}" ]; then
    echo "ERROR: Dataset not found: ${DATA_DIR}/${DATASET}"
    echo "Please run: python 2_8_26/h2_contamination/generate_ablation_datasets.py"
    exit 1
fi

# Run inference
python inference_scaling/scripts/vllm_infer_ensemble.py \
    --model_name_or_path ${MODEL} \
    --dataset ${DATASET} \
    --dataset_dir ${DATA_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --save_name ${OUTPUT_FILE} \
    --n_generations 1 \
    --vllm_config "${ROPE_SCALING}"

echo "Inference complete for ${LEVEL}"

# Extract results (skip for title_only since it generates abstracts, not decisions)
if [ "${LEVEL}" != "title_only" ]; then
    python inference_scaling/scripts/extract_results.py \
        --input ${OUTPUT_FILE} \
        --output ${OUTPUT_DIR}/results_single.jsonl \
        --strategy single
    echo "Results extracted to ${OUTPUT_DIR}/results_single.jsonl"
fi

echo "=============================================="
echo "Done: ${LEVEL}"
echo "=============================================="
