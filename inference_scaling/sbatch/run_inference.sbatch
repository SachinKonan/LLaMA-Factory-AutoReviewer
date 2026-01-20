#!/bin/bash
#SBATCH --job-name=inf_scaling
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0
#SBATCH --output=logs/inference_scaling/infer_%A_%a.out
#SBATCH --error=logs/inference_scaling/infer_%A_%a.err

# ============================================================================
# Inference Scaling Job Array
# ============================================================================
# This script runs inference across all prompt variants and modalities.
#
# Job Array Configuration (0-8 = 9 jobs):
#   - 3 modalities: clean (text), clean_images (text+images), vision (vision-only)
#   - 3 prompt variants: original, new, new_fewshot
#
# Model Selection:
#   - clean: Qwen2.5-7B-Instruct-1M (qwen template)
#   - clean_images, vision: Qwen2.5-VL-7B-Instruct (qwen2_vl template)
#
# For prompts 3-5 (new_fewshot with ensembling):
#   - Single run with n_generations=5
#   - Results used for: single (1st output), majority (5 outputs), meta-review
# ============================================================================

set -e

# Configuration
PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
source .venv_vllm_inf/bin/activate
# export TRANSFORMERS_OFFLINE=1

# Base data directory (original datasets)
BASE_DATA_DIR="/n/fs/vision-mix/sk7524/LLaMA-Factory/data"

# Generated datasets directory
GENERATED_DATA_DIR="${PROJECT_DIR}/inference_strategies/data"

# Results directory
RESULTS_DIR="${PROJECT_DIR}/inference_strategies/results"

# Define modalities and their dataset patterns
MODALITIES=(
    "clean"
    "clean_images"
    "vision"
)

# Define prompt variants
PROMPT_VARIANTS=(
    "original"
    "new"
    "new_fewshot"
)

# Calculate indices
MODALITY_IDX=$((SLURM_ARRAY_TASK_ID / 3))
PROMPT_IDX=$((SLURM_ARRAY_TASK_ID % 3))

MODALITY="${MODALITIES[$MODALITY_IDX]}"
PROMPT_VARIANT="${PROMPT_VARIANTS[$PROMPT_IDX]}"

# Model configuration based on modality
# - clean (text-only): Qwen2.5-7B-Instruct-1M with qwen template
# - clean_images, vision: Qwen2.5-VL-7B-Instruct with qwen2_vl template
if [ "$MODALITY" == "clean" ]; then
    MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-1M}"
    TEMPLATE="${TEMPLATE:-qwen}"
    MEDIA_DIR=""
else
    MODEL="${MODEL_VL:-Qwen/Qwen2.5-VL-7B-Instruct}"
    TEMPLATE="${TEMPLATE_VL:-qwen2_vl}"
    MEDIA_DIR="."
fi

# Construct dataset name
BASE_DATASET="iclr_2020_2025_85_5_10_split6_balanced_${MODALITY}_binary_noreviews_v6"
DATASET_NAME="${BASE_DATASET}_test_${PROMPT_VARIANT}"

# Determine n_generations (5 for fewshot ensembling, 1 otherwise)
if [ "$PROMPT_VARIANT" == "new_fewshot" ]; then
    N_GENERATIONS=5
else
    N_GENERATIONS=1
fi

# Determine cutoff_len based on prompt variant
# Original: ~20k, New: ~20k, New+Fewshot: ~60k
if [ "$PROMPT_VARIANT" == "new_fewshot" ]; then
    CUTOFF_LEN=65536
else
    CUTOFF_LEN=20480
fi

# Output file
OUTPUT_DIR="${RESULTS_DIR}/${MODALITY}/${PROMPT_VARIANT}"
OUTPUT_FILE="${OUTPUT_DIR}/predictions.jsonl"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "logs/inference_scaling"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Modality: ${MODALITY}"
echo "Prompt Variant: ${PROMPT_VARIANT}"
echo "Dataset: ${DATASET_NAME}"
echo "Model: ${MODEL}"
echo "Template: ${TEMPLATE}"
echo "N Generations: ${N_GENERATIONS}"
echo "Cutoff Length: ${CUTOFF_LEN}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

# Check if generated dataset exists
if [ ! -d "${GENERATED_DATA_DIR}/${DATASET_NAME}" ]; then
    echo "ERROR: Generated dataset not found at ${GENERATED_DATA_DIR}/${DATASET_NAME}"
    echo "Please run generate_datasets.py first."
    exit 1
fi

# Build inference command
INFERENCE_CMD="python inference_strategies/scripts/vllm_infer_ensemble.py \
    --model_name_or_path ${MODEL} \
    --dataset ${DATASET_NAME} \
    --dataset_dir ${GENERATED_DATA_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_new_tokens 20480 \
    --save_name ${OUTPUT_FILE} \
    --n_generations ${N_GENERATIONS} \
    --temperature 0.7 \
    --top_p 0.9"

# Add media_dir for VL models
if [ -n "${MEDIA_DIR}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --media_dir ${MEDIA_DIR}"
fi

# Run inference
eval ${INFERENCE_CMD}

echo "=============================================="
echo "Inference complete for ${MODALITY}/${PROMPT_VARIANT}"
echo "Output saved to: ${OUTPUT_FILE}"
echo "=============================================="
