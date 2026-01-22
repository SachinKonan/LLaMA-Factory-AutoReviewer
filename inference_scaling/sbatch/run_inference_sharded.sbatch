#!/bin/bash
#SBATCH --job-name=inf_sharded
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-23
#SBATCH --output=logs/inference_scaling/%A/infer_%A_%a.out
#SBATCH --error=logs/inference_scaling/%A/infer_%A_%a.err

# ============================================================================
# Sharded Inference for 5-Generation (new_fewshot) Case
# ============================================================================
# This script shards the new_fewshot dataset across multiple GPUs for faster
# inference with n_generations=5.
#
# Job Array Configuration (0-23 = 24 jobs):
#   - 3 modalities: clean, clean_images, vision
#   - 8 shards per modality
#
# Each GPU processes 1/8 of the dataset, then results are merged.
#
# Usage:
#   sbatch inference_scaling/sbatch/run_inference_sharded.sbatch
#
#   # Run only clean modality (8 shards):
#   sbatch --array=0-7 inference_scaling/sbatch/run_inference_sharded.sbatch
#
#   # Run only clean_images modality (8 shards):
#   sbatch --array=8-15 inference_scaling/sbatch/run_inference_sharded.sbatch
#
#   # Run only vision modality (8 shards):
#   sbatch --array=16-23 inference_scaling/sbatch/run_inference_sharded.sbatch
# ============================================================================

set -e

# Configuration
NUM_SHARDS=8
PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
source .vllm/bin/activate

# Base data directory (original datasets)
BASE_DATA_DIR="/n/fs/vision-mix/sk7524/LLaMA-Factory/data"

# Generated datasets directory
GENERATED_DATA_DIR="${PROJECT_DIR}/inference_scaling/data"

# Results directory
RESULTS_DIR="${PROJECT_DIR}/inference_scaling/results"

# Define modalities
MODALITIES=(
    "clean"
    "clean_images"
    "vision"
)

# Calculate indices
MODALITY_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SHARDS))
SHARD_ID=$((SLURM_ARRAY_TASK_ID % NUM_SHARDS))

MODALITY="${MODALITIES[$MODALITY_IDX]}"
PROMPT_VARIANT="new_fewshot"

# Model configuration based on modality
if [ "$MODALITY" == "clean" ]; then
    MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    TEMPLATE="${TEMPLATE:-qwen}"
    MEDIA_DIR=""
    # YaRN rope scaling: 32768 * 4 = 131072 context length
    ROPE_SCALING='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
else
    MODEL="${MODEL_VL:-Qwen/Qwen2.5-VL-7B-Instruct}"
    TEMPLATE="${TEMPLATE_VL:-qwen2_vl}"
    MEDIA_DIR="."
    ROPE_SCALING=""
fi

# Construct dataset name
BASE_DATASET="iclr_2020_2025_85_5_10_split6_balanced_${MODALITY}_binary_noreviews_v6"
DATASET_NAME="${BASE_DATASET}_test_${PROMPT_VARIANT}"

# Fixed parameters for new_fewshot
N_GENERATIONS=5
CUTOFF_LEN=50000

# Output file (per-shard)
OUTPUT_DIR="${RESULTS_DIR}/${MODALITY}/${PROMPT_VARIANT}"
OUTPUT_FILE="${OUTPUT_DIR}/predictions_shard_${SHARD_ID}.jsonl"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "logs/inference_scaling/${SLURM_ARRAY_JOB_ID}"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Modality: ${MODALITY}"
echo "Prompt Variant: ${PROMPT_VARIANT}"
echo "Shard: ${SHARD_ID}/${NUM_SHARDS}"
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

# Build inference command with sharding
INFERENCE_CMD="python inference_scaling/scripts/vllm_infer_ensemble.py \
    --model_name_or_path ${MODEL} \
    --dataset ${DATASET_NAME} \
    --dataset_dir ${GENERATED_DATA_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_new_tokens 10240 \
    --save_name ${OUTPUT_FILE} \
    --n_generations ${N_GENERATIONS} \
    --shard_id ${SHARD_ID} \
    --num_shards ${NUM_SHARDS}"

# Add media_dir for VL models
if [ -n "${MEDIA_DIR}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --media_dir ${MEDIA_DIR}"
fi

# Add rope_scaling config for extended context (clean modality)
if [ -n "${ROPE_SCALING}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --vllm_config '${ROPE_SCALING}'"
fi

# Run inference
eval ${INFERENCE_CMD}

echo "=============================================="
echo "Shard ${SHARD_ID} complete for ${MODALITY}/${PROMPT_VARIANT}"
echo "Output saved to: ${OUTPUT_FILE}"
echo "=============================================="

# If this is the last shard, merge all shards
if [ "$SHARD_ID" -eq "$((NUM_SHARDS - 1))" ]; then
    echo ""
    echo "This is the last shard. To merge all shards, run:"
    echo "  cat ${OUTPUT_DIR}/predictions_shard_*.jsonl > ${OUTPUT_DIR}/predictions.jsonl"
    echo "  rm ${OUTPUT_DIR}/predictions_shard_*.jsonl"
fi
