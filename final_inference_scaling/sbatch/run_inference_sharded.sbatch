#!/bin/bash
#SBATCH --job-name=inf_sharded
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-15
#SBATCH --output=logs/final_inference_scaling/%A/infer_%A_%a.out
#SBATCH --error=logs/final_inference_scaling/%A/infer_%A_%a.err

# ============================================================================
# Sharded Inference for N-Generation Case (e.g., PDR with n=5)
# ============================================================================
# This script shards a specific dataset across multiple GPUs for faster
# inference with n_generations > 1.
#
# Job Array Configuration (0-15 = 16 jobs):
#   - 2 modalities: text, vision
#   - 8 shards per modality
#
# Each GPU processes 1/8 of the dataset, then results are merged.
#
# Usage:
#   sbatch final_inference_scaling/sbatch/run_inference_sharded.sbatch
#
#   # Run only text modality (8 shards):
#   sbatch --array=0-7 final_inference_scaling/sbatch/run_inference_sharded.sbatch
#
#   # Run only vision modality (8 shards):
#   sbatch --array=8-15 final_inference_scaling/sbatch/run_inference_sharded.sbatch
# ============================================================================

set -e

# Configuration
NUM_SHARDS=8
PROJECT_DIR="/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
source .venv_vllm_inf/bin/activate
export TRANSFORMERS_OFFLINE=1

# Generated datasets directory
GENERATED_DATA_DIR="${PROJECT_DIR}/final_inference_scaling/data"

# Results directory
RESULTS_DIR="${PROJECT_DIR}/final_inference_scaling/results"

# Models and Context Length Configuration
MODEL_CLEAN="${MODEL_CLEAN:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_VISION="${MODEL_VISION:-/scratch/gpfs/ZHUANGL/jl0796/.hf_home/hub/models--Qwen--qwen2.5-VL-7B-instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"

CTX_LEN_NOFEWSHOT="${CTX_LEN_NOFEWSHOT:-20480}"
CTX_LEN_FEWSHOT="${CTX_LEN_FEWSHOT:-60000}"

# Target Prompt Variant to run 5 generations on
PROMPT_VARIANT="${PROMPT_VARIANT_N_GEN:-critical_fewshot_1-1_json}"
N_GENERATIONS="${N_GENERATIONS:-5}"

# Define modalities
MODALITIES=(
    "text"
    "vision"
)

BASE_DATASETS=(
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480"
)

# Calculate indices
MODALITY_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SHARDS))
SHARD_ID=$((SLURM_ARRAY_TASK_ID % NUM_SHARDS))

MODALITY="${MODALITIES[$MODALITY_IDX]}"
BASE_DATASET="${BASE_DATASETS[$MODALITY_IDX]}"

# Determine if this variant uses fewshot to set context length
if [[ "$PROMPT_VARIANT" == *"fewshot"* ]] && [[ "$PROMPT_VARIANT" != *"nofewshot"* ]]; then
    CUTOFF_LEN=$CTX_LEN_FEWSHOT
else
    CUTOFF_LEN=$CTX_LEN_NOFEWSHOT
fi

# Model configuration based on modality
if [ "$MODALITY" == "text" ]; then
    MODEL="${MODEL_CLEAN}"
    TEMPLATE="${TEMPLATE:-qwen}"
    MEDIA_DIR=""
    # YaRN rope scaling: 32768 * 4 = 131072 context length
    ROPE_SCALING='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
else
    MODEL="${MODEL_VISION}"
    TEMPLATE="${TEMPLATE_VL:-qwen2_vl}"
    MEDIA_DIR="."
    ROPE_SCALING=""
fi


# Clean model name for the output path
CLEAN_MODEL_NAME=$(basename "$MODEL")

# Construct dataset name
DATASET_NAME="${BASE_DATASET}_test_${PROMPT_VARIANT}"

# Output file path (Includes model name, context len, and generations to prevent overwrites)
OUTPUT_DIR="${RESULTS_DIR}/${CLEAN_MODEL_NAME}/${MODALITY}/${PROMPT_VARIANT}_ctx${CUTOFF_LEN}_gen${N_GENERATIONS}"
OUTPUT_FILE="${OUTPUT_DIR}/predictions_shard_${SHARD_ID}.jsonl"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "logs/final_inference_scaling/${SLURM_ARRAY_JOB_ID}"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Modality: ${MODALITY}"
echo "Prompt Variant: ${PROMPT_VARIANT}"
echo "Shard: ${SHARD_ID}/${NUM_SHARDS}"
echo "Dataset: ${DATASET_NAME}"
echo "Model: ${MODEL} (Cleaned for path: ${CLEAN_MODEL_NAME})"
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
INFERENCE_CMD="python final_inference_scaling/scripts/vllm_infer_ensemble.py \
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

# If this is the last shard, print merge instructions
if [ "$SHARD_ID" -eq "$((NUM_SHARDS - 1))" ]; then
    echo ""
    echo "This is the last shard. To merge all shards, run:"
    echo "  cat ${OUTPUT_DIR}/predictions_shard_*.jsonl > ${OUTPUT_DIR}/predictions.jsonl"
fi
