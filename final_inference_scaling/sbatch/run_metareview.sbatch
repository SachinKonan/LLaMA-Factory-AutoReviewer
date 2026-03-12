#!/bin/bash
#SBATCH --job-name=metareview
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --array=0-1
#SBATCH --output=logs/inference_scaling/metareview_%A_%a.out
#SBATCH --error=logs/inference_scaling/metareview_%A_%a.err

# ============================================================================
# Meta-Review Job Array
# ============================================================================
# This script runs meta-review inference for ensembled predictions.
# Should be run AFTER run_inference.sbatch completes for new_fewshot variants.
#
# Job Array Configuration (0-1 = 2 jobs):
#   - 2 modalities: text, vision
#
# Note: This expects sharded predictions to already be merged into
# a single predictions.jsonl file via the instructions printed
# at the end of run_inference_sharded.sbatch
# ============================================================================

set -e

# Configuration
PROJECT_DIR="/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
source .venv_vllm_inf/bin/activate
export TRANSFORMERS_OFFLINE=1

# Results directory
RESULTS_DIR="${PROJECT_DIR}/final_inference_scaling/results"
GENERATED_DATA_DIR="${PROJECT_DIR}/final_inference_scaling/data"

# Meta-review source target
PROMPT_VARIANT="${PROMPT_VARIANT_N_GEN:-critical_fewshot_1-1_json}"
N_GENERATIONS="${N_GENERATIONS:-5}"

# Define modalities
MODALITIES=(
    "text"
    "vision"
)

MODALITY="${MODALITIES[$SLURM_ARRAY_TASK_ID]}"

# Model configuration based on modality
# - clean (text-only): Qwen2.5-7B-Instruct with YaRN rope scaling for 131k context
# - clean_images, vision: Qwen2.5-VL-7B-Instruct with qwen2_vl template
if [ "$MODALITY" == "text" ]; then
    MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    TEMPLATE="${TEMPLATE:-qwen}"
    MEDIA_DIR=""
    # YaRN rope scaling: 32768 * 4 = 131072 context length
    ROPE_SCALING='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
else
    MODEL="${MODEL_VL:-/scratch/gpfs/ZHUANGL/jl0796/.hf_home/hub/models--Qwen--qwen2.5-VL-7B-instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
    TEMPLATE="${TEMPLATE_VL:-qwen2_vl}"
    MEDIA_DIR="."
    ROPE_SCALING=""
fi

# Paths
CLEAN_MODEL_NAME=$(basename "$MODEL")
if [[ "$PROMPT_VARIANT" == *"fewshot"* ]] && [[ "$PROMPT_VARIANT" != *"nofewshot"* ]]; then
    CUTOFF_LEN=${CTX_LEN_FEWSHOT:-60000}
else
    CUTOFF_LEN=${CTX_LEN_NOFEWSHOT:-24480}
fi

SOURCE_DIR="${RESULTS_DIR}/${CLEAN_MODEL_NAME}/${MODALITY}/${PROMPT_VARIANT}_ctx${CUTOFF_LEN}_gen${N_GENERATIONS}"
PREDICTIONS_FILE="${SOURCE_DIR}/predictions.jsonl"
METAREVIEW_DATASET_DIR="${GENERATED_DATA_DIR}/${MODALITY}_metareview_${PROMPT_VARIANT}"
METAREVIEW_PREDICTIONS="${SOURCE_DIR}/metareview_predictions.jsonl"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Modality: ${MODALITY}"
echo "Input: ${PREDICTIONS_FILE}"
echo "Output: ${METAREVIEW_PREDICTIONS}"
echo "=============================================="

# Check if input exists
if [ ! -f "${PREDICTIONS_FILE}" ]; then
    echo "ERROR: Predictions file not found: ${PREDICTIONS_FILE}"
    echo "Please run run_inference_sharded.sbatch first, AND merge the shards:"
    echo "  cat ${SOURCE_DIR}/predictions_shard_*.jsonl > ${PREDICTIONS_FILE}"
    exit 1
fi

# Create meta-review dataset
mkdir -p "${METAREVIEW_DATASET_DIR}"
python final_inference_scaling/scripts/run_metareview.py create \
    --input "${PREDICTIONS_FILE}" \
    --output "${METAREVIEW_DATASET_DIR}/data.json"

# Build meta-review inference command
INFERENCE_CMD="python final_inference_scaling/scripts/vllm_infer_ensemble.py \
    --model_name_or_path ${MODEL} \
    --dataset ${MODALITY}_metareview_${PROMPT_VARIANT} \
    --dataset_dir ${GENERATED_DATA_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len 32768 \
    --max_new_tokens 2048 \
    --save_name ${METAREVIEW_PREDICTIONS} \
    --n_generations 1 \
    --temperature 0.7 \
    --top_p 0.9"

# Add media_dir for VL models
if [ -n "${MEDIA_DIR}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --media_dir ${MEDIA_DIR}"
fi

# Add rope_scaling config for extended context (clean modality)
if [ -n "${ROPE_SCALING}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --vllm_config '${ROPE_SCALING}'"
fi

# Run meta-review inference
eval ${INFERENCE_CMD}

echo "=============================================="
echo "Meta-review inference complete for ${MODALITY}"
echo "Output saved to: ${METAREVIEW_PREDICTIONS}"
echo "=============================================="
