#!/bin/bash
#SBATCH --job-name=metareview
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=logs/inference_scaling/metareview_%A_%a.out
#SBATCH --error=logs/inference_scaling/metareview_%A_%a.err

# ============================================================================
# Meta-Review Job Array
# ============================================================================
# This script runs meta-review inference for ensembled predictions.
# Should be run AFTER run_inference.sbatch completes for new_fewshot variants.
#
# Job Array Configuration (0-2 = 3 jobs):
#   - 3 modalities: clean, clean_images, vision
# ============================================================================

set -e

# Configuration
PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
source .venv_vllm_inf/bin/activate
export TRANSFORMERS_OFFLINE=1

# Results directory
RESULTS_DIR="${PROJECT_DIR}/inference_strategies/results"
GENERATED_DATA_DIR="${PROJECT_DIR}/inference_strategies/data"

# Define modalities
MODALITIES=(
    "clean"
    "clean_images"
    "vision"
)

MODALITY="${MODALITIES[$SLURM_ARRAY_TASK_ID]}"

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

# Paths
PREDICTIONS_FILE="${RESULTS_DIR}/${MODALITY}/new_fewshot/predictions.jsonl"
METAREVIEW_DATASET_DIR="${GENERATED_DATA_DIR}/${MODALITY}_metareview"
METAREVIEW_PREDICTIONS="${RESULTS_DIR}/${MODALITY}/new_fewshot/metareview_predictions.jsonl"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Modality: ${MODALITY}"
echo "Input: ${PREDICTIONS_FILE}"
echo "Output: ${METAREVIEW_PREDICTIONS}"
echo "=============================================="

# Check if input exists
if [ ! -f "${PREDICTIONS_FILE}" ]; then
    echo "ERROR: Predictions file not found: ${PREDICTIONS_FILE}"
    echo "Please run run_inference.sbatch first."
    exit 1
fi

# Create meta-review dataset
mkdir -p "${METAREVIEW_DATASET_DIR}"
python inference_strategies/scripts/run_metareview.py create \
    --input "${PREDICTIONS_FILE}" \
    --output "${METAREVIEW_DATASET_DIR}/data.json"

# Build meta-review inference command
INFERENCE_CMD="python inference_strategies/scripts/vllm_infer_ensemble.py \
    --model_name_or_path ${MODEL} \
    --dataset ${MODALITY}_metareview \
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

# Run meta-review inference
eval ${INFERENCE_CMD}

echo "=============================================="
echo "Meta-review inference complete for ${MODALITY}"
echo "Output saved to: ${METAREVIEW_PREDICTIONS}"
echo "=============================================="
