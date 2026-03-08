#!/bin/bash
#SBATCH --job-name=h1_base_model
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=2_8_26/logs/h1_base_model/infer_%j.out
#SBATCH --error=2_8_26/logs/h1_base_model/infer_%j.err
#SBATCH --comment="H1: Base model vs Instruct - bias investigation"

# ============================================================================
# H1: Base Model Inference
# ============================================================================
# Tests Qwen2.5-7B (base, non-instruct) to see if instruct finetuning
# causes the optimism bias (95-100% acceptance rate).
#
# Usage:
#   # Generate dataset first
#   python 2_8_26/h1_base_model/generate_dataset.py
#
#   # Run inference
#   sbatch 2_8_26/h1_base_model/run_inference.sbatch
# ============================================================================

set -e

PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
source .vllm/bin/activate

mkdir -p 2_8_26/logs/h1_base_model
mkdir -p 2_8_26/h1_base_model/results

# Data
DATA_DIR="${PROJECT_DIR}/2_8_26/h1_base_model/data"
DATASET="iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test_base_model"

# Model: Base (non-instruct) Qwen 2.5 7B
MODEL="Qwen/Qwen2.5-7B"
TEMPLATE="default"
CUTOFF_LEN=20480
MAX_NEW_TOKENS=2048

# Output
OUTPUT_FILE="2_8_26/h1_base_model/results/predictions.jsonl"

echo "=============================================="
echo "H1: Base Model Inference"
echo "Job: $SLURM_JOB_ID"
echo "Model: ${MODEL}"
echo "Template: ${TEMPLATE}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

# Check dataset exists
if [ ! -d "${DATA_DIR}/${DATASET}" ]; then
    echo "ERROR: Dataset not found: ${DATA_DIR}/${DATASET}"
    echo "Please run: python 2_8_26/h1_base_model/generate_dataset.py"
    exit 1
fi

# Run inference using the shared vllm_infer_ensemble.py
# Note: base model uses template=default (no chat formatting)
# YaRN rope scaling for extended context
ROPE_SCALING='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'

python inference_scaling/scripts/vllm_infer_ensemble.py \
    --model_name_or_path ${MODEL} \
    --dataset ${DATASET} \
    --dataset_dir ${DATA_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --save_name ${OUTPUT_FILE} \
    --n_generations 1 \
    --temperature 0.7 \
    --vllm_config "${ROPE_SCALING}"

echo "=============================================="
echo "Base model inference complete"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

# Extract results
python inference_scaling/scripts/extract_results.py \
    --input ${OUTPUT_FILE} \
    --output 2_8_26/h1_base_model/results/results_single.jsonl \
    --strategy single

echo "Results extracted to 2_8_26/h1_base_model/results/results_single.jsonl"
