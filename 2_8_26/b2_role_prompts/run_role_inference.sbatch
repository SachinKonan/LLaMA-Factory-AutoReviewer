#!/bin/bash
#SBATCH --job-name=b2_role_prompts
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --output=2_8_26/logs/b2_role_prompts/role_%A_%a.out
#SBATCH --error=2_8_26/logs/b2_role_prompts/role_%A_%a.err
#SBATCH --comment="B2: Role-playing prompt inference"

# ============================================================================
# B2: Role-Playing Prompt Inference
# ============================================================================
# Runs inference with different reviewer personas to test bias mitigation.
#
# Array jobs (3 roles × 3 modalities = 9):
#   0: clean/critical       3: clean_images/critical       6: vision/critical
#   1: clean/enthusiastic   4: clean_images/enthusiastic   7: vision/enthusiastic
#   2: clean/standard       5: clean_images/standard       8: vision/standard
#
#  9, 10, 11: SACHIN_RL prompt on clean, clean_images, vision.

# Usage:
#   # Generate datasets first
#   python 2_8_26/b2_role_prompts/generate_role_datasets.py
#
#   # Run all roles
#   sbatch 2_8_26/b2_role_prompts/run_role_inference.sbatch
#
#   # Run only critical role
#   sbatch --array=0,3,6 2_8_26/b2_role_prompts/run_role_inference.sbatch
# ============================================================================

set -e

PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
source .vllm/bin/activate

mkdir -p 2_8_26/logs/b2_role_prompts

# Configuration
MODALITIES=("clean" "clean_images" "vision")
ROLES=("critical" "enthusiastic" "standard")

# Calculate indices
if [ $SLURM_ARRAY_TASK_ID -ge 9 ]; then
    # Jobs 9-11: SACHIN_RL on clean, clean_images, vision
    MODALITY_IDX=$((SLURM_ARRAY_TASK_ID - 9))
    MODALITY="${MODALITIES[$MODALITY_IDX]}"
    ROLE="sachin_rl"
else
    # Jobs 0-8: 3 roles × 3 modalities
    MODALITY_IDX=$((SLURM_ARRAY_TASK_ID / 3))
    ROLE_IDX=$((SLURM_ARRAY_TASK_ID % 3))
    MODALITY="${MODALITIES[$MODALITY_IDX]}"
    ROLE="${ROLES[$ROLE_IDX]}"
fi

# Data
DATA_DIR="${PROJECT_DIR}/2_8_26/b2_role_prompts/data"
BASE_PREFIX="iclr_2020_2025_85_5_10_split7_balanced"
DATASET="${BASE_PREFIX}_${MODALITY}_v7_test_${ROLE}"

# Model selection
if [ "$MODALITY" == "clean" ]; then
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    TEMPLATE="qwen"
    MEDIA_DIR=""
    VLLM_CONFIG=""
else
    MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    TEMPLATE="qwen2_vl"
    MEDIA_DIR="."
    VLLM_CONFIG=""
fi

CUTOFF_LEN=20400
MAX_NEW_TOKENS=10240

# Output
OUTPUT_DIR="2_8_26/b2_role_prompts/results/${MODALITY}/${ROLE}"
OUTPUT_FILE="${OUTPUT_DIR}/predictions.jsonl"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "B2: Role Inference - ${MODALITY}/${ROLE}"
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

# Check dataset
if [ ! -d "${DATA_DIR}/${DATASET}" ]; then
    echo "ERROR: Dataset not found: ${DATA_DIR}/${DATASET}"
    echo "Please run: python 2_8_26/b2_role_prompts/generate_role_datasets.py"
    exit 1
fi

# Build inference command
INFERENCE_CMD="python inference_scaling/scripts/vllm_infer_ensemble.py \
    --model_name_or_path ${MODEL} \
    --dataset ${DATASET} \
    --dataset_dir ${DATA_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --save_name ${OUTPUT_FILE} \
    --n_generations 1"

if [ -n "${MEDIA_DIR}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --media_dir ${MEDIA_DIR}"
fi
if [ -n "${VLLM_CONFIG}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --vllm_config '${VLLM_CONFIG}'"
fi

eval ${INFERENCE_CMD}

# Extract results
python inference_scaling/scripts/extract_results.py \
    --input ${OUTPUT_FILE} \
    --output ${OUTPUT_DIR}/results_single.jsonl \
    --strategy single

echo "=============================================="
echo "Done: ${MODALITY}/${ROLE}"
echo "=============================================="
