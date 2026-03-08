#!/bin/bash
#SBATCH --job-name=ablation_v1_infer
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-38
#SBATCH --output=ablations/logs/inference/e7/inference_v1_%A_%a.out
#SBATCH --error=ablations/logs/inference/e7/inference_v1_%A_%a.err
#SBATCH --comment="Ablation V1 Inference"

# ============================================================================
# Ablation V1 Inference Job Array
# ============================================================================
# This script runs inference for ablation v1 datasets (42 total)-- less critical modifier only for gemini.
#
# Dataset Types:
#   1. Critical/Less Critical Modifiers (6 datasets): 0-5
#      - 1 modifiers x 2 output formats x 3 modalities
#   2. Fewshot Full Paper (9 datasets): 6-14
#      - 3 variations x 3 modalities
#   3. Fewshot Paper Parts (24 datasets): 15-38
#      - 4 parts x 3 variations x 2 modalities (no vision)
#   4. Base Datasets (3 datasets): 39-41
#      - 3 modalities (clean, clean_images, vision)
#
# For base-only submission: sbatch --array=39-41 inference_experiment_v1.sbatch
#
# Model Selection:
#   - clean: Qwen2.5-7B-Instruct (qwen template)
#   - clean_images, vision: Qwen2.5-VL-7B-Instruct (qwen2_vl template)
# ============================================================================

set -e

# Configuration
PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
source .vllm/bin/activate

# Create logs dir
mkdir -p ablations/logs

# Data directory
DATA_DIR="${PROJECT_DIR}/data"

# Results directory
RESULTS_DIR="${PROJECT_DIR}/ablations/results_v1"
mkdir -p "${RESULTS_DIR}"

# Output prefix for datasets
PREFIX="iclr_2020_2025_85_5_10_split7_ablation_v1"

# ============================================================================
# Dataset Array (42 total)
# ============================================================================
DATASETS=(
    # === Critical/Less Critical Modifiers (12 datasets) ===
    # Critical modifier 0-5
    "${PREFIX}_critical_boxed_clean_test"
    "${PREFIX}_critical_boxed_clean_images_test"
    "${PREFIX}_critical_boxed_vision_test"
    "${PREFIX}_critical_json_clean_test"
    "${PREFIX}_critical_json_clean_images_test"
    "${PREFIX}_critical_json_vision_test"
    # Less critical modifier (only for gemini)
    # "${PREFIX}_less_critical_boxed_clean_test"
    # "${PREFIX}_less_critical_boxed_clean_images_test"
    # "${PREFIX}_less_critical_boxed_vision_test"
    # "${PREFIX}_less_critical_json_clean_test"
    # "${PREFIX}_less_critical_json_clean_images_test"
    # "${PREFIX}_less_critical_json_vision_test"

    # === Fewshot Full Paper (9 datasets) ===
    # 2 accept, 0 reject -- 6-8
    "${PREFIX}_fewshot_fullpaper_2acc_0rej_clean_test"
    "${PREFIX}_fewshot_fullpaper_2acc_0rej_clean_images_test"
    "${PREFIX}_fewshot_fullpaper_2acc_0rej_vision_test"
    # 1 accept, 1 reject -- 9-11
    "${PREFIX}_fewshot_fullpaper_1acc_1rej_clean_test"
    "${PREFIX}_fewshot_fullpaper_1acc_1rej_clean_images_test"
    "${PREFIX}_fewshot_fullpaper_1acc_1rej_vision_test"
    # 0 accept, 2 reject -- 12-14
    "${PREFIX}_fewshot_fullpaper_0acc_2rej_clean_test"
    "${PREFIX}_fewshot_fullpaper_0acc_2rej_clean_images_test"
    "${PREFIX}_fewshot_fullpaper_0acc_2rej_vision_test"

    # === Fewshot Paper Parts (24 datasets) ===
    # Introduction -- 15-20
    "${PREFIX}_fewshot_intro_2acc_0rej_clean_test"
    "${PREFIX}_fewshot_intro_2acc_0rej_clean_images_test"
    "${PREFIX}_fewshot_intro_1acc_1rej_clean_test"
    "${PREFIX}_fewshot_intro_1acc_1rej_clean_images_test"
    "${PREFIX}_fewshot_intro_0acc_2rej_clean_test"
    "${PREFIX}_fewshot_intro_0acc_2rej_clean_images_test"
    # Related Work -- 21-26
    "${PREFIX}_fewshot_related_2acc_0rej_clean_test"
    "${PREFIX}_fewshot_related_2acc_0rej_clean_images_test"
    "${PREFIX}_fewshot_related_1acc_1rej_clean_test"
    "${PREFIX}_fewshot_related_1acc_1rej_clean_images_test"
    "${PREFIX}_fewshot_related_0acc_2rej_clean_test"
    "${PREFIX}_fewshot_related_0acc_2rej_clean_images_test"
    # Methodology -- 27-32
    "${PREFIX}_fewshot_method_2acc_0rej_clean_test"
    "${PREFIX}_fewshot_method_2acc_0rej_clean_images_test"
    "${PREFIX}_fewshot_method_1acc_1rej_clean_test"
    "${PREFIX}_fewshot_method_1acc_1rej_clean_images_test"
    "${PREFIX}_fewshot_method_0acc_2rej_clean_test"
    "${PREFIX}_fewshot_method_0acc_2rej_clean_images_test"
    # Experimental Results -- 33-38
    "${PREFIX}_fewshot_results_2acc_0rej_clean_test"
    "${PREFIX}_fewshot_results_2acc_0rej_clean_images_test"
    "${PREFIX}_fewshot_results_1acc_1rej_clean_test"
    "${PREFIX}_fewshot_results_1acc_1rej_clean_images_test"
    "${PREFIX}_fewshot_results_0acc_2rej_clean_test"
    "${PREFIX}_fewshot_results_0acc_2rej_clean_images_test"

    # === Base Datasets (3 datasets) ===
    # For base-only submission, use: sbatch --array=39-41 inference_experiment_v1.sbatch
    "${PREFIX}_base_clean_test"           # 39
    "${PREFIX}_base_clean_images_test"    # 40
    "${PREFIX}_base_vision_test"          # 41
)

# Get current dataset
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

# ============================================================================
# Model Configuration
# ============================================================================
# Determine model based on modality in dataset name
if [[ "$DATASET" == *"vision"* ]] || [[ "$DATASET" == *"clean_images"* ]]; then
    MODEL="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/saves/qwen2.5vl-3b/full/sft_ds3/checkpoint-2500"
    TEMPLATE="qwen2_vl"
    MEDIA_DIR="."
else
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    TEMPLATE="qwen"
    MEDIA_DIR=""
fi

ROPE_SCALING=""
# Cutoff length - longer for fewshot datasets
if [[ "$DATASET" == *"fewshot"* ]] && ([[ "$DATASET" == *"vision"* ]] || [[ "$DATASET" == *"clean_images"* ]]); then
    CUTOFF_LEN=64000 # 2^14*3 - 1024
    ROPE_SCALING='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
else
    CUTOFF_LEN=30000
fi

# Output file
OUTPUT_FILE="${RESULTS_DIR}/${DATASET}_predictions.jsonl"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Template: ${TEMPLATE}"
echo "Cutoff Length: ${CUTOFF_LEN}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

# Check if dataset exists
if [ ! -d "${DATA_DIR}/${DATASET}" ]; then
    echo "ERROR: Dataset not found: ${DATA_DIR}/${DATASET}"
    echo "Please run generate_datasets.py first!"
    exit 1
fi

# Build inference command
INFERENCE_CMD="python ablations/run_inference.py \
    --model_name_or_path ${MODEL} \
    --dataset ${DATASET} \
    --dataset_dir ${DATA_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_new_tokens 1024 \
    --save_name ${OUTPUT_FILE} \
    --n_generations 1"

# Add media_dir for VL models
if [ -n "${MEDIA_DIR}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --media_dir ${MEDIA_DIR}"
fi

# Add rope_scaling config for extended context 
if [ -n "${ROPE_SCALING}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --vllm_config '${ROPE_SCALING}'"
fi

# Run inference
eval ${INFERENCE_CMD}

echo "=============================================="
echo "Inference complete for ${DATASET}"
echo "Output saved to: ${OUTPUT_FILE}"
echo "=============================================="
