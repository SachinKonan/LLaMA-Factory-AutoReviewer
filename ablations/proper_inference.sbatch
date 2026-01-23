#!/bin/bash
#SBATCH --job-name=ablation_infer
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-20
#SBATCH --output=ablations/logs/inference_%A_%a.out
#SBATCH --error=ablations/logs/inference_%A_%a.err
#SBATCH --comment="Ablation Inference"

# ============================================================================
# Ablation Inference Job Array
# ============================================================================
# This script runs inference for ablation study datasets.
#
# Job Array Configuration (0-20 = 21 jobs):
#   - Counterfactual reject datasets (clean, vision)
#   - Section filtering datasets (only_X, no_X for various sections)
#   - Critical analysis dataset (clean, 100 samples)
#   - Core technical (related_work + methodology + experimental_results)
#   - Fewshot variants with different accept/reject mixtures
#
# Model Selection:
#   - clean: Qwen2.5-3B-Instruct
#   - vision: Qwen2.5-VL-3B-Instruct (qwen2_vl template)
# ============================================================================

set -e

# Configuration
PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
source .vllm/bin/activate

# Create logs dir
mkdir -p ablations/logs

# Generated datasets directory
GENERATED_DATA_DIR="${PROJECT_DIR}/data"

# Results directory
RESULTS_DIR="${PROJECT_DIR}/ablations/results"

# Array of datasets
DATASETS=(
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_counterfactual_reject_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_discussion_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_experimental_results_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_introduction_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_intro_discussion_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_methodology_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_related_work_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_discussion_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_experimental_results_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_introduction_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_intro_discussion_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_methodology_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_related_work_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_vision_binary_noreviews_counterfactual_reject_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_critical_analysis_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_test" #baseline
    # Core technical (related_work + methodology + experimental_results)
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_core_technical_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_core_technical_v6_test"
    # Fewshot variants
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_core_technical_fewshot_2acc_2rej_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_core_technical_fewshot_3acc_1rej_v6_test"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_core_technical_fewshot_1acc_3rej_v6_test"
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

# Model configuration based on dataset modality
if [[ "$DATASET" == *"vision"* ]]; then
    MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
    TEMPLATE="qwen2_vl"
    MEDIA_DIR="."
else
    MODEL="Qwen/Qwen2.5-3B-Instruct"
    TEMPLATE="qwen"
    MEDIA_DIR=""
fi

# Cutoff length - longer for fewshot datasets (they include example papers)
if [[ "$DATASET" == *"fewshot"* ]]; then
    CUTOFF_LEN=20480
else
    CUTOFF_LEN=20480
fi

# Output file
OUTPUT_DIR="${RESULTS_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/${DATASET}_predictions.jsonl"

# Create directories
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Template: ${TEMPLATE}"
echo "Cutoff Length: ${CUTOFF_LEN}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

# Build inference command
INFERENCE_CMD="python ablations/run_inference.py \
    --model_name_or_path ${MODEL} \
    --dataset ${DATASET} \
    --dataset_dir ${GENERATED_DATA_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_new_tokens 1024 \
    --save_name ${OUTPUT_FILE} \
    --n_generations 1"

# Add media_dir for VL models
if [ -n "${MEDIA_DIR}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --media_dir ${MEDIA_DIR}"
fi

# Run inference
eval ${INFERENCE_CMD}

echo "=============================================="
echo "Inference complete for ${DATASET}"
echo "Output saved to: ${OUTPUT_FILE}"
echo "=============================================="
