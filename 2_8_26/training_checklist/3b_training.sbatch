#!/bin/bash
#SBATCH --job-name=training_check
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:8
#SBATCH --array=0-6
#SBATCH --output=2_8_26/logs/training_checklist/train_%A_%a.out
#SBATCH --error=2_8_26/logs/training_checklist/train_%A_%a.err
#SBATCH --comment="Training 3b with various data options"

# ============================================================================
# Ablation Training Job Array
# ============================================================================
# Full fine-tuning of Qwen2.5/vl-3B-Instruct using DeepSpeed ZeRO-3
# Each array job trains on a different ablation dataset.
#
# Model Selection:
#   - clean: Qwen2.5-3B-Instruct
#   - vision: Qwen2.5-VL-3B-Instruct
# ============================================================================

set -e  # Exit on error

cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

# Create logs directory
mkdir -p 2_8_26/training_checklist

# Array of training datasets
DATASETS=(
    ""
    ""
    ""
)

# Corresponding validation datasets
EVAL_DATASETS=(
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_counterfactual_reject_v6_validation"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_discussion_v6_validation"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_experimental_results_v6_validation"
)

# Select dataset based on array task ID
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
EVAL_DATASET="${EVAL_DATASETS[$SLURM_ARRAY_TASK_ID]}"

# Extract a short name for the output directory
SHORT_NAME=$(echo "$DATASET" | sed 's/iclr_2020_2025_85_5_10_split6_original_//' | sed 's/_v6_train//')
OUTPUT_DIR="ablations/saves/${SHORT_NAME}"

# Select model and template based on dataset modality
if [[ "$DATASET" == *"vision"* ]]; then
    MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
    TEMPLATE="qwen2_vl"
else
    MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
    TEMPLATE="qwen"
fi

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"
echo "=============================================="
echo "Dataset: ${DATASET}"
echo "Eval Dataset: ${EVAL_DATASET}"
echo "Model: ${MODEL_PATH}"
echo "Template: ${TEMPLATE}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# Set memory allocator for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create output directory and config directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "ablations/configs/generated"

# Generate job-specific YAML config from template
CONFIG_FILE="ablations/configs/generated/${SHORT_NAME}.yaml"
sed -e "s|PLACEHOLDER_DATASET|${DATASET}|g" \
    -e "s|PLACEHOLDER_EVAL_DATASET|${EVAL_DATASET}|g" \
    -e "s|PLACEHOLDER_OUTPUT_DIR|${OUTPUT_DIR}|g" \
    -e "s|model_name_or_path: Qwen/Qwen2.5-3B-Instruct|model_name_or_path: ${MODEL_PATH}|g" \
    -e "s|template: qwen|template: ${TEMPLATE}|g" \
    ablations/configs/ablation_base.yaml > "${CONFIG_FILE}"

echo "Generated config: ${CONFIG_FILE}"
echo ""
echo "Starting full fine-tuning with DeepSpeed ZeRO-3..."

# Run DeepSpeed ZeRO-3 training with torchrun
torchrun --nproc_per_node=8 \
    src/train.py "${CONFIG_FILE}"

echo ""
echo "=============================================="
echo "Training complete for ${SHORT_NAME}"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=============================================="