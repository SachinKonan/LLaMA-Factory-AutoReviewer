#!/bin/bash
#SBATCH --job-name=checklist_train
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:8
#SBATCH --array=0-5
#SBATCH --output=2_8_26/training_checklist/logs/train_%A_%a.out
#SBATCH --error=2_8_26/training_checklist/logs/train_%A_%a.err
#SBATCH --comment="Checklist-optimized training (SFT+DPO)"

# ============================================================================
# Training Pipeline: SFT and DPO with Checklist-Optimized Reviews
# ============================================================================
# Job array mapping:
#   0: SFT clean (text-only)
#   1: SFT clean_images (text + images)
#   2: SFT vision (images-focused)
#   3: DPO clean (waits for job 0)
#   4: DPO clean_images (waits for job 1)
#   5: DPO vision (waits for job 2)
# ============================================================================

set -e  # Exit on error

cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

# Create logs directory
mkdir -p 2_8_26/training_checklist/logs
mkdir -p 2_8_26/training_checklist/configs/generated

# Configuration arrays
STAGES=(sft sft sft dpo dpo dpo)
MODALITIES=(clean clean_images vision clean clean_images vision)
MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "PLACEHOLDER"  # Will be set to SFT checkpoint
    "PLACEHOLDER"
    "PLACEHOLDER"
)
TEMPLATES=(qwen qwen2_vl qwen2_vl qwen qwen2_vl qwen2_vl)

STAGE="${STAGES[$SLURM_ARRAY_TASK_ID]}"
MODALITY="${MODALITIES[$SLURM_ARRAY_TASK_ID]}"
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
TEMPLATE="${TEMPLATES[$SLURM_ARRAY_TASK_ID]}"

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"
echo "=============================================="
echo "Stage: $STAGE"
echo "Modality: $MODALITY"
echo "Model (initial): $MODEL"
echo "Template: $TEMPLATE"
echo "=============================================="

# For DPO jobs, wait for corresponding SFT checkpoint
if [ "$STAGE" == "dpo" ]; then
    SFT_JOB_ID=$((SLURM_ARRAY_TASK_ID - 3))
    SFT_MODALITY="${MODALITIES[$SFT_JOB_ID]}"
    SFT_CHECKPOINT="2_8_26/training_checklist/checkpoints/sft_${SFT_MODALITY}"

    echo "DPO job detected: waiting for SFT checkpoint..."
    echo "  SFT checkpoint path: $SFT_CHECKPOINT"

    # Wait for SFT checkpoint (polling approach)
    MAX_WAIT_HOURS=24
    MAX_WAIT_SECONDS=$((MAX_WAIT_HOURS * 3600))
    WAIT_INTERVAL=300  # Check every 5 minutes
    ELAPSED=0

    while [ ! -f "$SFT_CHECKPOINT/config.json" ]; do
        if [ $ELAPSED -ge $MAX_WAIT_SECONDS ]; then
            echo "ERROR: Timeout waiting for SFT checkpoint after $MAX_WAIT_HOURS hours"
            exit 1
        fi

        echo "  Waiting for SFT training to complete... (elapsed: ${ELAPSED}s)"
        sleep $WAIT_INTERVAL
        ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    done

    echo "  SFT checkpoint found!"
    MODEL="$SFT_CHECKPOINT"
fi

# Output directory
OUTPUT_DIR="2_8_26/training_checklist/checkpoints/${STAGE}_${MODALITY}"
mkdir -p "$OUTPUT_DIR"

# Generate job-specific config from base template
CONFIG_FILE="2_8_26/training_checklist/configs/generated/${STAGE}_${MODALITY}.yaml"
BASE_CONFIG="2_8_26/training_checklist/configs/${STAGE}_base.yaml"

echo ""
echo "Generating config from template..."
echo "  Base config: $BASE_CONFIG"
echo "  Output config: $CONFIG_FILE"

# Perform substitutions
sed -e "s|model_name_or_path: .*|model_name_or_path: ${MODEL}|g" \
    -e "s|template: .*|template: ${TEMPLATE}|g" \
    -e "s|dataset: .*|dataset: training_checklist_${STAGE}_${MODALITY}_train|g" \
    -e "s|val_dataset: .*|val_dataset: training_checklist_${STAGE}_${MODALITY}_val|g" \
    -e "s|output_dir: .*|output_dir: ${OUTPUT_DIR}|g" \
    "$BASE_CONFIG" > "$CONFIG_FILE"

# Add vision-specific parameters for clean_images and vision modalities
if [[ "$MODALITY" == *"vision"* ]] || [[ "$MODALITY" == *"images"* ]]; then
    echo ""
    echo "Adding vision-specific parameters..."
    echo "image_max_pixels: 250880" >> "$CONFIG_FILE"
    echo "image_min_pixels: 196" >> "$CONFIG_FILE"
    echo "media_dir: ." >> "$CONFIG_FILE"

    # Reduce cutoff_len for vision models (due to image tokens)
    sed -i "s|cutoff_len: 28000|cutoff_len: 16384|g" "$CONFIG_FILE"
fi

echo ""
echo "=============================================="
echo "FINAL CONFIGURATION"
echo "=============================================="
echo "Stage: $STAGE"
echo "Modality: $MODALITY"
echo "Model: $MODEL"
echo "Template: $TEMPLATE"
echo "Dataset (train): training_checklist_${STAGE}_${MODALITY}_train"
echo "Dataset (val): training_checklist_${STAGE}_${MODALITY}_val"
echo "Output directory: $OUTPUT_DIR"
echo "Config file: $CONFIG_FILE"
echo "=============================================="

# Set memory allocator for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print config preview
echo ""
echo "Config file preview (first 20 lines):"
head -n 20 "$CONFIG_FILE"
echo "..."

echo ""
echo "=============================================="
echo "STARTING TRAINING"
echo "=============================================="

# Run training with DeepSpeed ZeRO-3
torchrun --nproc_per_node=8 \
    src/train.py "$CONFIG_FILE"

echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "=============================================="
echo "Stage: $STAGE"
echo "Modality: $MODALITY"
echo "Model saved to: $OUTPUT_DIR"
echo "=============================================="

# List checkpoint files
echo ""
echo "Checkpoint files:"
ls -lh "$OUTPUT_DIR"

echo ""
echo "Done!"
