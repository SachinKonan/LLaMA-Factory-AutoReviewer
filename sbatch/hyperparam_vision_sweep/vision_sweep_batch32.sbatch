#!/bin/bash
#SBATCH --job-name=hp_vis_b32
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --array=0-1
#SBATCH --output=logs/hyperparam_vision_sweep/%A_%a.out
#SBATCH --error=logs/hyperparam_vision_sweep/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 2 combinations: clean_images only (vision batch32 already running as 3538083_5, 3538083_7)
# Format: "FREEZE_PROJ DATASET SHORT_NAME"
CONFIGS=(
    "true iclr_clean+images_binary_no_reviews_v3 clean_images"
    "false iclr_clean+images_binary_no_reviews_v3 clean_images"
)

# Parse hyperparameters from array index
read FREEZE_PROJ DATASET SHORT_NAME <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Config for clean+images
CONFIG_FILE="configs/hyperparam_clean_and_images_sweep_batch32.yaml"
INFER_CUTOFF=30000
INFER_MAX_PIXELS=589824

# Short name for freeze setting
if [ "$FREEZE_PROJ" == "true" ]; then
    FREEZE_SHORT="frozen"
else
    FREEZE_SHORT="unfrozen"
fi

# Directory names - unique port per job (base 29600 for batch32, different from batch16)
CONFIG_NAME="bs32_proj_${FREEZE_SHORT}_${SHORT_NAME}"
MODEL_DIR="saves/qwen2.5-vl-7b/full/hyperparam_vision_sweep/${CONFIG_NAME}"
RESULTS_DIR="results/hyperparam_vision_sweep/${CONFIG_NAME}"
MASTER_PORT=$((29600 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Config: ${CONFIG_NAME}"
echo "Batch Size: 32, Freeze Projector: ${FREEZE_PROJ}"
echo "Dataset: ${DATASET}"
echo "Config File: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "Master Port: ${MASTER_PORT}"
echo "=============================================="

mkdir -p logs/hyperparam_vision_sweep "${MODEL_DIR}" "${RESULTS_DIR}"

# ======================
# PHASE 1: TRAINING
# ======================
echo ""
echo "=== PHASE 1: TRAINING ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_4gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py "${CONFIG_FILE}" \
    dataset="${DATASET}_train" \
    eval_dataset="${DATASET}_test" \
    output_dir="${MODEL_DIR}" \
    freeze_multi_modal_projector="${FREEZE_PROJ}"

echo ""
echo "=== Training complete! ==="
echo ""

# ======================
# PHASE 2: INFERENCE
# ======================
echo ""
echo "=== PHASE 2: INFERENCE ==="
echo ""

# Use single GPU for inference
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | head -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "Using checkpoint: ${CKPT_DIR}"

python scripts/vllm_infer.py \
    --model_name_or_path "${CKPT_DIR}" \
    --dataset "${DATASET}_test" \
    --template qwen2_vl \
    --cutoff_len "${INFER_CUTOFF}" \
    --max_new_tokens 10240 \
    --image_min_pixels 784 \
    --image_max_pixels "${INFER_MAX_PIXELS}" \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${CONFIG_NAME}"
echo "Model: ${CKPT_DIR}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
