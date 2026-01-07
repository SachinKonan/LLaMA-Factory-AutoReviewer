#!/bin/bash
#SBATCH --job-name=hp_vision
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --array=0-7
#SBATCH --output=logs/hyperparam_vision_sweep/%A_%a.out
#SBATCH --error=logs/hyperparam_vision_sweep/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 8 combinations: 2 batch sizes x 2 freeze_projector settings x 2 datasets
# Format: "BATCH_SIZE FREEZE_PROJ DATASET SHORT_NAME"
CONFIGS=(
    "16 true iclr_clean+images_binary_no_reviews_v3 clean_images"
    "16 true iclr_vision_binary_no_reviews_v3 vision"
    "16 false iclr_clean+images_binary_no_reviews_v3 clean_images"
    "16 false iclr_vision_binary_no_reviews_v3 vision"
    "32 true iclr_clean+images_binary_no_reviews_v3 clean_images"
    "32 true iclr_vision_binary_no_reviews_v3 vision"
    "32 false iclr_clean+images_binary_no_reviews_v3 clean_images"
    "32 false iclr_vision_binary_no_reviews_v3 vision"
)

# Parse hyperparameters from array index
read BATCH_SIZE FREEZE_PROJ DATASET SHORT_NAME <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Select config file and accelerate config based on batch size
if [ "$BATCH_SIZE" == "16" ]; then
    CONFIG_FILE="configs/hyperparam_vision_sweep_batch16.yaml"
    ACCEL_CONFIG="configs/fsdp2_2gpu_config.yaml"
    # Limit to 2 GPUs for batch16
    export CUDA_VISIBLE_DEVICES=0,1
else
    CONFIG_FILE="configs/hyperparam_vision_sweep_batch32.yaml"
    ACCEL_CONFIG="configs/fsdp2_4gpu_config.yaml"
fi

# Short name for freeze setting
if [ "$FREEZE_PROJ" == "true" ]; then
    FREEZE_SHORT="frozen"
else
    FREEZE_SHORT="unfrozen"
fi

# Directory names - unique port per job
CONFIG_NAME="bs${BATCH_SIZE}_proj_${FREEZE_SHORT}_${SHORT_NAME}"
MODEL_DIR="saves/qwen2.5-vl-7b/full/hyperparam_vision_sweep/${CONFIG_NAME}"
RESULTS_DIR="results/hyperparam_vision_sweep/${CONFIG_NAME}"
MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Config: ${CONFIG_NAME}"
echo "Batch Size: ${BATCH_SIZE}, Freeze Projector: ${FREEZE_PROJ}"
echo "Dataset: ${DATASET}"
echo "Accelerate Config: ${ACCEL_CONFIG}"
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
    --config_file "${ACCEL_CONFIG}" \
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

# Reset CUDA_VISIBLE_DEVICES for inference (use single GPU)
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
    --cutoff_len 20480 \
    --max_new_tokens 10240 \
    --image_min_pixels 784 \
    --image_max_pixels 1003520 \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${CONFIG_NAME}"
echo "Model: ${CKPT_DIR}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
