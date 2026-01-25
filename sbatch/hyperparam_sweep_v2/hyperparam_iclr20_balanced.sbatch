#!/bin/bash
#SBATCH --job-name=hp_iclr20bal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --array=0-1
#SBATCH --output=logs/hyperparam_iclr20_balanced/%A_%a.out
#SBATCH --error=logs/hyperparam_iclr20_balanced/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Dataset: iclr20_balanced (index 3 from data_sweep_v2)
DATASET_CLEAN="iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6"
DATASET_VISION="iclr_2020_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6"

# 2 configurations: clean and vision
# Using same lr/batch as hyperparam_sweep_pli_v2 but with 3 epochs, 0 warmup
CONFIGS=(
    "clean"
    "vision"
)

MODALITY="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Set config and dataset based on modality
if [ "$MODALITY" = "clean" ]; then
    CONFIG_FILE="configs/data_sweep_v2_clean.yaml"
    DATASET="$DATASET_CLEAN"
    TEMPLATE="qwen"
    IMAGE_MAX_PIXELS=0
else
    CONFIG_FILE="configs/data_sweep_v2_vision.yaml"
    DATASET="$DATASET_VISION"
    TEMPLATE="qwen2_vl"
    IMAGE_MAX_PIXELS=1003520
fi

# Hyperparams: lr=3e-6, batch=32 (4 GPUs * 2 per_device * 4 grad_accum)
LR="3.0e-6"
PER_DEVICE_BATCH=2
GRAD_ACCUM=4
NUM_EPOCHS=3
WARMUP_STEPS=0

BATCH_SIZE=$((PER_DEVICE_BATCH * 4 * GRAD_ACCUM))
SHORT_NAME="iclr20bal_lr${LR}_b${BATCH_SIZE}_e${NUM_EPOCHS}_w${WARMUP_STEPS}_${MODALITY}"

MODEL_DIR="saves/hyperparam_iclr20_balanced/${SHORT_NAME}"
RESULTS_DIR="results/hyperparam_iclr20_balanced/${SHORT_NAME}"
MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: iclr20_balanced"
echo "Modality: ${MODALITY}"
echo "LR: ${LR}, Batch: ${BATCH_SIZE}, Epochs: ${NUM_EPOCHS}, Warmup: ${WARMUP_STEPS}"
echo "GPUs: 4 (H100), per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Config: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "=============================================="

mkdir -p logs/hyperparam_iclr20_balanced "${MODEL_DIR}" "${RESULTS_DIR}"

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
    output_dir="${MODEL_DIR}" \
    learning_rate="${LR}" \
    per_device_train_batch_size="${PER_DEVICE_BATCH}" \
    gradient_accumulation_steps="${GRAD_ACCUM}" \
    num_train_epochs="${NUM_EPOCHS}" \
    warmup_steps="${WARMUP_STEPS}" \
    save_total_limit=3

echo ""
echo "=== Training complete! ==="
echo ""

# ======================
# PHASE 2: INFERENCE
# ======================
echo ""
echo "=== PHASE 2: INFERENCE ==="
echo ""

CKPT_DIR=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "Using checkpoint: ${CKPT_DIR}"

if [ "$IMAGE_MAX_PIXELS" -gt 0 ]; then
    python scripts/vllm_infer.py \
        --model_name_or_path "${CKPT_DIR}" \
        --dataset "${DATASET}_test" \
        --template "${TEMPLATE}" \
        --cutoff_len 24480 \
        --max_new_tokens 1280 \
        --image_min_pixels 784 \
        --image_max_pixels "${IMAGE_MAX_PIXELS}" \
        --save_name "${RESULTS_DIR}/finetuned.jsonl"
else
    python scripts/vllm_infer.py \
        --model_name_or_path "${CKPT_DIR}" \
        --dataset "${DATASET}_test" \
        --template "${TEMPLATE}" \
        --cutoff_len 24480 \
        --max_new_tokens 1280 \
        --save_name "${RESULTS_DIR}/finetuned.jsonl"
fi

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
