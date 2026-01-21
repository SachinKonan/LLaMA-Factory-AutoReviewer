#!/bin/bash
#SBATCH --job-name=nips_acc_24
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=250G
#SBATCH --gres=gpu:4
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --array=0-2
#SBATCH --output=logs/data_sweep_nips_accepts_24/%A_%a.out
#SBATCH --error=logs/data_sweep_nips_accepts_24/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 3 configurations: clean, clean_images, vision
# Dataset: nips_accepts from 2024-2025 (iclr original + nips accepts)
# Format: "DATASET_BASE SHORT_NAME CONFIG_FILE TEMPLATE"
CONFIGS=(
    "iclr_nips_2024_2025_85_5_10_split6_balanced_nips_accepts_clean_binary_noreviews_v6 nips_acc24_clean configs/data_sweep_v2_clean.yaml qwen2.5"
    "iclr_nips_2024_2025_85_5_10_split6_balanced_nips_accepts_clean_images_binary_noreviews_v6 nips_acc24_clean_images configs/data_sweep_clean_images.yaml qwen2_vl"
    "iclr_nips_2024_2025_85_5_10_split6_balanced_nips_accepts_vision_binary_noreviews_v6 nips_acc24_vision configs/data_sweep_v2_vision.yaml qwen2_vl"
)

read DATASET SHORT_NAME CONFIG_FILE TEMPLATE <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Determine if vision model (for inference settings)
if [[ "$TEMPLATE" == "qwen2_vl" ]]; then
    IMAGE_MAX_PIXELS=1003520
else
    IMAGE_MAX_PIXELS=0
fi
CUTOFF_LEN=24480

# 4 H100 GPUs, per_device=2, grad_accum=2 → batch = 2*4*2 = 16
PER_DEVICE_BATCH=2
GRAD_ACCUM=2
NUM_GPUS=4

MODEL_DIR="saves/data_sweep_nips_accepts_24/${SHORT_NAME}"
RESULTS_DIR="results/data_sweep_nips_accepts_24/${SHORT_NAME}"
MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "Template: ${TEMPLATE}"
echo "GPUs: ${NUM_GPUS} (H100), per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Config: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "=============================================="

mkdir -p logs/data_sweep_nips_accepts_24 "${MODEL_DIR}" "${RESULTS_DIR}"

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
    eval_dataset="${DATASET}_validation" \
    output_dir="${MODEL_DIR}" \
    per_device_train_batch_size="${PER_DEVICE_BATCH}" \
    gradient_accumulation_steps="${GRAD_ACCUM}" \
    num_train_epochs=4 \
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

if [[ "$TEMPLATE" == "qwen2_vl" ]]; then
    python scripts/vllm_infer.py \
        --model_name_or_path "${CKPT_DIR}" \
        --dataset "${DATASET}_test" \
        --template "${TEMPLATE}" \
        --cutoff_len "${CUTOFF_LEN}" \
        --max_new_tokens 1280 \
        --image_min_pixels 784 \
        --image_max_pixels "${IMAGE_MAX_PIXELS}" \
        --save_name "${RESULTS_DIR}/finetuned.jsonl"
else
    python scripts/vllm_infer.py \
        --model_name_or_path "${CKPT_DIR}" \
        --dataset "${DATASET}_test" \
        --template "${TEMPLATE}" \
        --cutoff_len "${CUTOFF_LEN}" \
        --max_new_tokens 1280 \
        --save_name "${RESULTS_DIR}/finetuned.jsonl"
fi

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
