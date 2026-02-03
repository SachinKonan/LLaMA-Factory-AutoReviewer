#!/bin/bash
#SBATCH --job-name=v8_rat_txt
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --array=0-1
#SBATCH --output=logs/testing_v8/%A_%a.out
#SBATCH --error=logs/testing_v8/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 2 text ratingbinary v8 datasets (year gap ratingbinary not available yet)
CONFIGS=(
    "2020_2025_train_2020_2026_valtest_clean_ratingbinary_noreviews_v8 ood_text_rating"
    "2020_2026_balanced_clean_ratingbinary_noreviews_v8 balanced_text_rating"
)

read DATASET SHORT_NAME <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

CONFIG_FILE="configs/testing_v8_clean.yaml"
TEMPLATE="qwen"
CUTOFF_LEN=24480

# 2 H200 GPUs, per_device=2, grad_accum=4 -> batch = 2*2*4 = 16
PER_DEVICE_BATCH=2
GRAD_ACCUM=4
LEARNING_RATE="2.0e-6"
NUM_GPUS=2

MODEL_DIR="saves/testing_v8/${SHORT_NAME}"
RESULTS_DIR="results/testing_v8/${SHORT_NAME}"
MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID + 10))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "GPUs: ${NUM_GPUS} (H200), per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}, LR: ${LEARNING_RATE}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Config: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "=============================================="

mkdir -p logs/testing_v8 "${MODEL_DIR}" "${RESULTS_DIR}"

# ======================
# PHASE 1: TRAINING
# ======================
echo ""
echo "=== PHASE 1: TRAINING ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py "${CONFIG_FILE}" \
    dataset="${DATASET}_train" \
    eval_dataset="${DATASET}_validation" \
    output_dir="${MODEL_DIR}" \
    per_device_train_batch_size="${PER_DEVICE_BATCH}" \
    gradient_accumulation_steps="${GRAD_ACCUM}" \
    learning_rate="${LEARNING_RATE}" \
    cutoff_len="${CUTOFF_LEN}"

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

python scripts/vllm_infer.py \
    --model_name_or_path "${CKPT_DIR}" \
    --dataset "${DATASET}_test" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 1280 \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
