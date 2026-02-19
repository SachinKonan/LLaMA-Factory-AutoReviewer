#!/bin/bash
#SBATCH --job-name=optim_text_bz64
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --array=0-1
#SBATCH --output=logs/final_sweep_v7_datasweepv3/optim_search/text_bz64/%A_%a.out
#SBATCH --error=logs/final_sweep_v7_datasweepv3/optim_search/text_bz64/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# BZ=64: 4 GPUs, per_device=4, grad_accum=4 → batch = 4*4*4 = 64
# LR grid: [2e-6, 4e-6]
LRS=("2e-6" "4e-6")
SHORT_NAMES=("bz64_lr2e-6_text" "bz64_lr4e-6_text")

LR="${LRS[$SLURM_ARRAY_TASK_ID]}"
SHORT_NAME="${SHORT_NAMES[$SLURM_ARRAY_TASK_ID]}"

DATASET="iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered"
CONFIG_FILE="configs/final_sweep_v7_clean.yaml"
TEMPLATE="qwen"
CUTOFF_LEN=24480

PER_DEVICE_BATCH=4
GRAD_ACCUM=4
NUM_GPUS=4

MODEL_DIR="saves/final_sweep_v7_datasweepv3/optim_search/${SHORT_NAME}"
RESULTS_DIR="results/final_sweep_v7_datasweepv3/optim_search/${SHORT_NAME}"
MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "LR: ${LR}"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Config: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/final_sweep_v7_datasweepv3/optim_search/text_bz64 "${MODEL_DIR}" "${RESULTS_DIR}"

# Write trainjob touch file
echo "${SLURM_JOB_ID}" > "${MODEL_DIR}/.trainjob.touch"

# ======================
# Start auto-inference watcher in background
# ======================
python scripts/auto_infer_watcher.py \
    --save_dir "${MODEL_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --dataset "${DATASET}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 1280 \
    --compute_train_accuracy \
    --delete_safetensors_posteval \
    --train_accuracy_max_samples 2000 \
    --train_accuracy_batch_size 2 \
    --poll_interval 60 &
AUTO_INFER_PID=$!
echo "Started auto-inference watcher (PID: ${AUTO_INFER_PID})"

# ======================
# PHASE 1: TRAINING (3 epochs)
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
    do_eval=false \
    output_dir="${MODEL_DIR}" \
    per_device_train_batch_size="${PER_DEVICE_BATCH}" \
    gradient_accumulation_steps="${GRAD_ACCUM}" \
    cutoff_len="${CUTOFF_LEN}" \
    learning_rate="${LR}" \
    num_train_epochs=3.0

echo ""
echo "=== Training complete! ==="
echo ""

# ======================
# PHASE 2: INFERENCE
# ======================
echo ""
echo "=== PHASE 2: INFERENCE ==="
echo ""

CKPT_DIR=$(for d in ${MODEL_DIR}/checkpoint-*/; do basename "$d"; done | sed 's/checkpoint-//' | sort -n | tail -1 | xargs -I{} echo "${MODEL_DIR}/checkpoint-{}")

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
    --temperature 0 \
    --save_logprobs \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

# Kill auto-inference watcher if still running
kill $AUTO_INFER_PID 2>/dev/null || true

# Cleanup safetensors from final checkpoint
if [ -d "$CKPT_DIR" ]; then
    SAFETENSOR_COUNT=$(ls "${CKPT_DIR}"/*.safetensors 2>/dev/null | wc -l)
    if [ "$SAFETENSOR_COUNT" -gt 0 ]; then
        rm -f "${CKPT_DIR}"/*.safetensors
        touch "${CKPT_DIR}/.cleaned.touch"
        echo "Cleaned ${SAFETENSOR_COUNT} safetensors file(s) from final checkpoint"
    fi
fi

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
