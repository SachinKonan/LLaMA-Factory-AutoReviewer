#!/bin/bash
#SBATCH --job-name=wd_text
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --array=0-2
#SBATCH --output=logs/final_sweep_v7_datasweepv3/wd_sweep/text/%A_%a.out
#SBATCH --error=logs/final_sweep_v7_datasweepv3/wd_sweep/text/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Weight decay sweep: 0, 0.001, 0.01
# Fixed: bz16, lr=1e-6, 6 epochs, cosine_then_constant (decay_ratio=0.5, min_lr_rate=0.001)
WDS=("0" "0.001" "0.01")
SHORT_NAMES=("bz16_lr1e-6_wd0_text" "bz16_lr1e-6_wd0.001_text" "bz16_lr1e-6_wd0.01_text")

WD="${WDS[$SLURM_ARRAY_TASK_ID]}"
SHORT_NAME="${SHORT_NAMES[$SLURM_ARRAY_TASK_ID]}"

DATASET="iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered"
CONFIG_FILE="configs/final_sweep_v7_clean.yaml"
TEMPLATE="qwen"
CUTOFF_LEN=24480

# BZ=16: 4 GPUs, per_device=4, grad_accum=1 -> batch = 4*4*1 = 16
PER_DEVICE_BATCH=4
GRAD_ACCUM=1
NUM_GPUS=4

MODEL_DIR="saves/final_sweep_v7_datasweepv3/wd_sweep/${SHORT_NAME}"
RESULTS_DIR="results/final_sweep_v7_datasweepv3/wd_sweep/${SHORT_NAME}"
MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "LR: 1e-6, Weight Decay: ${WD}"
echo "Scheduler: cosine_then_constant (decay_ratio=0.5, min_lr_rate=0.001)"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Config: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/final_sweep_v7_datasweepv3/wd_sweep/text "${MODEL_DIR}" "${RESULTS_DIR}"

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
# PHASE 1: TRAINING (6 epochs)
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
    learning_rate=1e-6 \
    num_train_epochs=6.0 \
    lr_scheduler_type=cosine \
    'lr_scheduler_kwargs={"custom_scheduler": "cosine_then_constant", "decay_ratio": 0.5, "min_lr_rate": 0.001}' \
    weight_decay="${WD}"

echo ""
echo "=== Training complete! Waiting for auto-inference watcher to finish... ==="
echo ""

# Wait for auto-inference watcher to process final checkpoint
wait $AUTO_INFER_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
