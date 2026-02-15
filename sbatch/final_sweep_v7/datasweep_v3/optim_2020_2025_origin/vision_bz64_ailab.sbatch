#!/bin/bash
#SBATCH --job-name=o25_vis64
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --array=0-1
#SBATCH --output=logs/final_sweep_v7_datasweepv3/optim_2020_2025_origin/vision_bz64/%A_%a.out
#SBATCH --error=logs/final_sweep_v7_datasweepv3/optim_2020_2025_origin/vision_bz64/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# BZ=64: 4 GPUs, per_device=8, grad_accum=2 → batch = 4*8*2 = 64
# LR grid: [4e-6, 5.5e-6]
LRS=("4e-6" "5.5e-6")
SHORT_NAMES=("bz64_lr4e-6_vision" "bz64_lr5.5e-6_vision")

LR="${LRS[$SLURM_ARRAY_TASK_ID]}"
SHORT_NAME="${SHORT_NAMES[$SLURM_ARRAY_TASK_ID]}"

DATASET="iclr_2020_2025_85_5_10_balanced_original_vision_corrected_v7_filtered"
CONFIG_FILE="configs/final_sweep_v7_vision.yaml"
TEMPLATE="qwen2_vl"
CUTOFF_LEN=24480
IMAGE_MIN_PIXELS=784
IMAGE_MAX_PIXELS=1003520

PER_DEVICE_BATCH=8
GRAD_ACCUM=2
NUM_GPUS=4

MODEL_DIR="saves/final_sweep_v7_datasweepv3/optim_2020_2025_origin/${SHORT_NAME}"
RESULTS_DIR="results/final_sweep_v7_datasweepv3/optim_2020_2025_origin/${SHORT_NAME}"
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

mkdir -p logs/final_sweep_v7_datasweepv3/optim_2020_2025_origin/vision_bz64 "${MODEL_DIR}" "${RESULTS_DIR}"

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
    --image_min_pixels "${IMAGE_MIN_PIXELS}" \
    --image_max_pixels "${IMAGE_MAX_PIXELS}" \
    --compute_train_accuracy \
    --delete_safetensors_posteval \
    --keep_safetensors_epoch_idx "2" \
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
    num_train_epochs=4.0 \
    lr_scheduler_type=cosine \
    'lr_scheduler_kwargs={"custom_scheduler": "cosine_then_constant", "decay_ratio": 0.75, "min_lr_rate": 0.001}'

echo ""
echo "=== Training complete! Waiting for auto-inference watcher to finish... ==="
echo ""

# Wait for auto-inference watcher to process final checkpoint
wait $AUTO_INFER_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
