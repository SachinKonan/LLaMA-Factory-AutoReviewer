#!/bin/bash
#SBATCH --job-name=arxiv_train_large_vision
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --output=logs/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/large/vision_arxiv_balanced_per_venue/%j.out
#SBATCH --error=logs/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/large/vision_arxiv_balanced_per_venue/%j.err

# Train Qwen2.5-VL-7B (vision) on the LARGE arxiv dataset (71k balanced_per_venue).
# 3 epochs, ~36h walltime. Auto-inference NOT run inside this sbatch.

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# BZ=16: 4 GPUs, per_device=4, grad_accum=1 -> batch = 16 (matches ratio_clean).
LR="1e-6"
SHORT_NAME="arxiv_balanced_per_venue_vision"

TRAIN_DATASET="arxiv_50_50_balanced_per_venue_vision_wmetadata_train"
EVAL_DATASET="arxiv_50_50_balanced_per_venue_vision_wmetadata_validation"

CONFIG_FILE="configs/final_sweep_v7_vision.yaml"
TEMPLATE="qwen2_vl"
CUTOFF_LEN=24480
IMAGE_MIN_PIXELS=784
IMAGE_MAX_PIXELS=1003520

PER_DEVICE_BATCH=4
GRAD_ACCUM=1
NUM_GPUS=4

MODEL_DIR="saves/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/large/${SHORT_NAME}"
RESULTS_DIR="results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/large/${SHORT_NAME}"
MASTER_PORT=29773

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Train dataset: ${TRAIN_DATASET} (~71k samples)"
echo "Short Name: ${SHORT_NAME}"
echo "LR: ${LR}"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Image pixels: min=${IMAGE_MIN_PIXELS} max=${IMAGE_MAX_PIXELS}"
echo "Epochs: 3 (instead of 4) due to ~3.4x larger dataset"
echo "Model Dir: ${MODEL_DIR}"
echo "Note: auto-inference is NOT run by this sbatch; sk7524's watcher handles it."
echo "=============================================="

mkdir -p "${MODEL_DIR}" "${RESULTS_DIR}"
echo "${SLURM_JOB_ID}" > "${MODEL_DIR}/.trainjob.touch"

# ======================
# TRAINING (3 epochs, ~36h walltime)
# ======================
echo ""
echo "=== TRAINING (3 epochs) ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_4gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py "${CONFIG_FILE}" \
    dataset="${TRAIN_DATASET}" \
    eval_dataset="${EVAL_DATASET}" \
    do_eval=false \
    output_dir="${MODEL_DIR}" \
    per_device_train_batch_size="${PER_DEVICE_BATCH}" \
    gradient_accumulation_steps="${GRAD_ACCUM}" \
    cutoff_len="${CUTOFF_LEN}" \
    learning_rate="${LR}" \
    num_train_epochs=3.0 \
    lr_scheduler_type=cosine \
    'lr_scheduler_kwargs={"custom_scheduler": "cosine_then_constant", "decay_ratio": 0.75, "min_lr_rate": 0.001}'

echo ""
echo "=============================================="
echo "COMPLETED training: ${SHORT_NAME}"
echo "=============================================="
