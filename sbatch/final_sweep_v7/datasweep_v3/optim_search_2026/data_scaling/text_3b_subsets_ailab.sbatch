#!/bin/bash
#SBATCH --job-name=ds_txt3b
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --array=0-1
#SBATCH --output=logs/final_sweep_v7_datasweepv3/optim_search_2026/text_3b_data_scaling/%A_%a.out
#SBATCH --error=logs/final_sweep_v7_datasweepv3/optim_search_2026/text_3b_data_scaling/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# BZ=32: 2 GPUs, per_device=4, grad_accum=4 → batch = 2*4*4 = 32
# Subset sweep: 50pct, 75pct (single LR = 1e-6, the chosen 100% baseline)
FRACTIONS=("50pct" "75pct")
SUBSET="${FRACTIONS[$SLURM_ARRAY_TASK_ID]}"
LR="1e-6"
SHORT_NAME="bz32_lr1e-6_text_3b_${SUBSET}"

BASE_DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
CONFIG_FILE="configs/final_sweep_v7_clean_3b.yaml"
TEMPLATE="qwen"
CUTOFF_LEN=24480

PER_DEVICE_BATCH=4
GRAD_ACCUM=4
NUM_GPUS=2

MODEL_DIR="saves/final_sweep_v7_datasweepv3/optim_search_2026/scaling/data_scaling/${SHORT_NAME}"
RESULTS_DIR="results/final_sweep_v7_datasweepv3/optim_search_2026/scaling/data_scaling/${SHORT_NAME}"
MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Subset: ${SUBSET}"
echo "Train Dataset: ${BASE_DATASET}_train_${SUBSET}"
echo "Short Name: ${SHORT_NAME}"
echo "LR: ${LR}"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Config: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/final_sweep_v7_datasweepv3/optim_search_2026/text_3b_data_scaling "${MODEL_DIR}" "${RESULTS_DIR}"

echo "${SLURM_JOB_ID}" > "${MODEL_DIR}/.trainjob.touch"

python scripts/auto_infer_watcher.py \
    --save_dir "${MODEL_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --dataset "${BASE_DATASET}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 1280 \
    --compute_train_accuracy \
    --delete_safetensors_posteval \
    --keep_safetensors_epoch_idx "2" \
    --train_accuracy_max_samples 2000 \
    --train_accuracy_batch_size 2 \
    --poll_interval 60 &
AUTO_INFER_PID=$!
echo "Started auto-inference watcher (PID: ${AUTO_INFER_PID})"

echo ""
echo "=== PHASE 1: TRAINING ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py "${CONFIG_FILE}" \
    dataset="${BASE_DATASET}_train_${SUBSET}" \
    eval_dataset="${BASE_DATASET}_validation" \
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

wait $AUTO_INFER_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
