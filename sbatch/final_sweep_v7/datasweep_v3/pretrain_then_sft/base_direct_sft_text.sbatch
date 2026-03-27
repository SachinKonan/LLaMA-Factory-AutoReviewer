#!/bin/bash
#SBATCH --job-name=base_txt
#SBATCH --time=13:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --output=logs/pretrain_then_sft/base_direct_text/%A.out
#SBATCH --error=logs/pretrain_then_sft/base_direct_text/%A.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

BASE_MODEL="/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"
DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
CONFIG_FILE="configs/final_sweep_v7_clean.yaml"
TEMPLATE="qwen"
CUTOFF_LEN=24480

PER_DEVICE_BATCH=4
GRAD_ACCUM=2
NUM_GPUS=4

MODEL_DIR="saves/pretrain_then_sft/base_direct_text"
RESULTS_DIR="results/pretrain_then_sft/base_direct_text"
MASTER_PORT=29700

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Base Model: ${BASE_MODEL}"
echo "Dataset: ${DATASET}"
echo "LR: 1e-6"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Config: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/pretrain_then_sft/base_direct_text "${MODEL_DIR}" "${RESULTS_DIR}"

# Write trainjob touch file
echo "${SLURM_JOB_ID}" > "${MODEL_DIR}/.trainjob.touch"

# ======================
# Start auto-inference watcher in background (no --delete_safetensors_posteval)
# ======================
python scripts/auto_infer_watcher.py \
    --save_dir "${MODEL_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --dataset "balanced_nips_icml_iclr_text_eval_test" \
    --train_dataset "${DATASET}_train" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 1280 \
    --compute_train_accuracy \
    --train_accuracy_max_samples 2000 \
    --train_accuracy_batch_size 2 \
    --poll_interval 60 &
AUTO_INFER_PID=$!
echo "Started auto-inference watcher (PID: ${AUTO_INFER_PID})"

# ======================
# TRAINING (4 epochs)
# ======================
echo ""
echo "=== TRAINING ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_4gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py "${CONFIG_FILE}" \
    model_name_or_path="${BASE_MODEL}" \
    dataset="${DATASET}_train" \
    eval_dataset="${DATASET}_validation" \
    do_eval=false \
    output_dir="${MODEL_DIR}" \
    per_device_train_batch_size="${PER_DEVICE_BATCH}" \
    gradient_accumulation_steps="${GRAD_ACCUM}" \
    cutoff_len="${CUTOFF_LEN}" \
    learning_rate="1e-6" \
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
echo "COMPLETED: base_direct_text"
echo "=============================================="
