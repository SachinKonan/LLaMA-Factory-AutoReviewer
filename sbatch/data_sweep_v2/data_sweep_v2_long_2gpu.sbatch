#!/bin/bash
#SBATCH --job-name=dsv2_long_2gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=170G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --array=0
#SBATCH --output=logs/data_sweep_v2_long/%A_%a.out
#SBATCH --error=logs/data_sweep_v2_long/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Single config: index 0 (iclr17_balanced clean)
DATASET="iclr_2017_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6"
SHORT_NAME="iclr17_balanced"
MODALITY="clean"

CONFIG_FILE="configs/data_sweep_v2_clean.yaml"
TEMPLATE="qwen"
IMAGE_MAX_PIXELS=0

# 2 GPUs, per_device=2, grad_accum=4 → batch = 2*2*4 = 16
PER_DEVICE_BATCH=2
GRAD_ACCUM=4
NUM_GPUS=2

MODEL_DIR="saves/data_sweep_v2_long/${SHORT_NAME}_${MODALITY}"
RESULTS_DIR="results/data_sweep_v2_long/${SHORT_NAME}_${MODALITY}"
MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}_${MODALITY}"
echo "GPUs: ${NUM_GPUS} (H200), per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Config: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "=============================================="

mkdir -p logs/data_sweep_v2_long "${MODEL_DIR}" "${RESULTS_DIR}"

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
    gradient_accumulation_steps="${GRAD_ACCUM}"

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
    --cutoff_len 24480 \
    --max_new_tokens 1280 \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}_${MODALITY}"
echo "=============================================="
