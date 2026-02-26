#!/bin/bash
#SBATCH --job-name=test_cls_text_1k
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --output=logs/testing_cls/%j.out
#SBATCH --error=logs/testing_cls/%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

# Test CLS training with text model (1000 samples)
DATASET="iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7"
SHORT_NAME="test_cls_text_1k"

TEMPLATE="qwen"
CUTOFF_LEN=2048
MAX_SAMPLES=1000

# 2 GPUs, per_device=1, grad_accum=1 → batch = 1*2*1 = 2
PER_DEVICE_BATCH=1
GRAD_ACCUM=1
NUM_GPUS=2

MODEL_DIR="saves/testing_cls/${SHORT_NAME}"
MASTER_PORT=29700

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "Max Samples: ${MAX_SAMPLES}"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Model Dir: ${MODEL_DIR}"
echo "=============================================="

mkdir -p logs/testing_cls "${MODEL_DIR}"

echo ""
echo "=== CLS TEXT TRAINING (1k samples) ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py \
    --stage cls \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset "${DATASET}_train" \
    --template "${TEMPLATE}" \
    --output_dir "${MODEL_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_strategy no \
    --bf16 true \
    --do_train \
    --finetuning_type full \
    --gradient_checkpointing true

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
