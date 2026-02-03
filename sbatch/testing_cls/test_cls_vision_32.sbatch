#!/bin/bash
#SBATCH --job-name=test_cls_vision_32
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=350G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --output=logs/testing_cls/%j.out
#SBATCH --error=logs/testing_cls/%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

# Test CLS training with vision model (32 samples)
DATASET="iclr_2020_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7"
SHORT_NAME="test_cls_vision_32"

TEMPLATE="qwen2_vl"
CUTOFF_LEN=2048
MAX_SAMPLES=32

# 2 GPUs, per_device=4, grad_accum=2 → batch = 4*2*2 = 16
PER_DEVICE_BATCH=4
GRAD_ACCUM=2
NUM_GPUS=2

SAVE_DIR="saves/testing_v8/${SHORT_NAME}"
RESULTS_DIR="results/testing_v8/${SHORT_NAME}"
MASTER_PORT=29700

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "Max Samples: ${MAX_SAMPLES}"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Save Dir: ${SAVE_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/testing_cls "${SAVE_DIR}" "${RESULTS_DIR}"

echo ""
echo "=== CLS VISION TRAINING (32 samples) + PREDICT ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py \
    --stage cls \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset "${DATASET}_train" \
    --eval_dataset "${DATASET}_test" \
    --template "${TEMPLATE}" \
    --output_dir "${SAVE_DIR}" \
    --predictions_dir "${RESULTS_DIR}" \
    --max_samples "${MAX_SAMPLES}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --per_device_eval_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 true \
    --do_train \
    --do_predict \
    --finetuning_type full \
    --gradient_checkpointing true \
    --overwrite_cache true

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
