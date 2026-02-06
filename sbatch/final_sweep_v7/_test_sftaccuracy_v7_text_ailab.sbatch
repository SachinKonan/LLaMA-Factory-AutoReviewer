#!/bin/bash
#SBATCH --job-name=test_sft_acc
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-test
#SBATCH --output=logs/test_sft_accuracy/%j.out
#SBATCH --error=logs/test_sft_accuracy/%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

# Test config
MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7"
TEMPLATE="qwen"
CUTOFF_LEN=24480

# Batch: 1 per GPU * 2 GPUs * 4 grad_accum = 8 effective batch
PER_DEVICE_BATCH=1
GRAD_ACCUM=4
NUM_GPUS=2

MODEL_DIR="saves/test_sft_accuracy_2"
MASTER_PORT=29700

echo "=============================================="
echo "Test: SFT Train Accuracy"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Max samples: 32"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "=============================================="

mkdir -p logs/test_sft_accuracy "${MODEL_DIR}"

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py \
    --stage sft \
    --model_name_or_path "${MODEL}" \
    --dataset "${DATASET}_train" \
    --template "${TEMPLATE}" \
    --output_dir "${MODEL_DIR}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_samples 32 \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate 2e-6 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_steps 1 \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 true \
    --report_to none \
    --do_train \
    --finetuning_type full \
    --gradient_checkpointing true \
    --overwrite_cache true \
    --ddp_timeout 180000000 \
    --preprocessing_num_workers 4 \
    --sft_train_accuracy true \
    --sft_train_accuracy_format boxed \
    --sft_positive_token Accept \
    --sft_negative_token Reject

echo ""
echo "=============================================="
echo "COMPLETED: SFT Train Accuracy Test"
echo "=============================================="
