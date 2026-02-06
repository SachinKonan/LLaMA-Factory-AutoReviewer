#!/bin/bash
#SBATCH --job-name=test_think_tokens
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-test
#SBATCH --output=logs/test_think_tokens/%j.out
#SBATCH --error=logs/test_think_tokens/%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

# Test config - TEXT model for faster testing
MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7"
TEMPLATE="qwen"
CUTOFF_LEN=24480

# Think token config
THINK_NUM_TOKENS=10
THINK_PLACEMENT="input"  # or "label"

# Batch: 1 per GPU * 2 GPUs * 4 grad_accum = 8 effective batch
PER_DEVICE_BATCH=1
GRAD_ACCUM=4
NUM_GPUS=2

MODEL_DIR="saves/test_think_tokens"
MASTER_PORT=29710

echo "=============================================="
echo "Test: Think Tokens Feature"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Think Tokens: ${THINK_NUM_TOKENS}"
echo "Think Placement: ${THINK_PLACEMENT}"
echo "Max samples: 32"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "=============================================="

mkdir -p logs/test_think_tokens "${MODEL_DIR}"

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
    --think_num_tokens "${THINK_NUM_TOKENS}" \
    --think_token_placement "${THINK_PLACEMENT}" \
    --sft_train_accuracy true \
    --sft_train_accuracy_format boxed \
    --sft_positive_token Accept \
    --sft_negative_token Reject

echo ""
echo "=============================================="
echo "COMPLETED: Think Tokens Test"
echo "=============================================="
echo ""
echo "Check the output for:"
echo "1. 'Think tokens enabled: ${THINK_NUM_TOKENS} x <|think|> at ${THINK_PLACEMENT}'"
echo "2. 'Add special tokens <|think|>' (NOT split into individual chars)"
echo "3. Think token should appear in input_ids (decoded output)"
echo "4. If placement=input: think token masked in labels"
echo "5. If placement=label: think token visible in labels"
