#!/bin/bash
#SBATCH --job-name=test_noise_em_text
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-test
#SBATCH --output=logs/testing_cls/%j_noise_em_text.out
#SBATCH --error=logs/testing_cls/%j_noise_em_text.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

# Test: Noise-aware loss, EM strategy, text model
DATASET="iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7"
SHORT_NAME="test_noise_em_text"

TEMPLATE="qwen"
CUTOFF_LEN=1024
PER_DEVICE_BATCH=1
GRAD_ACCUM=4
NUM_GPUS=2

MODEL_DIR="saves/testing_cls/${SHORT_NAME}"
MASTER_PORT=29700

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Test: Noise-aware EM text"
echo "Dataset: ${DATASET}"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "=============================================="

mkdir -p logs/testing_cls "${MODEL_DIR}"

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py \
    --stage cls \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset "${DATASET}_train" \
    --template "${TEMPLATE}" \
    --output_dir "${MODEL_DIR}" \
    --max_samples 1000 \
    --cutoff_len "${CUTOFF_LEN}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate 1e-5 \
    --max_steps 10 \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 true \
    --do_train \
    --finetuning_type full \
    --gradient_checkpointing true \
    --noise_aware_loss true \
    --noise_strategy em \
    --noise_em_model_epochs 1.0 \
    --noise_em_noise_epochs 0.5

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
