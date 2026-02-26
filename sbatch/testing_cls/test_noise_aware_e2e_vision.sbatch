#!/bin/bash
#SBATCH --job-name=test_noise_e2e_vision
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=350G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-test
#SBATCH --output=logs/testing_cls/%j_noise_e2e_vision.out
#SBATCH --error=logs/testing_cls/%j_noise_e2e_vision.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

# Test: Noise-aware loss, end-to-end strategy, vision model
DATASET="iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7"
SHORT_NAME="test_noise_e2e_vision"

TEMPLATE="qwen2_vl"
CUTOFF_LEN=24480
IMAGE_MIN_PIXELS=784
IMAGE_MAX_PIXELS=1003520
PER_DEVICE_BATCH=1
GRAD_ACCUM=4
NUM_GPUS=2

MODEL_DIR="saves/testing_cls/${SHORT_NAME}"
MASTER_PORT=29700

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Test: Noise-aware E2E vision"
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
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset "${DATASET}_train" \
    --template "${TEMPLATE}" \
    --output_dir "${MODEL_DIR}" \
    --max_samples 1000 \
    --cutoff_len "${CUTOFF_LEN}" \
    --image_min_pixels "${IMAGE_MIN_PIXELS}" \
    --image_max_pixels "${IMAGE_MAX_PIXELS}" \
    --media_dir /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/images \
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
    --freeze_vision_tower true \
    --freeze_multi_modal_projector true \
    --noise_aware_loss true \
    --noise_alpha_init 0.506 \
    --noise_beta_init 0.15

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
