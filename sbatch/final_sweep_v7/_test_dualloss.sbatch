#!/bin/bash
#SBATCH --job-name=test_dualloss
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu-test
#SBATCH --output=logs/final_sweep_v7/%j_test_dualloss.out
#SBATCH --error=logs/final_sweep_v7/%j_test_dualloss.err

# Test script for dual loss (decision + rating prediction)
# 4 GPUs, per_device=1, grad_accum=1 → batch = 1*4*1 = 4

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

# Model configuration
MODEL="Qwen/Qwen2.5-7B-Instruct"
TEMPLATE="qwen"
CUTOFF_LEN=24480

# Training configuration
LEARNING_RATE="1.75e-6"
WEIGHT_DECAY=0.015
LABEL_SMOOTHING_EPS=0.05
MAX_STEPS=5

# Multi-task learning configuration
ADD_PREDICT_RATINGS=true
RATING_LOSS_WEIGHT=0.1

# Batch configuration: 4 GPUs, per_device=1, grad_accum=1 → batch = 1*4*1 = 4
PER_DEVICE_BATCH=1
GRAD_ACCUM=1
NUM_GPUS=4

# Dataset
DATASET="iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7"
SHORT_NAME="test_dualloss"

# Output directories
MODEL_DIR="saves/final_sweep_v7/${SHORT_NAME}"
RESULTS_DIR="results/final_sweep_v7/${SHORT_NAME}"
MASTER_PORT=29750

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Label Smoothing Eps: ${LABEL_SMOOTHING_EPS}"
echo "Max Steps: ${MAX_STEPS}"
echo "Add Predict Ratings: ${ADD_PREDICT_RATINGS}"
echo "Rating Loss Weight: ${RATING_LOSS_WEIGHT}"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/final_sweep_v7 "${MODEL_DIR}" "${RESULTS_DIR}"

echo ""
echo "=== TEST: CLS TEXT TRAINING + RATING PREDICTION ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_4gpu_config.yaml \
    --main_process_port ${MASTER_PORT} \
    src/train.py \
    --stage cls \
    --model_name_or_path ${MODEL} \
    --dataset ${DATASET}_train \
    --eval_dataset ${DATASET}_test \
    --template ${TEMPLATE} \
    --output_dir ${MODEL_DIR} \
    --predictions_dir ${RESULTS_DIR} \
    --cutoff_len ${CUTOFF_LEN} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH} \
    --per_device_eval_batch_size ${PER_DEVICE_BATCH} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --label_smoothing_eps ${LABEL_SMOOTHING_EPS} \
    --add_predict_ratings ${ADD_PREDICT_RATINGS} \
    --rating_loss_weight ${RATING_LOSS_WEIGHT} \
    --max_steps ${MAX_STEPS} \
    --lr_scheduler_type cosine \
    --warmup_steps 0 \
    --logging_steps 1 \
    --save_strategy no \
    --eval_strategy steps \
    --eval_steps 5 \
    --eval_on_start true \
    --bf16 true \
    --report_to none \
    --do_train \
    --do_predict \
    --finetuning_type full \
    --gradient_checkpointing true \
    --overwrite_cache true \
    --ddp_timeout 180000000 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 2

echo ""
echo "=============================================="
echo "TEST COMPLETED: ${SHORT_NAME}"
echo "=============================================="
