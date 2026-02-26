#!/bin/bash
#SBATCH --job-name=noise_em_txt
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --output=logs/lr_experiment_v7/noise_aware/%A_%a.out
#SBATCH --error=logs/lr_experiment_v7/noise_aware/%A_%a.err
#SBATCH --array=0-1

# Text CLS training: noise-aware loss, EM strategy
# Array job: 0=trainagreeing, 1=balanced (both exclude 2024)

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
LEARNING_RATE="2e-6"
WEIGHT_DECAY=0.001
NUM_EPOCHS=3

# Batch configuration: 2 GPUs, per_device=2, grad_accum=4 → batch = 2*2*4 = 16
PER_DEVICE_BATCH=2
GRAD_ACCUM=4
NUM_GPUS=2

# Dataset selection based on array task ID
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    DATASET="iclr_2020_2023_2025_85_5_10_trainagreeing_original_text_v7_filtered"
    SHORT_NAME="trainagreeing_no2024_noise_em"
else
    DATASET="iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered"
    SHORT_NAME="balanced_no2024_noise_em"
fi

# Output directories
MODEL_DIR="saves/lr_experiment_v7/text_${SHORT_NAME}"
RESULTS_DIR="results/lr_experiment_v7/text_${SHORT_NAME}"
MASTER_PORT=$((29702 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Noise Strategy: em (model=0.5 epoch, noise=0.05 epoch)"
echo "Epochs: ${NUM_EPOCHS}"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/lr_experiment_v7/noise_aware "${MODEL_DIR}" "${RESULTS_DIR}"

echo ""
echo "=== CLS TEXT TRAINING (NOISE EM) ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
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
    --noise_aware_loss true \
    --noise_alpha_init 0.506 \
    --noise_beta_init 0.15 \
    --noise_strategy em \
    --noise_em_model_epochs 0.5 \
    --noise_em_noise_epochs 0.05 \
    --num_train_epochs ${NUM_EPOCHS} \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --eval_strategy epoch \
    --eval_on_start true \
    --bf16 true \
    --report_to none \
    --do_train \
    --do_predict \
    --finetuning_type full \
    --save_only_model true \
    --gradient_checkpointing true \
    --plot_loss true \
    --overwrite_cache true \
    --ddp_timeout 180000000 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4

echo ""
echo "=============================================="
echo "COMPLETED: text_${SHORT_NAME}"
echo "=============================================="
