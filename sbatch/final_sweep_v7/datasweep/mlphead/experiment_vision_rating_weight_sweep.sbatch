#!/bin/bash
#SBATCH --job-name=rw_vis
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=350G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --output=logs/lr_experiment_v7/rating_weight_sweep/%A_%a.out
#SBATCH --error=logs/lr_experiment_v7/rating_weight_sweep/%A_%a.err
#SBATCH --array=0-2

# Vision CLS training: LR=2e-6, sweep rating prediction
# Array: 0=no rating, 1=rating_loss_weight=0.001, 2=rating_loss_weight=0.01
# Dataset: iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

# Model configuration
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
TEMPLATE="qwen2_vl"
CUTOFF_LEN=24480
IMAGE_MIN_PIXELS=784
IMAGE_MAX_PIXELS=1003520

# Training configuration
LEARNING_RATE="2e-6"
WEIGHT_DECAY=0.015
LABEL_SMOOTHING_EPS=0.05
NUM_EPOCHS=4

# Rating sweep configuration
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    RATING_FLAGS=""
    SHORT_NAME="balanced_vision_lr2e6_norating"
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    RATING_FLAGS="--add_predict_ratings true --rating_loss_weight 0.001"
    SHORT_NAME="balanced_vision_lr2e6_rw0.001"
else
    RATING_FLAGS="--add_predict_ratings true --rating_loss_weight 0.01"
    SHORT_NAME="balanced_vision_lr2e6_rw0.01"
fi

# Batch configuration: 2 GPUs, per_device=2, grad_accum=4 → batch = 2*2*4 = 16
PER_DEVICE_BATCH=2
GRAD_ACCUM=4
NUM_GPUS=2

DATASET="iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered"

# Output directories
MODEL_DIR="saves/lr_experiment_v7/vision_${SHORT_NAME}"
RESULTS_DIR="results/lr_experiment_v7/vision_${SHORT_NAME}"
MASTER_PORT=$((29703 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Weight Decay: ${WEIGHT_DECAY}"
echo "Label Smoothing Eps: ${LABEL_SMOOTHING_EPS}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Rating Flags: ${RATING_FLAGS:-none}"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Image pixels: min=${IMAGE_MIN_PIXELS}, max=${IMAGE_MAX_PIXELS}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/lr_experiment_v7/rating_weight_sweep "${MODEL_DIR}" "${RESULTS_DIR}"

echo ""
echo "=== CLS VISION TRAINING ==="
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
    --image_min_pixels ${IMAGE_MIN_PIXELS} \
    --image_max_pixels ${IMAGE_MAX_PIXELS} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH} \
    --per_device_eval_batch_size ${PER_DEVICE_BATCH} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --label_smoothing_eps ${LABEL_SMOOTHING_EPS} \
    ${RATING_FLAGS} \
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
    --freeze_vision_tower true \
    --freeze_multi_modal_projector true \
    --gradient_checkpointing true \
    --overwrite_cache true \
    --ddp_timeout 180000000 \
    --media_dir /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/images \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4

echo ""
echo "=============================================="
echo "COMPLETED: vision_${SHORT_NAME}"
echo "=============================================="
