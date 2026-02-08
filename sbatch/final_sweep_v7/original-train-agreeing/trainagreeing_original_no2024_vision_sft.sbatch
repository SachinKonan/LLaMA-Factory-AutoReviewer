#!/bin/bash
#SBATCH --job-name=orig_no2024_vision_sft
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=350G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --output=logs/lr_experiment_v7/%j.out
#SBATCH --error=logs/lr_experiment_v7/%j.err

# Vision SFT (generation) training on original trainagreeing (2024 filtered)

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
TEMPLATE="qwen2_vl"
CUTOFF_LEN=24480
IMAGE_MIN_PIXELS=784
IMAGE_MAX_PIXELS=1003520
LEARNING_RATE="2e-6"
WEIGHT_DECAY=0.01
NUM_EPOCHS=6
PER_DEVICE_BATCH=2
GRAD_ACCUM=4
NUM_GPUS=2
DATASET="iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_original_vision_binary_noreviews_v7"
SHORT_NAME="trainagreeing_original_no2024_vision_sft"
MODEL_DIR="saves/lr_experiment_v7/${SHORT_NAME}"
RESULTS_DIR="results/lr_experiment_v7/${SHORT_NAME}"
MASTER_PORT=29805

echo "=============================================="
echo "Job: $SLURM_JOB_ID | Dataset: ${DATASET}"
echo "Model: ${MODEL} | Stage: SFT (Generation)"
echo "=============================================="

mkdir -p logs/lr_experiment_v7 "${MODEL_DIR}" "${RESULTS_DIR}"

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
    --main_process_port ${MASTER_PORT} \
    src/train.py \
    --stage sft \
    --model_name_or_path ${MODEL} \
    --dataset ${DATASET}_train \
    --template ${TEMPLATE} \
    --output_dir ${MODEL_DIR} \
    --cutoff_len ${CUTOFF_LEN} \
    --image_min_pixels ${IMAGE_MIN_PIXELS} \
    --image_max_pixels ${IMAGE_MAX_PIXELS} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_train_epochs ${NUM_EPOCHS} \
    --lr_scheduler_type cosine \
    --warmup_steps 10 \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --eval_strategy no \
    --bf16 true \
    --report_to none \
    --do_train \
    --finetuning_type full \
    --save_only_model true \
    --freeze_vision_tower true \
    --freeze_multi_modal_projector true \
    --gradient_checkpointing true \
    --enable_liger_kernel \
    --overwrite_cache true \
    --ddp_timeout 180000000 \
    --media_dir /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/data/images \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4

echo "COMPLETED: ${SHORT_NAME}"
