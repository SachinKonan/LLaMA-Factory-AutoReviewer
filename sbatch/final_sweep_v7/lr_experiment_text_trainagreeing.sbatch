#!/bin/bash
#SBATCH --job-name=lr_exp_text
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --array=0-2
#SBATCH --output=logs/lr_experiment_v7/%A_%a.out
#SBATCH --error=logs/lr_experiment_v7/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# LR Configurations
# Index 0: LR=2e-5 (both backbone and head)
# Index 1: LR=2e-6 (both backbone and head)
# Index 2: LR=2e-5 (head), 2e-6 (backbone) - separate LRs
LR_CONFIGS=(
    "2e-5 none lr_2e5"              # Index 0: uniform 2e-5
    "2e-6 none lr_2e6"              # Index 1: uniform 2e-6
    "2e-5 2e-6 lr_2e5_backbone_2e6" # Index 2: separate LRs
)

read HEAD_LR BACKBONE_LR SHORT_NAME <<< "${LR_CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Model and dataset configuration
MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7"
TEMPLATE="qwen"
CUTOFF_LEN=24480

# Batch configuration: 2 GPUs, per_device=4, grad_accum=2 → batch = 4*2*2 = 16
PER_DEVICE_BATCH=4
GRAD_ACCUM=2
NUM_GPUS=2

# Output directories
MODEL_DIR="saves/lr_experiment_v7/text_trainagreeing_${SHORT_NAME}"
RESULTS_DIR="results/lr_experiment_v7/text_trainagreeing_${SHORT_NAME}"
MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "Learning Rates: head=${HEAD_LR}, backbone=${BACKBONE_LR}"
echo "GPUs: ${NUM_GPUS} (H200), per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/lr_experiment_v7 "${MODEL_DIR}" "${RESULTS_DIR}"

echo ""
echo "=== CLS TEXT TRAINING + PREDICT ==="
echo ""

# Build the command with conditional cls_backbone_learning_rate flag
CMD="accelerate launch \
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
    --learning_rate ${HEAD_LR}"

# Add backbone learning rate only if not "none"
if [ "$BACKBONE_LR" != "none" ]; then
    CMD="$CMD --cls_backbone_learning_rate ${BACKBONE_LR}"
fi

# Continue with remaining arguments
CMD="$CMD \
    --num_train_epochs 6.0 \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --eval_strategy epoch \
    --bf16 true \
    --do_train \
    --do_predict \
    --finetuning_type full \
    --gradient_checkpointing true \
    --overwrite_cache true"

# Execute the command
eval $CMD

echo ""
echo "=============================================="
echo "COMPLETED: text_trainagreeing_${SHORT_NAME}"
echo "=============================================="
