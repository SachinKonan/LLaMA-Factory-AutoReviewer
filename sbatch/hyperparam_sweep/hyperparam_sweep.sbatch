#!/bin/bash
#SBATCH --job-name=hp_sweep
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --array=0-14
#SBATCH --output=logs/hyperparam_sweep/%A_%a.out
#SBATCH --error=logs/hyperparam_sweep/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# All 15 hyperparameter combinations
# Format: "BATCH_SIZE LR SCHEDULER WARMUP"
# Skipped: (16, 2e-6, constant_with_warmup, 0.03) - already done
CONFIGS=(
    "16 2.0e-6 cosine 0"
    "16 2.0e-6 cosine 0.03"
    "16 2.0e-6 constant_with_warmup 0"
    "16 2.0e-5 cosine 0"
    "16 2.0e-5 cosine 0.03"
    "16 2.0e-5 constant_with_warmup 0"
    "16 2.0e-5 constant_with_warmup 0.03"
    "32 2.0e-6 cosine 0"
    "32 2.0e-6 cosine 0.03"
    "32 2.0e-6 constant_with_warmup 0"
    "32 2.0e-6 constant_with_warmup 0.03"
    "32 2.0e-5 cosine 0"
    "32 2.0e-5 cosine 0.03"
    "32 2.0e-5 constant_with_warmup 0"
    "32 2.0e-5 constant_with_warmup 0.03"
)

# Parse hyperparameters from array index
read BATCH_SIZE LR SCHEDULER WARMUP <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Select config file based on batch size
if [ "$BATCH_SIZE" == "16" ]; then
    CONFIG_FILE="configs/hyperparam_sweep_batch16.yaml"
else
    CONFIG_FILE="configs/hyperparam_sweep_batch32.yaml"
fi

# Short scheduler name for directory
if [ "$SCHEDULER" == "constant_with_warmup" ]; then
    SCHED_SHORT="const"
else
    SCHED_SHORT="cos"
fi

# Directory names
CONFIG_NAME="bs${BATCH_SIZE}_lr${LR}_${SCHED_SHORT}_w${WARMUP}"
MODEL_DIR="saves/qwen2.5-7b/full/hyperparam_sweep/${CONFIG_NAME}"
RESULTS_DIR="results/hyperparam_sweep/${CONFIG_NAME}"
MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Config: ${CONFIG_NAME}"
echo "Batch Size: ${BATCH_SIZE}, LR: ${LR}"
echo "Scheduler: ${SCHEDULER}, Warmup: ${WARMUP}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/hyperparam_sweep "${MODEL_DIR}" "${RESULTS_DIR}"

# ======================
# PHASE 1: TRAINING
# ======================
echo ""
echo "=== PHASE 1: TRAINING ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py "${CONFIG_FILE}" \
    dataset="iclr_clean_binary_no_reviews_v2_train" \
    eval_dataset="iclr_clean_binary_no_reviews_v2_test" \
    output_dir="${MODEL_DIR}" \
    learning_rate="${LR}" \
    lr_scheduler_type="${SCHEDULER}" \
    warmup_ratio="${WARMUP}"

echo ""
echo "=== Training complete! ==="
echo ""

# ======================
# PHASE 2: INFERENCE
# ======================
echo ""
echo "=== PHASE 2: INFERENCE ==="
echo ""

# Find the checkpoint directory (only one saved at end)
CKPT_DIR=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | head -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "Using checkpoint: ${CKPT_DIR}"

python scripts/vllm_infer.py \
    --model_name_or_path "${CKPT_DIR}" \
    --dataset "iclr_clean_binary_no_reviews_v2_test" \
    --template qwen \
    --cutoff_len 20480 \
    --max_new_tokens 10240 \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${CONFIG_NAME}"
echo "Model: ${CKPT_DIR}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
