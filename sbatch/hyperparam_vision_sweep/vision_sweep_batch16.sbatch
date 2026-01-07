#!/bin/bash
#SBATCH --job-name=hp_vis_b16
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --array=0-3
#SBATCH --output=logs/hyperparam_vision_sweep/%A_%a.out
#SBATCH --error=logs/hyperparam_vision_sweep/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 4 combinations: 2 freeze_projector settings x 2 datasets
# Format: "FREEZE_PROJ DATASET SHORT_NAME"
CONFIGS=(
    "true iclr_clean+images_binary_no_reviews_v3 clean_images"
    "true iclr_vision_binary_no_reviews_v3 vision"
    "false iclr_clean+images_binary_no_reviews_v3 clean_images"
    "false iclr_vision_binary_no_reviews_v3 vision"
)

# Parse hyperparameters from array index
read FREEZE_PROJ DATASET SHORT_NAME <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Select config file based on dataset
if [ "$SHORT_NAME" == "clean_images" ]; then
    CONFIG_FILE="configs/hyperparam_clean_and_images_sweep_batch16.yaml"
    INFER_CUTOFF=30000
    INFER_MAX_PIXELS=589824
else
    CONFIG_FILE="configs/hyperparam_vision_sweep_batch16.yaml"
    INFER_CUTOFF=20480
    INFER_MAX_PIXELS=1003520
fi

# Short name for freeze setting
if [ "$FREEZE_PROJ" == "true" ]; then
    FREEZE_SHORT="frozen"
else
    FREEZE_SHORT="unfrozen"
fi

# Directory names - unique port per job (base 29500 for batch16)
CONFIG_NAME="bs16_proj_${FREEZE_SHORT}_${SHORT_NAME}"
MODEL_DIR="saves/qwen2.5-vl-7b/full/hyperparam_vision_sweep/${CONFIG_NAME}"
RESULTS_DIR="results/hyperparam_vision_sweep/${CONFIG_NAME}"
MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Config: ${CONFIG_NAME}"
echo "Batch Size: 16, Freeze Projector: ${FREEZE_PROJ}"
echo "Dataset: ${DATASET}"
echo "Config File: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "Master Port: ${MASTER_PORT}"
echo "=============================================="

mkdir -p logs/hyperparam_vision_sweep "${MODEL_DIR}" "${RESULTS_DIR}"

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
    dataset="${DATASET}_train" \
    eval_dataset="${DATASET}_test" \
    output_dir="${MODEL_DIR}" \
    freeze_multi_modal_projector="${FREEZE_PROJ}"

echo ""
echo "=== Training complete! ==="
echo ""

# ======================
# PHASE 2: INFERENCE
# ======================
echo ""
echo "=== PHASE 2: INFERENCE ==="
echo ""

CKPT_DIR=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | head -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "Using checkpoint: ${CKPT_DIR}"

python scripts/vllm_infer.py \
    --model_name_or_path "${CKPT_DIR}" \
    --dataset "${DATASET}_test" \
    --template qwen2_vl \
    --cutoff_len "${INFER_CUTOFF}" \
    --max_new_tokens 10240 \
    --image_min_pixels 784 \
    --image_max_pixels "${INFER_MAX_PIXELS}" \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${CONFIG_NAME}"
echo "Model: ${CKPT_DIR}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
