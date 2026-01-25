#!/bin/bash
#SBATCH --job-name=vis_dsweep
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --array=0-3
#SBATCH --output=logs/hyperparam_vision_data_sweep/%A_%a.out
#SBATCH --error=logs/hyperparam_vision_data_sweep/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 4 combinations: 2 dataset types (clean+images, vision) x 2 data variants (balanced_deepreview, balanced_deepreview_agreeing)
# Format: "DATASET_TYPE DATA_VARIANT SHORT_NAME CUTOFF_LEN"
CONFIGS=(
    "clean+images balanced_deepreview ci_bd 29000"
    "clean+images balanced_deepreview_agreeing ci_bda 30000"
    "vision balanced_deepreview vis_bd 22480"
    "vision balanced_deepreview_agreeing vis_bda 22480"
)

# Parse configuration from array index
read DATASET_TYPE DATA_VARIANT SHORT_NAME CUTOFF_LEN <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Build dataset name based on type and variant
if [ "$DATASET_TYPE" == "clean+images" ]; then
    DATASET="iclr_2020_2025_80_20_split5_${DATA_VARIANT}_clean+images_binary_no_reviews_titleabs_corrected_v3"
else
    DATASET="iclr_2020_2025_80_20_split5_${DATA_VARIANT}_vision_binary_no_reviews_titleabs_corrected_v3"
fi

# Directory names - unique port per job
CONFIG_NAME="${SHORT_NAME}"
MODEL_DIR="saves/qwen2.5-vl-7b/full/hyperparam_vision_data_sweep/${CONFIG_NAME}"
RESULTS_DIR="results/hyperparam_vision_data_sweep/${CONFIG_NAME}"
MASTER_PORT=$((29600 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Config: ${CONFIG_NAME}"
echo "Dataset Type: ${DATASET_TYPE}"
echo "Data Variant: ${DATA_VARIANT}"
echo "Dataset: ${DATASET}"
echo "Cutoff Length: ${CUTOFF_LEN}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "Master Port: ${MASTER_PORT}"
echo "=============================================="

mkdir -p logs/hyperparam_vision_data_sweep "${MODEL_DIR}" "${RESULTS_DIR}"

# ======================
# PHASE 1: TRAINING
# ======================
echo ""
echo "=== PHASE 1: TRAINING ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py configs/hyperparam_vision_data_sweep.yaml \
    dataset="${DATASET}_train" \
    eval_dataset="${DATASET}_test" \
    output_dir="${MODEL_DIR}" \
    cutoff_len="${CUTOFF_LEN}"

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
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 10240 \
    --image_min_pixels 2352 \
    --image_max_pixels 589824 \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${CONFIG_NAME}"
echo "Model: ${CKPT_DIR}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
