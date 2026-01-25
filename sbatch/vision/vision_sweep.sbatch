#!/bin/bash
#SBATCH --job-name=vision_sweep
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --array=0-1
#SBATCH --output=logs/vision_sweep/%A_%a.out
#SBATCH --error=logs/vision_sweep/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 2 vision dataset variants
DATASETS=(
    "iclr_clean+images_binary_no_reviews_v3"
    "iclr_vision_binary_no_reviews_v3"
)

SHORT_NAMES=(
    "clean_images"
    "vision"
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
SHORT_NAME="${SHORT_NAMES[$SLURM_ARRAY_TASK_ID]}"

CONFIG_FILE="configs/vision_sweep.yaml"
MODEL_DIR="saves/qwen2.5-vl-7b/full/vision_sweep/${SHORT_NAME}"
RESULTS_DIR="results/vision_sweep/${SHORT_NAME}"
MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/vision_sweep "${MODEL_DIR}" "${RESULTS_DIR}"

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
    output_dir="${MODEL_DIR}"

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
    --cutoff_len 20480 \
    --max_new_tokens 10240 \
    --image_min_pixels 2352 \
    --image_max_pixels 2104256 \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "Model: ${CKPT_DIR}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
