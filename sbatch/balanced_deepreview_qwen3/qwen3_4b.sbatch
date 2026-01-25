#!/bin/bash
#SBATCH --job-name=qwen3_4b_bdrev
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --output=logs/balanced_deepreview_qwen3/%j_4b.out
#SBATCH --error=logs/balanced_deepreview_qwen3/%j_4b.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Dataset
DATASET="iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3"

# Model config
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
CONFIG_FILE="configs/qwen3_4b_balanced_deepreview.yaml"
FSDP_CONFIG="configs/fsdp2_1gpu_config.yaml"
SHORT_NAME="qwen3_4b"

# Directory setup
MODEL_DIR="saves/${SHORT_NAME}/full/balanced_deepreview"
RESULTS_DIR="results/balanced_deepreview_qwen3/${SHORT_NAME}"
MASTER_PORT=29701

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Model: ${MODEL_PATH}"
echo "Config: ${CONFIG_FILE}"
echo "FSDP Config: ${FSDP_CONFIG}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "Master Port: ${MASTER_PORT}"
echo "=============================================="

mkdir -p logs/balanced_deepreview_qwen3 "${MODEL_DIR}" "${RESULTS_DIR}"

# ======================
# PHASE 1: TRAINING
# ======================
echo ""
echo "=== PHASE 1: TRAINING ==="
echo ""

accelerate launch \
    --config_file "${FSDP_CONFIG}" \
    --main_process_port "${MASTER_PORT}" \
    src/train.py "${CONFIG_FILE}" \
    model_name_or_path="${MODEL_PATH}" \
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

# Use single GPU for inference
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | head -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "Using checkpoint: ${CKPT_DIR}"

python scripts/vllm_infer.py \
    --model_name_or_path "${CKPT_DIR}" \
    --dataset "${DATASET}_test" \
    --template qwen3_nothink \
    --cutoff_len 20480 \
    --max_new_tokens 10240 \
    --enable_thinking false \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "Model: ${CKPT_DIR}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
