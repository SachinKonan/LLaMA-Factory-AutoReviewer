#!/bin/bash
#SBATCH --job-name=hp_infer_b16
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --array=0-6
#SBATCH --output=logs/hyperparam_sweep/%A_%a.out
#SBATCH --error=logs/hyperparam_sweep/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 7 model directories to run inference on
MODEL_DIRS=(
    "saves/qwen2.5-7b/full/hyperparam_sweep/bs16_lr2.0e-5_const_w0"
    "saves/qwen2.5-7b/full/hyperparam_sweep/bs16_lr2.0e-5_const_w0.03"
    "saves/qwen2.5-7b/full/hyperparam_sweep/bs16_lr2.0e-5_cos_w0"
    "saves/qwen2.5-7b/full/hyperparam_sweep/bs16_lr2.0e-5_cos_w0.03"
    "saves/qwen2.5-7b/full/hyperparam_sweep/bs16_lr2.0e-6_const_w0"
    "saves/qwen2.5-7b/full/hyperparam_sweep/bs16_lr2.0e-6_cos_w0"
    "saves/qwen2.5-7b/full/hyperparam_sweep/bs16_lr2.0e-6_cos_w0.03"
)

MODEL_DIR="${MODEL_DIRS[$SLURM_ARRAY_TASK_ID]}"
CONFIG_NAME=$(basename "${MODEL_DIR}")
RESULTS_DIR="results/hyperparam_sweep/${CONFIG_NAME}"

# Find the checkpoint directory
CKPT_DIR=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | head -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Config: ${CONFIG_NAME}"
echo "Checkpoint: ${CKPT_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/hyperparam_sweep "${RESULTS_DIR}"

echo ""
echo "=== RUNNING INFERENCE ==="
echo ""

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
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
