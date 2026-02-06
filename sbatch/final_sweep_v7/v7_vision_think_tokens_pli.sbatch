#!/bin/bash
#SBATCH --job-name=v7_think_tokens
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --array=0-5
#SBATCH --output=logs/final_sweep_v7_pli/%A_%a.out
#SBATCH --error=logs/final_sweep_v7_pli/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 6 think token configurations: (1, 10, 100) x (input, label)
THINK_NUM_TOKENS=(10 10 100 100 1000 1000)
THINK_PLACEMENTS=("input" "label" "input" "label" "input" "label")

NUM_TOKENS=${THINK_NUM_TOKENS[$SLURM_ARRAY_TASK_ID]}
PLACEMENT=${THINK_PLACEMENTS[$SLURM_ARRAY_TASK_ID]}

DATASET="iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7"
SHORT_NAME="trainagreeing_vision_think${NUM_TOKENS}_${PLACEMENT}"

CONFIG_FILE="configs/final_sweep_v7_vision.yaml"
TEMPLATE="qwen2_vl"
CUTOFF_LEN=26480
IMAGE_MIN_PIXELS=784
IMAGE_MAX_PIXELS=1003520

# 4 H100 GPUs, per_device=2, grad_accum=2 → batch = 2*4*2 = 16
PER_DEVICE_BATCH=1
GRAD_ACCUM=4
NUM_GPUS=4

MODEL_DIR="saves/final_sweep_v7_pli/${SHORT_NAME}"
RESULTS_DIR="results/final_sweep_v7_pli/${SHORT_NAME}"
MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "Think Tokens: ${NUM_TOKENS}"
echo "Think Placement: ${PLACEMENT}"
echo "GPUs: ${NUM_GPUS} (H200), per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Config: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "=============================================="

mkdir -p logs/final_sweep_v7_pli "${MODEL_DIR}" "${RESULTS_DIR}"

# ======================
# Start auto-inference watcher in background
# ======================
python scripts/auto_infer_watcher.py \
    --save_dir "${MODEL_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --dataset "${DATASET}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 1280 \
    --image_min_pixels "${IMAGE_MIN_PIXELS}" \
    --image_max_pixels "${IMAGE_MAX_PIXELS}" \
    --compute_train_accuracy \
    --poll_interval 60 &
AUTO_INFER_PID=$!
echo "Started auto-inference watcher (PID: ${AUTO_INFER_PID})"

# ======================
# PHASE 1: TRAINING
# ======================
echo ""
echo "=== PHASE 1: TRAINING ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_4gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py "${CONFIG_FILE}" \
    dataset="${DATASET}_train" \
    eval_dataset="${DATASET}_validation" \
    output_dir="${MODEL_DIR}" \
    per_device_train_batch_size="${PER_DEVICE_BATCH}" \
    gradient_accumulation_steps="${GRAD_ACCUM}" \
    cutoff_len="${CUTOFF_LEN}" \
    think_num_tokens="${NUM_TOKENS}" \
    think_token_placement="${PLACEMENT}"

echo ""
echo "=== Training complete! ==="
echo ""

# ======================
# PHASE 2: INFERENCE
# ======================
echo ""
echo "=== PHASE 2: INFERENCE ==="
echo ""

CKPT_DIR=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "Using checkpoint: ${CKPT_DIR}"

python scripts/vllm_infer.py \
    --model_name_or_path "${CKPT_DIR}" \
    --dataset "${DATASET}_test" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 1280 \
    --image_min_pixels "${IMAGE_MIN_PIXELS}" \
    --image_max_pixels "${IMAGE_MAX_PIXELS}" \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

# Kill auto-inference watcher if still running
kill $AUTO_INFER_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
