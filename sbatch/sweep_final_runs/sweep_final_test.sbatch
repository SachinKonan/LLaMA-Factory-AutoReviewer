#!/bin/bash
#SBATCH --job-name=sweep_final
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --array=0-3
#SBATCH --output=logs/sweep_final_runs/%A_%a.out
#SBATCH --error=logs/sweep_final_runs/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 4 configurations: clean, clean_images, vision_768, vision_1280
# Format: "CONFIG_FILE DATASET SHORT_NAME TEMPLATE CUTOFF_LEN IMAGE_MAX_PIXELS"
CONFIGS=(
    "configs/sweep_final_clean.yaml iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3 clean qwen 20480 0"
    "configs/sweep_final_clean_images.yaml iclr_2020_2025_80_20_split5_balanced_deepreview_clean+images_binary_no_reviews_titleabs_corrected_v3 clean_images qwen2_vl 32000 156800"
    "configs/sweep_final_vision_768.yaml iclr_2020_2025_80_20_split5_balanced_deepreview_vision_binary_no_reviews_titleabs_corrected_v3 vision_768 qwen2_vl 20480 589824"
    "configs/sweep_final_vision_1280.yaml iclr_2020_2025_80_20_split5_balanced_deepreview_vision_binary_no_reviews_titleabs_corrected_v3 vision_1280 qwen2_vl 20480 1003520"
)

read CONFIG_FILE DATASET SHORT_NAME TEMPLATE CUTOFF_LEN IMAGE_MAX_PIXELS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

MODEL_DIR="saves/sweep_final_runs_test/${SHORT_NAME}"
RESULTS_DIR="results/sweep_final_runs_test/${SHORT_NAME}"
MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Config: ${SHORT_NAME}"
echo "Config File: ${CONFIG_FILE}"
echo "Dataset: ${DATASET}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/sweep_final_runs_test "${MODEL_DIR}" "${RESULTS_DIR}"

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
    eval_dataset="${DATASET}_test" \
    output_dir="${MODEL_DIR}"

echo ""
echo "=== Training complete! ==="
echo ""

# ======================
# PHASE 2: INFERENCE (on final checkpoint)
# ======================
echo ""
echo "=== PHASE 2: INFERENCE ==="
echo ""

# Find the latest checkpoint (epoch 4)
CKPT_DIR=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "Using checkpoint: ${CKPT_DIR}"

# Build inference command
if [ "$IMAGE_MAX_PIXELS" -gt 0 ]; then
    python scripts/vllm_infer.py \
        --model_name_or_path "${CKPT_DIR}" \
        --dataset "${DATASET}_test" \
        --template "${TEMPLATE}" \
        --cutoff_len "${CUTOFF_LEN}" \
        --max_new_tokens 1280 \
        --image_min_pixels 784 \
        --image_max_pixels "${IMAGE_MAX_PIXELS}" \
        --save_name "${RESULTS_DIR}/finetuned.jsonl"
else
    python scripts/vllm_infer.py \
        --model_name_or_path "${CKPT_DIR}" \
        --dataset "${DATASET}_test" \
        --template "${TEMPLATE}" \
        --cutoff_len "${CUTOFF_LEN}" \
        --max_new_tokens 1280 \
        --save_name "${RESULTS_DIR}/finetuned.jsonl"
fi

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "Model: ${CKPT_DIR}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
