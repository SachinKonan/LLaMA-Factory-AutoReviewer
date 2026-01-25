#!/bin/bash
#SBATCH --job-name=data_sweep_v2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --array=0-11
#SBATCH --output=logs/data_sweep_v2/%A_%a.out
#SBATCH --error=logs/data_sweep_v2/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 12 configurations: 6 dataset variants x 2 modalities (clean/vision)
# Format: "DATASET_BASE SHORT_NAME MODALITY"
CONFIGS=(
    # ICLR 2017-2025 balanced 0,1 needs rerun
    "iclr_2017_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6 iclr17_balanced clean"
    "iclr_2017_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6 iclr17_balanced vision"
    # ICLR 2020-2025 balanced 2,3
    "iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6 iclr20_balanced clean"
    "iclr_2020_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6 iclr20_balanced vision"
    # ICLR 2020-2025 trainagreeing 4,5
    "iclr_2020_2025_85_5_10_split6_balanced_trainagreeing_clean_binary_noreviews_v6 iclr20_trainagreeing clean"
    "iclr_2020_2025_85_5_10_split6_balanced_trainagreeing_vision_binary_noreviews_v6 iclr20_trainagreeing vision"
    # ICLR 2020-2025 original 6,7 needs rerun
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6 iclr20_original clean"
    "iclr_2020_2025_85_5_10_split6_original_vision_binary_noreviews_v6 iclr20_original vision"
    # ICLR+NeurIPS 2020-2025 balanced 8,9
    "iclr_nips_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6 iclr_nips_balanced clean"
    "iclr_nips_2020_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6 iclr_nips_balanced vision"
    # ICLR+NeurIPS 2020-2025 nips_accepts 10,11 needs rerurn
    "iclr_nips_2020_2025_85_5_10_split6_balanced_nips_accepts_clean_binary_noreviews_v6 iclr_nips_accepts clean"
    "iclr_nips_2020_2025_85_5_10_split6_balanced_nips_accepts_vision_binary_noreviews_v6 iclr_nips_accepts vision"
)

read DATASET SHORT_NAME MODALITY <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

# Set config and template based on modality
if [ "$MODALITY" = "clean" ]; then
    CONFIG_FILE="configs/data_sweep_v2_clean.yaml"
    TEMPLATE="qwen"
    IMAGE_MAX_PIXELS=0
else
    CONFIG_FILE="configs/data_sweep_v2_vision.yaml"
    TEMPLATE="qwen2_vl"
    IMAGE_MAX_PIXELS=1003520
fi

MODEL_DIR="saves/data_sweep_v2/${SHORT_NAME}_${MODALITY}"
RESULTS_DIR="results/data_sweep_v2/${SHORT_NAME}_${MODALITY}"
MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}_${MODALITY}"
echo "Config: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/data_sweep_v2 "${MODEL_DIR}" "${RESULTS_DIR}"

# ======================
# PHASE 1: TRAINING (with eval on validation)
# ======================
echo ""
echo "=== PHASE 1: TRAINING (eval on validation every epoch) ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_2gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py "${CONFIG_FILE}" \
    dataset="${DATASET}_train" \
    eval_dataset="${DATASET}_validation" \
    output_dir="${MODEL_DIR}"

echo ""
echo "=== Training complete! ==="
echo ""

# ======================
# PHASE 2: INFERENCE (on test set)
# ======================
echo ""
echo "=== PHASE 2: INFERENCE (on test set) ==="
echo ""

# Find the latest checkpoint
CKPT_DIR=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found in ${MODEL_DIR}"
    exit 1
fi

echo "Using checkpoint: ${CKPT_DIR}"

# Run inference on TEST set
if [ "$IMAGE_MAX_PIXELS" -gt 0 ]; then
    python scripts/vllm_infer.py \
        --model_name_or_path "${CKPT_DIR}" \
        --dataset "${DATASET}_test" \
        --template "${TEMPLATE}" \
        --cutoff_len 24480 \
        --max_new_tokens 1280 \
        --image_min_pixels 784 \
        --image_max_pixels "${IMAGE_MAX_PIXELS}" \
        --save_name "${RESULTS_DIR}/finetuned.jsonl"
else
    python scripts/vllm_infer.py \
        --model_name_or_path "${CKPT_DIR}" \
        --dataset "${DATASET}_test" \
        --template "${TEMPLATE}" \
        --cutoff_len 24480 \
        --max_new_tokens 1280 \
        --save_name "${RESULTS_DIR}/finetuned.jsonl"
fi

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}_${MODALITY}"
echo "Model: ${CKPT_DIR}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
