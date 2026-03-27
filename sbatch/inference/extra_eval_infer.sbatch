#!/bin/bash
#SBATCH --job-name=extra_eval_infer
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=350G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --array=0-5
#SBATCH --output=logs/extra_eval_infer/%A_%a.out
#SBATCH --error=logs/extra_eval_infer/%A_%a.err

# Job array for vLLM inference on extra eval datasets (NIPS 2021-2025, ICLR 2017-2019, ICML/COLM 2024-2025)
# using the top-performing 2020-2026 text and vision models.
#
# Array tasks:
#   0: Text  model → nips_2021_2025_original_text_v7_noref_eval500
#   1: Text  model → iclr_2017_2019_original_text_v7_eval200_balanced
#   2: Text  model → icml_colm_2024_2025_clean_binary_noref_eval350
#   3: Vision model → nips_2021_2025_original_vision_v7_eval500
#   4: Vision model → iclr_2017_2019_original_vision_v7_eval200_balanced
#   5: Vision model → icml_colm_2024_2025_vision_binary_eval350

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Model checkpoints
TEXT_CHECKPOINT="saves/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/checkpoint-1322"
VISION_CHECKPOINT="saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/checkpoint-2648"

# Results directories
TEXT_RESULTS_DIR="results/extra_eval/text_bz32_ckpt1322"
VISION_RESULTS_DIR="results/extra_eval/vision_bz16_ckpt2648"

# Inference parameters
CUTOFF_LEN=24480
MAX_NEW_TOKENS=1280
TEMPERATURE=0

# Vision image parameters (matching training config)
IMAGE_MIN_PIXELS=784
IMAGE_MAX_PIXELS=1003520

# Configure per array task
case $SLURM_ARRAY_TASK_ID in
    0)
        MODEL_PATH="${TEXT_CHECKPOINT}"
        DATASET="nips_2021_2025_original_text_v7_noref_eval500"
        TEMPLATE="qwen"
        RESULTS_DIR="${TEXT_RESULTS_DIR}"
        OUTPUT_FILE="${RESULTS_DIR}/nips_2021_2025_eval500.jsonl"
        EXTRA_ARGS=""
        ;;
    1)
        MODEL_PATH="${TEXT_CHECKPOINT}"
        DATASET="iclr_2017_2019_original_text_v7_eval200_balanced"
        TEMPLATE="qwen"
        RESULTS_DIR="${TEXT_RESULTS_DIR}"
        OUTPUT_FILE="${RESULTS_DIR}/iclr_2017_2019_eval200_balanced.jsonl"
        EXTRA_ARGS=""
        ;;
    2)
        MODEL_PATH="${TEXT_CHECKPOINT}"
        DATASET="icml_colm_2024_2025_clean_binary_noref_eval350"
        TEMPLATE="qwen"
        RESULTS_DIR="${TEXT_RESULTS_DIR}"
        OUTPUT_FILE="${RESULTS_DIR}/icml_colm_2024_2025_eval350.jsonl"
        EXTRA_ARGS=""
        ;;
    3)
        MODEL_PATH="${VISION_CHECKPOINT}"
        DATASET="nips_2021_2025_original_vision_v7_eval500"
        TEMPLATE="qwen2_vl"
        RESULTS_DIR="${VISION_RESULTS_DIR}"
        OUTPUT_FILE="${RESULTS_DIR}/nips_2021_2025_eval500.jsonl"
        EXTRA_ARGS="--image_min_pixels ${IMAGE_MIN_PIXELS} --image_max_pixels ${IMAGE_MAX_PIXELS}"
        ;;
    4)
        MODEL_PATH="${VISION_CHECKPOINT}"
        DATASET="iclr_2017_2019_original_vision_v7_eval200_balanced"
        TEMPLATE="qwen2_vl"
        RESULTS_DIR="${VISION_RESULTS_DIR}"
        OUTPUT_FILE="${RESULTS_DIR}/iclr_2017_2019_eval200_balanced.jsonl"
        EXTRA_ARGS="--image_min_pixels ${IMAGE_MIN_PIXELS} --image_max_pixels ${IMAGE_MAX_PIXELS}"
        ;;
    5)
        MODEL_PATH="${VISION_CHECKPOINT}"
        DATASET="icml_colm_2024_2025_vision_binary_eval350"
        TEMPLATE="qwen2_vl"
        RESULTS_DIR="${VISION_RESULTS_DIR}"
        OUTPUT_FILE="${RESULTS_DIR}/icml_colm_2024_2025_eval350.jsonl"
        EXTRA_ARGS="--image_min_pixels ${IMAGE_MIN_PIXELS} --image_max_pixels ${IMAGE_MAX_PIXELS}"
        ;;
    *)
        echo "ERROR: Unknown array task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "=============================================="
echo "Extra Eval Inference"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET}"
echo "Template: ${TEMPLATE}"
echo "Output: ${OUTPUT_FILE}"
echo "Cutoff Len: ${CUTOFF_LEN}"
echo "Max New Tokens: ${MAX_NEW_TOKENS}"
echo "Temperature: ${TEMPERATURE}"
echo "Extra Args: ${EXTRA_ARGS}"
echo "=============================================="

mkdir -p logs/extra_eval_infer "${RESULTS_DIR}"

python scripts/vllm_infer.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    ${EXTRA_ARGS} \
    --save_name "${OUTPUT_FILE}"

echo ""
echo "=============================================="
echo "COMPLETED: Task $SLURM_ARRAY_TASK_ID"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="
