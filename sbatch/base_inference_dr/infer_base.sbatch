#!/bin/bash
#SBATCH --job-name=base_infer_dr
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --array=0-2
#SBATCH --output=logs/base_inference_dr/%A_%a.out
#SBATCH --error=logs/base_inference_dr/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 3 jobs: clean (text), clean+images, vision
# Format: "MODEL DATASET SHORT_NAME TEMPLATE CUTOFF_LEN IMAGE_MAX_PIXELS"
CONFIGS=(
    "Qwen/Qwen2.5-7B-Instruct iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3 clean qwen 20480 0"
    "Qwen/Qwen2.5-VL-7B-Instruct iclr_2020_2025_80_20_split5_balanced_deepreview_clean+images_binary_no_reviews_titleabs_corrected_v3 clean_images qwen2_vl 30000 589824"
    "Qwen/Qwen2.5-VL-7B-Instruct iclr_2020_2025_80_20_split5_balanced_deepreview_vision_binary_no_reviews_titleabs_corrected_v3 vision qwen2_vl 20480 589824"
)

read MODEL DATASET SHORT_NAME TEMPLATE CUTOFF_LEN IMAGE_MAX_PIXELS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

RESULTS_DIR="results/base_inference_dr/${SHORT_NAME}"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}_test"
echo "Template: ${TEMPLATE}"
echo "Cutoff Len: ${CUTOFF_LEN}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/base_inference_dr "${RESULTS_DIR}"

# Build inference command based on whether it's vision model
if [ "$IMAGE_MAX_PIXELS" -gt 0 ]; then
    python scripts/vllm_infer.py \
        --model_name_or_path "${MODEL}" \
        --dataset "${DATASET}_test" \
        --template "${TEMPLATE}" \
        --cutoff_len "${CUTOFF_LEN}" \
        --max_new_tokens 10240 \
        --image_min_pixels 2352 \
        --image_max_pixels "${IMAGE_MAX_PIXELS}" \
        --save_name "${RESULTS_DIR}/base.jsonl"
else
    python scripts/vllm_infer.py \
        --model_name_or_path "${MODEL}" \
        --dataset "${DATASET}_test" \
        --template "${TEMPLATE}" \
        --cutoff_len "${CUTOFF_LEN}" \
        --max_new_tokens 10240 \
        --save_name "${RESULTS_DIR}/base.jsonl"
fi

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "Results: ${RESULTS_DIR}/base.jsonl"
echo "=============================================="
