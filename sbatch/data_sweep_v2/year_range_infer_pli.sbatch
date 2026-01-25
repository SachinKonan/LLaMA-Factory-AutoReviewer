#!/bin/bash
#SBATCH --job-name=yr_infer
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --array=0-5
#SBATCH --output=logs/data_sweep_year_range/%A_%a_infer.out
#SBATCH --error=logs/data_sweep_year_range/%A_%a_infer.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 6 configurations: dataset short_name, checkpoint step, template, modality
# Format: "SHORT_NAME CKPT_STEP TEMPLATE DATASET_BASE"
CONFIGS=(
    "iclr22_clean 2718 qwen2.5 iclr_2022_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6"
    "iclr22_clean_images 2640 qwen2_vl iclr_2022_2025_85_5_10_split6_balanced_clean_images_binary_noreviews_v6"
    "iclr22_vision 2742 qwen2_vl iclr_2022_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6"
    "iclr24_clean 1250 qwen2.5 iclr_2024_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6"
    "iclr24_clean_images 1824 qwen2_vl iclr_2024_2025_85_5_10_split6_balanced_clean_images_binary_noreviews_v6"
    "iclr24_vision 1896 qwen2_vl iclr_2024_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6"
)

read SHORT_NAME CKPT_STEP TEMPLATE DATASET <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

MODEL_DIR="saves/data_sweep_year_range/${SHORT_NAME}"
RESULTS_DIR="results/data_sweep_year_range/${SHORT_NAME}"
CKPT_DIR="${MODEL_DIR}/checkpoint-${CKPT_STEP}"

# Set cutoff_len and image params based on template
if [ "$TEMPLATE" == "qwen2.5" ]; then
    CUTOFF_LEN=24480
    IMAGE_ARGS=""
else
    CUTOFF_LEN=24480
    IMAGE_ARGS="--image_min_pixels 784 --image_max_pixels 1003520"
fi

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Short Name: ${SHORT_NAME}"
echo "Dataset: ${DATASET}"
echo "Checkpoint: ${CKPT_DIR}"
echo "Template: ${TEMPLATE}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/data_sweep_year_range "${RESULTS_DIR}"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT_DIR}"
    exit 1
fi

python scripts/vllm_infer.py \
    --model_name_or_path "${CKPT_DIR}" \
    --dataset "${DATASET}_test" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 1280 \
    ${IMAGE_ARGS} \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
