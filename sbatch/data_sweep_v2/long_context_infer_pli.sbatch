#!/bin/bash
#SBATCH --job-name=lc_infer
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --array=0-1
#SBATCH --output=logs/data_sweep_long_context_pli/%A_%a_infer.out
#SBATCH --error=logs/data_sweep_long_context_pli/%A_%a_infer.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 2 configurations: clean_vision epoch1 and vision_clean full
# Format: "MODEL_PATH SHORT_NAME DATASET"
CONFIGS=(
    "saves/data_sweep_long_context/balanced_clean_vision_epoch1/checkpoint-1067 balanced_clean_vision_epoch1 iclr_2020_2025_85_5_10_split6_balanced_clean_vision_binary_noreviews_v6"
    "saves/data_sweep_long_context/balanced_vision_clean balanced_vision_clean iclr_2020_2025_85_5_10_split6_balanced_vision_clean_binary_noreviews_v6"
)

read MODEL_PATH SHORT_NAME DATASET <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

TEMPLATE="qwen2_vl"
IMAGE_MAX_PIXELS=1003520
CUTOFF_LEN=50680
RESULTS_DIR="results/data_sweep_long_context_pli/${SHORT_NAME}"

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Short Name: ${SHORT_NAME}"
echo "Model Path: ${MODEL_PATH}"
echo "Dataset: ${DATASET}"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/data_sweep_long_context_pli "${RESULTS_DIR}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found: ${MODEL_PATH}"
    exit 1
fi

python scripts/vllm_infer.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${DATASET}_test" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 1280 \
    --image_min_pixels 784 \
    --image_max_pixels "${IMAGE_MAX_PIXELS}" \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
