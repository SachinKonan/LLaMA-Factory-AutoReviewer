#!/bin/bash
#SBATCH --job-name=dc_infer_v
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=350G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/final_sweep_v7_datasweepv3/data_cleaning/infer/%j.out
#SBATCH --error=logs/final_sweep_v7_datasweepv3/data_cleaning/infer/%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Vision inference on training set (with logprobs)
CHECKPOINT="saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/checkpoint-2648"
DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train"
TEMPLATE="qwen2_vl"
OUTPUT_FILE="results/data_cleaning/vision_train_predictions.jsonl"

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET}"
echo "Template: ${TEMPLATE}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

mkdir -p logs/final_sweep_v7_datasweepv3/data_cleaning/infer results/data_cleaning

python scripts/vllm_infer.py \
    --model_name_or_path "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    --template "${TEMPLATE}" \
    --cutoff_len 24480 \
    --max_new_tokens 128 \
    --image_min_pixels 784 \
    --image_max_pixels 1003520 \
    --save_logprobs \
    --save_name "${OUTPUT_FILE}"

echo ""
echo "=============================================="
echo "COMPLETED: Vision inference"
echo "Output saved to: ${OUTPUT_FILE}"
echo "=============================================="
