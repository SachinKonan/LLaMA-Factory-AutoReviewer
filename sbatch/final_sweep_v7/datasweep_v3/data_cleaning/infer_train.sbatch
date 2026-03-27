#!/bin/bash
#SBATCH --job-name=dc_infer
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/final_sweep_v7_datasweepv3/data_cleaning/infer/%j.out
#SBATCH --error=logs/final_sweep_v7_datasweepv3/data_cleaning/infer/%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Text inference on training set (with logprobs)
CHECKPOINT="saves/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/checkpoint-1322"
DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train"
TEMPLATE="qwen"
OUTPUT_FILE="results/data_cleaning/text_train_predictions.jsonl"

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
    --save_logprobs \
    --save_name "${OUTPUT_FILE}"

echo ""
echo "=============================================="
echo "COMPLETED: Text inference"
echo "Output saved to: ${OUTPUT_FILE}"
echo "=============================================="
