#!/bin/bash
#SBATCH --job-name=qwen_inf_text
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/qwen_inf/sft_ds3_%j.out
#SBATCH --error=logs/qwen_inf/sft_ds3_%j.err

set -e  # Exit on error

cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer
source .venv_vllm_inf/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"


# 1. Define the model path (the output_dir from your training YAML)
TRAINED_MODEL_PATH="saves/qwen2.5-3b/full/sft_ds3"

# 2. Define your test dataset (must be registered in dataset_info.json)
TEST_DATASET="iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_validation"

# 3. Define output path
OUTPUT_FILE="results/predictions/qwen2_5_3b_ds3_predictions.jsonl"
mkdir -p results/predictions

# Run inference using 1 GPU (using CUDA_VISIBLE_DEVICES=0 to avoid conflicts)
CUDA_VISIBLE_DEVICES=0 python scripts/vllm_infer.py \
    --model_name_or_path "$TRAINED_MODEL_PATH" \
    --dataset "$TEST_DATASET" \
    --template qwen \
    --cutoff_len 20480 \
    --max_new_tokens 1024 \
    --save_name "$OUTPUT_FILE"

echo "Inference complete! Results saved to $OUTPUT_FILE"