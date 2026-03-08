#!/bin/bash
#SBATCH --job-name=qwen_sft
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/qwen_sft/sft_v2_%j.out
#SBATCH --error=logs/qwen_sft/sft_v2_%j.err

# Full fine-tuning of Qwen2.5-3B-Instruct using LLaMA-Factory with FSDP2
#
# Usage:
#   sbatch sbatch/qwen2_5_3b_finetune.sbatch

set -e  # Exit on error

cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"

# Create logs directory if needed
mkdir -p logs

echo ""
echo "Starting Qwen2.5-3B-Instruct full fine-tuning with FSDP2..."

# Run FSDP2 training with accelerate
accelerate launch \
    --config_file configs/fsdp2_8gpu_config.yaml \
    src/train.py /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/configs/qwen2_5_3b_full_sft_fsdp2.yaml

echo ""
echo "Training complete. Starting Inference..."

# --- NEW INFERENCE STAGE ---

# 1. Define the model path (the output_dir from your training YAML)
TRAINED_MODEL_PATH="saves/qwen2.5-3b/full/sft"

# 2. Define your test dataset (must be registered in dataset_info.json)
TEST_DATASET="iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3_test"

# 3. Define output path
OUTPUT_FILE="results/predictions/qwen2_5_3b_predictions.jsonl"
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