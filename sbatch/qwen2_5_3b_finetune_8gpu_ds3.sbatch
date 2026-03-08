#!/bin/bash
#SBATCH --job-name=qwen_sft_ds3
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/qwen_sft/sft_ds3_%j.out
#SBATCH --error=logs/qwen_sft/sft_ds3_%j.err

# Full fine-tuning of Qwen2.5-3B-Instruct using LLaMA-Factory with DeepSpeed ZeRO-3
#
# Usage:
#   sbatch sbatch/qwen2_5_3b_finetune_8gpu_ds3.sbatch

set -e  # Exit on error

cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"

# Create logs directory if needed
mkdir -p logs/qwen_sft_v2

# Set memory allocator for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "Starting Qwen2.5-3B-Instruct full fine-tuning with DeepSpeed ZeRO-3..."

# Run DeepSpeed ZeRO-3 training with torchrun
torchrun --nproc_per_node=8 \
    src/train.py configs/qwen2_5_3b_full_sft_ds3.yaml

echo ""
echo "Training complete. Starting Inference..."

# --- INFERENCE STAGE ---

# 1. Define the model path (the output_dir from your training YAML)
TRAINED_MODEL_PATH="saves/qwen2.5-3b_v2/full/sft_ds3"

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
