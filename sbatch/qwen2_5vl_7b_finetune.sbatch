#!/bin/bash
#SBATCH --job-name=qwen_vl_sft
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/qwen_vl_sft_%j.out
#SBATCH --error=logs/qwen_vl_sft_%j.err

# Full fine-tuning of Qwen2.5-VL-7B-Instruct using LLaMA-Factory with FSDP2
#
# Usage:
#   sbatch sbatch/qwen2_5vl_7b_finetune.sbatch

set -e  # Exit on error

cd /n/fs/vision-mix/sk7524/LLaMA-Factory
source .venv/bin/activate

# Set HuggingFace cache
export HF_HOME="/n/fs/vision-mix/sk7524/caches/.hf"

echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"

# Create logs directory if needed
mkdir -p logs

echo ""
echo "Starting Qwen2.5-VL-7B-Instruct full fine-tuning with FSDP2..."

# Run FSDP2 training with accelerate
accelerate launch \
    --config_file configs/fsdp2_4gpu_config.yaml \
    src/train.py configs/qwen2_5vl_7b_full_sft_fsdp2.yaml

echo ""
echo "Training complete"
