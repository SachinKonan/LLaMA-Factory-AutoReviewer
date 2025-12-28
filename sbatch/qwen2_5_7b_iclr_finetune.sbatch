#!/bin/bash
#SBATCH --job-name=qwen_iclr
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/qwen_iclr_%j.out
#SBATCH --error=logs/qwen_iclr_%j.err

# Full fine-tuning of Qwen2.5-7B-Instruct on ICLR data using FSDP2
#
# Usage:
#   sbatch sbatch/qwen2_5_7b_iclr_finetune.sbatch

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
echo "Starting Qwen2.5-7B-Instruct ICLR fine-tuning with FSDP2..."

# Run FSDP2 training with accelerate
accelerate launch \
    --config_file configs/fsdp2_4gpu_config.yaml \
    src/train.py configs/qwen2_5_7b_iclr_sft_fsdp2.yaml

echo ""
echo "Training complete"
