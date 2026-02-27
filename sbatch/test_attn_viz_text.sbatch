#!/bin/bash
#SBATCH --job-name=attn_viz_text_test
#SBATCH --time=01:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --output=logs/attn_viz_text_test_%j.out
#SBATCH --error=logs/attn_viz_text_test_%j.err

set -e

# Use the current directory
cd /scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

# Create logs directory if needed
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true

# Run prediction with the text attention visualization config
torchrun --nproc_per_node=1 src/train.py configs/test_attention_viz_text.yaml

echo "Test complete."
