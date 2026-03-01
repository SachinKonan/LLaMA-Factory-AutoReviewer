#!/bin/bash
#SBATCH --job-name=attn_viz_text_balanced
#SBATCH --time=04:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --output=logs/attn_viz_text_balanced_%j.out
#SBATCH --error=logs/attn_viz_text_balanced_%j.err

set -e

cd /scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true

# Random port for distributed
MASTER_PORT=$(( RANDOM % 10000 + 20000 ))

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT src/train.py configs/test_attention_viz_text_balanced.yaml

echo "Balanced Text Test complete."
