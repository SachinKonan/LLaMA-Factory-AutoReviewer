#!/bin/bash
#SBATCH --job-name=cache_tokenize
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --output=logs/cache_tokenize_%j.out
#SBATCH --error=logs/cache_tokenize_%j.err

# Pre-tokenize dataset on CPU to avoid GPU idle timeout during training
#
# Usage:
#   sbatch sbatch/cache_tokenized_dataset.sbatch
#
# After this completes, update your training config to use:
#   tokenized_path: cache/tokenized_qwen2_5_3bvl_sft

set -e

cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Create directories
mkdir -p logs cache

echo ""
echo "Starting dataset tokenization (CPU-only)..."
echo "This will save the tokenized dataset to cache/tokenized_qwen2_5_3bvl_sft_text+images"
echo ""

# Run with CUDA disabled to force CPU-only
CUDA_VISIBLE_DEVICES="" python src/train.py configs/qwen2_5_3bvl_cache_tokenized_text+images.yaml

echo ""
echo "Tokenization complete!"
echo ""
echo "Next steps:"
echo "1. Add 'tokenized_path: cache/tokenized_qwen2_5_3bvl_sft_text+images' to your training config"
echo "2. Set 'overwrite_cache: false' in your training config"
echo "3. Run your GPU training job"
