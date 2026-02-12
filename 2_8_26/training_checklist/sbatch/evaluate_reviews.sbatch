#!/bin/bash
#SBATCH --job-name=eval_reviews
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --output=2_8_26/training_checklist/logs/evaluate_reviews_%j.out
#SBATCH --error=2_8_26/training_checklist/logs/evaluate_reviews_%j.err
#SBATCH --comment="Evaluate reviews with checklist (vLLM)"

set -e

cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer

# Activate vLLM environment
source .vllm/bin/activate

# Create logs directory
mkdir -p 2_8_26/training_checklist/logs
mkdir -p 2_8_26/training_checklist/data

echo "=============================================="
echo "EVALUATE REVIEWS WITH CHECKLIST"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# Run evaluation
python 2_8_26/training_checklist/scripts/evaluate_reviews.py \
    --checklist 2_8_26/checklist_optimization/data/optimal_checklist.json \
    --output 2_8_26/training_checklist/data/review_evaluations.jsonl \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --batch_size 16

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "=============================================="
echo "Output: 2_8_26/training_checklist/data/review_evaluations.jsonl"

# Show summary
if [ -f "2_8_26/training_checklist/data/review_evaluations.jsonl" ]; then
    echo "Total reviews evaluated: $(wc -l < 2_8_26/training_checklist/data/review_evaluations.jsonl)"
else
    echo "ERROR: Output file not found!"
    exit 1
fi

echo "=============================================="
