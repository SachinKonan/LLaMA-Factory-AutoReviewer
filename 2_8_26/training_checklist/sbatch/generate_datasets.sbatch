#!/bin/bash
#SBATCH --job-name=gen_datasets
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=2_8_26/training_checklist/logs/generate_datasets_%j.out
#SBATCH --error=2_8_26/training_checklist/logs/generate_datasets_%j.err
#SBATCH --comment="Generate SFT and DPO datasets"

set -e

cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer

# Can use base Python environment
source .venv/bin/activate

echo "=============================================="
echo "GENERATE TRAINING DATASETS"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "=============================================="

# Check input exists
if [ ! -f "2_8_26/training_checklist/data/review_evaluations.jsonl" ]; then
    echo "ERROR: review_evaluations.jsonl not found!"
    echo "Please run evaluate_reviews.sbatch first"
    exit 1
fi

# Generate datasets
python 2_8_26/training_checklist/scripts/generate_training_data.py \
    --review_evaluations 2_8_26/training_checklist/data/review_evaluations.jsonl \
    --output_dir 2_8_26/training_checklist/data \
    --top_percentile 0.8 \
    --max_dpo_pairs 2 \
    --seed 42

echo ""
echo "=============================================="
echo "VALIDATING DATASETS"
echo "=============================================="

# Validate datasets
python 2_8_26/training_checklist/scripts/validate_datasets.py \
    --data_dir 2_8_26/training_checklist/data \
    --stage all

echo ""
echo "=============================================="
echo "DATASET GENERATION COMPLETE"
echo "=============================================="
echo "Datasets ready in: 2_8_26/training_checklist/data/"
echo ""
echo "Next step: Submit training jobs"
echo "  sbatch --array=0-5 2_8_26/training_checklist/sbatch/train_pipeline.sbatch"
echo "=============================================="
