#!/bin/bash
#SBATCH --job-name=token_stats
#SBATCH --array=0-7
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/token_stats/%A_%a.out
#SBATCH --error=logs/token_stats/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

mkdir -p logs/token_stats figures/token_stats

# Dataset array: 4 text + 4 vision = 8 datasets
DATASETS=(
    # Text datasets
    "iclr_2017_2019_original_text_v7"
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_v7_filtered"
    "nips_2021_2025_original_text_v7_noref"
    "icml_colm_2024_2025_clean_binary_noref"
    # Vision datasets
    "iclr_2017_2019_original_vision_v7"
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_v7_filtered"
    "nips_2021_2025_original_vision_v7"
    "icml_colm_2024_2025_vision_binary"
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

echo "=============================================="
echo "Job: $SLURM_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "=============================================="

python scripts/count_token_stats.py \
    --dataset "${DATASET}" \
    --workers 16

echo ""
echo "=============================================="
echo "COMPLETED: ${DATASET}"
echo "=============================================="
