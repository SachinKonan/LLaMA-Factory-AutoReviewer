#!/bin/bash
#SBATCH --job-name=count_tokens
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/count_tokens/%j.out
#SBATCH --error=logs/count_tokens/%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

mkdir -p logs/count_tokens

# Arguments passed via sbatch command line or defaults
CONFIG="${CONFIG:-configs/sweep_final_clean_images.yaml}"
DATASET="${DATASET:-iclr_2020_2025_80_20_split5_balanced_deepreview_clean+images_binary_no_reviews_titleabs_corrected_v3}"
FILTER="${FILTER:-}"
WORKERS="${WORKERS:-16}"

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Config: ${CONFIG}"
echo "Dataset: ${DATASET}"
echo "Workers: ${WORKERS}"
echo "Filter: ${FILTER:-false}"
echo "=============================================="

if [ -n "$FILTER" ]; then
    python scripts/run_tokenization_and_count_overctx.py \
        --config "${CONFIG}" \
        --dataset "${DATASET}" \
        --workers "${WORKERS}" \
        --filter
else
    python scripts/run_tokenization_and_count_overctx.py \
        --config "${CONFIG}" \
        --dataset "${DATASET}" \
        --workers "${WORKERS}"
fi

echo ""
echo "=============================================="
echo "COMPLETED"
echo "=============================================="
