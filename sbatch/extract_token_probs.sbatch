#!/bin/bash
#SBATCH --job-name=extract_probs
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/extract_probs_%j.out
#SBATCH --error=logs/extract_probs_%j.err

set -e
cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer
source .venv_vllm_inf/bin/activate
export TRANSFORMERS_OFFLINE=1

# ============================================
# Fill in these paths
# ============================================
# Text
CHECKPOINT="saves/qwen2.5-3b/full/sft_ds3/checkpoint-3750"
DATASET="iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_test"
OUTPUT="results/probs/text-checkpoint-3750_probs.jsonl"
TEMPLATE="qwen"
BOXED="--boxed"


# Vision
# CHECKPOINT="saves/qwen2.5vl-3b/full/sft_ds3/checkpoint-2500"
# DATASET="iclr_2020_2025_85_5_10_split6_original_vision_binary_noreviews_v6_test"
# OUTPUT="results/probs/vision-checkpoint-2500_probs.jsonl"
# TEMPLATE="qwen2_vl"
# BOXED="--boxed"

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET}"
echo "Template: ${TEMPLATE}"
echo "Output: ${OUTPUT}"
echo "Boxed mode: ${BOXED:-disabled}"
echo "=============================================="

mkdir -p "$(dirname "${OUTPUT}")"

# Add --media_dir only for vision models
MEDIA_DIR_ARG=""
if [[ "${TEMPLATE}" == *"vl"* ]]; then
    MEDIA_DIR_ARG="--media_dir ."
fi

python scripts/extract_token_probs.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    ${MEDIA_DIR_ARG} \
    --template "${TEMPLATE}" \
    --cutoff_len 16384 \
    --output "${OUTPUT}" \
    --batch_size 1 \
    ${BOXED}

echo "Extraction complete!"
