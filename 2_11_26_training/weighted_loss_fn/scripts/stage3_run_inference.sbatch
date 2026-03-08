#!/bin/bash
#SBATCH --job-name=infer_weighted_loss
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-18
#SBATCH --output=2_11_26_training/weighted_loss_fn/logs/infer_%A_%a.out
#SBATCH --error=2_11_26_training/weighted_loss_fn/logs/infer_%A_%a.err

# ============================================================================
# Weighted Loss Inference on Test Set
# ============================================================================
# Runs inference on the ORIGINAL TEST SET (2024 samples, 50:50 distribution)
# for all 19 trained models (including baseline).
# ============================================================================

set -e

PROJECT_DIR="/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
export HF_HOME="/scratch/gpfs/ZHUANGL/jl0796/.hf_home"

source .venv/bin/activate

mkdir -p 2_11_26_training/weighted_loss_fn/results

# Decode array index (same logic as training script)
TASK_ID=${SLURM_ARRAY_TASK_ID}

# BASELINE experiment (index 0)
if [ $TASK_ID -eq 0 ]; then
    PROPORTION="1_2"
    VARIANT="baseline"
    GAMMA=1.0

# Balanced dataset weighted experiments (indices 1-6)
elif [ $TASK_ID -le 6 ]; then
    ADJUSTED_ID=$((TASK_ID - 1))
    PROPORTION="1_2"
    VARIANT_IDX=$((ADJUSTED_ID / 3))
    GAMMA_IDX=$((ADJUSTED_ID % 3))

    VARIANTS=("accept" "reject")
    GAMMAS=(2.0 4.0 8.0)

    VARIANT=${VARIANTS[$VARIANT_IDX]}
    GAMMA=${GAMMAS[$GAMMA_IDX]}

# Imbalanced dataset experiments (indices 7-18)
else
    IMBAL_IDX=$((TASK_ID - 7))

    PROP_IDX=$((IMBAL_IDX / 4))
    PROPORTIONS=("1_3" "1_4" "1_8")
    PROPORTION=${PROPORTIONS[$PROP_IDX]}

    LOCAL_IDX=$((IMBAL_IDX % 4))
    VARIANT_IDX=$((LOCAL_IDX / 2))
    GAMMA_IDX=$((LOCAL_IDX % 2))

    VARIANTS=("accept" "reject")
    GAMMAS=(2.0 4.0)

    VARIANT=${VARIANTS[$VARIANT_IDX]}
    GAMMA=${GAMMAS[$GAMMA_IDX]}
fi

# Model checkpoint directory
MODEL_DIR="saves/weighted_loss/${VARIANT}_gamma${GAMMA}_prop${PROPORTION}"

# Output directory
OUTPUT_DIR="2_11_26_training/weighted_loss_fn/results/${VARIANT}_gamma${GAMMA}_prop${PROPORTION}"
mkdir -p "${OUTPUT_DIR}"

# Test dataset (ALWAYS the original test set)
TEST_DATASET="iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test"

echo "=============================================="
echo "Inference: ${VARIANT}_gamma${GAMMA}_prop${PROPORTION}"
echo "Model: ${MODEL_DIR}"
echo "Test Dataset: ${TEST_DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# Run inference with vLLM
python inference_scaling/scripts/vllm_infer_ensemble.py \
    --model_name_or_path "${MODEL_DIR}" \
    --dataset "${TEST_DATASET}" \
    --dataset_dir 2_11_26_training/weighted_loss_fn \
    --template qwen \
    --cutoff_len 4096 \
    --max_new_tokens 1024 \
    --save_name "${OUTPUT_DIR}/predictions.jsonl" \
    --n_generations 1

echo "Inference complete"
