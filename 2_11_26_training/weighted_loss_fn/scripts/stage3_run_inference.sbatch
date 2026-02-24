#!/bin/bash
#SBATCH --job-name=infer_weighted_loss
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-35
#SBATCH --output=2_11_26_training/weighted_loss_fn/logs/infer_%A_%a.out
#SBATCH --error=2_11_26_training/weighted_loss_fn/logs/infer_%A_%a.err

# ============================================================================
# Weighted Loss Inference on Test Set
# ============================================================================
# Runs inference on the ORIGINAL TEST SET (2024 samples, 50:50 distribution)
# for trained models.
# ============================================================================

set -e

PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
export HF_HOME="/n/fs/vision-mix/jl0796/.hf_home"

source .venv/bin/activate

mkdir -p 2_11_26_training/weighted_loss_fn/results

# Decode array index (MUST match stage2_train_models_baseline.sbatch)
TASK_ID=${SLURM_ARRAY_TASK_ID}

ALL_PROPORTIONS=("1_2" "1_3" "1_4" "1_8")
GAMMAS=(2.0 3.0 4.0 8.0)
VARIANTS=("accept" "reject")

# Indices 0-3: Standard BCE baselines (no weighted loss)
if [ $TASK_ID -le 3 ]; then
    PROPORTION=${ALL_PROPORTIONS[$TASK_ID]}
    VARIANT="baseline"
    GAMMA=1.0

# Indices 4-35: Weighted loss experiments for all proportions
# (Note: Logic generalized code from stage2 for all indices 4-35)
else
    WEIGHTED_IDX=$((TASK_ID - 4))  # Offset to 0-31

    # 8 experiments per proportion (4 gammas × 2 variants)
    PROP_IDX=$((WEIGHTED_IDX / 8))
    LOCAL_IDX=$((WEIGHTED_IDX % 8))
    GAMMA_IDX=$((LOCAL_IDX / 2))
    VARIANT_IDX=$((LOCAL_IDX % 2))

    PROPORTION=${ALL_PROPORTIONS[$PROP_IDX]}
    GAMMA=${GAMMAS[$GAMMA_IDX]}
    VARIANT=${VARIANTS[$VARIANT_IDX]}
fi

# Model checkpoint directory
MODEL_DIR="saves/weighted_loss/${VARIANT}_gamma${GAMMA}_prop${PROPORTION}/checkpoint-268"

# Check if model exists; skip if missing
if [ ! -d "${MODEL_DIR}" ]; then
    echo "============================================="
    echo "SKIPPING Task $TASK_ID: Model not found"
    echo "Expected: ${MODEL_DIR}"
    echo "Status: Training may still be in progress"
    echo "============================================="
    exit 0
fi

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
    --cutoff_len 20000 \
    --max_new_tokens 1024 \
    --save_name "${OUTPUT_DIR}/predictions.jsonl" \
    --n_generations 1

echo "Inference complete"
