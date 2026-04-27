#!/bin/bash
#SBATCH --job-name=train_acc_only
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/auto_inference/train_acc_%j.out
#SBATCH --error=logs/auto_inference/train_acc_%j.err

# Standalone train-accuracy evaluation (Step 1 of text/vision_inference.sbatch).
# Computes eval_sft_accuracy on a 2k slice of the model's training data.
#
# Required env vars:
#   CHECKPOINT_DIR   - Path to checkpoint
#   TRAIN_DATASET    - Train dataset name (model's own training distribution)
#   TEMPLATE         - Template name (qwen for text, qwen2_vl for vision)
#   CUTOFF_LEN       - Cutoff length (24480)
#   CKPT_STEP        - Checkpoint step number
#   SAVE_DIR         - Where to write train-ckpt-{step}.json
#
# Optional:
#   TRAIN_ACCURACY_MAX_SAMPLES (default 2000)
#   TRAIN_ACCURACY_BATCH_SIZE  (default 2)
#   IMAGE_MIN_PIXELS           (vision only)
#   IMAGE_MAX_PIXELS           (vision only)
#   TEST_DATASET               (default: TRAIN_DATASET with _train -> _test)

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Validate
for v in CHECKPOINT_DIR TRAIN_DATASET TEMPLATE CKPT_STEP SAVE_DIR; do
    if [ -z "${!v}" ]; then
        echo "ERROR: $v not set"
        exit 1
    fi
done

CUTOFF_LEN=${CUTOFF_LEN:-24480}
TRAIN_ACCURACY_MAX_SAMPLES=${TRAIN_ACCURACY_MAX_SAMPLES:-2000}
TRAIN_ACCURACY_BATCH_SIZE=${TRAIN_ACCURACY_BATCH_SIZE:-2}

# Default test dataset (used to label the eval JSON; not actually evaluated on)
if [ -z "$TEST_DATASET" ]; then
    TEST_DATASET="${TRAIN_DATASET%_train}_test"
fi

mkdir -p logs/auto_inference "${SAVE_DIR}"
SAVE_NAME="${SAVE_DIR}/train-ckpt-${CKPT_STEP}.json"

echo "=============================================="
echo "Train-Acc Only Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Train Dataset: ${TRAIN_DATASET}"
echo "Test Dataset (label): ${TEST_DATASET}"
echo "Template: ${TEMPLATE}"
echo "Step: ${CKPT_STEP}"
echo "Save: ${SAVE_NAME}"
echo "Cutoff Len: ${CUTOFF_LEN}"
echo "Max Samples: ${TRAIN_ACCURACY_MAX_SAMPLES}"
echo "Batch Size: ${TRAIN_ACCURACY_BATCH_SIZE}"
echo "Image Min/Max: ${IMAGE_MIN_PIXELS:-<none>} / ${IMAGE_MAX_PIXELS:-<none>}"
echo "=============================================="

EXTRA_ARGS=""
if [ -n "${IMAGE_MIN_PIXELS:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --image_min_pixels ${IMAGE_MIN_PIXELS}"
fi
if [ -n "${IMAGE_MAX_PIXELS:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --image_max_pixels ${IMAGE_MAX_PIXELS}"
fi

python scripts/eval_training_ckpt.py \
    --model_name_or_path "${CHECKPOINT_DIR}" \
    --dataset "${TRAIN_DATASET}" \
    --template "${TEMPLATE}" \
    --cutoff_len ${CUTOFF_LEN} \
    --save_name "${SAVE_NAME}" \
    --sft_accuracy_format boxed \
    --sft_positive_token Accept \
    --sft_negative_token Reject \
    --per_device_eval_batch_size ${TRAIN_ACCURACY_BATCH_SIZE} \
    --max_samples ${TRAIN_ACCURACY_MAX_SAMPLES} \
    --test_dataset "${TEST_DATASET}" \
    ${EXTRA_ARGS}

echo "DONE"
