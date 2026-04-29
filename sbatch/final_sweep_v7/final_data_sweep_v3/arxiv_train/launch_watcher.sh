#!/bin/bash
# Launch the auto-inference watcher for ONE arxiv-train cell.
# Runs `nohup python scripts/auto_infer_watcher.py ...` in the background;
# the watcher polls saves/.../arxiv_train/<size>/<short>/ for new
# checkpoints and submits inference to PLI under the invoking user's
# account (sk7524). The training itself runs separately on ailab from a
# collaborator's account; this script is for sk7524 to run on login.
#
# Usage:
#   bash launch_watcher.sh <cell>
#
# <cell> is one of:
#   small_text   small_vision   large_text   large_vision
#
# Each watcher writes its own log under logs/auto_inference/.

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
mkdir -p logs/auto_inference

CELL="${1:?usage: launch_watcher.sh <small_text|small_vision|large_text|large_vision>}"

ICLR_TEXT_TEST="iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test"
ICLR_VISION_TEST="iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test"
ARXIV_TEXT_Y24UP="arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test"
ARXIV_VISION_Y24UP="arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_test"

case "$CELL" in
small_text)
    SHORT=arxiv_21k_text
    TEMPLATE=qwen
    SIZE=small
    DATASET=arxiv_50_50_21k_text_wmetadata_filtered24480_train
    TESTS="${ARXIV_TEXT_Y24UP}:arxiv_y24up,${ICLR_TEXT_TEST}:iclr_2526"
    IMG_ARGS=()
    ;;
small_vision)
    SHORT=arxiv_21k_vision
    TEMPLATE=qwen2_vl
    SIZE=small
    DATASET=arxiv_50_50_21k_vision_wmetadata_filtered24480_train
    TESTS="${ARXIV_VISION_Y24UP}:arxiv_y24up,${ICLR_VISION_TEST}:iclr_2526"
    IMG_ARGS=(--image_min_pixels 784 --image_max_pixels 1003520)
    ;;
large_text)
    SHORT=arxiv_balanced_per_venue_text
    TEMPLATE=qwen
    SIZE=large
    DATASET=arxiv_50_50_balanced_per_venue_text_wmetadata_train
    TESTS="${ARXIV_TEXT_Y24UP}:arxiv_y24up,${ICLR_TEXT_TEST}:iclr_2526"
    IMG_ARGS=()
    ;;
large_vision)
    SHORT=arxiv_balanced_per_venue_vision
    TEMPLATE=qwen2_vl
    SIZE=large
    DATASET=arxiv_50_50_balanced_per_venue_vision_wmetadata_train
    TESTS="${ARXIV_VISION_Y24UP}:arxiv_y24up,${ICLR_VISION_TEST}:iclr_2526"
    IMG_ARGS=(--image_min_pixels 784 --image_max_pixels 1003520)
    ;;
*)
    echo "ERROR: unknown cell '$CELL'"
    echo "  must be one of: small_text small_vision large_text large_vision"
    exit 2
    ;;
esac

SAVE_DIR="saves/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/${SIZE}/${SHORT}"
RES_DIR="results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/${SIZE}/${SHORT}"
mkdir -p "$SAVE_DIR" "$RES_DIR"

LOG_FILE="logs/auto_inference/watcher_${CELL}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting watcher for cell=${CELL}"
echo "  save_dir:    ${SAVE_DIR}"
echo "  results_dir: ${RES_DIR}"
echo "  template:    ${TEMPLATE}"
echo "  tests:       ${TESTS}"
echo "  log:         ${LOG_FILE}"

nohup python scripts/auto_infer_watcher.py \
    --save_dir "$SAVE_DIR" \
    --results_dir "$RES_DIR" \
    --dataset "$DATASET" \
    --test_datasets "$TESTS" \
    --template "$TEMPLATE" \
    --cutoff_len 24480 \
    --max_new_tokens 1280 \
    --save_logprobs \
    --keep_safetensors_epoch_idx 2 \
    --poll_interval 60 \
    "${IMG_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &

WATCHER_PID=$!
disown $WATCHER_PID
echo "watcher started: pid=${WATCHER_PID} cell=${CELL}"
echo "tail -f $LOG_FILE   # to follow"
echo "kill ${WATCHER_PID}    # to stop"
