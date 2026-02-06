#!/bin/bash
#
# Test script for auto_infer_watcher.py
#
# This script simulates a training run and tests the auto-inference watcher
# in dry-run mode to verify it would submit the correct jobs.
#
# Usage:
#   ./sbatch/final_sweep_v7/_test_auto_infer_watcher.sh
#
# What it does:
#   1. Creates a temp directory for fake training outputs
#   2. Starts the auto_infer_watcher in dry-run mode (background)
#   3. Runs simulate_training.py to create fake checkpoints
#   4. Watches the watcher detect and "submit" jobs (dry-run)
#   5. Cleans up
#

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

# Test configuration
TEST_DIR="/tmp/test_auto_infer_$$"
MODEL_DIR="${TEST_DIR}/saves/test_experiment"
RESULTS_DIR="${TEST_DIR}/results/test_experiment"
DATASET="iclr_2020_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7"
TEMPLATE="qwen2_vl"
CUTOFF_LEN=24480
IMAGE_MIN_PIXELS=784
IMAGE_MAX_PIXELS=1003520

# Cleanup function
cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    kill $WATCHER_PID 2>/dev/null || true
    rm -rf "$TEST_DIR"
    echo "Removed test directory: $TEST_DIR"
}
trap cleanup EXIT

echo "=============================================="
echo "Auto-Inference Watcher Test"
echo "=============================================="
echo "Test directory: ${TEST_DIR}"
echo "Model directory: ${MODEL_DIR}"
echo "Results directory: ${RESULTS_DIR}"
echo "Dataset: ${DATASET}"
echo "Template: ${TEMPLATE}"
echo "=============================================="

# Create directories
mkdir -p "${MODEL_DIR}" "${RESULTS_DIR}"

# ======================
# Start watcher in dry-run mode
# ======================
echo ""
echo "=== Starting auto-inference watcher (dry-run mode) ==="
echo ""

python scripts/auto_infer_watcher.py \
    --save_dir "${MODEL_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --dataset "${DATASET}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 1280 \
    --image_min_pixels "${IMAGE_MIN_PIXELS}" \
    --image_max_pixels "${IMAGE_MAX_PIXELS}" \
    --poll_interval 5 \
    --dry_run &
WATCHER_PID=$!
echo "Watcher started with PID: ${WATCHER_PID}"

# Give watcher time to start
sleep 2

# ======================
# Simulate training (create fake checkpoints)
# ======================
echo ""
echo "=== Starting training simulator ==="
echo ""

python scripts/simulate_training.py \
    --save_dir "${MODEL_DIR}" \
    --num_checkpoints 4 \
    --checkpoint_interval 8 \
    --steps_per_checkpoint 100

# Wait a bit for watcher to process final checkpoint detection
echo ""
echo "=== Waiting for watcher to detect completion ==="
sleep 10

# ======================
# Show results
# ======================
echo ""
echo "=== Test Results ==="
echo ""

echo "Checkpoints created:"
ls -la "${MODEL_DIR}"/checkpoint-* 2>/dev/null || echo "  (none)"

echo ""
echo "Touch files created (jobs that would have been submitted):"
find "${MODEL_DIR}" -name ".infer.touch" -exec echo "  {}" \; -exec cat {} \; 2>/dev/null || echo "  (none)"

echo ""
echo "Watcher log:"
if [ -f "${MODEL_DIR}/auto_infer_watcher.log" ]; then
    cat "${MODEL_DIR}/auto_infer_watcher.log"
else
    echo "  (no log file found)"
fi

echo ""
echo "=============================================="
echo "Test Complete!"
echo "=============================================="
