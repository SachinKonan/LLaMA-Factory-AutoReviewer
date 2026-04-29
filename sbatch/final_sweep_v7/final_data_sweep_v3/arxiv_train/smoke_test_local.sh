#!/bin/bash
# Local-mode counterpart of smoke_test_gputest.sbatch.
#
# Run directly on a node that already has a GPU (interactive srun, della-vis*,
# any local GPU). No sbatch -- launches python in the foreground. Same
# end-to-end pipeline test:
#   - cd into repo (group rX), source venv, set HF_HOME
#   - load Qwen2.5-3B-Instruct from sk7524's HF cache (group readable)
#   - load arxiv val dataset (group readable)
#   - run trainer for 10 steps (writes go to saves/, group writable + setgid)
#   - clean up the smoke save dir on EXIT (success or failure)
#
# Usage (from anywhere):
#   bash /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/sbatch/final_sweep_v7/final_data_sweep_v3/arxiv_train/smoke_test_local.sh
#
# Or after cd-ing into the repo:
#   ./sbatch/final_sweep_v7/final_data_sweep_v3/arxiv_train/smoke_test_local.sh

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export PYTHONUNBUFFERED=1

# Unique save-dir per invocation (timestamp + PID, since no SLURM_JOB_ID)
SMOKE_TAG="local_$(date +%Y%m%d_%H%M%S)_$$"
SMOKE_DIR="saves/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/_smoke/${SMOKE_TAG}"
mkdir -p "$SMOKE_DIR"
trap 'echo ""; echo "[SMOKE] cleaning $SMOKE_DIR"; rm -rf "$SMOKE_DIR"; echo "[SMOKE] cleanup done"' EXIT

echo "=============================================="
echo "[SMOKE] LOCAL mode (no sbatch)"
echo "[SMOKE] Tag:    $SMOKE_TAG"
echo "[SMOKE] Node:   $(hostname)"
echo "[SMOKE] User:   $(id -un)"
echo "[SMOKE] Groups: $(id -Gn)"
echo "[SMOKE] Save dir (will be deleted on exit): $SMOKE_DIR"
echo "[SMOKE] Parent perms (group should have rwxs):"
ls -ld "$(dirname "$SMOKE_DIR")"
echo "=============================================="

# Sanity-check GPU availability (fail fast if no GPU visible)
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[SMOKE] FAIL: nvidia-smi not on PATH. Are you on a GPU node?"
    exit 1
fi
n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$n_gpu" -lt 1 ]; then
    echo "[SMOKE] FAIL: 0 GPUs visible. Run on a GPU node (dellaviz*, srun --gres=gpu:1, etc.)."
    nvidia-smi 2>&1 | head -10
    exit 1
fi
echo "[SMOKE] $n_gpu GPU(s) visible:"
nvidia-smi -L 2>&1 | head -4

python src/train.py configs/final_sweep_v7_clean.yaml \
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct" \
    dataset="arxiv_50_50_21k_text_wmetadata_filtered24480_validation" \
    do_eval=false \
    output_dir="$SMOKE_DIR" \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=1 \
    cutoff_len=4096 \
    max_steps=10 \
    save_strategy=no \
    learning_rate=1e-6 \
    report_to=none

echo ""
echo "[SMOKE] trainer exited cleanly (10 steps)"
echo "[SMOKE] artifacts before cleanup:"
ls -la "$SMOKE_DIR" 2>/dev/null | head -10 || echo "  (none)"
echo ""
echo "[SMOKE] PASS: perms work, trainer runs, cleanup will fire"
