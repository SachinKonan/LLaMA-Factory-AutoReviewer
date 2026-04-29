#!/bin/bash
#SBATCH --job-name=arxiv_train_smoke
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=logs/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/_smoke/%j.out
#SBATCH --error=logs/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/_smoke/%j.err

# Smoke test that the collaborator runs FIRST to validate end-to-end:
#   - perms allow them to read repo + write to saves/
#   - venv activates and HF cache resolves a model offline
#   - trainer initializes, loads dataset, takes a few optimizer steps
#   - new files inherit the zhuangl group via setgid
#
# 3B Qwen + 1232-sample validation set + max_steps=10 = ~5-10 min on a single
# H100. Cleans up its save dir on EXIT (success or failure) so smoke runs
# leave no residue in saves/.

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export PYTHONUNBUFFERED=1

SMOKE_DIR="saves/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/_smoke/${SLURM_JOB_ID}"
mkdir -p "$SMOKE_DIR"
trap 'echo "[SMOKE] cleaning $SMOKE_DIR"; rm -rf "$SMOKE_DIR"; echo "[SMOKE] cleanup done"' EXIT

echo "=============================================="
echo "[SMOKE] Job: $SLURM_JOB_ID"
echo "[SMOKE] Node: $(hostname)"
echo "[SMOKE] User: $(id -un)"
echo "[SMOKE] Groups: $(id -Gn)"
echo "[SMOKE] Save dir (will be deleted on exit): $SMOKE_DIR"
echo "[SMOKE] Parent perms (group should have rwxs):"
ls -ld "$(dirname "$SMOKE_DIR")"
echo "=============================================="

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
