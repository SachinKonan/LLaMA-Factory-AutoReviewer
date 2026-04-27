#!/bin/bash
#SBATCH --job-name=test_keepalive
#SBATCH --partition=gpu-test
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --time=00:30:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/logs/keepalive_smoke_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/logs/keepalive_smoke_%j.err

# Smoke test for scripts/gpu_keepalive.py.
# Required env: MODE in {success, fail, fail-release}, MODEL_DIR (per-test scratch dir).
# Optional:     SELF_RELEASE_AFTER (seconds; default 90, only used for fail-release).

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p logs

if [ -z "$MODE" ] || [ -z "$MODEL_DIR" ]; then
    echo "ERROR: MODE and MODEL_DIR env vars must be set"
    exit 2
fi

SELF_RELEASE_AFTER=${SELF_RELEASE_AFTER:-90}

echo "=============================================="
echo "Keepalive smoke test"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $(hostname)"
echo "  Mode:      $MODE"
echo "  Model dir: $MODEL_DIR"
echo "  Self-release after: ${SELF_RELEASE_AFTER}s (fail-release only)"
echo "=============================================="

mkdir -p "$MODEL_DIR"

python scripts/test_keepalive.py \
    --mode "$MODE" \
    --model_dir "$MODEL_DIR" \
    --self_release_after "$SELF_RELEASE_AFTER"

RC=$?
echo "=============================================="
echo "test_keepalive.py exited rc=$RC"
echo "=============================================="
exit $RC
