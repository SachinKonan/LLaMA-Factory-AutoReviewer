#!/bin/bash
#SBATCH --job-name=pl_golden_scoring
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-test
#SBATCH --output=tests/scoring/runs/gen_%j.out
#SBATCH --error=tests/scoring/runs/gen_%j.err
#
# Generate golden scoring fixtures + expected_scores.json on a GPU node.
#   sbatch tests/scoring/generate_fixtures.sbatch
# Run from the paperlenstraininfer worktree root.

set -euo pipefail

TI=/scratch/gpfs/ZHUANGL/sk7524/PaperLens/paperlens-training-and-inference
PAPERPREP=/scratch/gpfs/ZHUANGL/sk7524/PaperLens/paperprep
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$TI"
mkdir -p tests/scoring/runs

echo "[sbatch] node=$(hostname)  job=$SLURM_JOB_ID"
nvidia-smi -L | head -1

# Reuse an existing work dir (skips MinerU if prep_out is present) by setting
#   sbatch --export=ALL,WORK=/abs/path/to/work_NNN  tests/scoring/generate_fixtures.sbatch
WORK="${WORK:-$TI/tests/scoring/runs/work_${SLURM_JOB_ID}}"
echo "[sbatch] WORK=$WORK"

# Driver runs under the paperlenstraininfer venv (the real deployment scorer
# env: vLLM 0.10.1, paperlens_cli installed). It spawns paperprep in the
# paperprep venv and vllm_infer / Scorer in this one.
"$TI/.venv/bin/python" tests/scoring/generate_fixtures.py \
    --work "$WORK" \
    --paperprep-venv "$PAPERPREP/.venv" \
    --paperprep-repo "$PAPERPREP"

echo "[sbatch] done — fixtures under tests/scoring/fixtures/, expected_scores.json written"
