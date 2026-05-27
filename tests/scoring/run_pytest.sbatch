#!/bin/bash
#SBATCH --job-name=pl_golden_pytest
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-test
#SBATCH --output=tests/scoring/runs/pytest_%j.out
#SBATCH --error=tests/scoring/runs/pytest_%j.err
#
# Run the golden scoring integration test against the checked-in
# expected_scores.json. Confirms the deployment Scorer reproduces the
# offline batch results on a fresh GPU allocation.

set -euo pipefail

TI=/scratch/gpfs/ZHUANGL/sk7524/PaperLens/paperlens-training-and-inference
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$TI"
echo "[sbatch] node=$(hostname)  job=$SLURM_JOB_ID"
nvidia-smi -L | head -1

"$TI/.venv/bin/python" -m pytest tests/scoring/test_golden_scoring.py -m gpu -v -s
