#!/bin/bash
#SBATCH --job-name=val_txt32
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=logs/auto_inference/val_text_%j.out
#SBATCH --error=logs/auto_inference/val_text_%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python scripts/vllm_infer.py \
  --model_name_or_path saves/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/checkpoint-1322 \
  --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation \
  --template qwen \
  --cutoff_len 24480 \
  --max_new_tokens 1280 \
  --save_logprobs \
  --save_name results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/validation-ckpt-1322.jsonl

echo "Done: validation text inference"
