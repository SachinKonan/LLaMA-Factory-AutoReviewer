#!/bin/bash
#SBATCH --job-name=val_vis16
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=logs/auto_inference/val_vision_%j.out
#SBATCH --error=logs/auto_inference/val_vision_%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python scripts/vllm_infer.py \
  --model_name_or_path saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/checkpoint-2648 \
  --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_validation \
  --template qwen2_vl \
  --cutoff_len 24480 \
  --max_new_tokens 1280 \
  --save_logprobs \
  --image_min_pixels 784 \
  --image_max_pixels 1003520 \
  --save_name results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/validation-ckpt-2648.jsonl

echo "Done: validation vision inference"
