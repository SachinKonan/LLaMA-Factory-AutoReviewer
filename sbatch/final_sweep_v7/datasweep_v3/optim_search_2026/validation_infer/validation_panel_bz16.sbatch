#!/bin/bash
#SBATCH --job-name=val_panel
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/logs/auto_inference/val_panel_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/logs/auto_inference/val_panel_%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

mkdir -p logs/auto_inference

python scripts/vllm_infer.py \
  --model_name_or_path saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/checkpoint-2648 \
  --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_panel_validation \
  --template qwen2_vl \
  --cutoff_len 24480 \
  --max_new_tokens 1280 \
  --save_logprobs \
  --image_min_pixels 784 \
  --image_max_pixels 4014080 \
  --save_name results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/validation-ckpt-2648-panel.jsonl

echo "Done: panel validation vision inference"
