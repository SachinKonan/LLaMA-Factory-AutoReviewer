#!/bin/bash
#SBATCH --job-name=i8p2axt
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/logs/auto_inference/iclr8page_to_arxiv_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/logs/auto_inference/iclr8page_to_arxiv_%j.err
set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
mkdir -p logs/auto_inference results/iclr_panel_to_arxiv
python scripts/vllm_infer.py \
  --model_name_or_path saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/checkpoint-2648 \
  --dataset arxiv_50_50_21k_vision_wmetadata_filtered24480_test \
  --template qwen2_vl --cutoff_len 24480 --max_new_tokens 1280 --save_logprobs \
  --image_min_pixels 784 --image_max_pixels 1003520 \
  --save_name results/iclr_panel_to_arxiv/EIGHTPAGE_arxiv_test-ckpt-2648.jsonl
echo done
