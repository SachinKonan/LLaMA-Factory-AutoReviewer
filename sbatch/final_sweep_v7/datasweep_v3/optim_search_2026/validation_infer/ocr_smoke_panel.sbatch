#!/bin/bash
#SBATCH --job-name=ocr_smoke
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/logs/auto_inference/ocr_smoke_%j.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/logs/auto_inference/ocr_smoke_%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf

mkdir -p logs/auto_inference results/ocr_smoke

# Base Qwen2.5-VL-7B-Instruct (NOT the FT'd checkpoint — we want raw OCR ability).
python scripts/vllm_infer.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset ocr_smoke \
  --template qwen2_vl \
  --cutoff_len 8192 \
  --max_new_tokens 5120 \
  --image_min_pixels 784 \
  --image_max_pixels 4014080 \
  --save_logprobs False \
  --save_name results/ocr_smoke/panel_2380x1512_base_qwen.jsonl

echo "Done: OCR smoke test"
