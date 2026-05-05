#!/bin/bash
#SBATCH --job-name=ipa2axt
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --array=0-3
#SBATCH --output=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/logs/auto_inference/iclr_panel_to_arxiv_%A_%a.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/logs/auto_inference/iclr_panel_to_arxiv_%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

mkdir -p logs/auto_inference results/iclr_panel_to_arxiv

CKPTS=(1324 2648 3972 5296)
CKPT="${CKPTS[$SLURM_ARRAY_TASK_ID]}"

MODEL=saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_panel/checkpoint-${CKPT}
OUT=results/iclr_panel_to_arxiv/arxiv_test-ckpt-${CKPT}.jsonl

echo "=========================================="
echo "Eval ICLR-trained panel checkpoint on arxiv panel test"
echo "  ckpt:    ${CKPT}"
echo "  model:   ${MODEL}"
echo "  dataset: arxiv_50_50_21k_vision_wmetadata_filtered24480_panel_test"
echo "  out:     ${OUT}"
echo "=========================================="

python scripts/vllm_infer.py \
  --model_name_or_path "${MODEL}" \
  --dataset arxiv_50_50_21k_vision_wmetadata_filtered24480_panel_test \
  --template qwen2_vl \
  --cutoff_len 8192 \
  --max_new_tokens 1280 \
  --save_logprobs \
  --image_min_pixels 784 \
  --image_max_pixels 4014080 \
  --save_name "${OUT}"

echo "Done: ckpt-${CKPT} -> ${OUT}"
