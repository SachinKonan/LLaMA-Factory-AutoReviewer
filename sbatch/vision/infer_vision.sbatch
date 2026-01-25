#!/bin/bash
#SBATCH --job-name=infer_vision
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/vision_sweep/%j.out
#SBATCH --error=logs/vision_sweep/%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

CKPT_DIR=$(ls -d saves/qwen2.5-vl-7b/full/vision_sweep/vision/checkpoint-* 2>/dev/null | head -1)
DATASET="iclr_vision_binary_no_reviews_v3"
RESULTS_DIR="results/vision_sweep/vision"

mkdir -p "${RESULTS_DIR}"

echo "=============================================="
echo "Job: $SLURM_JOB_ID"
echo "Checkpoint: ${CKPT_DIR}"
echo "Dataset: ${DATASET}_test"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="

python scripts/vllm_infer.py \
    --model_name_or_path "${CKPT_DIR}" \
    --dataset "${DATASET}_test" \
    --template qwen2_vl \
    --cutoff_len 20480 \
    --max_new_tokens 10240 \
    --image_min_pixels 2352 \
    --image_max_pixels 589824 \
    --save_name "${RESULTS_DIR}/finetuned.jsonl"

echo "=============================================="
echo "COMPLETED"
echo "Results: ${RESULTS_DIR}/finetuned.jsonl"
echo "=============================================="
