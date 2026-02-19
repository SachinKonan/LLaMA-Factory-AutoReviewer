#!/bin/bash
#SBATCH --job-name=val_infer_tv
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=350G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-test
#SBATCH --array=0-1
#SBATCH --output=logs/final_sweep_v7_datasweepv3/textvisionevalruns/%A_%a.out
#SBATCH --error=logs/final_sweep_v7_datasweepv3/textvisionevalruns/%A_%a.err

# Validation inference for ensemble analysis (array job):
#   Task 0: Text model on text validation set
#   Task 1: Vision model on vision validation set
# Outputs used by plot_text_vs_vision_v7.py Part 7 (ensemble meta-learner)

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

CUTOFF_LEN=24480
MAX_NEW_TOKENS=1280

mkdir -p logs/final_sweep_v7_datasweepv3/textvisionevalruns

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    # ============================================
    # Task 0: Text model inference on validation set
    # ============================================
    CHECKPOINT="saves/final_sweep_v7_datasweepv3/textvisionevalruns/text/checkpoint-1594"
    DATASET="iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered_validation"
    OUTPUT="results/final_sweep_v7_datasweepv3/textvisionevalruns/text/val-ckpt-1594.jsonl"
    TEMPLATE="qwen"
    IMAGE_ARGS=""
    MODALITY="Text"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    # ============================================
    # Task 1: Vision model inference on validation set
    # ============================================
    CHECKPOINT="saves/final_sweep_v7_datasweepv3/textvisionevalruns/vision/checkpoint-798"
    DATASET="iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered_validation"
    OUTPUT="results/final_sweep_v7_datasweepv3/textvisionevalruns/vision/val-ckpt-798.jsonl"
    TEMPLATE="qwen2_vl"
    IMAGE_ARGS="--image_min_pixels 784 --image_max_pixels 1003520"
    MODALITY="Vision"
fi

mkdir -p "$(dirname "${OUTPUT}")"

echo "=============================================="
echo "Validation Inference: ${MODALITY}"
echo "Job ID: ${SLURM_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT}"
echo "Template: ${TEMPLATE}"
echo "=============================================="

python scripts/vllm_infer.py \
    --model_name_or_path "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --save_logprobs \
    ${IMAGE_ARGS} \
    --save_name "${OUTPUT}"

echo ""
echo "=============================================="
echo "COMPLETED: ${MODALITY} validation inference"
echo "Output: ${OUTPUT}"
echo "=============================================="
