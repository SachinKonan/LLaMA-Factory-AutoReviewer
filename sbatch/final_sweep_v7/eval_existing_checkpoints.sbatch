#!/bin/bash
#SBATCH --job-name=eval_ckpts
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=350G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --array=0-11
#SBATCH --output=logs/final_sweep_v7/%A_%a.out
#SBATCH --error=logs/final_sweep_v7/%A_%a.err

# Array job: 12 tasks (0-11)
# Tasks 0-5: balanced_trainagreeing_no2024_vision checkpoints
# Tasks 6-11: balanced_trainagreeing_no2024_vision_lr1e-6 checkpoints

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true

DATASET="iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7"
CHECKPOINTS=(713 1426 2139 2852 3565 4278)

# Determine experiment and checkpoint from array task ID
if [ $SLURM_ARRAY_TASK_ID -lt 6 ]; then
    EXP_DIR="saves/final_sweep_v7/balanced_trainagreeing_no2024_vision"
    RESULTS_DIR="results/final_sweep_v7/balanced_trainagreeing_no2024_vision"
    CKPT_IDX=$SLURM_ARRAY_TASK_ID
else
    EXP_DIR="saves/final_sweep_v7/balanced_trainagreeing_no2024_vision_lr1e-6"
    RESULTS_DIR="results/final_sweep_v7/balanced_trainagreeing_no2024_vision_lr1e-6"
    CKPT_IDX=$((SLURM_ARRAY_TASK_ID - 6))
fi

CKPT=${CHECKPOINTS[$CKPT_IDX]}

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Experiment: ${EXP_DIR}"
echo "Checkpoint: ${CKPT}"
echo "Results: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p "${RESULTS_DIR}" logs/final_sweep_v7

python scripts/eval_training_ckpt.py \
    --model_name_or_path "${EXP_DIR}/checkpoint-${CKPT}" \
    --dataset "${DATASET}_train" \
    --template qwen2_vl \
    --cutoff_len 24480 \
    --image_max_pixels 1003520 \
    --image_min_pixels 784 \
    --per_device_eval_batch_size 4 \
    --max_samples 5000 \
    --save_name "${RESULTS_DIR}/train-ckpt-${CKPT}.json" \
    --test_dataset "${DATASET}_test"

echo "=============================================="
echo "COMPLETED: ${EXP_DIR}/checkpoint-${CKPT}"
echo "=============================================="
