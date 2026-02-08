#!/bin/bash
# Resubmit all failed think token checkpoint evaluations

DATASET="iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7"
TEMPLATE="qwen2_vl"
CUTOFF_LEN=26480
IMAGE_MIN_PIXELS=784
IMAGE_MAX_PIXELS=1003520

# Array of all checkpoints that need evaluation
CHECKPOINTS=(
  "saves/final_sweep_v7_pli/trainagreeing_vision_think1000_input/checkpoint-713"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think1000_input/checkpoint-1426"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think1000_input/checkpoint-2139"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think1000_input/checkpoint-2852"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think1000_input/checkpoint-3565"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think1000_label/checkpoint-713"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think1000_label/checkpoint-1426"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think1000_label/checkpoint-2139"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think1000_label/checkpoint-2852"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think1000_label/checkpoint-3565"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think100_input/checkpoint-713"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think100_input/checkpoint-1426"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think100_input/checkpoint-2139"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think100_input/checkpoint-2852"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think100_input/checkpoint-3565"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think100_input/checkpoint-4278"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think100_label/checkpoint-713"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think100_label/checkpoint-1426"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think100_label/checkpoint-2139"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think100_label/checkpoint-2852"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think100_label/checkpoint-3565"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think100_label/checkpoint-4278"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think10_input/checkpoint-713"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think10_input/checkpoint-1426"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think10_input/checkpoint-2139"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think10_input/checkpoint-2852"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think10_input/checkpoint-3565"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think10_input/checkpoint-4278"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think10_label/checkpoint-713"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think10_label/checkpoint-1426"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think10_label/checkpoint-2139"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think10_label/checkpoint-2852"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think10_label/checkpoint-3565"
  "saves/final_sweep_v7_pli/trainagreeing_vision_think10_label/checkpoint-4278"
)

echo "Submitting ${#CHECKPOINTS[@]} checkpoint evaluations..."

for CKPT_PATH in "${CHECKPOINTS[@]}"; do
  # Extract info from path
  EXP_NAME=$(echo $CKPT_PATH | cut -d'/' -f3)
  CKPT_NUM=$(basename $CKPT_PATH | cut -d'-' -f2)
  RESULT_FILE="results/final_sweep_v7_pli/${EXP_NAME}/train-ckpt-${CKPT_NUM}.json"

  # Create results directory
  mkdir -p "results/final_sweep_v7_pli/${EXP_NAME}"

  # Submit sbatch job
  sbatch --partition=pli --account=llm_explore \
    --job-name=auto_infer_vision \
    --time=02:00:00 \
    --cpus-per-task=10 \
    --mem=200G \
    --gres=gpu:1 \
    --output="logs/auto_inference/%j.out" \
    --error="logs/auto_inference/%j.err" \
    --wrap="cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer && \
source .venv/bin/activate && \
export TRANSFORMERS_OFFLINE=1 && \
python scripts/eval_training_ckpt.py \
  --model_name_or_path '${CKPT_PATH}' \
  --dataset '${DATASET}_train' \
  --template '${TEMPLATE}' \
  --cutoff_len ${CUTOFF_LEN} \
  --save_name '${RESULT_FILE}' \
  --sft_accuracy_format boxed \
  --sft_positive_token Accept \
  --sft_negative_token Reject \
  --per_device_eval_batch_size 2 \
  --max_samples 2000 \
  --test_dataset '${DATASET}_test' \
  --image_max_pixels ${IMAGE_MAX_PIXELS} \
  --image_min_pixels ${IMAGE_MIN_PIXELS}"

  echo "Submitted: $EXP_NAME checkpoint-${CKPT_NUM}"
  sleep 0.5  # Small delay to avoid overwhelming scheduler
done

echo ""
echo "All ${#CHECKPOINTS[@]} jobs submitted!"
echo "Monitor with: squeue -u \$USER"
