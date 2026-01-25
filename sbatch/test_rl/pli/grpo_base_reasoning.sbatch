#!/bin/bash
#SBATCH --job-name=grpo_base_r_pli
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/test_rl/pli/grpo_base_reasoning_%j.out
#SBATCH --error=logs/test_rl/pli/grpo_base_reasoning_%j.err

module load proxy/default

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate

export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export WANDB_API_KEY="a9c6eb313dc5aab38bec7680526824ccdbb7f5f0"
export TRANSFORMERS_OFFLINE=1

mkdir -p logs/test_rl/pli

accelerate launch \
    --config_file configs/fsdp2_4gpu_config.yaml \
    --main_process_port 29602 \
    src/train.py configs/grpo_binary.yaml \
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct" \
    dataset="iclr_clean_binary_no_reviews_v2_train" \
    output_dir="saves/qwen2.5-7b/grpo/binary_no_reviews_base_reasoning" \
    grpo_append_reasoning_explicit=true

echo "GRPO training on base model with reasoning complete!"
