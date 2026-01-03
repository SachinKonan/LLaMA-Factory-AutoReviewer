#!/bin/bash
#SBATCH --job-name=grpo_ft2k_pli
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/test_rl/pli/grpo_ft2k_%j.out
#SBATCH --error=logs/test_rl/pli/grpo_ft2k_%j.err

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
    --main_process_port 29601 \
    src/train.py configs/grpo_binary.yaml \
    model_name_or_path="saves/qwen2.5-7b/full/grid_searchv2/iclr_clean_binary_no_reviews_v2/checkpoint-2000" \
    dataset="iclr_clean_binary_no_reviews_v2_train" \
    output_dir="saves/qwen2.5-7b/grpo/binary_no_reviews_ft2000"

echo "GRPO training on finetuned-2000 model complete!"
