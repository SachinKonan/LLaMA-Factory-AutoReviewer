#!/bin/bash
#SBATCH --job-name=alignment_text_infer
#SBATCH --output=logs/alignment_text_infer_%A_%a.out
#SBATCH --error=logs/alignment_text_infer_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=8

cd /scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer
source .venv_vllm_inf/bin/activate

mkdir -p alignment_analysis/results

python alignment_analysis/vllm_infer.py \
  --model_name_or_path /scratch/gpfs/ZHUANGL/jl0796/shared/saves/best_2025_2026_text/checkpoint-1322 \
  --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test \
  --dataset_dir /scratch/gpfs/ZHUANGL/jl0796/shared/data \
  --template qwen \
  --cutoff_len 24480 \
  --shard_id $SLURM_ARRAY_TASK_ID \
  --num_shards 4 \
  --save_name alignment_analysis/results/text_predictions_shard_${SLURM_ARRAY_TASK_ID}.jsonl \
  --save_logprobs True \
  --ids_file iclr_2020_2023_2025_2026_85_5_10_balanced_original_v7_filtered_test_500_subset.txt

echo "Shard ${SLURM_ARRAY_TASK_ID} complete."
