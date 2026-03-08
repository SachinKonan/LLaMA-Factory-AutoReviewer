#!/bin/bash
#SBATCH --job-name=ablation_infer
#SBATCH --error=ablations/logs/inference_%j.err
#SBATCH --output=ablations/logs/inference_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=4:00:00
#SBATCH --comment="Ablation Inference"

#SBATCH --array=0-14

source /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/.vllm/bin/activate

# Create logs dir
mkdir -p ablations/logs

# Array of datasets
DATASETS=(
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_counterfactual_reject_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_discussion_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_experimental_results_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_introduction_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_intro_discussion_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_methodology_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_related_work_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_discussion_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_experimental_results_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_introduction_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_intro_discussion_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_methodology_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_related_work_v6_test"
"iclr_2020_2025_85_5_10_split6_original_vision_binary_noreviews_counterfactual_reject_v6_test"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_critical_analysis_v6_test"
)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Select model based on dataset modality
if [[ "$DATASET" == *"vision"* ]]; then
    MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
    TEMPLATE="qwen2_vl"
else
    MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
    TEMPLATE="qwen"
fi

echo "Running inference on $DATASET with model $MODEL_PATH (Template: $TEMPLATE)"

python ablations/run_inference.py \
    --model_name_or_path $MODEL_PATH \
    --template $TEMPLATE \
    --dataset $DATASET \
    --dataset_dir data \
    --cutoff_len 20480 \
    --max_new_tokens 1024 \
    --save_name ablations/results/${DATASET}_predictions.jsonl \
    --pipeline_parallel_size 1

echo "Done"




