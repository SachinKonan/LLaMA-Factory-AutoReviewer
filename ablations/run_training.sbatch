#!/bin/bash
#SBATCH --job-name=ablation_train
#SBATCH --output=ablations/logs/training/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --comment="Ablation Training"

#SBATCH --array=0-13

source /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/.venv/bin/activate

# Create logs dir
mkdir -p ablations/logs

# Array of datasets
DATASETS=(
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_counterfactual_reject_v6_train"
# "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_discussion_v6_train"
# "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_experimental_results_v6_train"
# "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_introduction_v6_train"
# "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_intro_discussion_v6_train"
# "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_methodology_v6_train"
# "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_no_related_work_v6_train"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_discussion_v6_train"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_experimental_results_v6_train"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_introduction_v6_train"
# "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_intro_discussion_v6_train"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_methodology_v6_train"
"iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_only_related_work_v6_train"
"iclr_2020_2025_85_5_10_split6_original_vision_binary_noreviews_counterfactual_reject_v6_train"
)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR="ablations/checkpoints/${DATASET}"

# Select model and config based on dataset modality
if [[ "$DATASET" == *"vision"* ]]; then
    MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
    CONFIG_FILE="ablations/configs/qwen2_5_vl_3b_ablation.yaml"
else
    MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
    CONFIG_FILE="ablations/configs/qwen2_5_3b_ablation.yaml"
fi

echo "Starting training on $DATASET"
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

llamafactory-cli train $CONFIG_FILE \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR

echo "Done"
