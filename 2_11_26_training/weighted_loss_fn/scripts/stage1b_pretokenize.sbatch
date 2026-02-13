#!/bin/bash
#SBATCH --job-name=pretokenize
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=2_11_26_training/weighted_loss_fn/logs/tokenize_%A_%a.out
#SBATCH --error=2_11_26_training/weighted_loss_fn/logs/tokenize_%A_%a.err

# ============================================================================
# Pre-tokenize all datasets before training.
#
# Array index mapping:
#   0: iclr_weighted_loss_train_1_2
#   1: iclr_weighted_loss_train_1_3
#   2: iclr_weighted_loss_train_1_4
#   3: iclr_weighted_loss_train_1_8
#
# Each job tokenizes one training dataset + the shared eval dataset and saves
# the result to disk.  The training sbatch then loads from tokenized_path
# instead of re-tokenizing on every rank.
#
# Usage:
#   sbatch 2_11_26_training/weighted_loss_fn/scripts/stage1b_pretokenize.sbatch
# ============================================================================

set -e

PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"

source .venv/bin/activate

mkdir -p 2_11_26_training/weighted_loss_fn/logs

ALL_PROPORTIONS=("1_2" "1_3" "1_4" "1_8")
PROPORTION=${ALL_PROPORTIONS[$SLURM_ARRAY_TASK_ID]}
DATASET="iclr_weighted_loss_train_${PROPORTION}"
TOKENIZED_PATH="2_11_26_training/weighted_loss_fn/tokenized/${DATASET}"

# Skip if already tokenized
if [ -d "${TOKENIZED_PATH}" ]; then
    echo "Tokenized data already exists at ${TOKENIZED_PATH}, skipping."
    exit 0
fi

mkdir -p 2_11_26_training/weighted_loss_fn/configs/generated
CONFIG_FILE="2_11_26_training/weighted_loss_fn/configs/generated/tokenize_${PROPORTION}.yaml"

cat > "${CONFIG_FILE}" << EOF
### model
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full

### dataset
dataset: ${DATASET}
dataset_dir: 2_11_26_training/weighted_loss_fn
template: qwen
cutoff_len: 16384
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/weighted_loss/tokenize_tmp_${PROPORTION}
overwrite_output_dir: true
report_to: none

### evaluation
do_eval: true
eval_dataset: iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test

### tokenized path (saves to disk and exits)
tokenized_path: ${TOKENIZED_PATH}

### train (minimal - just needed to satisfy config parsing)
per_device_train_batch_size: 1
num_train_epochs: 1.0
bf16: true
EOF

echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATASET}"
echo "Tokenized path: ${TOKENIZED_PATH}"
echo "Running on: $(hostname)"
echo "=============================================="

llamafactory-cli train "${CONFIG_FILE}"

echo "Pre-tokenization complete for ${DATASET}"
