#!/bin/bash
#SBATCH --job-name=train_weighted_loss
#SBATCH --time=14:30:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --array=0-35
#SBATCH --output=2_11_26_training/weighted_loss_fn/logs/train_%A_%a.out
#SBATCH --error=2_11_26_training/weighted_loss_fn/logs/train_%A_%a.err

# ============================================================================
# Weighted Loss Function Training
# ============================================================================
#
# Job Array: 0-35 (36 total experiments)
# Use --array=X,Y,Z to select specific experiments to run.
#
# Experiment Matrix:
#   0-3:   Standard BCE baselines (gamma=1.0) for each proportion
#   4-35:  Weighted loss: 4 proportions × 4 gammas × 2 variants = 32 experiments
#
# Array index mapping:
#   0:  BCE baseline, proportion 1/2
#   1:  BCE baseline, proportion 1/3
#   2:  BCE baseline, proportion 1/4
#   3:  BCE baseline, proportion 1/8
#   4-11:  Proportion 1/2, gammas [2,3,4,8] × variants [accept,reject]
#   12-19: Proportion 1/3, gammas [2,3,4,8] × variants [accept,reject]
#   20-27: Proportion 1/4, gammas [2,3,4,8] × variants [accept,reject]
#   28-35: Proportion 1/8, gammas [2,3,4,8] × variants [accept,reject]
#
# Within each 8-index block (e.g. 4-11):
#   +0: accept gamma=2  +1: reject gamma=2
#   +2: accept gamma=3  +3: reject gamma=3
#   +4: accept gamma=4  +5: reject gamma=4
#   +6: accept gamma=8  +7: reject gamma=8
# ============================================================================

set -e

# Navigate to project directory
PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"

# Activate training environment
source .venv/bin/activate

# Set HuggingFace cache
export HF_HOME="/n/fs/vision-mix/jl0796/.hf_home"

# Increase NCCL heartbeat timeout to allow slow tokenization on non-main ranks
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

# Create logs directory
mkdir -p 2_11_26_training/weighted_loss_fn/logs

# Decode array index to experiment configuration
TASK_ID=${SLURM_ARRAY_TASK_ID}

ALL_PROPORTIONS=("1_2" "1_3" "1_4" "1_8")
GAMMAS=(2.0 3.0 4.0 8.0)
VARIANTS=("accept" "reject")

# Indices 0-3: Standard BCE baselines (no weighted loss)
if [ $TASK_ID -le 3 ]; then
    PROPORTION=${ALL_PROPORTIONS[$TASK_ID]}
    VARIANT="baseline"
    GAMMA=1.0

# Indices 4-35: Weighted loss experiments
else
    WEIGHTED_IDX=$((TASK_ID - 4))  # Offset to 0-31

    # 8 experiments per proportion (4 gammas × 2 variants)
    PROP_IDX=$((WEIGHTED_IDX / 8))
    LOCAL_IDX=$((WEIGHTED_IDX % 8))
    GAMMA_IDX=$((LOCAL_IDX / 2))
    VARIANT_IDX=$((LOCAL_IDX % 2))

    PROPORTION=${ALL_PROPORTIONS[$PROP_IDX]}
    GAMMA=${GAMMAS[$GAMMA_IDX]}
    VARIANT=${VARIANTS[$VARIANT_IDX]}
fi

# Construct dataset name and tokenized path
DATASET="iclr_weighted_loss_train_${PROPORTION}"
TOKENIZED_PATH="2_11_26_training/weighted_loss_fn/tokenized/${DATASET}"

# Construct output directory
OUTPUT_DIR="saves/weighted_loss/${VARIANT}_gamma${GAMMA}_prop${PROPORTION}"

# Construct config file path (will be generated dynamically)
CONFIG_FILE="2_11_26_training/weighted_loss_fn/configs/generated/${VARIANT}_gamma${GAMMA}_prop${PROPORTION}.yaml"

# Create config directory
mkdir -p 2_11_26_training/weighted_loss_fn/configs/generated

# Generate config file from base
cat > "${CONFIG_FILE}" << EOF
### model
model_name_or_path: /n/fs/vision-mix/jl0796/.hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
trust_remote_code: true
flash_attn: fa2
enable_liger_kernel: True

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json
gradient_checkpointing: true
gradient_checkpointing_kwargs: {"use_reentrant": false}

### dataset
dataset: ${DATASET}
dataset_dir: 2_11_26_training/weighted_loss_fn
template: qwen
cutoff_len: 16384
preprocessing_num_workers: 16
dataloader_num_workers: 4
tokenized_path: ${TOKENIZED_PATH}

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 10
save_steps: 134
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-5
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### evaluation
do_eval: true
eval_dataset: iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test
eval_strategy: steps
eval_steps: 501
per_device_eval_batch_size: 2

### weighted loss
# For baseline (index 0), use standard BCE (can set use_weighted_loss=false or gamma=1.0)
use_weighted_loss: $([ "$GAMMA" == "1.0" ] && echo "false" || echo "true")
weighted_loss_variant: ${VARIANT}
weighted_loss_gamma: ${GAMMA}
EOF

echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Variant: ${VARIANT}"
echo "Gamma: ${GAMMA}"
echo "Proportion: ${PROPORTION}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "Tokenized: ${TOKENIZED_PATH}"
echo "Config: ${CONFIG_FILE}"
echo "Running on: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
if [ "$TASK_ID" -le 3 ]; then
    echo "**BASELINE EXPERIMENT** (Standard BCE, no weighted loss)"
fi
echo "=============================================="

# Run training with llamafactory-cli
llamafactory-cli train "${CONFIG_FILE}"

echo "Training complete for ${VARIANT}_gamma${GAMMA}_prop${PROPORTION}"
