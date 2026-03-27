#!/bin/bash
#SBATCH --job-name=pt_sft_test
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:4
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --output=logs/pretrain_then_sft/%A_test.out
#SBATCH --error=logs/pretrain_then_sft/%A_test.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

BASE_MODEL="/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
PT_LR="2e-6"
SHORT_NAME="test_run"

PT_DATASET="pretrain_iclr_nips_icml"
SFT_DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
TEMPLATE="qwen"
CUTOFF_LEN=24480

PT_MODEL_DIR="saves/pretrain_then_sft/${SHORT_NAME}/pt"
SFT_MODEL_DIR="saves/pretrain_then_sft/${SHORT_NAME}/sft"
MASTER_PORT=29710

echo "=============================================="
echo "TEST RUN: PT → SFT (10 steps each)"
echo "Base Model: ${BASE_MODEL}"
echo "PT Model Dir: ${PT_MODEL_DIR}"
echo "SFT Model Dir: ${SFT_MODEL_DIR}"
echo "=============================================="

mkdir -p logs/pretrain_then_sft "${PT_MODEL_DIR}" "${SFT_MODEL_DIR}"

# ======================
# PHASE 1: PRE-TRAINING (10 steps)
# ======================
echo ""
echo "=== PHASE 1: PRE-TRAINING (10 steps) ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_4gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py configs/pretrain_papers.yaml \
    model_name_or_path="${BASE_MODEL}" \
    dataset="${PT_DATASET}" \
    output_dir="${PT_MODEL_DIR}" \
    learning_rate="${PT_LR}" \
    max_steps=10 \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=1 \
    cutoff_len="${CUTOFF_LEN}" \
    save_strategy=steps \
    save_steps=10

echo ""
echo "=== Pre-training complete! ==="
echo ""

# Verify PT checkpoint exists
if [ ! -f "${PT_MODEL_DIR}/config.json" ] && [ ! -d "${PT_MODEL_DIR}/checkpoint-10" ]; then
    echo "ERROR: PT checkpoint not found in ${PT_MODEL_DIR}"
    ls -la "${PT_MODEL_DIR}/"
    exit 1
fi

# If saved as checkpoint-10 subdir, point to it
if [ -d "${PT_MODEL_DIR}/checkpoint-10" ]; then
    PT_CKPT="${PT_MODEL_DIR}/checkpoint-10"
else
    PT_CKPT="${PT_MODEL_DIR}"
fi

echo "PT checkpoint: ${PT_CKPT}"
ls "${PT_CKPT}/"

# ======================
# PHASE 2: SFT (10 steps, using PT checkpoint)
# ======================
echo ""
echo "=== PHASE 2: SFT (10 steps) ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_4gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py configs/final_sweep_v7_clean.yaml \
    model_name_or_path="${PT_CKPT}" \
    dataset="${SFT_DATASET}_train" \
    eval_dataset="${SFT_DATASET}_validation" \
    do_eval=false \
    output_dir="${SFT_MODEL_DIR}" \
    max_steps=10 \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=1 \
    cutoff_len="${CUTOFF_LEN}" \
    learning_rate="1e-6" \
    lr_scheduler_type=cosine \
    save_strategy=steps \
    save_steps=10

echo ""
echo "=============================================="
echo "TEST COMPLETED SUCCESSFULLY"
echo "PT checkpoint: ${PT_CKPT}"
echo "SFT output: ${SFT_MODEL_DIR}"
ls "${SFT_MODEL_DIR}/"
echo "=============================================="
