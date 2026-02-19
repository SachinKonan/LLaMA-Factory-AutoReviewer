
#!/bin/bash
#SBATCH --job-name=dsv3_text
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --array=0-5%4
#SBATCH --output=logs/final_sweep_v7_datasweepv3/text/%A_%a.out
#SBATCH --error=logs/final_sweep_v7_datasweepv3/text/%A_%a.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 6 balanced text datasets: 2 input representations x 3 year ranges
# Idx 4 uses _corrected (2020_2025 includes 2024, needs anonymized data)
CONFIGS=(
    "iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered bal_no2024_orig_text"
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_v7_filtered bal_no2024_w2026_orig_text"
    "iclr_2020_2025_85_5_10_balanced_original_text_corrected_v7_filtered bal_w2024_orig_text"
    "iclr_2020_2023_2025_85_5_10_balanced_pdftitleabs_clean_v7_filtered bal_no2024_clean"
    "iclr_2020_2023_2025_2026_85_5_10_balanced_pdftitleabs_clean_v7_filtered bal_no2024_w2026_clean"
    "iclr_2020_2025_85_5_10_balanced_pdftitleabs_clean_v7_filtered bal_w2024_clean"
)

read DATASET SHORT_NAME <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

CONFIG_FILE="configs/final_sweep_v7_clean.yaml"
TEMPLATE="qwen"
CUTOFF_LEN=24480

# 4 GPUs, per_device=4, grad_accum=4 → batch = 4*4*4 = 64
PER_DEVICE_BATCH=4
GRAD_ACCUM=4
NUM_GPUS=4

MODEL_DIR="saves/final_sweep_v7_datasweepv3/${SHORT_NAME}"
RESULTS_DIR="results/final_sweep_v7_datasweepv3/${SHORT_NAME}"
MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))

echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET}"
echo "Short Name: ${SHORT_NAME}"
echo "GPUs: ${NUM_GPUS}, per_device: ${PER_DEVICE_BATCH}, grad_accum: ${GRAD_ACCUM}"
echo "Effective batch: $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM))"
echo "Cutoff len: ${CUTOFF_LEN}"
echo "Config: ${CONFIG_FILE}"
echo "Model Dir: ${MODEL_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "=============================================="

mkdir -p logs/final_sweep_v7_datasweepv3/text "${MODEL_DIR}" "${RESULTS_DIR}"

# Write trainjob touch file
echo "${SLURM_JOB_ID}" > "${MODEL_DIR}/.trainjob.touch"

# ======================
# Start auto-inference watcher in background
# ======================
python scripts/auto_infer_watcher.py \
    --save_dir "${MODEL_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --dataset "${DATASET}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 1280 \
    --compute_train_accuracy \
    --delete_safetensors_posteval \
    --train_accuracy_max_samples 2000 \
    --train_accuracy_batch_size 2 \
    --save_logprobs \
    --temperature 0 \
    --poll_interval 60 &
AUTO_INFER_PID=$!
echo "Started auto-inference watcher (PID: ${AUTO_INFER_PID})"

# ======================
# TRAINING
# ======================
echo ""
echo "=== TRAINING ==="
echo ""

accelerate launch \
    --config_file configs/fsdp2_4gpu_config.yaml \
    --main_process_port "${MASTER_PORT}" \
    src/train.py "${CONFIG_FILE}" \
    dataset="${DATASET}_train" \
    eval_dataset="${DATASET}_validation" \
    do_eval=false \
    output_dir="${MODEL_DIR}" \
    per_device_train_batch_size="${PER_DEVICE_BATCH}" \
    gradient_accumulation_steps="${GRAD_ACCUM}" \
    cutoff_len="${CUTOFF_LEN}" \
    learning_rate=4e-6 \
    num_train_epochs=9.0

echo ""
echo "=== Training complete! ==="
echo ""

# Wait for watcher to finish processing final checkpoint
sleep 120
kill $AUTO_INFER_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
