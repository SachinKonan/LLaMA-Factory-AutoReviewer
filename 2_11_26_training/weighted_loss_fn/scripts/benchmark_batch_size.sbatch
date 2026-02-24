#!/bin/bash
#SBATCH --job-name=benchmark_bs
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=2_11_26_training/weighted_loss_fn/logs/benchmark_bs_%j.out
#SBATCH --error=2_11_26_training/weighted_loss_fn/logs/benchmark_bs_%j.err

set -e

# Navigate to project directory
PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"

# Activate training environment
source .venv/bin/activate

export HF_HOME="/n/fs/vision-mix/jl0796/.hf_home"

# Use a small subset of raw data for benchmarking if tokenized is not ready
# But here we will try to use the raw data directly to avoid waiting for pre-tokenization
DATASET="iclr_weighted_loss_train_1_2" 
# Note: LlamaFactory can tokenize on the fly if tokenized_path is invalid or ignored.
# We will use a temporary config for benchmarking

mkdir -p 2_11_26_training/weighted_loss_fn/configs/benchmark
mkdir -p saves/benchmark

# Function to run benchmark
run_benchmark() {
    BS=$1
    echo "----------------------------------------------------------------"
    echo "Benchmarking per-device batch size: ${BS}"
    echo "----------------------------------------------------------------"
    
    CONFIG_FILE="2_11_26_training/weighted_loss_fn/configs/benchmark/bs_${BS}.yaml"
    
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
gradient_checkpointing: true
gradient_checkpointing_kwargs: {"use_reentrant": false}

### dataset
dataset: ${DATASET}
dataset_dir: 2_11_26_training/weighted_loss_fn
template: qwen
cutoff_len: 16384
preprocessing_num_workers: 4
dataloader_num_workers: 2

### output
output_dir: saves/benchmark/bs_${BS}
logging_steps: 1
save_steps: 1000
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: ${BS}
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
max_steps: 5  # Run only 5 steps

### weighted loss (disable for benchmark)
use_weighted_loss: false
EOF

    echo "Running with config: ${CONFIG_FILE}"
    # We use || true to prevent script exit on OOM
    llamafactory-cli train "${CONFIG_FILE}" && echo "SUCCESS: Batch size ${BS} passed" || echo "FAILURE: Batch size ${BS} failed"
}

# Test increasing batch sizes
# 16k context is heavy. 
# BS=1 is known safe.
# Try 2, 4, 8, 16.

for bs in 2 4 8 16; do
    run_benchmark $bs
    echo ""
    sleep 5
done

echo "Benchmark complete. Check logs for failures (OOM)."
