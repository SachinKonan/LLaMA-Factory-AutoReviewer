#!/bin/bash
#SBATCH --job-name=textgrad-opt-v2-14b
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --output=textgrad_experiments_v2/logs/textgrad_%j.out
#SBATCH --error=textgrad_experiments_v2/logs/textgrad_%j.err

set -e

# Configuration
BASEDIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
WORKDIR="${BASEDIR}/textgrad_experiments_v2"
DATA_DIR="${BASEDIR}/data"
MAIN_VENV="${BASEDIR}/.venv_vllm_inf"           # Main venv with vLLM
TEXTGRAD_VENV="${WORKDIR}/.venv"        # TextGrad venv
INFERENCE_MODEL="${BASEDIR}/saves/qwen2.5-3b/full/sft_ds3/checkpoint-3750"
GRADIENT_MODEL="Qwen/Qwen3-14B"
INFERENCE_PORT=8000
GRADIENT_PORT=8001

cd "${WORKDIR}"

echo "=============================================="
echo "TextGrad Prompt Optimization"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "=============================================="
echo "Inference Model: ${INFERENCE_MODEL}"
echo "Gradient Model: ${GRADIENT_MODEL}"
echo "Main venv: ${MAIN_VENV}"
echo "TextGrad venv: ${TEXTGRAD_VENV}"
echo "=============================================="

# Create logs directory if needed
mkdir -p logs

# Start vLLM inference server (GPU 0) using MAIN venv (has vLLM)
echo "Starting inference server on port ${INFERENCE_PORT}..."
CUDA_VISIBLE_DEVICES=0 ${MAIN_VENV}/bin/python -m vllm.entrypoints.openai.api_server \
    --model "${INFERENCE_MODEL}" \
    --port ${INFERENCE_PORT} \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --trust-remote-code \
    2>&1 | tee logs/vllm_inference_${SLURM_JOB_ID}.log &
PID_INFERENCE=$!
echo "Inference server PID: ${PID_INFERENCE}"

# Start vLLM gradient server (GPU 1) using MAIN venv (has vLLM)
echo "Starting gradient server on port ${GRADIENT_PORT}..."
CUDA_VISIBLE_DEVICES=1 ${MAIN_VENV}/bin/python -m vllm.entrypoints.openai.api_server \
    --model "${GRADIENT_MODEL}" \
    --port ${GRADIENT_PORT} \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --trust-remote-code \
    2>&1 | tee logs/vllm_gradient_${SLURM_JOB_ID}.log &
PID_GRADIENT=$!
echo "Gradient server PID: ${PID_GRADIENT}"

# Wait for servers to start
echo "Waiting for servers to initialize (180s)..."
sleep 180

# Test server connectivity
echo "Testing server connectivity..."
for port in ${INFERENCE_PORT} ${GRADIENT_PORT}; do
    if curl -s "http://localhost:${port}/v1/models" > /dev/null; then
        echo "  Port ${port}: OK"
    else
        echo "  Port ${port}: FAILED"
        echo "  Checking if process is running..."
        ps aux | grep -E "vllm.*${port}" || true
        echo "  Server logs (last 50 lines):"
        tail -n 50 logs/vllm_*_${SLURM_JOB_ID}.log 2>/dev/null || true
    fi
done

# Activate TextGrad venv for optimization script
source ${TEXTGRAD_VENV}/bin/activate

# Run optimization
echo ""
echo "Starting optimization..."
echo "=============================================="

python optimize_prompt.py \
    --data_dir "${DATA_DIR}" \
    --train_dataset "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_train" \
    --val_dataset "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_validation" \
    --inference_model "${INFERENCE_MODEL}" \
    --gradient_model "${GRADIENT_MODEL}" \
    --inference_port ${INFERENCE_PORT} \
    --gradient_port ${GRADIENT_PORT} \
    --num_epochs 20 \
    --batch_size 16 \
    --eval_batch_size 50 \
    --save_dir results/

# Cleanup
echo ""
echo "=============================================="
echo "Cleaning up..."
kill ${PID_INFERENCE} ${PID_GRADIENT} 2>/dev/null || true

echo "Done!"
echo "=============================================="
