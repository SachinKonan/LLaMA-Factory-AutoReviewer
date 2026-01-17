#!/bin/bash
#SBATCH --job-name=textgrad-opt-qwen32b
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:5
#SBATCH --output=textgrad_experiments/logs/textgrad_%j.out
#SBATCH --error=textgrad_experiments/logs/textgrad_%j.err

set -e

# =========================
# Configuration
# =========================
BASEDIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
WORKDIR="${BASEDIR}/textgrad_experiments"
DATA_DIR="${BASEDIR}/data"

MAIN_VENV="${BASEDIR}/.venv_vllm_inf"     # vLLM env
TEXTGRAD_VENV="${WORKDIR}/.venv"          # TextGrad env

INFERENCE_MODEL="Qwen/Qwen2.5-3B-Instruct"
GRADIENT_MODEL="Qwen/Qwen3-32B"

INFERENCE_PORT=8000
GRADIENT_PORT=8001

# =========================
# Setup
# =========================
cd "${WORKDIR}"
mkdir -p logs

echo "=============================================="
echo "TextGrad Prompt Optimization"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "Inference Model: ${INFERENCE_MODEL}"
echo "Gradient Model: ${GRADIENT_MODEL}"
echo "=============================================="

# =========================
# Inference server (GPU 0)
# =========================
echo "Starting inference server on GPU 0 (port ${INFERENCE_PORT})..."

CUDA_VISIBLE_DEVICES=0 \
${MAIN_VENV}/bin/python -m vllm.entrypoints.openai.api_server \
    --model "${INFERENCE_MODEL}" \
    --port ${INFERENCE_PORT} \
    --gpu-memory-utilization 0.35 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --trust-remote-code \
    2>&1 | tee logs/vllm_inference_${SLURM_JOB_ID}.log &

PID_INFERENCE=$!
echo "Inference server PID: ${PID_INFERENCE}"

# Give inference time to fully initialize and claim GPU 0
sleep 60

# =========================
# Gradient server (GPUs 1,2 | TP=2)
# =========================
echo "Starting gradient server on GPUs 1,2 (port ${GRADIENT_PORT})..."

CUDA_VISIBLE_DEVICES=1,2,3,4 \
${MAIN_VENV}/bin/python -m vllm.entrypoints.openai.api_server \
    --model "${GRADIENT_MODEL}" \
    --port ${GRADIENT_PORT} \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --trust-remote-code \
    2>&1 | tee logs/vllm_gradient_${SLURM_JOB_ID}.log &

PID_GRADIENT=$!
echo "Gradient server PID: ${PID_GRADIENT}"

# =========================
# Wait for servers
# =========================
echo "Waiting for servers to initialize (180s)..."
sleep 180

# =========================
# Sanity check
# =========================
echo "Testing server connectivity..."
for port in ${INFERENCE_PORT} ${GRADIENT_PORT}; do
    if curl -s "http://localhost:${port}/v1/models" > /dev/null; then
        echo "  Port ${port}: OK"
    else
        echo "  Port ${port}: FAILED"
        echo "  Recent logs:"
        tail -n 50 logs/vllm_*_${SLURM_JOB_ID}.log || true
        exit 1
    fi
done

# =========================
# Run TextGrad
# =========================
source "${TEXTGRAD_VENV}/bin/activate"

echo "=============================================="
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

# =========================
# Cleanup
# =========================
echo "=============================================="
echo "Cleaning up..."
kill ${PID_INFERENCE} ${PID_GRADIENT} 2>/dev/null || true
echo "Done."
echo "=============================================="
