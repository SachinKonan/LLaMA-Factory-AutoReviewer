#!/bin/bash
#SBATCH --job-name=test_lora
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80"
#SBATCH --partition=gpu-test
#SBATCH --output=logs/testing/test_lora_e2e_%j.out
#SBATCH --error=logs/testing/test_lora_e2e_%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

BASE_MODEL="/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
SAVE_DIR="saves/testing/lora_e2e_test"
RESULTS_DIR="results/testing"

mkdir -p "${SAVE_DIR}" "${RESULTS_DIR}" logs/testing

echo "=============================================="
echo "LoRA E2E Test"
echo "Job: $SLURM_JOB_ID"
echo "=============================================="

# =============================================
# PHASE 1: Train 1 step of LoRA → save checkpoint
# =============================================
echo ""
echo "=== PHASE 1: LoRA Training (1 step) ==="
echo ""

python src/train.py configs/final_sweep_v7_clean.yaml \
    finetuning_type=lora \
    lora_rank=64 \
    lora_alpha=64 \
    lora_target=all \
    lora_dropout=0.0 \
    model_name_or_path="${BASE_MODEL}" \
    dataset="${DATASET}_train" \
    eval_dataset="${DATASET}_validation" \
    do_eval=false \
    output_dir="${SAVE_DIR}" \
    max_steps=1 \
    save_strategy=steps \
    save_steps=1 \
    save_total_limit=1 \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=1 \
    learning_rate=2e-4

echo ""
echo "=== PHASE 1 Complete ==="
echo ""

# Verify checkpoint contents
echo "=== Checkpoint contents ==="
CKPT_DIR=$(ls -d ${SAVE_DIR}/checkpoint-* | head -1)
echo "Checkpoint dir: ${CKPT_DIR}"
ls -la "${CKPT_DIR}/"
echo ""

# Verify it's a LoRA checkpoint (adapter files, not model files)
if [ -f "${CKPT_DIR}/adapter_config.json" ]; then
    echo "OK: adapter_config.json found (LoRA checkpoint)"
    cat "${CKPT_DIR}/adapter_config.json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'base_model: {d[\"base_model_name_or_path\"]}')"
else
    echo "FAIL: adapter_config.json NOT found"
    exit 1
fi

if ls "${CKPT_DIR}"/adapter_model*.safetensors 1>/dev/null 2>&1; then
    echo "OK: adapter_model.safetensors found"
    ls -lh "${CKPT_DIR}"/adapter_model*.safetensors
else
    echo "FAIL: adapter_model.safetensors NOT found"
    exit 1
fi

if ls "${CKPT_DIR}"/model-*.safetensors 1>/dev/null 2>&1; then
    echo "WARN: model-*.safetensors found (unexpected for LoRA)"
else
    echo "OK: no model-*.safetensors (correct for LoRA - only adapter saved)"
fi

# =============================================
# PHASE 2: vLLM inference with LoRA adapter
# =============================================
echo ""
echo "=== PHASE 2: vLLM Inference with LoRA adapter ==="
echo ""

python scripts/vllm_infer.py \
    --model_name_or_path "${BASE_MODEL}" \
    --adapter_name_or_path "${CKPT_DIR}" \
    --dataset "${DATASET}_test" \
    --template qwen \
    --cutoff_len 24480 \
    --max_new_tokens 1280 \
    --max_samples 10 \
    --vllm_config '{"max_lora_rank": 64}' \
    --save_name "${RESULTS_DIR}/lora_e2e_test.jsonl"

echo ""
echo "=== PHASE 2 Complete ==="
echo ""

# Verify inference output
if [ -f "${RESULTS_DIR}/lora_e2e_test.jsonl" ]; then
    LINE_COUNT=$(wc -l < "${RESULTS_DIR}/lora_e2e_test.jsonl")
    echo "OK: ${RESULTS_DIR}/lora_e2e_test.jsonl exists (${LINE_COUNT} lines)"
    echo "First prediction:"
    head -1 "${RESULTS_DIR}/lora_e2e_test.jsonl" | python3 -c "import sys,json; d=json.loads(sys.stdin.readline()); print(d.get('predict','')[:200])"
else
    echo "FAIL: inference output not found"
    exit 1
fi

echo ""
echo "=============================================="
echo "LoRA E2E Test PASSED"
echo "=============================================="
