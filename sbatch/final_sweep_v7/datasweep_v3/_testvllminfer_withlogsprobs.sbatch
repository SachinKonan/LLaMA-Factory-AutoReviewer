#!/bin/bash
#SBATCH --job-name=test_logprobs
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --output=logs/final_sweep_v7_datasweepv3/test_logprobs_%j.out
#SBATCH --error=logs/final_sweep_v7_datasweepv3/test_logprobs_%j.err

# Quick test: verify --save_logprobs works with vllm_infer.py
# Uses an existing checkpoint and a small sample to validate output format.

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

CKPT_DIR="saves/final_sweep_v7/balanced_clean/checkpoint-5345"
DATASET="iclr_2020_2025_85_5_10_balanced_pdftitleabs_clean_v7_filtered_test"
TEMPLATE="qwen"
CUTOFF_LEN=24480
RESULTS_DIR="results/final_sweep_v7_datasweepv3/_test_logprobs"

mkdir -p logs/final_sweep_v7_datasweepv3 "${RESULTS_DIR}"

echo "=============================================="
echo "Test: vllm_infer.py --save_logprobs"
echo "Checkpoint: ${CKPT_DIR}"
echo "Dataset: ${DATASET}"
echo "=============================================="

# Run with --save_logprobs and --temperature 0, small sample
python scripts/vllm_infer.py \
    --model_name_or_path "${CKPT_DIR}" \
    --dataset "${DATASET}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 1280 \
    --max_samples 20 \
    --temperature 0 \
    --save_logprobs \
    --save_name "${RESULTS_DIR}/test_logprobs.jsonl"

echo ""
echo "=== Validating output ==="

# Check that the output file exists and has token_logprobs
if [ ! -f "${RESULTS_DIR}/test_logprobs.jsonl" ]; then
    echo "FAIL: Output file not created"
    exit 1
fi

LINE_COUNT=$(wc -l < "${RESULTS_DIR}/test_logprobs.jsonl")
echo "Lines in output: ${LINE_COUNT}"

HAS_LOGPROBS=$(python -c "
import json
with open('${RESULTS_DIR}/test_logprobs.jsonl') as f:
    lines = [json.loads(l) for l in f]
has_field = all('token_logprobs' in l for l in lines)
has_values = any(len(l['token_logprobs']) > 0 for l in lines)
print(f'All entries have token_logprobs field: {has_field}')
print(f'At least one entry has non-empty logprobs: {has_values}')
print(f'Sample logprobs (first entry, first 5): {lines[0][\"token_logprobs\"][:5]}')
if has_field and has_values:
    print('PASS')
else:
    print('FAIL')
")
echo "${HAS_LOGPROBS}"

echo ""
echo "=============================================="
echo "Test complete. Output at: ${RESULTS_DIR}/test_logprobs.jsonl"
echo "=============================================="
