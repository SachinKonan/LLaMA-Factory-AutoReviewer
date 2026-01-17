#!/bin/bash
#SBATCH --job-name=vllm-textgrad
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --output=logs/grid_searchv2/infer_textgrad_%A_%a.out
#SBATCH --error=logs/grid_searchv2/infer_textgrad_%A_%a.err

set -e
cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer
source .venv_vllm_inf/bin/activate
export TRANSFORMERS_OFFLINE=1

# TextGrad optimized datasets
# Array indices:
# 0: validation_textgrad_8b
# 1: test_textgrad_8b
# 2: validation_textgrad_14b
# 3: test_textgrad_14b
# 4: validation_textgrad_32b
# 5: test_textgrad_32b

DATASETS=(
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_validation_textgrad_8b"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_test_textgrad_8b"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_validation_textgrad_14b"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_test_textgrad_14b"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_validation_textgrad_32b"
    "iclr_2020_2025_85_5_10_split6_original_clean_binary_noreviews_v6_test_textgrad_32b"
)

DATASET_BASE="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
TEST_DATASET="${DATASET_BASE}_test"
BASE_MODEL="saves/qwen2.5-3b/full/sft_ds3/checkpoint-3750"
RESULTS_DIR="results/grid_searchv2/textgrad_prompts"

# Determine output name based on dataset
if [[ "$DATASET_BASE" == *"textgrad_8b"* ]]; then
    PROMPT_TYPE="8b"
elif [[ "$DATASET_BASE" == *"textgrad_14b"* ]]; then
    PROMPT_TYPE="14b"
else
    PROMPT_TYPE="32b"
fi

if [[ "$DATASET_BASE" == *"validation"* ]]; then
    SPLIT="validation"
else
    SPLIT="test"
fi

OUTPUT_FILE="${RESULTS_DIR}/ckpt3750_textgrad_${PROMPT_TYPE}_${SPLIT}.jsonl"

echo "=============================================="
echo "TextGrad Optimized Prompt Inference"
echo "=============================================="
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: ${DATASET_BASE}"
echo "Model: ${BASE_MODEL}"
echo "Prompt variant: ${PROMPT_TYPE}"
echo "Split: ${SPLIT}"
echo "Output: ${OUTPUT_FILE}"
echo "=============================================="

mkdir -p "${RESULTS_DIR}"

python scripts/vllm_infer.py \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset "${TEST_DATASET}" \
    --template qwen \
    --cutoff_len 16384 \
    --max_new_tokens 10240 \
    --save_name "${OUTPUT_FILE}"

echo ""
echo "=============================================="
echo "Inference complete!"
echo "=============================================="

# Quick inline analysis
echo ""
echo "Running quick analysis..."
python -c "
import json
import re

def extract_boxed(text):
    if text is None:
        return None
    match = re.search(r'\\\\boxed\{([^}]+)\}', text)
    return match.group(1).strip().lower() if match else None

def normalize(label):
    if label is None:
        return None
    l = label.lower()
    if l in ['accept', 'accepted']:
        return 'accepted'
    if l in ['reject', 'rejected']:
        return 'rejected'
    return None

with open('${OUTPUT_FILE}') as f:
    data = [json.loads(line) for line in f]

preds = []
labels = []
for item in data:
    pred = normalize(extract_boxed(item.get('predict', '')))
    label = normalize(extract_boxed(item.get('label', '')))
    preds.append(pred)
    labels.append(label)

valid = [(p, l) for p, l in zip(preds, labels) if p is not None and l is not None]
valid_rate = len(valid) / len(data) if data else 0
correct = sum(1 for p, l in valid if p == l)
accuracy = correct / len(valid) if valid else 0

print(f'Results for: ${OUTPUT_FILE}')
print(f'  Total samples: {len(data)}')
print(f'  Valid predictions: {len(valid)} ({valid_rate:.1%})')
print(f'  Correct: {correct}')
print(f'  Accuracy: {accuracy:.1%}')
"

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
