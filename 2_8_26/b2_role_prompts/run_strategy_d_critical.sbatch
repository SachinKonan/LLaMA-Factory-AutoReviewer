#!/bin/bash
#SBATCH --job-name=b2_strat_d_crit
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=2_8_26/logs/b2_role_prompts/strategy_d_critical_%A_%a.out
#SBATCH --error=2_8_26/logs/b2_role_prompts/strategy_d_critical_%A_%a.err
#SBATCH --comment="B2: Strategy D Critical - Critical Meta-Review Variant"

# ============================================================================
# B2: Strategy D Critical — Critical Meta-Review Variant
# ============================================================================
# Similar to Strategy D but with critical instructions:
# - Weigh whether weaknesses outweigh strengths
# - Reject when unclear
# - Only accept when strengths clearly outweigh weaknesses
#
# Prerequisites: Run run_role_inference.sbatch first (critical + enthusiastic).
#
# Array jobs:
#   0: clean
#   1: clean_images
#   2: vision
#
# Usage:
#   sbatch 2_8_26/b2_role_prompts/run_strategy_d_critical.sbatch
# ============================================================================

set -e

PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
source .vllm/bin/activate

mkdir -p 2_8_26/logs/b2_role_prompts

MODALITIES=("clean" "clean_images" "vision")
MODALITY="${MODALITIES[$SLURM_ARRAY_TASK_ID]}"

# Model selection
if [ "$MODALITY" == "clean" ]; then
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    TEMPLATE="qwen"
    VLLM_CONFIG=""
else
    MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    TEMPLATE="qwen2_vl"
    VLLM_CONFIG=""
fi

CUTOFF_LEN=20400
MAX_NEW_TOKENS=10240

# Input: critical and enthusiastic predictions
CRITICAL_PRED="2_8_26/b2_role_prompts/results/${MODALITY}/critical/predictions.jsonl"
ENTHUSIASTIC_PRED="2_8_26/b2_role_prompts/results/${MODALITY}/enthusiastic/predictions.jsonl"

# Output
OUTPUT_DIR="2_8_26/b2_role_prompts/results/${MODALITY}/strategy_d_critical"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "B2: Strategy D Critical - ${MODALITY}"
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "=============================================="

# Check prerequisites
if [ ! -f "${CRITICAL_PRED}" ]; then
    echo "ERROR: Critical predictions not found: ${CRITICAL_PRED}"
    echo "Please run run_role_inference.sbatch (critical role) first."
    exit 1
fi
if [ ! -f "${ENTHUSIASTIC_PRED}" ]; then
    echo "ERROR: Enthusiastic predictions not found: ${ENTHUSIASTIC_PRED}"
    echo "Please run run_role_inference.sbatch (enthusiastic role) first."
    exit 1
fi

# Step 1: Create Strategy D Critical dataset
echo "Creating Strategy D Critical dataset..."
python -c "
import json, os, sys
sys.path.insert(0, '2_8_26')
from shared.prompt_templates import STRATEGY_D_CRITICAL_SYSTEM_PROMPT, STRATEGY_D_CRITICAL_USER_TEMPLATE

critical = [json.loads(l) for l in open('${CRITICAL_PRED}')]
enthusiastic = [json.loads(l) for l in open('${ENTHUSIASTIC_PRED}')]

assert len(critical) == len(enthusiastic), f'Mismatch: {len(critical)} vs {len(enthusiastic)}'

dataset = []
for c, e in zip(critical, enthusiastic):
    c_review = c['predict'] if isinstance(c['predict'], str) else c['predict'][0]
    e_review = e['predict'] if isinstance(e['predict'], str) else e['predict'][0]

    entry = {
        'conversations': [
            {'from': 'system', 'value': STRATEGY_D_CRITICAL_SYSTEM_PROMPT},
            {'from': 'human', 'value': STRATEGY_D_CRITICAL_USER_TEMPLATE.format(
                critical_review=c_review, enthusiastic_review=e_review)},
            {'from': 'gpt', 'value': c['label']}  # Ground truth from original
        ]
    }
    dataset.append(entry)

# Save as dataset
out_dir = '${OUTPUT_DIR}/strategy_d_critical_dataset'
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'data.json'), 'w') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

# Create dataset_info.json
info = {
    'strategy_d_critical_dataset': {
        'file_name': 'strategy_d_critical_dataset/data.json',
        'formatting': 'sharegpt',
        'columns': {'messages': 'conversations'},
        'tags': {'role_tag': 'from', 'content_tag': 'value',
                 'user_tag': 'human', 'assistant_tag': 'gpt', 'system_tag': 'system'}
    }
}
with open(os.path.join('${OUTPUT_DIR}', 'dataset_info.json'), 'w') as f:
    json.dump(info, f, indent=2)

print(f'Created Strategy D Critical dataset: {len(dataset)} samples')
"

# Step 2: Run meta-review inference
echo "Running Strategy D Critical inference..."
INFERENCE_CMD="python inference_scaling/scripts/vllm_infer_ensemble.py \
    --model_name_or_path ${MODEL} \
    --dataset strategy_d_critical_dataset \
    --dataset_dir ${OUTPUT_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --save_name ${OUTPUT_DIR}/predictions.jsonl \
    --n_generations 1"

if [ -n "${VLLM_CONFIG}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --vllm_config '${VLLM_CONFIG}'"
fi

eval ${INFERENCE_CMD}

# Step 3: Extract results
python inference_scaling/scripts/extract_results.py \
    --input ${OUTPUT_DIR}/predictions.jsonl \
    --output ${OUTPUT_DIR}/results_single.jsonl \
    --strategy single

echo "=============================================="
echo "Strategy D Critical complete for ${MODALITY}"
echo "Results: ${OUTPUT_DIR}/results_single.jsonl"
echo "=============================================="
