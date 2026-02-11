#!/bin/bash
#SBATCH --job-name=b2_strategy_d
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=2_8_26/logs/b2_role_prompts/strategy_d_%A_%a.out
#SBATCH --error=2_8_26/logs/b2_role_prompts/strategy_d_%A_%a.err
#SBATCH --comment="B2: Strategy D - Multi-Perspective Meta-Review"

# ============================================================================
# B2: Strategy D — Multi-Perspective Meta-Review
# ============================================================================
# Pipeline: Takes critical + enthusiastic review → meta-reviewer synthesizes.
# This explicitly separates concern-finding from strength-finding to reduce
# the "halo effect" of well-presented papers.
#
# Prerequisites: Run run_role_inference.sbatch first (critical + enthusiastic).
#
# Array jobs:
#   0: clean
#   1: clean_images
#   2: vision
#
# Usage:
#   sbatch 2_8_26/b2_role_prompts/run_strategy_d.sbatch
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
    ROPE_SCALING='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
else
    MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    TEMPLATE="qwen2_vl"
    ROPE_SCALING=""
fi

# Input: critical and enthusiastic predictions
CRITICAL_PRED="2_8_26/b2_role_prompts/results/${MODALITY}/critical/predictions.jsonl"
ENTHUSIASTIC_PRED="2_8_26/b2_role_prompts/results/${MODALITY}/enthusiastic/predictions.jsonl"

# Output
OUTPUT_DIR="2_8_26/b2_role_prompts/results/${MODALITY}/strategy_d"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "B2: Strategy D - ${MODALITY}"
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

# Step 1: Create Strategy D meta-review dataset
# Pairs each paper's critical + enthusiastic reviews for the meta-reviewer
echo "Creating Strategy D dataset..."
python -c "
import json, os, sys
sys.path.insert(0, '2_8_26')
from shared.prompt_templates import STRATEGY_D_SYSTEM_PROMPT, STRATEGY_D_USER_TEMPLATE

critical = [json.loads(l) for l in open('${CRITICAL_PRED}')]
enthusiastic = [json.loads(l) for l in open('${ENTHUSIASTIC_PRED}')]

assert len(critical) == len(enthusiastic), f'Mismatch: {len(critical)} vs {len(enthusiastic)}'

dataset = []
for c, e in zip(critical, enthusiastic):
    c_review = c['predict'] if isinstance(c['predict'], str) else c['predict'][0]
    e_review = e['predict'] if isinstance(e['predict'], str) else e['predict'][0]

    entry = {
        'conversations': [
            {'from': 'system', 'value': STRATEGY_D_SYSTEM_PROMPT},
            {'from': 'human', 'value': STRATEGY_D_USER_TEMPLATE.format(
                critical_review=c_review, enthusiastic_review=e_review)},
            {'from': 'gpt', 'value': c['label']}  # Ground truth from original
        ]
    }
    dataset.append(entry)

# Save as dataset
out_dir = '${OUTPUT_DIR}/strategy_d_dataset'
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'data.json'), 'w') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

# Create dataset_info.json
info = {
    'strategy_d_dataset': {
        'file_name': 'strategy_d_dataset/data.json',
        'formatting': 'sharegpt',
        'columns': {'messages': 'conversations'},
        'tags': {'role_tag': 'from', 'content_tag': 'value',
                 'user_tag': 'human', 'assistant_tag': 'gpt', 'system_tag': 'system'}
    }
}
with open(os.path.join('${OUTPUT_DIR}', 'dataset_info.json'), 'w') as f:
    json.dump(info, f, indent=2)

print(f'Created Strategy D dataset: {len(dataset)} samples')
"

# Step 2: Run meta-review inference
echo "Running Strategy D meta-review inference..."
INFERENCE_CMD="python inference_scaling/scripts/vllm_infer_ensemble.py \
    --model_name_or_path ${MODEL} \
    --dataset strategy_d_dataset \
    --dataset_dir ${OUTPUT_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len 50000 \
    --max_new_tokens 4096 \
    --save_name ${OUTPUT_DIR}/predictions.jsonl \
    --n_generations 1"

if [ -n "${ROPE_SCALING}" ]; then
    INFERENCE_CMD="${INFERENCE_CMD} --vllm_config '${ROPE_SCALING}'"
fi

eval ${INFERENCE_CMD}

# Step 3: Extract results
python inference_scaling/scripts/extract_results.py \
    --input ${OUTPUT_DIR}/predictions.jsonl \
    --output ${OUTPUT_DIR}/results_single.jsonl \
    --strategy single

echo "=============================================="
echo "Strategy D complete for ${MODALITY}"
echo "Results: ${OUTPUT_DIR}/results_single.jsonl"
echo "=============================================="
