#!/bin/bash
#SBATCH --job-name=b1_pdr
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=2_8_26/logs/b1_pdr/pdr_%A_%a.out
#SBATCH --error=2_8_26/logs/b1_pdr/pdr_%A_%a.err
#SBATCH --comment="B1: Parallel Distill Response (PDR) ensembling"

# ============================================================================
# B1: Parallel Distill Response (PDR)
# ============================================================================
# Generates 5 diverse reviews per paper, then uses a meta-reviewer to
# distill them into a single decision. Compared to simple majority voting,
# PDR allows the meta-reviewer to reason over the diverse perspectives.
#
# Reuses existing infrastructure:
#   - vllm_infer_ensemble.py with n_generations=5
#   - run_metareview.py for meta-review aggregation
#
# Array jobs:
#   0: clean (text-only)
#   1: clean_images (text+images)
#   2: vision (images-only)
#
# Usage:
#   sbatch 2_8_26/b1_pdr/run_pdr.sbatch
# ============================================================================

set -e

PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"
source .vllm/bin/activate

mkdir -p 2_8_26/logs/b1_pdr

# Modalities
MODALITIES=("clean" "clean_images" "vision")
MODALITY="${MODALITIES[$SLURM_ARRAY_TASK_ID]}"

# Use v7 split7 datasets — generate new_fewshot from v7 if needed
DATA_DIR="${PROJECT_DIR}/inference_scaling/data"
BASE_DATASET="iclr_2020_2025_85_5_10_split7_balanced_${MODALITY}_binary_noreviews_v7_test_new_fewshot"

# Fallback: if v7 new_fewshot dataset doesn't exist, use the v7 base dataset
# with n_generations=5 (the new_fewshot prompt variant is not strictly needed for PDR)
if [ ! -d "${DATA_DIR}/${BASE_DATASET}" ]; then
    echo "v7 new_fewshot dataset not found. Using v7 base dataset with JSON prompt..."
    BASE_DATASET="iclr_2020_2025_85_5_10_split7_balanced_${MODALITY}_binary_noreviews_v7_test_new"
    if [ ! -d "${DATA_DIR}/${BASE_DATASET}" ]; then
        # Use the B2 standard dataset (same model, same data, no role modifier)
        DATA_DIR="${PROJECT_DIR}/2_8_26/b2_role_prompts/data"
        BASE_DATASET="iclr_2020_2025_85_5_10_split7_balanced_${MODALITY}_v7_test_standard"
    fi
fi

# Model selection
if [ "$MODALITY" == "clean" ]; then
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    TEMPLATE="qwen"
    MEDIA_DIR=""
    ROPE_SCALING=""
else
    MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    TEMPLATE="qwen2_vl"
    MEDIA_DIR="."
    ROPE_SCALING=""
fi

# Output
OUTPUT_DIR="2_8_26/b1_pdr/results/${MODALITY}"
mkdir -p "${OUTPUT_DIR}"
PRED_FILE="${OUTPUT_DIR}/predictions.jsonl"

echo "=============================================="
echo "B1: PDR - ${MODALITY}"
echo "Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Model: ${MODEL}"
echo "Dataset: ${BASE_DATASET}"
echo "=============================================="

# Check if predictions already exist with sufficient samples (>100 lines)
EXISTING_PREDS="${PROJECT_DIR}/inference_scaling/results/${MODALITY}/new_fewshot/predictions.jsonl"
if [ -f "${EXISTING_PREDS}" ] && [ "$(wc -l < "${EXISTING_PREDS}")" -gt 100 ]; then
    echo "Using existing predictions from: ${EXISTING_PREDS} ($(wc -l < "${EXISTING_PREDS}") samples)"
    cp "${EXISTING_PREDS}" "${PRED_FILE}"
else
    echo "No existing predictions found. Running inference with n_generations=5..."

    # Check dataset
    if [ ! -d "${DATA_DIR}/${BASE_DATASET}" ]; then
        echo "ERROR: Dataset not found: ${DATA_DIR}/${BASE_DATASET}"
        exit 1
    fi

    INFERENCE_CMD="python inference_scaling/scripts/vllm_infer_ensemble.py \
        --model_name_or_path ${MODEL} \
        --dataset ${BASE_DATASET} \
        --dataset_dir ${DATA_DIR} \
        --template ${TEMPLATE} \
        --cutoff_len 20000 \
        --max_new_tokens 10240 \
        --save_name ${PRED_FILE} \
        --n_generations 5"

    if [ -n "${MEDIA_DIR}" ]; then
        INFERENCE_CMD="${INFERENCE_CMD} --media_dir ${MEDIA_DIR}"
    fi
    if [ -n "${ROPE_SCALING}" ]; then
        INFERENCE_CMD="${INFERENCE_CMD} --vllm_config '${ROPE_SCALING}'"
    fi

    eval ${INFERENCE_CMD}
fi

echo "Predictions ready. Extracting results..."

# Extract: single strategy (first of 5)
python inference_scaling/scripts/extract_results.py \
    --input ${PRED_FILE} \
    --output ${OUTPUT_DIR}/results_single.jsonl \
    --strategy single

# Extract: majority vote (5 votes)
python inference_scaling/scripts/extract_results.py \
    --input ${PRED_FILE} \
    --output ${OUTPUT_DIR}/results_majority.jsonl \
    --strategy majority

# Create meta-review dataset (PDR distillation step)
echo "Creating meta-review dataset..."
METAREVIEW_DATASET_DIR="${OUTPUT_DIR}/metareview_dataset"
METAREVIEW_DATA="${METAREVIEW_DATASET_DIR}/data.json"
mkdir -p ${METAREVIEW_DATASET_DIR}

python inference_scaling/scripts/run_metareview.py create \
    --input ${PRED_FILE} \
    --output ${METAREVIEW_DATA}

# Create dataset_info.json for LLaMA Factory
echo "Creating dataset_info.json..."
cat > ${OUTPUT_DIR}/dataset_info.json <<EOF
{
  "metareview_dataset": {
    "file_name": "metareview_dataset/data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "system_tag": "system"
    }
  }
}
EOF

# Run meta-review inference
echo "Running meta-review inference (PDR distillation)..."
METAREVIEW_PRED="${OUTPUT_DIR}/metareview_predictions.jsonl"

python inference_scaling/scripts/vllm_infer_ensemble.py \
    --model_name_or_path ${MODEL} \
    --dataset metareview_dataset \
    --dataset_dir ${OUTPUT_DIR} \
    --template ${TEMPLATE} \
    --cutoff_len 20000 \
    --max_new_tokens 4096 \
    --save_name ${METAREVIEW_PRED} \
    --n_generations 1

# Extract meta-review results
python inference_scaling/scripts/run_metareview.py extract \
    --input ${METAREVIEW_PRED} \
    --output ${OUTPUT_DIR}/results_metareview.jsonl

echo "=============================================="
echo "B1 PDR complete for ${MODALITY}"
echo "Results:"
echo "  Single:     ${OUTPUT_DIR}/results_single.jsonl"
echo "  Majority:   ${OUTPUT_DIR}/results_majority.jsonl"
echo "  Metareview: ${OUTPUT_DIR}/results_metareview.jsonl"
echo "=============================================="
