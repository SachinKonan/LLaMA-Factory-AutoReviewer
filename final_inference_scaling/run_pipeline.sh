#!/bin/bash
# ============================================================================
# Final Inference Scaling Pipeline
# ============================================================================
# This script orchestrates the full inference scaling experiment pipeline.
#
# Steps:
# 1. Generate modified datasets
# 2. Run inference on all configurations
# 3. Run meta-review inference for ensembled predictions
# 4. Extract results using different strategies
# 5. Compute metrics and generate plots
#
# Usage:
#   ./run_pipeline.sh [step]
#
# Steps:
#   all        - Run entire pipeline
#   generate   - Generate datasets only
#   inference  - Submit inference jobs only
#   metareview - Submit meta-review jobs only
#   extract    - Extract results only
#   metrics    - Compute metrics and plots only
# ============================================================================

set -e

PROJECT_DIR="/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"

# ============================================================================
# Configuration: Set LIMIT to control dataset size
# ============================================================================
# Set to 2 for testing, or 4000 (larger than all datasets) for full runs
LIMIT=4000

# Directory paths
DATA_DIR="./final_inference_scaling/data"
RESULTS_DIR="./final_inference_scaling/results"
METRICS_DIR="./final_inference_scaling/metrics"
LOGS_DIR="./logs/final_inference_scaling"

# Create necessary directories
mkdir -p ${LOGS_DIR}
mkdir -p ${RESULTS_DIR}
mkdir -p ${METRICS_DIR}

STEP="${1:-all}"

# ============================================================================
# Step 1: Generate Datasets
# ============================================================================
generate_datasets() {
    echo "=============================================="
    echo "Step 1: Generating modified datasets"
    echo "Limit: ${LIMIT} samples per dataset"
    echo "=============================================="

    source .venv/bin/activate

    python final_inference_scaling/scripts/generate_datasets.py \
        --base_data_dir "/scratch/gpfs/ZHUANGL/jl0796/shared/data" \
        --output_dir "${DATA_DIR}" \
        --splits test \
        --limit ${LIMIT}

    echo "Dataset generation complete!"
}

# ============================================================================
# Step 2: Submit Inference Jobs
# ============================================================================
submit_inference() {
    echo "=============================================="
    echo "Step 2: Submitting inference jobs"
    echo "=============================================="

    # Submit main inference job array
    INFERENCE_JOB=$(sbatch --parsable final_inference_scaling/sbatch/run_inference.sbatch)
    echo "Submitted inference job array: ${INFERENCE_JOB}"

    echo "Monitor with: squeue -u \$USER"
    echo "Inference job ID: ${INFERENCE_JOB}"
}

# ============================================================================
# Step 3: Submit Meta-Review Jobs
# ============================================================================
submit_metareview() {
    echo "=============================================="
    echo "Step 3: Submitting meta-review jobs"
    echo "=============================================="

    echo "WARNING: run_metareview.sbatch needs to be updated to match the final pipeline structure."
    echo "Please ensure the input/output paths in the sbatch target the 5-generation runs correctly."
    
    METAREVIEW_JOB=$(sbatch --parsable final_inference_scaling/sbatch/run_metareview.sbatch)
    echo "Submitted meta-review job array: ${METAREVIEW_JOB}"
}

# ============================================================================
# Step 4: Extract Results
# ============================================================================
extract_results_for_dir() {
    local BASE_RESULTS_DIR="$1"
    local DIR_NAME="$2"

    echo "Processing results in ${DIR_NAME}..."
    
    # We dynamically find all prediction.jsonl files
    find "${BASE_RESULTS_DIR}" -name "predictions.jsonl" | while read pred_file; do
        eval_dir=$(dirname "${pred_file}")
        
        # Check if predictions are non-empty
        non_empty=$(python3 -c "
import json
count = 0
with open('${pred_file}') as f:
    for line in f:
        if line.strip():
            count += 1
            break
print(count)
")
        if [ "${non_empty}" == "0" ]; then
            echo "  Skipping ${eval_dir}: all predictions are empty"
            continue
        fi

        echo "  Processing ${eval_dir}..."

        # Single strategy
        python final_inference_scaling/scripts/extract_results.py \
            --input "${pred_file}" \
            --output "${eval_dir}/results_single.jsonl" \
            --strategy single

        # Calibrated strategy (using overall score threshold)
        python final_inference_scaling/scripts/extract_results.py \
            --input "${pred_file}" \
            --output "${eval_dir}/results_calibrated.jsonl" \
            --strategy single \
            --use_calibration \
            --threshold 6

        # Check if this represents a 5-generation run or pdr run
        if [[ "${eval_dir}" == *"_gen5"* ]] || [[ "${eval_dir}" == *"pdr"* ]]; then
            python final_inference_scaling/scripts/extract_results.py \
                --input "${pred_file}" \
                --output "${eval_dir}/results_majority.jsonl" \
                --strategy majority

            # Meta-review results (if available)
            meta_pred="${eval_dir}/metareview_predictions.jsonl"
            if [ -f "${meta_pred}" ]; then
                python final_inference_scaling/scripts/run_metareview.py extract \
                    --input "${meta_pred}" \
                    --output "${eval_dir}/results_metareview.jsonl"
            fi
        fi
    done
}

extract_results() {
    echo "=============================================="
    echo "Step 4: Extracting results"
    echo "=============================================="

    source .venv/bin/activate

    extract_results_for_dir "${RESULTS_DIR}" "Final Results"

    echo "Result extraction complete!"
}

# ============================================================================
# Step 5: Compute Metrics
# ============================================================================
compute_metrics() {
    echo "=============================================="
    echo "Step 5: Computing metrics and generating plots"
    echo "=============================================="

    source .venv/bin/activate

    python final_inference_scaling/scripts/compute_metrics.py \
        --results_dir "${RESULTS_DIR}" \
        --output_dir "${METRICS_DIR}" \
        --base_data_dir "/scratch/gpfs/ZHUANGL/jl0796/shared/data"

    echo "Metrics computation complete!"
    echo "Results saved to: ${METRICS_DIR}"
}

# ============================================================================
# Main
# ============================================================================
case "${STEP}" in
    all)
        generate_datasets
        submit_inference
        echo ""
        echo "Inference jobs submitted. After they complete, run:"
        echo "  ./run_pipeline.sh metareview  # Submit meta-review jobs"
        echo "  ./run_pipeline.sh extract     # Extract results"
        echo "  ./run_pipeline.sh metrics     # Compute metrics"
        ;;
    generate)
        generate_datasets
        ;;
    inference)
        submit_inference
        ;;
    metareview)
        submit_metareview
        ;;
    extract)
        extract_results
        ;;
    metrics)
        compute_metrics
        ;;
    *)
        echo "Unknown step: ${STEP}"
        echo "Usage: ./run_pipeline.sh [all|generate|inference|metareview|extract|metrics]"
        exit 1
        ;;
esac

echo ""
echo "Done!"
