#!/bin/bash
# ============================================================================
# Inference Scaling Pipeline
# ============================================================================
# This script orchestrates the full inference scaling experiment pipeline.
#
# Steps:
# 1. Generate modified datasets (original, new, new_fewshot prompts)
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

PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"

# ============================================================================
# Configuration: Set LIMIT to control dataset size
# ============================================================================
# Set to 2 for testing, or 4000 (larger than all datasets) for full runs
LIMIT=4000  # Change to 2 for testing

# Create necessary directories
mkdir -p logs/inference_scaling
mkdir -p inference_scaling/results
mkdir -p inference_scaling/metrics

STEP="${1:-all}"

# ============================================================================
# Step 1: Generate Datasets
# ============================================================================
generate_datasets() {
    echo "=============================================="
    echo "Step 1: Generating modified datasets"
    echo "Limit: ${LIMIT} samples per dataset"
    echo "=============================================="

    source .venv_vllm_inf/bin/activate

    python inference_scaling/scripts/generate_datasets.py \
        --base_data_dir "/n/fs/vision-mix/sk7524/LLaMA-Factory/data" \
        --output_dir "./inference_scaling/data" \
        --splits test \
        --seed 42 \
        --limit ${LIMIT}

    # Generate dataset_info.json for LlamaFactory
    python inference_scaling/scripts/generate_dataset_info.py \
        --data_dir "./inference_scaling/data" \
        --output "./inference_scaling/data/dataset_info.json"

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
    INFERENCE_JOB=$(sbatch --parsable inference_scaling/sbatch/run_inference.sbatch)
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

    # Check if inference results exist
    for modality in clean clean_images vision; do
        pred_file="inference_scaling/results/${modality}/new_fewshot/predictions.jsonl"
        if [ ! -f "${pred_file}" ]; then
            echo "Warning: ${pred_file} not found. Meta-review may fail for ${modality}."
        fi
    done

    METAREVIEW_JOB=$(sbatch --parsable inference_scaling/sbatch/run_metareview.sbatch)
    echo "Submitted meta-review job array: ${METAREVIEW_JOB}"
}

# ============================================================================
# Step 4: Extract Results
# ============================================================================
extract_results() {
    echo "=============================================="
    echo "Step 4: Extracting results"
    echo "=============================================="

    source .venv_vllm_inf/bin/activate

    RESULTS_DIR="inference_scaling/results"

    for modality in clean clean_images vision; do
        for variant in original new new_fewshot; do
            pred_file="${RESULTS_DIR}/${modality}/${variant}/predictions.jsonl"

            if [ ! -f "${pred_file}" ]; then
                echo "Skipping ${modality}/${variant}: predictions not found"
                continue
            fi

            echo "Processing ${modality}/${variant}..."

            # Single strategy (variants 1-3)
            python inference_scaling/scripts/extract_results.py \
                --input "${pred_file}" \
                --output "${RESULTS_DIR}/${modality}/${variant}/results_single.jsonl" \
                --strategy single

            # Calibrated strategy (using overall score threshold)
            python inference_scaling/scripts/extract_results.py \
                --input "${pred_file}" \
                --output "${RESULTS_DIR}/${modality}/${variant}/results_calibrated.jsonl" \
                --strategy single \
                --use_calibration \
                --threshold 6

            # Majority strategy (for ensemble predictions)
            if [ "${variant}" == "new_fewshot" ]; then
                python inference_scaling/scripts/extract_results.py \
                    --input "${pred_file}" \
                    --output "${RESULTS_DIR}/${modality}/${variant}/results_majority.jsonl" \
                    --strategy majority

                # Meta-review results (if available)
                meta_pred="${RESULTS_DIR}/${modality}/${variant}/metareview_predictions.jsonl"
                if [ -f "${meta_pred}" ]; then
                    python inference_scaling/scripts/run_metareview.py extract \
                        --input "${meta_pred}" \
                        --output "${RESULTS_DIR}/${modality}/${variant}/results_metareview.jsonl"
                fi
            fi
        done
    done

    echo "Result extraction complete!"
}

# ============================================================================
# Step 5: Compute Metrics
# ============================================================================
compute_metrics() {
    echo "=============================================="
    echo "Step 5: Computing metrics and generating plots"
    echo "=============================================="

    source .venv_vllm_inf/bin/activate

    python inference_scaling/scripts/compute_metrics.py \
        --results_dir "./inference_scaling/results" \
        --output_dir "./inference_scaling/metrics" \
        --base_data_dir "/n/fs/vision-mix/sk7524/LLaMA-Factory/data"

    echo "Metrics computation complete!"
    echo "Results saved to: ./inference_scaling/metrics/"
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
