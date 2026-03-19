#!/usr/bin/env bash
# final_inference_scaling/scripts/gemini_process_results.sh
# Process Gemini batch results: extract and compute metrics.

set -e

RESULTS_DIR="final_inference_scaling/results"
METRICS_DIR="final_inference_scaling/metrics/gemini"

# Models to process (nested directories)
MODELS=("gemini-2.5-flash")

echo "=============================================="
echo "Step 1: Extracting Results"
echo "=============================================="

for model in "${MODELS[@]}"; do
    model_path="${RESULTS_DIR}/${model}"
    if [ ! -d "${model_path}" ]; then
        echo "Skipping ${model}: directory not found"
        continue
    fi

    echo "Processing model: ${model}"
    
    # Dynamically find all predictions.jsonl files
    find "${model_path}" -name "predictions.jsonl" | while read pred_file; do
        eval_dir=$(dirname "${pred_file}")
        echo "  Extracting ${eval_dir}..."

        # Extraction logic (mirroring run_pipeline.sh)
        python final_inference_scaling/scripts/extract_results.py \
            --input "${pred_file}" \
            --output "${eval_dir}/results_single.jsonl" \
            --strategy single

        # Calibrated strategy
        python final_inference_scaling/scripts/extract_results.py \
            --input "${pred_file}" \
            --output "${eval_dir}/results_calibrated.jsonl" \
            --strategy single \
            --use_calibration \
            --threshold 6
    done
done

echo ""
echo "=============================================="
echo "Step 2: Computing Metrics"
echo "=============================================="

# Use the patched compute_metrics.py which handles nested directories
python final_inference_scaling/scripts/compute_metrics.py \
    --results_dir "${RESULTS_DIR}" \
    --output_dir "${METRICS_DIR}" \
    --model "gemini-2.5-flash"

echo ""
echo "Done! Metrics available in ${METRICS_DIR}"
