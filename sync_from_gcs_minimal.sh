#!/bin/bash
# Minimum dataset sync from GCS for the 50/50 ICLR experiments.
#
# Pulls: text + vision train/val/test JSONs, dataset_info.json, and
# the images/ tree (vision pages — ~91 GB).
#
# Skips: ratio-modified test sets (40_60, 30_70), reversed datasets,
# year-conditioned datasets, paper-stats variants, gemini/qwen reviews
# augmentations, scaling/cleaned variants, NIPS, ICML/COLM, ICLR 2017-2019.
#
# Usage:
#   bash sync_from_gcs_minimal.sh        # 50/50 text + vision + images
#   bash sync_from_gcs_minimal.sh text   # text only (no images, ~2 GB)
#   bash sync_from_gcs_minimal.sh vision # vision only (~91 GB with images)
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (`gcloud auth login`)
#   - Project's dataset is in gs://autoreviewer-data/autoreviewer_data/

set -e

GCS_BASE="gs://autoreviewer-data/autoreviewer_data"
LOCAL_BASE="data"

MODE=${1:-both}  # text | vision | both

mkdir -p "$LOCAL_BASE"

TEXT_BASE="iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
VISION_BASE="iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480"

sync_dir() {
    local name=$1
    echo "  -> ${name}"
    gcloud alpha storage rsync -r --gzip-in-flight-all \
        "${GCS_BASE}/${name}" "${LOCAL_BASE}/${name}"
}

echo "=== dataset_info.json (the dataset registry) ==="
gcloud alpha storage cp "${GCS_BASE}/dataset_info.json" "${LOCAL_BASE}/dataset_info.json"

if [[ "$MODE" == "text" || "$MODE" == "both" ]]; then
    echo ""
    echo "=== TEXT (~2 GB total) ==="
    for split in train validation test; do
        sync_dir "${TEXT_BASE}_${split}"
    done
fi

if [[ "$MODE" == "vision" || "$MODE" == "both" ]]; then
    echo ""
    echo "=== VISION JSONs (~50 MB) ==="
    for split in train validation test; do
        sync_dir "${VISION_BASE}_${split}"
    done

    echo ""
    echo "=== VISION IMAGES (~91 GB) ==="
    echo "  -> images/  (this dir is the bulk of the data)"
    gcloud alpha storage rsync -r --gzip-in-flight-all \
        "${GCS_BASE}/images" "${LOCAL_BASE}/images"
fi

echo ""
echo "=== DONE ==="
echo ""
echo "Verify:"
echo "  ls ${LOCAL_BASE}/${TEXT_BASE}_test/data.json"
echo "  ls ${LOCAL_BASE}/${VISION_BASE}_test/data.json"
echo "  ls ${LOCAL_BASE}/images | wc -l   # ~24,925 submission dirs"
