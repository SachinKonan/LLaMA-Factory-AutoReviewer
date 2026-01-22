#!/bin/bash
# Upload test images to GCS for Gemini batch inference
#
# Usage:
#   ./inference_scaling/upload_test_images.sh
#
# This uploads only the images needed for the test set (~8.5GB instead of 339GB)

set -e
cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer
source .venv_vllm_inf/bin/activate

# Use application default credentials
export GOOGLE_APPLICATION_CREDENTIALS="/u/jl0796/.config/gcloud/application_default_credentials.json"
export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="/u/jl0796/.config/gcloud/application_default_credentials.json"


SOURCE_DIR="/n/fs/vision-mix/sk7524/LLaMA-Factory"
IMAGE_LIST="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/inference_scaling/test_images_list.txt"
GCS_BUCKET="gs://jl0796-autoreviewer-staging"

echo "=============================================="
echo "Uploading test images to GCS"
echo "Source: ${SOURCE_DIR}"
echo "Destination: ${GCS_BUCKET}"
echo "Total images: $(wc -l < ${IMAGE_LIST})"
echo "=============================================="

cd "${SOURCE_DIR}"

# Upload directly using gsutil with the file list
# Each image path is like: data/images/XXXXX/filename.jpg
echo "Starting parallel upload to GCS..."
cat "${IMAGE_LIST}" | gsutil -m cp -I "${GCS_BUCKET}/"

echo ""
echo "Upload complete!"
echo "Images available at: ${GCS_BUCKET}/data/images/"

# Grant Gemini service account access
echo ""
echo "Granting Gemini service account access..."
gsutil iam ch serviceAccount:service-969596589080@gcp-sa-aiplatform.iam.gserviceaccount.com:objectViewer "${GCS_BUCKET}"

echo ""
echo "=============================================="
echo "Done! Resubmit vision jobs with:"
echo "=============================================="
echo "python inference_scaling/scripts/gemini_inference.py submit \\"
echo "    --data_dir inference_scaling/data \\"
echo "    --output_dir inference_scaling/results/gemini \\"
echo "    --project hip-gecko-485003-c4 \\"
echo "    --gcs_staging gs://jl0796-autoreviewer-staging/inference_scaling \\"
echo "    --gcs_base gs://jl0796-autoreviewer-staging \\"
echo "    --modality clean_images"
