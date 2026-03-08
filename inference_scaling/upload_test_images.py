#!/usr/bin/env python3
"""Upload test images to GCS for Gemini batch inference.

Usage:
    python inference_scaling/upload_test_images.py
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set credentials before importing google.cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/u/jl0796/.config/gcloud/application_default_credentials.json"

from google.cloud import storage

# Configuration
SOURCE_DIR = Path("/n/fs/vision-mix/sk7524/LLaMA-Factory")
IMAGE_LIST = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/inference_scaling/test_images_list.txt")
GCS_BUCKET = "jl0796-autoreviewer-staging"
PROJECT = "hip-gecko-485003-c4"
MAX_WORKERS = 32  # Parallel uploads


def upload_file(client, bucket_name, local_path, gcs_path):
    """Upload a single file to GCS."""
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        return True, gcs_path
    except Exception as e:
        return False, f"{gcs_path}: {e}"


def main():
    print("=" * 60)
    print("Uploading test images to GCS")
    print("=" * 60)

    # Load image list
    with open(IMAGE_LIST) as f:
        images = [line.strip() for line in f if line.strip()]

    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: gs://{GCS_BUCKET}")
    print(f"Total images: {len(images)}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print()

    # Initialize client
    client = storage.Client(project=PROJECT)

    # Upload files in parallel
    uploaded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for img_path in images:
            local_path = SOURCE_DIR / img_path
            if local_path.exists():
                future = executor.submit(
                    upload_file, client, GCS_BUCKET, str(local_path), img_path
                )
                futures[future] = img_path

        for i, future in enumerate(as_completed(futures)):
            success, result = future.result()
            if success:
                uploaded += 1
            else:
                failed += 1
                print(f"FAILED: {result}")

            # Progress update every 1000 files
            if (i + 1) % 1000 == 0:
                print(f"Progress: {i + 1}/{len(futures)} ({uploaded} uploaded, {failed} failed)")

    print()
    print("=" * 60)
    print(f"Upload complete!")
    print(f"  Uploaded: {uploaded}")
    print(f"  Failed: {failed}")
    print(f"  Images at: gs://{GCS_BUCKET}/data/images/")
    print("=" * 60)

    # Grant Gemini service account access
    print()
    print("Granting Gemini service account access...")
    try:
        bucket = client.bucket(GCS_BUCKET)
        policy = bucket.get_iam_policy(requested_policy_version=3)

        # Add objectViewer role for Gemini service account
        service_account = "serviceAccount:service-969596589080@gcp-sa-aiplatform.iam.gserviceaccount.com"
        role = "roles/storage.objectViewer"

        # Check if binding already exists
        binding_exists = False
        for binding in policy.bindings:
            if binding["role"] == role and service_account in binding["members"]:
                binding_exists = True
                break

        if not binding_exists:
            policy.bindings.append({
                "role": role,
                "members": [service_account]
            })
            bucket.set_iam_policy(policy)
            print(f"  Granted {role} to Gemini service account")
        else:
            print("  Gemini service account already has access")
    except Exception as e:
        print(f"  Warning: Could not set IAM policy: {e}")
        print("  You may need to run manually:")
        print(f"    gsutil iam ch {service_account}:objectViewer gs://{GCS_BUCKET}")

    print()
    print("=" * 60)
    print("Done! Resubmit vision jobs with:")
    print("=" * 60)
    print(f"""python inference_scaling/scripts/gemini_inference.py submit \\
    --data_dir inference_scaling/data \\
    --output_dir inference_scaling/results/gemini \\
    --project {PROJECT} \\
    --gcs_staging gs://{GCS_BUCKET}/inference_scaling \\
    --gcs_base gs://{GCS_BUCKET} \\
    --modality clean_images""")


if __name__ == "__main__":
    main()
