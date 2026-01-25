#!/usr/bin/env python3
"""
Gemini Batch Inference for Inference Scaling Experiments - Ablation V1.

Submits batch jobs to Gemini API for Ablation V1 datasets.
Supports both original (boxed) and new (JSON) prompt formats.

The `--test_subset` flag can be used to submit only the JSON variants of the less critical 
modifier datasets (10 samples each): 
- iclr_2020_2025_85_5_10_split7_ablation_v1_fewshot_fullpaper_2acc_0rej_clean_test
- iclr_2020_2025_85_5_10_split7_ablation_v1_fewshot_fullpaper_2acc_0rej_clean_images_test
- iclr_2020_2025_85_5_10_split7_ablation_v1_fewshot_fullpaper_2acc_0rej_vision_test

Use:
source .venv_vllm_inf/bin/activate

Usage:
    # Submit ablation v1 datasets
    python ablations/scripts_v1/infer_gemini.py submit \
        --data_dir data \
        --output_dir ablations/results_v1/gemini_2.5_flash \
        --project hip-gecko-485003-c4 \
        --gcs_staging gs://jl0796-autoreviewer-staging/ablation_v1 \
        --model gemini-2.5-flash \
        --thinking_budget 1000 \
        --max_tokens 1000 \
        --gcs_base gs://jl0796-autoreviewer-staging/autoreviewer_data/images \
        --upload_images # SOMETIMES.

    # Submit test subset (10 samples of JSON less critical datasets)
    python ablations/scripts_v1/infer_gemini.py submit \
        --test_subset \
        --upload_images \
        --project hip-gecko-485003-c4 \
        --model gemini-2.5-flash \
        --thinking_budget 1000 \
        --max_tokens 1000 \
        --gcs_base gs://jl0796-autoreviewer-staging/autoreviewer_data/images

    # Check status of all jobs
    python ablations/scripts_v1/infer_gemini.py status \
        --output_dir ablations/results_v1/gemini \
        --project hip-gecko-485003-c4

    # Retrieve completed results
    python ablations/scripts_v1/infer_gemini.py retrieve \
        --output_dir ablations/results_v1/gemini \
        --project hip-gecko-485003-c4
"""

import argparse
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_GCS_IMAGES_BASE = "gs://jl0796-autoreviewer-staging/autoreviewer_data/images"
LOCAL_IMAGES_PREFIX = "data/images/"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-central1"
DEFAULT_PROJECT = "hip-gecko-485003-c4"
DEFAULT_GCS_STAGING = "gs://jl0796-autoreviewer-staging/inference_scaling"

# Ablation V1 dataset prefix
ABLATION_V1_PREFIX = "iclr_2020_2025_85_5_10_split7_ablation_v1"

# Ablation V1 datasets: Less Critical Modifier + Fewshot Full Paper
ABLATION_V1_DATASETS = [
    # Less Critical Modifier (6 datasets)
    f"{ABLATION_V1_PREFIX}_less_critical_boxed_clean_test",
    f"{ABLATION_V1_PREFIX}_less_critical_boxed_clean_images_test",
    f"{ABLATION_V1_PREFIX}_less_critical_boxed_vision_test",
    f"{ABLATION_V1_PREFIX}_less_critical_json_clean_test",
    f"{ABLATION_V1_PREFIX}_less_critical_json_clean_images_test",
    f"{ABLATION_V1_PREFIX}_less_critical_json_vision_test",
    # Fewshot Full Paper (9 datasets)
    f"{ABLATION_V1_PREFIX}_fewshot_fullpaper_2acc_0rej_clean_test",
    f"{ABLATION_V1_PREFIX}_fewshot_fullpaper_2acc_0rej_clean_images_test",
    f"{ABLATION_V1_PREFIX}_fewshot_fullpaper_2acc_0rej_vision_test",
    f"{ABLATION_V1_PREFIX}_fewshot_fullpaper_1acc_1rej_clean_test",
    f"{ABLATION_V1_PREFIX}_fewshot_fullpaper_1acc_1rej_clean_images_test",
    f"{ABLATION_V1_PREFIX}_fewshot_fullpaper_1acc_1rej_vision_test",
    f"{ABLATION_V1_PREFIX}_fewshot_fullpaper_0acc_2rej_clean_test",
    f"{ABLATION_V1_PREFIX}_fewshot_fullpaper_0acc_2rej_clean_images_test",
    f"{ABLATION_V1_PREFIX}_fewshot_fullpaper_0acc_2rej_vision_test",
]

# Test Subset: JSON variants of Less Critical Modifier
TEST_SUBSET_DATASETS = [
    f"{ABLATION_V1_PREFIX}_fewshot_fullpaper_2acc_0rej_clean_test",
    f"{ABLATION_V1_PREFIX}_fewshot_fullpaper_2acc_0rej_clean_images_test",
    f"{ABLATION_V1_PREFIX}_fewshot_fullpaper_2acc_0rej_vision_test",
]


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(data_dir: Path, modality: str, variant: str) -> list[dict]:
    """Load dataset from generated data directory.

    Args:
        data_dir: Path to inference_scaling/data
        modality: clean, clean_images, or vision
        variant: original, new, or new_fewshot

    Returns:
        List of dataset entries
    """
    dataset_name = DATASET_PATTERN.format(modality=modality, variant=variant)
    dataset_path = data_dir / dataset_name / "data.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} samples")
    return data


def load_dataset_by_name(data_dir: Path, dataset_name: str) -> list[dict]:
    """Load dataset by exact name.

    Args:
        data_dir: Path to data directory
        dataset_name: Full dataset name

    Returns:
        List of dataset entries
    """
    dataset_path = data_dir / dataset_name / "data.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} samples")
    return data


def get_variant_from_dataset_name(dataset_name: str) -> str:
    """Determine variant (boxed/json) from dataset name for format hints."""
    if "_boxed_" in dataset_name:
        return "original"  # boxed format
    elif "_json_" in dataset_name:
        return "new"  # JSON format
    else:
        return "new"  # default to JSON for fewshot


def parse_conversations(entry: dict) -> dict:
    """Parse conversations to extract system, human, and gpt messages."""
    result = {"system": None, "human": None, "gpt": None}

    for msg in entry.get("conversations", []):
        role = msg.get("from", "")
        value = msg.get("value", "")

        if role == "system":
            result["system"] = value
        elif role == "human":
            result["human"] = value
        elif role == "gpt":
            result["gpt"] = value

    return result


def convert_image_path_to_gcs(local_path: str, gcs_base: str) -> str:
    """Convert local image path to GCS URI."""
    if local_path.startswith(LOCAL_IMAGES_PREFIX):
        relative_path = local_path[len(LOCAL_IMAGES_PREFIX):]
        return f"{gcs_base.rstrip('/')}/{relative_path}"
    return local_path


def get_mime_type(path: str) -> str:
    """Get MIME type from file extension."""
    ext = path.lower().split(".")[-1]
    mime_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    return mime_types.get(ext, "image/png")


# ============================================================================
# BATCH REQUEST CREATION
# ============================================================================

def build_content_parts(human_message: str, images: list[str], gcs_base: str) -> list[dict]:
    """Build content parts array with text and images."""
    parts = []

    # Split message by <image> placeholders
    segments = re.split(r"<image>", human_message)
    image_idx = 0

    for i, segment in enumerate(segments):
        # Add text segment if non-empty
        if segment.strip():
            parts.append({"text": segment})

        # Add image part after each segment (except the last)
        if i < len(segments) - 1 and image_idx < len(images):
            gcs_uri = convert_image_path_to_gcs(images[image_idx], gcs_base)
            mime_type = get_mime_type(images[image_idx])
            parts.append({
                "fileData": {
                    "fileUri": gcs_uri,
                    "mimeType": mime_type
                }
            })
            image_idx += 1

    # If no segments or all empty, add the full message
    if not parts:
        parts.append({"text": human_message})

    return parts


def create_batch_request(
    entry: dict,
    variant: str,
    gcs_base: str,
    temperature: float = 0.7,
    max_tokens: int = 250,
    thinking_budget: int = 2000,
    n_generations: int = 1,
) -> dict:
    """Create a batch request for a single entry.

    Args:
        entry: Dataset entry
        variant: Prompt variant (original, new, new_fewshot)
        gcs_base: GCS base URI for images
        temperature: Sampling temperature
        max_tokens: Max output tokens
        thinking_budget: Thinking budget tokens (0 to disable)
        n_generations: Number of generations (for ensembling)

    Returns:
        Batch request dict
    """
    messages = parse_conversations(entry)
    images = entry.get("images", [])

    # Build content parts
    parts = build_content_parts(messages["human"] or "", images, gcs_base)

    # For original variant, add format reminder
    if variant == "original":
        parts.append({"text": "\n\nPLEASE FORMAT ANSWER AS: \\boxed{Accept} or \\boxed{Reject}"})

    # Build request
    request = {
        "contents": [
            {"role": "user", "parts": parts}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "candidateCount": n_generations,
        }
    }

    # Add thinking config if enabled
    if thinking_budget > 0:
        request["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": thinking_budget
        }

    # Add system instruction if present
    if messages["system"]:
        request["systemInstruction"] = {
            "parts": [{"text": messages["system"]}]
        }

    return {"request": request}


def create_batch_jsonl(
    data: list[dict],
    output_path: str,
    variant: str,
    gcs_base: str,
    temperature: float = 0.7,
    max_tokens: int = 250,
    thinking_budget: int = 2000,
    n_generations: int = 1,
) -> tuple[str, list[dict]]:
    """Create JSONL file for batch processing.

    Returns:
        Tuple of (output_path, metadata_list)
    """
    print(f"Creating batch JSONL: {output_path}")

    metadata_list = []

    with open(output_path, "w") as f:
        for idx, entry in enumerate(data):
            request = create_batch_request(
                entry, variant, gcs_base, temperature, max_tokens,
                thinking_budget, n_generations
            )
            f.write(json.dumps(request) + "\n")

            # Store metadata
            messages = parse_conversations(entry)
            metadata_list.append({
                "idx": idx,
                "submission_id": entry.get("_metadata", {}).get("submission_id", ""),
                "label": messages["gpt"] or "",
                "ground_truth": entry.get("_metadata", {}).get("answer", ""),
                "human_message": messages["human"] or "",
            })

    file_size = os.path.getsize(output_path)
    print(f"  Created {len(data)} requests, file size: {file_size / 1024 / 1024:.2f} MB")

    return output_path, metadata_list


# ============================================================================
# GCS AND CLIENT UTILITIES
# ============================================================================

def get_client(project: str, location: str) -> genai.Client:
    """Get Gemini client for Vertex AI."""
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=HttpOptions(api_version="v1")
    )


def upload_to_gcs(local_path: str, gcs_uri: str, project: str) -> str:
    """Upload local file to GCS."""
    from google.cloud import storage

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

    print(f"  Uploaded to: {gcs_uri}")
    return gcs_uri


def download_from_gcs(gcs_uri: str, local_path: str, project: str) -> str:
    """Download file from GCS."""
    from google.cloud import storage

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

    return local_path


def list_gcs_files(gcs_prefix: str, project: str) -> list[str]:
    """List files in GCS with prefix."""
    from google.cloud import storage

    parts = gcs_prefix[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    return [f"gs://{bucket_name}/{blob.name}" for blob in blobs]


def upload_images_parallel(
    image_paths: list[str],
    source_dir: Path,
    bucket_name: str,
    gcs_prefix: str,
    project: str,
    max_workers: int = 32
) -> tuple[int, int]:
    """Upload multiple images to GCS in parallel.

    Args:
        image_paths: List of relative image paths (e.g., 'data/images/xxx/0.png')
        source_dir: Local base directory where images are stored
        bucket_name: GCS bucket name
        gcs_prefix: Prefix within the bucket (e.g., 'autoreviewer_data/images')
        project: GCP project ID
        max_workers: Number of parallel workers
    """
    from google.cloud import storage
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)

    print(f"\nUploading {len(image_paths)} images to gs://{bucket_name}/{gcs_prefix}...")
    print(f"  Source: {source_dir}")
    print(f"  Max workers: {max_workers}")

    def upload_single(rel_path):
        try:
            local_path = source_dir / rel_path
            if not local_path.exists():
                return False, f"File not found: {local_path}"
            
            # Match the logic in convert_image_path_to_gcs:
            # strip 'data/images/' and prepend gcs_prefix
            if rel_path.startswith(LOCAL_IMAGES_PREFIX):
                clean_rel_path = rel_path[len(LOCAL_IMAGES_PREFIX):]
            else:
                clean_rel_path = rel_path
            
            target_blob_path = f"{gcs_prefix.rstrip('/')}/{clean_rel_path}"
            blob = bucket.blob(target_blob_path)
            
            blob.upload_from_filename(str(local_path))
            return True, target_blob_path
        except Exception as e:
            return False, f"{rel_path}: {e}"

    uploaded = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(upload_single, path): path for path in image_paths}
        
        for i, future in enumerate(as_completed(futures)):
            success, result = future.result()
            if success:
                uploaded += 1
            else:
                failed += 1
                print(f"    FAILED: {result}")
            
            if (i + 1) % 500 == 0:
                print(f"    Progress: {i + 1}/{len(image_paths)} ({uploaded} uploaded, {failed} failed)")

    print(f"  Upload complete: {uploaded} success, {failed} failed")
    return uploaded, failed


# ============================================================================
# BATCH JOB MANAGEMENT
# ============================================================================

def submit_batch_job(
    client: genai.Client,
    model: str,
    input_uri: str,
    output_uri: str,
    display_name: str = None,
) -> any:
    """Submit a batch prediction job."""
    if display_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        display_name = f"inf_scaling_{timestamp}"

    print(f"\nSubmitting batch job...")
    print(f"  Model: {model}")
    print(f"  Input: {input_uri}")
    print(f"  Output: {output_uri}")

    job = client.batches.create(
        model=model,
        src=input_uri,
        config=CreateBatchJobConfig(
            dest=output_uri,
            display_name=display_name,
        )
    )

    print(f"  Job submitted: {job.name}")
    print(f"  State: {job.state}")

    return job


def check_job_status(client: genai.Client, job_name: str) -> dict:
    """Check job status and return info dict."""
    job = client.batches.get(name=job_name)
    return {
        "name": job.name,
        "state": str(job.state),
        "completed": "SUCCEEDED" in str(job.state),
    }


# ============================================================================
# RESULT PROCESSING
# ============================================================================

def parse_prediction(response: dict) -> list[str]:
    """Extract prediction texts from batch response.

    Returns list of predictions (one per candidate).
    """
    predictions = []
    try:
        candidates = response.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            text = ""
            for part in parts:
                if "text" in part:
                    text += part.get("text", "")
            predictions.append(text)
    except (KeyError, IndexError, TypeError):
        pass
    return predictions if predictions else [""]


def process_batch_results(
    output_uri: str,
    metadata_list: list[dict],
    output_path: str,
    project: str,
    n_generations: int = 1,
) -> int:
    """Process batch results and save in vllm_infer format."""
    print(f"\nProcessing batch results from: {output_uri}")

    # List result files
    result_files = list_gcs_files(output_uri, project)
    result_files = [f for f in result_files if f.endswith(".jsonl")]

    if not result_files:
        print("  No result files found!")
        return 0

    print(f"  Found {len(result_files)} result file(s)")

    # Download and parse results
    all_results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for gcs_file in result_files:
            local_file = os.path.join(tmpdir, os.path.basename(gcs_file))
            download_from_gcs(gcs_file, local_file, project)

            with open(local_file) as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))

    print(f"  Parsed {len(all_results)} results")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Also save raw Gemini responses for inspection
    raw_output_path = output_path.replace(".jsonl", "_raw.jsonl")

    processed = 0
    with open(output_path, "w") as f, open(raw_output_path, "w") as f_raw:
        for idx, result in enumerate(all_results):
            if idx >= len(metadata_list):
                break

            meta = metadata_list[idx]
            response = result.get("response", {})
            predictions = parse_prediction(response)

            # Format compatible with vllm_infer output
            output_entry = {
                "prompt": meta["human_message"],
                "predict": predictions[0] if predictions else "",
                "label": meta["label"],
                "n_generations": n_generations,
            }

            # If multiple generations, include all
            if n_generations > 1:
                output_entry["all_predictions"] = predictions

            f.write(json.dumps(output_entry) + "\n")

            # Save raw Gemini response (full result object)
            raw_entry = {
                "idx": idx,
                "submission_id": meta.get("submission_id", ""),
                "ground_truth": meta.get("ground_truth", ""),
                "raw_response": result,
            }
            f_raw.write(json.dumps(raw_entry) + "\n")

            processed += 1

    print(f"  Saved {processed} results to: {output_path}")
    print(f"  Saved raw responses to: {raw_output_path}")
    return processed


# ============================================================================
# MAIN COMMANDS
# ============================================================================

def cmd_submit(args):
    """Submit batch inference jobs for Ablation V1."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    client = get_client(args.project, args.location)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    jobs_info = []
    all_image_paths = set()

    # Select datasets to submit
    if args.test_subset:
        datasets_to_submit = TEST_SUBSET_DATASETS
        # Default limit to 10 for test subset unless specified otherwise
        if args.limit is None:
            args.limit = 10
        print(f"Test Subset mode enabled: submitting 10f samples for {len(datasets_to_submit)} datasets")
    else:
        datasets_to_submit = ABLATION_V1_DATASETS

    # Pre-collect data and image paths
    dataset_worklist = []
    for dataset_name in datasets_to_submit:
        try:
            data = load_dataset_by_name(data_dir, dataset_name)
            if args.limit:
                data = data[:args.limit]
            
            dataset_worklist.append((dataset_name, data))
            
            # Collect all image paths
            for entry in data:
                for img in entry.get("images", []):
                    all_image_paths.add(img)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

    # Handle image uploads if requested
    if args.upload_images and all_image_paths:
        # Extract bucket name and prefix from gcs_base
        # gs://bucket/path -> bucket, path
        match = re.match(r"gs://([^/]+)/?(.*)", args.gcs_base)
        if not match:
            print(f"Error: Could not extract bucket from gcs_base: {args.gcs_base}")
        else:
            bucket_name = match.group(1)
            gcs_prefix = match.group(2) or ""
            source_dir = Path(args.source_dir)
            upload_images_parallel(
                list(all_image_paths),
                source_dir,
                bucket_name,
                gcs_prefix,
                args.project
            )

    for dataset_name, data in dataset_worklist:
        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*70}")

        try:
            # Determine variant for format hints
            variant = get_variant_from_dataset_name(dataset_name)

            # n_generations = 1 for all ablation datasets
            n_generations = 1

            # Create short name for files
            short_name = dataset_name.replace(f"{ABLATION_V1_PREFIX}_", "")

            # Create batch JSONL
            local_jsonl = f"/tmp/batch_{short_name}_{timestamp}.jsonl"
            _, metadata_list = create_batch_jsonl(
                data, local_jsonl, variant, args.gcs_base,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                thinking_budget=args.thinking_budget,
                n_generations=n_generations,
            )

            # Upload to GCS
            input_uri = f"{args.gcs_staging}/ablation_v1_{short_name}_{timestamp}_input.jsonl"
            output_uri = f"{args.gcs_staging}/ablation_v1_{short_name}_{timestamp}_output/"
            upload_to_gcs(local_jsonl, input_uri, args.project)

            # Submit job
            display_name = f"abl_{short_name[:30]}_{timestamp}"
            job = submit_batch_job(client, args.model, input_uri, output_uri, display_name)

            # Save job info
            job_info = {
                "dataset_name": dataset_name,
                "short_name": short_name,
                "variant": variant,
                "job_name": job.name,
                "model": args.model,
                "num_samples": len(data),
                "n_generations": n_generations,
                "input_uri": input_uri,
                "output_uri": output_uri,
                "submitted_at": timestamp,
            }
            jobs_info.append(job_info)

            # Save metadata - use dataset short name as directory
            result_dir = output_dir / short_name
            result_dir.mkdir(parents=True, exist_ok=True)

            metadata_path = result_dir / f"gemini_metadata_{timestamp}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata_list, f)

            job_info_path = result_dir / f"gemini_job_info_{timestamp}.json"
            with open(job_info_path, "w") as f:
                json.dump(job_info, f, indent=2)

            print(f"  Metadata saved: {metadata_path}")
            print(f"  Job info saved: {job_info_path}")

            # Cleanup
            os.remove(local_jsonl)

        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

    # Save all jobs info
    all_jobs_path = output_dir / f"all_jobs_{timestamp}.json"
    with open(all_jobs_path, "w") as f:
        json.dump(jobs_info, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Submitted {len(jobs_info)} batch jobs")
    print(f"All jobs info saved to: {all_jobs_path}")
    print(f"{'='*70}")


def find_job_info_files(output_dir: Path, modality: str, variant: str) -> list[Path]:
    """Find all job info files for a modality/variant combination.

    Supports both legacy (gemini_job_info.json) and timestamped
    (gemini_job_info_{timestamp}.json) formats.

    Returns:
        List of job info file paths, sorted by modification time (newest first)
    """
    result_dir = output_dir / modality / variant
    if not result_dir.exists():
        return []

    job_files = []

    # Legacy format (single file)
    legacy_path = result_dir / "gemini_job_info.json"
    if legacy_path.exists():
        job_files.append(legacy_path)

    # Timestamped format (multiple files)
    for f in result_dir.glob("gemini_job_info_*.json"):
        if f not in job_files:
            job_files.append(f)

    # Sort by modification time, newest first
    job_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return job_files


def find_ablation_v1_job_info_files(output_dir: Path) -> list[Path]:
    """Find all job info files for ablation_v1 datasets.

    Returns:
        List of job info file paths, sorted by modification time (newest first)
    """
    job_files = []

    # Look in each ablation dataset subdirectory
    for dataset_name in ABLATION_V1_DATASETS:
        short_name = dataset_name.replace(f"{ABLATION_V1_PREFIX}_", "")
        result_dir = output_dir / short_name

        if not result_dir.exists():
            continue

        # Find job info files
        for f in result_dir.glob("gemini_job_info_*.json"):
            job_files.append(f)

    # Sort by modification time, newest first
    job_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return job_files


def cmd_status(args):
    """Check status of Ablation V1 jobs."""
    output_dir = Path(args.output_dir)
    client = get_client(args.project, args.location)

    print(f"\n{'='*70}")
    print("Job Status")
    print(f"{'='*70}\n")

    found_any = False
    job_files = find_ablation_v1_job_info_files(output_dir)

    for job_info_path in job_files:
        with open(job_info_path) as f:
            job_info = json.load(f)

        # Filter by dataset name if specified
        if args.dataset and args.dataset not in job_info.get("dataset_name", ""):
            continue

        # Filter by timestamp if specified
        timestamp = job_info.get("submitted_at", "unknown")
        if args.timestamp and args.timestamp not in timestamp:
            continue

        found_any = True
        status = check_job_status(client, job_info["job_name"])
        state = status["state"]

        # Color coding
        if "SUCCEEDED" in state:
            marker = "[DONE]"
        elif "FAILED" in state:
            marker = "[FAIL]"
        elif "RUNNING" in state:
            marker = "[RUN]"
        else:
            marker = "[PEND]"

        # Include short name, timestamp, and model
        short_name = job_info.get("short_name", "unknown")
        model = job_info.get("model", "unknown")
        print(f"{marker} {short_name} ({timestamp}) [{model}]: {state}")

    if not found_any:
        print("No jobs found matching the specified filters.")


def cmd_retrieve(args):
    """Retrieve results for Ablation V1 jobs."""
    output_dir = Path(args.output_dir)
    client = get_client(args.project, args.location)

    job_files = find_ablation_v1_job_info_files(output_dir)

    if not job_files:
        print("No ablation job info found.")
        return

    for job_info_path in job_files:
        with open(job_info_path) as f:
            job_info = json.load(f)

        # Filter by dataset name if specified
        if args.dataset and args.dataset not in job_info.get("dataset_name", ""):
            continue

        # Filter by timestamp if specified
        submitted_at = job_info.get("submitted_at", "unknown")
        if args.timestamp and args.timestamp not in submitted_at:
            continue

        # Determine corresponding metadata file and output path
        short_name = job_info.get("short_name", "unknown")
        result_dir = output_dir / short_name
        
        job_filename = job_info_path.name
        ts = job_filename.replace("gemini_job_info_", "").replace(".json", "")
        metadata_path = result_dir / f"gemini_metadata_{ts}.json"
        output_path = result_dir / f"predictions_{ts}.jsonl"

        if not metadata_path.exists():
            print(f"Skipping {job_info_path.name}: metadata file not found at {metadata_path}")
            continue

        # Skip already retrieved jobs unless forced
        if job_info.get("status") == "retrieved":
            print(f"Skipping {short_name} ({submitted_at}): already retrieved")
            continue

        print(f"\n{'='*70}")
        print(f"Retrieving: {short_name} ({submitted_at})")
        print(f"{'='*70}")

        # Check status
        status = check_job_status(client, job_info["job_name"])
        print(f"  Status: {status['state']}")

        if not status["completed"]:
            print(f"  Job not completed yet, skipping")
            continue

        # Load metadata
        with open(metadata_path) as f:
            metadata_list = json.load(f)

        # Process results
        n_generations = job_info.get("n_generations", 1)

        num_processed = process_batch_results(
            job_info["output_uri"],
            metadata_list,
            str(output_path),
            project=args.project,
            n_generations=n_generations,
        )

        # Update job info
        job_info["status"] = "retrieved"
        job_info["retrieved_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_info["num_results"] = num_processed
        with open(job_info_path, "w") as f:
            json.dump(job_info, f, indent=2)

        print(f"  Saved {num_processed} results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini Batch Inference for Inference Scaling"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit batch jobs for Ablation V1")
    submit_parser.add_argument("--data_dir", type=str, default="data")
    submit_parser.add_argument("--output_dir", type=str, default="ablations/results_v1/gemini")
    submit_parser.add_argument("--test_subset", action="store_true",
                               help="Submit only the JSON variants of the less critical modifier datasets (10 samples each)")
    submit_parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    submit_parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    submit_parser.add_argument("--gcs_base", type=str, default=DEFAULT_GCS_IMAGES_BASE)
    submit_parser.add_argument("--gcs_staging", type=str, default=DEFAULT_GCS_STAGING)
    submit_parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    submit_parser.add_argument("--temperature", type=float, default=1.0)
    submit_parser.add_argument("--max_tokens", type=int, default=1000)
    submit_parser.add_argument("--thinking_budget", type=int, default=500)
    submit_parser.add_argument("--limit", type=int, help="Limit samples per dataset")
    submit_parser.add_argument("--upload_images", action="store_true", help="Automatically upload images to GCS")
    submit_parser.add_argument("--source_dir", type=str, default="/n/fs/vision-mix/sk7524/LLaMA-Factory",
                               help="Source directory for local images")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check status of Ablation V1 jobs")
    status_parser.add_argument("--output_dir", type=str, default="ablations/results_v1/gemini")
    status_parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    status_parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    status_parser.add_argument("--dataset", type=str, help="Filter by dataset name (substring)")
    status_parser.add_argument("--timestamp", type=str, help="Filter by timestamp (e.g., 20260124)")

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve results for Ablation V1 jobs")
    retrieve_parser.add_argument("--output_dir", type=str, default="ablations/results_v1/gemini")
    retrieve_parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    retrieve_parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    retrieve_parser.add_argument("--dataset", type=str, help="Filter by dataset name (substring)")
    retrieve_parser.add_argument("--timestamp", type=str, help="Filter by timestamp (e.g., 20260124)")

    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)


if __name__ == "__main__":
    main()
