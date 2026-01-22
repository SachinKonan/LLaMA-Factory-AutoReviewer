#!/usr/bin/env python3
"""
Gemini Batch Inference for Inference Scaling Experiments.

Submits batch jobs to Gemini API for all modalities and prompt variants.
Supports both original (boxed) and new (JSON) prompt formats.

source .venv_vllm_inf/bin/activate

Usage:
    # Submit all inference jobs
    python inference_scaling/scripts/gemini_inference.py submit \
        --data_dir inference_scaling/data \
        --output_dir inference_scaling/results/gemini \
        --project hip-gecko-485003-c4 \
        --gcs_staging gs://jl0796-autoreviewer-staging/inference_scaling

    # Check status of all jobs
    python inference_scaling/scripts/gemini_inference.py status \
        --output_dir inference_scaling/results/gemini \
        --project hip-gecko-485003-c4

    # Retrieve completed results
    python inference_scaling/scripts/gemini_inference.py retrieve \
        --output_dir inference_scaling/results/gemini \
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

from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_GCS_IMAGES_BASE = "gs://autoreviewer-data/autoreviewer_data/images"
LOCAL_IMAGES_PREFIX = "data/images/"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-central1"
DEFAULT_PROJECT = "ringed-inn-474523-u3"
DEFAULT_GCS_STAGING = "gs://jl0796-autoreviewer-staging/inference_scaling"

MODALITIES = ["clean", "clean_images", "vision"]
VARIANTS = ["original", "new", "new_fewshot"]

# Dataset name pattern
DATASET_PATTERN = "iclr_2020_2025_85_5_10_split6_balanced_{modality}_binary_noreviews_v6_test_{variant}"


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

    processed = 0
    with open(output_path, "w") as f:
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
            processed += 1

    print(f"  Saved {processed} results to: {output_path}")
    return processed


# ============================================================================
# MAIN COMMANDS
# ============================================================================

def cmd_submit(args):
    """Submit batch inference jobs."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Determine which jobs to submit
    modalities = [args.modality] if args.modality else MODALITIES
    variants = [args.variant] if args.variant else VARIANTS

    client = get_client(args.project, args.location)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    jobs_info = []

    for modality in modalities:
        for variant in variants:
            print(f"\n{'='*70}")
            print(f"Processing: {modality}/{variant}")
            print(f"{'='*70}")

            try:
                # Load dataset
                data = load_dataset(data_dir, modality, variant)

                # Apply limit if specified
                if args.limit:
                    data = data[:args.limit]
                    print(f"  Limited to {len(data)} samples")

                # Determine n_generations
                n_generations = 5 if variant == "new_fewshot" else 1

                # Create batch JSONL
                local_jsonl = f"/tmp/batch_{modality}_{variant}_{timestamp}.jsonl"
                _, metadata_list = create_batch_jsonl(
                    data, local_jsonl, variant, args.gcs_base,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    thinking_budget=args.thinking_budget,
                    n_generations=n_generations,
                )

                # Upload to GCS
                input_uri = f"{args.gcs_staging}/{modality}_{variant}_{timestamp}_input.jsonl"
                output_uri = f"{args.gcs_staging}/{modality}_{variant}_{timestamp}_output/"
                upload_to_gcs(local_jsonl, input_uri, args.project)

                # Submit job
                display_name = f"inf_{modality}_{variant}_{timestamp}"
                job = submit_batch_job(client, args.model, input_uri, output_uri, display_name)

                # Save job info
                job_info = {
                    "modality": modality,
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

                # Save metadata
                result_dir = output_dir / modality / variant
                result_dir.mkdir(parents=True, exist_ok=True)

                metadata_path = result_dir / "gemini_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata_list, f)

                job_info_path = result_dir / "gemini_job_info.json"
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


def cmd_status(args):
    """Check status of all jobs."""
    output_dir = Path(args.output_dir)
    client = get_client(args.project, args.location)

    print(f"\n{'='*70}")
    print("Job Status")
    print(f"{'='*70}\n")

    for modality in MODALITIES:
        for variant in VARIANTS:
            job_info_path = output_dir / modality / variant / "gemini_job_info.json"

            if not job_info_path.exists():
                continue

            with open(job_info_path) as f:
                job_info = json.load(f)

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

            print(f"{marker} {modality}/{variant}: {state}")


def cmd_retrieve(args):
    """Retrieve results from completed jobs."""
    output_dir = Path(args.output_dir)
    client = get_client(args.project, args.location)

    # Determine which jobs to retrieve
    modalities = [args.modality] if args.modality else MODALITIES
    variants = [args.variant] if args.variant else VARIANTS

    for modality in modalities:
        for variant in variants:
            result_dir = output_dir / modality / variant
            job_info_path = result_dir / "gemini_job_info.json"
            metadata_path = result_dir / "gemini_metadata.json"

            if not job_info_path.exists():
                print(f"Skipping {modality}/{variant}: no job info found")
                continue

            print(f"\n{'='*70}")
            print(f"Retrieving: {modality}/{variant}")
            print(f"{'='*70}")

            with open(job_info_path) as f:
                job_info = json.load(f)

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
            output_path = result_dir / "predictions.jsonl"
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

            print(f"  Saved {num_processed} results")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini Batch Inference for Inference Scaling"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit batch jobs")
    submit_parser.add_argument("--data_dir", type=str, default="inference_scaling/data")
    submit_parser.add_argument("--output_dir", type=str, default="inference_scaling/results/gemini")
    submit_parser.add_argument("--modality", type=str, choices=MODALITIES, help="Specific modality")
    submit_parser.add_argument("--variant", type=str, choices=VARIANTS, help="Specific variant")
    submit_parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    submit_parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    submit_parser.add_argument("--gcs_base", type=str, default=DEFAULT_GCS_IMAGES_BASE)
    submit_parser.add_argument("--gcs_staging", type=str, default=DEFAULT_GCS_STAGING)
    submit_parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    submit_parser.add_argument("--temperature", type=float, default=0.7)
    submit_parser.add_argument("--max_tokens", type=int, default=250)
    submit_parser.add_argument("--thinking_budget", type=int, default=2000)
    submit_parser.add_argument("--limit", type=int, help="Limit samples per dataset")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("--output_dir", type=str, default="inference_scaling/results/gemini")
    status_parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    status_parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve results")
    retrieve_parser.add_argument("--output_dir", type=str, default="inference_scaling/results/gemini")
    retrieve_parser.add_argument("--modality", type=str, choices=MODALITIES, help="Specific modality")
    retrieve_parser.add_argument("--variant", type=str, choices=VARIANTS, help="Specific variant")
    retrieve_parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    retrieve_parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)

    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)


if __name__ == "__main__":
    main()
