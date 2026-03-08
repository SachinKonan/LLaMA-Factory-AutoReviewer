#!/usr/bin/env python3
"""
Gemini Batch Inference for Title-Only Contamination Check.

Submits batch jobs to Gemini API for title-only datasets to check whether
models can predict accept/reject from paper titles alone.

source .venv_vllm_inf/bin/activate

Usage:
    # Submit title-only inference
    python 2_8_26/gemini_title_infer.py submit \
        --data_dir inference_scaling/data \
        --output_dir inference_scaling/results/gemini_title \
        --project hip-gecko-485003-c4 \
        --gcs_staging gs://jl0796-autoreviewer-staging/title_only \
        --model gemini-2.5-flash

    # Check status
    python 2_8_26/gemini_title_infer.py status \
        --output_dir inference_scaling/results/gemini_title \
        --project hip-gecko-485003-c4

    # Retrieve results
    python 2_8_26/gemini_title_infer.py retrieve \
        --output_dir inference_scaling/results/gemini_title \
        --project hip-gecko-485003-c4
"""

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-central1"
DEFAULT_PROJECT = "ringed-inn-474523-u3"
DEFAULT_GCS_STAGING = "gs://jl0796-autoreviewer-staging/title_only"

# Title-only: clean modality only, single variant
MODALITIES = ["clean"]
VARIANTS = ["title_only"]

DATASET_PATTERN = "iclr_2020_2025_85_5_10_split7_balanced_{modality}_binary_noreviews_v7_test_title_only"


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(data_dir: Path, modality: str, variant: str) -> list[dict]:
    """Load dataset from generated data directory."""
    dataset_name = DATASET_PATTERN.format(modality=modality)
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


# ============================================================================
# BATCH REQUEST CREATION
# ============================================================================

def create_batch_request(
    entry: dict,
    temperature: float = 0.7,
    max_tokens: int = 250,
    thinking_budget: int = 500,
) -> dict:
    """Create a batch request for a single entry.

    Title-only entries are pure text with no images.
    """
    messages = parse_conversations(entry)

    request = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": messages["human"] or ""}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "candidateCount": 1,
        },
    }

    if thinking_budget > 0:
        request["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": thinking_budget
        }

    if messages["system"]:
        request["systemInstruction"] = {
            "parts": [{"text": messages["system"]}]
        }

    return {"request": request}


def create_batch_jsonl(
    data: list[dict],
    output_path: str,
    temperature: float = 0.7,
    max_tokens: int = 250,
    thinking_budget: int = 500,
) -> tuple[str, list[dict]]:
    """Create JSONL file for batch processing."""
    print(f"Creating batch JSONL: {output_path}")

    metadata_list = []

    with open(output_path, "w") as f:
        for idx, entry in enumerate(data):
            request = create_batch_request(
                entry, temperature, max_tokens, thinking_budget
            )
            f.write(json.dumps(request) + "\n")

            messages = parse_conversations(entry)
            metadata_list.append({
                "idx": idx,
                "submission_id": entry.get("_metadata", {}).get("submission_id", ""),
                "label": messages["gpt"] or "",
                "ground_truth": entry.get("_metadata", {}).get("answer", ""),
                "human_message": messages["human"] or "",
            })

    file_size = os.path.getsize(output_path)
    print(f"  Created {len(data)} requests, file size: {file_size / 1024:.1f} KB")

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
        http_options=HttpOptions(api_version="v1"),
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
        display_name = f"title_only_{timestamp}"

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
        ),
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

def parse_prediction(response: dict) -> str:
    """Extract prediction text from batch response."""
    try:
        candidates = response.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            text = ""
            for part in parts:
                if "text" in part:
                    text += part.get("text", "")
            return text
    except (KeyError, IndexError, TypeError):
        pass
    return ""


def process_batch_results(
    output_uri: str,
    metadata_list: list[dict],
    output_path: str,
    project: str,
) -> int:
    """Process batch results and save in predictions.jsonl format."""
    print(f"\nProcessing batch results from: {output_uri}")

    result_files = list_gcs_files(output_uri, project)
    result_files = [f for f in result_files if f.endswith(".jsonl")]

    if not result_files:
        print("  No result files found!")
        return 0

    print(f"  Found {len(result_files)} result file(s)")

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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed = 0
    with open(output_path, "w") as f:
        for idx, result in enumerate(all_results):
            if idx >= len(metadata_list):
                break

            meta = metadata_list[idx]
            response = result.get("response", {})
            prediction = parse_prediction(response)

            output_entry = {
                "prompt": meta["human_message"],
                "predict": prediction,
                "label": meta["label"],
                "n_generations": 1,
            }

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

    client = get_client(args.project, args.location)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    jobs_info = []

    for modality in MODALITIES:
        for variant in VARIANTS:
            print(f"\n{'='*70}")
            print(f"Processing: {modality}/{variant}")
            print(f"{'='*70}")

            try:
                data = load_dataset(data_dir, modality, variant)

                if args.limit:
                    data = data[:args.limit]
                    print(f"  Limited to {len(data)} samples")

                local_jsonl = f"/tmp/batch_title_{modality}_{timestamp}.jsonl"
                _, metadata_list = create_batch_jsonl(
                    data,
                    local_jsonl,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    thinking_budget=args.thinking_budget,
                )

                input_uri = f"{args.gcs_staging}/title_{modality}_{timestamp}_input.jsonl"
                output_uri = f"{args.gcs_staging}/title_{modality}_{timestamp}_output/"
                upload_to_gcs(local_jsonl, input_uri, args.project)

                display_name = f"title_{modality}_{timestamp}"
                job = submit_batch_job(client, args.model, input_uri, output_uri, display_name)

                job_info = {
                    "modality": modality,
                    "variant": variant,
                    "job_name": job.name,
                    "model": args.model,
                    "num_samples": len(data),
                    "n_generations": 1,
                    "input_uri": input_uri,
                    "output_uri": output_uri,
                    "submitted_at": timestamp,
                }
                jobs_info.append(job_info)

                result_dir = output_dir / modality / variant
                result_dir.mkdir(parents=True, exist_ok=True)

                metadata_path = result_dir / f"gemini_metadata_{timestamp}.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata_list, f)

                job_info_path = result_dir / f"gemini_job_info_{timestamp}.json"
                with open(job_info_path, "w") as f:
                    json.dump(job_info, f, indent=2)

                print(f"  Metadata saved: {metadata_path}")
                print(f"  Job info saved: {job_info_path}")

                os.remove(local_jsonl)

            except FileNotFoundError as e:
                print(f"  Skipping: {e}")
                continue

    all_jobs_path = output_dir / f"all_jobs_{timestamp}.json"
    with open(all_jobs_path, "w") as f:
        json.dump(jobs_info, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Submitted {len(jobs_info)} batch jobs")
    print(f"All jobs info saved to: {all_jobs_path}")
    print(f"{'='*70}")


def find_job_info_files(output_dir: Path, modality: str, variant: str) -> list[Path]:
    """Find all job info files for a modality/variant combination."""
    result_dir = output_dir / modality / variant
    if not result_dir.exists():
        return []

    job_files = []

    legacy_path = result_dir / "gemini_job_info.json"
    if legacy_path.exists():
        job_files.append(legacy_path)

    for f in result_dir.glob("gemini_job_info_*.json"):
        if f not in job_files:
            job_files.append(f)

    job_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return job_files


def cmd_status(args):
    """Check status of all jobs."""
    output_dir = Path(args.output_dir)
    client = get_client(args.project, args.location)

    print(f"\n{'='*70}")
    print("Job Status")
    print(f"{'='*70}\n")

    found_any = False
    for modality in MODALITIES:
        for variant in VARIANTS:
            job_files = find_job_info_files(output_dir, modality, variant)

            if not job_files:
                continue

            for job_info_path in job_files:
                with open(job_info_path) as f:
                    job_info = json.load(f)

                timestamp = job_info.get("submitted_at", "unknown")
                if args.timestamp and args.timestamp not in timestamp:
                    continue

                found_any = True
                status = check_job_status(client, job_info["job_name"])
                state = status["state"]

                if "SUCCEEDED" in state:
                    marker = "[DONE]"
                elif "FAILED" in state:
                    marker = "[FAIL]"
                elif "RUNNING" in state:
                    marker = "[RUN]"
                else:
                    marker = "[PEND]"

                model = job_info.get("model", "unknown")
                print(f"{marker} {modality}/{variant} ({timestamp}) [{model}]: {state}")

    if not found_any:
        print("No jobs found matching the specified filters.")


def cmd_retrieve(args):
    """Retrieve results from completed jobs."""
    output_dir = Path(args.output_dir)
    client = get_client(args.project, args.location)

    for modality in MODALITIES:
        for variant in VARIANTS:
            result_dir = output_dir / modality / variant
            job_files = find_job_info_files(output_dir, modality, variant)

            if not job_files:
                print(f"Skipping {modality}/{variant}: no job info found")
                continue

            for job_info_path in job_files:
                job_filename = job_info_path.name
                if job_filename == "gemini_job_info.json":
                    metadata_path = result_dir / "gemini_metadata.json"
                    output_path = result_dir / "predictions.jsonl"
                else:
                    ts = job_filename.replace("gemini_job_info_", "").replace(".json", "")
                    metadata_path = result_dir / f"gemini_metadata_{ts}.json"
                    output_path = result_dir / f"predictions_{ts}.jsonl"

                if not metadata_path.exists():
                    print(f"Skipping {job_info_path.name}: metadata file not found")
                    continue

                with open(job_info_path) as f:
                    job_info = json.load(f)

                submitted_at = job_info.get("submitted_at", "unknown")
                if args.timestamp and args.timestamp not in submitted_at:
                    continue

                if job_info.get("status") == "retrieved":
                    print(f"Skipping {modality}/{variant} ({submitted_at}): already retrieved")
                    continue

                print(f"\n{'='*70}")
                print(f"Retrieving: {modality}/{variant} ({submitted_at})")
                print(f"{'='*70}")

                status = check_job_status(client, job_info["job_name"])
                print(f"  Status: {status['state']}")

                if not status["completed"]:
                    print(f"  Job not completed yet, skipping")
                    continue

                with open(metadata_path) as f:
                    metadata_list = json.load(f)

                num_processed = process_batch_results(
                    job_info["output_uri"],
                    metadata_list,
                    str(output_path),
                    project=args.project,
                )

                job_info["status"] = "retrieved"
                job_info["retrieved_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                job_info["num_results"] = num_processed
                with open(job_info_path, "w") as f:
                    json.dump(job_info, f, indent=2)

                print(f"  Saved {num_processed} results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini Batch Inference for Title-Only Contamination Check"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit batch jobs")
    submit_parser.add_argument("--data_dir", type=str, default="inference_scaling/data")
    submit_parser.add_argument("--output_dir", type=str, default="inference_scaling/results/gemini_title")
    submit_parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    submit_parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    submit_parser.add_argument("--gcs_staging", type=str, default=DEFAULT_GCS_STAGING)
    submit_parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    submit_parser.add_argument("--temperature", type=float, default=1.0)
    submit_parser.add_argument("--max_tokens", type=int, default=500)
    submit_parser.add_argument("--thinking_budget", type=int, default=500)
    submit_parser.add_argument("--limit", type=int, help="Limit samples per dataset")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("--output_dir", type=str, default="inference_scaling/results/gemini_title")
    status_parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    status_parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    status_parser.add_argument("--timestamp", type=str, help="Filter by timestamp")

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve results")
    retrieve_parser.add_argument("--output_dir", type=str, default="inference_scaling/results/gemini_title")
    retrieve_parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    retrieve_parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    retrieve_parser.add_argument("--timestamp", type=str, help="Filter by timestamp")

    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)


if __name__ == "__main__":
    main()
