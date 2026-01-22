#!/usr/bin/env python3
"""
Gemini Batch Metareview for Inference Scaling Experiments.

Creates and submits metareview batch jobs that aggregate multiple reviewer outputs.

Usage:
    # Submit metareview jobs (after main inference is complete)
    python inference_scaling/scripts/gemini_metareview.py submit \
        --results_dir inference_scaling/results/gemini \
        --project hip-gecko-485003-c4

    # Check status
    python inference_scaling/scripts/gemini_metareview.py status \
        --results_dir inference_scaling/results/gemini \
        --project hip-gecko-485003-c4

    # Retrieve results
    python inference_scaling/scripts/gemini_metareview.py retrieve \
        --results_dir inference_scaling/results/gemini \
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
DEFAULT_PROJECT = "hip-gecko-485003-c4"
DEFAULT_GCS_STAGING = "gs://jl0796-autoreviewer-staging/inference_scaling"

MODALITIES = ["clean", "clean_images", "vision"]

METAREVIEW_SYSTEM_PROMPT = """You are a senior area chair reviewing paper submissions. You will be given multiple reviewer opinions about a paper. Your job is to synthesize these reviews and make a final decision."""

METAREVIEW_USER_TEMPLATE = """Here are {n_reviews} reviewer opinions for a paper submission:

{reviews}

Based on these reviews, provide your final assessment. Consider:
1. The consensus among reviewers
2. The strength of arguments for/against
3. Any critical issues raised

Respond with your reasoning followed by a JSON block:
```json
{{
  "summary": "Brief synthesis of reviewer opinions",
  "consensus": "agreement/disagreement/mixed",
  "key_strengths": "Main strengths identified",
  "key_weaknesses": "Main weaknesses identified",
  "decision": "accept" or "reject"
}}
```"""


# ============================================================================
# DATASET CREATION
# ============================================================================

def load_predictions(predictions_path: Path) -> list[dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(predictions_path) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def create_metareview_dataset(predictions: list[dict]) -> list[dict]:
    """Create metareview dataset from ensemble predictions.

    Args:
        predictions: List of prediction entries with 'all_predictions' field

    Returns:
        List of metareview entries
    """
    metareview_data = []

    for pred in predictions:
        # Get all predictions (ensemble outputs)
        all_preds = pred.get("all_predictions", [pred.get("predict", "")])

        if not all_preds or len(all_preds) < 2:
            continue

        # Format reviews
        reviews_text = ""
        for i, review in enumerate(all_preds, 1):
            reviews_text += f"\n--- Reviewer {i} ---\n{review}\n"

        # Create metareview prompt
        user_message = METAREVIEW_USER_TEMPLATE.format(
            n_reviews=len(all_preds),
            reviews=reviews_text
        )

        metareview_data.append({
            "system": METAREVIEW_SYSTEM_PROMPT,
            "human": user_message,
            "label": pred.get("label", ""),
            "original_prompt": pred.get("prompt", "")[:500],  # Truncated for reference
        })

    return metareview_data


def create_batch_request(entry: dict, temperature: float = 0.3, max_tokens: int = 2048) -> dict:
    """Create a batch request for metareview."""
    request = {
        "contents": [
            {"role": "user", "parts": [{"text": entry["human"]}]}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }

    if entry.get("system"):
        request["systemInstruction"] = {
            "parts": [{"text": entry["system"]}]
        }

    return {"request": request}


def create_batch_jsonl(data: list[dict], output_path: str, temperature: float = 0.3) -> tuple[str, list[dict]]:
    """Create JSONL file for batch processing."""
    print(f"Creating metareview batch JSONL: {output_path}")

    metadata_list = []

    with open(output_path, "w") as f:
        for idx, entry in enumerate(data):
            request = create_batch_request(entry, temperature)
            f.write(json.dumps(request) + "\n")

            metadata_list.append({
                "idx": idx,
                "label": entry.get("label", ""),
                "human_message": entry["human"],
            })

    print(f"  Created {len(data)} requests")
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


def submit_batch_job(client: genai.Client, model: str, input_uri: str, output_uri: str, display_name: str) -> any:
    """Submit a batch prediction job."""
    print(f"\nSubmitting metareview batch job...")
    print(f"  Model: {model}")
    print(f"  Input: {input_uri}")

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
    """Check job status."""
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
    """Extract prediction text from response."""
    try:
        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text", "")
    except (KeyError, IndexError, TypeError):
        pass
    return ""


def process_batch_results(output_uri: str, metadata_list: list[dict], output_path: str, project: str) -> int:
    """Process batch results."""
    print(f"\nProcessing metareview results from: {output_uri}")

    result_files = list_gcs_files(output_uri, project)
    result_files = [f for f in result_files if f.endswith(".jsonl")]

    if not result_files:
        print("  No result files found!")
        return 0

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
            }
            f.write(json.dumps(output_entry) + "\n")
            processed += 1

    print(f"  Saved {processed} results to: {output_path}")
    return processed


# ============================================================================
# COMMANDS
# ============================================================================

def cmd_submit(args):
    """Submit metareview batch jobs."""
    results_dir = Path(args.results_dir)
    client = get_client(args.project, args.location)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    modalities = [args.modality] if args.modality else MODALITIES

    for modality in modalities:
        # Check for new_fewshot predictions (which have ensemble outputs)
        pred_path = results_dir / modality / "new_fewshot" / "predictions.jsonl"

        if not pred_path.exists():
            print(f"Skipping {modality}: no predictions.jsonl found")
            continue

        print(f"\n{'='*70}")
        print(f"Creating metareview for: {modality}")
        print(f"{'='*70}")

        # Load predictions
        predictions = load_predictions(pred_path)
        print(f"  Loaded {len(predictions)} predictions")

        # Create metareview dataset
        metareview_data = create_metareview_dataset(predictions)
        print(f"  Created {len(metareview_data)} metareview entries")

        if not metareview_data:
            print(f"  No valid ensemble predictions, skipping")
            continue

        # Create batch JSONL
        local_jsonl = f"/tmp/metareview_{modality}_{timestamp}.jsonl"
        _, metadata_list = create_batch_jsonl(metareview_data, local_jsonl, args.temperature)

        # Upload to GCS
        input_uri = f"{args.gcs_staging}/metareview_{modality}_{timestamp}_input.jsonl"
        output_uri = f"{args.gcs_staging}/metareview_{modality}_{timestamp}_output/"
        upload_to_gcs(local_jsonl, input_uri, args.project)

        # Submit job
        display_name = f"metareview_{modality}_{timestamp}"
        job = submit_batch_job(client, args.model, input_uri, output_uri, display_name)

        # Save job info
        job_info = {
            "modality": modality,
            "job_name": job.name,
            "model": args.model,
            "num_samples": len(metareview_data),
            "input_uri": input_uri,
            "output_uri": output_uri,
            "submitted_at": timestamp,
        }

        result_dir = results_dir / modality / "new_fewshot"
        job_info_path = result_dir / "gemini_metareview_job_info.json"
        with open(job_info_path, "w") as f:
            json.dump(job_info, f, indent=2)

        metadata_path = result_dir / "gemini_metareview_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata_list, f)

        print(f"  Job info saved: {job_info_path}")

        os.remove(local_jsonl)

    print(f"\n{'='*70}")
    print("Metareview jobs submitted!")
    print(f"{'='*70}")


def cmd_status(args):
    """Check status of metareview jobs."""
    results_dir = Path(args.results_dir)
    client = get_client(args.project, args.location)

    print(f"\n{'='*70}")
    print("Metareview Job Status")
    print(f"{'='*70}\n")

    for modality in MODALITIES:
        job_info_path = results_dir / modality / "new_fewshot" / "gemini_metareview_job_info.json"

        if not job_info_path.exists():
            continue

        with open(job_info_path) as f:
            job_info = json.load(f)

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

        print(f"{marker} {modality}/metareview: {state}")


def cmd_retrieve(args):
    """Retrieve metareview results."""
    results_dir = Path(args.results_dir)
    client = get_client(args.project, args.location)

    modalities = [args.modality] if args.modality else MODALITIES

    for modality in modalities:
        result_dir = results_dir / modality / "new_fewshot"
        job_info_path = result_dir / "gemini_metareview_job_info.json"
        metadata_path = result_dir / "gemini_metareview_metadata.json"

        if not job_info_path.exists():
            continue

        print(f"\n{'='*70}")
        print(f"Retrieving metareview: {modality}")
        print(f"{'='*70}")

        with open(job_info_path) as f:
            job_info = json.load(f)

        status = check_job_status(client, job_info["job_name"])
        print(f"  Status: {status['state']}")

        if not status["completed"]:
            print(f"  Job not completed, skipping")
            continue

        with open(metadata_path) as f:
            metadata_list = json.load(f)

        output_path = result_dir / "metareview_predictions.jsonl"
        num_processed = process_batch_results(
            job_info["output_uri"],
            metadata_list,
            str(output_path),
            args.project,
        )

        job_info["status"] = "retrieved"
        job_info["retrieved_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_info["num_results"] = num_processed
        with open(job_info_path, "w") as f:
            json.dump(job_info, f, indent=2)

        print(f"  Saved {num_processed} results")


def main():
    parser = argparse.ArgumentParser(description="Gemini Metareview for Inference Scaling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit metareview jobs")
    submit_parser.add_argument("--results_dir", type=str, default="inference_scaling/results/gemini")
    submit_parser.add_argument("--modality", type=str, choices=MODALITIES)
    submit_parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    submit_parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    submit_parser.add_argument("--gcs_staging", type=str, default=DEFAULT_GCS_STAGING)
    submit_parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    submit_parser.add_argument("--temperature", type=float, default=0.3)

    # Status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("--results_dir", type=str, default="inference_scaling/results/gemini")
    status_parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    status_parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve results")
    retrieve_parser.add_argument("--results_dir", type=str, default="inference_scaling/results/gemini")
    retrieve_parser.add_argument("--modality", type=str, choices=MODALITIES)
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
