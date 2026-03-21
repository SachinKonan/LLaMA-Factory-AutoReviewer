#!/usr/bin/env python3
"""
Gemini Batch API - Retrieve results from previously submitted batch jobs.

Usage:
    # Scan for all jobs in results_dir and check status
    python final_inference_scaling/gemini_batch_retrieve.py --results_dir final_inference_scaling/results --status_only

    # Retrieve all completed jobs found in results_dir
    python final_inference_scaling/gemini_batch_retrieve.py --results_dir final_inference_scaling/results --project YOUR_PROJECT

    # Retrieve specific job
    python final_inference_scaling/gemini_batch_retrieve.py --job_name projects/... --output predictions.jsonl --project YOUR_PROJECT

in practice, completed:
python /scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/final_inference_scaling/gemini_batch_retrieve.py --results_dir final_inference_scaling/results
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime

from google import genai
from google.genai.types import HttpOptions


DEFAULT_LOCATION = "us-central1"


def get_client(project: str, location: str) -> genai.Client:
    """Get Gemini client for Vertex AI."""
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=HttpOptions(api_version="v1")
    )


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


def parse_prediction(response: dict) -> tuple[list[str], list[str]]:
    """Extract prediction text and finish reasons from batch response.
    
    Returns (predictions, finish_reasons).
    """
    predictions = []
    finish_reasons = []
    try:
        candidates = response.get("candidates", [])
        for candidate in candidates:
            # Get text
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            text = ""
            for part in parts:
                if "text" in part:
                    text += part.get("text", "")
            predictions.append(text)
            
            # Get finish reason
            # Note: Might be "finishReason" (Vertex) or "finish_reason" (SDK)
            reason = candidate.get("finishReason") or candidate.get("finish_reason", "UNKNOWN")
            finish_reasons.append(str(reason))
    except (KeyError, IndexError, TypeError):
        pass
        
    if not predictions:
        predictions = [""]
        finish_reasons = ["ERROR"]
        
    return predictions, finish_reasons


def process_batch_results(
    output_uri: str,
    metadata_list: list[dict],
    output_path: str,
    project: str,
) -> int:
    """Process batch results and save in vllm_infer.py format."""
    print(f"  Retrieving results from: {output_uri}")

    # List result files
    try:
        result_files = list_gcs_files(output_uri, project)
    except Exception as e:
        print(f"    Error listing GCS files: {e}")
        return 0
        
    result_files = [f for f in result_files if f.endswith(".jsonl")]

    if not result_files:
        print("    No result files found in GCS output directory.")
        return 0

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

    # Match results with metadata and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed = 0
    reason_counts = {}
    
    with open(output_path, "w") as f:
        for idx, result in enumerate(all_results):
            if idx >= len(metadata_list):
                break

            meta = metadata_list[idx]
            response = result.get("response", {})
            predictions, finish_reasons = parse_prediction(response)

            # Record reason counts for summary
            primary_reason = finish_reasons[0] if finish_reasons else "UNKNOWN"
            reason_counts[primary_reason] = reason_counts.get(primary_reason, 0) + 1

            output_entry = {
                "prompt": meta["human_message"],
                "predict": predictions[0] if predictions else "",
                "label": meta["label"],
                "finish_reason": primary_reason,
            }
            if len(predictions) > 1:
                output_entry["all_predictions"] = predictions
                output_entry["all_finish_reasons"] = finish_reasons
                
            f.write(json.dumps(output_entry) + "\n")
            processed += 1

    print(f"    Saved {processed} results to: {output_path}")
    print(f"    Finish Reasons breakdown:")
    for reason, count in reason_counts.items():
        print(f"      - {reason}: {count}")
        
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve results from Gemini batch jobs"
    )

    # Bulk processing
    parser.add_argument("--results_dir", type=str, help="Root results directory to scan for jobs")
    
    # Specific job
    parser.add_argument("--job_name", type=str, help="Specific batch job name")
    parser.add_argument("--output", type=str, help="Specific output JSONL path")
    parser.add_argument("--metadata_path", type=str, help="Specific metadata JSON path")

    # Global config
    parser.add_argument("--project", type=str, help="GCP project ID")
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    
    # Flags
    parser.add_argument("--status_only", action="store_true", help="Only check job status, don't retrieve")
    parser.add_argument("--force", action="store_true", help="Re-retrieve even if already done")

    args = parser.parse_args()

    if not args.results_dir and not args.job_name:
        parser.error("Either --results_dir or --job_name must be provided")

    client = get_client(args.project, args.location) if args.project else None

    # Case 1: Scanning directory structure
    if args.results_dir:
        results_path = Path(args.results_dir)
        job_info_files = list(results_path.glob("**/gemini_job_info.json"))
        
        if not job_info_files:
            print(f"No gemini_job_info.json files found in {args.results_dir}")
            return

        print(f"Found {len(job_info_files)} Gemini jobs. Checking status...")
        
        for job_info_path in sorted(job_info_files):
            with open(job_info_path) as f:
                job_info = json.load(f)
            
            job_name = job_info["job_name"]
            modality = job_info.get("modality", "unknown")
            variant = job_info.get("variant", "unknown")
            model = job_info.get("model", "unknown")
            
            # Use project from job_info if not provided on CLI
            project = args.project or job_name.split("/")[1]
            if not client:
                client = get_client(project, args.location)

            print(f"\nJob: {modality}/{variant} ({model})")
            print(f"  Name: {job_name}")
            
            try:
                job = client.batches.get(name=job_name)
                state = str(job.state)
                print(f"  State: {state}")
                
                if args.status_only:
                    continue
                    
                if "SUCCEEDED" in state or state == "JOB_STATE_SUCCEEDED":
                    if job_info.get("status") == "completed" and not args.force:
                        print(f"  Already retrieved. Use --force to re-retrieve.")
                        continue
                        
                    # Find metadata file (should be in same dir)
                    metadata_path = job_info_path.parent / "gemini_metadata.json"
                    if not metadata_path.exists():
                        print(f"  Error: Metadata file not found at {metadata_path}")
                        continue
                        
                    with open(metadata_path) as f:
                        metadata_list = json.load(f)
                    
                    output_path = job_info.get("predictions_file") or str(job_info_path.parent / "predictions.jsonl")
                    
                    num_processed = process_batch_results(
                        job_info["output_uri"], metadata_list, str(output_path), project
                    )
                    
                    if num_processed > 0:
                        job_info["status"] = "completed"
                        job_info["num_results"] = num_processed
                        job_info["retrieved_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                        with open(job_info_path, "w") as f:
                            json.dump(job_info, f, indent=2)
                else:
                    print(f"  Job not ready (State: {state})")
            except Exception as e:
                print(f"  Error checking/retrieving job: {e}")

    # Case 2: Specific job retrieval
    elif args.job_name:
        if not args.output or not args.project:
            parser.error("--output and --project are required with --job_name")
            
        metadata_path = args.metadata_path or args.output.replace(".jsonl", "_metadata.json")
        if not os.path.exists(metadata_path):
            parser.error(f"Metadata file not found: {metadata_path}. Provide --metadata_path")
            
        with open(metadata_path) as f:
            metadata_list = json.load(f)
            
        job = client.batches.get(name=args.job_name)
        state = str(job.state)
        print(f"Job: {args.job_name}")
        print(f"  State: {state}")
        
        if args.status_only:
            return
            
        if "SUCCEEDED" in state or state == "JOB_STATE_SUCCEEDED":
            # We need the output URI. If it's not in a job_info file, we might have to scrape it 
            # or it might be in the job object. Vertex AI BatchPredictionJob has an 'output_info' usually?
            # Actually GenAI SDK Batch object has 'config.dest'.
            output_uri = getattr(job.config, 'dest', None)
            if not output_uri:
                print("Error: Could not determine output URI from job object.")
                return
                
            process_batch_results(output_uri, metadata_list, args.output, args.project)
        else:
            print(f"Job not ready (State: {state})")

if __name__ == "__main__":
    main()