#!/usr/bin/env python3
"""
Gemini Batch API - Submit batch jobs for ICLR review prediction using Vertex AI.

Updated to work with the 13 prompt variants × 2 modalities structure used in the
final_inference_scaling pipeline.

Usage:
    # Text-only inference, standard no-fewshot variant
    python final_inference_scaling/gemini_batch_submit.py \
        --modality text \
        --variant standard_nofewshot_boxed \
        --project YOUR_PROJECT \
        --output final_inference_scaling/results/gemini/text/standard_nofewshot_boxed/predictions.jsonl

    # Dry run (test with 10 samples, doesn't actually submit)
    python final_inference_scaling/gemini_batch_submit.py \
        --modality vision \
        --variant critical_fewshot_1-1_json \
        --limit 10 \
        --dry_run \
        --project YOUR_PROJECT
        
    # Filter by specific paper IDs
    python final_inference_scaling/gemini_batch_submit.py \
        --modality text \
        --variant pdr_boxed \
        --ids_file selective_papers.txt \
        --project YOUR_PROJECT

        Actually submitted:                                                                                                             │
│  34 - - default model is gemini-2.5-flash.                                                                                            │
│  35 - python final_inference_scaling/gemini_batch_submit.py     --modality all     --variant                                          │
│     standard_nofewshot_boxed,standard_nofewshot_json --ids_file                                                                       │
│     /scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/iclr_2020_2023_2025_2026_85_5_10_balanced_original_v7_filtered_test_500_s │
│     ubset.txt --max_tokens 5000 --thinking_budget 2000 --submit_only                                                                  │
│  36 - python final_inference_scaling/gemini_batch_submit.py     --modality all     --variant                                          │
│     critical_nofewshot_boxed,critical_nofewshot_json --ids_file                                                                       │
│     /scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/iclr_2020_2023_2025_2026_85_5_10_balanced_original_v7_filtered_test_500_s │
│     ubset.txt --max_tokens 5000 --thinking_budget 2000 --submit_only                                                                  │
│  37 - python final_inference_scaling/gemini_batch_submit.py \                                                                         │
│  38 -         --modality all     --variant \                                                                                          │
│  39 -         critical_fewshot_1-1_boxed,critical_fewshot_0-2_boxed,standard_fewshot_1-1_boxed,standard_fewshot_0-2_boxed  \          │
│  40 -         --ids_file                                                                                                              │
│     /scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/iclr_2020_2023_2025_2026_85_5_10_balanced_original_v7_filtered_test_500_s │
│     ubset.txt \                                                                                                                       │
│  41 -         --max_tokens 5000 --thinking_budget 2000 --submit_only                                                                  │
│  42 -                                                                                                                                 │
│  43 - 5x, prior to metareview:                                                                                                        │
│  44 - python final_inference_scaling/gemini_batch_submit.py     --modality all     --variant                                          │
│     standard_nofewshot_boxed,critical_fewshot_0-2_boxed,standard_fewshot_0-2_boxed --ids_file                                         │
│     /scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/iclr_2020_2023_2025_2026_85_5_10_balanced_original_v7_filtered_test_500_s │
│     ubset.txt --max_tokens 5000 --thinking_budget 2000 --samples 5 --submit_only --limit 10 --temp 1.0


5x (correct):
python final_inference_scaling/gemini_batch_submit.py     --modality all     --variant standard_nofewshot_boxed,critical_fewshot_0-2_boxed,standard_fewshot_0-2_boxed     --ids_file /scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/iclr_2020_2023_2025_2026_85_5_10_balanced_original_v7_filtered_test_500_subset.txt     --max_tokens 5000     --thinking_budget 2000     --samples 5     --submit_only     --limit 10     --temperature 1.0


"""

#!/usr/bin/env python3
"""
Gemini Batch API - Submit batch jobs for ICLR review prediction using Vertex AI.

Updated to work with the 13 prompt variants x 2 modalities structure used in the
final_inference_scaling pipeline.

Usage:
    # Text-only inference, standard no-fewshot variant
    python final_inference_scaling/gemini_batch_submit.py \
        --modality text \
        --variant standard_nofewshot_boxed \
        --project YOUR_PROJECT \
        --output final_inference_scaling/results/gemini/text/standard_nofewshot_boxed/predictions.jsonl

    # Dry run (test with 10 samples, doesn't actually submit)
    python final_inference_scaling/gemini_batch_submit.py \
        --modality vision \
        --variant critical_fewshot_1-1_json \
        --limit 10 \
        --dry_run \
        --project YOUR_PROJECT
        
    # Filter by specific paper IDs
    python final_inference_scaling/gemini_batch_submit.py \
        --modality text \
        --variant pdr_boxed \
        --ids_file selective_papers.txt \
        --project YOUR_PROJECT

"""

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_GCS_IMAGES_BASE = "gs://jl0796-autoreviewer-staging/data/images"
LOCAL_IMAGES_PREFIX = "/scratch/gpfs/ZHUANGL/jl0796/shared/data/images/"
DEFAULT_MODEL = "gemini-2.5-flash" # Updated to newest
DEFAULT_LOCATION = "us-central1"
DEFAULT_GCS_STAGING = "gs://jl0796-autoreviewer-staging/inference_scaling"

BASE_DATASETS = {
    "text": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered",
    "vision": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480",
}

PROMPT_VARIANTS = [
    "standard_nofewshot_boxed", "standard_nofewshot_json",
    "standard_fewshot_1-1_boxed", "standard_fewshot_1-1_json",
    "standard_fewshot_0-2_boxed", "standard_fewshot_0-2_json",
    "critical_nofewshot_boxed", "critical_nofewshot_json",
    "critical_fewshot_1-1_boxed", "critical_fewshot_1-1_json",
    "critical_fewshot_0-2_boxed", "critical_fewshot_0-2_json",
    "pdr_boxed",
]


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(data_dir: Path, modality: str, variant: str) -> list[dict]:
    """Load dataset generated by generate_datasets.py."""
    if modality not in BASE_DATASETS:
        raise ValueError(f"Invalid modality: {modality}. Must be one of {list(BASE_DATASETS.keys())}")
    
    base_name = BASE_DATASETS[modality]
    dataset_name = f"{base_name}_test_{variant}"
    dataset_path = data_dir / dataset_name / "data.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at expected path: {dataset_path}")

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} samples")
    return data


def filter_by_ids(data: list[dict], ids_file: str) -> list[dict]:
    """Filter dataset to only include submission IDs from the provided file."""
    if not os.path.exists(ids_file):
        print(f"Warning: IDs file not found: {ids_file}. No filtering applied.")
        return data

    with open(ids_file, "r") as f:
        target_ids = set(line.strip() for line in f if line.strip())

    filtered_data = []
    for entry in data:
        sub_id = entry.get("_metadata", {}).get("submission_id")
        if sub_id in target_ids:
            filtered_data.append(entry)
    
    print(f"  Filtered: {len(data)} -> {len(filtered_data)} samples using {ids_file}")
    return filtered_data


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
    
    if local_path.startswith("data/images/"):
        relative_path = local_path[len("data/images/"):]
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
    segments = re.split(r"<image>", human_message)
    image_idx = 0

    for i, segment in enumerate(segments):
        if segment.strip():
            parts.append({"text": segment})

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

    if not parts:
        parts.append({"text": human_message})

    return parts


def create_batch_request(
    entry: dict,
    gcs_base: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    thinking_budget: int = 4000,
) -> dict:
    """Create a batch request for a single entry."""
    messages = parse_conversations(entry)
    images = entry.get("images", [])
    parts = build_content_parts(messages["human"] or "", images, gcs_base)

    request = {
        "contents": [
            {"role": "user", "parts": parts}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
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


def get_request_key(req: dict) -> str:
    """Extract a stable SHA-256 key from a request dict to match shuffled responses to metadata."""
    parts_str = []
    try:
        if "systemInstruction" in req:
            for part in req["systemInstruction"].get("parts", []):
                if "text" in part:
                    parts_str.append(part["text"])
        for content in req.get("contents", []):
            for part in content.get("parts", []):
                if "text" in part:
                    parts_str.append(part["text"])
                elif "fileData" in part:
                    parts_str.append(part["fileData"].get("fileUri", ""))
    except Exception:
        pass
    return hashlib.sha256("".join(parts_str).encode('utf-8')).hexdigest()


def create_batch_jsonl(
    data: list[dict],
    output_path: str,
    gcs_base: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    thinking_budget: int = 2000,
    samples: int = 1,
) -> tuple[str, list[dict]]:
    """Create JSONL file for batch processing."""
    print(f"Creating batch JSONL: {output_path}")

    metadata_list = []
    with open(output_path, "w") as f:
        global_idx = 0
        for s_idx in range(samples):
            for idx, entry in enumerate(data):
                request = create_batch_request(
                    entry, gcs_base, temperature, max_tokens, thinking_budget
                )
                f.write(json.dumps(request) + "\n")

                messages = parse_conversations(entry)
                req_key = get_request_key(request["request"])

                metadata_list.append({
                    "req_key": req_key,
                    "idx": global_idx,
                    "sample_idx": s_idx,
                    "original_idx": idx,
                    "submission_id": entry.get("_metadata", {}).get("submission_id", ""),
                    "label": messages["gpt"] or "",
                    "ground_truth": entry.get("_metadata", {}).get("answer", ""),
                    "human_message": messages["human"] or "",
                })
                global_idx += 1

    file_size = os.path.getsize(output_path)
    print(f"  Created {global_idx} requests ({samples} samples x {len(data)} papers), file size: {file_size / 1024 / 1024:.2f} MB")

    return output_path, metadata_list


# ============================================================================
# BATCH JOB MANAGEMENT
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
        display_name = f"autoreviewer_{timestamp}"

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

    print(f"  Job name: {job.name}")
    print(f"  State: {job.state}")
    return job


def poll_until_complete(
    client: genai.Client,
    job_name: str,
    interval: int = 300,
    max_wait: int = 86400,
) -> any:
    """Poll job until completion."""
    print(f"\nPolling job status (interval: {interval}s)...")
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise RuntimeError(f"Job timed out after {elapsed/3600:.1f} hours")
        job = client.batches.get(name=job_name)
        state = str(job.state)
        elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m"
        print(f"  [{elapsed_str}] State: {state}")
        if "SUCCEEDED" in state or state == "JOB_STATE_SUCCEEDED":
            print(f"\n  Job completed successfully!")
            return job
        if "FAILED" in state or "CANCELLED" in state:
            raise RuntimeError(f"Batch job failed or cancelled: {state}")
        time.sleep(interval)


# ============================================================================
# RESULT PROCESSING
# ============================================================================

def parse_prediction(response: dict) -> str:
    """Extract prediction text from batch response."""
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


def process_batch_results(
    output_uri: str,
    metadata_list: list[dict],
    output_path: str,
    project: str,
) -> int:
    """Process batch results and save in vllm_infer.py format."""
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
    
    # Map metadata by the uniquely hashed request payload
    meta_map = defaultdict(list)
    for meta in metadata_list:
        meta_map[meta["req_key"]].append(meta)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed = 0
    
    with open(output_path, "w") as f:
        for result in all_results:
            req = result.get("request", {})
            req_key = get_request_key(req)
            
            # Map API returned request safely back to original request context regardless of shuffling
            if req_key in meta_map and len(meta_map[req_key]) > 0:
                meta = meta_map[req_key].pop(0)
            else:
                print(f"  Warning: Could not match a result back to metadata. Skipped.")
                continue

            response = result.get("response", {})
            prediction = parse_prediction(response)
            
            output_entry = {
                "prompt": meta["human_message"],
                "predict": prediction,
                "label": meta["label"],
                "submission_id": meta["submission_id"],
                "sample_idx": meta.get("sample_idx", 0),
            }
            f.write(json.dumps(output_entry) + "\n")
            processed += 1
            
    print(f"  Saved {processed} results to: {output_path}")
    return processed


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gemini Batch API for Inference Scaling Experiments"
    )
    parser.add_argument("--modality", type=str, help="Modality (text, vision, or comma-separated list, or 'all')")
    parser.add_argument("--variant", type=str, help="Prompt variant (e.g., standard_nofewshot_boxed, or comma-separated list, or 'all')")
    parser.add_argument("--dataset_path", type=str, help="Direct path to a data.json dataset (bypasses modality/variant lookup)")
    parser.add_argument("--data_dir", type=str, default="final_inference_scaling/data")
    parser.add_argument("--output_dir", type=str, default="final_inference_scaling/results")
    parser.add_argument("--gcs_staging", type=str, default=DEFAULT_GCS_STAGING)
    parser.add_argument("--gcs_base", type=str, default=DEFAULT_GCS_IMAGES_BASE)
    parser.add_argument("--limit", type=int, help="Limit to first N samples")
    parser.add_argument("--ids_file", type=str, help="Text file with submission_ids to include")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--project", type=str, default="hip-gecko-485003-c4", help="GCP project ID")
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=1, help="Number of samples per paper")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--thinking_budget", type=int, default=2000, help="0 to disable thinking")
    parser.add_argument("--dry_run", action="store_true", help="Create JSONL locally but don't submit")
    parser.add_argument("--submit_only", action="store_true", help="Submit and exit without polling")
    parser.add_argument("--poll_interval", type=int, default=300)

    args = parser.parse_args()

    # --dataset_path mode: single custom dataset, skip modality/variant validation
    if args.dataset_path:
        if not args.modality:
            args.modality = "custom"
        if not args.variant:
            args.variant = Path(args.dataset_path).parent.name
        modalities = [args.modality]
        variants = [args.variant]
    else:
        if args.modality == "all":
            modalities = list(BASE_DATASETS.keys())
        elif args.modality:
            modalities = [m.strip() for m in args.modality.split(",")]
        else:
            parser.error("--modality is required")

        if args.variant == "all":
            variants = PROMPT_VARIANTS
        elif args.variant:
            variants = [v.strip() for v in args.variant.split(",")]
        else:
            parser.error("--variant is required")

        for m in modalities:
            if m not in BASE_DATASETS:
                parser.error(f"Invalid modality: {m}. Choices: {list(BASE_DATASETS.keys())}")
        for v in variants:
            if v not in PROMPT_VARIANTS:
                parser.error(f"Invalid variant: {v}. Choices: {PROMPT_VARIANTS}")

    if args.gcs_staging is None:
        args.gcs_staging = DEFAULT_GCS_STAGING

    for modality in modalities:
        for variant in variants:
            print(f"\n{'='*70}")
            print(f"PROCESS: {modality} / {variant}")
            print(f"{'='*70}")

            try:
                if args.dataset_path:
                    dataset_path = Path(args.dataset_path)
                    if not dataset_path.exists():
                        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
                    print(f"Loading custom dataset: {dataset_path}")
                    with open(dataset_path) as f:
                        data = json.load(f)
                    print(f"  Loaded {len(data)} samples")
                else:
                    data_dir = Path(args.data_dir)
                    data = load_dataset(data_dir, modality, variant)
                if args.ids_file:
                    data = filter_by_ids(data, args.ids_file)
                if args.limit:
                    data = data[:args.limit]
                    print(f"  Limited to first {len(data)} samples")
                if not data:
                    print(f"Skipping {modality}/{variant}: No samples after filtering.")
                    continue

                cutoff_len = 60000 if "fewshot" in variant and "nofewshot" not in variant else 24480
                model_name_clean = args.model.replace("/", "--")
                gen_suffix = "gen1" if args.samples == 1 else f"gen{args.samples}"
                job_results_dir = Path(args.output_dir) / model_name_clean / modality / f"{variant}_ctx{cutoff_len}_{gen_suffix}"
                job_results_dir.mkdir(parents=True, exist_ok=True)
                
                predictions_path = job_results_dir / "predictions.jsonl"
                metadata_path = job_results_dir / "gemini_metadata.json"
                job_info_path = job_results_dir / "gemini_job_info.json"

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                local_jsonl = f"/tmp/gemini_batch_{modality}_{variant}_{timestamp}.jsonl"
                _, metadata_list = create_batch_jsonl(
                    data, local_jsonl, args.gcs_base,
                    args.temperature, args.max_tokens, args.thinking_budget,
                    samples=args.samples
                )

                with open(metadata_path, "w") as f:
                    json.dump(metadata_list, f)
                print(f"Metadata saved to: {metadata_path}")

                if args.dry_run:
                    print(f"DRY RUN complete for {modality}/{variant}.")
                    if os.path.exists(local_jsonl):
                        os.remove(local_jsonl)
                    continue

                gcs_input_uri = f"{args.gcs_staging.rstrip('/')}/{modality}_{variant}_{timestamp}_input.jsonl"
                gcs_output_uri = f"{args.gcs_staging.rstrip('/')}/{modality}_{variant}_{timestamp}_output/"
                upload_to_gcs(local_jsonl, gcs_input_uri, args.project)

                client = get_client(args.project, args.location)
                display_name = f"{modality}_{variant}_{timestamp}"
                job = submit_batch_job(client, args.model, gcs_input_uri, gcs_output_uri, display_name)

                job_info = {
                    "job_name": job.name,
                    "model": args.model,
                    "modality": modality,
                    "variant": variant,
                    "num_samples": len(data),
                    "samples": args.samples,
                    "input_uri": gcs_input_uri,
                    "output_uri": gcs_output_uri,
                    "submitted_at": timestamp,
                    "predictions_file": str(predictions_path),
                }
                with open(job_info_path, "w") as f:
                    json.dump(job_info, f, indent=2)
                print(f"Job info saved to: {job_info_path}")

                if args.submit_only:
                    print(f"Submitted {modality}/{variant}. Moving to next.")
                else:
                    poll_until_complete(client, job.name, interval=args.poll_interval)
                    process_batch_results(gcs_output_uri, metadata_list, str(predictions_path), args.project)
                    job_info["status"] = "completed"
                    with open(job_info_path, "w") as f:
                        json.dump(job_info, f, indent=2)
                    print(f"Completed {modality}/{variant}.")

                if os.path.exists(local_jsonl):
                    os.remove(local_jsonl)

            except Exception as e:
                print(f"Error processing {modality}/{variant}: {e}")
                continue

    print("\n" + "=" * 70)
    print("ALL COMBINATIONS PROCESSED.")
    print("=" * 70)

if __name__ == "__main__":
    main()


# import argparse
# import json
# import os
# import re
# import tempfile
# import time
# from datetime import datetime
# from pathlib import Path

# from google import genai
# from google.genai.types import CreateBatchJobConfig, HttpOptions


# # ============================================================================
# # CONSTANTS
# # ============================================================================

# DEFAULT_GCS_IMAGES_BASE = "gs://jl0796-autoreviewer-staging/data/images"
# LOCAL_IMAGES_PREFIX = "/scratch/gpfs/ZHUANGL/jl0796/shared/data/images/"
# DEFAULT_MODEL = "gemini-2.5-flash" # Updated to newest
# DEFAULT_LOCATION = "us-central1"
# DEFAULT_GCS_STAGING = "gs://jl0796-autoreviewer-staging/inference_scaling"

# BASE_DATASETS = {
#     "text": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered",
#     "vision": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480",
# }

# PROMPT_VARIANTS = [
#     "standard_nofewshot_boxed", "standard_nofewshot_json",
#     "standard_fewshot_1-1_boxed", "standard_fewshot_1-1_json",
#     "standard_fewshot_0-2_boxed", "standard_fewshot_0-2_json",
#     "critical_nofewshot_boxed", "critical_nofewshot_json",
#     "critical_fewshot_1-1_boxed", "critical_fewshot_1-1_json",
#     "critical_fewshot_0-2_boxed", "critical_fewshot_0-2_json",
#     "pdr_boxed",
# ]


# # ============================================================================
# # DATASET LOADING
# # ============================================================================

# def load_dataset(data_dir: Path, modality: str, variant: str) -> list[dict]:
#     """Load dataset generated by generate_datasets.py."""
#     if modality not in BASE_DATASETS:
#         raise ValueError(f"Invalid modality: {modality}. Must be one of {list(BASE_DATASETS.keys())}")
    
#     base_name = BASE_DATASETS[modality]
#     dataset_name = f"{base_name}_test_{variant}"
#     dataset_path = data_dir / dataset_name / "data.json"

#     if not dataset_path.exists():
#         raise FileNotFoundError(f"Dataset not found at expected path: {dataset_path}")

#     print(f"Loading dataset: {dataset_path}")
#     with open(dataset_path) as f:
#         data = json.load(f)

#     print(f"  Loaded {len(data)} samples")
#     return data


# def filter_by_ids(data: list[dict], ids_file: str) -> list[dict]:
#     """Filter dataset to only include submission IDs from the provided file."""
#     if not os.path.exists(ids_file):
#         print(f"Warning: IDs file not found: {ids_file}. No filtering applied.")
#         return data

#     with open(ids_file, "r") as f:
#         target_ids = set(line.strip() for line in f if line.strip())

#     filtered_data = []
#     for entry in data:
#         sub_id = entry.get("_metadata", {}).get("submission_id")
#         if sub_id in target_ids:
#             filtered_data.append(entry)
    
#     print(f"  Filtered: {len(data)} -> {len(filtered_data)} samples using {ids_file}")
#     return filtered_data


# def parse_conversations(entry: dict) -> dict:
#     """Parse conversations to extract system, human, and gpt messages."""
#     result = {"system": None, "human": None, "gpt": None}

#     for msg in entry.get("conversations", []):
#         role = msg.get("from", "")
#         value = msg.get("value", "")

#         if role == "system":
#             result["system"] = value
#         elif role == "human":
#             result["human"] = value
#         elif role == "gpt":
#             result["gpt"] = value

#     return result


# def convert_image_path_to_gcs(local_path: str, gcs_base: str) -> str:
#     """Convert local image path to GCS URI."""
#     if local_path.startswith(LOCAL_IMAGES_PREFIX):
#         relative_path = local_path[len(LOCAL_IMAGES_PREFIX):]
#         return f"{gcs_base.rstrip('/')}/{relative_path}"
    
#     if local_path.startswith("data/images/"):
#         relative_path = local_path[len("data/images/"):]
#         return f"{gcs_base.rstrip('/')}/{relative_path}"
        
#     return local_path


# def get_mime_type(path: str) -> str:
#     """Get MIME type from file extension."""
#     ext = path.lower().split(".")[-1]
#     mime_types = {
#         "png": "image/png",
#         "jpg": "image/jpeg",
#         "jpeg": "image/jpeg",
#         "webp": "image/webp",
#         "gif": "image/gif",
#     }
#     return mime_types.get(ext, "image/png")


# # ============================================================================
# # BATCH REQUEST CREATION
# # ============================================================================

# def build_content_parts(human_message: str, images: list[str], gcs_base: str) -> list[dict]:
#     """Build content parts array with text and images."""
#     parts = []
#     segments = re.split(r"<image>", human_message)
#     image_idx = 0

#     for i, segment in enumerate(segments):
#         if segment.strip():
#             parts.append({"text": segment})

#         if i < len(segments) - 1 and image_idx < len(images):
#             gcs_uri = convert_image_path_to_gcs(images[image_idx], gcs_base)
#             mime_type = get_mime_type(images[image_idx])
#             parts.append({
#                 "fileData": {
#                     "fileUri": gcs_uri,
#                     "mimeType": mime_type
#                 }
#             })
#             image_idx += 1

#     if not parts:
#         parts.append({"text": human_message})

#     return parts


# def create_batch_request(
#     entry: dict,
#     gcs_base: str,
#     temperature: float = 0.0,
#     max_tokens: int = 2000,
#     thinking_budget: int = 4000,
# ) -> dict:
#     """Create a batch request for a single entry."""
#     messages = parse_conversations(entry)
#     images = entry.get("images", [])
#     parts = build_content_parts(messages["human"] or "", images, gcs_base)

#     request = {
#         "contents": [
#             {"role": "user", "parts": parts}
#         ],
#         "generationConfig": {
#             "temperature": temperature,
#             "maxOutputTokens": max_tokens,
#         }
#     }

#     if thinking_budget > 0:
#         request["generationConfig"]["thinkingConfig"] = {
#             "thinkingBudget": thinking_budget
#         }

#     if messages["system"]:
#         request["systemInstruction"] = {
#             "parts": [{"text": messages["system"]}]
#         }

#     return {"request": request}


# def create_batch_jsonl(
#     data: list[dict],
#     output_path: str,
#     gcs_base: str,
#     temperature: float = 0.0,
#     max_tokens: int = 1000,
#     thinking_budget: int = 2000,
#     samples: int = 1,
# ) -> tuple[str, list[dict]]:
#     """Create JSONL file for batch processing."""
#     print(f"Creating batch JSONL: {output_path}")

#     metadata_list = []
#     with open(output_path, "w") as f:
#         global_idx = 0
#         for s_idx in range(samples):
#             for idx, entry in enumerate(data):
#                 request = create_batch_request(
#                     entry, gcs_base, temperature, max_tokens, thinking_budget
#                 )
#                 f.write(json.dumps(request) + "\n")

#                 messages = parse_conversations(entry)
#                 metadata_list.append({
#                     "idx": global_idx,
#                     "sample_idx": s_idx,
#                     "original_idx": idx,
#                     "submission_id": entry.get("_metadata", {}).get("submission_id", ""),
#                     "label": messages["gpt"] or "",
#                     "ground_truth": entry.get("_metadata", {}).get("answer", ""),
#                     "human_message": messages["human"] or "",
#                 })
#                 global_idx += 1

#     file_size = os.path.getsize(output_path)
#     print(f"  Created {global_idx} requests ({samples} samples x {len(data)} papers), file size: {file_size / 1024 / 1024:.2f} MB")

#     return output_path, metadata_list


# # ============================================================================
# # BATCH JOB MANAGEMENT
# # ============================================================================

# def get_client(project: str, location: str) -> genai.Client:
#     """Get Gemini client for Vertex AI."""
#     return genai.Client(
#         vertexai=True,
#         project=project,
#         location=location,
#         http_options=HttpOptions(api_version="v1")
#     )


# def upload_to_gcs(local_path: str, gcs_uri: str, project: str) -> str:
#     """Upload local file to GCS."""
#     from google.cloud import storage
#     parts = gcs_uri[5:].split("/", 1)
#     bucket_name = parts[0]
#     blob_name = parts[1] if len(parts) > 1 else ""
#     client = storage.Client(project=project)
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#     blob.upload_from_filename(local_path)
#     print(f"  Uploaded to: {gcs_uri}")
#     return gcs_uri


# def download_from_gcs(gcs_uri: str, local_path: str, project: str) -> str:
#     """Download file from GCS."""
#     from google.cloud import storage
#     parts = gcs_uri[5:].split("/", 1)
#     bucket_name = parts[0]
#     blob_name = parts[1] if len(parts) > 1 else ""
#     client = storage.Client(project=project)
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#     blob.download_to_filename(local_path)
#     return local_path


# def list_gcs_files(gcs_prefix: str, project: str) -> list[str]:
#     """List files in GCS with prefix."""
#     from google.cloud import storage
#     parts = gcs_prefix[5:].split("/", 1)
#     bucket_name = parts[0]
#     prefix = parts[1] if len(parts) > 1 else ""
#     client = storage.Client(project=project)
#     bucket = client.bucket(bucket_name)
#     blobs = bucket.list_blobs(prefix=prefix)
#     return [f"gs://{bucket_name}/{blob.name}" for blob in blobs]


# def submit_batch_job(
#     client: genai.Client,
#     model: str,
#     input_uri: str,
#     output_uri: str,
#     display_name: str = None,
# ) -> any:
#     """Submit a batch prediction job."""
#     if display_name is None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         display_name = f"autoreviewer_{timestamp}"

#     print(f"\nSubmitting batch job...")
#     print(f"  Model: {model}")
#     print(f"  Input: {input_uri}")
#     print(f"  Output: {output_uri}")

#     job = client.batches.create(
#         model=model,
#         src=input_uri,
#         config=CreateBatchJobConfig(
#             dest=output_uri,
#             display_name=display_name,
#         )
#     )

#     print(f"  Job name: {job.name}")
#     print(f"  State: {job.state}")
#     return job


# def poll_until_complete(
#     client: genai.Client,
#     job_name: str,
#     interval: int = 300,
#     max_wait: int = 86400,
# ) -> any:
#     """Poll job until completion."""
#     print(f"\nPolling job status (interval: {interval}s)...")
#     start_time = time.time()
#     while True:
#         elapsed = time.time() - start_time
#         if elapsed > max_wait:
#             raise RuntimeError(f"Job timed out after {elapsed/3600:.1f} hours")
#         job = client.batches.get(name=job_name)
#         state = str(job.state)
#         elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m"
#         print(f"  [{elapsed_str}] State: {state}")
#         if "SUCCEEDED" in state or state == "JOB_STATE_SUCCEEDED":
#             print(f"\n  Job completed successfully!")
#             return job
#         if "FAILED" in state or "CANCELLED" in state:
#             raise RuntimeError(f"Batch job failed or cancelled: {state}")
#         time.sleep(interval)


# # ============================================================================
# # RESULT PROCESSING
# # ============================================================================

# def parse_prediction(response: dict) -> str:
#     """Extract prediction text from batch response."""
#     try:
#         candidates = response.get("candidates", [])
#         if candidates:
#             content = candidates[0].get("content", {})
#             parts = content.get("parts", [])
#             if parts:
#                 return parts[0].get("text", "")
#     except (KeyError, IndexError, TypeError):
#         pass
#     return ""


# def process_batch_results(
#     output_uri: str,
#     metadata_list: list[dict],
#     output_path: str,
#     project: str,
# ) -> int:
#     """Process batch results and save in vllm_infer.py format."""
#     print(f"\nProcessing batch results from: {output_uri}")
#     result_files = list_gcs_files(output_uri, project)
#     result_files = [f for f in result_files if f.endswith(".jsonl")]
#     if not result_files:
#         print("  No result files found!")
#         return 0
#     print(f"  Found {len(result_files)} result file(s)")

#     all_results = []
#     with tempfile.TemporaryDirectory() as tmpdir:
#         for gcs_file in result_files:
#             local_file = os.path.join(tmpdir, os.path.basename(gcs_file))
#             download_from_gcs(gcs_file, local_file, project)
#             with open(local_file) as f:
#                 for line in f:
#                     if line.strip():
#                         all_results.append(json.loads(line))

#     print(f"  Parsed {len(all_results)} results")
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     processed = 0
#     with open(output_path, "w") as f:
#         for idx, result in enumerate(all_results):
#             if idx >= len(metadata_list):
#                 break
#             meta = metadata_list[idx]
#             response = result.get("response", {})
#             prediction = parse_prediction(response)
#             output_entry = {
#                 "prompt": meta["human_message"],
#                 "predict": prediction,
#                 "label": meta["label"],
#                 "submission_id": meta["submission_id"],
#                 "sample_idx": meta.get("sample_idx", 0),
#             }
#             f.write(json.dumps(output_entry) + "\n")
#             processed += 1
#     print(f"  Saved {processed} results to: {output_path}")
#     return processed


# # ============================================================================
# # MAIN
# # ============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description="Gemini Batch API for Inference Scaling Experiments"
#     )
#     parser.add_argument("--modality", type=str, help="Modality (text, vision, or comma-separated list, or 'all')")
#     parser.add_argument("--variant", type=str, help="Prompt variant (e.g., standard_nofewshot_boxed, or comma-separated list, or 'all')")
#     parser.add_argument("--data_dir", type=str, default="final_inference_scaling/data")
#     parser.add_argument("--output_dir", type=str, default="final_inference_scaling/results")
#     parser.add_argument("--gcs_staging", type=str, default=DEFAULT_GCS_STAGING)
#     parser.add_argument("--gcs_base", type=str, default=DEFAULT_GCS_IMAGES_BASE)
#     parser.add_argument("--limit", type=int, help="Limit to first N samples")
#     parser.add_argument("--ids_file", type=str, help="Text file with submission_ids to include")
#     parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
#     parser.add_argument("--project", type=str, default="hip-gecko-485003-c4", help="GCP project ID")
#     parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
#     parser.add_argument("--temperature", type=float, default=0.0)
#     parser.add_argument("--samples", type=int, default=1, help="Number of samples per paper")
#     parser.add_argument("--max_tokens", type=int, default=1000)
#     parser.add_argument("--thinking_budget", type=int, default=2000, help="0 to disable thinking")
#     parser.add_argument("--dry_run", action="store_true", help="Create JSONL locally but don't submit")
#     parser.add_argument("--submit_only", action="store_true", help="Submit and exit without polling")
#     parser.add_argument("--poll_interval", type=int, default=300)

#     args = parser.parse_args()

#     if args.modality == "all":
#         modalities = list(BASE_DATASETS.keys())
#     elif args.modality:
#         modalities = [m.strip() for m in args.modality.split(",")]
#     else:
#         parser.error("--modality is required")

#     if args.variant == "all":
#         variants = PROMPT_VARIANTS
#     elif args.variant:
#         variants = [v.strip() for v in args.variant.split(",")]
#     else:
#         parser.error("--variant is required")

#     for m in modalities:
#         if m not in BASE_DATASETS:
#             parser.error(f"Invalid modality: {m}. Choices: {list(BASE_DATASETS.keys())}")
#     for v in variants:
#         if v not in PROMPT_VARIANTS:
#             parser.error(f"Invalid variant: {v}. Choices: {PROMPT_VARIANTS}")

#     if args.gcs_staging is None:
#         args.gcs_staging = DEFAULT_GCS_STAGING

#     for modality in modalities:
#         for variant in variants:
#             print(f"\n{'='*70}")
#             print(f"PROCESS: {modality} / {variant}")
#             print(f"{'='*70}")

#             try:
#                 data_dir = Path(args.data_dir)
#                 data = load_dataset(data_dir, modality, variant)
#                 if args.ids_file:
#                     data = filter_by_ids(data, args.ids_file)
#                 if args.limit:
#                     data = data[:args.limit]
#                     print(f"  Limited to first {len(data)} samples")
#                 if not data:
#                     print(f"Skipping {modality}/{variant}: No samples after filtering.")
#                     continue

#                 cutoff_len = 60000 if "fewshot" in variant and "nofewshot" not in variant else 24480
#                 model_name_clean = args.model.replace("/", "--")
#                 gen_suffix = "gen1" if args.samples == 1 else f"gen{args.samples}"
#                 job_results_dir = Path(args.output_dir) / model_name_clean / modality / f"{variant}_ctx{cutoff_len}_{gen_suffix}"
#                 job_results_dir.mkdir(parents=True, exist_ok=True)
                
#                 predictions_path = job_results_dir / "predictions.jsonl"
#                 metadata_path = job_results_dir / "gemini_metadata.json"
#                 job_info_path = job_results_dir / "gemini_job_info.json"

#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 local_jsonl = f"/tmp/gemini_batch_{modality}_{variant}_{timestamp}.jsonl"
#                 _, metadata_list = create_batch_jsonl(
#                     data, local_jsonl, args.gcs_base,
#                     args.temperature, args.max_tokens, args.thinking_budget,
#                     samples=args.samples
#                 )

#                 with open(metadata_path, "w") as f:
#                     json.dump(metadata_list, f)
#                 print(f"Metadata saved to: {metadata_path}")

#                 if args.dry_run:
#                     print(f"DRY RUN complete for {modality}/{variant}.")
#                     if os.path.exists(local_jsonl):
#                         os.remove(local_jsonl)
#                     continue

#                 gcs_input_uri = f"{args.gcs_staging.rstrip('/')}/{modality}_{variant}_{timestamp}_input.jsonl"
#                 gcs_output_uri = f"{args.gcs_staging.rstrip('/')}/{modality}_{variant}_{timestamp}_output/"
#                 upload_to_gcs(local_jsonl, gcs_input_uri, args.project)

#                 client = get_client(args.project, args.location)
#                 display_name = f"{modality}_{variant}_{timestamp}"
#                 job = submit_batch_job(client, args.model, gcs_input_uri, gcs_output_uri, display_name)

#                 job_info = {
#                     "job_name": job.name,
#                     "model": args.model,
#                     "modality": modality,
#                     "variant": variant,
#                     "num_samples": len(data),
#                     "samples": args.samples,
#                     "input_uri": gcs_input_uri,
#                     "output_uri": gcs_output_uri,
#                     "submitted_at": timestamp,
#                     "predictions_file": str(predictions_path),
#                 }
#                 with open(job_info_path, "w") as f:
#                     json.dump(job_info, f, indent=2)
#                 print(f"Job info saved to: {job_info_path}")

#                 if args.submit_only:
#                     print(f"Submitted {modality}/{variant}. Moving to next.")
#                 else:
#                     poll_until_complete(client, job.name, interval=args.poll_interval)
#                     process_batch_results(gcs_output_uri, metadata_list, str(predictions_path), args.project)
#                     job_info["status"] = "completed"
#                     with open(job_info_path, "w") as f:
#                         json.dump(job_info, f, indent=2)
#                     print(f"Completed {modality}/{variant}.")

#                 if os.path.exists(local_jsonl):
#                     os.remove(local_jsonl)

#             except Exception as e:
#                 print(f"Error processing {modality}/{variant}: {e}")
#                 continue

#     print("\n" + "=" * 70)
#     print("ALL COMBINATIONS PROCESSED.")
#     print("=" * 70)

# if __name__ == "__main__":
#     main()