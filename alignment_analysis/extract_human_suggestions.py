#!/usr/bin/env python3
"""
Step 3: Extract human reviewer suggestions from ground truth reviews using Gemini batch API.

For each paper in the 500-paper subset, parses original_reviews from the metadata CSV
and uses Gemini 2.5 Flash to extract discrete improvement suggestions per reviewer.

Usage:
    # Submit batch job
    python alignment_analysis/extract_human_suggestions.py \
        --project hip-gecko-485003-c4 \
        --submit_only

    # Retrieve after job completes
    python alignment_analysis/extract_human_suggestions.py \
        --project hip-gecko-485003-c4 \
        --retrieve
"""

import argparse
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from google import genai
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig, HttpOptions


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-central1"
DEFAULT_GCS_STAGING = "gs://jl0796-autoreviewer-staging/alignment_analysis"
METADATA_CSV = "/scratch/gpfs/ZHUANGL/jl0796/shared/data/massive_metadata_v7_5.csv"
IDS_FILE = "iclr_2020_2023_2025_2026_85_5_10_balanced_original_v7_filtered_test_500_subset.txt"
JOB_INFO_PATH = "alignment_analysis/results/human_suggestions_job_info.json"
OUTPUT_PATH = "alignment_analysis/results/human_suggestions.jsonl"

SYSTEM_PROMPT = (
    "You are a research assistant specializing in academic peer review analysis. "
    "Your task is to extract discrete, actionable improvement suggestions from reviewer comments."
)

EXTRACTION_PROMPT_TEMPLATE = """Below is a reviewer's comment on an academic paper. Extract all discrete, actionable improvement suggestions mentioned by the reviewer. Each suggestion should be a specific, concrete request (e.g., "Add an ablation study comparing X and Y", not vague statements like "The paper needs improvement").

Reviewer comment:
{review_text}

Return a JSON array of strings, each string being one discrete suggestion. If no actionable suggestions are found, return an empty array [].
Example output: ["Add ablation study for hyperparameter sensitivity", "Clarify the proof of Theorem 2", "Include comparison with baseline method X"]

Return only the JSON array, no other text."""


def get_client(project: str, location: str) -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=HttpOptions(api_version="v1"),
    )


def upload_to_gcs(local_path: str, gcs_uri: str, project: str) -> str:
    parts = gcs_uri[5:].split("/", 1)
    bucket_name, blob_name = parts[0], parts[1] if len(parts) > 1 else ""
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"  Uploaded to: {gcs_uri}")
    return gcs_uri


def download_from_gcs(gcs_uri: str, local_path: str, project: str) -> str:
    parts = gcs_uri[5:].split("/", 1)
    bucket_name, blob_name = parts[0], parts[1] if len(parts) > 1 else ""
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path


def list_gcs_files(gcs_prefix: str, project: str) -> list[str]:
    parts = gcs_prefix[5:].split("/", 1)
    bucket_name, prefix = parts[0], parts[1] if len(parts) > 1 else ""
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [f"gs://{bucket_name}/{blob.name}" for blob in blobs]


def parse_reviews(original_reviews_field) -> list[str]:
    """Parse the original_reviews field into a list of reviewer text strings."""
    if not original_reviews_field or (isinstance(original_reviews_field, float)):
        return []

    reviews_str = str(original_reviews_field)

    # Try parsing as JSON
    try:
        parsed = json.loads(reviews_str)
        if isinstance(parsed, list):
            texts = []
            for item in parsed:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    # Try common fields
                    for key in ["review", "comment", "text", "body"]:
                        if key in item:
                            texts.append(str(item[key]))
                            break
                    else:
                        texts.append(json.dumps(item))
            return texts
        elif isinstance(parsed, dict):
            return [json.dumps(parsed)]
    except (json.JSONDecodeError, ValueError):
        pass

    # Treat as raw text (single reviewer)
    return [reviews_str]


def build_batch_requests(papers: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Build Gemini batch requests, one per (paper, reviewer) pair.
    Returns (requests, metadata_list).
    """
    requests = []
    metadata_list = []

    for paper in papers:
        submission_id = paper["submission_id"]
        reviews = paper["reviews"]
        for reviewer_idx, review_text in enumerate(reviews):
            if not review_text.strip():
                continue

            prompt = EXTRACTION_PROMPT_TEMPLATE.format(review_text=review_text)  # do not truncate long reviews
            request = {
                "request": {
                    "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 3000,
                    },
                }
            }
            requests.append(request)
            metadata_list.append({
                "submission_id": submission_id,
                "reviewer_idx": reviewer_idx,
            })

    return requests, metadata_list


def parse_suggestions_response(text: str) -> list[str]:
    """Parse the Gemini response into a list of suggestion strings."""
    text = text.strip()
    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(s) for s in result if s]
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: split by newline and treat each line as a suggestion
    return [line.strip("- •*").strip() for line in text.split("\n") if line.strip() and not line.startswith("[")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="hip-gecko-485003-c4")
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--gcs_staging", type=str, default=DEFAULT_GCS_STAGING)
    parser.add_argument("--ids_file", type=str, default=IDS_FILE)
    parser.add_argument("--metadata_csv", type=str, default=METADATA_CSV)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--job_info", type=str, default=JOB_INFO_PATH)
    parser.add_argument("--submit_only", action="store_true", help="Submit job and exit without polling")
    parser.add_argument("--retrieve", action="store_true", help="Retrieve results from a previously submitted job")
    parser.add_argument("--dry_run", action="store_true", help="Build JSONL locally but don't submit")
    args = parser.parse_args()

    os.makedirs("alignment_analysis/results", exist_ok=True)

    # --- Load paper IDs ---
    with open(args.ids_file) as f:
        target_ids = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(target_ids)} paper IDs from {args.ids_file}")

    # --- Load metadata ---
    print(f"Loading metadata from {args.metadata_csv}")
    df = pd.read_csv(args.metadata_csv)
    df = df[df["submission_id"].isin(target_ids)].copy()
    print(f"  Found {len(df)} matching papers in metadata")

    # Parse reviews
    papers = []
    for _, row in df.iterrows():
        reviews = parse_reviews(row.get("original_reviews"))
        if reviews:
            papers.append({"submission_id": row["submission_id"], "reviews": reviews})
        else:
            print(f"  WARNING: No reviews for {row['submission_id']}")

    print(f"  {len(papers)} papers with reviews, total reviewer slots: {sum(len(p['reviews']) for p in papers)}")

    # --- RETRIEVE mode ---
    if args.retrieve:
        if not os.path.exists(args.job_info):
            print(f"ERROR: Job info not found at {args.job_info}")
            return

        with open(args.job_info) as f:
            job_info = json.load(f)

        metadata_list = job_info["metadata_list"]
        output_uri = job_info["output_uri"]

        print(f"Retrieving results from {output_uri}")
        result_files = list_gcs_files(output_uri, args.project)
        result_files = [f for f in result_files if f.endswith(".jsonl")]

        all_results = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for gcs_file in result_files:
                local_file = os.path.join(tmpdir, os.path.basename(gcs_file))
                download_from_gcs(gcs_file, local_file, args.project)
                with open(local_file) as f:
                    for line in f:
                        if line.strip():
                            all_results.append(json.loads(line))

        print(f"  Parsed {len(all_results)} results")

        # Aggregate by submission_id
        paper_suggestions: dict[str, list[list[str]]] = {}
        for idx, result in enumerate(all_results):
            if idx >= len(metadata_list):
                break
            meta = metadata_list[idx]
            sub_id = meta["submission_id"]
            response = result.get("response", {})
            try:
                candidates = response.get("candidates", [])
                text = ""
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
            except Exception:
                text = ""

            suggestions = parse_suggestions_response(text)
            if sub_id not in paper_suggestions:
                paper_suggestions[sub_id] = []
            paper_suggestions[sub_id].append(suggestions)

        with open(args.output, "w") as f:
            for sub_id, reviewer_suggestions in paper_suggestions.items():
                entry = {"submission_id": sub_id, "reviewer_suggestions": reviewer_suggestions}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Saved {len(paper_suggestions)} papers to {args.output}")
        return

    # --- BUILD + SUBMIT mode ---
    requests, metadata_list = build_batch_requests(papers)
    print(f"Built {len(requests)} batch requests")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_jsonl = f"/tmp/human_suggestions_requests_{timestamp}.jsonl"
    with open(local_jsonl, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    print(f"Wrote batch requests to {local_jsonl}")

    if args.dry_run:
        print("Dry run — not submitting.")
        return

    # Upload to GCS
    gcs_input_uri = f"{args.gcs_staging}/human_suggestions_requests_{timestamp}.jsonl"
    gcs_output_uri = f"{args.gcs_staging}/human_suggestions_results_{timestamp}/"
    upload_to_gcs(local_jsonl, gcs_input_uri, args.project)

    # Submit batch job
    client = get_client(args.project, args.location)
    job = client.batches.create(
        model=args.model,
        src=gcs_input_uri,
        config=CreateBatchJobConfig(
            dest=gcs_output_uri,
            display_name=f"alignment_human_suggestions_{timestamp}",
        ),
    )
    print(f"Submitted job: {job.name}")
    print(f"State: {job.state}")

    # Save job info
    job_info = {
        "job_name": job.name,
        "output_uri": gcs_output_uri,
        "submitted_at": timestamp,
        "metadata_list": metadata_list,
    }
    with open(args.job_info, "w") as f:
        json.dump(job_info, f, indent=2)
    print(f"Job info saved to {args.job_info}")

    if args.submit_only:
        print("submit_only=True — exiting. Run with --retrieve when job completes.")
        return

    # Poll until complete
    print("Polling for completion...")
    while True:
        job = client.batches.get(name=job.name)
        state = str(job.state)
        print(f"  State: {state}")
        if "SUCCEEDED" in state or state == "JOB_STATE_SUCCEEDED":
            break
        if "FAILED" in state or "CANCELLED" in state:
            raise RuntimeError(f"Job failed: {state}")
        time.sleep(300)

    print("Job complete. Run with --retrieve to fetch results.")


if __name__ == "__main__":
    main()
