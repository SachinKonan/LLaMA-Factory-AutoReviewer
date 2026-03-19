#!/usr/bin/env python3
"""
Step 5: Grade alignment between AI suggestions and human suggestions using Gemini as judge.

For each paper and each variant (conditioned, unconditioned), submits a Gemini batch
request asking it to match each AI suggestion to a human suggestion index (or null).

Usage:
    # Submit batch jobs
    python alignment_analysis/grade_alignment.py \
        --project hip-gecko-485003-c4 \
        --submit_only

    # Retrieve results after jobs complete
    python alignment_analysis/grade_alignment.py \
        --project hip-gecko-485003-c4 \
        --retrieve

    # Compute aggregate metrics (no Gemini needed, run after retrieve)
    python alignment_analysis/grade_alignment.py --metrics_only
"""

import argparse
import json
import os
import tempfile
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd
from google import genai
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig, HttpOptions


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-central1"
DEFAULT_GCS_STAGING = "gs://jl0796-autoreviewer-staging/alignment_analysis"

SYSTEM_PROMPT = (
    "You are an expert at comparing academic paper review suggestions. "
    "Your task is to determine whether AI-generated suggestions correspond to human reviewer suggestions."
)

GRADING_PROMPT_TEMPLATE = """You are comparing AI-generated paper improvement suggestions against human reviewer suggestions.

Human reviewer suggestions (ground truth):
{human_suggestions_numbered}

AI-generated suggestions:
{ai_suggestions_lettered}

For each AI suggestion (A through {last_letter}), determine whether it substantially corresponds to one of the human suggestions. Two suggestions correspond if they address the same core issue or request the same type of improvement, even if worded differently.

Rules:
- Each human suggestion index can be matched to at most one AI suggestion.
- If an AI suggestion does not clearly correspond to any human suggestion, assign null.
- Be conservative: only match if there is a clear, specific correspondence.

Return a JSON object mapping each AI suggestion letter to either a human suggestion index (0-based integer) or null.
Example: {{"A": 0, "B": 2, "C": null, "D": null, "E": 1}}

Return only the JSON object, no other text."""

LENIENT_GRADING_PROMPT_TEMPLATE = """You are comparing AI-generated paper improvement suggestions against human reviewer suggestions.

Human reviewer suggestions (ground truth):
{human_suggestions_numbered}

AI-generated suggestions:
{ai_suggestions_lettered}

For each AI suggestion (A through {last_letter}), determine whether it substantially corresponds to one of the human suggestions. Two suggestions correspond if they address the same core issue or request the same type of improvement, even if worded differently.

Rules:
- Each human suggestion index can be matched to at most one AI suggestion.
- If an AI suggestion does not clearly correspond to any human suggestion, assign null.
- Be liberal: match if the AI suggestion addresses the same general topic, issue, or area as the human suggestion, even if wording/scope/specifics differ. Err on the side of matching.

Return a JSON object mapping each AI suggestion letter to either a human suggestion index (0-based integer) or null.
Example: {{"A": 0, "B": 2, "C": null, "D": null, "E": 1}}

Return only the JSON object, no other text."""


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


def build_grading_request(
    sub_id: str,
    human_suggestions: list[str],
    ai_suggestions: list[str],
    lenient: bool = False,
) -> tuple[dict, dict]:
    """Build a Gemini batch request for grading one paper."""
    letters = [chr(ord("A") + i) for i in range(len(ai_suggestions))]
    last_letter = letters[-1] if letters else "E"

    human_numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(human_suggestions))
    ai_lettered = "\n".join(f"{l}: {s}" for l, s in zip(letters, ai_suggestions))

    template = LENIENT_GRADING_PROMPT_TEMPLATE if lenient else GRADING_PROMPT_TEMPLATE
    prompt = template.format(
        human_suggestions_numbered=human_numbered,
        ai_suggestions_lettered=ai_lettered,
        last_letter=last_letter,
    )

    request = {
        "request": {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 2048,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
    }
    metadata = {
        "submission_id": sub_id,
        "human_suggestions": human_suggestions,
        "ai_suggestions": ai_suggestions,
        "letters": letters,
    }
    return request, metadata


def parse_grading_response(text: str, letters: list[str]) -> dict:
    """Parse the Gemini grading response into a letter→index mapping."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            # Normalize: ensure all letters are present
            matches = {}
            for letter in letters:
                val = result.get(letter)
                matches[letter] = int(val) if val is not None and str(val).isdigit() else None
            return matches
    except (json.JSONDecodeError, ValueError):
        pass
    # Return all null on parse failure
    return {letter: None for letter in letters}


def compute_metrics(entry: dict) -> dict:
    """Compute intersection metrics from a graded alignment entry."""
    human_suggestions = entry["human_suggestions"]
    ai_suggestions = entry["ai_suggestions"]
    matches = entry["matches"]

    matched_human_indices = set()
    matched_ai_count = 0
    for letter, human_idx in matches.items():
        if human_idx is not None:
            matched_human_indices.add(human_idx)
            matched_ai_count += 1

    intersection_count = len(matched_human_indices)
    n_human = len(human_suggestions)
    n_ai = len(ai_suggestions)

    return {
        "submission_id": entry["submission_id"],
        "intersection_count": intersection_count,
        "human_coverage": intersection_count / n_human if n_human > 0 else 0.0,
        "ai_coverage": matched_ai_count / n_ai if n_ai > 0 else 0.0,
        "unrelated_ai_count": n_ai - matched_ai_count,
        "n_human": n_human,
        "n_ai": n_ai,
    }


def load_flat_human_suggestions(path: str) -> dict[str, list[str]]:
    """Load human_suggestions.jsonl and flatten to a single list per paper."""
    result = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sub_id = entry["submission_id"]
            # Flatten: reviewer_suggestions is list of lists
            flat = []
            for reviewer_list in entry.get("reviewer_suggestions", []):
                flat.extend(reviewer_list)
            result[sub_id] = flat
    return result


def submit_grading_variant(
    variant: str,
    human_map: dict[str, list[str]],
    ai_path: str,
    args,
    timestamp: str,
    lenient: bool = False,
) -> dict:
    """Build, upload, submit grading batch for one variant."""
    prompt_variant = "lenient" if lenient else "strict"
    full_variant = f"{variant}_{prompt_variant}"

    # Load AI suggestions
    ai_map: dict[str, list[str]] = {}
    with open(ai_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                ai_map[entry["submission_id"]] = entry.get("suggestions", [])

    requests, metadata_list = [], []
    for sub_id, human_suggestions in human_map.items():
        ai_suggestions = ai_map.get(sub_id)
        if not ai_suggestions:
            print(f"  WARNING: No AI suggestions for {sub_id} in {variant}, skipping")
            continue
        if not human_suggestions:
            print(f"  WARNING: No human suggestions for {sub_id}, skipping")
            continue
        req, meta = build_grading_request(sub_id, human_suggestions, ai_suggestions, lenient=lenient)
        requests.append(req)
        metadata_list.append(meta)

    print(f"  {full_variant}: {len(requests)} grading requests")

    local_jsonl = f"/tmp/grading_{full_variant}_{timestamp}.jsonl"
    with open(local_jsonl, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    if args.dry_run:
        print(f"  Dry run — not submitting.")
        return {}

    gcs_input_uri = f"{args.gcs_staging}/grading_{full_variant}_requests_{timestamp}.jsonl"
    gcs_output_uri = f"{args.gcs_staging}/grading_{full_variant}_results_{timestamp}/"
    upload_to_gcs(local_jsonl, gcs_input_uri, args.project)

    client = get_client(args.project, args.location)
    job = client.batches.create(
        model=args.model,
        src=gcs_input_uri,
        config=CreateBatchJobConfig(
            dest=gcs_output_uri,
            display_name=f"alignment_grading_{full_variant}_{timestamp}",
        ),
    )
    print(f"  Submitted job: {job.name}")

    return {
        "job_name": job.name,
        "output_uri": gcs_output_uri,
        "variant": full_variant,
        "submitted_at": timestamp,
        "metadata_list": metadata_list,
    }


def retrieve_grading_variant(
    job_info: dict,
    output_path: str,
    project: str,
):
    """Retrieve and parse grading results for one variant."""
    metadata_list = job_info["metadata_list"]
    output_uri = job_info["output_uri"]

    print(f"Retrieving from {output_uri}")
    result_files = list_gcs_files(output_uri, project)
    result_files = [f for f in result_files if f.endswith(".jsonl")]

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

    with open(output_path, "w") as f:
        for idx, result in enumerate(all_results):
            if idx >= len(metadata_list):
                break
            meta = metadata_list[idx]
            response = result.get("response", {})
            try:
                candidates = response.get("candidates", [])
                text = ""
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
            except Exception:
                text = ""

            letters = meta.get("letters", [chr(ord("A") + i) for i in range(len(meta["ai_suggestions"]))])
            matches = parse_grading_response(text, letters)

            # Convert letter keys to positional index keys for downstream use
            matches_positional = {str(i): matches.get(letter) for i, letter in enumerate(letters)}

            entry = {
                "submission_id": meta["submission_id"],
                "human_suggestions": meta["human_suggestions"],
                "ai_suggestions": meta["ai_suggestions"],
                "matches": matches_positional,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved to {output_path}")


def print_metrics_summary(variant: str, alignment_path: str):
    """Print aggregate metrics for a variant."""
    if not os.path.exists(alignment_path):
        print(f"  {variant}: file not found at {alignment_path}")
        return

    all_metrics = []
    with open(alignment_path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                all_metrics.append(compute_metrics(entry))

    if not all_metrics:
        print(f"  {variant}: no entries")
        return

    n = len(all_metrics)
    avg_intersection = sum(m["intersection_count"] for m in all_metrics) / n
    avg_human_cov = sum(m["human_coverage"] for m in all_metrics) / n
    avg_ai_cov = sum(m["ai_coverage"] for m in all_metrics) / n
    avg_unrelated = sum(m["unrelated_ai_count"] for m in all_metrics) / n

    print(f"\n=== {variant.upper()} ({n} papers) ===")
    print(f"  Avg intersection count:  {avg_intersection:.3f}")
    print(f"  Avg human coverage:      {avg_human_cov:.3f}")
    print(f"  Avg AI coverage:         {avg_ai_cov:.3f}")
    print(f"  Avg unrelated AI count:  {avg_unrelated:.3f}")

    # Save per-paper metrics
    metrics_path = alignment_path.replace("alignment_matches_", "alignment_metrics_")
    df = pd.DataFrame(all_metrics)
    df.to_csv(metrics_path.replace(".jsonl", ".csv"), index=False)
    print(f"  Per-paper metrics saved to {metrics_path.replace('.jsonl', '.csv')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="hip-gecko-485003-c4")
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--gcs_staging", type=str, default=DEFAULT_GCS_STAGING)
    parser.add_argument("--human_suggestions", type=str, default="alignment_analysis/results/human_suggestions.jsonl")
    parser.add_argument("--ai_conditioned", type=str, default=None,
                        help="Path to conditioned AI suggestions (default derived from --ai_prompt_variant)")
    parser.add_argument("--ai_unconditioned", type=str, default=None,
                        help="Path to unconditioned AI suggestions (default derived from --ai_prompt_variant)")
    parser.add_argument("--submit_only", action="store_true")
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--metrics_only", action="store_true", help="Only compute metrics from existing results")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--lenient", action="store_true", help="Use lenient matching prompt instead of strict")
    parser.add_argument(
        "--ai_prompt_variant", type=str, default="original", choices=["original", "specific"],
        help="Which AI suggestion set to grade: 'original' or 'specific'",
    )
    args = parser.parse_args()

    # Resolve AI suggestion paths from --ai_prompt_variant if not explicitly provided
    apv = args.ai_prompt_variant
    if args.ai_conditioned is None:
        args.ai_conditioned = f"alignment_analysis/results/ai_suggestions_conditioned_{apv}.jsonl"
    if args.ai_unconditioned is None:
        args.ai_unconditioned = f"alignment_analysis/results/ai_suggestions_unconditioned_{apv}.jsonl"

    os.makedirs("alignment_analysis/results", exist_ok=True)

    grading_variant = "lenient" if args.lenient else "strict"
    # Include AI prompt variant in filenames to avoid collisions across the 8 job combinations
    name_suffix = f"{apv}_{grading_variant}"
    conditioned_job_info_path = f"alignment_analysis/results/grading_conditioned_{name_suffix}_job_info.json"
    unconditioned_job_info_path = f"alignment_analysis/results/grading_unconditioned_{name_suffix}_job_info.json"
    conditioned_output = f"alignment_analysis/results/alignment_matches_conditioned_{name_suffix}.jsonl"
    unconditioned_output = f"alignment_analysis/results/alignment_matches_unconditioned_{name_suffix}.jsonl"
    prompt_variant = name_suffix  # used in display strings below

    # --- METRICS ONLY ---
    if args.metrics_only:
        print_metrics_summary(f"conditioned_{prompt_variant}", conditioned_output)
        print_metrics_summary(f"unconditioned_{prompt_variant}", unconditioned_output)
        return

    # --- RETRIEVE ---
    if args.retrieve:
        for job_info_path, output_path, variant in [
            (conditioned_job_info_path, conditioned_output, f"conditioned_{prompt_variant}"),
            (unconditioned_job_info_path, unconditioned_output, f"unconditioned_{prompt_variant}"),
        ]:
            if os.path.exists(job_info_path):
                with open(job_info_path) as f:
                    job_info = json.load(f)
                retrieve_grading_variant(job_info, output_path, args.project)
            else:
                print(f"WARNING: Job info not found: {job_info_path}")

        print("\nComputing metrics...")
        print_metrics_summary(f"conditioned_{prompt_variant}", conditioned_output)
        print_metrics_summary(f"unconditioned_{prompt_variant}", unconditioned_output)
        return

    # --- SUBMIT ---
    human_map = load_flat_human_suggestions(args.human_suggestions)
    print(f"Loaded human suggestions for {len(human_map)} papers")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for variant, ai_path, job_info_path in [
        ("conditioned", args.ai_conditioned, conditioned_job_info_path),
        ("unconditioned", args.ai_unconditioned, unconditioned_job_info_path),
    ]:
        print(f"\nProcessing variant: {variant} ({prompt_variant})")
        job_info = submit_grading_variant(variant, human_map, ai_path, args, timestamp, lenient=args.lenient)
        if job_info:
            with open(job_info_path, "w") as f:
                json.dump(job_info, f, indent=2)
            print(f"  Job info saved to {job_info_path}")

    if args.submit_only or args.dry_run:
        print("\nDone submitting. Run with --retrieve when jobs complete.")
        return

    # Poll for completion
    client = get_client(args.project, args.location)
    for variant, job_info_path in [
        (f"conditioned_{prompt_variant}", conditioned_job_info_path),
        (f"unconditioned_{prompt_variant}", unconditioned_job_info_path),
    ]:
        if not os.path.exists(job_info_path):
            continue
        with open(job_info_path) as f:
            job_info = json.load(f)
        job_name = job_info.get("job_name")
        if not job_name:
            continue
        while True:
            job = client.batches.get(name=job_name)
            state = str(job.state)
            print(f"  [{variant}] State: {state}")
            if "SUCCEEDED" in state or state == "JOB_STATE_SUCCEEDED":
                break
            if "FAILED" in state or "CANCELLED" in state:
                raise RuntimeError(f"Job failed: {state}")
            time.sleep(300)

    print("All jobs complete. Run with --retrieve to fetch and compute metrics.")


if __name__ == "__main__":
    main()
