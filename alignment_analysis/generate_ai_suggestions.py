#!/usr/bin/env python3
"""
Step 4: Generate AI suggestions using Gemini 2.5 Flash (two variants: conditioned, unconditioned).

For each paper in the 500-paper subset, reads the paper markdown from clean_md_path
and generates 5 improvement suggestions. The conditioned variant also receives the
text model's prediction (Accept/Reject) and confidence score.

Usage:
    # Submit both batch jobs
    python alignment_analysis/generate_ai_suggestions.py \
        --project hip-gecko-485003-c4 \
        --submit_only

    # Retrieve results after jobs complete
    python alignment_analysis/generate_ai_suggestions.py \
        --project hip-gecko-485003-c4 \
        --retrieve
"""

import argparse
import json
import math
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig, HttpOptions


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-central1"
DEFAULT_GCS_STAGING = "gs://jl0796-autoreviewer-staging/alignment_analysis"
IDS_FILE = "iclr_2020_2023_2025_2026_85_5_10_balanced_original_v7_filtered_test_500_subset.txt"
CONFIDENCE_PATH = "alignment_analysis/results/confidence.jsonl"
DATASET_PATH = "/scratch/gpfs/ZHUANGL/jl0796/shared/data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json"

SYSTEM_PROMPT = (
    "You are an expert academic paper reviewer. Your task is to provide specific, "
    "actionable suggestions for improving research papers."
)

SPECIFIC_SYSTEM_PROMPT = (
    "You are an expert academic paper reviewer. Your task is to provide highly specific, "
    "actionable suggestions that reference concrete elements of the paper by name — sections, "
    "figures, tables, equations, experiments, baselines, or datasets."
)

UNCONDITIONED_PROMPT_TEMPLATE = """Please review the following academic paper and generate exactly 5 specific, actionable suggestions for improving it. Focus on concrete, implementable improvements (e.g., "Add an ablation study comparing X and Y across datasets A and B", not vague statements like "The experiments need improvement").

Paper content:
{paper_content}

Return a JSON array of exactly 5 strings, each being one specific improvement suggestion. Return only the JSON array, no other text."""

CONDITIONED_PROMPT_TEMPLATE = """Please review the following academic paper and generate exactly 5 specific, actionable suggestions for improving it.

Note: This paper has been predicted to be **{decision}** (model confidence: {confidence_pct:.1f}%). Take this signal into account when generating suggestions — if the paper is predicted to be rejected, focus more on fundamental weaknesses; if accepted, focus on strengthening and refining the work.

Focus on concrete, implementable improvements (e.g., "Add an ablation study comparing X and Y across datasets A and B", not vague statements like "The experiments need improvement").

Paper content:
{paper_content}

Return a JSON array of exactly 5 strings, each being one specific improvement suggestion. Return only the JSON array, no other text."""

SPECIFIC_UNCONDITIONED_PROMPT_TEMPLATE = """Please review the following academic paper and generate exactly 5 highly specific, actionable improvement suggestions.

Each suggestion must:
1. Name the exact paper element it targets (e.g. "Section 3.2", "Figure 4", "Table 2", "Equation 5", "the ResNet-50 baseline", "the CIFAR-10 experiment")
2. State precisely what should be added, changed, or removed — not why it matters
3. Be self-contained: a reader should be able to act on the suggestion without reading the full paper

Bad example (too vague): "The experiments need more baselines."
Good example: "In Table 3, add a comparison against DINOv2 and MAE on the ImageNet-1K linear probe benchmark to contextualize the reported 78.4% top-1 accuracy."

Paper content:
{paper_content}

Return a JSON array of exactly 5 strings, each being one specific improvement suggestion. Return only the JSON array, no other text."""

SPECIFIC_CONDITIONED_PROMPT_TEMPLATE = """Please review the following academic paper and generate exactly 5 highly specific, actionable improvement suggestions.

Note: This paper has been predicted to be **{decision}** (model confidence: {confidence_pct:.1f}%). If predicted Reject, prioritize the most fundamental gaps; if Accept, focus on tightening the strongest claims.

Each suggestion must:
1. Name the exact paper element it targets (e.g. "Section 3.2", "Figure 4", "Table 2", "Equation 5", "the ResNet-50 baseline", "the CIFAR-10 experiment")
2. State precisely what should be added, changed, or removed — not why it matters
3. Be self-contained: a reader should be able to act on the suggestion without reading the full paper

Bad example (too vague): "The experiments need more baselines."
Good example: "In Table 3, add a comparison against DINOv2 and MAE on the ImageNet-1K linear probe benchmark to contextualize the reported 78.4% top-1 accuracy."

Paper content:
{paper_content}

Return a JSON array of exactly 5 strings, each being one specific improvement suggestion. Return only the JSON array, no other text."""


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


def compute_confidence_pct(decision: str, decision_logit: float) -> float:
    """Convert log-probability to confidence percentage for the predicted class."""
    if decision_logit is None:
        return 50.0
    prob = math.exp(decision_logit)
    # prob is P(decision token), which directly gives model confidence for that class
    return prob * 100.0


def load_papers_from_dataset(dataset_path: str, target_ids: set, max_chars: int = 60000) -> dict[str, str]:
    """Load paper content from LLaMA-Factory dataset JSON, keyed by submission_id.

    The human conversation turn contains the full paper text after the prompt prefix.
    """
    with open(dataset_path) as f:
        data = json.load(f)

    papers = {}
    for entry in data:
        sub_id = entry.get("_metadata", {}).get("submission_id")
        if sub_id not in target_ids:
            continue
        for conv in entry.get("conversations", []):
            if conv.get("from") == "human":
                content = conv["value"][:max_chars]
                papers[sub_id] = content
                break
    return papers


def build_requests(
    papers: list[dict],
    confidence_map: dict[str, dict],
    variant: str,
    prompt_variant: str = "original",
) -> tuple[list[dict], list[dict]]:
    """Build batch requests for conditioned or unconditioned variant."""
    specific = prompt_variant == "specific"
    system_prompt = SPECIFIC_SYSTEM_PROMPT if specific else SYSTEM_PROMPT
    requests = []
    metadata_list = []

    for paper in papers:
        sub_id = paper["submission_id"]
        paper_content = paper["content"]

        if not paper_content.strip():
            print(f"  WARNING: Empty content for {sub_id}, skipping")
            continue

        if variant == "conditioned":
            conf = confidence_map.get(sub_id)
            if conf is None or conf.get("decision") is None:
                print(f"  WARNING: No confidence for {sub_id}, skipping conditioned variant")
                continue
            decision = conf["decision"]
            decision_logit = conf.get("decision_logit")
            confidence_pct = compute_confidence_pct(decision, decision_logit)
            template = SPECIFIC_CONDITIONED_PROMPT_TEMPLATE if specific else CONDITIONED_PROMPT_TEMPLATE
            prompt = template.format(
                decision=decision,
                confidence_pct=confidence_pct,
                paper_content=paper_content,
            )
        else:  # unconditioned
            template = SPECIFIC_UNCONDITIONED_PROMPT_TEMPLATE if specific else UNCONDITIONED_PROMPT_TEMPLATE
            prompt = template.format(paper_content=paper_content)

        request = {
            "request": {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 4096,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }
        }
        requests.append(request)
        metadata_list.append({"submission_id": sub_id})

    return requests, metadata_list


def parse_suggestions_response(text: str) -> list[str]:
    """Parse the Gemini response into a list of suggestion strings."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(s) for s in result if s]
    except (json.JSONDecodeError, ValueError):
        pass
    return [line.strip("- •*").strip() for line in text.split("\n") if line.strip()]


def retrieve_results(
    job_info: dict,
    output_path: str,
    project: str,
):
    """Retrieve and parse batch results."""
    metadata_list = job_info["metadata_list"]
    output_uri = job_info["output_uri"]

    print(f"Retrieving results from {output_uri}")
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

            suggestions = parse_suggestions_response(text)
            entry = {"submission_id": meta["submission_id"], "suggestions": suggestions}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_results)} results to {output_path}")


def submit_variant(
    papers: list[dict],
    confidence_map: dict[str, dict],
    variant: str,
    args,
    timestamp: str,
    prompt_variant: str = "original",
) -> dict:
    """Build, upload, and submit a batch job for one variant. Returns job_info dict."""
    full_variant = f"{variant}_{prompt_variant}"
    requests, metadata_list = build_requests(papers, confidence_map, variant, prompt_variant=prompt_variant)
    print(f"  {full_variant}: {len(requests)} requests")

    if not requests:
        print(f"  Skipping {full_variant} — no requests to submit.")
        return {}

    local_jsonl = f"/tmp/ai_suggestions_{full_variant}_{timestamp}.jsonl"
    with open(local_jsonl, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    if args.dry_run:
        print(f"  Dry run — not submitting {full_variant}.")
        return {}

    gcs_input_uri = f"{args.gcs_staging}/ai_suggestions_{full_variant}_requests_{timestamp}.jsonl"
    gcs_output_uri = f"{args.gcs_staging}/ai_suggestions_{full_variant}_results_{timestamp}/"
    upload_to_gcs(local_jsonl, gcs_input_uri, args.project)

    client = get_client(args.project, args.location)
    job = client.batches.create(
        model=args.model,
        src=gcs_input_uri,
        config=CreateBatchJobConfig(
            dest=gcs_output_uri,
            display_name=f"alignment_ai_suggestions_{full_variant}_{timestamp}",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="hip-gecko-485003-c4")
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--gcs_staging", type=str, default=DEFAULT_GCS_STAGING)
    parser.add_argument("--ids_file", type=str, default=IDS_FILE)
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH,
                        help="Path to text dataset data.json (contains paper content in human turn)")
    parser.add_argument("--confidence_path", type=str, default=CONFIDENCE_PATH)
    parser.add_argument("--submit_only", action="store_true")
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--prompt_variant", type=str, default="original", choices=["original", "specific"],
        help="Prompt style: 'original' (default) or 'specific' (requires naming paper elements by name)",
    )
    args = parser.parse_args()

    os.makedirs("alignment_analysis/results", exist_ok=True)
    pv = args.prompt_variant

    # --- Load paper IDs ---
    with open(args.ids_file) as f:
        target_ids = set(line.strip() for line in f if line.strip())

    # --- Load paper content from dataset JSON ---
    print(f"Loading paper content from {args.dataset_path}")
    content_map = load_papers_from_dataset(args.dataset_path, target_ids)
    print(f"  Loaded content for {len(content_map)} papers")

    missing = target_ids - set(content_map.keys())
    if missing:
        print(f"  WARNING: {len(missing)} IDs from ids_file not found in dataset")

    papers = [{"submission_id": sid, "content": content} for sid, content in content_map.items()]
    print(f"Loaded {len(papers)} papers")

    # --- Load confidence ---
    confidence_map: dict[str, dict] = {}
    if os.path.exists(args.confidence_path):
        with open(args.confidence_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    confidence_map[entry["submission_id"]] = entry
        print(f"Loaded confidence for {len(confidence_map)} papers")
    else:
        print(f"WARNING: confidence file not found at {args.confidence_path}")

    job_info_conditioned_path = f"alignment_analysis/results/ai_suggestions_conditioned_{pv}_job_info.json"
    job_info_unconditioned_path = f"alignment_analysis/results/ai_suggestions_unconditioned_{pv}_job_info.json"

    # --- RETRIEVE mode ---
    if args.retrieve:
        for variant, job_info_path, output_path in [
            ("conditioned", job_info_conditioned_path,
             f"alignment_analysis/results/ai_suggestions_conditioned_{pv}.jsonl"),
            ("unconditioned", job_info_unconditioned_path,
             f"alignment_analysis/results/ai_suggestions_unconditioned_{pv}.jsonl"),
        ]:
            if os.path.exists(job_info_path):
                with open(job_info_path) as f:
                    job_info = json.load(f)
                retrieve_results(job_info, output_path, args.project)
            else:
                print(f"WARNING: Job info not found for {variant}: {job_info_path}")
        return

    # --- SUBMIT mode ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for variant, job_info_path in [
        ("conditioned", job_info_conditioned_path),
        ("unconditioned", job_info_unconditioned_path),
    ]:
        print(f"\nProcessing variant: {variant} ({pv})")
        job_info = submit_variant(papers, confidence_map, variant, args, timestamp, prompt_variant=pv)
        if job_info:
            with open(job_info_path, "w") as f:
                json.dump(job_info, f, indent=2)
            print(f"  Job info saved to {job_info_path}")

    if args.submit_only or args.dry_run:
        print("\nDone submitting. Run with --retrieve when jobs complete.")
        return

    print("\nPolling for job completion...")
    client = get_client(args.project, args.location)
    for variant, job_info_path in [
        (f"conditioned_{pv}", job_info_conditioned_path),
        (f"unconditioned_{pv}", job_info_unconditioned_path),
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

    print("All jobs complete. Run with --retrieve to fetch results.")


if __name__ == "__main__":
    main()
