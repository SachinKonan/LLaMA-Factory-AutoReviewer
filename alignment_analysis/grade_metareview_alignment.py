#!/usr/bin/env python3
"""
Grade which human reviewer suggestions are mentioned/addressed in the metareview.

For each paper, sends a Gemini batch request asking: "Given this metareview, which of these
human reviewer suggestions are explicitly addressed or reflected in it?"

Outputs metareview_alignment.jsonl. A follow-up stats section (--stats_only) cross-tabs
these results against the AI alignment matches to answer:
  "Does the AI preferentially match human suggestions that the AC also found important?"

Usage:
    # Preview one prompt without submitting
    python alignment_analysis/grade_metareview_alignment.py --dry_run

    # Submit batch job
    python alignment_analysis/grade_metareview_alignment.py --submit_only

    # Retrieve after job completes
    python alignment_analysis/grade_metareview_alignment.py --retrieve

    # Compute stats (no Gemini needed)
    python alignment_analysis/grade_metareview_alignment.py --stats_only
"""

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime

from google import genai
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig, HttpOptions

DEFAULT_PROJECT = "hip-gecko-485003-c4"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_GCS_STAGING = "gs://jl0796-autoreviewer-staging/alignment_analysis"

RESULTS_DIR = "alignment_analysis/results"
HUMAN_SUGGESTIONS_PATH = f"{RESULTS_DIR}/human_suggestions.jsonl"
METADATA_CSV = "/scratch/gpfs/ZHUANGL/jl0796/shared/data/massive_metadata_v7_5.csv"
OUTPUT_PATH = f"{RESULTS_DIR}/metareview_alignment.jsonl"
JOB_INFO_PATH = f"{RESULTS_DIR}/metareview_alignment_job_info.json"

SYSTEM_PROMPT = (
    "You are an expert at analyzing academic paper reviews and meta-reviews. "
    "Your task is to determine which reviewer suggestions are reflected in an area chair's meta-review."
)

GRADING_PROMPT_TEMPLATE = """You are analyzing whether human reviewer improvement suggestions are addressed or reflected in an area chair's meta-review.

Meta-review:
{metareview_text}

Human reviewer improvement suggestions:
{suggestions_numbered}

For each suggestion (by index), determine whether it is explicitly mentioned, addressed, or clearly reflected in the meta-review above.

Rules:
- Mark true if the meta-review explicitly mentions the same concern, issue, or improvement request, even if worded differently.
- Mark false if the meta-review does not address this suggestion at all, or only discusses it in passing in the context of strengths.
- Be conservative: only mark true if there is a clear connection.

Return a JSON object mapping each suggestion index (as a string) to true or false.
Example for 3 suggestions: {{"0": true, "1": false, "2": true}}

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


def load_metareview_map(csv_path: str, sub_ids: set) -> dict[str, str]:
    """Load normalized_metareview from metadata CSV for given submission IDs."""
    result = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["submission_id"]
            if sid not in sub_ids:
                continue
            raw = row.get("normalized_metareview", "").strip()
            if not raw:
                continue
            # Parse JSON if it's a JSON object with summary/strengths/weaknesses
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    parts = []
                    if parsed.get("summary"):
                        parts.append(f"Summary: {parsed['summary']}")
                    if parsed.get("weaknesses"):
                        parts.append(f"Weaknesses / Areas for improvement: {parsed['weaknesses']}")
                    if parsed.get("strengths"):
                        parts.append(f"Strengths: {parsed['strengths']}")
                    raw = "\n\n".join(parts)
            except (json.JSONDecodeError, TypeError):
                pass  # use raw text as-is
            result[sid] = raw
    return result


def load_flat_human_suggestions(path: str) -> dict[str, list[str]]:
    result = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sub_id = entry["submission_id"]
            flat = []
            for reviewer_list in entry.get("reviewer_suggestions", []):
                flat.extend(reviewer_list)
            result[sub_id] = flat
    return result


def build_request(sub_id: str, suggestions: list[str], metareview_text: str) -> tuple[dict, dict]:
    suggestions_numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(suggestions))
    prompt = GRADING_PROMPT_TEMPLATE.format(
        metareview_text=metareview_text,
        suggestions_numbered=suggestions_numbered,
    )
    request = {
        "request": {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 1024,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
    }
    metadata = {"submission_id": sub_id, "human_suggestions": suggestions}
    return request, metadata


def parse_response(text: str, n: int) -> dict[str, bool]:
    """Parse Gemini response into {str(idx): bool} for n suggestions."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            out = {}
            for i in range(n):
                val = result.get(str(i))
                out[str(i)] = bool(val) if val is not None else False
            return out
    except (json.JSONDecodeError, ValueError):
        pass
    return {str(i): False for i in range(n)}


def submit(args):
    human_map = load_flat_human_suggestions(HUMAN_SUGGESTIONS_PATH)
    sub_ids = set(human_map.keys())
    meta_map = load_metareview_map(METADATA_CSV, sub_ids)

    missing_meta = sub_ids - set(meta_map.keys())
    if missing_meta:
        print(f"WARNING: {len(missing_meta)} papers have no metareview, skipping them")

    requests, metadata_list = [], []
    for sub_id, suggestions in human_map.items():
        metareview_text = meta_map.get(sub_id)
        if not metareview_text:
            continue
        req, meta = build_request(sub_id, suggestions, metareview_text)
        requests.append(req)
        metadata_list.append(meta)

    print(f"Built {len(requests)} requests")

    if args.dry_run:
        # Print the first request's prompt in full for review
        print("\n" + "=" * 70)
        print("DRY RUN — sample prompt (paper 1):")
        print("=" * 70)
        first_prompt = requests[0]["request"]["contents"][0]["parts"][0]["text"]
        print(first_prompt)
        print("=" * 70)
        print("\nGenerationConfig:", requests[0]["request"]["generationConfig"])
        print("\nNot submitting. Re-run without --dry_run to submit.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_jsonl = f"/tmp/metareview_alignment_requests_{timestamp}.jsonl"
    with open(local_jsonl, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    gcs_input_uri = f"{args.gcs_staging}/metareview_alignment_requests_{timestamp}.jsonl"
    gcs_output_uri = f"{args.gcs_staging}/metareview_alignment_results_{timestamp}/"
    upload_to_gcs(local_jsonl, gcs_input_uri, args.project)

    client = get_client(args.project, args.location)
    job = client.batches.create(
        model=args.model,
        src=gcs_input_uri,
        config=CreateBatchJobConfig(
            dest=gcs_output_uri,
            display_name=f"metareview_alignment_{timestamp}",
        ),
    )
    print(f"Submitted job: {job.name}")

    job_info = {
        "job_name": job.name,
        "output_uri": gcs_output_uri,
        "submitted_at": timestamp,
        "metadata_list": metadata_list,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(JOB_INFO_PATH, "w") as f:
        json.dump(job_info, f, indent=2)
    print(f"Job info saved to {JOB_INFO_PATH}")


def retrieve(args):
    with open(JOB_INFO_PATH) as f:
        job_info = json.load(f)

    metadata_list = job_info["metadata_list"]
    output_uri = job_info["output_uri"]
    print(f"Retrieving from {output_uri}")

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

    with open(OUTPUT_PATH, "w") as f:
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

            n = len(meta["human_suggestions"])
            in_metareview = parse_response(text, n)
            meta_count = sum(1 for v in in_metareview.values() if v)

            entry = {
                "submission_id": meta["submission_id"],
                "human_suggestions": meta["human_suggestions"],
                "in_metareview": in_metareview,
                "metareview_coverage": meta_count / n if n > 0 else 0.0,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved to {OUTPUT_PATH}")


def stats(args):
    """Cross-tab metareview alignment with AI alignment matches."""
    if not os.path.exists(OUTPUT_PATH):
        print(f"ERROR: {OUTPUT_PATH} not found. Run --retrieve first.")
        return

    # Load metareview alignment
    meta_align = {}
    with open(OUTPUT_PATH) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                meta_align[entry["submission_id"]] = entry

    print(f"Loaded metareview alignment for {len(meta_align)} papers")

    # Overall metareview coverage stats
    coverages = [e["metareview_coverage"] for e in meta_align.values()]
    n_papers = len(coverages)
    avg_cov = sum(coverages) / n_papers if n_papers else 0
    n_total_suggs = sum(len(e["human_suggestions"]) for e in meta_align.values())
    n_in_meta = sum(
        sum(1 for v in e["in_metareview"].values() if v)
        for e in meta_align.values()
    )
    print(f"\n=== METAREVIEW COVERAGE ===")
    print(f"  Papers with metareview: {n_papers}")
    print(f"  Total human suggestions: {n_total_suggs}")
    print(f"  In metareview:           {n_in_meta} ({n_in_meta/n_total_suggs*100:.1f}%)")
    print(f"  Avg coverage per paper:  {avg_cov:.3f}")

    # Cross-tab: for each alignment variant, are AI-matched suggestions more likely in metareview?
    alignment_files = {
        "original strict  | conditioned  ": f"{RESULTS_DIR}/alignment_matches_conditioned_strict.jsonl",
        "original strict  | unconditioned": f"{RESULTS_DIR}/alignment_matches_unconditioned_strict.jsonl",
        "original lenient | conditioned  ": f"{RESULTS_DIR}/alignment_matches_conditioned_lenient.jsonl",
        "original lenient | unconditioned": f"{RESULTS_DIR}/alignment_matches_unconditioned_lenient.jsonl",
        "specific strict  | conditioned  ": f"{RESULTS_DIR}/alignment_matches_conditioned_specific_strict.jsonl",
        "specific lenient | conditioned  ": f"{RESULTS_DIR}/alignment_matches_conditioned_specific_lenient.jsonl",
        "specific lenient | unconditioned": f"{RESULTS_DIR}/alignment_matches_unconditioned_specific_lenient.jsonl",
    }

    print(f"\n=== AI MATCH × METAREVIEW CROSS-TAB ===")
    print(f"{'Variant':<45}  {'P(meta|matched)':>15}  {'P(meta|unmatched)':>17}  {'lift':>6}")
    print("-" * 90)

    for label, path in alignment_files.items():
        if not os.path.exists(path):
            print(f"{label:<45}  MISSING")
            continue

        matched_in_meta, matched_not_meta = 0, 0
        unmatched_in_meta, unmatched_not_meta = 0, 0

        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                sub_id = entry["submission_id"]
                if sub_id not in meta_align:
                    continue

                meta_entry = meta_align[sub_id]
                in_meta = meta_entry["in_metareview"]  # {str(idx): bool}
                matches = entry["matches"]  # {str(ai_pos): human_idx or null}

                # Which human indices were AI-matched?
                ai_matched_human = set()
                for ai_pos, h_idx in matches.items():
                    if h_idx is not None:
                        ai_matched_human.add(str(h_idx))

                n_human = len(entry["human_suggestions"])
                for i in range(n_human):
                    in_m = in_meta.get(str(i), False)
                    ai_matched = i in {int(x) for x in ai_matched_human}
                    if ai_matched:
                        if in_m:
                            matched_in_meta += 1
                        else:
                            matched_not_meta += 1
                    else:
                        if in_m:
                            unmatched_in_meta += 1
                        else:
                            unmatched_not_meta += 1

        total_matched = matched_in_meta + matched_not_meta
        total_unmatched = unmatched_in_meta + unmatched_not_meta
        p_meta_given_matched = matched_in_meta / total_matched if total_matched > 0 else 0
        p_meta_given_unmatched = unmatched_in_meta / total_unmatched if total_unmatched > 0 else 0
        lift = p_meta_given_matched / p_meta_given_unmatched if p_meta_given_unmatched > 0 else float("nan")

        print(
            f"{label:<45}  {p_meta_given_matched:>15.3f}  {p_meta_given_unmatched:>17.3f}  {lift:>6.2f}"
        )

    # Save per-paper stats
    save_path = f"{RESULTS_DIR}/metareview_alignment_stats.csv"
    rows = []
    for sub_id, entry in meta_align.items():
        n = len(entry["human_suggestions"])
        n_in = sum(1 for v in entry["in_metareview"].values() if v)
        rows.append({
            "submission_id": sub_id,
            "n_human": n,
            "n_in_metareview": n_in,
            "metareview_coverage": entry["metareview_coverage"],
        })
    import csv as csv_mod
    with open(save_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-paper stats saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gcs_staging", default=DEFAULT_GCS_STAGING)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print a sample prompt and exit without submitting")
    parser.add_argument("--submit_only", action="store_true")
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--stats_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.stats_only:
        stats(args)
    elif args.retrieve:
        retrieve(args)
        stats(args)
    else:
        submit(args)


if __name__ == "__main__":
    main()
