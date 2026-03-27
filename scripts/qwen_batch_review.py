#!/usr/bin/env python3
"""
Qwen Batch Review Generation — Generate AI reviews for ICLR papers using a
local vLLM server (OpenAI-compatible API) with Qwen3.5-122B-A10B.

Same data, prompts, and JSONL output format as gemini_batch_review.py.
Uses a simple threadpool to call the vLLM server concurrently.

Usage:
    # Start vLLM server first, then:
    python scripts/qwen_batch_review.py \
        --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered \
        --split train --mode blind --candidate_count 3 \
        --api_base http://localhost:8000 --workers 5 \
        --output results/reviews/qwen_blind_text_train.jsonl

    # Test with 100 samples
    python scripts/qwen_batch_review.py \
        --dataset ... --split train --mode blind --dry_run 100 \
        --output results/reviews/qwen_test.jsonl
"""

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from pydantic import BaseModel, Field


# ============================================================================
# CONSTANTS
# ============================================================================

REVIEW_FIELDS_TO_STRIP = {"rating", "decision"}


# ============================================================================
# PYDANTIC SCHEMA (for output validation only, NOT sent to model)
# ============================================================================

class ReviewOutput(BaseModel):
    """Original schema (used by Gemini with structured output enforcement)."""
    strengths: list[str] = Field(description="Top 3 strengths, 3-4 sentences each")
    weaknesses: list[str] = Field(description="Top 3 weaknesses, 3-4 sentences each")
    critical_disputes: str = Field(
        description="What reviewers would likely disagree about, 3-4 sentences"
    )
    most_important_strength: str = Field(
        description="Single most compelling aspect, 3-4 sentences"
    )
    most_important_weakness: str = Field(
        description="Single biggest concern, 3-4 sentences"
    )
    relation_to_other_work: str = Field(
        description="How this paper relates to and builds on prior work, 3-4 sentences"
    )
    paper_quality: str = Field(
        description="Overall 'paper gestalt' — the gut feeling of quality. Consider: Is the writing polished with full, well-structured paragraphs? Are figures professional, clear, and easy to understand? Are tables well-formatted and informative? Does the paper feel like a mature, carefully prepared submission or a rushed draft? 3-4 sentences"
    )


class QwenReviewOutput(BaseModel):
    """Simplified schema for local LLM inference — all fields are plain strings, easier to generate."""
    strengths: str = Field(description="Top 3 strengths, each 3-4 sentences, separated by newlines")
    weaknesses: str = Field(description="Top 3 weaknesses, each 3-4 sentences, separated by newlines")
    critical_disputes: str = Field(description="What reviewers would likely disagree about, 3-4 sentences")
    most_important_strength: str = Field(description="Single most compelling aspect, 3-4 sentences")
    most_important_weakness: str = Field(description="Single biggest concern, 3-4 sentences")
    relation_to_other_work: str = Field(description="How this paper relates to and builds on prior work, 3-4 sentences")
    paper_quality: str = Field(description="Overall paper gestalt — gut feeling of quality, writing polish, figure clarity, 3-4 sentences")
    accept: bool = Field(description="Whether the paper should be accepted")


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

PROMPT_BLIND_TEXT = """\
You are an expert reviewer at a top ML conference in {year}.
It is currently {year}. Evaluate this paper ONLY using knowledge available up to {year}.
Do NOT use hindsight or reference any work published after {year}.
Do NOT predict the acceptance decision.
Note: This paper was converted from PDF to Markdown, so ignore any formatting artifacts from the conversion process — they are not reflective of the authors' original submission quality.
Provide exactly 3 strengths and 3 weaknesses, each 3-4 sentences. All other fields: 3-4 sentences.

{paper_text}"""

PROMPT_BLIND_VISION = """\
You are an expert reviewer at a top ML conference in {year}.
It is currently {year}. Evaluate this paper ONLY using knowledge available up to {year}.
Do NOT use hindsight or reference any work published after {year}.
Do NOT predict the acceptance decision.
Consider visual elements (figures, tables, diagrams) in your analysis where relevant.
Provide exactly 3 strengths and 3 weaknesses, each 3-4 sentences. All other fields: 3-4 sentences.

{paper_text}"""

PROMPT_INFORMED = """\
You are an expert reviewer at a top ML conference in {year}.
It is currently {year}. Evaluate this paper ONLY using knowledge available up to {year}.
Do NOT use hindsight or reference any work published after {year}.
You are also given real peer reviews for reference. Use them to inform your analysis \
but write your own independent assessment. Do NOT predict the acceptance decision.
Note: This paper was converted from PDF to Markdown, so ignore any formatting artifacts from the conversion process — they are not reflective of the authors' original submission quality.
Provide exactly 3 strengths and 3 weaknesses, each 3-4 sentences. All other fields: 3-4 sentences.

{paper_text}

## Peer Reviews (ratings redacted)
{peer_reviews}"""

SYSTEM_MESSAGE = """\
You are an expert ML reviewer. Be direct and critical. Do not soften weaknesses or hedge with qualifiers like "may" or "potentially". If the paper is poor quality, say so plainly.

Example tone for a weak paper (ratings [3,3,1,1]): "The contribution is incremental — the proposed method is a straightforward combination of existing techniques without meaningful novelty. Key baselines are missing and the experimental setup does not support the claims."

Example tone for a strong paper (ratings [10,10,10,10]): "The method is well-motivated and the results convincingly demonstrate improvements over strong baselines. The writing is clear and the evaluation thorough, though coverage of related work in adjacent fields could be improved."

Respond ONLY with a JSON object (no markdown fences, no extra text) with these exact keys (all values are strings except "accept" which is a boolean):
{"strengths": "...", "weaknesses": "...", "critical_disputes": "...", "most_important_strength": "...", "most_important_weakness": "...", "relation_to_other_work": "...", "paper_quality": "...", "accept": true/false}"""


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(data_dir: Path, dataset_name: str, split: str) -> list[dict]:
    dataset_path = data_dir / f"{dataset_name}_{split}" / "data.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples")
    return data


def is_vision_dataset(dataset_name: str) -> bool:
    return "vision" in dataset_name or "clean_images" in dataset_name or "clean+images" in dataset_name


def extract_paper_text(entry: dict) -> str:
    for msg in entry.get("conversations", []):
        if msg.get("from") == "human":
            text = msg["value"]
            match = re.search(r"\n(#\s)", text)
            if match:
                return text[match.start():].strip()
            return text
    return ""


def extract_year(entry: dict) -> int:
    return entry.get("_metadata", {}).get("year", 2024)


def extract_submission_id(entry: dict) -> str:
    return entry.get("_metadata", {}).get("submission_id", "")


# ============================================================================
# HUMAN REVIEW LOADING (informed mode)
# ============================================================================

def load_metadata_reviews(metadata_dir: str) -> dict[str, str]:
    from datasets import load_from_disk

    print(f"Loading metadata reviews from: {metadata_dir}")
    ds = load_from_disk(metadata_dir)
    print(f"  Loaded {len(ds)} metadata entries")

    reviews_lookup = {}
    for row in ds:
        sid = row["submission_id"]
        raw = row.get("original_reviews")
        if not raw:
            continue
        try:
            reviews_list = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        formatted = format_reviews_for_prompt(reviews_list)
        if formatted:
            reviews_lookup[sid] = formatted

    print(f"  Built review lookup for {len(reviews_lookup)} papers")
    return reviews_lookup


def format_reviews_for_prompt(reviews_list: list[dict]) -> str:
    parts = []
    for i, review in enumerate(reviews_list, 1):
        lines = [f"### Review {i}"]
        for key, value in review.items():
            if key.lower() in REVIEW_FIELDS_TO_STRIP:
                continue
            if isinstance(value, str) and value.strip():
                lines.append(f"**{key}**: {value}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ============================================================================
# PROMPT BUILDING
# ============================================================================

def build_prompt_text(entry, mode, is_vision, reviews_lookup=None):
    paper_text = extract_paper_text(entry)
    year = extract_year(entry)
    sid = extract_submission_id(entry)

    if mode == "informed" and reviews_lookup:
        peer_reviews = reviews_lookup.get(sid, "(No peer reviews available)")
        return PROMPT_INFORMED.format(year=year, paper_text=paper_text, peer_reviews=peer_reviews)
    elif is_vision:
        return PROMPT_BLIND_VISION.format(year=year, paper_text=paper_text)
    else:
        return PROMPT_BLIND_TEXT.format(year=year, paper_text=paper_text)


# ============================================================================
# RECORD PREPARATION
# ============================================================================

def prepare_records(data, mode, is_vision, reviews_lookup=None):
    records = []
    for entry in data:
        prompt = build_prompt_text(entry, mode, is_vision, reviews_lookup)
        year = extract_year(entry)
        sid = extract_submission_id(entry)
        split = entry.get("_metadata", {}).get("_split", "")

        records.append({
            "submission_id": sid,
            "year": year,
            "split": split,
            "prompt": prompt,
        })
    return records


# ============================================================================
# vLLM API CALL
# ============================================================================

def parse_review_text(raw_text: str):
    """Try to parse JSON review from raw text. Returns (review_dict, error_str)."""
    text_to_parse = raw_text.strip()
    if text_to_parse.startswith("```"):
        text_to_parse = re.sub(r"^```(?:json)?\s*", "", text_to_parse)
        text_to_parse = re.sub(r"\s*```$", "", text_to_parse)

    # Fix unescaped backslashes: escape any lone \ not followed by a valid JSON escape char
    # Use a negative lookbehind to avoid double-escaping already-escaped \\
    text_to_parse = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu\\])', r'\\\\', text_to_parse)

    review = json.loads(text_to_parse)
    QwenReviewOutput(**review)  # validate
    return review


def call_vllm(record: dict, api_base: str, model: str, temperature: float,
              max_tokens: int, candidate_count: int) -> list[dict]:
    """Call vLLM with n=candidate_count. Returns list of output dicts (one per candidate)."""
    url = f"{api_base.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": record["prompt"]},
        ],
        "n": candidate_count,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    results = []
    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()

        for choice in data["choices"]:
            raw_text = choice["message"]["content"] or ""
            review = None
            error = None
            try:
                review = parse_review_text(raw_text)
            except (json.JSONDecodeError, ValueError) as e:
                error = f"Parse error: {e}"
            except Exception as e:
                error = f"Validation error: {e}"

            results.append({
                "submission_id": record["submission_id"],
                "year": record["year"],
                "split": record["split"],
                "candidate_idx": choice["index"],
                "review": review,
                "raw_text": raw_text,
                "error": error,
            })
    except requests.exceptions.RequestException as e:
        # HTTP failure — return one error entry per candidate
        for i in range(candidate_count):
            results.append({
                "submission_id": record["submission_id"],
                "year": record["year"],
                "split": record["split"],
                "candidate_idx": i,
                "review": None,
                "raw_text": "",
                "error": f"HTTP error: {e}",
            })

    return results


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AI reviews via vLLM server (OpenAI-compatible API)",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", nargs="+", default=["train", "test"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--mode", choices=["blind", "informed"], default="blind")
    parser.add_argument("--metadata_dir")

    parser.add_argument("--model", default="default", help="Model name for API payload (default: auto-detect from /v1/models)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--candidate_count", type=int, default=1)

    parser.add_argument("--api_base", default="http://localhost:8000")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent worker threads (default: 5)")

    parser.add_argument("--shard_idx", type=int, default=None,
                        help="Shard index (default: $SLURM_ARRAY_TASK_ID or 0)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards (default: 1 = no sharding)")
    parser.add_argument("--shard_subset", default=None,
                        help="Slice of global records for this partition, e.g. '0:11700' or '11700:'")
    parser.add_argument("--shard_prefix", default=None,
                        help="Prefix for shard output files to avoid collisions, e.g. 'ailab' or 'gputest'")
    parser.add_argument("--retry_file", default=None,
                        help="File with submission_ids to retry (one per line)")
    parser.add_argument("--wait_for_server", type=int, default=0,
                        help="Poll /v1/models every 10s until server is ready, up to N seconds (0=disabled)")

    parser.add_argument("--output", required=True)
    parser.add_argument("--dry_run", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    is_vision = is_vision_dataset(args.dataset)

    # Wait for vLLM server to be ready
    if args.wait_for_server > 0:
        models_url = f"{args.api_base.rstrip('/')}/v1/models"
        deadline = time.time() + args.wait_for_server
        print(f"Waiting up to {args.wait_for_server}s for vLLM server at {models_url} ...")
        while time.time() < deadline:
            try:
                r = requests.get(models_url, timeout=5)
                if r.status_code == 200:
                    print(f"  Server ready after {args.wait_for_server - int(deadline - time.time())}s")
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(10)
        else:
            print(f"  WARNING: server not ready after {args.wait_for_server}s — proceeding anyway")

    # Auto-detect model name from vLLM server
    if args.model == "default":
        try:
            resp = requests.get(f"{args.api_base.rstrip('/')}/v1/models", timeout=10)
            resp.raise_for_status()
            models = resp.json()["data"]
            args.model = models[0]["id"]
            print(f"Auto-detected model: {args.model}")
        except Exception as e:
            print(f"Warning: could not auto-detect model ({e}), using 'default'")

    print(f"=== Qwen Batch Review Generation ===")
    print(f"  Dataset: {args.dataset}")
    print(f"  Splits: {args.split}")
    print(f"  Mode: {args.mode}")
    print(f"  Model: {args.model}")
    print(f"  API: {args.api_base}")
    print(f"  Workers: {args.workers}")
    print(f"  Candidate count: {args.candidate_count}")
    print(f"  Temperature: {args.temperature}")
    if args.dry_run:
        print(f"  DRY RUN: {args.dry_run} samples")

    # Load human reviews if informed mode
    reviews_lookup = None
    if args.mode == "informed":
        if not args.metadata_dir:
            raise ValueError("--metadata_dir required for informed mode")
        reviews_lookup = load_metadata_reviews(args.metadata_dir)

    # Load and prepare data
    all_records = []
    for split in args.split:
        data = load_dataset(data_dir, args.dataset, split)
        if args.dry_run:
            data = data[:args.dry_run]
            print(f"  (dry run: truncated to {len(data)} samples)")
        records = prepare_records(data, args.mode, is_vision, reviews_lookup)
        all_records.extend(records)

    # Retry filter: keep only papers from the retry file
    if args.retry_file:
        with open(args.retry_file) as f:
            retry_sids = {line.strip() for line in f if line.strip()}
        before = len(all_records)
        all_records = [r for r in all_records if r["submission_id"] in retry_sids]
        print(f"  Retry filter: {len(all_records)} papers (from {len(retry_sids)} IDs, was {before})")

    # Sharding (block-based with optional subset for partition splitting)
    if args.shard_idx is None:
        args.shard_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if args.shard_subset:
        parts = args.shard_subset.split(":")
        s = int(parts[0]) if parts[0] else 0
        e = int(parts[1]) if parts[1] else len(all_records)
        all_records = all_records[s:e]
        print(f"  Subset [{s}:{e}]: {len(all_records)} papers")
    if args.num_shards > 1:
        total = len(all_records)
        shard_size = math.ceil(total / args.num_shards)
        start = args.shard_idx * shard_size
        end = min(start + shard_size, total)
        all_records = all_records[start:end]
        print(f"  Shard {args.shard_idx}/{args.num_shards}: records [{start}:{end}] = {len(all_records)} papers")
        if "_shard" not in args.output:
            base, ext = os.path.splitext(args.output)
            prefix = f"_{args.shard_prefix}" if args.shard_prefix else ""
            args.output = f"{base}{prefix}_shard{args.shard_idx}{ext}"

    print(f"\nTotal records (papers): {len(all_records)}")
    print(f"  Each gets n={args.candidate_count} completions")
    if not all_records:
        print("No records to process.")
        return

    # Warmup: send one short request to trigger CUDA graph compilation before blasting
    print("\nWarming up vLLM server...")
    try:
        warmup_resp = requests.post(
            f"{args.api_base.rstrip('/')}/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": "Say OK."}],
                "max_tokens": 8,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=300,
        )
        warmup_resp.raise_for_status()
        print("  Warmup OK")
    except Exception as e:
        print(f"  Warmup failed: {e} (continuing anyway)")

    # Run with threadpool
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t0 = time.time()
    total_papers = 0
    total_candidates = 0
    parsed_ok = 0
    errors = 0

    print(f"\nProcessing with {args.workers} workers...")
    with open(args.output, "w") as f, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:

        futures = {
            pool.submit(call_vllm, rec, args.api_base, args.model,
                        args.temperature, args.max_tokens, args.candidate_count): i
            for i, rec in enumerate(all_records)
        }

        for future in as_completed(futures):
            results = future.result()  # list of dicts, one per candidate
            total_papers += 1

            for result in results:
                total_candidates += 1
                if result["review"] is not None:
                    parsed_ok += 1
                if result.get("error"):
                    errors += 1
                    if errors <= 5:
                        print(f"  Error [{result['submission_id']}]: {result['error']}")

                # Remove error field before writing
                result.pop("error", None)
                f.write(json.dumps(result) + "\n")
            f.flush()

            if total_papers % 10 == 0 or total_papers == len(all_records):
                elapsed = time.time() - t0
                rate = total_papers / elapsed if elapsed > 0 else 0
                print(f"  [{total_papers}/{len(all_records)} papers] "
                      f"{parsed_ok}/{total_candidates} parsed ok, {errors} err, "
                      f"{rate:.1f} papers/s, {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nDone! {total_papers} papers, {total_candidates} candidates in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Parsed OK: {parsed_ok}/{total_candidates}, Errors: {errors}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
