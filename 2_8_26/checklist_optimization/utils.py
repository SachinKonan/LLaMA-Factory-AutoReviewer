"""Shared utilities for checklist optimization experiment.

Uses local Qwen/Qwen3-30B-A3B-Thinking-2507 via vLLM for inference
and real ICLR reviews from the HuggingFace Arrow dataset.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import pointbiserialr


# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# HuggingFace dataset (Arrow format) with original_reviews column
DEFAULT_HF_DATASET = (
    "/n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer/data/"
    "hf_dataset_new8_noref_cropped_2017_2026_with_decisions"
)

# LLaMA Factory test split (used to filter to test-set papers)
DEFAULT_TEST_SPLIT = (
    "/n/fs/vision-mix/sk7524/LLaMA-Factory/data/"
    "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test/data.json"
)


# ============================================================================
# vLLM Local Inference
# ============================================================================

def get_vllm_model(
    model_name: str = MODEL_NAME,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
):
    """Load vLLM model for local inference.

    Args:
        model_name: HuggingFace model name
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum sequence length
        gpu_memory_utilization: Ignored (kept for API compatibility)

    Returns:
        vLLM LLM instance
    """
    from vllm import LLM

    return LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        disable_log_stats=True,
    )


def query_vllm(
    prompt: str,
    llm,
    temperature: float = 0.7,
    max_tokens: int = 500,
    system_prompt: Optional[str] = None,
) -> str:
    """Query local vLLM model.

    Args:
        prompt: User prompt
        llm: vLLM LLM instance
        temperature: Sampling temperature
        max_tokens: Max output tokens
        system_prompt: Optional system instruction

    Returns:
        Model response text
    """
    from vllm import SamplingParams

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
    )

    outputs = llm.chat(messages=[messages], sampling_params=sampling_params)

    if outputs and outputs[0].outputs:
        text = outputs[0].outputs[0].text
        # Strip thinking tags if present (Qwen3 thinking model)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return text

    return ""


def query_vllm_batch(
    prompts: List[str],
    llm,
    temperature: float = 0.7,
    max_tokens: int = 500,
    system_prompt: Optional[str] = None,
    show_progress: bool = True,
) -> List[str]:
    """Batch query local vLLM model.

    Args:
        prompts: List of user prompts
        llm: vLLM LLM instance
        temperature: Sampling temperature
        max_tokens: Max output tokens
        system_prompt: Optional system instruction
        show_progress: Show progress

    Returns:
        List of responses (same order as prompts)
    """
    from vllm import SamplingParams

    # Build message lists
    all_messages = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        all_messages.append(messages)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
    )

    if show_progress:
        print(f"  Running batch inference on {len(prompts)} prompts...")

    outputs = llm.chat(messages=all_messages, sampling_params=sampling_params)

    responses = []
    for output in outputs:
        if output.outputs:
            text = output.outputs[0].text
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            responses.append(text)
        else:
            responses.append("")

    return responses


# ============================================================================
# HuggingFace Dataset Loading (Real Reviews)
# ============================================================================

def get_test_submission_ids(test_split_path: str = DEFAULT_TEST_SPLIT) -> set:
    """Load submission IDs from the LLaMA Factory test split."""
    with open(test_split_path, "r") as f:
        data = json.load(f)

    ids = set()
    for sample in data:
        sid = sample.get("_metadata", {}).get("submission_id", "")
        if sid:
            ids.add(sid)

    print(f"Loaded {len(ids)} test submission IDs from {test_split_path}")
    return ids


def load_real_reviews(
    hf_dataset_path: str = DEFAULT_HF_DATASET,
    test_split_path: str = DEFAULT_TEST_SPLIT,
    filter_to_test: bool = True,
) -> List[Dict]:
    """Load real ICLR reviews from the HuggingFace Arrow dataset.

    Each paper has an `original_reviews` column with real OpenReview data.

    Args:
        hf_dataset_path: Path to HF Arrow dataset
        test_split_path: Path to test split JSON for filtering
        filter_to_test: Whether to filter to test-set papers only

    Returns:
        List of review dicts with text, rating, decision, submission_id, year
    """
    try:
        from datasets import load_from_disk
    except ImportError:
        print("Error: datasets library not installed. pip install datasets")
        sys.exit(1)

    print(f"Loading HF dataset from {hf_dataset_path}...")
    ds = load_from_disk(hf_dataset_path)
    print(f"  Total papers in dataset: {len(ds)}")

    test_ids = None
    if filter_to_test:
        test_ids = get_test_submission_ids(test_split_path)

    all_reviews = []
    skipped = 0

    for row in ds:
        submission_id = row["submission_id"]

        if test_ids is not None and submission_id not in test_ids:
            continue

        reviews_json = row.get("original_reviews", "")
        if not reviews_json or reviews_json == "null":
            skipped += 1
            continue

        try:
            reviews = json.loads(reviews_json)
        except (json.JSONDecodeError, TypeError):
            skipped += 1
            continue

        if not isinstance(reviews, list):
            skipped += 1
            continue

        # Get decision (normalize to capitalized: Accept/Reject)
        tech_json = row.get("technical_indicators", "")
        try:
            tech = json.loads(tech_json) if tech_json else {}
        except (json.JSONDecodeError, TypeError):
            tech = {}
        decision = tech.get("binary_decision", "")
        if decision:
            decision = decision.capitalize()  # accept -> Accept, reject -> Reject

        for idx, review in enumerate(reviews):
            if not isinstance(review, dict):
                continue

            # Build text from review fields
            text_parts = []
            for field in ["summary", "strengths", "weaknesses", "questions"]:
                val = review.get(field, "")
                if val:
                    text_parts.append(f"{field}: {val}")

            text = "\n".join(text_parts)
            if len(text) < 50:
                continue

            # Parse rating
            rating_raw = review.get("rating", "")
            rating_match = re.search(r'(\d+)', str(rating_raw))
            rating = int(rating_match.group(1)) if rating_match else None

            all_reviews.append({
                "text": text,
                "review_idx": idx,
                "submission_id": submission_id,
                "year": row.get("year", 0),
                "decision": decision,
                "rating": rating,
                "confidence": review.get("confidence", ""),
                "soundness": review.get("soundness", ""),
                "presentation": review.get("presentation", ""),
                "contribution": review.get("contribution", ""),
            })

    if skipped:
        print(f"  Skipped {skipped} papers with no/invalid reviews")
    print(f"  Parsed {len(all_reviews)} individual reviews")

    return all_reviews


def get_paper_level_data(reviews: List[Dict]) -> List[Dict]:
    """Aggregate reviews to paper level with decision labels.

    Args:
        reviews: List of individual review dicts

    Returns:
        List of paper dicts, each with reviews list and decision
    """
    papers = {}
    for r in reviews:
        sid = r["submission_id"]
        if sid not in papers:
            papers[sid] = {
                "submission_id": sid,
                "decision": r["decision"],
                "year": r["year"],
                "reviews": [],
            }
        papers[sid]["reviews"].append(r)

    return list(papers.values())


# ============================================================================
# Data Loading (Legacy - for backward compatibility with results files)
# ============================================================================

def load_predictions(path: str) -> List[Dict]:
    """Load predictions.jsonl file."""
    predictions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line.strip()))
    return predictions


def load_results(path: str) -> List[Dict]:
    """Load results_single.jsonl file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results


# ============================================================================
# Metrics
# ============================================================================

def compute_point_biserial(binary_answers: List[int], binary_outcomes: List[int]) -> tuple[float, float]:
    """Compute point-biserial correlation."""
    if len(binary_answers) != len(binary_outcomes):
        raise ValueError("Input arrays must have same length")

    if len(set(binary_answers)) == 1 or len(set(binary_outcomes)) == 1:
        return 0.0, 1.0

    return pointbiserialr(binary_answers, binary_outcomes)


def compute_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Compute accuracy."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
    return correct / len(predictions)


# ============================================================================
# File I/O
# ============================================================================

def save_jsonl(data: List[Dict], path: str):
    """Save data as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def save_json(data: Any, path: str, indent: int = 2):
    """Save data as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data
