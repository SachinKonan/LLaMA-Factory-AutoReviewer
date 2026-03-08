#!/usr/bin/env python3
"""
Stage 2: Filter Questions by Enforceability and Answerability.

Tests each candidate question on real ICLR reviews using local Qwen3-30B-A3B:
1. Enforceability: Consistent answers across multiple trials
2. Answerability: Can LLM provide yes/no answers (not "unclear")
3. Variability: Not always yes or always no

Usage:
    # Full filtering (needs 2x L40 GPUs)
    python 2_8_26/checklist_optimization/stage2_filter_questions.py \
        --input_questions 2_8_26/checklist_optimization/data/candidate_questions.jsonl \
        --output 2_8_26/checklist_optimization/data/filtered_questions.jsonl \
        --consistency_threshold 0.80 --n_test_reviews 5 --n_repeats 3

    # Debug mode
    python 2_8_26/checklist_optimization/stage2_filter_questions.py \
        --input_questions 2_8_26/checklist_optimization/data/candidate_questions.jsonl \
        --output 2_8_26/checklist_optimization/data/filtered_questions.jsonl \
        --consistency_threshold 0.60 --n_test_reviews 2 --n_repeats 2 --debug
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_vllm_model,
    load_jsonl,
    load_real_reviews,
    query_vllm_batch,
    save_json,
    save_jsonl,
)


# ============================================================================
# Enforceability Testing
# ============================================================================

def test_enforceability(
    question: dict,
    reviews: list[dict],
    n_repeats: int = 3,
    llm=None,
) -> dict:
    """Test enforceability of a question across reviews.

    Enforceability = consistency of answers when asked multiple times.
    """
    question_text = question.get("text", "")
    consistency_scores = []
    all_answers = []

    # Build all prompts at once for batch processing
    prompts = []
    prompt_indices = []  # (review_idx, repeat_idx)

    for r_idx, review in enumerate(reviews):
        review_text = review["text"][:2000]
        for rep in range(n_repeats):
            prompt = f"""Review: {review_text}

Question: {question_text}

Answer only "Yes" or "No" based strictly on the review content above."""
            prompts.append(prompt)
            prompt_indices.append((r_idx, rep))

    # Batch query
    responses = query_vllm_batch(
        prompts, llm, temperature=0.3, max_tokens=10, show_progress=False
    )

    # Parse and organize answers by review
    review_answers = {i: [] for i in range(len(reviews))}
    for (r_idx, _), response in zip(prompt_indices, responses):
        response_lower = response.lower().strip()
        if "yes" in response_lower:
            answer = "Yes"
        elif "no" in response_lower:
            answer = "No"
        else:
            answer = "Unclear"
        review_answers[r_idx].append(answer)
        all_answers.append(answer)

    # Compute consistency per review
    for r_idx in range(len(reviews)):
        answers = review_answers[r_idx]
        if answers:
            counts = Counter(answers)
            majority_count = max(counts.values())
            consistency_scores.append(majority_count / len(answers))

    if consistency_scores:
        answer_dist = Counter(all_answers)
        total = len(all_answers)
        return {
            "avg_consistency": float(np.mean(consistency_scores)),
            "consistency_scores": [float(x) for x in consistency_scores],
            "answer_distribution": dict(answer_dist),
            "fraction_yes": answer_dist.get("Yes", 0) / total if total > 0 else 0,
            "fraction_no": answer_dist.get("No", 0) / total if total > 0 else 0,
            "fraction_unclear": answer_dist.get("Unclear", 0) / total if total > 0 else 0,
        }
    else:
        return {
            "avg_consistency": 0.0,
            "consistency_scores": [],
            "answer_distribution": {},
            "fraction_yes": 0.0,
            "fraction_no": 0.0,
            "fraction_unclear": 0.0,
        }


# ============================================================================
# Filtering Logic
# ============================================================================

def filter_questions(
    questions: list[dict],
    reviews: list[dict],
    consistency_threshold: float = 0.80,
    n_test_reviews: int = 5,
    n_repeats: int = 3,
    llm=None,
) -> tuple[list[dict], dict]:
    """Filter questions based on enforceability criteria."""
    random.seed(42)
    test_reviews = random.sample(reviews, min(n_test_reviews, len(reviews)))

    filtered = []
    enforceability_scores = {}

    for i, question in enumerate(questions):
        qid = question.get("question_id", f"q{i + 1:03d}")
        print(f"\nTesting question {i + 1}/{len(questions)}: {qid}")
        print(f"  {question.get('text', '')}")

        metrics = test_enforceability(question, test_reviews, n_repeats, llm)
        enforceability_scores[qid] = metrics

        keep = True
        reasons = []

        if metrics["avg_consistency"] < consistency_threshold:
            keep = False
            reasons.append(f"low_consistency ({metrics['avg_consistency']:.2f})")

        if metrics["fraction_unclear"] > 0.2:
            keep = False
            reasons.append(f"too_many_unclear ({metrics['fraction_unclear']:.2f})")

        if metrics["fraction_yes"] > 0.9:
            keep = False
            reasons.append(f"always_yes ({metrics['fraction_yes']:.2f})")
        if metrics["fraction_no"] > 0.9:
            keep = False
            reasons.append(f"always_no ({metrics['fraction_no']:.2f})")

        if keep:
            filtered.append(question)
            print(f"  KEEP (consistency: {metrics['avg_consistency']:.2f})")
        else:
            print(f"  REJECT ({', '.join(reasons)})")

    return filtered, enforceability_scores


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Filter questions by enforceability")
    parser.add_argument("--input_questions", type=str, required=True, help="Path to candidate_questions.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output path for filtered_questions.jsonl")
    parser.add_argument("--consistency_threshold", type=float, default=0.80)
    parser.add_argument("--n_test_reviews", type=int, default=5, help="Reviews to test per question")
    parser.add_argument("--n_repeats", type=int, default=3, help="Repeats per question per review")
    parser.add_argument("--hf_dataset", type=str, default=None)
    parser.add_argument("--test_split", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.consistency_threshold = 0.60
        args.n_test_reviews = min(args.n_test_reviews, 2)
        args.n_repeats = min(args.n_repeats, 2)
        print("DEBUG MODE")
        print(f"  consistency_threshold: {args.consistency_threshold}")
        print(f"  n_test_reviews: {args.n_test_reviews}")
        print(f"  n_repeats: {args.n_repeats}")

    # Load questions
    print(f"\nLoading candidate questions from: {args.input_questions}")
    questions = load_jsonl(args.input_questions)
    print(f"  Loaded {len(questions)} questions")

    # Load real reviews
    print(f"\nLoading real ICLR reviews...")
    kwargs = {}
    if args.hf_dataset:
        kwargs["hf_dataset_path"] = args.hf_dataset
    if args.test_split:
        kwargs["test_split_path"] = args.test_split
    reviews = load_real_reviews(**kwargs)

    # Load vLLM model
    print(f"\nLoading vLLM model...")
    llm = get_vllm_model(tensor_parallel_size=args.tensor_parallel_size)

    # Filter questions
    print(f"\nFiltering questions (threshold: {args.consistency_threshold})...")
    filtered, scores = filter_questions(
        questions, reviews,
        consistency_threshold=args.consistency_threshold,
        n_test_reviews=args.n_test_reviews,
        n_repeats=args.n_repeats,
        llm=llm,
    )

    # Save filtered questions
    print(f"\nFiltered {len(questions)} → {len(filtered)} questions ({len(filtered) / len(questions):.1%} kept)")
    save_jsonl(filtered, args.output)
    print(f"Saved to: {args.output}")

    # Save enforceability scores
    scores_path = args.output.replace("filtered_questions.jsonl", "").rstrip("/")
    scores_path = scores_path.replace("/data", "/results") + "/enforceability_scores.json"
    save_json(scores, scores_path)
    print(f"Saved enforceability scores to: {scores_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
