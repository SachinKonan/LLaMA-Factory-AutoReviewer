#!/usr/bin/env python3
"""
Evaluate Reviews with Checklist.

Applies the optimal checklist from checklist_optimization to ALL individual reviews
(not paper-level aggregation). Each review gets its own checkmark score.

Usage:
    python 2_8_26/training_checklist/scripts/evaluate_reviews.py \
        --checklist 2_8_26/checklist_optimization/data/optimal_checklist.json \
        --output 2_8_26/training_checklist/data/review_evaluations.jsonl \
        --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
        --batch_size 64
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "checklist_optimization"))

from utils import (
    get_vllm_model,
    load_json,
    load_real_reviews,
    query_vllm_batch,
    save_jsonl,
)


def answer_questions_for_reviews(
    questions: list[dict],
    reviews: list[dict],
    llm,
    batch_size: int = 64,
) -> list[dict]:
    """
    Apply checklist questions to each individual review using vLLM.

    Args:
        questions: List of question dicts from optimal checklist
        reviews: List of individual review dicts (from HF dataset)
        llm: vLLM model instance
        batch_size: Number of prompts per batch

    Returns:
        List of evaluation entries (one per review) with checkmark scores
    """
    # Build question text for prompt
    question_list = []
    qids = []
    for i, q in enumerate(questions, 1):
        qid = q.get("id", q.get("question_id", f"q{i:03d}"))
        qids.append(qid)
        text = q.get("text", "")
        question_list.append(f'{i}. [{qid}] {text}')
    questions_text = "\n".join(question_list)

    print(f"Applying {len(questions)} checklist questions to {len(reviews)} reviews...")
    print(f"Questions:\n{questions_text}\n")

    # Build all prompts
    all_prompts = []
    for review in reviews:
        review_text = review["text"][:3000]  # Truncate long reviews
        prompt = f"""You are analyzing an academic paper review. Answer questions about what topics the review DISCUSSES or MENTIONS (not what opinions it expresses).

Review:
{review_text}

Questions (answer "Yes" if the review discusses/mentions this topic, "No" if it doesn't):

{questions_text}

Respond in JSON format with ONLY Yes or No for each question:
{{
  "answers": [
    {{"question_id": "q001", "answer": "Yes"}},
    {{"question_id": "q002", "answer": "No"}},
    ...
  ]
}}"""
        all_prompts.append(prompt)

    # Process in batches
    evaluations = []
    for batch_start in range(0, len(all_prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(all_prompts))
        batch_prompts = all_prompts[batch_start:batch_end]

        print(f"  Processing reviews {batch_start + 1}-{batch_end}/{len(all_prompts)}...")
        responses = query_vllm_batch(batch_prompts, llm, temperature=0.1, max_tokens=2000, show_progress=False)

        for j, response in enumerate(responses):
            review_idx = batch_start + j
            review = reviews[review_idx]

            # Parse JSON response
            try:
                # Extract JSON from markdown code blocks
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    response = response[json_start:json_end].strip()
                elif "```" in response:
                    json_start = response.find("```") + 3
                    json_end = response.find("```", json_start)
                    response = response[json_start:json_end].strip()

                # Find JSON object boundaries
                brace_start = response.find("{")
                brace_end = response.rfind("}") + 1
                if brace_start >= 0 and brace_end > brace_start:
                    response = response[brace_start:brace_end]

                data = json.loads(response)
                answers_list = data.get("answers", [])

                # Convert list to dict
                answers = {}
                for ans in answers_list:
                    qid = ans.get("question_id", "")
                    answer = ans.get("answer", "No")
                    answers[qid] = answer

            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Warning: Failed to parse response for review {review_idx}: {e}")
                # Default to "No" for all questions
                answers = {qid: "No" for qid in qids}

            # Compute checkmark score
            yes_count = sum(1 for qid in qids if answers.get(qid, "No") == "Yes")
            checkmark_score = yes_count / len(qids) if qids else 0.0

            evaluations.append({
                "submission_id": review.get("submission_id", ""),
                "review_idx": review.get("review_idx", review_idx),
                "review_text": review.get("text", "")[:500],  # Truncated for storage
                "rating": review.get("rating"),
                "decision": review.get("decision", ""),
                "year": review.get("year", 0),
                "confidence": review.get("confidence", ""),
                "soundness": review.get("soundness", ""),
                "presentation": review.get("presentation", ""),
                "contribution": review.get("contribution", ""),
                "checklist_answers": answers,
                "checkmark_count": yes_count,
                "checkmark_score": checkmark_score,
            })

    return evaluations


def main():
    parser = argparse.ArgumentParser(description="Evaluate reviews with checklist")
    parser.add_argument(
        "--checklist",
        type=str,
        required=True,
        help="Path to optimal_checklist.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output review_evaluations.jsonl",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Thinking-2507",
        help="vLLM model to use for evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for vLLM inference",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only process first 100 reviews",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("EVALUATE REVIEWS WITH CHECKLIST")
    print("=" * 80)

    # Load optimal checklist
    print(f"\nLoading checklist from {args.checklist}...")
    checklist = load_json(args.checklist)
    questions = checklist.get("questions", [])
    print(f"  Loaded {len(questions)} questions")

    # Load all individual reviews from HF dataset
    print("\nLoading individual reviews from HF dataset...")
    reviews = load_real_reviews(
        filter_to_test=True,  # Only test set reviews
    )
    print(f"  Loaded {len(reviews)} individual reviews")

    if args.debug:
        print(f"\n[DEBUG MODE] Processing only first 100 reviews")
        reviews = reviews[:100]

    # Initialize vLLM model
    print(f"\nInitializing vLLM model: {args.model}...")
    llm = get_vllm_model(args.model)

    # Apply checklist to all reviews
    print("\nApplying checklist to reviews...")
    evaluations = answer_questions_for_reviews(
        questions=questions,
        reviews=reviews,
        llm=llm,
        batch_size=args.batch_size,
    )

    # Save results
    print(f"\nSaving results to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(evaluations, str(output_path))

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total reviews evaluated: {len(evaluations)}")

    scores = [e["checkmark_score"] for e in evaluations]
    print(f"\nCheckmark score distribution:")
    print(f"  Min:    {min(scores):.3f}")
    print(f"  Q1:     {sorted(scores)[len(scores)//4]:.3f}")
    print(f"  Median: {sorted(scores)[len(scores)//2]:.3f}")
    print(f"  Q3:     {sorted(scores)[3*len(scores)//4]:.3f}")
    print(f"  Max:    {max(scores):.3f}")
    print(f"  Mean:   {sum(scores)/len(scores):.3f}")

    # Top 20% threshold
    threshold_80 = sorted(scores)[int(len(scores) * 0.8)]
    print(f"\n80th percentile (top 20% threshold): {threshold_80:.3f}")
    top_20_pct = sum(1 for s in scores if s >= threshold_80)
    print(f"Reviews in top 20%: {top_20_pct}")

    # Decision distribution
    decisions = [e["decision"] for e in evaluations if e["decision"]]
    accept_count = sum(1 for d in decisions if d == "Accept")
    print(f"\nDecision distribution:")
    print(f"  Accept: {accept_count} ({accept_count/len(decisions)*100:.1f}%)")
    print(f"  Reject: {len(decisions)-accept_count} ({(len(decisions)-accept_count)/len(decisions)*100:.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
