#!/usr/bin/env python3
"""
Stage 4: Beam Search Optimization.

Answers all deduplicated questions on all real ICLR reviews using local
Qwen3-30B-A3B via vLLM, then runs beam search to find the optimal subset
of questions that maximizes composite score (accuracy + diversity + parsimony).

Usage:
    # Full beam search (needs 2x L40 GPUs)
    python 2_8_26/checklist_optimization/stage4_beam_search.py \
        --input_questions 2_8_26/checklist_optimization/data/deduplicated_questions.jsonl \
        --output 2_8_26/checklist_optimization/data/optimal_checklist.json \
        --beam_width 10 --max_questions 20

    # Debug mode (fewer reviews, smaller beam)
    python 2_8_26/checklist_optimization/stage4_beam_search.py \
        --input_questions 2_8_26/checklist_optimization/data/deduplicated_questions.jsonl \
        --output 2_8_26/checklist_optimization/data/optimal_checklist.json \
        --beam_width 3 --max_questions 5 --n_reviews 50 --debug
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_paper_level_data,
    get_vllm_model,
    load_jsonl,
    load_real_reviews,
    query_vllm_batch,
    save_json,
    save_jsonl,
)


# ============================================================================
# Answer All Questions on All Reviews
# ============================================================================

def answer_questions_batch(
    questions: list[dict],
    reviews: list[dict],
    llm,
    batch_size: int = 64,
) -> list[dict]:
    """Answer all questions on all reviews using local vLLM.

    Batches prompts efficiently for vLLM.

    Args:
        questions: List of question dicts
        reviews: List of review dicts (from real ICLR reviews)
        llm: vLLM model instance
        batch_size: Number of prompts per batch

    Returns:
        List of evaluation entries (one per review) with answers dict
    """
    # Build question text for prompt
    question_list = []
    qids = []
    for i, q in enumerate(questions, 1):
        qid = q.get("question_id", f"q{i:03d}")
        qids.append(qid)
        question_list.append(f'{i}. [{qid}] {q.get("text", "")}')
    questions_text = "\n".join(question_list)

    # Build all prompts
    all_prompts = []
    for review in reviews:
        review_text = review["text"][:3000]
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
            # Parse JSON response
            try:
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    response = response[json_start:json_end].strip()
                elif "```" in response:
                    json_start = response.find("```") + 3
                    json_end = response.find("```", json_start)
                    response = response[json_start:json_end].strip()

                # Try to find JSON object
                brace_start = response.find("{")
                brace_end = response.rfind("}") + 1
                if brace_start >= 0 and brace_end > brace_start:
                    response = response[brace_start:brace_end]

                data = json.loads(response)
                answers_list = data.get("answers", [])

                answers = {}
                for ans in answers_list:
                    qid = ans.get("question_id", "")
                    answer = ans.get("answer", "No")
                    answers[qid] = answer

            except (json.JSONDecodeError, KeyError):
                # Default to "No" for all questions
                answers = {qid: "No" for qid in qids}

            evaluations.append({
                "review_idx": review_idx,
                "submission_id": reviews[review_idx].get("submission_id", ""),
                "answers": answers,
            })

    return evaluations


# ============================================================================
# Paper-Level Aggregation
# ============================================================================

def aggregate_to_paper_level(evaluations: list[dict], reviews: list[dict]) -> list[dict]:
    """Aggregate review-level checklist answers to paper level.

    For each paper, a question is "Yes" if ANY review for that paper answers "Yes".
    This gives a paper-level checklist evaluation.

    Args:
        evaluations: Review-level evaluations
        reviews: Review dicts (to map reviews to papers)

    Returns:
        Paper-level evaluations with decision labels
    """
    # Group by submission_id
    paper_evals = {}
    for eval_entry, review in zip(evaluations, reviews):
        sid = review["submission_id"]
        if sid not in paper_evals:
            paper_evals[sid] = {
                "submission_id": sid,
                "decision": review["decision"],
                "year": review["year"],
                "review_answers": [],
            }
        paper_evals[sid]["review_answers"].append(eval_entry["answers"])

    # Aggregate: for each question, take majority vote across reviews
    paper_level = []
    for sid, pe in paper_evals.items():
        aggregated_answers = {}
        all_qids = set()
        for ans in pe["review_answers"]:
            all_qids.update(ans.keys())

        for qid in all_qids:
            yes_count = sum(1 for ans in pe["review_answers"] if ans.get(qid) == "Yes")
            total = len(pe["review_answers"])
            # Majority vote
            aggregated_answers[qid] = "Yes" if yes_count > total / 2 else "No"

        paper_level.append({
            "submission_id": sid,
            "decision": pe["decision"],
            "year": pe["year"],
            "answers": aggregated_answers,
            "n_reviews": len(pe["review_answers"]),
        })

    return paper_level


# ============================================================================
# Beam Search
# ============================================================================

def compute_composite_score(
    question_subset: list[str],
    evaluations: list[dict],
    ground_truth: list[str],
    question_to_cluster: dict,
) -> float:
    """Compute composite score for a question subset.

    Score = 0.6 * accuracy # + 0.3 * diversity # + 0.01 * length_penalty
    """
    if not question_subset:
        return 0.0

    checklist_scores = []
    for eval_entry in evaluations:
        answers = eval_entry.get("answers", {})
        yes_count = sum(1 for qid in question_subset if answers.get(qid, "No") == "Yes")
        checklist_scores.append(yes_count / len(question_subset))

    threshold = np.median(checklist_scores)
    predictions = ["Accept" if score >= threshold else "Reject" for score in checklist_scores]

    correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
    accuracy = correct / len(ground_truth)

    cluster_ids = [question_to_cluster.get(qid, 0) for qid in question_subset]
    diversity = len(set(cluster_ids)) / len(cluster_ids) if cluster_ids else 0.0

    # length_penalty = max(0.0, 1.0 - (len(question_subset) / 30.0))

    return 0.6 * accuracy # + 0.3 * diversity +#0.1 * length_penalty


def beam_search(
    questions: list[dict],
    evaluations: list[dict],
    ground_truth: list[str],
    beam_width: int = 10,
    max_questions: int = 20,
) -> tuple[list[str], list[dict]]:
    """Run beam search to find optimal question subset."""
    print(f"\nRunning beam search (beam_width={beam_width}, max_questions={max_questions})...")

    all_question_ids = [q.get("question_id", f"q{i:03d}") for i, q in enumerate(questions)]
    question_to_cluster = {q.get("question_id", f"q{i:03d}"): q.get("cluster_id", 0)
                           for i, q in enumerate(questions)}

    # Initialize beam with single best question
    print("\nInitializing beam with single questions...")
    initial_candidates = []
    for qid in all_question_ids:
        score = compute_composite_score([qid], evaluations, ground_truth, question_to_cluster)
        initial_candidates.append((score, [qid]))

    beam = sorted(initial_candidates, reverse=True)[:beam_width]
    best_score = beam[0][0]
    beam_trace = [{"step": 0, "best_score": best_score, "best_subset": beam[0][1]}]
    print(f"  Step 0: best_score = {best_score:.4f}, best_qid = {beam[0][1]}")

    for step in range(1, max_questions):
        print(f"\nStep {step}...")
        candidates = []

        for score, subset in beam:
            for qid in all_question_ids:
                if qid not in subset:
                    new_subset = subset + [qid]
                    new_score = compute_composite_score(
                        new_subset, evaluations, ground_truth, question_to_cluster
                    )
                    candidates.append((new_score, new_subset))

        beam = sorted(candidates, reverse=True)[:beam_width]
        current_best_score = beam[0][0]

        beam_trace.append({
            "step": step,
            "best_score": current_best_score,
            "best_subset": beam[0][1],
        })

        print(f"  Best score: {current_best_score:.4f} (subset size: {len(beam[0][1])})")

        if current_best_score <= best_score:
            print(f"  No improvement, stopping early")
            break

        best_score = current_best_score

    final_score, final_subset = beam[0]
    print(f"\nBeam search complete! Final score: {final_score:.4f}, subset size: {len(final_subset)}")

    return final_subset, beam_trace


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Beam search optimization")
    parser.add_argument("--input_questions", type=str, required=True, help="Path to deduplicated_questions.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output path for optimal_checklist.json")
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--max_questions", type=int, default=20)
    parser.add_argument("--n_reviews", type=int, default=None, help="Limit number of reviews (debug)")
    parser.add_argument("--hf_dataset", type=str, default=None)
    parser.add_argument("--test_split", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for vLLM inference")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.n_reviews = args.n_reviews or 100
        print(f"DEBUG MODE: Using {args.n_reviews} reviews")

    # Load questions
    print(f"\nLoading questions from: {args.input_questions}")
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

    if args.n_reviews:
        reviews = reviews[:args.n_reviews]
    print(f"  Using {len(reviews)} reviews")

    # Load vLLM model
    print(f"\nLoading vLLM model...")
    llm = get_vllm_model(tensor_parallel_size=args.tensor_parallel_size)

    # Answer all questions on all reviews
    print(f"\nAnswering {len(questions)} questions on {len(reviews)} reviews...")
    evaluations = answer_questions_batch(questions, reviews, llm, batch_size=args.batch_size)

    # Save review-level evaluations
    eval_path = args.output.replace("optimal_checklist.json", "checklist_evaluations.jsonl")
    print(f"\nSaving review-level evaluations to: {eval_path}")
    save_jsonl(evaluations, eval_path)

    # Aggregate to paper level
    print(f"\nAggregating to paper level...")
    paper_evaluations = aggregate_to_paper_level(evaluations, reviews)
    ground_truth = [pe["decision"] for pe in paper_evaluations]
    print(f"  {len(paper_evaluations)} papers, "
          f"{sum(1 for gt in ground_truth if gt == 'Accept')} Accept / "
          f"{sum(1 for gt in ground_truth if gt == 'Reject')} Reject")

    # Save paper-level evaluations
    paper_eval_path = args.output.replace("optimal_checklist.json", "paper_evaluations.jsonl")
    save_jsonl(paper_evaluations, paper_eval_path)

    # Run beam search on paper-level evaluations
    best_subset, beam_trace = beam_search(
        questions, paper_evaluations, ground_truth,
        beam_width=args.beam_width, max_questions=args.max_questions,
    )

    # Build optimal checklist
    optimal_checklist = {
        "questions": [],
        "threshold": 0.5,
        "beam_search_steps": len(beam_trace),
        "final_score": beam_trace[-1]["best_score"],
        "beam_width": args.beam_width,
        "n_papers": len(paper_evaluations),
        "n_reviews": len(reviews),
    }

    for qid in best_subset:
        q = next((q for q in questions if q.get("question_id") == qid), None)
        if q:
            optimal_checklist["questions"].append({
                "id": qid,
                "text": q.get("text", ""),
                "category": q.get("category", "unknown"),
                "cluster_id": q.get("cluster_id", 0),
            })

    # Save
    print(f"\nSaving optimal checklist to: {args.output}")
    save_json(optimal_checklist, args.output)

    trace_path = args.output.replace("optimal_checklist.json", "").rstrip("/")
    trace_path = trace_path.replace("/data", "/results") + "/beam_search_trace.jsonl"
    save_jsonl(beam_trace, trace_path)

    print(f"\nOptimal checklist ({len(optimal_checklist['questions'])} questions):")
    for q in optimal_checklist["questions"]:
        print(f"  [{q['id']}] {q['text']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
