#!/usr/bin/env python3
"""
Stage 1: Generate Candidate Checklist Questions.

Seeds initial questions from official ICLR review criteria, then uses a local
Qwen3-30B-A3B model via vLLM to expand and refine them using real ICLR reviews
from the HuggingFace Arrow dataset.

Usage:
    # Full generation (uses 2x L40 GPUs)
    python 2_8_26/checklist_optimization/stage1_generate_candidates.py \
        --n_questions 100 --sample_size 50

    # Debug mode
    python 2_8_26/checklist_optimization/stage1_generate_candidates.py \
        --n_questions 20 --sample_size 10 --debug
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_vllm_model,
    load_real_reviews,
    query_vllm,
    save_jsonl,
)

# ============================================================================
# ICLR Review Criteria Seed Questions
# ============================================================================

# These questions ask about REVIEW COVERAGE (what topics are discussed),
# not review opinions (what the review says). This identifies comprehensive
# reviews that correlate with better decision-making.
ICLR_SEED_QUESTIONS = [
    # --- Novelty & Originality ---
    {"text": "Does the review discuss whether the paper presents a novel contribution?", "category": "novelty"},
    {"text": "Does the review discuss whether the approach is incremental over prior work?", "category": "novelty"},
    {"text": "Does the review evaluate the originality or creativity of the paper's idea?", "category": "novelty"},
    {"text": "Does the review address the novelty of the contribution?", "category": "novelty"},
    {"text": "Does the review compare the contribution to existing methods?", "category": "novelty"},

    # --- Technical Soundness ---
    {"text": "Does the review evaluate the mathematical or logical correctness?", "category": "soundness"},
    {"text": "Does the review discuss the correctness of proofs or derivations?", "category": "soundness"},
    {"text": "Does the review assess the soundness of the methodology?", "category": "soundness"},
    {"text": "Does the review discuss assumptions or justifications for claims?", "category": "soundness"},
    {"text": "Does the review evaluate the experimental setup or design?", "category": "soundness"},
    {"text": "Does the review discuss the rigor of the theoretical analysis?", "category": "soundness"},

    # --- Clarity & Presentation ---
    {"text": "Does the review evaluate the quality of the writing?", "category": "clarity"},
    {"text": "Does the review discuss the clarity or readability of the paper?", "category": "clarity"},
    {"text": "Does the review comment on the writing, organization, or presentation?", "category": "clarity"},
    {"text": "Does the review discuss the figures, tables, or visualizations?", "category": "clarity"},

    # --- Experimental Validation ---
    {"text": "Does the review evaluate the comprehensiveness of the experiments?", "category": "validation"},
    {"text": "Does the review discuss baselines or comparisons?", "category": "validation"},
    {"text": "Does the review discuss the statistical significance or robustness of results?", "category": "validation"},
    {"text": "Does the review suggest or discuss ablation studies?", "category": "validation"},
    {"text": "Does the review comment on the magnitude of improvements?", "category": "validation"},
    {"text": "Does the review discuss the evaluation metrics used?", "category": "validation"},
    {"text": "Does the review discuss the datasets or benchmarks used?", "category": "validation"},

    # --- Impact & Significance ---
    {"text": "Does the review evaluate the significance of the contribution?", "category": "impact"},
    {"text": "Does the review discuss the practical applicability of the work?", "category": "impact"},
    {"text": "Does the review discuss the importance of the problem being addressed?", "category": "impact"},
    {"text": "Does the review comment on the paper's potential impact or broad interest?", "category": "impact"},

    # --- Reproducibility ---
    {"text": "Does the review discuss reproducibility?", "category": "reproducibility"},
    {"text": "Does the review mention code or data availability?", "category": "reproducibility"},
    {"text": "Does the review discuss implementation details?", "category": "reproducibility"},

    # --- Related Work ---
    {"text": "Does the review evaluate the related work discussion?", "category": "related_work"},
    {"text": "Does the review discuss citations or relevant prior work?", "category": "related_work"},

    # --- Overall Assessment ---
    {"text": "Does the review provide an explicit recommendation (accept/reject)?", "category": "sentiment"},
    {"text": "Does the review discuss major concerns or blocking issues?", "category": "sentiment"},
    {"text": "Does the review provide a structured list of strengths?", "category": "sentiment"},
    {"text": "Does the review provide a structured list of weaknesses?", "category": "sentiment"},
    {"text": "Does the review provide questions for the authors?", "category": "sentiment"},

    # --- Scope & Limitations ---
    {"text": "Does the review discuss the scope or generality of the evaluation?", "category": "scope"},
    {"text": "Does the review discuss limitations of the proposed method?", "category": "scope"},
    {"text": "Does the review identify limitations acknowledged by the authors?", "category": "scope"},
]


# ============================================================================
# Review Sampling
# ============================================================================

def sample_reviews(reviews: list[dict], n_samples: int = 50, seed: int = 42) -> dict:
    """Sample balanced reviews (50% Accept, 50% Reject)."""
    random.seed(seed)

    accept_reviews = [r for r in reviews if r.get("decision") == "Accept"]
    reject_reviews = [r for r in reviews if r.get("decision") == "Reject"]

    n_per_class = n_samples // 2
    sampled_accept = random.sample(accept_reviews, min(n_per_class, len(accept_reviews)))
    sampled_reject = random.sample(reject_reviews, min(n_per_class, len(reject_reviews)))

    return {"accept": sampled_accept, "reject": sampled_reject}


def format_sample_reviews(samples: dict, max_per_class: int = 5) -> str:
    """Format sample reviews for prompt context."""
    output = []

    output.append("REVIEWS FOR ACCEPTED PAPERS:\n")
    for i, r in enumerate(samples["accept"][:max_per_class]):
        output.append(f"Review {i + 1} (rating={r.get('rating', '?')}):\n{r['text'][:600]}...\n")

    output.append("\nREVIEWS FOR REJECTED PAPERS:\n")
    for i, r in enumerate(samples["reject"][:max_per_class]):
        output.append(f"Review {i + 1} (rating={r.get('rating', '?')}):\n{r['text'][:600]}...\n")

    return "\n".join(output)


# ============================================================================
# Question Generation via LLM Expansion
# ============================================================================

def expand_questions_with_llm(
    seed_questions: list[dict],
    samples: dict,
    n_target: int = 100,
    llm=None,
) -> list[dict]:
    """Use LLM to expand seed questions into a larger diverse pool.

    Takes the ICLR-criteria seed questions and real review samples, asks the
    model to generate additional binary questions that capture patterns in
    the reviews.

    Args:
        seed_questions: Initial ICLR-criteria questions
        samples: Dict with 'accept' and 'reject' review samples
        n_target: Target number of total questions (seed + generated)
        llm: vLLM model instance

    Returns:
        Combined list of seed + generated questions
    """
    n_to_generate = max(0, n_target - len(seed_questions))
    if n_to_generate == 0:
        return seed_questions

    sample_text = format_sample_reviews(samples, max_per_class=3)

    system_prompt = """You are an expert in academic peer review for machine learning conferences (ICLR, NeurIPS, ICML). Your task is to generate binary (yes/no) checklist questions about REVIEW COVERAGE - what topics the review discusses, not what opinions it expresses. Comprehensive reviews that cover key evaluation criteria correlate with better accept/reject decisions."""

    user_prompt = f"""Here are some existing checklist questions we already have:

{chr(10).join(f'- {q["text"]} [{q["category"]}]' for q in seed_questions[:15])}

And here are sample real ICLR reviews for context:

{sample_text}

Generate {n_to_generate} NEW binary (yes/no) questions that ask about REVIEW COVERAGE (what is discussed):
1. Ask "Does the review discuss/evaluate/address/mention [topic]?" NOT "Does the review say [opinion]?"
2. Focus on whether the review covers important evaluation criteria
3. Answerable purely from the review text (not the paper)
4. Cover diverse aspects: novelty, soundness, clarity, experiments, impact, reproducibility, limitations
5. Not redundant with the existing questions above

GOOD examples:
- "Does the review discuss the novelty of the contribution?"
- "Does the review evaluate the experimental methodology?"
- "Does the review mention limitations?"

BAD examples:
- "Does the review say the paper is novel?" (opinion, not coverage)
- "Does the review recommend acceptance?" (outcome, not coverage)

Output as a JSON list:
[
  {{"text": "Does the review discuss/evaluate/mention...", "category": "novelty"}},
  ...
]

Categories: novelty, soundness, clarity, validation, impact, reproducibility, related_work, sentiment, scope"""

    print(f"  Generating {n_to_generate} additional questions via LLM...")
    response = query_vllm(
        user_prompt,
        llm,
        temperature=0.8,
        max_tokens=4000,
        system_prompt=system_prompt,
    )

    # Parse JSON response
    generated = []
    try:
        # Extract JSON from response (may be wrapped in markdown)
        if "```json" in response:
            json_start = response.find("```json") + 7
            json_end = response.find("```", json_start)
            response = response[json_start:json_end].strip()
        elif "```" in response:
            json_start = response.find("```") + 3
            json_end = response.find("```", json_start)
            response = response[json_start:json_end].strip()

        # Try to find a JSON array
        bracket_start = response.find("[")
        bracket_end = response.rfind("]") + 1
        if bracket_start >= 0 and bracket_end > bracket_start:
            response = response[bracket_start:bracket_end]

        data = json.loads(response)
        if isinstance(data, list):
            generated = data
        elif isinstance(data, dict) and "questions" in data:
            generated = data["questions"]

        print(f"  Successfully parsed {len(generated)} generated questions")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Warning: Failed to parse LLM response: {e}")
        print(f"  Response preview: {response[:300]}...")

    return seed_questions + generated


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Generate candidate checklist questions")
    parser.add_argument("--output", type=str,
                        default="2_8_26/checklist_optimization/data/candidate_questions.jsonl",
                        help="Output path for candidate_questions.jsonl")
    parser.add_argument("--n_questions", type=int, default=100,
                        help="Target number of questions (seed + generated, default: 100)")
    parser.add_argument("--sample_size", type=int, default=50,
                        help="Number of reviews to sample for context (default: 50)")
    parser.add_argument("--hf_dataset", type=str, default=None, help="Override HF dataset path")
    parser.add_argument("--test_split", type=str, default=None, help="Override test split path")
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="GPUs for TP (default: 2)")
    parser.add_argument("--debug", action="store_true", help="Debug mode (fewer questions, smaller samples)")
    args = parser.parse_args()

    if args.debug:
        args.n_questions = min(args.n_questions, 50)
        args.sample_size = min(args.sample_size, 10)
        print("DEBUG MODE: Using reduced parameters")
        print(f"  n_questions: {args.n_questions}")
        print(f"  sample_size: {args.sample_size}")

    # Load real ICLR reviews from HF dataset
    print("\nLoading real ICLR reviews...")
    kwargs = {}
    if args.hf_dataset:
        kwargs["hf_dataset_path"] = args.hf_dataset
    if args.test_split:
        kwargs["test_split_path"] = args.test_split
    reviews = load_real_reviews(**kwargs)

    # Sample balanced reviews for LLM context
    print(f"\nSampling {args.sample_size} reviews (balanced by outcome)...")
    samples = sample_reviews(reviews, n_samples=args.sample_size)
    print(f"  Sampled {len(samples['accept'])} accept, {len(samples['reject'])} reject")

    # Start with ICLR seed questions
    print(f"\nUsing {len(ICLR_SEED_QUESTIONS)} ICLR-criteria seed questions")

    # Expand with LLM if needed
    if args.n_questions > len(ICLR_SEED_QUESTIONS):
        print(f"\nLoading vLLM model for question expansion...")
        llm = get_vllm_model(tensor_parallel_size=args.tensor_parallel_size)

        all_questions = expand_questions_with_llm(
            ICLR_SEED_QUESTIONS, samples, n_target=args.n_questions, llm=llm
        )
    else:
        all_questions = ICLR_SEED_QUESTIONS[:args.n_questions]

    # Format output with IDs
    output = []
    for i, q in enumerate(all_questions):
        output.append({
            "question_id": f"q{i + 1:03d}",
            "text": q.get("text", ""),
            "category": q.get("category", "unknown"),
            "source": "iclr_criteria" if i < len(ICLR_SEED_QUESTIONS) else "llm_generated",
        })

    # Save
    print(f"\nSaving {len(output)} questions to: {args.output}")
    save_jsonl(output, args.output)

    # Print summary
    by_category = {}
    by_source = {}
    for q in output:
        cat = q["category"]
        src = q["source"]
        by_category[cat] = by_category.get(cat, 0) + 1
        by_source[src] = by_source.get(src, 0) + 1

    print("\nQuestions by category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")

    print(f"\nQuestions by source:")
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")

    print("\nSample questions:")
    for q in output[:5]:
        print(f"  [{q['question_id']}] {q['text']} ({q['category']}, {q['source']})")


if __name__ == "__main__":
    main()
