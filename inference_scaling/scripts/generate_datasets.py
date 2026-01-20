#!/usr/bin/env python3
"""
Generate modified datasets with different prompt variants for inference scaling experiments.

Prompt variants:
1. Original: Keep the original prompt (baseline)
2. New: Replace with the new detailed reviewer prompt (no few-shot)
3. New + Fewshot: New prompt with few-shot examples (1 accept + 1 reject)

Usage:
    python generate_datasets.py --base_data_dir /path/to/base/data --output_dir ./data
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Original prompt from existing datasets
ORIGINAL_SYSTEM_PROMPT = "You are an expert academic reviewer tasked with evaluating research papers."
ORIGINAL_USER_PREFIX = """I am giving you a paper. I want to predict its acceptance outcome at ICLR.
 - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}
 - Note: ICLR generally has a ~30% acceptance rate

"""

# New detailed system prompt
NEW_SYSTEM_PROMPT = """You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue. Be critical and cautious in your decision. If a paper is bad or you are unsure, give it bad scores and reject it."""

# New detailed user prompt (prefix - before paper content)
NEW_USER_PREFIX = """Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions. When writing your review, please keep in mind that after decisions have been made, reviews and meta-reviews of accepted papers and opted-in rejected papers will be made public.

Summarized in one line, a review aims to determine whether a submission will bring sufficient value to the community and contribute new knowledge. The process can be broken down into the following main reviewer tasks:

Read the paper: It's important to carefully read through the entire paper, and to look up any related work and citations that will help you comprehensively evaluate it. Be sure to give yourself sufficient time for this step.
While reading, consider the following:
Objective of the work: What is the goal of the paper? Is it to better address a known application or problem, draw attention to a new application or problem, or to introduce and/or explain a new theoretical finding? A combination of these? Different objectives will require different considerations as to potential value and impact.
Strong points: is the submission clear, technically correct, experimentally rigorous, reproducible, does it present novel findings (e.g. theoretically, algorithmically, etc.)?
Weak points: is it weak in any of the aspects listed in b.?
Be mindful of potential biases and try to be open-minded about the value and interest a paper can hold for the entire ICLR community, even if it may not be very interesting for you.
Answer three key questions for yourself, to make a recommendation to Accept or Reject:
What is the specific question and/or problem tackled by the paper?
Is the approach well motivated, including being well-placed in the literature?
Does the paper support the claims? This includes determining if results, whether theoretical or empirical, are correct and if they are scientifically rigorous.
Write your initial review, organizing it as follows:
Summarize what the paper claims to contribute. Be positive and generous.
List strong and weak points of the paper. Be as comprehensive as possible.
Clearly state your recommendation (accept or reject) with one or two key reasons for this choice.
Provide supporting arguments for your recommendation.
Ask questions you would like answered by the authors to help you clarify your understanding of the paper and provide the additional evidence you need to be confident in your assessment.
Provide additional feedback with the aim to improve the paper. Make it clear that these points are here to help, and not necessarily part of your decision assessment.
General points to consider:
Be polite in your review. Ask yourself whether you'd be happy to receive a review written like the one you wrote.
Be precise and concrete. For example, include references to back up any claims, especially claims about novelty and prior work
Provide constructive feedback.
It's also fine to explicitly state where you are uncertain and what you don't quite understand. The authors may be able to resolve this in their response.
Don't reject a paper just because you don't find it "interesting". This should not be a criterion at all for accepting/rejecting a paper. The research community is so big that somebody will find some value in the paper (maybe even a few years down the road), even if you don't see it right now.
Complete the CoE report: ICLR has adopted the following Code of Ethics (CoE). Please check your assigned papers for conflicts with the code of ethics and mark them in your review form. If you are uncertain, please reach out to your area chair.
Engage in discussion: During the discussion phase, reviewers, authors and area chairs engage in asynchronous discussion, and authors are allowed to revise their submissions to address concerns that arise. It is crucial that you are actively engaged and responsive during this phase, i.e., you should be able to respond to comments/requests within 3 business days.
Provide final recommendation: Update your review, taking into account the new information collected during the discussion phase, and any revisions to the submission. Maintain a spirit of openness to changing your initial recommendation (either to a more positive or more negative) rating.

{fewshot_placeholder}Here is the paper you are asked to review: """

# Suffix after the paper content
NEW_USER_SUFFIX = """
Respond with a reasoning trace followed by a strictly formatted JSON block.

1. First, provide your reasoning under the section "THOUGHT:".
2. Then, provide the review in a valid JSON object inside a markdown code block.

Use the following JSON schema:
```json
{
  "summary": "string",
  "questions": "string",
  "limitations": "string",
  "strengths": "string",
  "weaknesses": "string",
  "ethical_concerns": boolean,
  "soundness": integer, // Score 1-5
  "presentation": integer, // Score 1-5
  "contribution": integer, // Score 1-5
  "overall": integer, // Score 1-10
  "confidence": integer, // Score 1-5
  "decision": "accept" OR "reject"
}
```"""

# Fewshot example placeholder text (to be replaced with actual examples)
FEWSHOT_TEMPLATE = """Here are example reviews for reference:

Example 1 (Accept):
Paper: {accept_paper}
Review: {accept_review}

Example 2 (Reject):
Paper: {reject_paper}
Review: {reject_review}

"""


def load_dataset(data_path: str) -> List[Dict]:
    """Load a dataset from a data.json file."""
    json_path = os.path.join(data_path, "data.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data: List[Dict], output_path: str):
    """Save a dataset to a data.json file."""
    os.makedirs(output_path, exist_ok=True)
    json_path = os.path.join(output_path, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} samples to {json_path}")


def extract_paper_content(user_message: str) -> str:
    """Extract the paper content from the original user message.

    The original user message format is:
    {ORIGINAL_USER_PREFIX}\n\n{paper_content}

    Where ORIGINAL_USER_PREFIX ends with a double newline.
    """
    # The prefix ends with "\n\n", so find that boundary
    # Look for the end of the known prefix pattern
    prefix_end_marker = " - Note: ICLR generally has a ~30% acceptance rate\n\n"

    if prefix_end_marker in user_message:
        # Split at the end of the prefix
        idx = user_message.find(prefix_end_marker)
        return user_message[idx + len(prefix_end_marker):].strip()

    # Fallback: try to find double newline after the first few lines
    lines = user_message.split('\n')
    for i, line in enumerate(lines):
        if i > 0 and line.strip() == '' and i < len(lines) - 1:
            # Check if this is a double newline (empty line)
            if lines[i-1].strip().endswith('rate'):
                return '\n'.join(lines[i+1:]).strip()

    # Last fallback: return as-is
    return user_message


def transform_to_original(sample: Dict) -> Dict:
    """Keep the original prompt format (no changes)."""
    return sample.copy()


def transform_to_new(sample: Dict, include_fewshot: bool = False,
                     fewshot_examples: Optional[str] = None) -> Dict:
    """Transform a sample to use the new prompt format."""
    new_sample = sample.copy()
    conversations = []

    for conv in sample["conversations"]:
        new_conv = conv.copy()
        if conv["from"] == "system":
            new_conv["value"] = NEW_SYSTEM_PROMPT
        elif conv["from"] == "human":
            # Extract paper content
            paper_content = extract_paper_content(conv["value"])

            # Build new user message
            if include_fewshot and fewshot_examples:
                fewshot_section = fewshot_examples
            else:
                fewshot_section = ""

            user_prefix = NEW_USER_PREFIX.format(fewshot_placeholder=fewshot_section)
            new_conv["value"] = user_prefix + paper_content + NEW_USER_SUFFIX
        # Keep gpt response unchanged
        conversations.append(new_conv)

    new_sample["conversations"] = conversations
    return new_sample


def get_fewshot_examples(validation_data: List[Dict], seed: int = 42) -> Tuple[Dict, Dict]:
    """
    Get one accept and one reject example from validation data.
    Uses fixed seed for reproducibility.
    """
    random.seed(seed)

    accepts = [s for s in validation_data if s.get("_metadata", {}).get("answer") == "Accept"]
    rejects = [s for s in validation_data if s.get("_metadata", {}).get("answer") == "Reject"]

    if not accepts or not rejects:
        raise ValueError("Validation data must contain both accept and reject examples")

    accept_example = random.choice(accepts)
    reject_example = random.choice(rejects)

    return accept_example, reject_example


def format_fewshot_string(accept_example: Dict, reject_example: Dict) -> str:
    """Format the few-shot examples into a string."""
    # Extract paper content from examples
    accept_paper = ""
    reject_paper = ""

    for conv in accept_example["conversations"]:
        if conv["from"] == "human":
            accept_paper = extract_paper_content(conv["value"])[:2000] + "..."  # Truncate for brevity
            break

    for conv in reject_example["conversations"]:
        if conv["from"] == "human":
            reject_paper = extract_paper_content(conv["value"])[:2000] + "..."
            break

    # Create example review outputs (simplified for demonstration)
    accept_review = '''{
  "summary": "This paper presents a novel approach...",
  "questions": "...",
  "limitations": "...",
  "strengths": "The paper is well-written and the experiments are thorough...",
  "weaknesses": "Some minor issues with clarity...",
  "ethical_concerns": false,
  "soundness": 4,
  "presentation": 4,
  "contribution": 4,
  "overall": 7,
  "confidence": 4,
  "decision": "accept"
}'''

    reject_review = '''{
  "summary": "This paper attempts to...",
  "questions": "...",
  "limitations": "...",
  "strengths": "The motivation is clear...",
  "weaknesses": "The experimental evaluation is limited and the claims are not well supported...",
  "ethical_concerns": false,
  "soundness": 2,
  "presentation": 3,
  "contribution": 2,
  "overall": 4,
  "confidence": 3,
  "decision": "reject"
}'''

    return FEWSHOT_TEMPLATE.format(
        accept_paper=accept_paper,
        accept_review=accept_review,
        reject_paper=reject_paper,
        reject_review=reject_review
    )


def generate_datasets(
    base_data_dir: str,
    output_dir: str,
    dataset_names: List[str],
    splits: List[str] = ["test"],
    fewshot_dir: Optional[str] = None,
    seed: int = 42,
    limit: Optional[int] = None
):
    """
    Generate modified datasets for all prompt variants.

    Args:
        base_data_dir: Directory containing base datasets
        output_dir: Directory to save modified datasets
        dataset_names: List of base dataset names (without split suffix)
        splits: List of splits to process (e.g., ["test", "validation"])
        fewshot_dir: Directory containing validation data for few-shot examples (optional)
        seed: Random seed for few-shot example selection
        limit: If set, only process the first N samples (for testing)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load few-shot examples if provided
    fewshot_string = None
    if fewshot_dir:
        validation_data = load_dataset(fewshot_dir)
        accept_ex, reject_ex = get_fewshot_examples(validation_data, seed)
        fewshot_string = format_fewshot_string(accept_ex, reject_ex)
        print(f"Loaded few-shot examples from {fewshot_dir}")

    for dataset_name in dataset_names:
        for split in splits:
            full_name = f"{dataset_name}_{split}"
            input_path = os.path.join(base_data_dir, full_name)

            if not os.path.exists(input_path):
                print(f"Warning: Dataset not found: {input_path}")
                continue

            print(f"\nProcessing {full_name}...")
            data = load_dataset(input_path)

            # Apply limit if specified (for testing)
            if limit is not None:
                data = data[:limit]
                print(f"  Limited to {len(data)} samples")

            # Variant 1: Original prompt (copy as-is)
            original_output = os.path.join(output_dir, f"{full_name}_original")
            save_dataset(data, original_output)

            # Variant 2: New prompt (no few-shot)
            new_data = [transform_to_new(s, include_fewshot=False) for s in data]
            new_output = os.path.join(output_dir, f"{full_name}_new")
            save_dataset(new_data, new_output)

            # Variant 3: New prompt + few-shot (only if fewshot_string is available)
            if fewshot_string:
                fewshot_data = [transform_to_new(s, include_fewshot=True,
                                                  fewshot_examples=fewshot_string) for s in data]
                fewshot_output = os.path.join(output_dir, f"{full_name}_new_fewshot")
                save_dataset(fewshot_data, fewshot_output)
            else:
                # Create placeholder with None fewshot
                fewshot_data = [transform_to_new(s, include_fewshot=True,
                                                  fewshot_examples="[Few-shot examples will be added here]\n\n") for s in data]
                fewshot_output = os.path.join(output_dir, f"{full_name}_new_fewshot")
                save_dataset(fewshot_data, fewshot_output)
                print(f"  Note: Few-shot examples not provided, using placeholder")


def main():
    parser = argparse.ArgumentParser(description="Generate modified datasets for inference scaling")
    parser.add_argument("--base_data_dir", type=str,
                        default="/n/fs/vision-mix/sk7524/LLaMA-Factory/data",
                        help="Base directory containing original datasets")
    parser.add_argument("--output_dir", type=str,
                        default="./inference_strategies/data",
                        help="Output directory for modified datasets")
    parser.add_argument("--splits", type=str, nargs="+", default=["test"],
                        help="Dataset splits to process")
    parser.add_argument("--fewshot_dir", type=str, default=None,
                        help="Directory containing validation data for few-shot examples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for few-shot example selection")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N samples (for testing)")

    args = parser.parse_args()

    # Define the three base dataset names (without split suffix)
    dataset_names = [
        "iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6",  # text only
        "iclr_2020_2025_85_5_10_split6_balanced_clean_images_binary_noreviews_v6",  # text + images
        "iclr_2020_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6",  # vision
    ]

    generate_datasets(
        base_data_dir=args.base_data_dir,
        output_dir=args.output_dir,
        dataset_names=dataset_names,
        splits=args.splits,
        fewshot_dir=args.fewshot_dir,
        seed=args.seed,
        limit=args.limit
    )

    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
