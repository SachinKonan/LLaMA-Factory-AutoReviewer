#!/usr/bin/env python3
"""
Run meta-review inference to aggregate multiple reviews into a single decision.

This script:
1. Takes predictions with n_generations > 1 (5 reviews per paper)
2. Formats them into a meta-review prompt
3. Runs inference to get the final decision

Usage:
    python run_metareview.py --input predictions.jsonl --output metareview_results.jsonl
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

# Meta-review system prompt
METAREVIEW_SYSTEM_PROMPT = """You are an Area Chair at a prestigious machine learning conference. You are in charge of meta-reviewing a paper that was reviewed by 5 reviewers. Your job is to aggregate the reviews into a single meta-review in the same format. Be critical and cautious in your decision, find consensus, and respect the opinion of all the reviewers."""

# Meta-review user prompt template
METAREVIEW_USER_TEMPLATE = """Review 1/5: {review1}

Review 2/5: {review2}

Review 3/5: {review3}

Review 4/5: {review4}

Review 5/5: {review5}

Respond with a reasoning trace followed by a strictly formatted JSON block.

1. First, provide your reasoning under the section "THOUGHT:".
2. Then, provide the metareview in a valid JSON object inside a markdown code block.

Use the following JSON schema:
```json
{{
  "metareview": "string",
  "soundness": integer, // Score 1-5
  "presentation": integer, // Score 1-5
  "contribution": integer, // Score 1-5
  "overall": integer, // Score 1-10
  "confidence": integer, // Score 1-5
  "decision": "accept" OR "reject"
}}
```"""


def parse_json_decision(text: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Extract decision and full review from JSON output."""
    # Try to find JSON block in markdown code fence
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            review = json.loads(json_match.group(1))
            decision = review.get("decision", "").lower()
            if decision in ["accept", "reject"]:
                return decision.capitalize(), review
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    try:
        json_start = text.rfind('{')
        json_end = text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = text[json_start:json_end]
            review = json.loads(json_str)
            decision = review.get("decision", "").lower()
            if decision in ["accept", "reject"]:
                return decision.capitalize(), review
    except json.JSONDecodeError:
        pass

    return None, None


def create_metareview_dataset(input_path: str, output_path: str):
    """
    Create a dataset for meta-review inference from predictions with multiple generations.

    Args:
        input_path: Path to predictions.jsonl with n_generations > 1
        output_path: Path to save the meta-review dataset
    """
    metareview_samples = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line {line_num}")
                continue

            predictions = data.get("predict", [])
            if isinstance(predictions, str):
                predictions = [predictions]

            # Need at least 5 predictions for meta-review
            if len(predictions) < 5:
                print(f"Warning: Line {line_num} has only {len(predictions)} predictions, skipping")
                continue

            # Format the meta-review prompt
            user_content = METAREVIEW_USER_TEMPLATE.format(
                review1=predictions[0],
                review2=predictions[1],
                review3=predictions[2],
                review4=predictions[3],
                review5=predictions[4]
            )

            # Create ShareGPT format sample
            sample = {
                "conversations": [
                    {"from": "system", "value": METAREVIEW_SYSTEM_PROMPT},
                    {"from": "human", "value": user_content},
                    {"from": "gpt", "value": data.get("label", "")}  # Keep original label for evaluation
                ],
                "_metadata": {
                    "original_predictions": predictions,
                    "original_label": data.get("label", ""),
                    "source_line": line_num
                }
            }

            metareview_samples.append(sample)

    # Save as dataset
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metareview_samples, f, indent=2, ensure_ascii=False)

    print(f"Created meta-review dataset with {len(metareview_samples)} samples")
    print(f"Saved to: {output_path}")

    return metareview_samples


def extract_metareview_results(predictions_path: str, output_path: str):
    """
    Extract final decisions from meta-review predictions.

    Args:
        predictions_path: Path to meta-review predictions.jsonl
        output_path: Path to save extracted results
    """
    results = []

    with open(predictions_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line {line_num}")
                continue

            prediction = data.get("predict", "")
            label = data.get("label", "")

            # Parse meta-review decision
            pred_decision, pred_review = parse_json_decision(prediction)

            # Parse ground truth
            gt_decision = None
            if "accept" in label.lower():
                gt_decision = "Accept"
            elif "reject" in label.lower():
                gt_decision = "Reject"

            result = {
                "prediction": pred_decision,
                "ground_truth": gt_decision,
                "correct": pred_decision == gt_decision if (pred_decision and gt_decision) else None,
                "metareview": pred_review,
                "raw_prediction": prediction
            }

            results.append(result)

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Print summary
    total = len(results)
    correct = sum(1 for r in results if r.get("correct") is True)
    incorrect = sum(1 for r in results if r.get("correct") is False)
    unparseable = sum(1 for r in results if r.get("correct") is None)

    print(f"\n{'='*60}")
    print(f"Meta-Review Results Summary")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct} ({100*correct/total:.2f}%)")
    print(f"Incorrect: {incorrect} ({100*incorrect/total:.2f}%)")
    print(f"Unparseable: {unparseable} ({100*unparseable/total:.2f}%)")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run meta-review for ensemble predictions")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create dataset subcommand
    create_parser = subparsers.add_parser("create", help="Create meta-review dataset from predictions")
    create_parser.add_argument("--input", type=str, required=True,
                               help="Input predictions JSONL with n_generations > 1")
    create_parser.add_argument("--output", type=str, required=True,
                               help="Output path for meta-review dataset (data.json)")

    # Extract results subcommand
    extract_parser = subparsers.add_parser("extract", help="Extract results from meta-review predictions")
    extract_parser.add_argument("--input", type=str, required=True,
                                help="Input meta-review predictions JSONL")
    extract_parser.add_argument("--output", type=str, required=True,
                                help="Output results JSONL")

    args = parser.parse_args()

    if args.command == "create":
        create_metareview_dataset(args.input, args.output)
    elif args.command == "extract":
        extract_metareview_results(args.input, args.output)


if __name__ == "__main__":
    main()
