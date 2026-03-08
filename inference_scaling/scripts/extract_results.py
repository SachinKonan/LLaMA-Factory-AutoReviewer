#!/usr/bin/env python3
"""
Extract decisions from model predictions using different strategies.

Strategies:
1. Single: Use the first prediction (or only prediction for n=1)
2. Majority: Use majority vote from multiple predictions
3. Direct Decision: Parse the decision field from JSON output
4. Calibrated: Use overall score threshold for decision

Usage:
    python extract_results.py --input predictions.jsonl --output results.jsonl --strategy single
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_boxed_decision(text: str) -> Optional[str]:
    """Extract decision from \\boxed{Accept} or \\boxed{Reject} format, or plain text."""
    # Match \boxed{Accept} or \boxed{Reject}
    match = re.search(r'\\boxed\{(Accept|Reject)\}', text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    # Also try plain "Accept" or "Reject" (e.g., from Gemini)
    text_stripped = text.strip().lower()
    if text_stripped == "accept":
        return "Accept"
    elif text_stripped == "reject":
        return "Reject"

    return None


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
        # Find the last occurrence of a JSON-like structure
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


def parse_calibrated_decision(text: str, threshold: int = 6) -> Optional[str]:
    """Extract decision based on overall score threshold."""
    _, review = parse_json_decision(text)
    if review and "overall" in review:
        try:
            overall = int(review["overall"])
            return "Accept" if overall >= threshold else "Reject"
        except (ValueError, TypeError):
            pass
    return None


def extract_decision(text: str, use_calibration: bool = False, threshold: int = 6) -> Optional[str]:
    """Extract decision from a single prediction text."""
    # Try boxed format first (original prompt)
    decision = parse_boxed_decision(text)
    if decision:
        return decision

    # Try JSON format (new prompt)
    if use_calibration:
        decision = parse_calibrated_decision(text, threshold)
    else:
        decision, _ = parse_json_decision(text)

    return decision


def extract_single(predictions: List[str], use_calibration: bool = False, threshold: int = 6) -> Optional[str]:
    """Extract decision using single strategy (first prediction)."""
    if not predictions:
        return None
    if isinstance(predictions, str):
        predictions = [predictions]
    return extract_decision(predictions[0], use_calibration, threshold)


def extract_majority(predictions: List[str], use_calibration: bool = False, threshold: int = 6) -> Optional[str]:
    """Extract decision using majority vote strategy."""
    if isinstance(predictions, str):
        return extract_decision(predictions, use_calibration, threshold)

    decisions = []
    for pred in predictions:
        decision = extract_decision(pred, use_calibration, threshold)
        if decision:
            decisions.append(decision)

    if not decisions:
        return None

    # Majority vote
    counter = Counter(decisions)
    majority_decision, count = counter.most_common(1)[0]

    # In case of tie, default to Reject (more conservative)
    if len(counter) > 1:
        accept_count = counter.get("Accept", 0)
        reject_count = counter.get("Reject", 0)
        if accept_count == reject_count:
            return "Reject"

    return majority_decision


def process_predictions_file(
    input_path: str,
    output_path: str,
    strategy: str = "single",
    use_calibration: bool = False,
    threshold: int = 6
):
    """Process predictions file and extract decisions."""
    results = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line {line_num}")
                continue

            predictions = data.get("predict", [])
            label = data.get("label", "")

            # Extract ground truth from label
            gt_decision = parse_boxed_decision(label)
            if gt_decision is None:
                gt_decision, _ = parse_json_decision(label)
            if gt_decision is None:
                # Try to extract from simple "Accept" or "Reject" in label
                if "accept" in label.lower():
                    gt_decision = "Accept"
                elif "reject" in label.lower():
                    gt_decision = "Reject"

            # Extract prediction based on strategy
            if strategy == "single":
                pred_decision = extract_single(predictions, use_calibration, threshold)
            elif strategy == "majority":
                pred_decision = extract_majority(predictions, use_calibration, threshold)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            result = {
                "prediction": pred_decision,
                "ground_truth": gt_decision,
                "correct": pred_decision == gt_decision if (pred_decision and gt_decision) else None,
                "raw_predictions": predictions if isinstance(predictions, list) else [predictions],
                "n_generations": data.get("n_generations", 1)
            }

            # Add vote breakdown for majority strategy
            if strategy == "majority" and isinstance(predictions, list):
                decisions = [extract_decision(p, use_calibration, threshold) for p in predictions]
                result["vote_breakdown"] = dict(Counter(d for d in decisions if d))

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
    print(f"Results Summary ({strategy} strategy)")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct} ({100*correct/total:.2f}%)")
    print(f"Incorrect: {incorrect} ({100*incorrect/total:.2f}%)")
    print(f"Unparseable: {unparseable} ({100*unparseable/total:.2f}%)")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract decisions from model predictions")
    parser.add_argument("--input", type=str, required=True, help="Input predictions JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output results JSONL file")
    parser.add_argument("--strategy", type=str, default="single",
                        choices=["single", "majority"],
                        help="Decision extraction strategy")
    parser.add_argument("--use_calibration", action="store_true",
                        help="Use overall score calibration instead of direct decision")
    parser.add_argument("--threshold", type=int, default=6,
                        help="Threshold for calibrated decision (score >= threshold = Accept)")

    args = parser.parse_args()

    process_predictions_file(
        input_path=args.input,
        output_path=args.output,
        strategy=args.strategy,
        use_calibration=args.use_calibration,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()
