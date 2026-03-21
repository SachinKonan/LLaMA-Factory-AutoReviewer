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
    """Extract decision from \boxed{Accept} or \boxed{Reject} format, or plain text.
    
    Greedy search to handle conversational models that omit the tag.
    """
    # 1. Standard \boxed logic (highest confidence)
    match = re.search(r'\\boxed\{(Accept|Reject)\}', text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    # 2. Look for common patterns like "Decision: Accept"
    # Prioritizes the LAST occurrence in the text
    patterns = [
        r'Decision:\s*(Accept|Reject)',
        r'Prediction:\s*(Accept|Reject)',
        r'Outcome:\s*(Accept|Reject)',
        r'Final\s*Decision:\s*(Accept|Reject)',
        r'Final\s*Prediction:\s*(Accept|Reject)'
    ]
    
    best_pos = -1
    best_decision = None
    
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if m.start() > best_pos:
                best_pos = m.start()
                best_decision = m.group(1).capitalize()
    
    if best_decision:
        return best_decision

    # 3. Last-ditch: look for the absolute last occurrence of "Accept" or "Reject"
    # This is a very greedy fallback for conversational responses.
    last_accept = text.lower().rfind("accept")
    last_reject = text.lower().rfind("reject")
    
    if last_accept > last_reject:
        return "Accept"
    elif last_reject > last_accept:
        return "Reject"

    return None


def parse_json_decision(text: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Extract decision and full review from JSON output.
    
    Tries multiple strategies to handle common LLM output issues:
    - Literal newlines (strict=False)
    - Unescaped backslashes (LaTeX)
    - Prematurely closed blocks (greedy matching)
    """
    # 1. Try greedy extraction within triple backticks
    fence_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    content = fence_match.group(1) if fence_match else text
    
    # 2. Find first '{' and LAST '}' to capture full block (greedy)
    json_start = content.find('{')
    json_end = content.rfind('}') + 1
    
    if json_start != -1 and json_end > json_start:
        json_str = content[json_start:json_end]
        
        # 3. Pre-process to handle common LLM errors
        # Double up backslashes that aren't valid JSON escapes (e.g., LaTeX \beta)
        json_str = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', json_str)
        
        try:
            # 4. Load with strict=False to allow literal newlines
            review = json.loads(json_str, strict=False)
            decision = review.get("decision", "").lower()
            if decision in ["accept", "reject"]:
                return decision.capitalize(), review
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    return None, None


def parse_calibrated_decision(text: str, threshold: int = 6) -> Optional[str]:
    """Extract decision based on score threshold."""
    _, review = parse_json_decision(text)
    if review:
        # Check for common score field names
        for key in ["score", "overall", "rating"]:
            if key in review:
                try:
                    val = int(review[key])
                    return "Accept" if val >= threshold else "Reject"
                except (ValueError, TypeError):
                    pass
    return None


def extract_decision(text: str, use_calibration: bool = False, threshold: int = 6) -> Optional[str]:
    """Extract decision from a single prediction text."""
    # 1. If calibration is requested, try that first (prefers JSON score)
    if use_calibration:
        decision = parse_calibrated_decision(text, threshold)
        if decision:
            return decision

    # 2. Try boxed format (original prompt / fallback)
    decision = parse_boxed_decision(text)
    if decision:
        return decision

    # 3. Try JSON decision field (fallback)
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
