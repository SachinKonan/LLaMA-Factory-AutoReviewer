"""Utility functions for textgrad prompt optimization."""

import json
import random
import re
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    if text is None:
        return None
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip().lower()
    return None


def normalize_label(label: str | None) -> str | None:
    """Normalize label to 'accepted' or 'rejected'."""
    if label is None:
        return None
    label_lower = label.lower()
    if label_lower in ["accept", "accepted"]:
        return "accepted"
    if label_lower in ["reject", "rejected"]:
        return "rejected"
    return None


def load_dataset(data_dir: str, dataset_name: str) -> list[dict]:
    """
    Load ShareGPT-format dataset.

    Returns list of dicts with keys:
    - system_prompt: str
    - human_message: str
    - expected_label: str ('accepted' or 'rejected')
    """
    # Find the dataset file
    dataset_info_path = Path(data_dir) / "dataset_info.json"
    with open(dataset_info_path) as f:
        info = json.load(f)

    if dataset_name not in info:
        raise ValueError(f"Dataset {dataset_name} not found in dataset_info.json")

    file_name = info[dataset_name]["file_name"]
    data_path = Path(data_dir) / file_name

    with open(data_path) as f:
        raw_data = json.load(f)

    # Parse ShareGPT format
    samples = []
    for item in raw_data:
        conversations = item.get("conversations", [])

        system_prompt = None
        human_message = None
        assistant_response = None

        for msg in conversations:
            role = msg.get("from", "")
            value = msg.get("value", "")

            if role == "system":
                system_prompt = value
            elif role == "human":
                human_message = value
            elif role in ["gpt", "assistant"]:
                assistant_response = value

        if human_message is None:
            continue

        # Extract expected label from assistant response
        expected_label = None
        if assistant_response:
            boxed = extract_boxed_answer(assistant_response)
            expected_label = normalize_label(boxed)

        if expected_label is None:
            continue

        samples.append({
            "system_prompt": system_prompt or "",
            "human_message": human_message,
            "expected_label": expected_label,
        })

    return samples


def sample_batch(data: list[dict], batch_size: int, seed: int | None = None) -> list[dict]:
    """Sample a random batch from the dataset."""
    if seed is not None:
        random.seed(seed)
    return random.sample(data, min(batch_size, len(data)))


def compute_metrics(predictions: list[str | None], labels: list[str]) -> dict:
    """Compute accuracy and other metrics for binary classification."""
    # Filter out None predictions
    valid_pairs = [(p, l) for p, l in zip(predictions, labels) if p is not None]

    if not valid_pairs:
        return {
            "accuracy": 0.0,
            "num_valid": 0,
            "num_total": len(predictions),
            "accept_f1": 0.0,
            "reject_f1": 0.0,
            "num_accept_correct": 0,
            "num_reject_correct": 0,
            "num_accept_total": sum(1 for l in labels if l == "accepted"),
            "num_reject_total": sum(1 for l in labels if l == "rejected"),
        }

    y_pred, y_true = zip(*valid_pairs)

    accuracy = accuracy_score(y_true, y_pred)
    accept_f1 = f1_score(y_true, y_pred, pos_label="accepted", zero_division=0)
    reject_f1 = f1_score(y_true, y_pred, pos_label="rejected", zero_division=0)

    num_accept_correct = sum(1 for p, l in valid_pairs if p == l == "accepted")
    num_reject_correct = sum(1 for p, l in valid_pairs if p == l == "rejected")

    return {
        "accuracy": accuracy,
        "num_valid": len(valid_pairs),
        "num_total": len(predictions),
        "accept_f1": accept_f1,
        "reject_f1": reject_f1,
        "num_accept_correct": num_accept_correct,
        "num_reject_correct": num_reject_correct,
        "num_accept_total": sum(1 for l in labels if l == "accepted"),
        "num_reject_total": sum(1 for l in labels if l == "rejected"),
    }


def describe_errors(predictions: list[str | None], labels: list[str]) -> str:
    """Generate a text description of prediction errors."""
    false_accepts = sum(1 for p, l in zip(predictions, labels) if p == "accepted" and l == "rejected")
    false_rejects = sum(1 for p, l in zip(predictions, labels) if p == "rejected" and l == "accepted")
    parse_failures = sum(1 for p in predictions if p is None)

    parts = []
    if false_accepts > 0:
        parts.append(f"{false_accepts} false accepts (predicted accept but was rejected)")
    if false_rejects > 0:
        parts.append(f"{false_rejects} false rejects (predicted reject but was accepted)")
    if parse_failures > 0:
        parts.append(f"{parse_failures} parse failures (could not extract \\boxed answer)")

    return "; ".join(parts) if parts else "No errors"


def suggest_improvements(predictions: list[str | None], labels: list[str]) -> str:
    """Suggest improvements based on error patterns."""
    false_accepts = sum(1 for p, l in zip(predictions, labels) if p == "accepted" and l == "rejected")
    false_rejects = sum(1 for p, l in zip(predictions, labels) if p == "rejected" and l == "accepted")
    parse_failures = sum(1 for p in predictions if p is None)

    suggestions = []

    if false_accepts > false_rejects:
        suggestions.append("Be more critical and rigorous in evaluation - currently too lenient")
    elif false_rejects > false_accepts:
        suggestions.append("Be more open to accepting novel contributions - currently too harsh")

    if parse_failures > 0:
        suggestions.append("Ensure responses end with \\boxed{Accept} or \\boxed{Reject}")

    if not suggestions:
        suggestions.append("Continue with current approach, balancing rigor and openness")

    return "; ".join(suggestions)


def format_loss_feedback(predictions: list[str | None], labels: list[str]) -> str:
    """Format loss feedback for textgrad optimization."""
    metrics = compute_metrics(predictions, labels)
    errors = describe_errors(predictions, labels)
    suggestions = suggest_improvements(predictions, labels)

    return f"""Evaluation Results for Paper Acceptance Prediction:
- Accuracy: {metrics['accuracy']:.1%} ({metrics['num_valid']}/{metrics['num_total']} predictions parsed)
- Accept F1: {metrics['accept_f1']:.3f}
- Reject F1: {metrics['reject_f1']:.3f}

Error Analysis:
{errors}

Suggestions for System Prompt Improvement:
{suggestions}

The goal is to improve accuracy by refining how the model approaches paper evaluation.
The system prompt should guide the model to be a better academic reviewer."""
