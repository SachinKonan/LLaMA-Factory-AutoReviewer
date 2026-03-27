"""Stdio MCP server that gates model prior access behind a completed independent review."""

import json
import os
import sys
from pathlib import Path

from fastmcp import FastMCP

MODEL_PRIOR_PATH = os.environ.get("MODEL_PRIOR_PATH", "model_priors.json")
CALL_BUDGET = int(os.environ.get("MODEL_PRIOR_BUDGET", "0"))  # 0 = unlimited

# Load model priors at startup
_path = Path(MODEL_PRIOR_PATH)
if not _path.exists():
    print(f"[model_prior] ERROR: MODEL_PRIOR_PATH not found: {_path}", file=sys.stderr)
    sys.exit(1)

with _path.open() as _f:
    MODEL_PRIORS: dict[str, dict] = json.load(_f)

print(f"[model_prior] Loaded {len(MODEL_PRIORS)} model priors from {_path}", file=sys.stderr)
if CALL_BUDGET > 0:
    print(f"[model_prior] Call budget: {CALL_BUDGET}", file=sys.stderr)
else:
    print("[model_prior] Call budget: unlimited", file=sys.stderr)

mcp = FastMCP("model_prior")

MIN_REVIEW_LENGTH = 800

# Track call count for budget enforcement
_calls_used = 0


def _check_budget(n: int = 1) -> str | None:
    """Check if we have budget for n calls. Returns error string if over budget."""
    global _calls_used
    if CALL_BUDGET <= 0:
        return None  # unlimited
    if _calls_used + n > CALL_BUDGET:
        remaining = max(0, CALL_BUDGET - _calls_used)
        return (
            f"Error: model prior call budget exhausted. "
            f"Budget: {CALL_BUDGET}, used: {_calls_used}, remaining: {remaining}, requested: {n}. "
            f"Plan your calls carefully — you cannot get more."
        )
    return None


def _consume_budget(n: int = 1) -> None:
    """Consume n calls from the budget."""
    global _calls_used
    _calls_used += n
    if CALL_BUDGET > 0:
        print(
            f"[model_prior] Budget: {_calls_used}/{CALL_BUDGET} used",
            file=sys.stderr,
        )


@mcp.tool()
async def get_model_prior(submission_id: str, prediction: str, review: str) -> str:
    """Get the SFT vision model's prediction for a paper, gated behind your independent review.

    You must provide your own prediction and a complete review BEFORE seeing the model's prediction.
    This enforces genuine two-stage review: independent assessment first, model signal second.

    Args:
        submission_id: The ICLR submission ID of the paper being reviewed.
        prediction: Your independent prediction, must be exactly "accept" or "reject".
        review: Your complete independent structured review (summary, strengths, weaknesses, etc.).
                Must be at least 800 characters to ensure it is a genuine review, not a stub.

    Returns:
        The SFT vision model's prediction and confidence for this paper.
    """
    # Check budget
    budget_err = _check_budget(1)
    if budget_err:
        return budget_err

    # Validate submission_id
    if submission_id not in MODEL_PRIORS:
        return f"Error: submission_id '{submission_id}' not found in model priors. Check the ID and try again."

    # Validate prediction
    if prediction not in ("accept", "reject"):
        return f"Error: prediction must be exactly 'accept' or 'reject', got '{prediction}'."

    # Validate review length
    if len(review) < MIN_REVIEW_LENGTH:
        return (
            f"Error: review is too short ({len(review)} characters). "
            f"Minimum is {MIN_REVIEW_LENGTH} characters. "
            f"Please provide a complete structured review (summary, strengths, weaknesses) "
            f"before requesting the model prior."
        )

    # All checks passed — consume budget and return model prior
    _consume_budget(1)
    prior = MODEL_PRIORS[submission_id]
    model_prediction = prior["model_prediction"]
    model_confidence = prior["model_confidence"]

    # Audit log to stderr
    print(
        f"[model_prior] submission_id={submission_id} | "
        f"agent_prediction={prediction} | "
        f"review_length={len(review)} | "
        f"model_prediction={model_prediction} | "
        f"model_confidence={model_confidence}",
        file=sys.stderr,
    )

    remaining = f" (budget: {_calls_used}/{CALL_BUDGET} used)" if CALL_BUDGET > 0 else ""
    return (
        f"Model prior for {submission_id}:\n"
        f"  model_prediction: {model_prediction}\n"
        f"  model_confidence: {model_confidence}\n\n"
        f"Use this to recalibrate your independent assessment. "
        f"If you disagree with the model, that is fine — override with confidence and explain why."
        f"{remaining}"
    )


@mcp.tool()
async def get_model_priors_batch(papers: str) -> str:
    """Batch version of get_model_prior. Submit multiple papers at once.

    Each paper in the batch counts as one call against the budget.

    Args:
        papers: JSON string of a list of objects, each with:
            - submission_id: str
            - prediction: "accept" or "reject"
            - review: str (minimum 800 characters)

    Returns:
        JSON with model predictions for all valid submissions.
        Invalid submissions get error messages inline.
    """
    try:
        entries = json.loads(papers)
    except json.JSONDecodeError as e:
        return f"Error: could not parse papers JSON: {e}"

    if not isinstance(entries, list):
        return "Error: papers must be a JSON array of objects."

    # Check budget for the entire batch upfront
    budget_err = _check_budget(len(entries))
    if budget_err:
        return budget_err

    results = {}
    valid_count = 0
    for entry in entries:
        sid = entry.get("submission_id", "")
        prediction = entry.get("prediction", "")
        review = entry.get("review", "")

        # Validate submission_id
        if sid not in MODEL_PRIORS:
            results[sid] = {"error": f"submission_id '{sid}' not found in model priors."}
            continue

        # Validate prediction
        if prediction not in ("accept", "reject"):
            results[sid] = {"error": f"prediction must be 'accept' or 'reject', got '{prediction}'."}
            continue

        # Validate review length
        if len(review) < MIN_REVIEW_LENGTH:
            results[sid] = {
                "error": f"review too short ({len(review)} chars, minimum {MIN_REVIEW_LENGTH})."
            }
            continue

        # All checks passed
        valid_count += 1
        prior = MODEL_PRIORS[sid]
        model_prediction = prior["model_prediction"]
        model_confidence = prior["model_confidence"]

        print(
            f"[model_prior_batch] submission_id={sid} | "
            f"agent_prediction={prediction} | "
            f"review_length={len(review)} | "
            f"model_prediction={model_prediction} | "
            f"model_confidence={model_confidence}",
            file=sys.stderr,
        )

        results[sid] = {
            "model_prediction": model_prediction,
            "model_confidence": model_confidence,
        }

    # Only consume budget for valid entries
    _consume_budget(valid_count)

    remaining = CALL_BUDGET - _calls_used if CALL_BUDGET > 0 else "unlimited"
    results["_budget"] = {"used": _calls_used, "total": CALL_BUDGET, "remaining": remaining}

    return json.dumps(results, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
