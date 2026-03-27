#!/bin/bash
# Reconciler for v3.3: combines weak CC reviews (2-example train) with SFT model predictions.
# Calibrates model accuracy using 400 training labels (no human reviews).
#
# Usage:
#   bash scripts/sh/claude_code_with_arxiv/run_reconciler_v33.sh

set -euo pipefail

REPO_DIR=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
CLAUDE_BIN="/home/sk7524/.local/bin/claude"
AGENT_DIR="${REPO_DIR}/coding_agents_2026_claudecode_v3.3"

mkdir -p "${REPO_DIR}/logs/claude_code_with_arxiv"

N_PAPERS=$(python3 -c "import json; print(len(json.load(open('${AGENT_DIR}/reconciler/reconciler_input.json'))))")

echo "============================================"
echo "Reconciler v3.3 (CC 2-train + SFT model)"
echo "============================================"
echo "Agent dir: ${AGENT_DIR}"
echo "Papers:    ${N_PAPERS}"
echo "============================================"

cd "${AGENT_DIR}"

RECONCILER_PROMPT="You are a reconciler. You have two imperfect sources of signal for predicting accept/reject on ICLR 2026 papers:

1. **CC reviews**: Independent AI paper reviews (decision, rating, and full review text). These were written by an agent with minimal calibration data. They are NOT ground truth — they have their own error rate.
2. **SFT model predictions**: A fine-tuned vision model's accept/reject prediction with a confidence score (0-1).

Neither source is reliable on its own. Your job is to figure out how to combine them to do better than either alone.

## Step 1: Analyze BOTH signals

Read reconciler/TRAIN_SAMPLE.csv. It has 400 labeled ICLR 2026 papers with columns: submission_id, label, model_prediction, model_confidence.

Write and run a Python script to thoroughly analyze the model's behavior:
- Accuracy overall and broken down by predicted class
- Precision and recall for accept and reject
- Accuracy at different confidence levels
- Error patterns

Also read reconciler/reconciler_input.json to analyze the CC reviews' behavior:
- What fraction does CC predict accept vs reject?
- How do CC ratings distribute?

Think about the relative strengths of each signal. Neither is authoritative.

## Step 2: Build a reconciliation rubric

Based on your analysis, write a rubric to a file called reconciler/rubric.md. The rubric should:
- Compare the two signals' strengths and weaknesses against each other (not just evaluate the model in isolation)
- Define when each signal should take priority
- Be derived entirely from what you observed in the data

## Step 3: Apply your rubric

Read reconciler/reconciler_input.json. It has ${N_PAPERS} papers, each with:
- cc_review: an independent AI review with all 11 fields
- model_prediction: accept/reject
- model_confidence: 0-1

Apply your rubric to each paper. For each, decide the final prediction.

If you change a decision, make sure the full review (rating, text, scores) is consistent with the new decision.

## Step 4: Output

Write PREDICTIONS_reconciled.json with the standard schema:
{
  \"submission_id\": {
    \"decision\": \"accept\" or \"reject\",
    \"rating\": 0-10,
    \"confidence\": 1-5,
    \"soundness\": 1-4,
    \"presentation\": 1-4,
    \"contribution\": 1-4,
    \"summary\": \"...\",
    \"strengths\": \"...\",
    \"weaknesses\": \"...\",
    \"questions\": \"...\",
    \"missing_references\": \"...\"
  }
}

Also write reconciler/reconciler_log.json with per-paper decisions:
{
  \"submission_id\": {
    \"cc_decision\": \"...\",
    \"cc_rating\": X,
    \"model_prediction\": \"...\",
    \"model_confidence\": 0.XX,
    \"final_decision\": \"...\",
    \"action\": \"kept_cc\" or \"flipped_to_model\",
    \"reason\": \"brief explanation\"
  }
}

## Step 5: Validate and summarize

Run: python main.py validate-predictions --path PREDICTIONS_reconciled.json

Print a summary of what you did."

echo "Starting reconciler ..."

"${CLAUDE_BIN}" -p "${RECONCILER_PROMPT}" \
    --model opus \
    --dangerously-skip-permissions \
    --max-budget-usd 20 \
    2>&1 | tee "${REPO_DIR}/logs/claude_code_with_arxiv/reconciler_v33.log"

echo "Reconciler complete."
