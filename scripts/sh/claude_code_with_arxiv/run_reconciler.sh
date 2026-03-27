#!/bin/bash
# Stage 2: Reconciler — combines independent CC reviews with model predictions.
#
# Input:  reconciler/reconciler_input.json (CC reviews + model preds)
#         reconciler/TRAIN_SAMPLE.csv (400 labeled papers for calibration)
# Output: PREDICTIONS_reconciled.json
#
# Usage:
#   bash scripts/sh/claude_code_with_arxiv/run_reconciler.sh

set -euo pipefail

REPO_DIR=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
CLAUDE_BIN="/home/sk7524/.local/bin/claude"
AGENT_DIR="${REPO_DIR}/coding_agents_2026_codex_arxiv_aligned_no_model_v3.1"

mkdir -p "${REPO_DIR}/logs/claude_code_with_arxiv"

echo "============================================"
echo "Stage 2: Reconciler"
echo "============================================"

cd "${AGENT_DIR}"

read -r -d '' RECONCILER_PROMPT <<'PROMPT_EOF' || true
You are a reconciler. Your job is to combine independent AI paper reviews with a fine-tuned vision model's predictions to produce final, calibrated accept/reject decisions.

## Step 1: Calibrate using training data

Read reconciler/TRAIN_SAMPLE.csv. It has 400 labeled ICLR 2026 papers with columns: submission_id, label, content_json, human_reviews_json, model_prediction, model_confidence.

Analyze the model's accuracy as a function of confidence:
- What is the model's overall accuracy?
- How accurate is it when confidence > 0.9? > 0.8? > 0.7? < 0.6?
- When the model says "accept" vs "reject", how often is it right?
- Are there confidence ranges where the model is nearly always correct vs barely better than random?

Write a Python script to compute these statistics. Run it and study the output carefully.

## Step 2: Read independent reviews and model predictions

Read reconciler/reconciler_input.json. It has 100 papers, each with:
- cc_review: an independent AI review (decision, rating, confidence, soundness, presentation, contribution, summary, strengths, weaknesses, questions, missing_references)
- model_prediction: the fine-tuned model's accept/reject prediction
- model_confidence: the model's confidence (0-1)

## Step 3: Reconcile

For each paper, decide the final prediction by weighing both signals:

- The CC review was written independently WITHOUT seeing the model prediction. Its decision and rating reflect genuine paper analysis.
- The model prediction comes from a fine-tuned vision model trained on historical ICLR data.

Use your calibration analysis to decide when to trust each signal:
- When the model has HIGH confidence AND the calibration shows it's reliable at that confidence level, lean toward the model.
- When the model has LOW confidence or the calibration shows it's unreliable, trust the CC review.
- When they AGREE, keep the CC review as-is.
- When they DISAGREE, think carefully. Be skeptical of the model — only flip if the calibration strongly supports the model AND the CC review's rating is borderline (e.g., rating 5-6).

If you decide to flip the decision:
- Adjust the rating to be consistent (e.g., if flipping reject→accept, bump rating from 4→6)
- Keep the rest of the review text (summary, strengths, weaknesses) unchanged — it was an honest assessment
- Note in missing_references or a comment that the decision was adjusted based on model calibration

## Step 4: Output

Write PREDICTIONS_reconciled.json with the standard schema from AGENTS.override.md:
{
  "submission_id": {
    "decision": "accept" or "reject",
    "rating": 0-10,
    "confidence": 1-5,
    "soundness": 1-4,
    "presentation": 1-4,
    "contribution": 1-4,
    "summary": "...",
    "strengths": "...",
    "weaknesses": "...",
    "questions": "...",
    "missing_references": "..."
  }
}

Also write reconciler/reconciler_log.json with per-paper decisions:
{
  "submission_id": {
    "cc_decision": "...",
    "model_prediction": "...",
    "model_confidence": 0.XX,
    "final_decision": "...",
    "action": "kept_cc" or "flipped_to_model",
    "reason": "brief explanation"
  }
}

## Step 5: Validate

Run: python main.py validate-predictions --path PREDICTIONS_reconciled.json

Then print a summary: how many papers were kept vs flipped, and the confidence distribution of flipped papers.
PROMPT_EOF

echo "Starting reconciler ..."

"${CLAUDE_BIN}" -p "${RECONCILER_PROMPT}" \
    --model opus \
    --dangerously-skip-permissions \
    --max-budget-usd 100 \
    2>&1 | tee "${REPO_DIR}/logs/claude_code_with_arxiv/reconciler.log"

echo "Reconciler complete."
