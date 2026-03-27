#!/bin/bash
# Run LLM-as-judge on v3.3 Claude Code reviews (2-train, no model) vs human reviews.
#
# Usage:
#   bash scripts/sh/claude_code_with_arxiv/run_judge_v33.sh

set -euo pipefail

REPO_DIR=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
CLAUDE_BIN="/home/sk7524/.local/bin/claude"
SCORING_DIR="${REPO_DIR}/coding_agents_2026_codex_aligned_scoring"

mkdir -p "${SCORING_DIR}/judge_scores_v33"
mkdir -p "${REPO_DIR}/logs/claude_code_with_arxiv"

N_PAPERS=$(python3 -c "import csv; print(sum(1 for _ in csv.DictReader(open('${SCORING_DIR}/judge_input_v33.csv'))))")

echo "============================================"
echo "LLM-as-Judge: v3.3 Claude Code Reviews"
echo "============================================"
echo "Papers: ${N_PAPERS}"
echo "============================================"

cd "${SCORING_DIR}"

JUDGE_PROMPT="You are an orchestrator for LLM-as-judge evaluation. Your task is to compare AI-generated paper reviews against human ICLR reviewer reviews and score alignment.

Read judge_input_v33.csv. It has columns: submission_id, ground_truth, human_reviews_json, agent_review_json.

Create the directory judge_scores_v33/ if it does not exist.

For each row, use the Agent tool to spawn a sub-agent. Pass it the submission_id, human_reviews_json, and agent_review_json. Tell the sub-agent to score the agent review against the human reviews on these 5 dimensions (each 1-5):

1. contribution_identification: Did the agent identify the same primary contribution as human reviewers? (1=completely missed, 5=perfect match)
2. top_weakness_identification: Did the agent identify the most important weakness flagged by humans? (1=missed all key weaknesses, 5=identified the top weakness precisely)
3. novel_concerns: Did the agent raise valid concerns NOT mentioned in human reviews? (1=no novel valid points, 3=some valid novel points, 5=substantial novel insights)
4. rating_alignment: How close is the agent's rating to the human reviewer consensus mean? (1=off by 3+, 2=off by 2, 3=off by 1.5, 4=off by 1, 5=off by <0.5)
5. overall_review_quality: Overall quality of the agent review compared to human reviews. (1=poor, 3=adequate, 5=comparable to human quality)

Each sub-agent MUST:
- Compare the agent review text (summary, strengths, weaknesses) against ALL human reviews
- Compute the human mean rating to score rating_alignment
- Write the scores as JSON to judge_scores_v33/{submission_id}.json in this format:
  {\"contribution_identification\": N, \"top_weakness_identification\": N, \"novel_concerns\": N, \"rating_alignment\": N, \"overall_review_quality\": N}
- Return ONLY \"Done: {submission_id}\" to keep context small

Launch sub-agents in parallel (multiple Agent calls per message) for efficiency.

After all sub-agents complete, write a Python script that reads all judge_scores_v33/*.json files and merges them into judge_scores_v33_merged.json as a dict keyed by submission_id. Run the script, then print summary statistics (mean score per dimension).

CRITICAL: Process ALL ${N_PAPERS} papers. Do not skip any."

echo "Starting Claude Code judge ..."

"${CLAUDE_BIN}" -p "${JUDGE_PROMPT}" \
    --model sonnet \
    --dangerously-skip-permissions \
    --max-budget-usd 20 \
    2>&1 | tee "${REPO_DIR}/logs/claude_code_with_arxiv/judge_v33.log"

echo "Judge complete."
