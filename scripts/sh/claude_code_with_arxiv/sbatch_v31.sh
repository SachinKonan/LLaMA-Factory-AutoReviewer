#!/bin/bash
#SBATCH --job-name=cc-arxiv-v31
#SBATCH --partition=all
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/claude_code_with_arxiv/sbatch_v31_%j.out
#SBATCH --error=logs/claude_code_with_arxiv/sbatch_v31_%j.err

# Claude Code v3.1 with arxiv search via MCP.
#
# Architecture:
#   srun (background): Retrieval server on gpu-test (1 GPU, max 1hr)
#   This node:         claude -p (headless) with stdio MCP server
#
# Usage:
#   sbatch scripts/sh/claude_code_with_arxiv/sbatch_v31.sh <AGENT_DIR>
#
# Examples:
#   sbatch scripts/sh/claude_code_with_arxiv/sbatch_v31.sh coding_agents_2026_codex_arxiv_aligned_with_model_v3.1
#   sbatch scripts/sh/claude_code_with_arxiv/sbatch_v31.sh coding_agents_2026_codex_arxiv_aligned_no_model_v3.1

set -euo pipefail

REPO_DIR=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
SKYRL_DIR=/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train
MCP_SERVER="${REPO_DIR}/scripts/mcp/arxiv_search_server.py"
CLAUDE_BIN="/home/sk7524/.local/bin/claude"

RETRIEVAL_PORT=8000

# Parameterized agent directory
AGENT_DIR_NAME="${1:-coding_agents_2026_codex_arxiv_aligned_with_model_v3.1}"
AGENT_DIR="${REPO_DIR}/${AGENT_DIR_NAME}"
AGENT_BASENAME=$(basename "${AGENT_DIR}")

if [ ! -d "${AGENT_DIR}" ]; then
    echo "ERROR: Agent directory does not exist: ${AGENT_DIR}"
    exit 1
fi

mkdir -p "${REPO_DIR}/logs/claude_code_with_arxiv"

echo "============================================"
echo "Claude Code v3.1 Agent with Arxiv Search"
echo "============================================"
echo "Agent dir: ${AGENT_DIR} (${AGENT_BASENAME})"
echo "SLURM job: ${SLURM_JOB_ID:-interactive}"
echo "Host:      $(hostname)"
echo "============================================"

# ============================================
# Cleanup trap — kill srun retrieval job on exit
# ============================================
SRUN_PID=""
cleanup() {
    echo "Cleaning up ..."
    if [ -n "${SRUN_PID}" ]; then
        kill "${SRUN_PID}" 2>/dev/null || true
        wait "${SRUN_PID}" 2>/dev/null || true
    fi
    echo "Cleanup done."
}
trap cleanup EXIT

# ============================================
# Start retrieval server via srun (background)
# ============================================
echo "Submitting srun for retrieval server (waiting for gpu-test allocation) ..."

srun --partition=gpu-test --gres=gpu:1 --constraint=gpu80 \
     --time=01:00:00 --mem=40G --cpus-per-task=4 \
     --job-name=cc-retr-${AGENT_BASENAME} \
     bash -c "
    set -e
    export no_proxy='127.0.0.1,localhost,\$(hostname)'
    export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
    export TRANSFORMERS_OFFLINE=1
    cd '${SKYRL_DIR}'

    echo \"[Retrieval] Starting on \$(hostname):${RETRIEVAL_PORT} ...\"
    bash examples/search/retriever/retrieval_launch.sh qwen3 \
        /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/qwen3_06_embed/qwen3_Flat.index \
        /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/arxiv_wikiformat.jsonl \
        ${RETRIEVAL_PORT} gpu \
        'Given a query about academic research, retrieve relevant arXiv papers that address the topic' \
        /scratch/gpfs/ZHUANGL/sk7524/SkyRL/skyrl-train/data/searchr1_original/arxiv/arxiv-metadata-oai-snapshot.jsonl

    echo '[Retrieval] Server launched, keeping node alive ...'
    while true; do sleep 60; done
" &
SRUN_PID=$!

# ============================================
# Discover retrieval node hostname via squeue
# ============================================
sleep 5
echo "Waiting for srun allocation ..."
RETRIEVAL_HOST=""
for i in $(seq 1 2880); do
    if ! kill -0 $SRUN_PID 2>/dev/null; then
        echo "ERROR: srun exited before allocation."
        exit 1
    fi
    RETRIEVAL_HOST=$(squeue --me --name="cc-retr-${AGENT_BASENAME}" -o "%N" --noheader 2>/dev/null | head -1)
    if [ -n "$RETRIEVAL_HOST" ] && [ "$RETRIEVAL_HOST" != "(null)" ]; then
        break
    fi
    sleep 5
done

if [ -z "$RETRIEVAL_HOST" ]; then
    echo "ERROR: Could not determine retrieval node after 4 hours."
    exit 1
fi

RETRIEVAL_URL="http://${RETRIEVAL_HOST}:${RETRIEVAL_PORT}/retrieve"
export RETRIEVAL_URL
export no_proxy="127.0.0.1,localhost,${RETRIEVAL_HOST}"

echo "Retrieval node: ${RETRIEVAL_HOST}"
echo "RETRIEVAL_URL:  ${RETRIEVAL_URL}"

# ============================================
# Wait for retrieval server to be ready
# ============================================
echo "Waiting for retrieval server ..."
for i in $(seq 1 60); do
    if curl -s -X POST "${RETRIEVAL_URL}" \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "topk": 1}' > /dev/null 2>&1; then
        echo "Retrieval server ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Retrieval server did not start within 5 minutes."
        exit 1
    fi
    sleep 5
done

# ============================================
# Write MCP config for Claude Code (.mcp.json in agent dir)
# ============================================
cat > "${AGENT_DIR}/.mcp.json" <<EOF
{
  "mcpServers": {
    "arxiv_search": {
      "type": "stdio",
      "command": "${REPO_DIR}/.venv/bin/python",
      "args": ["${MCP_SERVER}"],
      "env": {
        "RETRIEVAL_URL": "${RETRIEVAL_URL}"
      }
    }
  }
}
EOF

echo "Wrote MCP config to ${AGENT_DIR}/.mcp.json"

# ============================================
# Create paper_images symlink for vision access
# ============================================
if [ ! -e "${AGENT_DIR}/paper_images" ]; then
    ln -s "${REPO_DIR}/data/images" "${AGENT_DIR}/paper_images"
    echo "Created paper_images symlink -> ${REPO_DIR}/data/images"
fi

# ============================================
# Build sub-agent definition (reviewer)
# ============================================
AGENTS_JSON='{
  "reviewer": {
    "description": "Paper reviewer sub-agent that reads papers, inspects page images, searches arxiv for novelty verification, and produces structured ICLR-format reviews.",
    "prompt": "You are a paper reviewer sub-agent. Read the paper content JSON and examine page images. Use search_arxiv to verify novelty and find missing baselines. Write a structured ICLR review with all 11 fields (decision, rating, confidence, soundness, presentation, contribution, summary, strengths, weaknesses, questions, missing_references). Return only the JSON review object.",
    "model": "sonnet",
    "mcpServers": ["arxiv_search"]
  }
}'

# ============================================
# Build orchestrator prompt
# ============================================
if head -1 "${AGENT_DIR}/TEST_SAMPLE.csv" | grep -q "model_prediction"; then
    # With-model prompt
    read -r -d '' CLAUDE_PROMPT <<'PROMPT_EOF' || true
Read AGENTS.override.md for the output schema, calibration guidance, and review rules. Follow every rule strictly.

Step 1: Run `python main.py train-summary` and study the calibration statistics — especially how model_prediction accuracy varies by model_confidence. Note the accept rate, rating distributions, and when to trust the model vs override it.

Step 2: Read TEST_SAMPLE.csv. It has columns: submission_id, content_json, model_prediction, model_confidence. Create the directory agent_reviews/ if it does not exist.

Step 3: For each paper in TEST_SAMPLE.csv, use the Agent tool to spawn a reviewer sub-agent. Pass it the paper's submission_id, content_json path, model_prediction, and model_confidence. Include your calibration findings so the sub-agent can make calibrated decisions. Tell each sub-agent:
- Read the paper content from the content_json file path
- Examine page images listed in the content JSON under img_pages
- Use search_arxiv to verify novelty claims and find missing references
- Use the model_prediction and model_confidence as a Bayesian prior
- Write a structured ICLR review with all 11 fields
- IMPORTANT: Write the review JSON to agent_reviews/{submission_id}.json using the Write tool. Return ONLY a short confirmation like "Done: {submission_id}" — do NOT return the full review text.

Launch sub-agents in parallel (multiple Agent calls in one message) for efficiency. Each sub-agent reviews one paper independently.

Step 4: After all sub-agents complete, write a Python script that reads all agent_reviews/*.json files, assembles them into PREDICTIONS.json with the schema from AGENTS.override.md, and ensures numeric fields are integers in valid ranges (rating: 0-10, confidence: 1-5, soundness/presentation/contribution: 1-4). Decisions must be exactly "accept" or "reject" (lowercase). Run the script.

Step 5: Run `python main.py validate-predictions --path PREDICTIONS.json` to verify.

CRITICAL: Every paper must be reviewed by a sub-agent that actually reads the paper content. Do NOT write scripts, use sklearn, or template-fill. Do NOT skip any of the 200 test papers. Sub-agents MUST write reviews to agent_reviews/{submission_id}.json and return only a short confirmation to avoid context overflow.
PROMPT_EOF
else
    # No-model prompt
    read -r -d '' CLAUDE_PROMPT <<'PROMPT_EOF' || true
Read AGENTS.override.md for the output schema, calibration guidance, and review rules. Follow every rule strictly.

Step 1: Run `python main.py train-summary` and study the calibration statistics — accept rate, rating distributions, confidence distributions, and how human reviewer scores map to decisions.

Step 2: Read TEST_SAMPLE.csv. It has columns: submission_id, content_json. Create the directory agent_reviews/ if it does not exist.

Step 3: For each paper in TEST_SAMPLE.csv, use the Agent tool to spawn a reviewer sub-agent. Pass it the paper's submission_id and content_json path. Include your calibration findings so the sub-agent can make calibrated decisions. Tell each sub-agent:
- Read the paper content from the content_json file path
- Examine page images listed in the content JSON under img_pages
- Use search_arxiv to verify novelty claims and find missing references
- Write a structured ICLR review with all 11 fields
- IMPORTANT: Write the review JSON to agent_reviews/{submission_id}.json using the Write tool. Return ONLY a short confirmation like "Done: {submission_id}" — do NOT return the full review text.

Launch sub-agents in parallel (multiple Agent calls in one message) for efficiency. Each sub-agent reviews one paper independently.

Step 4: After all sub-agents complete, write a Python script that reads all agent_reviews/*.json files, assembles them into PREDICTIONS.json with the schema from AGENTS.override.md, and ensures numeric fields are integers in valid ranges (rating: 0-10, confidence: 1-5, soundness/presentation/contribution: 1-4). Decisions must be exactly "accept" or "reject" (lowercase). Run the script.

Step 5: Run `python main.py validate-predictions --path PREDICTIONS.json` to verify.

CRITICAL: Every paper must be reviewed by a sub-agent that actually reads the paper content. Do NOT write scripts, use sklearn, or template-fill. Do NOT skip any of the 200 test papers. Sub-agents MUST write reviews to agent_reviews/{submission_id}.json and return only a short confirmation to avoid context overflow.
PROMPT_EOF
fi

# ============================================
# Run Claude Code (on this node — no GPU needed)
# ============================================
echo "============================================"
echo "Starting Claude Code for ${AGENT_BASENAME} ..."
echo "============================================"

cd "${AGENT_DIR}"

"${CLAUDE_BIN}" -p "${CLAUDE_PROMPT}" \
    --model opus \
    --dangerously-skip-permissions \
    --mcp-config "${AGENT_DIR}/.mcp.json" \
    --agents "${AGENTS_JSON}" \
    --max-budget-usd 100

echo "Claude Code complete for ${AGENT_BASENAME}."
