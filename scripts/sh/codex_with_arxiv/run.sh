#!/bin/bash
# Codex agent with arxiv search via MCP.
#
# Architecture:
#   srun (background): Retrieval server on gpu-test (1 GPU, max 1hr)
#   This node:         Codex exec with stdio MCP server (wraps retrieval API)
#
# Usage (from repo root):
#   bash scripts/sh/codex_with_arxiv/run.sh <AGENT_DIR>
#
# Examples:
#   bash scripts/sh/codex_with_arxiv/run.sh coding_agents_2026_codex_arxiv_unbiased_quarter
#   bash scripts/sh/codex_with_arxiv/run.sh coding_agents_2026_codex_arxiv_with_model_unbiased

set -euo pipefail

REPO_DIR=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
SKYRL_DIR=/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train
MCP_SERVER="${REPO_DIR}/scripts/mcp/arxiv_search_server.py"
NODE_DIR=/scratch/gpfs/ZHUANGL/sk7524/.cache/.nvm/versions/node/v24.14.0/bin
CODEX_BIN="${NODE_DIR}/codex"
export PATH="${NODE_DIR}:${PATH}"

RETRIEVAL_PORT=8000

# Parameterized agent directory
AGENT_DIR_NAME="${1:-coding_agents_2026_codex_arxiv_unbiased_quarter}"
AGENT_DIR="${REPO_DIR}/${AGENT_DIR_NAME}"
AGENT_BASENAME=$(basename "${AGENT_DIR}")

if [ ! -d "${AGENT_DIR}" ]; then
    echo "ERROR: Agent directory does not exist: ${AGENT_DIR}"
    exit 1
fi

mkdir -p "${REPO_DIR}/logs/codex_with_arxiv"

echo "============================================"
echo "Codex Agent with Arxiv Search"
echo "============================================"
echo "Agent dir: ${AGENT_DIR} (${AGENT_BASENAME})"
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
     --job-name=codex-retr-${AGENT_BASENAME} \
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
    RETRIEVAL_HOST=$(squeue --me --name="codex-retr-${AGENT_BASENAME}" -o "%N" --noheader 2>/dev/null | head -1)
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
# Write MCP config for codex (project-level)
# ============================================
CODEX_CONFIG_DIR="${AGENT_DIR}/.codex"
AGENTS_CONFIG_DIR="${CODEX_CONFIG_DIR}/agents"
mkdir -p "${AGENTS_CONFIG_DIR}"

# Resolve MODEL_PRIORS_PATH early (needed by v2.3 reviewer.toml)
MODEL_PRIORS_PATH="${REPO_DIR}/data/model_priors_${AGENT_BASENAME}.json"
if [ ! -f "${MODEL_PRIORS_PATH}" ]; then
    GATED_SUFFIX=""
    case "${AGENT_BASENAME}" in
        *_gated_v2.5_quarter) GATED_SUFFIX="gated_v2.5_quarter" ;;
        *_gated_v2.5)         GATED_SUFFIX="gated_v2.5_full" ;;
        *_gated_v2.4_quarter) GATED_SUFFIX="gated_v2.4_quarter" ;;
        *_gated_v2.4)         GATED_SUFFIX="gated_v2.4_full" ;;
        *_gated_v2.3_quarter) GATED_SUFFIX="gated_v2.3_quarter" ;;
        *_gated_v2.3)         GATED_SUFFIX="gated_v2.3_full" ;;
        *_gated_v2.2_quarter) GATED_SUFFIX="gated_v2.2_quarter" ;;
        *_gated_v2.2)         GATED_SUFFIX="gated_v2.2_full" ;;
        *_gated_v2.1_quarter) GATED_SUFFIX="gated_v2.1_quarter" ;;
        *_gated_v2.1)         GATED_SUFFIX="gated_v2.1_full" ;;
        *_gated_v2_quarter)   GATED_SUFFIX="gated_v2_quarter" ;;
        *_gated_v2)           GATED_SUFFIX="gated_v2_full" ;;
        *_gated_quarter)      GATED_SUFFIX="gated_quarter" ;;
        *_gated)              GATED_SUFFIX="gated_full" ;;
    esac
    if [ -n "${GATED_SUFFIX}" ]; then
        MODEL_PRIORS_PATH="${REPO_DIR}/data/model_priors_${GATED_SUFFIX}.json"
    fi
fi

# Detect version for config generation
AGENT_VERSION=""
case "${AGENT_BASENAME}" in
    *_aligned_*) AGENT_VERSION="aligned" ;;
    *_gated_v2.5*) AGENT_VERSION="v2.5" ;;
    *_gated_v2.4*) AGENT_VERSION="v2.4" ;;
    *_gated_v2.3*) AGENT_VERSION="v2.3" ;;
esac

if [ "${AGENT_VERSION}" = "aligned" ]; then
    # Aligned: single reviewer role, arxiv search, structured ICLR output
    # No model_prior MCP needed — model predictions are inline (with-model) or absent (no-model)

    cat > "${AGENTS_CONFIG_DIR}/reviewer.toml" <<EOF
model = "gpt-5.4"
model_reasoning_effort = "medium"
developer_instructions = "You are a paper reviewer sub-agent producing structured ICLR-format reviews. Read the paper text and page images carefully. Use search_arxiv to verify novelty and find missing baselines. Output a complete structured review with all required numeric scores (rating 0-10, confidence 1-5, soundness 1-4, presentation 1-4, contribution 1-4) and text fields (summary, strengths, weaknesses, questions, missing_references). Study the human review patterns from training data to calibrate your scores."

[mcp_servers.arxiv_search]
command = "${REPO_DIR}/.venv/bin/python"
args = ["${MCP_SERVER}"]
startup_timeout_sec = 120

[mcp_servers.arxiv_search.env]
RETRIEVAL_URL = "${RETRIEVAL_URL}"
EOF

    cat > "${CODEX_CONFIG_DIR}/config.toml" <<EOF
[agents]
max_threads = 20
job_max_runtime_seconds = 3600

[agents.reviewer]
description = "Paper reviewer sub-agent that reads papers, inspects page images, searches arxiv, and produces structured ICLR-format reviews with numeric scores."
config_file = "agents/reviewer.toml"

[mcp_servers.arxiv_search]
command = "${REPO_DIR}/.venv/bin/python"
args = ["${MCP_SERVER}"]
startup_timeout_sec = 120

[mcp_servers.arxiv_search.env]
RETRIEVAL_URL = "${RETRIEVAL_URL}"
EOF

elif [ "${AGENT_VERSION}" = "v2.5" ] || [ "${AGENT_VERSION}" = "v2.4" ]; then
    # v2.4/v2.5: two-role pipeline — reviewer (arxiv) + reconciler (model_prior)
    # Both use gpt-5.4 at medium reasoning

    # Determine budget for model prior calls
    # v2.5: budget = test_size * 2 (enough for test + calibration subset)
    # v2.4: unlimited
    MODEL_PRIOR_BUDGET="0"
    if [ "${AGENT_VERSION}" = "v2.5" ]; then
        # Count test papers to set budget
        TEST_COUNT=$(python3 -c "import json; print(len(json.load(open('${AGENT_DIR}/TEST_SAMPLE.json'))))" 2>/dev/null || echo "200")
        MODEL_PRIOR_BUDGET=$((TEST_COUNT * 2))
        echo "Model prior budget: ${MODEL_PRIOR_BUDGET} (test=${TEST_COUNT} × 2)"
    fi

    # Reviewer: reads papers + images, writes independent review, encouraged to use arxiv search
    cat > "${AGENTS_CONFIG_DIR}/reviewer.toml" <<EOF
model = "gpt-5.4"
model_reasoning_effort = "medium"
developer_instructions = "You are a paper reviewer sub-agent. Read the paper text and page images, write a thorough independent structured review (summary, strengths, weaknesses, missing references). You have access to search_arxiv — use it to verify novelty claims, find missing baselines, and substantiate critiques with real prior work. Output your review and independent accept/reject prediction."

[mcp_servers.arxiv_search]
command = "${REPO_DIR}/.venv/bin/python"
args = ["${MCP_SERVER}"]
startup_timeout_sec = 120

[mcp_servers.arxiv_search.env]
RETRIEVAL_URL = "${RETRIEVAL_URL}"
EOF

    # Reconciler: takes reviewer output, MUST call get_model_prior, recalibrates
    cat > "${AGENTS_CONFIG_DIR}/reconciler.toml" <<EOF
model = "gpt-5.4"
model_reasoning_effort = "medium"
developer_instructions = "You are a recalibration sub-agent. You receive an independent paper review with a prediction. You MUST call get_model_prior(submission_id, prediction, review) to obtain the SFT vision model's prediction and confidence. Then genuinely recalibrate: compare the independent review against the model signal, using the calibration summary provided. Output the final decision with reasoning that explains how you weighed both signals. Do NOT skip the get_model_prior call — it is mandatory."

[mcp_servers.model_prior]
command = "${REPO_DIR}/.venv/bin/python"
args = ["${REPO_DIR}/scripts/mcp/model_prior_server.py"]
startup_timeout_sec = 30

[mcp_servers.model_prior.env]
MODEL_PRIOR_PATH = "${MODEL_PRIORS_PATH:-}"
MODEL_PRIOR_BUDGET = "${MODEL_PRIOR_BUDGET}"
EOF

    cat > "${CODEX_CONFIG_DIR}/config.toml" <<EOF
[agents]
max_threads = 20
job_max_runtime_seconds = 3600

[agents.reviewer]
description = "Paper reviewer sub-agent that reads papers, inspects page images, and writes independent structured reviews. Has access to arxiv search for novelty verification."
config_file = "agents/reviewer.toml"

[agents.reconciler]
description = "Recalibration sub-agent that takes a reviewer's independent assessment, calls get_model_prior to obtain the model signal, and produces a final accept/reject decision."
config_file = "agents/reconciler.toml"

[mcp_servers.arxiv_search]
command = "${REPO_DIR}/.venv/bin/python"
args = ["${MCP_SERVER}"]
startup_timeout_sec = 120

[mcp_servers.arxiv_search.env]
RETRIEVAL_URL = "${RETRIEVAL_URL}"
EOF

elif [ "${AGENT_VERSION}" = "v2.3" ]; then
    # v2.3: single-role, sub-agents handle full pipeline
    cat > "${AGENTS_CONFIG_DIR}/reviewer.toml" <<EOF
model = "gpt-5.1"
model_reasoning_effort = "medium"
developer_instructions = "You are a paper reviewer sub-agent. You handle the full review pipeline: read the paper text and page images, search arxiv for novelty verification, write an independent structured review, call get_model_prior with your review to receive the model signal, recalibrate your decision, and output your final verdict. Follow the rules in the task instructions provided by the orchestrator."

[mcp_servers.arxiv_search]
command = "${REPO_DIR}/.venv/bin/python"
args = ["${MCP_SERVER}"]
startup_timeout_sec = 120

[mcp_servers.arxiv_search.env]
RETRIEVAL_URL = "${RETRIEVAL_URL}"

[mcp_servers.model_prior]
command = "${REPO_DIR}/.venv/bin/python"
args = ["${REPO_DIR}/scripts/mcp/model_prior_server.py"]
startup_timeout_sec = 30

[mcp_servers.model_prior.env]
MODEL_PRIOR_PATH = "${MODEL_PRIORS_PATH:-}"
EOF

    cat > "${CODEX_CONFIG_DIR}/config.toml" <<EOF
[agents]
max_threads = 20
job_max_runtime_seconds = 3600

[agents.reviewer]
description = "Paper reviewer sub-agent for reading and evaluating individual papers"
config_file = "agents/reviewer.toml"

[mcp_servers.arxiv_search]
command = "${REPO_DIR}/.venv/bin/python"
args = ["${MCP_SERVER}"]
startup_timeout_sec = 120

[mcp_servers.arxiv_search.env]
RETRIEVAL_URL = "${RETRIEVAL_URL}"
EOF
else
    # Default config (v2.2 and earlier)
    cat > "${AGENTS_CONFIG_DIR}/reviewer.toml" <<EOF
model_reasoning_effort = "medium"
developer_instructions = "You are a paper reviewer sub-agent. Read the full paper content provided to you and produce a structured review following the format in the parent task instructions."
EOF

    cat > "${CODEX_CONFIG_DIR}/config.toml" <<EOF
[agents]
max_threads = 10

[agents.reviewer]
description = "Paper reviewer sub-agent for reading and evaluating individual papers"
config_file = "agents/reviewer.toml"

[mcp_servers.arxiv_search]
command = "${REPO_DIR}/.venv/bin/python"
args = ["${MCP_SERVER}"]
startup_timeout_sec = 120

[mcp_servers.arxiv_search.env]
RETRIEVAL_URL = "${RETRIEVAL_URL}"
EOF
fi

# Conditionally add model_prior MCP server to config.toml (MODEL_PRIORS_PATH resolved above)
if [ -f "${MODEL_PRIORS_PATH}" ]; then
    cat >> "${CODEX_CONFIG_DIR}/config.toml" <<EOF

[mcp_servers.model_prior]
command = "${REPO_DIR}/.venv/bin/python"
args = ["${REPO_DIR}/scripts/mcp/model_prior_server.py"]
startup_timeout_sec = 30

[mcp_servers.model_prior.env]
MODEL_PRIOR_PATH = "${MODEL_PRIORS_PATH}"
MODEL_PRIOR_BUDGET = "${MODEL_PRIOR_BUDGET:-0}"
EOF
    echo "Added model_prior MCP server (priors at ${MODEL_PRIORS_PATH}, budget=${MODEL_PRIOR_BUDGET:-0})"

    # For v2.3/v2.4/v2.5, update agent configs with the resolved MODEL_PRIORS_PATH
    if [ -n "${AGENT_VERSION}" ]; then
        for toml_file in "${AGENTS_CONFIG_DIR}"/*.toml; do
            if grep -q "MODEL_PRIOR_PATH" "$toml_file" 2>/dev/null; then
                sed -i "s|MODEL_PRIOR_PATH = \".*\"|MODEL_PRIOR_PATH = \"${MODEL_PRIORS_PATH}\"|" "$toml_file"
                echo "Updated $(basename "$toml_file") MODEL_PRIOR_PATH"
            fi
        done
    fi
fi

echo "Wrote MCP config to ${CODEX_CONFIG_DIR}/config.toml"
echo "Wrote reviewer agent config to ${AGENTS_CONFIG_DIR}/reviewer.toml"
echo "============================================"

# ============================================
# Run codex (on this node — no GPU needed)
# ============================================
cd "${REPO_DIR}"

# Create paper_images symlink for vision experiments (v2.3+)
if [ ! -e "${AGENT_DIR}/paper_images" ]; then
    ln -s "${REPO_DIR}/data/images" "${AGENT_DIR}/paper_images"
    echo "Created paper_images symlink -> ${REPO_DIR}/data/images"
fi

# Detect data format: CSV (v3.1+) or JSON (earlier)
if [ -f "${AGENT_DIR}/TEST_SAMPLE.csv" ] && [ ! -f "${AGENT_DIR}/TEST_SAMPLE.json" ]; then
    # CSV format (v3.1+): directly invoke spawn_agents_on_csv in the prompt
    # This matches the Codex docs pattern where the user prompt tells the agent to call the tool

    # Detect with-model vs no-model by checking CSV header for model_prediction column
    if head -1 "${AGENT_DIR}/TEST_SAMPLE.csv" | grep -q "model_prediction"; then
        # With-model: sub-agent instruction includes model prior columns
        read -r -d '' CODEX_PROMPT <<'PROMPT_EOF' || true
Read AGENTS.override.md for output schema, calibration guidance, and review rules.

Run: python main.py train-summary
Study the calibration statistics it prints — especially how model_prediction accuracy varies by model_confidence. You will embed key findings in the sub-agent instruction below.

Then call spawn_agents_on_csv with:
- csv_path: TEST_SAMPLE.csv
- id_column: submission_id
- instruction: "You are reviewing ICLR 2026 paper {submission_id}. A fine-tuned vision model predicts {model_prediction} with confidence {model_confidence} — use this as a Bayesian prior. Read the paper content from {content_json}. Examine the page images listed in that JSON file under img_pages. Call search_arxiv multiple times to verify novelty and find missing references. Write a structured ICLR review with all 11 fields (decision, rating, confidence, soundness, presentation, contribution, summary, strengths, weaknesses, questions, missing_references). Strengths and weaknesses must cite specific sections, theorems, tables, or figures. Return your review as JSON via report_agent_job_result."
- output_csv_path: reviews.csv
- output_schema: an object with required string fields submission_id, decision, rating, confidence, soundness, presentation, contribution, summary, strengths, weaknesses, questions, missing_references
- max_concurrency: 20

After reviews.csv is complete, parse it and assemble PREDICTIONS.json with the schema from AGENTS.override.md (convert numeric strings to integers: rating 0-10, confidence 1-5, soundness/presentation/contribution 1-4). Then run: python main.py validate-predictions --path PREDICTIONS.json
PROMPT_EOF
    else
        # No-model: sub-agent instruction has no model prior
        read -r -d '' CODEX_PROMPT <<'PROMPT_EOF' || true
Read AGENTS.override.md for output schema, calibration guidance, and review rules.

Run: python main.py train-summary
Study the calibration statistics it prints. You will embed key findings in the sub-agent instruction below.

Then call spawn_agents_on_csv with:
- csv_path: TEST_SAMPLE.csv
- id_column: submission_id
- instruction: "You are reviewing ICLR 2026 paper {submission_id}. Read the paper content from {content_json}. Examine the page images listed in that JSON file under img_pages. Call search_arxiv multiple times to verify novelty and find missing references. Write a structured ICLR review with all 11 fields (decision, rating, confidence, soundness, presentation, contribution, summary, strengths, weaknesses, questions, missing_references). Strengths and weaknesses must cite specific sections, theorems, tables, or figures. Return your review as JSON via report_agent_job_result."
- output_csv_path: reviews.csv
- output_schema: an object with required string fields submission_id, decision, rating, confidence, soundness, presentation, contribution, summary, strengths, weaknesses, questions, missing_references
- max_concurrency: 20

After reviews.csv is complete, parse it and assemble PREDICTIONS.json with the schema from AGENTS.override.md (convert numeric strings to integers: rating 0-10, confidence 1-5, soundness/presentation/contribution 1-4). Then run: python main.py validate-predictions --path PREDICTIONS.json
PROMPT_EOF
    fi
else
    CODEX_PROMPT="Read AGENTS.override.md for your task instructions. Predict accept/reject for all papers in TEST_SAMPLE.json and write PREDICTIONS.json."
fi

echo "Starting codex exec for ${AGENT_BASENAME} ..."
"${CODEX_BIN}" exec --sandbox workspace-write \
    -C "${AGENT_DIR}" \
    "${CODEX_PROMPT}" \
    -c model_reasoning_effort=high

echo "Codex complete for ${AGENT_BASENAME}."
