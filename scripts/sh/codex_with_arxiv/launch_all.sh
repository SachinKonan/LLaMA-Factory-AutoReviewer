#!/bin/bash
# Launch all 4 arxiv-search Codex experiments.
# Each run.sh is backgrounded — it srun's its own retrieval server on gpu-test
# and runs Codex on this node.
#
# Usage (from repo root):
#   bash scripts/sh/codex_with_arxiv/launch_all.sh
#   bash scripts/sh/codex_with_arxiv/launch_all.sh --quarter-only
#   bash scripts/sh/codex_with_arxiv/launch_all.sh --full-only

set -euo pipefail

REPO_DIR=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
RUN_SCRIPT="${REPO_DIR}/scripts/sh/codex_with_arxiv/run.sh"

mkdir -p "${REPO_DIR}/logs/codex_with_arxiv"

QUARTER_ONLY=false
FULL_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --quarter-only) QUARTER_ONLY=true ;;
        --full-only) FULL_ONLY=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

PIDS=()

echo "============================================"
echo "Launching Codex Arxiv Search Experiments"
echo "============================================"

# Quarter runs (fast validation — 50 test papers each)
if [ "$FULL_ONLY" = false ]; then
    echo ""
    echo "--- Quarter Runs (50 test papers) ---"

    bash "${RUN_SCRIPT}" coding_agents_2026_codex_arxiv_unbiased_quarter \
        > "${REPO_DIR}/logs/codex_with_arxiv/quarter_nomodel.log" 2>&1 &
    PIDS+=($!)
    echo "Launched: quarter-nomodel (pid ${PIDS[-1]})"

    bash "${RUN_SCRIPT}" coding_agents_2026_codex_arxiv_with_model_unbiased_quarter \
        > "${REPO_DIR}/logs/codex_with_arxiv/quarter_model.log" 2>&1 &
    PIDS+=($!)
    echo "Launched: quarter-model   (pid ${PIDS[-1]})"
fi

# Full runs (200 test papers each)
if [ "$QUARTER_ONLY" = false ]; then
    echo ""
    echo "--- Full Runs (200 test papers) ---"

    bash "${RUN_SCRIPT}" coding_agents_2026_codex_arxiv_unbiased \
        > "${REPO_DIR}/logs/codex_with_arxiv/full_nomodel.log" 2>&1 &
    PIDS+=($!)
    echo "Launched: full-nomodel    (pid ${PIDS[-1]})"

    bash "${RUN_SCRIPT}" coding_agents_2026_codex_arxiv_with_model_unbiased \
        > "${REPO_DIR}/logs/codex_with_arxiv/full_model.log" 2>&1 &
    PIDS+=($!)
    echo "Launched: full-model      (pid ${PIDS[-1]})"
fi

echo ""
echo "All experiments launched. PIDs: ${PIDS[*]}"
echo "Logs in: ${REPO_DIR}/logs/codex_with_arxiv/"
echo ""
echo "Monitor retrieval servers: squeue --me | grep codex-retr"
echo "Wait for all: wait ${PIDS[*]}"
