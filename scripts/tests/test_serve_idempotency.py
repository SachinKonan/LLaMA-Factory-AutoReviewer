"""Idempotency: paperlens-serve /score == scripts/vllm_infer.py == RANKER.md §6.1 parquet.

The contract: identical sharegpt input must produce identical ``p_accept``
across every code path that touches the model -- training-time eval,
``scripts/vllm_infer.py`` offline batches, and the persistent
``paperlens serve`` HTTP endpoint. This works because all three call into
LlamaFactory's ``get_dataset(template_obj, ..., "ppo", ...)`` tokenization
path; switching to a different chat-template / image-processor invocation
silently shifts logprobs in the third decimal place.

Gold reference: ``litsearch_eval/passover/predictions_3b.parquet`` from
RANKER.md §6.1 (28,664 rows of ``(arxiv_id, p_accept_3b)`` produced by
``scripts/vllm_infer.py`` with ``--save_logprobs --temperature 0`` on the
arxiv_21k_vision_3b/checkpoint-5236 model).

This test has THREE assertions, each gated by an env var so we can choose
which paths to exercise in a given run:

1. ``test_serve_matches_offline_parquet`` -- POSTs N rows to a running
   paperlens-serve and compares against the parquet. Requires
   ``PAPERLENS_SERVE_URL``.
2. ``test_offline_vllm_infer_matches_parquet`` -- re-runs
   ``scripts/vllm_infer.py`` on the same N rows via subprocess and compares
   against the parquet. Catches model/code drift since the §6.1 run.
   Requires ``PAPERLENS_REGEN_FIXTURE=1``.
3. ``test_serve_is_deterministic`` -- two consecutive /score calls return
   byte-identical p_accept. Requires ``PAPERLENS_SERVE_URL``.

Tolerance: 1e-4 (fp16/bf16 last-bit noise + softmax rounding).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Configuration via env / module-level paths
# ---------------------------------------------------------------------------

# Gold parquet (RANKER.md §6.1)
GOLD_PARQUET = Path(os.environ.get(
    "PAPERLENS_GOLD_PARQUET",
    "/scratch/gpfs/ZHUANGL/sk7524/litsearch_eval/passover/predictions_3b.parquet",
))

# Source repo whose data/ tree contains the per_venue vision sharegpt files
# (look up arxiv_id -> sharegpt row across train/val/test).
SOURCE_REPO = Path(os.environ.get(
    "PAPERLENS_SOURCE_REPO",
    "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer",
))

# The three splits of the dataset RANKER.md §1.2 was scored on.
VISION_BASE = "arxiv_50_50_balanced_per_venue_vision_wmetadata_filtered24480"
SPLITS = ("train", "validation", "test")

# Tolerance — fp16/bf16 last-bit noise + softmax rounding
P_ACCEPT_TOL = 1e-4

# How many rows to test (small for speed; gold parquet has 28k).
N_TEST = int(os.environ.get("PAPERLENS_N_IDEMPOTENCY", 5))

# Optional: a paperlens-serve already running on this URL.
SERVE_URL = os.environ.get("PAPERLENS_SERVE_URL")

# Optional: enable the subprocess re-run of vllm_infer.py (heavy; needs GPU).
REGEN_FIXTURE = bool(os.environ.get("PAPERLENS_REGEN_FIXTURE"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _index_sharegpt_by_arxiv_id() -> dict[str, dict]:
    """Build {arxiv_id -> sharegpt row} across the three per_venue vision splits."""
    out: dict[str, dict] = {}
    for split in SPLITS:
        p = SOURCE_REPO / "data" / f"{VISION_BASE}_{split}" / "data.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        for row in rows:
            md = row.get("_metadata") or {}
            aid = md.get("arxiv_id")
            if aid and aid not in out:
                out[aid] = row
    return out


@pytest.fixture(scope="module")
def gold_subset():
    """N rows from the parquet, joined with their sharegpt sources."""
    if not GOLD_PARQUET.exists():
        pytest.skip(f"gold parquet missing: {GOLD_PARQUET}")
    if not (SOURCE_REPO / "data" / f"{VISION_BASE}_test" / "data.json").exists():
        pytest.skip(f"source repo data/ missing under {SOURCE_REPO}")

    import pandas as pd
    gold = pd.read_parquet(GOLD_PARQUET).head(max(N_TEST * 3, 30))   # over-fetch in case some aren't in sharegpt index
    sg_by_aid = _index_sharegpt_by_arxiv_id()

    rows = []
    for _, r in gold.iterrows():
        aid = r["arxiv_id"]
        if aid in sg_by_aid:
            rows.append({
                "arxiv_id": aid,
                "p_accept_gold": float(r["p_accept_3b"]),
                "sharegpt": sg_by_aid[aid],
            })
        if len(rows) >= N_TEST:
            break

    if len(rows) < N_TEST:
        pytest.skip(f"could only resolve {len(rows)} / {N_TEST} arxiv_ids in sharegpt index")
    return rows


# ---------------------------------------------------------------------------
# Test 1: paperlens-serve vs gold parquet
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SERVE_URL, reason="set PAPERLENS_SERVE_URL to run")
def test_serve_matches_offline_parquet(gold_subset):
    import requests

    sharegpt_rows = [r["sharegpt"] for r in gold_subset]
    payload = {"papers": sharegpt_rows}
    resp = requests.post(f"{SERVE_URL.rstrip('/')}/score", json=payload, timeout=600)
    resp.raise_for_status()
    body = resp.json()
    assert "scores" in body, f"missing 'scores' in response: {list(body)}"
    assert len(body["scores"]) == len(gold_subset), \
        f"score-count mismatch: got {len(body['scores'])}, expected {len(gold_subset)}"

    failures = []
    for ref, srv in zip(gold_subset, body["scores"]):
        p_gold = ref["p_accept_gold"]
        p_srv = float(srv["p_accept"])
        diff = abs(p_srv - p_gold)
        if diff >= P_ACCEPT_TOL:
            failures.append(
                f"  arxiv_id={ref['arxiv_id']}: gold={p_gold:.6f} serve={p_srv:.6f} |diff|={diff:.6f}"
            )
    if failures:
        msg = (
            f"paperlens-serve drifted from RANKER.md §6.1 parquet on "
            f"{len(failures)} / {len(gold_subset)} rows (tol={P_ACCEPT_TOL}):\n"
            + "\n".join(failures)
            + "\n\nLikely cause: serve is NOT going through llamafactory.data.get_dataset, "
            + "so the chat template / image processor produced different token ids."
        )
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Test 2: scripts/vllm_infer.py vs gold parquet (regen-fixture sanity check)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REGEN_FIXTURE, reason="set PAPERLENS_REGEN_FIXTURE=1 to run")
def test_offline_vllm_infer_matches_parquet(gold_subset, tmp_path: Path):
    """Re-run scripts/vllm_infer.py on the N rows and confirm match.

    Catches drift in the model/ckpt/library since predictions_3b.parquet
    was produced. Heavy: loads a 3B model + does inference on N rows
    (~30 sec per row on H100). Requires a GPU on the test host.
    """
    ckpt = os.environ.get("PAPERLENS_TEST_CKPT_PATH")
    if not ckpt:
        pytest.skip("set PAPERLENS_TEST_CKPT_PATH to the 3B vision ckpt to regen")

    # 1. Write the N rows as a temp dataset under tmp_path
    ds_name = "idempotency_check"
    ds_dir = tmp_path / "data" / ds_name
    ds_dir.mkdir(parents=True)
    sharegpt_rows = [r["sharegpt"] for r in gold_subset]
    (ds_dir / "data.json").write_text(json.dumps(sharegpt_rows))
    info = {
        ds_name: {
            "file_name": f"{ds_name}/data.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {
                "role_tag": "from", "content_tag": "value",
                "user_tag": "human", "assistant_tag": "gpt",
                "system_tag": "system",
            },
        }
    }
    (tmp_path / "data" / "dataset_info.json").write_text(json.dumps(info))

    out_path = tmp_path / "out.jsonl"
    cmd = [
        sys.executable, str(SOURCE_REPO / "scripts" / "vllm_infer.py"),
        "--model_name_or_path", ckpt,
        "--dataset", ds_name,
        "--dataset_dir", str(tmp_path / "data"),
        "--template", "qwen2_vl",
        "--cutoff_len", "24480",
        "--max_new_tokens", "8",
        "--temperature", "0",
        "--save_logprobs", "True",
        "--save_name", str(out_path),
        "--batch_size", str(N_TEST),
    ]
    env = {**os.environ, "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}
    subprocess.run(cmd, check=True, env=env)
    assert out_path.exists(), f"vllm_infer.py did not produce {out_path}"

    # The output is JSONL; rows match the input order.
    actual = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    assert len(actual) == len(gold_subset)

    DECISION_IDX = 5
    failures = []
    for ref, row in zip(gold_subset, actual):
        la = row["logprob_accept"][DECISION_IDX]
        lr = row["logprob_reject"][DECISION_IDX]
        if la is None or lr is None:
            failures.append(f"  arxiv_id={ref['arxiv_id']}: missing logprob at idx {DECISION_IDX}")
            continue
        m = max(la, lr)
        p_actual = math.exp(la - m) / (math.exp(la - m) + math.exp(lr - m))
        diff = abs(p_actual - ref["p_accept_gold"])
        if diff >= P_ACCEPT_TOL:
            failures.append(
                f"  arxiv_id={ref['arxiv_id']}: gold={ref['p_accept_gold']:.6f} "
                f"regen={p_actual:.6f} |diff|={diff:.6f}"
            )
    if failures:
        pytest.fail(
            f"vllm_infer.py drifted from gold parquet on {len(failures)} rows:\n"
            + "\n".join(failures)
        )


# ---------------------------------------------------------------------------
# Test 3: serve is deterministic across calls
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SERVE_URL, reason="set PAPERLENS_SERVE_URL to run")
def test_serve_is_deterministic(gold_subset):
    """Two consecutive /score calls with the same input must return identical p_accept."""
    import requests

    sharegpt_rows = [r["sharegpt"] for r in gold_subset]
    payload = {"papers": sharegpt_rows}
    a = requests.post(f"{SERVE_URL.rstrip('/')}/score", json=payload, timeout=600).json()
    b = requests.post(f"{SERVE_URL.rstrip('/')}/score", json=payload, timeout=600).json()
    for sa, sb in zip(a["scores"], b["scores"]):
        assert sa["p_accept"] == sb["p_accept"], \
            f"non-deterministic: {sa['p_accept']} vs {sb['p_accept']}"
