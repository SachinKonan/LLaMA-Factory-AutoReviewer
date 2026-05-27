"""Golden scoring integration test (GPU).

Re-scores the 6 golden papers with the deployment Scorer
(``paperlens_cli.scoring.Scorer``) and asserts the results match the
checked-in ``expected_scores.json`` — which was produced by the offline
batch path (``scripts/vllm_infer.py``) in ``generate_fixtures.py``. This is
the regression gate proving the persistent server reproduces the local
inference results for the local arxiv-3b + iclr-3b checkpoints.

Run (needs a GPU + the paperlenstraininfer venv):

    cd <worktree>
    .venv/bin/python -m pytest tests/scoring/test_golden_scoring.py -m gpu -s

Each domain is scored in its own subprocess because two vLLM engines cannot
share one interpreter. The subprocess (``score_with_scorer.py``) loads the
domain's checkpoint, scores the staged fixtures, and writes a JSON the test
diffs against ``expected_scores.json``.

Tolerance: the Scorer and the offline batch share the LF data pipeline and
the same vLLM engine, so on identical hardware they agree to ~1e-6. We allow
``PAPERLENS_SCORING_TOL`` (default 0.05) to absorb cross-GPU-class drift
(scoring.py documents ~0.02 mean / 0.21 max across GPU classes) while still
catching a genuine pipeline regression. The Accept/Reject decision must match
whenever the expected p_accept is confidently off the 0.5 boundary.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
EXPECTED_PATH = HERE / "expected_scores.json"
WORKER = HERE / "score_with_scorer.py"

TOL = float(os.environ.get("PAPERLENS_SCORING_TOL", "0.05"))

pytestmark = [pytest.mark.gpu]


def _have_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except Exception:
        return False


@pytest.fixture(scope="session")
def expected() -> dict:
    if not EXPECTED_PATH.exists():
        pytest.skip(f"no expected_scores.json — run tests/scoring/generate_fixtures.sbatch first")
    return json.loads(EXPECTED_PATH.read_text())


@pytest.fixture(scope="session")
def gpu():
    if not _have_gpu():
        pytest.skip("no CUDA GPU available")
    return True


def _domains(expected: dict) -> dict[str, list[str]]:
    by_domain: dict[str, list[str]] = {}
    for pid, e in expected.items():
        by_domain.setdefault(e["domain"], []).append(pid)
    return by_domain


def _run_scorer(ckpt: str, pids: list[str], tmp_path: Path) -> dict:
    out = tmp_path / "scored.json"
    cmd = [
        sys.executable, str(WORKER),
        "--ckpt", ckpt,
        "--fixtures-dir", str(FIXTURES),
        "--pids", ",".join(pids),
        "--out", str(out),
    ]
    r = subprocess.run(cmd, cwd=str(HERE.parent.parent))
    assert r.returncode == 0, f"score_with_scorer failed rc={r.returncode}"
    return json.loads(out.read_text())


def test_fixtures_present(expected):
    """Every expected paper has a staged, self-contained fixture."""
    for pid in expected:
        entry = FIXTURES / pid / "sharegpt_vision_entry.json"
        assert entry.exists(), f"missing staged fixture for {pid}"
        row = json.loads(entry.read_text())
        assert row["images"], f"{pid}: no images in staged entry"
        # [system, human] + the placeholder gpt turn the deployment appends so
        # LF's ppo converter accepts the row (generation still runs from the
        # [system, human] prompt; the gpt turn is the dropped label).
        assert [c["from"] for c in row["conversations"]] == ["system", "human", "gpt"]
        # every referenced PNG exists beside the entry
        for img in row["images"]:
            png = FIXTURES / pid / Path(img).name
            assert png.exists() and png.stat().st_size > 0, f"{pid}: missing PNG {img}"


@pytest.mark.parametrize("domain", ["arxiv", "iclr"])
def test_scorer_reproduces_reference(domain, expected, gpu, tmp_path):
    """The Scorer reproduces the offline-batch p_accept for each domain's ckpt."""
    by_domain = _domains(expected)
    if domain not in by_domain:
        pytest.skip(f"no {domain} papers in expected_scores.json")
    pids = by_domain[domain]
    ckpt = expected[pids[0]]["ckpt"]

    scored = _run_scorer(ckpt, pids, tmp_path)

    for pid in pids:
        exp = expected[pid]
        got = scored[pid]
        diff = abs(got["p_accept"] - exp["p_accept"])
        assert diff <= TOL, (
            f"{pid}: p_accept {got['p_accept']:.6f} vs expected {exp['p_accept']:.6f} "
            f"(diff {diff:.4f} > tol {TOL})"
        )
        # Decision must match when expected is confidently off the boundary.
        if abs(exp["p_accept"] - 0.5) > TOL:
            got_decision = "Accept" if got["p_accept"] >= 0.5 else "Reject"
            assert got_decision == exp["decision"], (
                f"{pid}: decision {got_decision} != expected {exp['decision']} "
                f"(p_accept got={got['p_accept']:.4f} exp={exp['p_accept']:.4f})"
            )
