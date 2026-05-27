"""Score staged golden fixtures with ``paperlens_cli.scoring.Scorer``.

Run as an isolated subprocess (one vLLM engine per process — two Scorers
cannot coexist in one interpreter). Both the fixture generator and the
pytest invoke this so the *exact* deployment scoring path is exercised:

    PYTHONPATH=<worktree>/src <LF-venv>/bin/python tests/scoring/score_with_scorer.py \
        --ckpt <checkpoint_dir> --fixtures-dir tests/scoring/fixtures \
        --pids arxiv_2201.00346,arxiv_2301.08170 --out /tmp/scores.json

Each ``<fixtures-dir>/<pid>/sharegpt_vision_entry.json`` holds an
inference-only ([system, human]) sharegpt row whose ``images`` are
*relative* PNG filenames living beside it; we resolve them to absolute
paths so the fixtures stay self-contained and relocatable.

Config mirrors ``configs/serve.yaml`` exactly (the deployment defaults) so
parity with the offline ``scripts/vllm_infer.py`` batch is meaningful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# Deployment defaults — identical to configs/serve.yaml. Keeping them here
# (rather than reading the yaml) lets the worker swap only ckpt_path between
# the arxiv and iclr checkpoints while everything else stays pinned.
TEMPLATE = "qwen2_vl"
CUTOFF_LEN = 24480
MAX_NEW_TOKENS = 8
IMAGE_MAX_PIXELS = 1003520        # 1024 * 980 — training image pixel cap
IMAGE_MIN_PIXELS = 784            # 28 * 28 — training image pixel floor
POSITIVE_TOKEN = "Accept"
NEGATIVE_TOKEN = "Reject"
DECISION_TOKEN_IDX = 5
MAX_MODEL_LEN = 24580
GPU_MEM_UTIL = 0.85


def _build_cfg(ckpt: str):
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "model": {
            "ckpt_path": ckpt,
            "template": TEMPLATE,
            "cutoff_len": CUTOFF_LEN,
            "enable_thinking": False,
            "dtype": "bfloat16",
        },
        "vllm": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "gpu_memory_utilization": GPU_MEM_UTIL,
            "max_model_len": MAX_MODEL_LEN,
            "disable_log_stats": True,
        },
        "scoring": {
            "positive_token": POSITIVE_TOKEN,
            "negative_token": NEGATIVE_TOKEN,
            "decision_token_idx": DECISION_TOKEN_IDX,
            "max_new_tokens": MAX_NEW_TOKENS,
            "image_max_pixels": IMAGE_MAX_PIXELS,
            "image_min_pixels": IMAGE_MIN_PIXELS,
        },
        "compute_arch": "test",
    })


def _load_row(entry_path: Path) -> dict:
    """Load a staged sharegpt entry and absolutize its relative image paths."""
    row = json.loads(entry_path.read_text())
    pid_dir = entry_path.parent
    if row.get("images"):
        row = dict(row)
        row["images"] = [str((pid_dir / Path(p).name).resolve()) for p in row["images"]]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir for this domain")
    ap.add_argument("--fixtures-dir", required=True, help="tests/scoring/fixtures")
    ap.add_argument("--pids", required=True, help="comma-separated paper ids to score")
    ap.add_argument("--out", required=True, help="where to write {pid: result} JSON")
    args = ap.parse_args()

    fixtures = Path(args.fixtures_dir)
    pids = [p for p in args.pids.split(",") if p]
    entry_paths = [fixtures / pid / "sharegpt_vision_entry.json" for pid in pids]
    for ep in entry_paths:
        if not ep.exists():
            raise FileNotFoundError(f"missing staged fixture: {ep}")

    rows = [_load_row(ep) for ep in entry_paths]

    from paperlens_cli.scoring import Scorer
    scorer = Scorer(_build_cfg(args.ckpt))
    results = scorer.score(rows)

    out = {
        pid: {
            "p_accept": r["p_accept"],
            "logp_accept": r["logp_accept"],
            "logp_reject": r["logp_reject"],
            "pred": r["pred"],
        }
        for pid, r in zip(pids, results)
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    for pid, r in out.items():
        print(f"[scorer] {pid}: p_accept={r['p_accept']:.6f} pred={r['pred']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
