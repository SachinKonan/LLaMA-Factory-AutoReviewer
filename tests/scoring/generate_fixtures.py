"""Generate the golden scoring fixtures + expected scores (run on a GPU node).

This is a *generation* tool, not a test. It produces the artifacts the
gpu-marked ``test_golden_scoring.py`` then re-verifies:

    tests/scoring/fixtures/<pid>/sharegpt_vision_entry.json   (relative images)
    tests/scoring/fixtures/<pid>/page_*.png
    tests/scoring/expected_scores.json

Pipeline (all GPU work runs as isolated subprocesses — two vLLM engines
cannot share one interpreter):

  1. paperprep  (paperprep venv) runs the 6 golden PDFs end to end ->
     <work>/prep_out/sharegpt/vision/data.json + page PNGs.
  2. Stage each paper's entry + PNGs into tests/scoring/fixtures/<pid>/,
     rewriting ``images`` to relative filenames so the fixtures are
     self-contained and relocatable.
  3. REFERENCE inference: scripts/vllm_infer.py (the offline batch path the
     paper used) over a per-domain sharegpt dataset, with arxiv papers
     scored by the arxiv ckpt and iclr papers by the iclr ckpt. Parse the
     ``logprob_accept[5]`` / ``logprob_reject[5]`` arrays -> p_accept. This
     is "the local inference result" we anchor to.
  4. CROSS-CHECK: score the same staged fixtures via the deployment
     Scorer (tests/scoring/score_with_scorer.py) and log the max diff vs
     the reference. They share the LF data pipeline + vLLM engine so they
     should agree to ~1e-6 on the same GPU; a large diff means the Scorer
     drifted from the offline batch.

Domains map to checkpoints (both local — the iclr HF repo is an empty stub):
    arxiv -> saves/.../arxiv_train/small/arxiv_21k_vision_3b/checkpoint-5236
    iclr  -> saves/.../optim_search_2026/scaling/bz16_lr1e-6_vision_3b/checkpoint-5296
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent                       # tests/scoring
REPO = HERE.parent.parent                                    # worktree root
FIXTURES = HERE / "fixtures"
EXPECTED = HERE / "expected_scores.json"

# Golden papers (paperprep tests/fixtures/golden/inputs/) by domain.
ARXIV_IDS = ["arxiv_2201.00346", "arxiv_2301.08170", "arxiv_2511.13144"]
ICLR_IDS = ["iclr_ryxOUTVYDH", "iclr_ryxyCeHtPB", "iclr_ryxgJTEYDr"]
ALL_IDS = ARXIV_IDS + ICLR_IDS

CKPTS = {
    "arxiv": "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/saves/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/small/arxiv_21k_vision_3b/checkpoint-5236",
    "iclr": "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/saves/final_sweep_v7_datasweepv3/optim_search_2026/scaling/bz16_lr1e-6_vision_3b/checkpoint-5296",
}
DOMAIN_IDS = {"arxiv": ARXIV_IDS, "iclr": ICLR_IDS}

# Inference config — must match tests/scoring/score_with_scorer.py and
# configs/serve.yaml so the reference and the Scorer are comparable.
TEMPLATE = "qwen2_vl"
CUTOFF_LEN = 24480
MAX_NEW_TOKENS = 8
IMAGE_MAX_PIXELS = 1003520
IMAGE_MIN_PIXELS = 784
POSITIVE_TOKEN = "Accept"
NEGATIVE_TOKEN = "Reject"
DECISION_TOKEN_IDX = 5

GOLDEN_INPUTS = Path("/scratch/gpfs/ZHUANGL/sk7524/PaperLens/paperprep/tests/fixtures/golden/inputs")

_SHAREGPT_TAGS = {
    "role_tag": "from", "content_tag": "value", "user_tag": "human",
    "assistant_tag": "gpt", "system_tag": "system",
}

# paperprep emits inference-only rows ([system, human], no gpt). LF's "ppo"
# converter requires an EVEN non-system message count (user+assistant pairs),
# so it drops a lone [human]. The deployment client (paperlensreview
# pipeline.py) appends this exact placeholder assistant turn before scoring;
# we mirror it so the fixtures are byte-faithful to real deployment input.
# It's the dropped "label" — generation still starts from [system, human], so
# the placeholder text does not affect the score.
PLACEHOLDER_GPT = {"from": "gpt", "value": "Outcome: \\boxed{Accept}"}


def _with_gpt_turn(convs: list[dict]) -> list[dict]:
    convs = list(convs)
    if not any(c.get("from") == "gpt" for c in convs):
        convs.append(dict(PLACEHOLDER_GPT))
    return convs


def _softmax2(la: float, lr: float) -> float:
    m = max(la, lr)
    ea, eb = math.exp(la - m), math.exp(lr - m)
    return ea / (ea + eb)


def run_paperprep(paperprep_venv: Path, paperprep_repo: Path, work: Path) -> Path:
    """Run paperprep on the 6 golden PDFs; return prep_out dir."""
    prep_out = work / "prep_out"
    vision_data = prep_out / "sharegpt" / "vision" / "data.json"
    if vision_data.exists():
        print(f"[prep] reuse existing {vision_data}")
        return prep_out

    manifest = work / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as f:
        for pid in ALL_IDS:
            f.write(json.dumps({"id": pid, "type": "pdf",
                                "path": str(GOLDEN_INPUTS / f"{pid}.pdf")}) + "\n")

    cmd = [
        str(paperprep_venv / "bin" / "python"), "-m", "paper_anonymizer.paperprep.cli", "run",
        "--input-manifest", str(manifest),
        "--output-dir", str(prep_out),
        "--stages", "compile,mineru,normalize,filter,export",
        "--no-resume",
    ]
    print(f"[prep] {' '.join(cmd)}")
    env_path = f"{paperprep_venv / 'bin'}:{__import__('os').environ.get('PATH','')}"
    r = subprocess.run(cmd, cwd=str(paperprep_repo),
                       env={**__import__('os').environ, "PATH": env_path})
    if r.returncode != 0:
        raise RuntimeError(f"paperprep failed rc={r.returncode}")
    if not vision_data.exists():
        raise RuntimeError(f"paperprep produced no vision data.json at {vision_data}")
    return prep_out


def stage_fixtures(prep_out: Path) -> dict[str, dict]:
    """Copy PNGs + write relative-image entries into tests/scoring/fixtures/<pid>/.

    Returns {pid: entry_with_absolute_images} for the reference inference step.
    """
    vision_data = json.loads((prep_out / "sharegpt" / "vision" / "data.json").read_text())
    by_id = {e["_metadata"]["id"]: e for e in vision_data}
    missing = set(ALL_IDS) - set(by_id)
    if missing:
        raise RuntimeError(f"paperprep output missing papers: {sorted(missing)}")

    abs_rows: dict[str, dict] = {}
    for pid in ALL_IDS:
        entry = by_id[pid]
        pid_dir = FIXTURES / pid
        if pid_dir.exists():
            shutil.rmtree(pid_dir)
        pid_dir.mkdir(parents=True)

        src_pngs = [Path(p) for p in entry.get("images", [])]
        rel_names: list[str] = []
        for src in src_pngs:
            dst = pid_dir / src.name
            shutil.copyfile(src, dst)
            rel_names.append(src.name)

        convs = _with_gpt_turn(entry["conversations"])

        # Staged entry: relative image filenames, self-contained.
        staged = {
            "conversations": convs,
            "images": rel_names,
            "_metadata": entry.get("_metadata", {}),
        }
        (pid_dir / "sharegpt_vision_entry.json").write_text(json.dumps(staged, indent=2))

        # Absolute-path copy for the reference inference dataset.
        abs_rows[pid] = {
            "conversations": convs,
            "images": [str((pid_dir / n).resolve()) for n in rel_names],
            "_metadata": entry.get("_metadata", {}),
        }
        print(f"[stage] {pid}: {len(rel_names)} pages -> {pid_dir}")
    return abs_rows


def _write_lf_dataset(rows: list[dict], ds_dir: Path, ds_name: str) -> None:
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / f"{ds_name}.json").write_text(json.dumps(rows))
    info = {ds_name: {
        "file_name": f"{ds_name}.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": _SHAREGPT_TAGS,
    }}
    (ds_dir / "dataset_info.json").write_text(json.dumps(info))


def reference_inference(domain: str, rows: list[dict], work: Path) -> list[dict]:
    """Run scripts/vllm_infer.py for one domain; return per-row p_accept dicts
    in the same order as `rows`."""
    ds_dir = work / f"refds_{domain}"
    ds_name = f"golden_{domain}"
    _write_lf_dataset(rows, ds_dir, ds_name)

    pred = work / f"predictions_{domain}.jsonl"
    cmd = [
        sys.executable, str(REPO / "scripts" / "vllm_infer.py"),
        "--model_name_or_path", CKPTS[domain],
        "--dataset", ds_name,
        "--dataset_dir", str(ds_dir),
        "--template", TEMPLATE,
        "--cutoff_len", str(CUTOFF_LEN),
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--temperature", "0",
        "--top_p", "1.0",
        "--top_k", "-1",
        "--repetition_penalty", "1.0",
        "--enable_thinking", "False",
        "--save_logprobs", "True",
        "--positive_token", POSITIVE_TOKEN,
        "--negative_token", NEGATIVE_TOKEN,
        "--image_max_pixels", str(IMAGE_MAX_PIXELS),
        "--image_min_pixels", str(IMAGE_MIN_PIXELS),
        "--save_name", str(pred),
    ]
    print(f"[ref:{domain}] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        raise RuntimeError(f"vllm_infer ({domain}) failed rc={r.returncode}")

    out = [json.loads(l) for l in pred.read_text().splitlines() if l.strip()]
    if len(out) != len(rows):
        raise RuntimeError(f"{domain}: vllm_infer produced {len(out)} rows, expected {len(rows)}")

    results = []
    for o in out:
        la_arr = o.get("logprob_accept") or []
        lr_arr = o.get("logprob_reject") or []
        idx = DECISION_TOKEN_IDX
        la = la_arr[idx] if idx < len(la_arr) else None
        lr = lr_arr[idx] if idx < len(lr_arr) else None
        if la is None or lr is None:
            results.append({"p_accept": 0.5, "logp_accept": la, "logp_reject": lr,
                            "pred": o.get("predict", "")})
        else:
            results.append({"p_accept": _softmax2(la, lr), "logp_accept": la,
                            "logp_reject": lr, "pred": o.get("predict", "")})
    return results


def scorer_check(domain: str, work: Path) -> dict[str, dict]:
    """Score the staged fixtures via the deployment Scorer (subprocess)."""
    out = work / f"scorer_{domain}.json"
    cmd = [
        sys.executable, str(HERE / "score_with_scorer.py"),
        "--ckpt", CKPTS[domain],
        "--fixtures-dir", str(FIXTURES),
        "--pids", ",".join(DOMAIN_IDS[domain]),
        "--out", str(out),
    ]
    print(f"[scorer:{domain}] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        raise RuntimeError(f"score_with_scorer ({domain}) failed rc={r.returncode}")
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="scratch work dir (shared FS)")
    ap.add_argument("--paperprep-venv", required=True)
    ap.add_argument("--paperprep-repo", required=True)
    ap.add_argument("--skip-scorer-check", action="store_true",
                    help="skip the Scorer cross-check (reference only)")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    prep_out = run_paperprep(Path(args.paperprep_venv), Path(args.paperprep_repo), work)
    abs_rows = stage_fixtures(prep_out)

    expected: dict[str, dict] = {}
    max_diff = 0.0
    for domain, ids in DOMAIN_IDS.items():
        rows = [abs_rows[pid] for pid in ids]
        ref = reference_inference(domain, rows, work)
        for pid, r in zip(ids, ref):
            expected[pid] = {
                "domain": domain,
                "ckpt": CKPTS[domain],
                "p_accept": r["p_accept"],
                "logp_accept": r["logp_accept"],
                "logp_reject": r["logp_reject"],
                "pred": r["pred"],
                "decision": "Accept" if r["p_accept"] >= 0.5 else "Reject",
            }
            print(f"[ref:{domain}] {pid}: p_accept={r['p_accept']:.6f} pred={r['pred']!r}")

        if not args.skip_scorer_check:
            scored = scorer_check(domain, work)
            for pid in ids:
                d = abs(scored[pid]["p_accept"] - expected[pid]["p_accept"])
                max_diff = max(max_diff, d)
                print(f"[diff:{domain}] {pid}: |scorer-ref|={d:.2e}")

    EXPECTED.write_text(json.dumps(expected, indent=2))
    print(f"\n[done] wrote {EXPECTED} ({len(expected)} papers)")
    if not args.skip_scorer_check:
        print(f"[done] max |Scorer - reference| p_accept diff = {max_diff:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
