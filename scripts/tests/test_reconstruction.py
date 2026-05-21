"""Round-trip test: reconstruct.py output must match the source data.json
byte-for-byte on conv[1] (system + human + gpt), _metadata, and image
dimensions (vision).

Runs against a locally-built HF release (passed via PAPERLENS_HF_LOCAL_DIR)
and the source main repo (passed via PAPERLENS_SOURCE_REPO). If either env
var is unset, the test self-skips.

Usage:

    export PAPERLENS_HF_LOCAL_DIR=/scratch/.../hf_release_local
    export PAPERLENS_SOURCE_REPO=/scratch/.../LLaMA-Factory-AutoReviewer
    pytest scripts/tests/test_reconstruction.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


SOURCE_REPO = os.environ.get("PAPERLENS_SOURCE_REPO")
HF_LOCAL = os.environ.get("PAPERLENS_HF_LOCAL_DIR")

# A small representative set covering text+vision, arxiv+iclr, base+yup
ROUND_TRIP_KEYS = [
    "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test",
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_test",
    "arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_test",
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_y25up_test",
]


pytestmark = pytest.mark.skipif(
    not (SOURCE_REPO and HF_LOCAL),
    reason="set PAPERLENS_SOURCE_REPO + PAPERLENS_HF_LOCAL_DIR to run",
)


def _load_source(key: str) -> dict[str, dict]:
    """Load source data.json from main repo, index by paper_id."""
    info = json.loads(Path(f"{SOURCE_REPO}/data/dataset_info.json").read_text())
    rel = info[key]["file_name"]
    p = Path(SOURCE_REPO) / "data" / rel
    if p.is_dir():
        p = p / "data.json"
    rows = json.loads(p.read_text())

    def pid(r):
        md = r.get("_metadata") or {}
        return md.get("arxiv_id") or md.get("submission_id")

    return {pid(r): r for r in rows if pid(r)}


def _load_reconstructed(key: str, work_root: Path) -> dict[str, dict]:
    p = work_root / "data" / key / "data.json"
    rows = json.loads(p.read_text())

    def pid(r):
        md = r.get("_metadata") or {}
        return md.get("arxiv_id") or md.get("submission_id")

    return {pid(r): r for r in rows if pid(r)}


@pytest.fixture(scope="module")
def reconstructed_data_root():
    """Run reconstruction.py on the test keys into a temp dir; return its data/ path."""
    with tempfile.TemporaryDirectory(prefix="paperlens_recon_") as td:
        td = Path(td)
        # Seed an empty dataset_info.json so reconstruction can append
        (td / "data").mkdir()
        (td / "data" / "dataset_info.json").write_text("{}")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "reconstruction.py"),
            "--local_dir", str(HF_LOCAL),
            "--data_root", str(td / "data"),
            "--dataset_keys", *ROUND_TRIP_KEYS,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"reconstruction failed:\n{r.stderr}"
        yield td / "data"


@pytest.mark.parametrize("key", ROUND_TRIP_KEYS)
def test_roundtrip_byte_identical(reconstructed_data_root, key):
    src_by_pid = _load_source(key)
    rec_by_pid = _load_reconstructed(key, reconstructed_data_root.parent)
    assert len(rec_by_pid) == len(src_by_pid), \
        f"row count mismatch: src={len(src_by_pid)} rec={len(rec_by_pid)}"

    for pid, src in src_by_pid.items():
        rec = rec_by_pid.get(pid)
        assert rec is not None, f"paper {pid} missing in reconstructed"

        # conv[0,1,2] byte-identical (sys + human + gpt)
        for i, turn in enumerate(src["conversations"]):
            assert rec["conversations"][i]["from"] == turn["from"], \
                f"[{key}] {pid}: turn[{i}].from mismatch"
            assert rec["conversations"][i]["value"] == turn["value"], \
                f"[{key}] {pid}: turn[{i}].value differs"

        # metadata round-trip (JSON-string -> dict -> dict comparison)
        rec_md = rec["_metadata"]
        src_md = src["_metadata"]
        assert set(rec_md.keys()) == set(src_md.keys()), \
            f"[{key}] {pid}: metadata key set differs"
        for k in src_md:
            sa, ra = src_md[k], rec_md[k]
            assert sa == ra or str(sa) == str(ra), \
                f"[{key}] {pid}: metadata[{k!r}] differs: src={sa!r} rec={ra!r}"

        # Vision: image count + dimension match
        if "vision" in key:
            src_imgs = src.get("images", []) or []
            rec_imgs = rec.get("images", []) or []
            assert len(rec_imgs) == len(src_imgs), \
                f"[{key}] {pid}: image count mismatch"
            if src_imgs:
                from PIL import Image
                src_size = Image.open(Path(SOURCE_REPO) / src_imgs[0]).size
                rec_size = Image.open(Path(reconstructed_data_root.parent) / rec_imgs[0]).size
                assert src_size == rec_size, \
                    f"[{key}] {pid}: image[0] size mismatch: src={src_size} rec={rec_size}"
