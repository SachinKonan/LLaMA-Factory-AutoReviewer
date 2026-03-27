#!/usr/bin/env python3
"""Resample the 200-paper 50/50 test set to a 30/70 accept/reject split
and create fresh coding agent directories for Claude Code inference.

Selects 30 accepts and 70 rejects (seed=42) from the existing test pool to
better reflect ICLR 2026's ~30% acceptance rate.

Creates new standalone directories:
  - coding_agents_2026_claudecode_v3.2a/  (no model prior)
  - coding_agents_2026_claudecode_v3.2b/  (with model prior)

Each contains: TEST_SAMPLE.csv, TRAIN_SAMPLE.csv, main.py, AGENTS.override.md,
papers/ (symlinked), paper_images/ (symlinked), agent_reviews/ (empty).
Also outputs: data/agent_ground_truth_2026_unbiased_30_70.json
"""

import csv
import json
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Source v3.1 dirs (for papers, train CSV, main.py, AGENTS.override.md)
SRC_NO_MODEL = ROOT / "coding_agents_2026_codex_arxiv_aligned_no_model_v3.1"
SRC_WITH_MODEL = ROOT / "coding_agents_2026_codex_arxiv_aligned_with_model_v3.1"

# New output dirs (neutral names — no split info to avoid leaking to the agent)
DST_NO_MODEL = ROOT / "coding_agents_2026_claudecode_v3.2a"
DST_WITH_MODEL = ROOT / "coding_agents_2026_claudecode_v3.2b"

GT_PATH = ROOT / "data" / "agent_ground_truth_2026_unbiased.json"
GT_OUT_PATH = ROOT / "data" / "agent_ground_truth_2026_unbiased_30_70.json"

SEED = 42
N_ACCEPT = 30
N_REJECT = 70


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_gt(path: Path) -> dict[str, str]:
    with open(path) as f:
        return json.load(f)


def setup_dir(dst: Path, src: Path, test_rows: list[dict], test_fieldnames: list[str]):
    """Create a fresh agent directory with symlinked data and new TEST_SAMPLE.csv."""
    dst.mkdir(parents=True, exist_ok=True)

    # Empty agent_reviews dir
    (dst / "agent_reviews").mkdir(exist_ok=True)

    # Symlink papers/ and paper_images/ from source
    for dirname in ["papers", "paper_images"]:
        src_path = src / dirname
        dst_path = dst / dirname
        if src_path.exists() and not dst_path.exists():
            os.symlink(src_path.resolve(), dst_path)

    # Copy main.py and AGENTS.override.md
    for fname in ["main.py", "AGENTS.override.md"]:
        src_file = src / fname
        dst_file = dst / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_file)

    # Copy TRAIN_SAMPLE.csv (same train set)
    src_train = src / "TRAIN_SAMPLE.csv"
    if src_train.exists():
        shutil.copy2(src_train, dst / "TRAIN_SAMPLE.csv")

    # Write the 30/70 test CSV as TEST_SAMPLE.csv (so main.py works as-is)
    with open(dst / "TEST_SAMPLE.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=test_fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)


def main():
    # --- Load ground truth ---
    gt = load_gt(GT_PATH)
    print(f"Loaded GT: {len(gt)} papers")

    # --- Load test CSVs from source dirs ---
    no_model_rows = load_csv(SRC_NO_MODEL / "TEST_SAMPLE.csv")
    with_model_rows = load_csv(SRC_WITH_MODEL / "TEST_SAMPLE.csv")

    test_ids_no_model = {r["submission_id"] for r in no_model_rows}
    test_ids_with_model = {r["submission_id"] for r in with_model_rows}

    assert test_ids_no_model == test_ids_with_model, "Test ID mismatch between dirs"
    test_ids = test_ids_no_model
    print(f"Test pool: {len(test_ids)} papers")

    # --- Partition by GT label ---
    accepts = sorted([sid for sid in test_ids if gt.get(sid) == "accept"])
    rejects = sorted([sid for sid in test_ids if gt.get(sid) == "reject"])
    print(f"  accepts: {len(accepts)}, rejects: {len(rejects)}")

    assert len(accepts) >= N_ACCEPT, f"Only {len(accepts)} accepts, need {N_ACCEPT}"
    assert len(rejects) >= N_REJECT, f"Only {len(rejects)} rejects, need {N_REJECT}"

    # --- Subsample ---
    rng = random.Random(SEED)
    sampled_accepts = sorted(rng.sample(accepts, N_ACCEPT))
    sampled_rejects = sorted(rng.sample(rejects, N_REJECT))
    sampled_ids = set(sampled_accepts + sampled_rejects)

    print(f"\nSampled: {len(sampled_accepts)} accepts + {len(sampled_rejects)} rejects = {len(sampled_ids)} total")

    # --- Filter rows ---
    no_model_filtered = [r for r in no_model_rows if r["submission_id"] in sampled_ids]
    with_model_filtered = [r for r in with_model_rows if r["submission_id"] in sampled_ids]

    # --- Create no-model dir ---
    setup_dir(
        DST_NO_MODEL, SRC_NO_MODEL,
        no_model_filtered,
        ["submission_id", "content_json"],
    )
    print(f"Created {DST_NO_MODEL.name}/ ({len(no_model_filtered)} test papers)")

    # --- Create with-model dir ---
    setup_dir(
        DST_WITH_MODEL, SRC_WITH_MODEL,
        with_model_filtered,
        ["submission_id", "content_json", "model_prediction", "model_confidence"],
    )
    print(f"Created {DST_WITH_MODEL.name}/ ({len(with_model_filtered)} test papers)")

    # --- Write filtered GT ---
    gt_filtered = {sid: gt[sid] for sid in sorted(sampled_ids)}
    with open(GT_OUT_PATH, "w") as f:
        json.dump(gt_filtered, f, indent=2)
    print(f"Wrote {GT_OUT_PATH} ({len(gt_filtered)} entries)")

    # --- Update AGENTS.override.md ---
    disk_save_rule = (
        "\n## Context Management\n\n"
        "**CRITICAL: Do NOT keep full review text in the main conversation context.**\n"
        "Each sub-agent must write its review directly to `agent_reviews/{submission_id}.json` on disk.\n"
        "The main agent should only receive the decision and rating back from the sub-agent, not the full review body.\n"
        "This prevents context window exhaustion when reviewing many papers.\n"
    )
    for dst in [DST_NO_MODEL, DST_WITH_MODEL]:
        agents_path = dst / "AGENTS.override.md"
        if agents_path.exists():
            text = agents_path.read_text()
            text = text.replace("200 unlabeled", "100 unlabeled")
            text = text.replace("Must predict all 200 test papers", "Must predict all 100 test papers")
            # Insert context management section before ## Environment
            text = text.replace("## Environment", disk_save_rule + "## Environment")
            agents_path.write_text(text)

    # --- Verification ---
    print("\n=== Verification ===")
    accept_count = sum(1 for sid in sampled_ids if gt[sid] == "accept")
    reject_count = sum(1 for sid in sampled_ids if gt[sid] == "reject")
    print(f"Accepts: {accept_count}, Rejects: {reject_count}, Total: {accept_count + reject_count}")
    print(f"Accept rate: {accept_count / (accept_count + reject_count):.1%}")

    # List dir contents
    for dst in [DST_NO_MODEL, DST_WITH_MODEL]:
        print(f"\n{dst.name}/")
        for item in sorted(dst.iterdir()):
            suffix = " -> " + str(item.resolve()) if item.is_symlink() else ""
            print(f"  {item.name}{suffix}")


if __name__ == "__main__":
    main()
