#!/usr/bin/env python3
"""Build 6 new Claude Code experiment directories for ICLR 2025/2026 papers.

Creates:
  - coding_agents_2026_claudecode_v3.3.1      (2026 batch 2, no model)
  - coding_agents_2026_claudecode_v3.3b.1     (2026 batch 2, with model)
  - coding_agents_2025_claudecode_v3.3_2025   (2025 batch 1, no model)
  - coding_agents_2025_claudecode_v3.3b_2025  (2025 batch 1, with model)
  - coding_agents_2025_claudecode_v3.3_2025.1  (2025 batch 2, no model)
  - coding_agents_2025_claudecode_v3.3b_2025.1 (2025 batch 2, with model)

Also writes ground truth files:
  - data/agent_ground_truth_2026_batch2.json
  - data/agent_ground_truth_2025_batch1.json
  - data/agent_ground_truth_2025_batch2.json

Usage:
  python scripts/build_cc_experiment_batches.py
"""

import csv
import json
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parent.parent

# ── Data sources ───────────────────────────────────────────────────────────
METADATA_DIR = ROOT / "data" / "massive_metadata_v7_5"
TEXT_TEST_SPLIT = ROOT / "data" / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test" / "data.json"
VISION_TEST_SPLIT = ROOT / "data" / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test" / "data.json"
VISION_PREDICTIONS = ROOT / "results" / "final_sweep_v7_datasweepv3" / "optim_search_2026" / "bz16_lr1e-6_vision" / "finetuned-ckpt-2648.jsonl"

# ── Existing experiment dirs (for train CSV, AGENTS.override.md, main.py) ─
SRC_NO_MODEL = ROOT / "coding_agents_2026_claudecode_v3.3"
SRC_WITH_MODEL = ROOT / "coding_agents_2026_claudecode_v3.3b"
PAPERS_SRC = ROOT / "coding_agents_2026_codex_arxiv_aligned_no_model_v3.1" / "papers"

# ── Exclusion sources ─────────────────────────────────────────────────────
GT_2026_PATH = ROOT / "data" / "agent_ground_truth_2026_unbiased.json"
TRAIN_SAMPLE_PATH = ROOT / "coding_agents_2026_codex_base" / "TRAIN_SAMPLE.json"

# ── Sampling params ───────────────────────────────────────────────────────
N_ACCEPT = 30
N_REJECT = 70

# Different seeds per batch for reproducibility
BATCH_CONFIGS = [
    {
        "name": "2026_batch2",
        "year": 2026,
        "seed": 100,
        "no_model_dir": "coding_agents_2026_claudecode_v3.3.1",
        "with_model_dir": "coding_agents_2026_claudecode_v3.3b.1",
        "gt_file": "agent_ground_truth_2026_batch2.json",
    },
    {
        "name": "2025_batch1",
        "year": 2025,
        "seed": 200,
        "no_model_dir": "coding_agents_2025_claudecode_v3.3_2025",
        "with_model_dir": "coding_agents_2025_claudecode_v3.3b_2025",
        "gt_file": "agent_ground_truth_2025_batch1.json",
    },
    {
        "name": "2025_batch2",
        "year": 2025,
        "seed": 300,
        "no_model_dir": "coding_agents_2025_claudecode_v3.3_2025.1",
        "with_model_dir": "coding_agents_2025_claudecode_v3.3b_2025.1",
        "gt_file": "agent_ground_truth_2025_batch2.json",
    },
]

# ── Content sanitization (from build_agent_sample_unbiased.py) ─────────────
KEEP_FIELDS = {
    "type", "text", "text_level", "text_format",
    "page_idx", "bbox",
    "image_caption", "image_footnote",
    "table_body", "table_caption", "table_footnote",
    "sub_type", "list_items",
}
LEAK_PATTERNS = [
    "we accept", "we reject", "paper is accepted", "paper is rejected",
    "decision: accept", "decision: reject",
    "overall rating", "reviewer confidence",
    "score:", "rating:",
]


def sanitize_element(elem):
    text = elem.get("text", "")
    if isinstance(text, str):
        text_lower = text.lower()
        for pattern in LEAK_PATTERNS:
            if pattern in text_lower:
                return None
    cleaned = {k: v for k, v in elem.items() if k in KEEP_FIELDS and v is not None}
    if "type" not in cleaned:
        return None
    return cleaned


def extract_headers(content_list):
    return [
        elem["text"]
        for elem in content_list
        if elem.get("text_level") is not None and elem.get("text")
    ]


def build_paper_content_json(content_json_str, submission_id):
    """Build content.json dict with content_list, headers, and img_pages."""
    content_list_raw = json.loads(content_json_str) if isinstance(content_json_str, str) else content_json_str
    content_list = [c for elem in content_list_raw if (c := sanitize_element(elem)) is not None]
    headers = extract_headers(content_list)

    # Build img_pages from data/images/{sid}/
    img_dir = ROOT / "data" / "images" / submission_id
    img_pages = []
    if img_dir.exists():
        for png in sorted(img_dir.iterdir()):
            if png.suffix == ".png":
                img_pages.append(f"paper_images/{submission_id}/{png.name}")

    return {"content_list": content_list, "headers": headers, "img_pages": img_pages}


# ── Vision prediction loading ─────────────────────────────────────────────
def extract_prediction_label(text):
    text_lower = (text or "").lower()
    accept_pos = text_lower.rfind("accept")
    reject_pos = text_lower.rfind("reject")
    if accept_pos == -1 and reject_pos == -1:
        return "unknown"
    return "accept" if accept_pos > reject_pos else "reject"


def find_decision_token_idx(all_logprobs):
    width = max(len(row) for row in all_logprobs)
    padded = []
    for row in all_logprobs:
        if len(row) < width:
            row = row + [row[-1]] * (width - len(row))
        padded.append(row)
    variances = []
    for col in range(width):
        values = [row[col] for row in padded]
        mean = sum(values) / len(values)
        variances.append(sum((v - mean) ** 2 for v in values) / len(values))
    return max(range(width), key=variances.__getitem__)


def load_vision_prediction_lookup():
    """Load vision predictions keyed by submission_id (covers all years)."""
    with open(VISION_TEST_SPLIT) as f:
        vision_rows = json.load(f)
    predictions = []
    with open(VISION_PREDICTIONS) as f:
        for line in f:
            predictions.append(json.loads(line))

    paired = list(zip(vision_rows, predictions))
    decision_token_idx = find_decision_token_idx(
        [pred["token_logprobs"] for _, pred in paired if pred.get("token_logprobs")]
    )

    pred_lookup = {}
    for row, pred in paired:
        submission_id = row["_metadata"]["submission_id"]
        decision = extract_prediction_label(pred.get("predict", ""))
        if decision == "unknown":
            continue
        token_logprobs = pred.get("token_logprobs") or []
        if not token_logprobs:
            continue
        conf_idx = min(decision_token_idx, len(token_logprobs) - 1)
        pred_lookup[submission_id] = {
            "model_prediction": decision,
            "model_confidence": round(math.exp(token_logprobs[conf_idx]), 4),
        }
    return pred_lookup


def setup_experiment_dir(dst: Path, src: Path, test_rows: list[dict],
                         test_fieldnames: list[str], papers_dir: Path):
    """Create a fresh experiment directory."""
    dst.mkdir(parents=True, exist_ok=True)

    # Empty agent_reviews dir
    (dst / "agent_reviews").mkdir(exist_ok=True)

    # Symlink paper_images -> data/images/
    images_target = ROOT / "data" / "images"
    dst_images = dst / "paper_images"
    if not dst_images.exists():
        os.symlink(images_target, dst_images)

    # Symlink papers -> shared papers dir
    dst_papers = dst / "papers"
    if not dst_papers.exists():
        os.symlink(papers_dir.resolve(), dst_papers)

    # Copy main.py, AGENTS.override.md, TRAIN_SAMPLE.csv from source
    for fname in ["main.py", "AGENTS.override.md", "TRAIN_SAMPLE.csv"]:
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(src_file, dst / fname)

    # Write TEST_SAMPLE.csv
    with open(dst / "TEST_SAMPLE.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=test_fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)


def main():
    print("=" * 60)
    print("Building 6 new Claude Code experiment batches")
    print("=" * 60)

    # ── Load text test split ──────────────────────────────────────────
    print("\nLoading text test split...")
    with open(TEXT_TEST_SPLIT) as f:
        text_rows = json.load(f)
    print(f"  Total: {len(text_rows)} papers")

    # ── Build exclusion set for 2026 ──────────────────────────────────
    with open(GT_2026_PATH) as f:
        gt_2026_existing = json.load(f)
    exclude_2026 = set(gt_2026_existing.keys())

    with open(TRAIN_SAMPLE_PATH) as f:
        train_sample = json.load(f)
    exclude_2026 |= set(train_sample.keys())
    print(f"  Excluding {len(exclude_2026)} existing 2026 IDs (GT + train)")

    # ── Load vision predictions ───────────────────────────────────────
    print("Loading vision predictions...")
    pred_lookup = load_vision_prediction_lookup()
    print(f"  {len(pred_lookup)} predictions loaded")

    # ── Load metadata for building content.json ───────────────────────
    print("Loading massive_metadata_v7_5...")
    ds = load_from_disk(str(METADATA_DIR))
    meta_index = {}
    for i in range(len(ds)):
        meta_index[ds[i]["submission_id"]] = i
    print(f"  {len(meta_index)} entries indexed")

    # ── Create shared papers directory for new papers ─────────────────
    # We'll write new content.json files here, alongside existing ones
    new_papers_dir = ROOT / "papers_cc_batches"
    new_papers_dir.mkdir(exist_ok=True)

    # Also need to include existing papers from v3.1 (for train examples)
    # We'll make new_papers_dir contain symlinks to existing + new papers

    # ── Build pools by year ───────────────────────────────────────────
    pools = {}
    for year in [2025, 2026]:
        exclude = exclude_2026 if year == 2026 else set()
        accepts = [r for r in text_rows
                   if r["_metadata"]["year"] == year
                   and r["_metadata"]["answer"].lower() == "accept"
                   and r["_metadata"]["submission_id"] not in exclude]
        rejects = [r for r in text_rows
                   if r["_metadata"]["year"] == year
                   and r["_metadata"]["answer"].lower() == "reject"
                   and r["_metadata"]["submission_id"] not in exclude]
        pools[year] = {"accepts": accepts, "rejects": rejects}
        print(f"\n  {year} pool: {len(accepts)} accepts, {len(rejects)} rejects")

    # ── Track used IDs across batches (to prevent overlap) ────────────
    used_ids = {2025: set(), 2026: set()}

    # ── Process each batch ────────────────────────────────────────────
    for config in BATCH_CONFIGS:
        year = config["year"]
        seed = config["seed"]
        name = config["name"]
        print(f"\n{'─' * 50}")
        print(f"Batch: {name} (year={year}, seed={seed})")
        print(f"{'─' * 50}")

        pool = pools[year]
        avail_accepts = [r for r in pool["accepts"]
                         if r["_metadata"]["submission_id"] not in used_ids[year]]
        avail_rejects = [r for r in pool["rejects"]
                         if r["_metadata"]["submission_id"] not in used_ids[year]]
        print(f"  Available: {len(avail_accepts)} accepts, {len(avail_rejects)} rejects")

        assert len(avail_accepts) >= N_ACCEPT, \
            f"Only {len(avail_accepts)} accepts available, need {N_ACCEPT}"
        assert len(avail_rejects) >= N_REJECT, \
            f"Only {len(avail_rejects)} rejects available, need {N_REJECT}"

        # Sample
        rng = random.Random(seed)
        sampled_accepts = rng.sample(avail_accepts, N_ACCEPT)
        sampled_rejects = rng.sample(avail_rejects, N_REJECT)
        sampled = sampled_accepts + sampled_rejects

        # Shuffle order (so accepts/rejects are mixed)
        rng.shuffle(sampled)

        sampled_ids = {r["_metadata"]["submission_id"] for r in sampled}
        used_ids[year] |= sampled_ids

        # Build ground truth
        ground_truth = {}
        for r in sampled:
            sid = r["_metadata"]["submission_id"]
            ground_truth[sid] = r["_metadata"]["answer"].lower()

        gt_labels = Counter(ground_truth.values())
        print(f"  Sampled: {gt_labels['accept']}A + {gt_labels['reject']}R = {len(ground_truth)} total")

        # Build content.json for each sampled paper
        test_rows_no_model = []
        test_rows_with_model = []

        for r in sampled:
            sid = r["_metadata"]["submission_id"]

            # Build content.json and write to papers dir
            paper_dir = new_papers_dir / sid
            paper_dir.mkdir(exist_ok=True)
            content_json_path = paper_dir / "content.json"

            if not content_json_path.exists():
                idx = meta_index[sid]
                content_data = build_paper_content_json(ds[idx]["content_list_json"], sid)
                with open(content_json_path, "w") as f:
                    json.dump(content_data, f)

            # CSV rows
            test_rows_no_model.append({
                "submission_id": sid,
                "content_json": f"papers/{sid}/content.json",
            })

            wm_row = {
                "submission_id": sid,
                "content_json": f"papers/{sid}/content.json",
            }
            if sid in pred_lookup:
                wm_row["model_prediction"] = pred_lookup[sid]["model_prediction"]
                wm_row["model_confidence"] = pred_lookup[sid]["model_confidence"]
            else:
                # Fallback if no prediction available
                wm_row["model_prediction"] = ""
                wm_row["model_confidence"] = ""
            test_rows_with_model.append(wm_row)

        # Check SFT accuracy on this batch
        sft_correct = sum(1 for r in sampled
                          if r["_metadata"]["submission_id"] in pred_lookup
                          and pred_lookup[r["_metadata"]["submission_id"]]["model_prediction"]
                          == r["_metadata"]["answer"].lower())
        sft_coverage = sum(1 for r in sampled if r["_metadata"]["submission_id"] in pred_lookup)
        print(f"  SFT accuracy: {sft_correct}/{sft_coverage} = {sft_correct/max(sft_coverage,1):.1%}")

        # ── Create no-model directory ─────────────────────────────────
        no_model_dir = ROOT / config["no_model_dir"]
        setup_experiment_dir(
            no_model_dir, SRC_NO_MODEL,
            test_rows_no_model,
            ["submission_id", "content_json"],
            new_papers_dir,
        )
        print(f"  Created {no_model_dir.name}/ ({len(test_rows_no_model)} test papers)")

        # ── Create with-model directory ───────────────────────────────
        with_model_dir = ROOT / config["with_model_dir"]
        setup_experiment_dir(
            with_model_dir, SRC_WITH_MODEL,
            test_rows_with_model,
            ["submission_id", "content_json", "model_prediction", "model_confidence"],
            new_papers_dir,
        )
        print(f"  Created {with_model_dir.name}/ ({len(test_rows_with_model)} test papers)")

        # ── Write ground truth ────────────────────────────────────────
        gt_path = ROOT / "data" / config["gt_file"]
        with open(gt_path, "w") as f:
            json.dump(ground_truth, f, indent=2)
        print(f"  Wrote {gt_path.name}")

    # ── Also symlink the 2 train papers into new_papers_dir ───────────
    # (5DpzzTPnJZ and j9k3Oamba8 need to be accessible via papers/ symlink)
    for train_sid in ["5DpzzTPnJZ", "j9k3Oamba8"]:
        dst_train = new_papers_dir / train_sid
        src_train = PAPERS_SRC / train_sid
        if not dst_train.exists() and src_train.exists():
            os.symlink(src_train.resolve(), dst_train)

    # ── Update AGENTS.override.md for 2025 experiments ────────────────
    # Change "ICLR 2026" to "ICLR 2025" in the 2025 experiment dirs
    for config in BATCH_CONFIGS:
        if config["year"] == 2025:
            for dir_key in ["no_model_dir", "with_model_dir"]:
                agents_path = ROOT / config[dir_key] / "AGENTS.override.md"
                if agents_path.exists():
                    text = agents_path.read_text()
                    text = text.replace("ICLR 2026", "ICLR 2025")
                    agents_path.write_text(text)

    # ── Final verification ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Check no overlap between batches
    gt_files = [
        ("2026 batch 1 (existing)", set(json.load(open(ROOT / "data" / "agent_ground_truth_2026_unbiased_30_70.json")).keys())),
        ("2026 batch 2", set(json.load(open(ROOT / "data" / "agent_ground_truth_2026_batch2.json")).keys())),
        ("2025 batch 1", set(json.load(open(ROOT / "data" / "agent_ground_truth_2025_batch1.json")).keys())),
        ("2025 batch 2", set(json.load(open(ROOT / "data" / "agent_ground_truth_2025_batch2.json")).keys())),
    ]

    for i, (name_i, ids_i) in enumerate(gt_files):
        for j, (name_j, ids_j) in enumerate(gt_files):
            if i >= j:
                continue
            overlap = ids_i & ids_j
            status = "OK" if not overlap else f"OVERLAP: {len(overlap)}"
            print(f"  {name_i} vs {name_j}: {status}")

    # Directory listing
    print()
    for config in BATCH_CONFIGS:
        for dir_key in ["no_model_dir", "with_model_dir"]:
            d = ROOT / config[dir_key]
            test_csv = d / "TEST_SAMPLE.csv"
            n_test = sum(1 for _ in open(test_csv)) - 1 if test_csv.exists() else 0
            train_csv = d / "TRAIN_SAMPLE.csv"
            n_train = sum(1 for _ in open(train_csv)) - 1 if train_csv.exists() else 0
            print(f"  {d.name}: {n_test} test, {n_train} train")

    print("\nDone! All 6 experiment directories created.")


if __name__ == "__main__":
    main()
