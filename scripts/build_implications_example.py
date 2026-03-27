#!/usr/bin/env python3
"""Find and format a side-by-side review example for the Discussion section.

Selects a paper where:
  - with-model got the decision correct, no-model got it wrong
  - Human reviews are available
  - The alignment difference is clearly visible

Outputs formatted excerpts for the LaTeX figure.
"""

import json
import os
import sys
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parent.parent


def get_dec(v):
    if isinstance(v, dict):
        return v.get("decision", "").strip().lower()
    return str(v).strip().lower()


def main():
    # Configs: (batch, no_model_dir, with_model_dir, gt_path)
    configs = [
        ("2026 b1", "coding_agents_2026_claudecode_v3.3", "coding_agents_2026_claudecode_v3.3b",
         "data/agent_ground_truth_2026_unbiased_30_70.json"),
        ("2026 b2", "coding_agents_2026_claudecode_v3.3.1", "coding_agents_2026_claudecode_v3.3b.1",
         "data/agent_ground_truth_2026_batch2.json"),
        ("2025 b1", "coding_agents_2025_claudecode_v3.3_2025", "coding_agents_2025_claudecode_v3.3b_2025",
         "data/agent_ground_truth_2025_batch1.json"),
        ("2025 b2", "coding_agents_2025_claudecode_v3.3_2025.1", "coding_agents_2025_claudecode_v3.3b_2025.1",
         "data/agent_ground_truth_2025_batch2.json"),
    ]

    # Find candidates
    candidates = []
    for batch, no_dir, wm_dir, gt_path in configs:
        gt = json.load(open(ROOT / gt_path))
        preds_no = json.load(open(ROOT / no_dir / "PREDICTIONS.json"))
        preds_wm = json.load(open(ROOT / wm_dir / "PREDICTIONS.json"))

        for sid in gt:
            if sid not in preds_no or sid not in preds_wm:
                continue
            g = gt[sid]
            p_no = get_dec(preds_no[sid])
            p_wm = get_dec(preds_wm[sid])

            if p_wm == g and p_no != g:
                no_review_path = ROOT / no_dir / "agent_reviews" / f"{sid}.json"
                wm_review_path = ROOT / wm_dir / "agent_reviews" / f"{sid}.json"
                if no_review_path.exists() and wm_review_path.exists():
                    candidates.append({
                        "sid": sid, "batch": batch, "gt": g,
                        "no_dec": p_no, "wm_dec": p_wm,
                        "no_dir": no_dir, "wm_dir": wm_dir,
                    })

    print(f"Found {len(candidates)} candidates")

    # Load metadata for human reviews
    ds = load_from_disk(str(ROOT / "data" / "massive_metadata_v7_5"))
    meta = {}
    for i in range(len(ds)):
        meta[ds[i]["submission_id"]] = i

    # Score candidates by review richness
    for c in candidates:
        sid = c["sid"]
        if sid not in meta:
            c["score"] = -1
            continue

        idx = meta[sid]
        reviews_raw = ds[idx].get("original_reviews")
        reviews = json.loads(reviews_raw) if isinstance(reviews_raw, str) else reviews_raw
        if not reviews:
            c["score"] = -1
            continue

        # Load agent reviews
        no_rev = json.load(open(ROOT / c["no_dir"] / "agent_reviews" / f"{sid}.json"))
        wm_rev = json.load(open(ROOT / c["wm_dir"] / "agent_reviews" / f"{sid}.json"))

        # Score: prefer papers with clear weakness text and rating differences
        no_weak = str(no_rev.get("weaknesses", ""))
        wm_weak = str(wm_rev.get("weaknesses", ""))
        human_weak = str(reviews[0].get("weaknesses", "")) if reviews else ""

        c["score"] = len(wm_weak) - len(no_weak) + len(human_weak) // 10
        c["human_reviews"] = reviews
        c["no_rev"] = no_rev
        c["wm_rev"] = wm_rev

    # Sort by score
    candidates.sort(key=lambda x: x.get("score", -1), reverse=True)

    # Print top 5
    print("\nTop candidates:")
    for c in candidates[:5]:
        if c.get("score", -1) < 0:
            continue
        print(f"\n{'='*60}")
        print(f"Paper: {c['sid']} | Batch: {c['batch']} | GT: {c['gt']}")
        print(f"No-model: {c['no_dec']} (rating {c['no_rev'].get('rating', '?')})")
        print(f"With-model: {c['wm_dec']} (rating {c['wm_rev'].get('rating', '?')})")

        # Human mean rating
        ratings = [r.get("rating") for r in c["human_reviews"] if r.get("rating")]
        if ratings:
            print(f"Human mean rating: {sum(ratings)/len(ratings):.1f} ({len(ratings)} reviewers)")

        print(f"\nNo-model top weakness:")
        no_weak = c["no_rev"].get("weaknesses", "")
        if isinstance(no_weak, list):
            print(f"  {no_weak[0][:200]}")
        else:
            print(f"  {str(no_weak)[:200]}")

        print(f"\nWith-model top weakness:")
        wm_weak = c["wm_rev"].get("weaknesses", "")
        if isinstance(wm_weak, list):
            print(f"  {wm_weak[0][:200]}")
        else:
            print(f"  {str(wm_weak)[:200]}")

        print(f"\nHuman top weakness:")
        h_weak = c["human_reviews"][0].get("weaknesses", "")
        print(f"  {str(h_weak)[:200]}")


if __name__ == "__main__":
    main()
