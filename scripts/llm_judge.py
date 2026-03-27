#!/usr/bin/env python3
"""Generic LLM-as-judge for evaluating agent reviews against human reviews.

Prepares batched input, runs Codex on each batch, merges scores.

Usage:
    python scripts/llm_judge.py \
        --predictions path/to/PREDICTIONS.json \
        --metadata data/massive_metadata_v7_5 \
        --ground-truth data/agent_ground_truth_2026_unbiased.json \
        --output-dir path/to/judge_output \
        --batch-size 10 \
        --max-papers 100

Or with pre-built judge input:
    python scripts/llm_judge.py \
        --judge-input path/to/judge_input.json \
        --output-dir path/to/judge_output

Phases:
    prepare  - Build judge_input.json from predictions + metadata (default: runs all)
    judge    - Run Codex on batches
    merge    - Merge batch results into final scores
    summary  - Print summary statistics
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CODEX_BIN = "/scratch/gpfs/ZHUANGL/sk7524/.cache/.nvm/versions/node/v24.14.0/bin/codex"
NODE_DIR = "/scratch/gpfs/ZHUANGL/sk7524/.cache/.nvm/versions/node/v24.14.0/bin"

REVIEW_FIELDS = [
    "rating", "confidence", "soundness", "presentation", "contribution",
    "summary", "strengths", "weaknesses", "questions",
]

JUDGE_RUBRIC = """Score the AI-generated review against the human reviews on these 5 dimensions (each 1-5):

1. contribution_identification: Did the agent identify the same primary contribution as human reviewers?
   (1=completely missed, 3=partially identified, 5=perfect match)
2. top_weakness_identification: Did the agent identify the most important weakness flagged by humans?
   (1=missed all key weaknesses, 3=identified some, 5=identified the top weakness precisely)
3. novel_concerns: Did the agent raise valid concerns NOT mentioned in human reviews?
   (1=no novel valid points, 3=some valid novel points, 5=substantial novel insights)
4. rating_alignment: How close is the agent's rating to the human reviewer consensus mean?
   (1=off by 3+, 2=off by 2, 3=off by 1.5, 4=off by 1, 5=off by <0.5)
5. overall_review_quality: Overall quality of the agent review compared to human reviews.
   (1=poor/generic, 2=below average, 3=adequate, 4=good, 5=comparable to human quality)"""


def load_human_reviews(metadata_path: str, submission_ids: set) -> dict:
    """Load human reviews from massive_metadata dataset."""
    try:
        from datasets import load_from_disk
        ds = load_from_disk(metadata_path)
    except Exception as e:
        print(f"ERROR: Could not load metadata: {e}")
        sys.exit(1)

    reviews = {}
    for row in ds:
        sid = row.get("submission_id")
        if sid not in submission_ids:
            continue
        raw = row.get("original_reviews")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(raw, list):
            continue
        filtered = []
        for rev in raw:
            if not isinstance(rev, dict):
                continue
            entry = {k: rev[k] for k in REVIEW_FIELDS if k in rev}
            if entry:
                filtered.append(entry)
        if filtered:
            reviews[sid] = filtered
    return reviews


def prepare(args):
    """Build judge_input.json from predictions + metadata."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = json.load(open(args.predictions))
    gt = json.load(open(args.ground_truth)) if args.ground_truth else {}

    # Optionally limit papers
    sids = sorted(preds.keys())
    if args.max_papers and args.max_papers < len(sids):
        sids = sids[:args.max_papers]

    print(f"Loading human reviews for {len(sids)} papers ...")
    human_reviews = load_human_reviews(args.metadata, set(sids))
    print(f"Human reviews found: {len(human_reviews)}")

    judge_input = {}
    for sid in sids:
        if sid not in human_reviews:
            continue
        judge_input[sid] = {
            "ground_truth": gt.get(sid, "unknown"),
            "human_reviews": human_reviews[sid],
            "agent_review": preds[sid],
        }

    out_path = output_dir / "judge_input.json"
    with open(out_path, "w") as f:
        json.dump(judge_input, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(judge_input)} entries to {out_path}")
    return judge_input


def judge(args):
    """Run Codex on batches."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load judge input
    judge_input_path = output_dir / "judge_input.json"
    if args.judge_input:
        judge_input_path = Path(args.judge_input)
    judge_input = json.load(open(judge_input_path))

    batch_size = args.batch_size
    sids = sorted(judge_input.keys())
    batches = [sids[i:i + batch_size] for i in range(0, len(sids), batch_size)]

    print(f"Judging {len(sids)} papers in {len(batches)} batches of {batch_size}")

    scores_dir = output_dir / "batch_scores"
    scores_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{NODE_DIR}:{env.get('PATH', '')}"

    for batch_idx, batch_sids in enumerate(batches):
        score_file = scores_dir / f"batch_{batch_idx}.json"
        if score_file.exists():
            existing = json.load(open(score_file))
            if len(existing) == len(batch_sids):
                print(f"Batch {batch_idx}/{len(batches)-1}: already complete ({len(existing)} scores)")
                continue

        # Build batch data
        batch_data = {sid: judge_input[sid] for sid in batch_sids}
        batch_input_file = scores_dir / f"batch_{batch_idx}_input.json"
        with open(batch_input_file, "w") as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)

        prompt = f"""You are an LLM judge evaluating AI-generated paper reviews against real human ICLR reviews.

Read the file {batch_input_file}. It contains {len(batch_sids)} papers, each with:
- "ground_truth": the real accept/reject decision
- "human_reviews": list of real ICLR reviewer assessments
- "agent_review": the AI-generated review to evaluate

For EACH paper, score the agent_review against the human_reviews:

{JUDGE_RUBRIC}

For rating_alignment, compute the mean of all human reviewers' ratings first, then compare to the agent's rating.

Write your scores to {score_file} as a JSON dict keyed by submission_id:
{{
  "SUBMISSION_ID": {{
    "contribution_identification": N,
    "top_weakness_identification": N,
    "novel_concerns": N,
    "rating_alignment": N,
    "overall_review_quality": N
  }},
  ...
}}

Score ALL {len(batch_sids)} papers. Do not skip any."""

        print(f"Batch {batch_idx}/{len(batches)-1}: judging {len(batch_sids)} papers ...")

        try:
            result = subprocess.run(
                [CODEX_BIN, "exec", "--sandbox", "workspace-write",
                 "-C", str(output_dir), prompt],
                env=env,
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                print(f"  WARNING: Codex returned {result.returncode}")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            print(f"  WARNING: Batch {batch_idx} timed out (600s)")
            continue

        # Verify output
        if score_file.exists():
            scores = json.load(open(score_file))
            print(f"  Scored {len(scores)}/{len(batch_sids)} papers")
        else:
            print(f"  WARNING: No output file created for batch {batch_idx}")


def merge(args):
    """Merge batch results into final scores."""
    output_dir = Path(args.output_dir)
    scores_dir = output_dir / "batch_scores"

    merged = {}
    for f in sorted(scores_dir.glob("batch_*.json")):
        if "_input" in f.name:
            continue
        try:
            batch = json.load(open(f))
            merged.update(batch)
            print(f"  {f.name}: {len(batch)} scores")
        except Exception as e:
            print(f"  {f.name}: ERROR {e}")

    out_path = output_dir / "judge_scores_merged.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged {len(merged)} scores to {out_path}")
    return merged


def summary(args):
    """Print summary statistics."""
    output_dir = Path(args.output_dir)
    scores_path = output_dir / "judge_scores_merged.json"
    if not scores_path.exists():
        print("No merged scores found. Run merge first.")
        return

    scores = json.load(open(scores_path))
    dims = ["contribution_identification", "top_weakness_identification",
            "novel_concerns", "rating_alignment", "overall_review_quality"]

    print(f"\n{'='*55}")
    print(f"  LLM Judge Summary ({len(scores)} papers)")
    print(f"{'='*55}")
    print(f"{'Dimension':<35} {'Mean':>6} {'Min':>5} {'Max':>5}")
    print("-" * 55)

    overall_sum = 0
    for dim in dims:
        vals = [v[dim] for v in scores.values() if dim in v]
        if vals:
            import statistics
            mean = statistics.mean(vals)
            overall_sum += mean
            print(f"{dim:<35} {mean:>6.2f} {min(vals):>5} {max(vals):>5}")

    print("-" * 55)
    print(f"{'OVERALL MEAN':<35} {overall_sum/len(dims):>6.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generic LLM-as-judge for review alignment")
    parser.add_argument("--predictions", help="Path to PREDICTIONS.json or PREDICTIONS_genuine.json")
    parser.add_argument("--metadata", default="data/massive_metadata_v7_5", help="Path to metadata dataset")
    parser.add_argument("--ground-truth", help="Path to ground truth JSON")
    parser.add_argument("--judge-input", help="Path to pre-built judge_input.json (skip prepare)")
    parser.add_argument("--output-dir", required=True, help="Output directory for judge results")
    parser.add_argument("--batch-size", type=int, default=10, help="Papers per Codex batch")
    parser.add_argument("--max-papers", type=int, default=None, help="Limit number of papers to judge")
    parser.add_argument("--phase", choices=["prepare", "judge", "merge", "summary", "all"],
                        default="all", help="Which phase to run")
    args = parser.parse_args()

    if args.phase in ("prepare", "all"):
        if not args.predictions:
            parser.error("--predictions required for prepare phase")
        prepare(args)

    if args.phase in ("judge", "all"):
        judge(args)

    if args.phase in ("merge", "all"):
        merge(args)

    if args.phase in ("summary", "all"):
        summary(args)


if __name__ == "__main__":
    main()