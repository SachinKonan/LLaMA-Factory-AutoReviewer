# Gemini Synthetic Reasoning Generation

Generate structured AI reviews for ICLR papers using Gemini's batch prediction API.
These synthetic reviews are appended to paper text before fine-tuning, giving the model
structured reasoning (strengths, weaknesses, disputes) to condition on.

## What This Does

For each paper in a dataset, Gemini generates a structured review containing:

| Field | Description |
|---|---|
| `strengths` | Top 3 strengths (3-4 sentences each) |
| `weaknesses` | Top 3 weaknesses (3-4 sentences each) |
| `critical_disputes` | What reviewers would likely disagree about |
| `most_important_strength` | Single most compelling aspect |
| `most_important_weakness` | Single biggest concern |
| `relation_to_other_work` | How paper relates to prior work |
| `paper_quality` | "Paper gestalt" — gut feeling of quality, writing polish, figure clarity |

Output is guaranteed valid JSON via Gemini's native `responseSchema` — no parsing failures.

## The Script: `scripts/gemini_batch_review.py`

### Generation Modes

- **`blind`** — Gemini sees only the paper text/images. No human reviews, no decision.
- **`informed`** — Gemini sees paper + real human reviews (ratings stripped to prevent leakage).

### Execution Modes

- **Batch** (default) — Uploads requests to GCS, submits Vertex AI batch prediction job(s), polls for completion. **50% cost savings** vs online.
- **Online** (`--nobatch`) — Direct async API calls with rate limiting. Immediate results, full price.

### Key Arguments

```
Dataset:
  --dataset NAME        Dataset name (without _train/_test suffix)
  --split SPLIT [...]   Splits to process (default: train test)
  --data_dir DIR        Data directory (default: data)

Mode:
  --mode {blind,informed}   Generation mode (default: blind)
  --metadata_dir DIR        Arrow dataset with human reviews (for informed mode)

Model:
  --model NAME          Gemini model (default: gemini-2.5-pro)
  --temperature FLOAT   Sampling temperature (default: 1.0)
  --max_tokens INT      Max output tokens (default: 2048)
  --candidate_count INT Candidates per request (default: 1, use 3 for diversity)
  --thinking_budget INT Thinking tokens budget (default: 512, 0 to disable)

GCP:
  --project ID          GCP project ID
  --location REGION     GCP region (default: us-central1)
  --gcs_staging URI     GCS path for batch input/output files

Sharding:
  --num_shards N        Number of parallel batch jobs to submit (default: 1)

Resume:
  --partial_predictions FILE   Predictions JSONL from a cancelled/partial batch job
  --partial_metadata FILE      Matching metadata JSON (maps line index → submission_id)

Control:
  --submit_only         Submit batch job(s) and exit (don't poll/wait)
  --dry_run N           Test with N samples only
  --nobatch             Online mode instead of batch
  --qps INT             Max concurrent requests in online mode (default: 10)
  --analyze_diversity   Compute diversity metrics across candidates
```

### Output Format

JSONL where each line is:
```json
{
  "submission_id": "abc123",
  "year": 2023,
  "candidate_idx": 0,
  "_split": "train",
  "review": {
    "strengths": ["...", "...", "..."],
    "weaknesses": ["...", "...", "..."],
    "critical_disputes": "...",
    "most_important_strength": "...",
    "most_important_weakness": "...",
    "relation_to_other_work": "...",
    "paper_quality": "..."
  }
}
```

With `candidate_count=3`, each paper produces 3 lines (candidate_idx 0, 1, 2).

## Shell Scripts

### `blind_text_batch.sh` — Fresh Run

Submits the entire text dataset as batch jobs. Default 10 shards for parallelism.

```bash
# Run with 10 shards (default)
bash scripts/sh/gemini_synthetic_reasoning/blind_text_batch.sh

# Run with custom shard count
bash scripts/sh/gemini_synthetic_reasoning/blind_text_batch.sh 4

# Run in background with logging
nohup bash scripts/sh/gemini_synthetic_reasoning/blind_text_batch.sh \
    > logs/synthetic_gemini/blind_text_batch.log 2>&1 &
tail -f logs/synthetic_gemini/blind_text_batch.log
```

Config: gemini-2.5-flash, candidate_count=3, thinking_budget=256, temp=1.0, train+test splits.

### `blind_text_batch_resume.sh` — Resume from Partial Results

If a previous batch job was cancelled partway through, this picks up where it left off.
Loads the partial predictions file, identifies which papers already have valid responses,
and only submits the remaining papers.

```bash
# Resume with 10 shards (default)
bash scripts/sh/gemini_synthetic_reasoning/blind_text_batch_resume.sh

# Resume with custom shard count
bash scripts/sh/gemini_synthetic_reasoning/blind_text_batch_resume.sh 8

# Run in background
nohup bash scripts/sh/gemini_synthetic_reasoning/blind_text_batch_resume.sh \
    > logs/synthetic_gemini/blind_text_resume.log 2>&1 &
```

After all jobs complete, merge the shard outputs:
```bash
cat results/reviews/blind_text_resume_shard*.jsonl > results/reviews/blind_text_resume_merged.jsonl
```

## How Sharding Works

When `--num_shards N` is set (N > 1), the script:

1. Loads the full dataset once (avoids loading 900MB+ JSON N times)
2. Splits it into N equal chunks
3. Creates a separate batch JSONL for each chunk
4. Uploads each to GCS
5. Submits N independent Vertex AI batch prediction jobs

Each shard gets its own output files:
```
results/reviews/blind_text_shard0.jsonl          # predictions
results/reviews/blind_text_shard0_job_info.json  # job metadata
results/reviews/blind_text_shard0_metadata.json  # submission_id mapping
results/reviews/blind_text_shard1.jsonl
...
```

With `--submit_only`, the script exits after submitting all jobs. Without it,
the script polls each job sequentially until all complete.

Shards are completely independent — if one fails, the others continue.

**Example with 10 shards on 22,577 papers:**
```
Shard 0:  entries [0, 2257)      → 2,257 papers
Shard 1:  entries [2257, 4514)   → 2,257 papers
...
Shard 9:  entries [20313, 22577) → 2,264 papers
```

**Speedup**: A single job processes ~930 papers in 32 min. With 10 parallel jobs,
all 22.5K papers should complete in ~2-3 hours instead of ~13 hours.

## How Partial Predictions / Resume Works

The `--partial_predictions` flag accepts a raw predictions JSONL file downloaded from GCS
(the output of a Vertex AI batch prediction job). This file has one line per request,
in the same order as the original input:

```json
// Completed — has response with candidates
{"request": {...}, "status": "...", "response": {"candidates": [...]}, "processed_time": "..."}

// Incomplete — request only (job was cancelled before processing)
{"request": {...}}
```

The `--partial_metadata` JSON file maps each line index to a `submission_id`:
```json
[{"idx": 0, "submission_id": "abc123", "year": 2023, "split": "train"}, ...]
```

The script:
1. Iterates through both files in parallel (by line index)
2. For lines with a `response` containing `candidates`, parses the JSON text
   from at least one candidate to verify it's valid
3. Collects the `submission_id` for all verified completions
4. Filters the dataset to exclude those IDs before sharding and submitting

This is conservative — only verified responses are skipped.

## Cost Estimates (gemini-2.5-flash, batch pricing = 50% off)

Batch: $0.03/MTok input, $2.50/MTok output (thinking included in output).

| Scenario | Papers | Est. Cost |
|---|---|---|
| Blind text, 1 candidate | 23.6K | ~$90 |
| Blind text, 3 candidates | 23.6K | ~$170-200 |
| Informed text, 3 candidates | 23.6K | ~$200-230 |
| Blind vision, 3 candidates | 23.6K | ~$200-270 |

Online mode (`--nobatch`) is roughly 2x these costs but results are immediate.

## Examples

```bash
# Fresh run, 10 shards, submit and exit
uv run python scripts/gemini_batch_review.py \
    --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered \
    --mode blind --model gemini-2.5-flash --candidate_count 3 --thinking_budget 256 \
    --num_shards 10 --submit_only \
    --output results/reviews/blind_text.jsonl

# Resume from partial results, 10 shards
uv run python scripts/gemini_batch_review.py \
    --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered \
    --mode blind --model gemini-2.5-flash --candidate_count 3 --thinking_budget 256 \
    --partial_predictions data/gemini_text-predictions.jsonl \
    --partial_metadata results/reviews/blind_text_metadata.json \
    --num_shards 10 --submit_only \
    --output results/reviews/blind_text_resume.jsonl

# Dry run — test 5 papers, online mode
uv run python scripts/gemini_batch_review.py \
    --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered \
    --dry_run 5 --mode blind --model gemini-2.5-flash \
    --candidate_count 3 --thinking_budget 256 --nobatch

# Informed mode with human reviews
uv run python scripts/gemini_batch_review.py \
    --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered \
    --mode informed --metadata_dir data/massive_metadata_v7_5 \
    --model gemini-2.5-flash --candidate_count 3 --thinking_budget 256

# Vision dataset
uv run python scripts/gemini_batch_review.py \
    --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480 \
    --mode blind --model gemini-2.5-flash --candidate_count 3 --thinking_budget 256

# Merge shards after completion
cat results/reviews/blind_text_shard*.jsonl > results/reviews/blind_text_merged.jsonl
```
