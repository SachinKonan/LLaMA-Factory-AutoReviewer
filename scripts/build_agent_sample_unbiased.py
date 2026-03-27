"""Build unbiased agent benchmark samples (no distribution hints).

The prompt tells the agent the train set is balanced but says nothing about
the test distribution or conference acceptance rate. This lets the agent
reveal its natural accept/reject bias without post-hoc calibration.

Builds both no-model and with-model variants, plus run dirs with
AGENTS.override.md copied from existing runs.

Usage:
  python scripts/build_agent_sample_unbiased.py
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

# ── Paths ──────────────────────────────────────────────────────────────────
METADATA_DIR = Path("data/massive_metadata_v7_5")

VISION_TEST_SPLIT = Path(
    "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json"
)
TEXT_TEST_SPLIT = Path(
    "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json"
)
VISION_PREDICTIONS = Path(
    "results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl"
)

# Reuse existing train sample (same 400 papers)
EXISTING_BASE_DIR = Path("coding_agents_2026_codex_base")

# Existing run dirs (for copying AGENTS.override.md template)
EXISTING_RUN_NO_MODEL = Path("coding_agents_2026_codex_2")
EXISTING_RUN_WITH_MODEL = Path("coding_agents_2026_codex_with_best_model_and_model_prior_2")

# Output
OUT_BASE_NO_MODEL = Path("coding_agents_2026_codex_unbiased_base")
OUT_BASE_WITH_MODEL = Path("coding_agents_2026_codex_with_best_model_and_model_prior_unbiased_base")
OUT_RUN_NO_MODEL = Path("coding_agents_2026_codex_unbiased")
OUT_RUN_WITH_MODEL = Path("coding_agents_2026_codex_with_best_model_and_model_prior_unbiased")
OUT_GT = Path("data/agent_ground_truth_2026_unbiased.json")

# ── Constants ──────────────────────────────────────────────────────────────
SEED = 6
TEST_PER_LABEL = 100  # 100 accept + 100 reject = 200
SHUFFLE_SEED = 20260312

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

# ── Unbiased AGENTS.override.md ───────────────────────────────────────────
AGENTS_OVERRIDE_TEMPLATE = """# ICLR Paper Acceptance Prediction — You Are the Reviewer

## Your Role

You are an **expert ICLR reviewer**. Your job is to read each paper, evaluate its quality, and predict whether it was accepted or rejected. You have a labeled training set to calibrate your judgment.

## Data

- **`TRAIN_SAMPLE.json`** — 400 labeled ICLR papers (200 accept, 200 reject) from ICLR 2026. Use these to **calibrate** — study what accepted vs rejected papers look like.
- **`TEST_SAMPLE.json`** — {n_test} unlabeled ICLR papers you must classify by reading and reviewing them.{model_prior_note}

## Goal

Predict `"accept"` or `"reject"` for each of the {n_test} test papers. Output: **`PREDICTIONS.json`** in this directory.

```json
{{
  "submission_id_1": {{
    "decision": "accept",
    "why": "Novel architecture with strong empirical results across 5 benchmarks, thorough ablations, and clear writing."
  }},
  "submission_id_2": {{
    "decision": "reject",
    "why": "Incremental contribution over prior work, weak baselines, and missing comparisons to recent methods."
  }}
}}
```

Each entry **must** include:
- `"decision"`: exactly `"accept"` or `"reject"` (lowercase)
- `"why"`: a 1-3 sentence explanation of your reasoning, referencing specific aspects of the paper. This forces you to actually read the paper — generic or vague reasoning is not acceptable.

## Environment

Run Python scripts directly:

```bash
python my_script.py
```

Pre-installed: numpy, pandas, scikit-learn, scipy, matplotlib. **Do not install additional packages.**

## How to Approach This

### Step 1: Calibrate on the Training Set

First, write a script to extract readable text from a handful of accepted and rejected papers in `TRAIN_SAMPLE.json`. Read ~10 accepted and ~10 rejected papers carefully. Pay attention to:
- What distinguishes accepted papers? (novelty, strong experiments, clear writing, rigorous ablations)
- What distinguishes rejected papers? (incremental contribution, weak baselines, missing comparisons, poor writing, overclaiming)
- What patterns can you spot?

### Step 2: Review Every Test Paper — This Is the Core Task

**You must actually read and review each test paper.** This is not optional. Do NOT skip this step and jump to building a classifier — statistical models on 400 papers with simple features perform near random (~55%). Your LLM reasoning about paper quality is far more powerful.

For each paper in `TEST_SAMPLE.json`:
1. Extract readable text (write a helper script to dump paper text to a file)
2. Read the paper content: title, abstract, introduction, methodology, experiments, results
3. Think critically as an ICLR reviewer: Is the contribution novel? Are experiments convincing? Is the writing clear? Are claims supported?
4. Decide: accept or reject

**Use sub-agents to parallelize**: spawn a sub-agent per paper (or small batch). Each sub-agent reads one paper, writes a brief review, and returns its verdict.

### Step 3 (Optional): Cross-check with Simple Statistics

If you want, you can also build a simple feature-based model as a secondary signal to cross-reference against your reviews. But this should supplement your reviews, not replace them.

## Data Schema

Each paper entry in both files has:

- **`content_list`**: Array of elements from the paper PDF:
  - `type`: `"text"`, `"image"`, `"table"`, `"equation"`, `"list"`, or `"page_number"`
  - `text`: Text content (for text/equation/page_number)
  - `text_level`: Header level integer (1 = section, 2 = subsection) — only on headers
  - `text_format`: e.g., `"latex"` for equations
  - `page_idx`: 0-indexed page number
  - `bbox`: Bounding box `[x0, y0, x1, y1]`
  - `image_caption`, `image_footnote`: Lists of strings
  - `table_body`: HTML table string
  - `table_caption`, `table_footnote`: Lists of strings
  - `sub_type`, `list_items`: For list elements

- **`headers`**: Ordered list of section header strings

- **`label`**: `"accept"` or `"reject"` — **only in TRAIN_SAMPLE.json**
{model_prior_schema}
## Rules

1. **No internet access** — work only with the provided data
2. **Must predict all {n_test} test papers** — every submission_id from `TEST_SAMPLE.json`
3. **The training set is balanced** (50% accept, 50% reject), but the test set distribution is unknown — evaluate each paper independently on its own merits. Do not force any particular accept/reject ratio.
4. **Each entry must have `"decision"` and `"why"`** — the `"why"` must reference specific content from the paper (not generic boilerplate)
5. **Decisions must be exactly** `"accept"` or `"reject"` (lowercase)
6. **Output file**: `PREDICTIONS.json` in this directory
7. **You must read the papers** — do not rely solely on statistical models or heuristics

## Final Step: Create REVIEWER_SKILLS.md

After you have finished producing `PREDICTIONS.json`, create a file called **`REVIEWER_SKILLS.md`** in this directory. This file should be a complete, self-contained guide that describes how to perform inference on **any** new `TEST_SAMPLE.json` file (assuming the same format as the one you just processed). It should include:

1. **Overview** — what the task is and what the expected input/output looks like
2. **Step-by-step workflow** — the exact procedure you followed, written as repeatable instructions for a future agent
3. **Paper review heuristics** — the key signals you learned for distinguishing accepts from rejects (from both calibration and your reviews)
4. **Parallelization strategy** — how to batch and distribute paper reviews efficiently
5. **Common pitfalls** — mistakes to avoid (e.g., generic reasoning, missing edge cases, over-relying on simple heuristics)

Write it so that a future Codex agent given a new `TEST_SAMPLE.json` could read `REVIEWER_SKILLS.md` and immediately know how to produce high-quality predictions without re-discovering the workflow from scratch.
"""

MODEL_PRIOR_NOTE = """
- Each test paper also includes `model_prediction` (`"accept"` or `"reject"`) and `model_confidence` (0–1) from a fine-tuned vision model. Use these as an additional signal alongside your own review — they are informative but not infallible."""

MODEL_PRIOR_SCHEMA = """
- **`model_prediction`**: `"accept"` or `"reject"` — prediction from a fine-tuned SFT vision model (**only in TEST_SAMPLE.json**)
- **`model_confidence`**: float 0–1 — model's confidence in its prediction (**only in TEST_SAMPLE.json**)
"""


# ── Helpers ────────────────────────────────────────────────────────────────
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


def build_paper_data(sub_id, content_json):
    content_list_raw = json.loads(content_json) if isinstance(content_json, str) else content_json
    content_list = [c for elem in content_list_raw if (c := sanitize_element(elem)) is not None]
    headers = extract_headers(content_list)
    return {"content_list": content_list, "headers": headers}


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


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Building unbiased agent benchmark (seed={})".format(SEED))

    # Load text test split to get 2026 paper IDs and labels
    with open(TEXT_TEST_SPLIT) as f:
        text_rows = json.load(f)

    # Get existing train IDs to exclude
    with open(EXISTING_BASE_DIR / "TRAIN_SAMPLE.json") as f:
        train_sample = json.load(f)
    train_ids = set(train_sample.keys())

    # Pool: 2026 papers in test split, not in train
    accepts = [row for row in text_rows
               if row["_metadata"]["year"] == 2026
               and row["_metadata"]["answer"].lower() == "accept"
               and row["_metadata"]["submission_id"] not in train_ids]
    rejects = [row for row in text_rows
               if row["_metadata"]["year"] == 2026
               and row["_metadata"]["answer"].lower() == "reject"
               and row["_metadata"]["submission_id"] not in train_ids]

    print(f"Pool: {len(accepts)} accepts, {len(rejects)} rejects (excl {len(train_ids)} train)")

    # Sample
    rng = random.Random(SEED)
    sampled_accepts = rng.sample(accepts, TEST_PER_LABEL)
    sampled_rejects = rng.sample(rejects, TEST_PER_LABEL)
    test_ids = {}
    for row in sampled_accepts:
        test_ids[row["_metadata"]["submission_id"]] = "accept"
    for row in sampled_rejects:
        test_ids[row["_metadata"]["submission_id"]] = "reject"

    assert len(test_ids) == 2 * TEST_PER_LABEL
    assert not (set(test_ids) & train_ids)

    # Load vision predictions
    pred_lookup = load_vision_prediction_lookup()
    print(f"Vision pred_lookup: {len(pred_lookup)} papers")

    # Check SFT accuracy on this sample
    sft_correct = sum(1 for sid, label in test_ids.items()
                      if sid in pred_lookup and pred_lookup[sid]["model_prediction"] == label)
    print(f"SFT Vision accuracy on this sample: {sft_correct}/{len(test_ids)} = {sft_correct/len(test_ids):.1%}")

    # Load metadata for paper content
    print("Loading massive_metadata_v7_5...")
    ds = load_from_disk(str(METADATA_DIR))
    meta_index = {}
    for i in range(len(ds)):
        meta_index[ds[i]["submission_id"]] = i

    # Build test samples
    n_test = len(test_ids)

    # No-model variant
    test_no_model = {}
    # With-model variant
    test_with_model = {}
    ground_truth = {}

    for sid, label in test_ids.items():
        idx = meta_index[sid]
        entry = build_paper_data(sid, ds[idx]["content_list_json"])
        test_no_model[sid] = entry

        wm_entry = dict(entry)
        if sid in pred_lookup:
            wm_entry["model_prediction"] = pred_lookup[sid]["model_prediction"]
            wm_entry["model_confidence"] = pred_lookup[sid]["model_confidence"]
        test_with_model[sid] = wm_entry

        ground_truth[sid] = label

    # Shuffle
    shuffled_ids = list(test_ids.keys())
    random.Random(SHUFFLE_SEED).shuffle(shuffled_ids)
    test_no_model = {sid: test_no_model[sid] for sid in shuffled_ids}
    test_with_model = {sid: test_with_model[sid] for sid in shuffled_ids}
    ground_truth = {sid: ground_truth[sid] for sid in shuffled_ids}

    # Write base dirs
    for out_dir, test_data, with_model in [
        (OUT_BASE_NO_MODEL, test_no_model, False),
        (OUT_BASE_WITH_MODEL, test_with_model, True),
    ]:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy train sample
        with open(out_dir / "TRAIN_SAMPLE.json", "w") as f:
            json.dump(train_sample, f)

        with open(out_dir / "TEST_SAMPLE.json", "w") as f:
            json.dump(test_data, f)

        # Write AGENTS.override.md
        prompt = AGENTS_OVERRIDE_TEMPLATE.format(
            n_test=n_test,
            model_prior_note=MODEL_PRIOR_NOTE if with_model else "",
            model_prior_schema=MODEL_PRIOR_SCHEMA if with_model else "",
        )
        with open(out_dir / "AGENTS.override.md", "w") as f:
            f.write(prompt)

        variant = "with-model" if with_model else "no-model"
        print(f"\n[{variant}] Wrote {out_dir}/")

    # Write ground truth
    with open(OUT_GT, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Wrote {OUT_GT}")
    print(f"Labels: {dict(sorted(Counter(ground_truth.values()).items()))}")

    # Create run dirs (copy base + main.py from existing)
    import shutil
    for base_dir, run_dir, existing_run in [
        (OUT_BASE_NO_MODEL, OUT_RUN_NO_MODEL, EXISTING_RUN_NO_MODEL),
        (OUT_BASE_WITH_MODEL, OUT_RUN_WITH_MODEL, EXISTING_RUN_WITH_MODEL),
    ]:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        shutil.copytree(base_dir, run_dir)
        # Copy main.py from existing run if it exists
        main_py = existing_run / "main.py"
        if main_py.exists():
            shutil.copy2(main_py, run_dir / "main.py")
            print(f"Copied main.py to {run_dir}/")

    print(f"\nDone! Run dirs ready:")
    print(f"  {OUT_RUN_NO_MODEL}/")
    print(f"  {OUT_RUN_WITH_MODEL}/")


if __name__ == "__main__":
    main()
