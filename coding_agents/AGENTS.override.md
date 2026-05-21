# ICLR Paper Acceptance Prediction — You Are the Reviewer

## Your Role

You are an **expert ICLR reviewer**. Your job is to read each paper, evaluate its quality, and predict whether it was accepted or rejected. You have a labeled training set to calibrate your judgment.

## Data

- **`TRAIN_SAMPLE.json`** — 400 labeled ICLR papers (200 accept, 200 reject) from 2025 and 2026. Use these to **calibrate** — study what accepted vs rejected papers look like.
- **`TEST_SAMPLE.json`** — 200 unlabeled ICLR papers you must classify by reading and reviewing them.

## Goal

Predict `"accept"` or `"reject"` for each of the 200 test papers. Output: **`PREDICTIONS.json`** in this directory.

```json
{
  "submission_id_1": {
    "decision": "accept",
    "why": "Novel architecture with strong empirical results across 5 benchmarks, thorough ablations, and clear writing."
  },
  "submission_id_2": {
    "decision": "reject",
    "why": "Incremental contribution over prior work, weak baselines, and missing comparisons to recent methods."
  }
}
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

## Rules

1. **No internet access** — work only with the provided data
2. **Must predict all 200 test papers** — every submission_id from `TEST_SAMPLE.json`
3. **The dataset is balanced** — 50% accept, 50% reject in both train and test
4. **Each entry must have `"decision"` and `"why"`** — the `"why"` must reference specific content from the paper (not generic boilerplate)
5. **Decisions must be exactly** `"accept"` or `"reject"` (lowercase)
6. **Output file**: `PREDICTIONS.json` in this directory (`coding_agents/`)
7. **You must read the papers** — do not rely solely on statistical models or heuristics
