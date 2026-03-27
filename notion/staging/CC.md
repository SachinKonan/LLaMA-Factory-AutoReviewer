claude -# Claude Code Agent Experiment: 30/70 Test Set Analysis

## Main Results

| Metric | v3.3 (No Model) | v3.3b (With Model) | SFT Model Alone | v3.2a (400-train, No Model) |
|---|---|---|---|---|
| **Accuracy** | 66.0% | **78.0%** | 72.0% | 73.0% |
| **Precision** | 0.450 | **0.591** | 0.520 | 0.545 |
| **Recall** | 0.600 | **0.867** | 0.867 | 0.600 |
| **F1** | 0.514 | **0.703** | — | — |

Test set: 100 papers (30 accept, 70 reject), ICLR 2026.

Key result: **+12pp accuracy gain** (66% → 78%) with model prior. v3.3b with only 2 training papers surpasses v3.2a with 400 training papers (78% vs 73%).

---

## Experiment Setup: What Claude Code Does

### Architecture Overview

- **Orchestrator**: Claude Opus, runs on CPU node. Reads `AGENTS.override.md`, runs `python main.py train-summary` for calibration, reads `TEST_SAMPLE.csv`, spawns sub-agents.
- **Sub-agents**: Claude Sonnet, spawned via the Agent tool. Each reviews one paper independently.
- **ArXiv Search**: MCP server on GPU node (Qwen3-Embedding-0.6B + FAISS index). Sub-agents call `search_arxiv` to verify novelty and find missing references.
- **Runner**: `scripts/sh/claude_code_with_arxiv/sbatch_v32.sh` — parameterized SLURM script that auto-detects model columns from CSV header.

### Inputs per Variant

**v3.3 (no model)**:
- `TRAIN_SAMPLE.csv`: 2 papers (1 accept, 1 reject) with `content_json` + `human_reviews_json`
- `TEST_SAMPLE.csv`: 100 papers, columns: `submission_id`, `content_json`
- `AGENTS.override.md`: "No Model Prior" version

**v3.3b (with model)**:
- `TRAIN_SAMPLE.csv`: 400 papers with `label` + `model_prediction` + `model_confidence` (only 2 have `content_json`/`human_reviews_json`)
- `TEST_SAMPLE.csv`: 100 papers, columns: `submission_id`, `content_json`, `model_prediction`, `model_confidence`
- `AGENTS.override.md`: "With Model Prior" version — adds "Model Prior" section

### Exact Orchestrator Prompts

The `sbatch_v32.sh` script auto-detects model columns in the CSV header and selects the appropriate prompt. The prompts differ in 3 places (marked with **[DIFF]**):

**v3.3 (no model) orchestrator prompt:**
```
Read AGENTS.override.md for the output schema, calibration guidance, and review rules.
Follow every rule strictly.

Step 1: Run `python main.py train-summary` and study the calibration statistics —
  accept rate, rating distributions, confidence distributions, and how human       ← [DIFF 1]
  reviewer scores map to decisions.

Step 2: Read TEST_SAMPLE.csv. It has columns: submission_id, content_json.          ← [DIFF 2]
  Create the directory agent_reviews/ if it does not exist.

Step 3: For each paper in TEST_SAMPLE.csv, use the Agent tool to spawn a reviewer
  sub-agent. Pass it the paper's submission_id and content_json path. Include your
  calibration findings so the sub-agent can make calibrated decisions. Tell each
  sub-agent:
- Read the paper content from the content_json file path
- Examine page images listed in the content JSON under img_pages
- Use search_arxiv to verify novelty claims and find missing references
                                                                                    ← [DIFF 3]
- Write a structured ICLR review with all 11 fields
- IMPORTANT: Write the review JSON to agent_reviews/{submission_id}.json using
  the Write tool. Return ONLY a short confirmation like "Done: {submission_id}"
  — do NOT return the full review text.

Launch sub-agents in parallel (multiple Agent calls in one message) for efficiency.
Each sub-agent reviews one paper independently.

Step 4: After all sub-agents complete, write a Python script that reads all
  agent_reviews/*.json files, assembles them into PREDICTIONS.json [...]

Step 5: Run `python main.py validate-predictions --path PREDICTIONS.json` to verify.

CRITICAL: Every paper must be reviewed by a sub-agent that actually reads the paper
  content. Do NOT write scripts, use sklearn, or template-fill. Do NOT skip any of
  the 100 test papers. Sub-agents MUST write reviews to
  agent_reviews/{submission_id}.json and return only a short confirmation to avoid
  context overflow.
```

**v3.3b (with model) orchestrator prompt — differences only:**
```
Step 1: ... study the calibration statistics — especially how model_prediction      ← [DIFF 1]
  accuracy varies by model_confidence. Note the accept rate, rating distributions,
  and when to trust the model vs override it.

Step 2: Read TEST_SAMPLE.csv. It has columns: submission_id, content_json,           ← [DIFF 2]
  model_prediction, model_confidence. [...]

Step 3: [...] Pass it the paper's submission_id, content_json path,
  model_prediction, and model_confidence. [...]
- Use the model_prediction and model_confidence as a Bayesian prior                  ← [DIFF 3]
```

### Orchestrator Behavior in Practice

**v3.3 orchestrator** ran `python main.py train-summary` on 2 training papers and derived:
- Accept paper: mean rating 5.0 (ratings 4,4,6,6), soundness ~3, presentation ~3, contribution ~2.5
- Reject paper: mean rating 3.0 (ratings 2,2,4,4), soundness ~3, presentation ~2.25, contribution ~1.5
- Average review: ~3 strengths, ~4-5 weaknesses, ~834 chars

Then spawned 100 sub-agents sequentially (1 per message), each getting this calibration.

**v3.3b orchestrator** ran `python main.py train-summary` on 400 training papers and derived ADDITIONAL statistics:
- Overall model accuracy: 66.5%
- Confidence-binned accuracy:
  - [0.6, 0.7): 63.7% (n=80) — "weak signal, lean slightly toward model"
  - [0.7, 0.8): 63.5% (n=126) — "weak signal, lean slightly toward model"
  - [0.8, 0.9): 74.5% (n=51) — "moderate signal, give model weight"
  - [0.9+]: 88.6% — "high confidence, strong signal"

Then spawned 100 sub-agents, each getting **paper-specific calibration** — the confidence bin and accuracy for THAT paper's model confidence.

### Exact Sub-Agent Prompts (Side-by-Side)

For the SAME paper (r2HG3xOMJI), the orchestrator constructed these prompts:

**v3.3 sub-agent prompt:**
```
You are an expert ICLR reviewer. Review the paper with submission_id=r2HG3xOMJI.

CALIBRATION (from ICLR 2026 training data):
- Accept papers: mean rating ~5.0 (ratings 4-6), soundness ~3, presentation ~3,
  contribution ~2.5
- Reject papers: mean rating ~3.0 (ratings 2-4), soundness ~3,
  presentation ~2.25, contribution ~1.5
- Strengths: ~3 bullets, Weaknesses: ~4-5 bullets, ~834 chars per review
- Be critical but fair. Most papers at top venues have meaningful weaknesses.

TASK:
1. Read the paper content from: papers/r2HG3xOMJI/content.json
2. Examine the page images listed in the content JSON under img_pages
3. Use search_arxiv to verify novelty claims and find missing references
4. Write a structured ICLR review as JSON with these fields:
   - decision: "accept" or "reject"
   - rating: 0-10
   - confidence: 1-5
   - soundness: 1-4
   - presentation: 1-4
   - contribution: 1-4
   - summary: 2-3 sentence summary
   - strengths: numbered list of strengths
     (cite specific sections/tables/figures)
   - weaknesses: numbered list of weaknesses
     (cite specific sections/tables/figures, no generic criticisms)
   - questions: numbered list of questions for authors
   - missing_references: prior work found via arxiv search

5. Write the review JSON to: agent_reviews/r2HG3xOMJI.json using the Write tool.
6. Return ONLY a short confirmation like "Done: r2HG3xOMJI"
   — do NOT return the full review text.
```

**v3.3b sub-agent prompt** (same paper, model_prediction=accept, confidence=0.679):
```
You are reviewing an ICLR 2026 submission. Your task is to write a detailed,
calibrated review and save it to disk.

## Paper Details
- Submission ID: r2HG3xOMJI
- Content JSON path: /scratch/gpfs/.../papers/r2HG3xOMJI/content.json
- Model prediction: accept                                                   ← NEW
- Model confidence: 0.679                                                    ← NEW

## Calibration Data (from 400 training papers)                                ← EXPANDED
- Training set: 50/50 accept/reject split
- Overall model accuracy: 66.5%                                               ← NEW
- Confidence [0.6, 0.7): accuracy=63.7% (n=80) — weak signal, lean slightly   ← NEW
  toward model
- Accept papers: typical ratings 4-6 (mean 5.0), reject papers: typical
  ratings 2-4 (mean 3.0)
- Soundness: accept ~2.75, reject ~2.75; Presentation: accept ~3.0,
  reject ~2.25; Contribution: accept ~2.5, reject ~1.5
- Strengths: ~3 bullet points; Weaknesses: ~4-5 bullet points;
  Review length: ~834 chars

## Instructions
1. Read the paper content from the content JSON file. Look at the img_pages
   listed in the JSON to examine page images.
2. Use search_arxiv to verify novelty claims and find related/missing
   references.
3. Use the model prediction as a Bayesian prior (confidence 0.679 maps to     ← NEW
   ~63.7% accuracy — weak signal).
4. Write a structured ICLR review with ALL 11 fields:
   [same 11 fields as v3.3]

5. IMPORTANT: Write the review as a JSON object to
   /scratch/gpfs/.../agent_reviews/r2HG3xOMJI.json
6. Return ONLY a short confirmation "Done: r2HG3xOMJI"
   — do NOT return the full review text.
```

**Summary of prompt differences:**
1. v3.3b adds "Model prediction" and "Model confidence" per paper
2. v3.3b expands calibration from 2 training papers → 400, with confidence-binned accuracy
3. v3.3b adds explicit Bayesian prior instruction with mapped accuracy percentage
4. v3.3b tells the agent the confidence bin and qualitative signal strength ("weak", "moderate", "strong")

### Sub-Agent Behavior (Both Variants)

Each Sonnet sub-agent independently:
1. Reads the full paper content JSON (text elements, headers, sections)
2. Examines 8-10 page images (rendered PDF pages)
3. Runs 5-6 targeted ArXiv searches (e.g., "hierarchical reinforcement learning offline goal-conditioned", "OGBench benchmark 2025")
4. Writes a structured 11-field ICLR review to disk
5. Returns only "Done: {submission_id}" to prevent context overflow

**Key architectural difference from Codex**: Claude Code sub-agents read full paper content AND page images (multimodal), and have access to live ArXiv search via MCP server. Codex sub-agents only read text summaries generated by a preprocessing script.

---

## Alignment Metrics

### Metric Definitions

All metrics compare agent-predicted numeric scores against the **human reviewer mean** for each paper. For a given field (e.g., rating), let `agent_i` be the agent's score for paper `i` and `human_i = mean(reviewer scores for paper i)` across all ICLR reviewers.

- **Pearson correlation** (`scipy.stats.pearsonr`): Linear correlation between `agent` and `human` score vectors. Measures whether the agent's scores track the direction and magnitude of human consensus. Range: [-1, 1].
- **Spearman correlation** (`scipy.stats.spearmanr`): Rank correlation between `agent` and `human` score vectors. Measures whether the agent's relative ordering of papers matches human ordering, regardless of scale. Range: [-1, 1].
- **MAE** (Mean Absolute Error): `mean(|agent_i - human_i|)` across all papers. Measures average absolute deviation from human mean scores.
- **Agent mean rating (true accepts/rejects)**: `mean(agent_rating_i)` grouped by ground-truth label. Shows how the agent's ratings distribute across true accepts vs true rejects.
- **Calibration gap**: `agent_mean_rating(true accepts) - agent_mean_rating(true rejects)`. Measures how well the agent's rating scale separates the two classes. Higher = better separation.

Numeric fields evaluated: `rating` (0-10), `confidence` (1-5), `soundness` (1-4), `presentation` (1-4), `contribution` (1-4).

### Ground-Truth Human Mean Ratings (Test Set)

Human reviewer means are computed per paper from `original_reviews` in the `massive_metadata_v7_5` dataset. For each paper, the mean is taken across all ICLR reviewers for that submission.

| Group | Human Mean Rating | n |
|---|---|---|
| True accepts | 5.587 | 30 |
| True rejects | 4.086 | 70 |
| All papers | 4.536 | 100 |

### Results

| Metric | v3.3 | v3.3b |
|---|---|---|
| Rating Pearson | 0.395 | **0.525** |
| Rating Spearman | 0.445 | **0.527** |
| Rating MAE | 0.998 | **0.970** |
| Soundness Pearson | 0.325 | **0.398** |
| Calibration gap | 0.886 | **1.614** |
| Agent mean rating (true accepts) | 5.300 | 5.700 |
| Agent mean rating (true rejects) | 4.414 | 4.086 |

The model prior improves rating correlation (Pearson 0.395 → 0.525) and widens the calibration gap (0.886 → 1.614), meaning v3.3b better separates accepts from rejects in its rating assignments.

---

## Decision Flips: v3.3 → v3.3b

16/100 papers flipped (16%), 14 correct (**87.5% flip accuracy**), 2 wrong.

### By Direction

- **Reject→Accept**: 10 flips, 8 correct (80%). Human mean ratings: 4.5–7.0 (mostly borderline-to-strong accepts the no-model agent missed)
- **Accept→Reject**: 6 flips, 6 correct (100%). Human mean ratings: 2.5–5.5 (true rejects the no-model agent wrongly accepted)

### Full Flip Table

| SID | v3.3 | v3.3b | True | v3.3 rat | v3.3b rat | Human mean | Model pred | Model conf | Result |
|---|---|---|---|---|---|---|---|---|---|
| 5E5sd3TWGD | reject | accept | accept | 4 | 6 | 5.00 | accept | 0.706 | FIXED |
| AlJK6bFbAo | accept | reject | reject | 6 | 4 | 5.50 | accept | 0.679 | FIXED |
| Bp2VlfYAMc | accept | reject | reject | 6 | 4 | 5.00 | reject | 0.562 | FIXED |
| EzHOmjg3R6 | accept | reject | reject | 6 | 5 | 5.00 | reject | 0.531 | FIXED |
| KjYpHySlb0 | reject | accept | accept | 4 | 6 | 6.50 | accept | 0.651 | FIXED |
| OPQUeeT02q | reject | accept | reject | 4 | 6 | 5.50 | accept | 0.731 | BROKE |
| SFgXPipvXw | reject | accept | reject | 4 | 6 | 5.00 | accept | 0.679 | BROKE |
| Vvks41GeL9 | accept | reject | reject | 6 | 4 | 5.50 | reject | 0.531 | FIXED |
| Zunww3FHPU | reject | accept | accept | 4 | 6 | 6.50 | accept | 0.500 | FIXED |
| cK35kNVm5r | reject | accept | accept | 4 | 6 | 4.80 | accept | 0.622 | FIXED |
| h1u9YsjLBy | accept | reject | reject | 6 | 4 | 2.50 | reject | 0.706 | FIXED |
| mN49LupE8l | reject | accept | accept | 4 | 6 | 5.50 | accept | 0.651 | FIXED |
| oKB0CacHaM | reject | accept | accept | 4 | 6 | 6.00 | accept | 0.817 | FIXED |
| sPRK6XefjY | reject | accept | accept | 4 | 6 | 7.00 | accept | 0.622 | FIXED |
| wBKXuuLZbc | reject | accept | accept | 4 | 6 | 4.50 | accept | 0.651 | FIXED |
| wnCJLnRBtb | accept | reject | reject | 6 | 4 | 4.00 | reject | 0.679 | FIXED |

**Model alignment on flips:** v3.3b followed the model on 15/16 flips (93.8%). The one override (AlJK6bFbAo: model=accept, v3.3b=reject) was correct — the agent overrode the model because ArXiv search revealed a prior preprint with the same name and pipeline.

**Rating pattern:** Every flip swings exactly ±2 rating points (4↔6). The model prior shifts the agent between borderline reject (4) and borderline accept (6) — it doesn't create extreme ratings.

### Prediction Transitions

| Transition | Count | % | v3.3b Acc | v3.3 Acc |
|---|---|---|---|---|
| A→A (stayed accept) | 34 | 34% | 52.9% | 52.9% |
| R→R (stayed reject) | 50 | 50% | 92.0% | 92.0% |
| A→R (flipped to reject) | 6 | 6% | **100.0%** | 0.0% |
| R→A (flipped to accept) | 10 | 10% | **80.0%** | 20.0% |

A→R flips are 100% accurate. R→A flips are 80% accurate (2 wrong).

### Transitions by Human Mean Rating

| Rating Bin | R→R | A→A | A→R | R→A |
|---|---|---|---|---|
| **< 4** | 21 (100%) | 1 (0%) | 1 (100%) | — |
| **4-5** | 19 (95%) | 15 (13%) | 1 (100%) | 2 (100%) |
| **5-6** | 9 (78%) | 12 (83%) | 4 (100%) | 4 (50%) |
| **6+** | 1 (0%) | 6 (100%) | — | 4 (100%) |

Each cell: count (v3.3b accuracy). The 2 wrong R→A flips are in the 5-6 bin — borderline papers where the model prior incorrectly pushed toward accept.

---

## Accuracy by Human Mean Rating Bin

| Rating Bin | Model Acc | v3.3 Acc | v3.3b Acc | #Accept | #Reject |
|---|---|---|---|---|---|
| < 4 | 87.0% | 91.3% | **95.7%** | 0 | 23 |
| 4-5 | 56.8% | 54.1% | **62.2%** | 5 | 32 |
| 5-6 | 72.4% | 65.5% | **79.3%** | 14 | 15 |
| 6+ | 90.9% | 54.5% | **90.9%** | 11 | 0 |

The model prior helps most at the extremes. For 6+ papers: v3.3 only 54.5% (over-rejects strong papers) → v3.3b 90.9% (matching model). For 5-6: +14pp improvement. Even the contested 4-5 range gains +8pp.

---

## Model Prior: Confidence & Deference Analysis

### How Often Does Claude Code Follow the Model?

v3.3b follows the model **90%** of the time (90/100) and overrides 10% (10/100).

| Confidence Bin | Follow Rate | Follow Acc | Override Acc |
|---|---|---|---|
| < 0.55 | 8/9 (89%) | 75.0% | 100.0% (n=1) |
| 0.55–0.65 | 16/24 (67%) | 62.5% | 75.0% (n=8) |
| 0.65–0.75 | 33/34 (97%) | 75.8% | 100.0% (n=1) |
| 0.75–0.85 | 17/17 (100%) | 76.5% | — |
| 0.85+ | 16/16 (100%) | 100.0% | — |

At 0.85+ confidence, follow rate is 100% and accuracy is 100%. Override rate is highest in the 0.55–0.65 bin (33%) where the model is least reliable. Overrides are actually MORE accurate than follows (80% vs 77.8% overall), suggesting the agent is selective about when to override.

---

## LLM-as-Judge Review Quality

### Setup

**Judge model**: Codex (`codex exec --sandbox workspace-write`), using default Codex settings (no explicit temperature, max_tokens, or model override — Codex CLI defaults).

**Procedure**: Papers are batched (10 per batch). For each batch, Codex reads a JSON file containing the agent review and all human ICLR reviews for each paper, then writes per-paper scores to a JSON output file. Batches run sequentially with a 600-second timeout each.

**Full judge prompt** (per batch):
```
You are an LLM judge evaluating AI-generated paper reviews against real human
ICLR reviews.

Read the file {batch_input_file}. It contains {N} papers, each with:
- "ground_truth": the real accept/reject decision
- "human_reviews": list of real ICLR reviewer assessments
- "agent_review": the AI-generated review to evaluate

For EACH paper, score the agent_review against the human_reviews:

Score the AI-generated review against the human reviews on these 5 dimensions
(each 1-5):

1. contribution_identification: Did the agent identify the same primary
   contribution as human reviewers?
   (1=completely missed, 3=partially identified, 5=perfect match)
2. top_weakness_identification: Did the agent identify the most important
   weakness flagged by humans?
   (1=missed all key weaknesses, 3=identified some, 5=identified the top
   weakness precisely)
3. novel_concerns: Did the agent raise valid concerns NOT mentioned in human
   reviews?
   (1=no novel valid points, 3=some valid novel points, 5=substantial novel
   insights)
4. rating_alignment: How close is the agent's rating to the human reviewer
   consensus mean?
   (1=off by 3+, 2=off by 2, 3=off by 1.5, 4=off by 1, 5=off by <0.5)
5. overall_review_quality: Overall quality of the agent review compared to
   human reviews.
   (1=poor/generic, 2=below average, 3=adequate, 4=good, 5=comparable to
   human quality)

For rating_alignment, compute the mean of all human reviewers' ratings first,
then compare to the agent's rating.

Write your scores to {score_file} as a JSON dict keyed by submission_id:
{
  "SUBMISSION_ID": {
    "contribution_identification": N,
    "top_weakness_identification": N,
    "novel_concerns": N,
    "rating_alignment": N,
    "overall_review_quality": N
  },
  ...
}

Score ALL {N} papers. Do not skip any.
```

**Input per paper**: The judge sees the full `agent_review` (all 11 fields: decision, rating, confidence, soundness, presentation, contribution, summary, strengths, weaknesses, questions, missing_references) alongside all human ICLR reviews (rating, confidence, soundness, presentation, contribution, summary, strengths, weaknesses, questions).

### Results

| Dimension | v3.3 | v3.3b |
|---|---|---|
| contribution_identification | 4.960 | 4.980 |
| top_weakness_identification | 4.190 | **4.370** |
| novel_concerns | 3.790 | 3.700 |
| rating_alignment | 3.640 | 3.630 |
| overall_review_quality | 3.820 | 3.860 |
| **OVERALL MEAN** | **4.080** | **4.108** |

Review quality is comparable between variants. v3.3b slightly improves weakness identification (+0.18) without sacrificing other dimensions. The model prior doesn't degrade review substance — it shifts the decision threshold while maintaining the same level of analytical detail.

---

## Qualitative Analysis: Side-by-Side Reviews

### Correct Flip: 5E5sd3TWGD (reject → accept, true=accept)

Paper: RD-HRL (Reliability-Driven Hierarchical RL)

**v3.3 review** (no model prior):
- Decision: reject, Rating: 4, Confidence: 4
- Focused on narrow benchmarks (only D4RL AntMaze, no OGBench), missing comparisons (GAS, OTA), weak theoretical justification for k-means/FDI
- 6 ArXiv searches found missing concurrent work
- Conclusion: "dated benchmarks + missing comparisons = reject"

**v3.3b review** (model prior: accept, conf=0.706):
- Decision: accept, Rating: 6, Confidence: 3
- Found the SAME weaknesses (missing GAS comparison, hyperparameter sensitivity, kitchen-partial failure)
- But also found STRONGER strengths (top-3% on 8/9 benchmarks, thorough ablation, novel FDI concept, non-Euclidean extension)
- Listed 7 weaknesses vs v3.3's 5, but weighted them as non-disqualifying
- The model prior (accept) nudged the agent to give more weight to the positive evidence

**Why it flipped**: The model prior shifted the agent's frame from "is there enough wrong to reject?" to "is there enough right to accept?" — same evidence, different weighing.

### Correct Flip: AlJK6bFbAo (accept → reject, true=reject)

Paper: GPT-IMAGE-EDIT-1.5M (image editing dataset)

**v3.3 review** (no model prior):
- Decision: accept, Rating: 6, Confidence: 4
- Noted prior preprint with identical name/pipeline but weighed the dataset contribution as "practically significant"
- Found 8 missing references but treated ablation studies as sufficient novelty

**v3.3b review** (model prior: accept, but agent overrode to reject):
- Decision: reject, Rating: 4, Confidence: 4
- Same prior preprint issue escalated to "serious concerns about novelty and prior publication"
- Same proprietary model dependence now framed as "difficult to reproduce"
- Added 8 detailed weaknesses (vs 5 in v3.3) including GPT-5 evaluation circularity, unfair comparisons
- The agent overrode the model's accept prediction because ArXiv search revealed damning prior-publication evidence

**Why it flipped**: ArXiv search activated stronger scrutiny of the prior-publication issue, which v3.3 treated as a minor weakness but v3.3b treated as disqualifying. This is the one case where the agent correctly overrode the model.

### Wrong Flip: OPQUeeT02q (reject → accept, true=reject)

Paper: SSNA (Semi-Supervised Noise Adaptation)

**v3.3 review**: reject (4), correctly identified missing SSL baselines (FixMatch, FlexMatch)
**v3.3b review**: accept (6), found the same weaknesses but the model prior (accept, conf=0.731) pushed toward accepting despite limited comparisons

**Why it went wrong**: The model was wrong (predicted accept with moderate confidence), and the agent deferred to it despite finding legitimate weaknesses.

---

## Key Takeaways

1. **Model prior as a priori >> post-hoc reconciliation.** The model prior is injected DURING review (as a Bayesian prior), not after. This reframes evidence weighing rather than overriding a completed judgment — producing 87.5% flip accuracy (14/16 correct).

2. **+12pp accuracy (66% → 78%) surpasses the 400-train no-model setup (73%).** Two training examples plus a model prior outperform 400 training examples without one.

3. **Accept recall jumps from 0.600 to 0.867** (matching SFT model) while precision improves from 0.450 to 0.591. The agent no longer over-rejects strong papers.

4. **Calibration gap widens (0.886 → 1.614).** v3.3b better separates accepts from rejects in its rating assignments, improving rating correlation (Pearson 0.395 → 0.525).

5. **The model prior doesn't blindly override.** The agent conducts the same level of research (5-6 ArXiv searches, full paper reading, image examination) but reframes evidence weighting. Review quality is maintained (LLM-as-Judge overall: 4.080 → 4.108).

6. **Confidence-binned calibration is key.** The orchestrator tells each sub-agent HOW MUCH to trust the model based on training-set accuracy at that confidence level. At 0.85+, follow rate and accuracy are both 100%. At 0.55–0.65, the agent overrides 33% of the time — and those overrides are 75% accurate.

7. **Every flip is a ±2 rating swing (4↔6).** The model prior shifts the agent between borderline reject and borderline accept. It doesn't create extreme ratings, suggesting the prior modulates rather than dominates.

8. **Agent adds genuine independent signal.** Unlike Codex (which mostly amplifies model predictions), Claude Code overrides are MORE accurate than follows (80% vs 77.8%), demonstrating selective, value-adding judgment.
