# Codex Agent Experiment: Unbiased (50/50) Test Set Analysis

## Main Results

| Metric | No Model Prior | With Model Prior | Delta |
|---|---|---|---|
| **Accuracy** | 61.0% (122/200) | **64.5%** (129/200) | +3.5pp |
| **Accept Recall** | **77.0%** (77/100) | 72.0% (72/100) | -5.0pp |
| **Reject Recall** | 45.0% (45/100) | **57.0%** (57/100) | +12.0pp |
| Predicted Accepts | 132/200 | 115/200 | — |

Test set: 200 papers (100 accept, 100 reject), balanced 50/50 from ICLR 2026.

---

## Experiment Setup: What Codex Does

### Inputs

Both variants receive the same core data:
- **`TRAIN_SAMPLE.json`** — 400 labeled ICLR 2026 papers (200 accept, 200 reject) for calibration
- **`TEST_SAMPLE.json`** — 200 unlabeled ICLR 2026 papers to classify
- **`AGENTS.override.md`** — Task instructions framing Codex as an expert ICLR reviewer

The **with-model** variant additionally receives `model_prediction` ("accept"/"reject") and `model_confidence` (0-1 float) per test paper, from a fine-tuned SFT vision model. The no-model variant has no access to any model signal.

### No-Model Variant Workflow

Codex autonomously builds a single-pass review pipeline:

**Step 1: Build Tooling** — Codex writes `paper_review_tools.py`, a Python script that converts raw `content_list` JSON entries into structured reviewer-friendly markdown briefs. The script extracts and organizes: section headers, introduction text, claimed contributions (via regex for "we propose/present/introduce"), method highlights, experimental evidence, reported results (via regex for "outperform/SOTA/improves"), and figure/table captions.

**Step 2: Calibrate on Training Set** — Generates markdown summaries for ~20 training papers (10 accept, 10 reject) into `calibration/train_balanced/`. Reads these to learn what distinguishes accepts from rejects.

**Step 3: Generate Test Paper Summaries** — Runs `paper_review_tools.py` on all 200 test papers, producing individual markdown briefs in `paper_summaries/test/` (one `.md` file per paper).

**Step 4: Batch and Parallelize Reviews** — Splits the 200 test papers into **10 batches of 20** papers each, writing batch manifests to `batch_manifests/batch_00.txt` through `batch_09.txt`.

**Step 5: Spawn Sub-Agent Workers** — Launches parallel sub-agents, one per batch. Each worker reads its batch manifest and paper summaries, reviews each paper as an ICLR reviewer, and writes `decision`, `why`, and `confidence` to `worker_reviews/batch_XX.json`.

**Step 6: Aggregate** — Merges all 10 worker outputs, verifies every test submission appears exactly once, and writes the final `PREDICTIONS.json`.

**Step 7: Document** — Writes `REVIEWER_SKILLS.md`, a self-contained guide for future agents.

### With-Model Variant Workflow (Significantly Different)

The with-model Codex autonomously designed a **multi-stage review-and-correct pipeline** — not just the same pipeline with model predictions appended. Key differences:

**Step 1: Build a Full CLI Tool (`main.py`)** — Instead of a simple text extractor, Codex builds a comprehensive CLI with subcommands: `export`, `export-calibration`, `make-batches`, `merge-reviews`, `audit-predictions`, and `stats`. This is a more sophisticated toolchain.

**Step 2: Statistical Analysis of Model Signal** — Before reviewing any papers, Codex runs `stats` to analyze the model prior distribution (confidence bins, accept/reject split). It generates `model_signal.tsv` with per-paper model predictions, confidence, and a text probability feature. It also generates `review_table.tsv` with extracted features (has_ablation, table count, figure count, header structure, intro text, contribution claims, experiment text) for all 200 papers.

**Step 3: Calibrate** — Same as no-model: reads 20 labeled training papers (10 accept, 10 reject).

**Step 4: Generate Paper Cards with Model Prior** — Creates `paper_cards/` with condensed markdown summaries. Critically, each card includes the model prediction and confidence at the top (e.g., "Model prior: reject (0.6512)"). Paper cards are then grouped into `paper_card_batches/` (20 batches of 10 papers each — smaller batches than no-model).

**Step 5: First-Pass Review (120/200 papers)** — Codex reviews 120 papers in a first wave, producing `review_notes.tsv`. Of these 120, only 11 override the model prior. This first pass covers 60% of papers.

**Step 6: Sub-Agent Batch Reviews (all 200 papers)** — Spawns 20 parallel sub-agents (batches of 10), each reading paper card batches and producing `agent_reviews/batch_XX.json`. All 200 papers are covered. Of the 200 agent reviews, 23 override the model prior (11.5%).

**Step 7: Correction Step** — Compares agent reviews against first-pass review notes. **13 papers have their decisions changed** between stages (e.g., 7 flipped accept→reject, 6 flipped reject→accept). This is the review-and-correct cycle — the agent revisits initial judgments.

**Step 8: Fallback Script (`build_predictions.py`)** — A safety net that uses agent reviews where available, and falls back to the raw model prediction with auto-generated "why" text for any unreviewed papers. In practice, all 200 papers were agent-reviewed so this fallback wasn't needed.

**Step 9: Merge and Audit** — Merges agent reviews into `PREDICTIONS.json`, then runs `audit-predictions` to check coverage, decision validity, and count model overrides.

### Key Differences Between Variants

| Aspect | No Model Prior | With Model Prior |
|---|---|---|
| **Tooling** | Simple extraction script | Full CLI with stats/audit/merge subcommands |
| **Batch size** | 20 papers/batch (10 batches) | 10 papers/batch (20 batches) |
| **Review stages** | Single pass | Multi-stage: first-pass notes → agent reviews → corrections |
| **Papers corrected** | 0 (single pass) | 13 changed between first-pass and final review |
| **Feature extraction** | Summaries only | Feature table (ablation flag, fig/table counts, etc.) + model signal analysis |
| **Model prior visible?** | No | Yes — shown at top of every paper card |
| **Override rate** | N/A | 11.5% (23/200 disagree with model) |
| **Fallback mechanism** | None | `build_predictions.py` defaults to model prior for unreviewed papers |

---

## Breakdown by Reviewer Rating

### Accuracy by Mean Rating Bin

| Rating Bin | No Model Acc | With Model Acc | #Accept | #Reject |
|---|---|---|---|---|
| mean < 4 | 60.0% | **82.9%** | 1 | 34 |
| mean 4-5 | 46.5% | 46.5% | 26 | 45 |
| mean 5-6 | 70.1% | 64.2% | 46 | 21 |
| mean 6+ | 79.2% | **91.7%** | 24 | 0 |

The model prior helps most at the extremes (mean < 4 and mean 6+) where signal is clearest, but provides no benefit in the contested 4-5 range.

### Accuracy by pct_rating (Percentile Rating)

| pct_rating | No Model Acc | With Model Acc | #Accept | #Reject |
|---|---|---|---|---|
| 0-20% | 60.0% | **82.9%** | 1 | 34 |
| 20-40% | 38.7% | 54.8% | 4 | 27 |
| 40-60% | 52.5% | 40.0% | 22 | 18 |
| 60-80% | 63.4% | 61.0% | 25 | 16 |
| 80-100% | 79.2% | 79.2% | 48 | 5 |

---

## Prediction Transitions: No Model -> With Model

### Overall

| Transition | Count | % | With-Model Acc | No-Model Acc |
|---|---|---|---|---|
| A->A (stayed accept) | 97 | 48.5% | 66.0% | 66.0% |
| R->R (stayed reject) | 50 | 25.0% | 70.0% | 70.0% |
| A->R (flipped to reject) | 35 | 17.5% | **62.9%** | 37.1% |
| R->A (flipped to accept) | 18 | 9.0% | 44.4% | 55.6% |

The model prior flips 53 papers (26.5%). A->R flips are mostly beneficial (62.9% accurate vs 37.1% before). R->A flips are mostly harmful (44.4% accurate vs 55.6% before).

### Transition Table by Mean Rating

| Rating Bin | R->R (stayed reject) | A->A (stayed accept) | A->R (flipped) | R->A (flipped) |
|---|---|---|---|---|
| **mean < 4** | 19 (94.7%) | 2 (0.0%) | 11 (100.0%) | 3 (0.0%) |
| **mean 4-5** | 15 (66.7%) | 38 (36.8%) | 14 (57.1%) | 4 (25.0%) |
| **mean 5-6** | 14 (50.0%) | 37 (81.1%) | 9 (33.3%) | 7 (42.9%) |
| **mean 6+** | 2 (0.0%) | 20 (100.0%) | 1 (0.0%) | 4 (100.0%) |

Each cell: count (with-model accuracy of those samples).

Key finding: A->R flips are 100% accurate for low-rated papers (mean < 4) but only 33% accurate for mid-rated papers (mean 5-6). The model prior over-rejects in the middle.

### Transition Table by pct_rating

| pct_rating | R->R (stayed reject) | A->A (stayed accept) | A->R (flipped) | R->A (flipped) |
|---|---|---|---|---|
| **0-20%** | 19 (94.7%) | 2 (0.0%) | 11 (100.0%) | 3 (0.0%) |
| **20-40%** | 7 (100.0%) | 13 (15.4%) | 10 (80.0%) | 1 (0.0%) |
| **40-60%** | 8 (37.5%) | 25 (48.0%) | 4 (0.0%) | 3 (33.3%) |
| **60-80%** | 11 (54.5%) | 21 (71.4%) | 5 (40.0%) | 4 (50.0%) |
| **80-100%** | 5 (20.0%) | 36 (97.2%) | 5 (20.0%) | 7 (71.4%) |

---

## Model Prior: Confidence & Deference Analysis

### How Often Does Codex Follow the Model?

Codex follows the model prior **88.5%** of the time (177/200) and overrides it only 11.5% (23/200).

| Model Confidence | Follow Rate | Follow Acc | Override Acc |
|---|---|---|---|
| < 0.5 | 100% | 66.7% | — |
| 0.5-0.6 | 89.1% | 51.2% | 60.0% |
| 0.6-0.7 | 78.0% | 59.0% | 27.3% |
| 0.7-0.8 | 89.3% | 70.0% | 50.0% |
| 0.8-0.9 | 95.5% | 85.7% | 100% (n=1) |
| 0.9+ | 100% | 87.0% | — |

Confidence is well-calibrated: following at 0.9+ yields 87% accuracy vs 51% at 0.5-0.6.

### Asymmetry in Override Direction

Codex almost never overrides accept predictions (3/100) but overrides reject predictions more often (19/100). This reflects an inherent accept bias — Codex resists reject signals from the model.

### Transitions by Model Confidence

| Confidence | R->R | A->A | A->R | R->A | Total | WM Acc |
|---|---|---|---|---|---|---|
| < 0.5 | — | 1 (100%) | — | 2 (50%) | 3 | 66.7% |
| 0.5-0.6 | 10 (40%) | 20 (50%) | 12 (67%) | 4 (50%) | 46 | 52.2% |
| 0.6-0.7 | 7 (43%) | 30 (63%) | 5 (20%) | 8 (38%) | 50 | 52.0% |
| 0.7-0.8 | 10 (80%) | 34 (71%) | 9 (56%) | 3 (33%) | 56 | 67.9% |
| 0.8-0.9 | 6 (100%) | 12 (83%) | 3 (67%) | 1 (100%) | 22 | 86.4% |
| 0.9+ | 17 (82%) | — | 6 (100%) | — | 23 | 87.0% |

### Follow Rate: Confidence x Mean Rating Cross-Tab

|  | r=<4 | r=4-5 | r=5-6 | r=6+ |
|---|---|---|---|---|
| < 0.5 | 1/1 (100%) | 1/1 (100%) | — | 1/1 (100%) |
| 0.5-0.6 | 4/4 (100%) | 13/15 (87%) | 21/22 (95%) | 3/5 (60%) |
| 0.6-0.7 | 1/1 (100%) | 20/26 (77%) | 12/16 (75%) | 6/7 (86%) |
| 0.7-0.8 | 7/9 (78%) | 18/18 (100%) | 19/23 (83%) | 6/6 (100%) |
| 0.8-0.9 | 4/4 (100%) | 7/7 (100%) | 3/4 (75%) | 7/7 (100%) |
| 0.9+ | 16/16 (100%) | 4/4 (100%) | 2/2 (100%) | 1/1 (100%) |

Lowest follow rates occur where model prediction conflicts with paper quality signal — 0.5-0.6 confidence for mean 6+ papers (60%) and 0.7-0.8 confidence for mean < 4 papers (78%).

---

## Qualitative Analysis: How Reasoning Changes in A->R Flips

We analyzed the "why" field in agent predictions for all 35 A->R transitions. 22 were correct (true reject), 13 were wrong (true accept).

### Correct A->R Flips: Four Rhetorical Patterns

**Pattern 1: Scope/Scale Skepticism** (most common)
The no-model agent describes what the paper does and finds it convincing. The with-model agent reframes the same experiments as limited in scope or scale.

- (IDQdJphVxR, mean=3.5, conf=0.75) No-model: "evaluates on MNIST, SVHN, CIFAR-10, CIFAR-100." With-model: same experiments framed as "limited to single-class forgetting on **small** image classification benchmarks."
- (90EZvjKMqK, mean=2.5, conf=0.98) No-model: "meaningful methodological advance." With-model: "empirical validation is still quite narrow: one main threat model, essentially one baseline."
- (3oo2d8TWG7, mean=2.0, conf=0.71) No-model: "genuinely nontrivial architecture... multiple layers of evidence." With-model: "the extracted evidence does not show comparably strong experimental validation or clean headline gains."

**Pattern 2: Incremental Reframing**
The no-model agent treats the contribution as novel. The with-model agent reframes the identical contribution as an incremental extension of prior work.

- (QB7w5JuchB, mean=3.5, conf=0.96) No-model: "meaningful modeling change over prior VAE." With-model: "the step from prior t3VAE/C-VAE variants feels **fairly incremental**."
- (Vvks41GeL9, mean=5.5, conf=0.53) No-model: "focused but well-executed contribution." With-model: "feels more like an **incremental tuning rule** than a major advance."
- (bJetbxYtRM, mean=4.0, conf=0.53) No-model: "new prompting framework for temporal interaction graphs." With-model: "incremental extension of existing graph prompting."

**Pattern 3: Complexity vs. Gains Tradeoff**
The with-model agent introduces cost-benefit skepticism — the method is complex but gains are modest.

- (gWBMEq9PDj, mean=3.5, conf=0.85) No-model accepts the multi-component pipeline. With-model: "a **heavy combination** of ImageBind, Vicuna, adapters... for **relatively modest improvements**."
- (fxVEVO8Z7M, mean=2.0, conf=0.97) No-model: "consistent gains." With-model: "contribution feels **incremental** over prior contrastive/alignment methods... presentation is sloppy."

**Pattern 4: Weakening Positive Evidence**
The with-model agent takes the exact same experimental results but reinterprets them as insufficient.

- (h1u9YsjLBy, mean=2.5, conf=0.71) No-model: "99.4% average accuracy... results are unusually strong." With-model: "less convincing than the headline accuracy suggests: the setup assumes privileged access."
- (ib1Mg4c5pl, mean=4.0, conf=0.97) No-model: "precision around 0.95 with recall 0.71." With-model: "**still reaches only 0.71 recall** for malware detection."

### Wrong A->R Flips: Over-Applying the Incremental Label

Wrong flips tend to have higher ratings (mean 4.5-6.0) and lower model confidence (0.53-0.84). The model prior pushes Codex to call papers "incremental" even when the community accepted them.

- (jUuXNrG7wh, mean=6.0, conf=0.71) With-model dismisses the contribution as "essentially a prompt-level heuristic" — but the community valued the diagnostic insight.
- (7L7kmHHfgf, mean=5.5, conf=0.84) With-model: "feels incremental relative to recent anomaly-detection design patterns." Accepted with solid [6,6,6,4] ratings.
- (CsoR8ztROC, mean=5.0, conf=0.75) With-model: "primarily a large modular sweep of known time-series design choices." Community accepted as a useful empirical contribution.
- (SSd3GENRAU, mean=5.5, conf=0.71) With-model: "fairly incremental combination of low-pass spectral filtering." Accepted with [6,4,6,6] ratings.

---

## Key Takeaways

1. **Model prior helps most at extremes**: +23pp accuracy on mean < 4 papers (60% -> 83%), +12pp on mean 6+ (79% -> 92%). No benefit in the contested mean 4-5 range (46.5% -> 46.5%).

2. **Codex is highly deferential**: Follows the model 88.5% of the time. At high confidence (0.8+), follow rate is 95-100%. Overrides are rare and usually wrong (27% accurate at 0.6-0.7 confidence).

3. **Accept bias persists**: Codex overrides 19% of model reject predictions but only 3% of model accept predictions. The agent resists reject signals.

4. **A->R flips use a consistent rhetorical template**: The model prior shifts reasoning through "scope skepticism," "incremental reframing," "complexity vs. gains," and "evidence weakening." These are valid for genuinely weak papers but get over-applied to borderline papers the community accepted.

5. **The agent adds little independent signal**: It mostly amplifies the model prediction. The net accuracy gain (+3.5pp) comes from there being more clear rejects to correctly flip than borderline papers to incorrectly damage.
