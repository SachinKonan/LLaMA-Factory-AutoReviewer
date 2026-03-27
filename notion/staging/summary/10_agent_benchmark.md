## 10. Agent Benchmark: LLM-as-Reviewer via Codex

We evaluate a Codex agent (GPT-4.1 backbone) on the paper acceptance prediction task, allowing it to read papers, build tools, and reason autonomously. This tests whether an LLM acting as a reviewer---with access to labeled calibration data but no fine-tuning---can match SFT models.

### 10.1 Task Setup

The agent receives:

- **Test set**: 200 unlabeled ICLR papers (100 accept, 100 reject; 50 per class from 2025 and 2026), drawn from the same v9 split used for SFT evaluation.
- **Training set**: 400 labeled ICLR papers (200 accept, 200 reject) for calibration---the agent can study these to learn what accepted vs. rejected papers look like.
- **Tools**: Python with numpy, pandas, scikit-learn, scipy, matplotlib. Sub-agent spawning for parallelism. No internet access.
- **Instructions**: The agent is framed as an "expert ICLR reviewer" and explicitly told to *read and review every test paper*, with ML models allowed only as a secondary cross-check. The prompt warns that statistical models on 400 papers perform near random (~55%).

The agent must output a `PREDICTIONS.json` file with `{"decision": "accept"/"reject", "why": "..."}` for all 200 papers, where the `why` field must reference specific paper content.

**Run artifacts** (all under `coding_agents/`):

| Artifact | Path |
|----------|------|
| Agent prompt | `AGENTS.override.md` |
| Training data (400 papers) | `TRAIN_SAMPLE.json` |
| Test data (200 papers) | `TEST_SAMPLE.json` |
| Extraction / review pipeline | `review_pipeline.py` |
| Compact paper markdowns | `derived/compact_{train,test}/` |
| Review rubric | `derived/review_rubric.md` |
| Batch reviews (20 batches) | `derived/reviews/batch_00.json` – `batch_19.json` |
| Logistic regression scores | `derived/model_scores.csv` |
| Final predictions | `PREDICTIONS.json` |
| Session logs (main + sub-agents) | `/scratch/gpfs/ZHUANGL/sk7524/.cache/.codex/sessions/2026/03/08/` |

### 10.2 Agent Strategy

The agent autonomously devised and executed a multi-stage pipeline:

**Stage 1: Paper extraction and compaction.** The agent built `review_pipeline.py`, which parses each paper's structured JSON (text blocks, tables, figures, equations) and extracts key sections (abstract, introduction, methods, experiments, ablations, conclusions) into compact markdown files. It uses keyword matching on section headers to select relevant sections and extracts "review signal" sentences containing terms like "we propose," "outperform," "ablation," and "limitation." All 600 papers (400 train + 200 test) were processed into `derived/compact_{train,test}/`.

**Stage 2: Rubric construction from training set.** After studying the 400 labeled papers, the agent wrote `derived/review_rubric.md`---a structured rubric codifying patterns that distinguish accepted from rejected papers at ICLR:

- *Accept signals*: explicit novelty beyond recombination of known components; strong benchmark coverage across multiple datasets; careful ablations or theoretical analysis; honest discussion of limitations.
- *Reject signals*: method is mostly a packaging/prompting variant of existing systems; results depend on cherry-picked settings or weak baselines; highlights interpretability/efficiency but trails standard task performance; attack/demo/observation paper without enough methodological depth.

The rubric explicitly sets the bar at "ICLR, not a workshop."

**Stage 3: Batch review of all test papers.** The agent divided the 200 test papers into 20 batches of 10, then reviewed each batch, producing `derived/reviews/batch_00.json` through `batch_19.json`. For each paper, it read the compact markdown, applied the rubric, and wrote a structured verdict with per-paper reasoning.

**Stage 4: Logistic regression as secondary signal.** The agent also trained a logistic regression model on features extracted from the training set, producing `derived/model_scores.csv` with per-paper acceptance probabilities. However, all 200 scores fell in the narrow range [0.40, 0.62] with mean 0.50---essentially random, confirming the prompt's warning about statistical models on small data.

**Stage 5: Merge and validate.** Batch reviews were merged into the final `PREDICTIONS.json`, and the agent ran its own validation to confirm all 200 papers were covered with valid decisions and non-empty `why` fields.

Total token usage: **173,036 tokens**.

### 10.3 Results

| Metric | Value |
|--------|-------|
| Accuracy | **63.0%** (126/200) |
| Precision (accept) | 63.0% |
| Recall (accept) | 63.0% |
| F1 (accept) | 63.0% |

**Confusion matrix** (rows = ground truth, columns = predicted):

|  | Pred Accept | Pred Reject |
|--|-------------|-------------|
| True Accept | 63 | 37 |
| True Reject | 37 | 63 |

**Prediction distribution**: 100 accept, 100 reject---perfectly matching the ground truth class balance.

**Quality of reasoning**: The `why` field averaged 369 characters (min 226, max 517), with substantive per-paper explanations referencing specific methods, datasets, and weaknesses. Example:

> *"The paper mainly studies brain alignment of existing multimodal encoders like ImageBind and TVLT on a single movie fMRI setup, rather than introducing a strong new modeling idea. The evidence is also limited..."* (correctly predicted reject)

**Comparison to other methods:**

| Method | Accuracy | Gap to SFT Vision |
|--------|----------|--------------------|
| Random | 50.0% | -20.4 pp |
| Zero-shot LLM (Qwen 3.5-122B) | 52.1% | -18.3 pp |
| TF-IDF + LogReg | 59.6% | -10.8 pp |
| **Codex Agent (GPT-4.1)** | **63.0%** | **-7.4 pp** |
| LogReg on 187 paper stats | 64.0% | -6.4 pp |
| Random Forest on 187 features | 65.0% | -5.4 pp |
| SFT Text (Qwen2.5-7B) | 66.9% | -3.5 pp |
| SFT Vision (Qwen2.5-VL-7B) | **70.4%** | --- |

### 10.3.1 Variant: Agent with Arxiv Search (GPT-5.4)

We ran a second agent experiment giving the Codex agent (upgraded to GPT-5.4 backbone) access to a semantic search tool over ~2.9 million arxiv papers via an MCP server. The search is automatically filtered by submission year to prevent information leakage (cutoff: December 15 of the year before the ICLR conference).

**Setup differences from baseline:**
- **Model**: GPT-5.4 (vs GPT-4.1 in baseline)
- **New tool**: `search_arxiv(submission_id, query, topk)` — semantic retrieval over arxiv, backed by Qwen3-0.6B embeddings + FAISS GPU index
- **Instructions**: Agent told to use search to verify novelty claims, check related work, and assess baseline completeness

**Tool usage**: The agent made **91 MCP tool calls** across 8 sub-agent sessions, searching for related work with substantive queries (e.g., `"teacher optimization synthetic instruction data influence function DPO student preference"`).

| Metric | Value |
|--------|-------|
| Accuracy | **58.0%** (116/200) |
| Precision (accept) | 58.0% |
| Recall (accept) | 58.0% |
| F1 (accept) | 58.0% |

**Confusion matrix** (rows = ground truth, columns = predicted):

|  | Pred Accept | Pred Reject |
|--|-------------|-------------|
| True Accept | 58 | 42 |
| True Reject | 42 | 58 |

**Prediction distribution**: 100 accept, 100 reject---again perfectly balanced.

**Comparison**: Adding arxiv search *decreased* accuracy by 5 pp (58% vs 63% baseline). The confusion matrix remains perfectly symmetric, but errors increased uniformly. Possible explanations: (1) search results introduced noise that caused the agent to second-guess borderline papers; (2) the agent may have over-weighted novelty signals from search (finding similar-sounding prior work for papers that were actually accepted); (3) the model upgrade from GPT-4.1 to GPT-5.4 may itself account for the difference, independent of search.

**Updated comparison table:**

| Method | Accuracy | Gap to SFT Vision |
|--------|----------|--------------------|
| Random | 50.0% | -20.4 pp |
| Zero-shot LLM (Qwen 3.5-122B) | 52.1% | -18.3 pp |
| **Codex Agent + Arxiv Search (GPT-5.4)** | **58.0%** | **-12.4 pp** |
| TF-IDF + LogReg | 59.6% | -10.8 pp |
| Codex Agent (GPT-4.1) | 63.0% | -7.4 pp |
| LogReg on 187 paper stats | 64.0% | -6.4 pp |
| Random Forest on 187 features | 65.0% | -5.4 pp |
| SFT Text (Qwen2.5-7B) | 66.9% | -3.5 pp |
| SFT Vision (Qwen2.5-VL-7B) | **70.4%** | --- |

### 10.4 Analysis

**Symmetric errors.** The confusion matrix is perfectly symmetric (37 FP, 37 FN), indicating no systematic bias toward accepting or rejecting. This contrasts with zero-shot LLMs, which often default to one label. The balanced prediction distribution (100/100) suggests the agent successfully calibrated its decision threshold using the labeled training set.

**Rubric quality vs. execution gap.** The agent's rubric closely mirrors criteria used by actual ICLR reviewers---novelty, experimental rigor, ablation depth, honest framing. The `why` fields demonstrate that the agent engaged with paper content at a substantive level. Yet accuracy remains only 13 pp above random, suggesting that while the agent can articulate reviewer-like reasoning, it struggles to reliably *discriminate* marginal papers at the ICLR acceptance boundary.

**Logistic regression was near-random.** The model scores (mean 0.50, range 0.40--0.62) indicate the statistical model contributed negligibly. The 63% accuracy is driven primarily by the agent's LLM-based paper reviews, not the logistic regression. This confirms that hand-crafted features on 400 papers are insufficient for this task.

**Comparison to classical ML.** Despite reading full papers and producing substantive reasoning, the Codex agent (63.0%) underperforms Random Forest on 187 engineered features (65.0%) by 2 pp. This is a striking finding: a frontier LLM given unlimited time to read and reason about papers barely matches a feature-engineering baseline that uses only surface statistics (page count, reference count, readability scores).

### 10.5 Discussion

The agent benchmark reveals the limits of LLM-as-reviewer in a zero-shot-ish setting. Even with 400 labeled calibration papers, a rubric built from those papers, and per-paper reasoning, the Codex agent achieves only 63%---7.4 pp below SFT Vision (70.4%) and 2 pp below classical Random Forest (65.0%).

**Why does SFT win?** Fine-tuning on thousands of papers with venue-specific labels allows the model to internalize subtle distributional patterns---stylistic norms, community expectations, topic-specific acceptance rates---that cannot be captured by reading individual papers in isolation. The SFT models see ~20,000 papers during training; the agent sees only 400 for calibration and reviews each test paper independently.

**The calibration bottleneck.** The agent's strategy was sound: study labeled examples, build a rubric, apply it systematically. But 400 calibration papers are too few to learn the nuanced decision boundary at a top venue, where the difference between accept and reject often comes down to relative positioning against concurrent submissions rather than absolute quality.

**Substantive reasoning with limited discrimination.** The agent's `why` fields are more informative than any model's logits---they provide actionable, paper-specific feedback. This suggests a hybrid approach: use SFT models for classification accuracy and LLM agents for generating reviewer-style explanations. The agent's reasoning could serve as an interpretability layer on top of the SFT model's predictions.

**Implications for automated reviewing.** These results caution against deploying LLMs as standalone reviewers. Even with careful calibration and paper-by-paper reasoning, frontier LLMs cannot reliably reproduce venue-specific acceptance decisions. Fine-tuning on venue-specific data remains critical for predictive accuracy.
