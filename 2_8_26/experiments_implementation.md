# Experiments Implementation Guide

**Date**: 2026-02-09
**Location**: `2_8_26/`
**Refines**: `working_plan_3.md`

## Directory Structure

```
2_8_26/
├── logs/                              # Unified SLURM logs
│   ├── h1_base_model/
│   ├── h2_contamination/
│   ├── b1_pdr/
│   ├── b2_role_prompts/
│   ├── c1_bayesian/
│   └── q1_reviewer_archetypes/
├── shared/
│   ├── prompt_templates.py            # Role prompts, base model prompts, meta-review prompts
│   └── analysis_utils.py              # Common plotting/metric helpers
├── h1_base_model/                     # Hypothesis 1: Instruct finetuning artifact
│   ├── data/
│   ├── results/
│   ├── metrics/
│   ├── generate_dataset.py
│   ├── run_inference.sbatch
│   └── analyze.py
├── h2_contamination/                  # Hypothesis 2: Data contamination
│   ├── data/
│   ├── results/
│   ├── metrics/
│   ├── generate_ablation_datasets.py
│   ├── run_ablation.sbatch
│   └── analyze.py
├── b1_pdr/                            # Parallel Distill Response
│   ├── results/
│   ├── metrics/
│   ├── run_pdr.sbatch
│   └── analyze.py
├── b2_role_prompts/                   # Role-playing / adversarial prompts
│   ├── data/
│   ├── results/
│   ├── metrics/
│   ├── generate_role_datasets.py
│   ├── run_role_inference.sbatch
│   ├── run_strategy_d.sbatch
│   └── analyze.py
├── c1_bayesian/                       # Bayesian decision correction
│   ├── results/
│   ├── metrics/
│   ├── estimate_priors.py
│   ├── bayesian_correction.py
│   └── analyze.py
└── q1_reviewer_archetypes/            # Reviewer style modeling
    ├── results/
    ├── metrics/
    ├── cluster_reviews.py
    ├── classify_generated.py
    └── analyze.py
```

---

## Shared Utilities

### `shared/prompt_templates.py`

Constants and a helper function used by all experiments:

| Constant | Used By | Description |
|----------|---------|-------------|
| `CRITICAL_MODIFIER` | B2 | "Be critical... When in doubt, reject" |
| `ENTHUSIASTIC_MODIFIER` | B2 | "Give benefit of doubt... Focus on novelty" |
| `STANDARD_MODIFIER` | B2 | Empty string (baseline) |
| `BASE_COMPLETION_SYSTEM_PROMPT` | H1 | Simple prompt for base (non-instruct) models |
| `METAREVIEW_SYSTEM_PROMPT` | B1 | "You are an Area Chair..." for meta-review aggregation |
| `METAREVIEW_USER_TEMPLATE` | B1 | Template for 5 reviews → meta-review |
| `STRATEGY_D_SYSTEM_PROMPT` | B2 | Synthesize critical + enthusiastic perspectives |
| `STRATEGY_D_USER_TEMPLATE` | B2 | Template for critical + enthusiastic → decision |
| `TITLE_ONLY_SYSTEM_PROMPT` | H2 | "Generate the abstract for this paper" |
| `TITLE_ONLY_USER_TEMPLATE` | H2 | "Title: {title}" |
| `CONTENT_ABLATION_SYSTEM_PROMPT` | H2 | Standard review prompt for partial content |
| `build_system_prompt(modifier, output_format)` | B2, H2 | Assembles system prompt from components |

### `shared/analysis_utils.py`

Reusable functions for all `analyze.py` scripts:

- **Data loading**: `load_results()`, `load_predictions()`, `load_dataset()`
- **Metrics**: `compute_accuracy()`, `compute_acceptance_rate()`, `compute_recall()`, `compute_precision()`, `compute_confusion_matrix()`, `compute_metrics_summary()`, `compute_by_year()`
- **Plotting**: `setup_plot_style()`, `plot_confusion_matrix()`, `plot_bar_comparison()`, `plot_line()`
- **I/O**: `save_metrics_json()`

---

## Experiment Details

### H1: Base Model (Instruct Finetuning Artifact)

**Hypothesis**: Instruct models are RLHF-tuned to be helpful/agreeable, causing extreme optimism (95-100% accept rate).

| Item | Detail |
|------|--------|
| **Data** | `iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test` |
| **Model** | `Qwen/Qwen2.5-7B` (base, non-instruct) |
| **Comparison** | `Qwen/Qwen2.5-7B-Instruct` (B2 standard results from same v7 split: `b2_role_prompts/results/clean/standard/`) |
| **Template** | `default` (completion-style, no chat tokens) |
| **Context** | 20480 tokens + YaRN rope scaling (4x) |

**Files**:
- `generate_dataset.py` — Reformats ShareGPT data: replaces system prompt with `BASE_COMPLETION_SYSTEM_PROMPT`, strips prefix from user message, removes images. Outputs to `data/` with `dataset_info.json`.
- `run_inference.sbatch` — Single SLURM job. Calls `vllm_infer_ensemble.py` with `template=default`, then `extract_results.py`.
- `analyze.py` — Loads base + instruct results (default: B2 standard/clean for same-split comparison). Computes acceptance rates, agreement statistics, Venn diagram, detailed disagreement analysis split by ground truth correctness.

**Plots**:
- `acceptance_rate_base_vs_instruct.png` — Bar chart comparing acceptance rates
- `accuracy_base_vs_instruct.png` — Bar chart comparing accuracy
- `confusion_base.png` / `confusion_instruct.png` — Confusion matrices
- `venn_agreement.png` — Venn diagram: base correct vs instruct correct, showing both-correct (overlap), only-base-correct, only-instruct-correct, both-wrong regions
- `disagreement_analysis.png` — Detailed disagreement breakdown by prediction type × ground truth + pie chart of agreement/correctness distribution

**Run**:
```bash
python 2_8_26/h1_base_model/generate_dataset.py
sbatch 2_8_26/h1_base_model/run_inference.sbatch
# After job completes:
python 2_8_26/h1_base_model/analyze.py
```

---

### H2: Data Contamination (Progressive Ablation)

**Hypothesis**: Papers (especially famous ones) are in the pretraining corpus, leading to memorized positive sentiment.

| Item | Detail |
|------|--------|
| **Data** | 5 content levels from same test set |
| **Models** | `Qwen/Qwen2.5-7B-Instruct` (vLLM) + `gemini-2.5-flash` (title-only, existing script) |
| **Existing assets** | `generate_title_dataset.py`, `gemini_title_infer.py` (already in `2_8_26/`) |

**Content Levels**:

| Level | Content | Cutoff | Max New Tokens |
|-------|---------|--------|----------------|
| `title_only` | Paper title only (generates abstract) | 4096 | 512 |
| `title_abstract` | Title + Abstract | 8192 | 2048 |
| `title_intro` | Title + Abstract + Introduction | 16384 | 4096 |
| `title_conclusion` | Title + Abstract + Conclusion | 16384 | 4096 |
| `full_paper` | Full paper (baseline) | 20480 | 10240 |

**Files**:
- `generate_ablation_datasets.py` — Extracts sections using header pattern matching. Generates 5 dataset variants. Handles missing sections gracefully.
- `run_ablation.sbatch` — Job array 0-4 (one per level). Calls `vllm_infer_ensemble.py` + `extract_results.py`. Skips extraction for `title_only` (generates abstracts, not decisions).
- `analyze.py` — Accuracy vs content level plots, acceptance rate by level, **plus title-only abstract similarity analysis**: computes cosine similarity between generated and true abstracts using sentence-transformers, plots distribution with examples at low/medium/high bins, and correlates similarity with citation percentile and average reviewer score.

**Plots**:
- `accuracy_vs_content_level.png` — Line+bar: accuracy at each level
- `acceptance_rate_by_level.png` — Colored bars (red=high, green=calibrated)
- `abstract_similarity_distribution.png` — Histogram of cosine similarity (generated vs true abstract) with color-coded bins (blue=low, orange=moderate, red=high/contaminated) + example abstracts at each similarity level
- `similarity_vs_citations.png` — Scatter plot: similarity vs citation percentile (normalized by year), with trend line and Pearson r, colored by accept/reject
- `similarity_vs_rating.png` — Scatter plot: similarity vs average reviewer score, with trend line and Pearson r, colored by accept/reject

**Run**:
```bash
python 2_8_26/h2_contamination/generate_ablation_datasets.py
sbatch 2_8_26/h2_contamination/run_ablation.sbatch
# After jobs complete:
python 2_8_26/h2_contamination/analyze.py
```

---

### B1: Parallel Distill Response (PDR)

**Method**: Generate N=5 diverse reviews → meta-reviewer synthesizes into single decision (unlike simple majority voting, PDR reasons over diverse perspectives).

| Item | Detail |
|------|--------|
| **Data** | v7 split datasets (falls back to B2 standard datasets if v7 new_fewshot unavailable) |
| **Models** | `Qwen/Qwen2.5-7B-Instruct` (clean) / `Qwen2.5-VL-7B-Instruct` (vision) |
| **Infrastructure** | Fully reuses `vllm_infer_ensemble.py` (n_gen=5) + `run_metareview.py` |

**Files**:
- `run_pdr.sbatch` — Job array 0-2 (clean, clean_images, vision). Uses v7 split datasets (with fallback chain: v7 new_fewshot → v7 new → B2 standard). Only reuses existing predictions if >100 samples (prevents copying stale 10-sample results). Then: extract single + majority + create metareview dataset + run metareview inference + extract metareview results.
- `analyze.py` — Compares single vs majority vs PDR (meta-review) across modalities. Pairwise agreement heatmap.

**Plots**:
- `strategy_comparison.png` — Grouped bar: accuracy by modality × strategy
- `strategy_agreement.png` — Heatmap: pairwise agreement between strategies

**Run**:
```bash
sbatch 2_8_26/b1_pdr/run_pdr.sbatch
# After jobs complete:
python 2_8_26/b1_pdr/analyze.py
```

---

### B2: Role-Playing Prompts

**Method**: Test if explicit reviewer personas reduce bias. Plus "Strategy D" — multi-perspective pipeline.

| Item | Detail |
|------|--------|
| **Data** | 3 roles × 3 modalities = 9 new datasets |
| **Models** | `Qwen/Qwen2.5-7B-Instruct` (clean) / `Qwen2.5-VL-7B-Instruct` (vision) |
| **Existing** | `ablations/scripts_v1/generate_datasets.py` has critical/less_critical prompts (reused) |

**Roles**:

| Role | System Prompt Modifier |
|------|----------------------|
| `critical` | "Be critical... Look for flaws... When in doubt, reject." |
| `enthusiastic` | "Give benefit of doubt... Focus on novelty and contribution." |
| `standard` | No modifier (baseline) |

**Strategy D Pipeline**:
1. Run critical inference → get weakness-focused reviews
2. Run enthusiastic inference → get strength-focused reviews
3. Meta-reviewer receives both and synthesizes final decision

**Files**:
- `generate_role_datasets.py` — Injects role system prompts via `build_system_prompt()`. Supports `--roles`, `--modalities`, `--output_format` flags.
- `run_role_inference.sbatch` — Job array 0-8 (3 roles × 3 modalities). Layout: `MODALITY_IDX = task_id / 3`, `ROLE_IDX = task_id % 3`.
- `run_strategy_d.sbatch` — Job array 0-2 (one per modality). Requires critical + enthusiastic results. Creates Strategy D dataset inline (Python one-liner), runs meta-review inference, extracts results.
- `analyze.py` — Acceptance rate by role, accuracy by role per modality, Strategy D comparison.

**Plots**:
- `acceptance_rate_by_role.png` — Grouped bar: role × modality
- `strategy_d_comparison.png` — Per-modality subplots comparing all roles + Strategy D

**Run**:
```bash
python 2_8_26/b2_role_prompts/generate_role_datasets.py
sbatch 2_8_26/b2_role_prompts/run_role_inference.sbatch
# After role inference completes:
sbatch 2_8_26/b2_role_prompts/run_strategy_d.sbatch
# After all jobs complete:
python 2_8_26/b2_role_prompts/analyze.py
```

---

### C1: Bayesian Decision Correction

**Method**: Use estimated confusion matrix to correct predictions via Bayes' rule. Find optimal decision threshold.

| Item | Detail |
|------|--------|
| **Data** | Any experiment's results (validation or test) |
| **Priors** | Estimated from data, or manual: P(Pred Reject \| True Accept) ≈ 50%, P(Pred Accept \| True Reject) ≈ 20% |
| **Dependencies** | Requires results from at least one other experiment |

**Files**:
- `estimate_priors.py` — Computes confusion matrix rates from results. Can scan a directory of results or process a single file. Outputs `priors.json`.
- `bayesian_correction.py` — Applies Bayes' rule: `P(True Label | Prediction)`. Grid searches over thresholds (0.1–0.9) to maximize accuracy. Supports manual priors via `--fpr`/`--fnr` flags or file via `--priors`.
- `analyze.py` — Runs the full pipeline: estimate priors → find optimal threshold → correct predictions. Generates before/after plots.

**Plots**:
- `roc_curve.png` — Accuracy vs threshold curve with optimal marked
- `confusion_before_after.png` — Side-by-side confusion matrices

**Run**:
```bash
# Default: uses B2 standard/clean results (v7 split, ~2024 samples)
python 2_8_26/c1_bayesian/analyze.py

# Custom input
python 2_8_26/c1_bayesian/analyze.py --input 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl

# Manual priors
python 2_8_26/c1_bayesian/bayesian_correction.py \
    --input 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl \
    --fpr 0.2 --fnr 0.5

# Estimate priors from all experiment results
python 2_8_26/c1_bayesian/estimate_priors.py --input_dir 2_8_26/b2_role_prompts/results
```

---

### Q1: Reviewer Archetypes

**Method**: Cluster ground truth reviews from the test set into archetypes using K-Means clustering. This analysis reveals the distribution of reviewer types (e.g., harsh critics, enthusiasts, balanced reviewers) in the real ICLR review data.

| Item | Detail |
|------|--------|
| **Data** | Ground truth reviews from HF dataset (`original_reviews` column) filtered to test-set papers |
| **HF Dataset** | `/n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer/data/hf_dataset_new8_noref_cropped_2017_2026_with_decisions` |
| **Test Filter** | Submission IDs from `*_split7_balanced_clean_binary_noreviews_v7_test/data.json` |
| **Embedding model** | `all-MiniLM-L6-v2` (sentence-transformers) |
| **Clustering** | K-Means (default k=5) |
| **Dependencies** | `pip install sentence-transformers scikit-learn datasets` |

**Review Data Format** (from HF dataset `original_reviews` column):
Each paper has a JSON list of review dicts with keys: `summary`, `strengths`, `weaknesses`, `questions`, `rating`, `confidence`, `soundness`, `presentation`, `contribution`, etc. Reviews are real OpenReview submissions.

**Files**:
- `cluster_reviews.py` — Loads GT reviews from the HF Arrow dataset via `datasets.load_from_disk()`, filters to test-set papers by submission ID, parses individual reviews from `original_reviews` JSON, embeds with sentence-transformers, clusters with K-Means, labels clusters by keyword frequency + avg rating + weakness/strength ratio. Saves: `gt_reviews_clustered.json`, `gt_embeddings.npy`, `kmeans_model.pkl`, `cluster_info.json`.
- `analyze.py` — t-SNE visualization of clustered reviews, archetype distribution histogram, cluster characteristics summary.

**Plots**:
- `review_embeddings_tsne.png` — t-SNE visualization of human reviews colored by cluster
- `archetype_distribution.png` — Bar chart showing frequency of each archetype

**Run**:
```bash
python 2_8_26/q1_reviewer_archetypes/cluster_reviews.py
python 2_8_26/q1_reviewer_archetypes/analyze.py
```

---

## Reused Infrastructure

All experiments reuse these existing scripts (no modifications needed):

| Script | Location | Used By |
|--------|----------|---------|
| `vllm_infer_ensemble.py` | `inference_scaling/scripts/` | H1, H2, B1, B2 |
| `extract_results.py` | `inference_scaling/scripts/` | H1, H2, B1, B2 |
| `run_metareview.py` | `inference_scaling/scripts/` | B1, B2 (Strategy D) |
| `gemini_title_infer.py` | `inference_scaling/scripts/` | H2 (Gemini title-only) |

---

## Models

| Model | Template | Used By | Context |
|-------|----------|---------|---------|
| `Qwen/Qwen2.5-7B` | `default` | H1 | 20480 + YaRN 4x |
| `Qwen/Qwen2.5-7B-Instruct` | `qwen` | H2, B1 (clean), B2 (clean) | 20480 + YaRN 4x |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen2_vl` | B1 (vision), B2 (vision) | 20480 |
| `gemini-2.5-flash` | N/A (batch API) | H2 (title-only) | N/A |

---

## Datasets

| Dataset | Source | Used By |
|---------|--------|---------|
| `*_clean_binary_noreviews_v7_test` | Base data dir | H1, H2, B2 |
| `*_clean_images_binary_noreviews_v7_test` | Base data dir | B2 |
| `*_vision_binary_noreviews_v7_test` | Base data dir | B2 |
| `*_v7_test_new_fewshot` or `*_v7_test_standard` | v7 split / B2 generated | B1 |
| `*_test_base_model` | Generated by H1 | H1 |
| `*_test_{title_only,title_abstract,...}` | Generated by H2 | H2 |
| `*_v7_test_{critical,enthusiastic,standard}` | Generated by B2 | B2 |
| `hf_dataset_new8_noref_cropped_2017_2026_with_decisions` | HF Arrow dataset (sk7524) | Q1 |

---

## Execution Phases

```
Phase 1 — Generate Datasets (parallel, ~5 min each):
    python 2_8_26/h1_base_model/generate_dataset.py
    python 2_8_26/h2_contamination/generate_ablation_datasets.py
    python 2_8_26/b2_role_prompts/generate_role_datasets.py

Phase 1 — Submit Inference (parallel, ~8-16 hrs each):
    sbatch 2_8_26/h1_base_model/run_inference.sbatch        # 1 job
    sbatch 2_8_26/h2_contamination/run_ablation.sbatch      # 5 jobs (array 0-4)
    sbatch 2_8_26/b2_role_prompts/run_role_inference.sbatch  # 9 jobs (array 0-8)
    sbatch 2_8_26/b1_pdr/run_pdr.sbatch                     # 3 jobs (array 0-2)

Phase 2 — Strategy D (needs critical + enthusiastic from Phase 1):
    sbatch 2_8_26/b2_role_prompts/run_strategy_d.sbatch      # 3 jobs (array 0-2)

Phase 3 — Analysis (needs inference results):
    python 2_8_26/h1_base_model/analyze.py
    python 2_8_26/h2_contamination/analyze.py
    python 2_8_26/b1_pdr/analyze.py
    python 2_8_26/b2_role_prompts/analyze.py
    python 2_8_26/c1_bayesian/analyze.py                          # Uses B2 standard/clean by default
    python 2_8_26/q1_reviewer_archetypes/cluster_reviews.py
    python 2_8_26/q1_reviewer_archetypes/classify_generated.py --predictions <predictions>
    python 2_8_26/q1_reviewer_archetypes/analyze.py
```

---

## Plots Summary

| Experiment | Plot File | Description |
|------------|-----------|-------------|
| H1 | `acceptance_rate_base_vs_instruct.png` | Bar: acceptance rates comparison |
| H1 | `accuracy_base_vs_instruct.png` | Bar: accuracy comparison |
| H1 | `confusion_base.png` / `confusion_instruct.png` | Confusion matrices |
| H1 | `venn_agreement.png` | Venn diagram: base correct vs instruct correct regions |
| H1 | `disagreement_analysis.png` | Disagreement breakdown by type × GT + pie chart |
| H2 | `accuracy_vs_content_level.png` | Line+bar: accuracy at each content level |
| H2 | `acceptance_rate_by_level.png` | Bar: acceptance rate per level |
| H2 | `abstract_similarity_distribution.png` | Histogram: generated vs true abstract similarity with examples |
| H2 | `similarity_vs_citations.png` | Scatter: similarity vs citation percentile |
| H2 | `similarity_vs_rating.png` | Scatter: similarity vs avg reviewer score |
| B1 | `strategy_comparison.png` | Grouped bar: single vs majority vs PDR |
| B1 | `strategy_agreement.png` | Heatmap: pairwise strategy agreement |
| B2 | `acceptance_rate_by_role.png` | Grouped bar: role × modality |
| B2 | `strategy_d_comparison.png` | Per-modality: all roles + Strategy D |
| C1 | `roc_curve.png` | Accuracy vs threshold curve |
| C1 | `confusion_before_after.png` | Side-by-side confusion matrices |
| Q1 | `review_embeddings_tsne.png` | t-SNE: human vs generated reviews |
| Q1 | `archetype_distribution.png` | Histogram: archetype frequency comparison |

---

## Verification Checklist

For each experiment after running:

1. **Dataset**: `wc -l <data_dir>/*/data.json` — check sample counts match expectations
2. **Inference**: `wc -l <results_dir>/predictions.jsonl` — should match dataset size
3. **Extraction**: `wc -l <results_dir>/results_single.jsonl` — should match predictions
4. **Metrics**: Check `<metrics_dir>/*.json` has non-zero values
5. **Plots**: Visually verify plots in `<metrics_dir>/`
6. **Sanity**: Acceptance rate should vary across experiments (not all 95%+)

---

## Current Status (2026-02-10)

| Experiment | Data | Inference | Analysis | Notes |
|------------|------|-----------|----------|-------|
| **H1** | 2024 samples | 2024 base predictions (2011 valid) | Metrics + plots done | Uses B2 standard/clean as instruct baseline (same v7 split) |
| **H2** | 5 levels × 2024 | All 5 levels complete | Content level metrics done | `title_only` generates abstracts; similarity analysis requires `sentence-transformers` |
| **B1** | **NEEDS RERUN** | `clean` has only 10 samples (stale v6 data) | Blocked on clean | Fixed sbatch to use v7; resubmit `sbatch --array=0 2_8_26/b1_pdr/run_pdr.sbatch` |
| **B2** | 9 datasets (3×3) all present | Missing: `vision/critical`, `vision/standard`, `vision/strategy_d` | Partial | Resubmit: `sbatch --array=6,8 run_role_inference.sbatch`, then `sbatch --array=2 run_strategy_d.sbatch` |
| **C1** | N/A (uses other results) | N/A | **Ready to run** | `python 2_8_26/c1_bayesian/analyze.py` (defaults to B2 standard/clean) |
| **Q1** | HF dataset + test IDs | N/A (clustering) | Not yet run | `python 2_8_26/q1_reviewer_archetypes/cluster_reviews.py` |

### Immediate Next Steps
```bash
# 1. Fix B1 clean (rerun with v7 data)
sbatch --array=0 2_8_26/b1_pdr/run_pdr.sbatch

# 2. Fix B2 vision missing roles
sbatch --array=6,8 2_8_26/b2_role_prompts/run_role_inference.sbatch

# 3. After B2 vision/critical + vision/enthusiastic complete:
sbatch --array=2 2_8_26/b2_role_prompts/run_strategy_d.sbatch

# 4. Run C1 analysis (no inference needed)
python 2_8_26/c1_bayesian/analyze.py

# 5. Run H2 similarity analysis (needs sentence-transformers)
python 2_8_26/h2_contamination/analyze.py

# 6. Run H1 with updated Venn diagram
python 2_8_26/h1_base_model/analyze.py
```
