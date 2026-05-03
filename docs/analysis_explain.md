# Analysis Handoff — `docs/objective_analysis.md`

This file is a self-contained guide for any agent who wants to understand,
verify, or extend the analysis in `docs/objective_analysis.md`. It documents:
where every input file lives, how the result jsonls are structured, what the
scoring conventions are, how each metric is computed, and how to add a new
model/dataset/section.

If you want to just re-run everything: see "Quick start" below. If you want
to extend it: see "Where to add things" at the bottom.

---

## 1. What the analysis covers

Two objectives, evaluated across model size × modality × train ratio × train
distribution × test distribution:

1. **General quality indicator** — does the model score track underlying
   paper quality? Metrics: AUC + Spearman ρ to `pct_rating` and
   `citation_normalized_by_year` (year-filtered).
2. **Conference acceptor** — does the binary decision match the venue's
   call? Metrics: balanced accuracy + accept-recall + reject-recall.

The doc has 8 sections; see `docs/objective_analysis.md` for the narrative.
This handoff focuses on the *plumbing*.

---

## 2. Quick start

```bash
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer

# Regenerate the doc + every figure (~3-5 minutes)
uv run python scripts/tmp_latex_dir/generate_objective_analysis.py

# Verify all 438 statistical checks (~2 minutes)
uv run python scripts/verify_objective_analysis.py
```

The generator reads result jsonls + metadata, computes everything in-memory,
writes `docs/objective_analysis.md` and `tmp_latex_dir/figures/objective_*.{pdf,png}`.
The verifier independently re-derives every number against sklearn/scipy
reference implementations and the values printed in the doc itself.

`uv run` is required because matplotlib + Pillow need a libtiff that's only
in the project's `.venv`. Running plain `python` will fail with libtiff errors.

---

## 3. File map

### 3.1 Generator + verifier (the only two scripts you usually touch)

| File | Purpose |
|---|---|
| `scripts/tmp_latex_dir/generate_objective_analysis.py` | Produces `docs/objective_analysis.md` + all figures |
| `scripts/verify_objective_analysis.py` | Audits every metric vs sklearn/scipy/doc |

### 3.2 Output

| File | Purpose |
|---|---|
| `docs/objective_analysis.md` | The published analysis |
| `docs/objective_analysis_figures/*.png` | PNG copies for embedded display |
| `tmp_latex_dir/figures/objective_*.{pdf,png}` | Canonical figures (PDF for LaTeX, PNG for markdown) |

### 3.3 Helper scripts referenced by the generator (do not need to touch)

| File | Purpose |
|---|---|
| `scripts/ratio_xeval_consolidated.py` | Earlier sibling script — has the same model registry + loaders. Source for path conventions. |
| `scripts/iclr_arxiv_calibrated_eval.py` | Original calibration logic. |
| `scripts/ratio_xeval_per_venue_year_quality.py` | Per-(venue, year) quality decomposition. Generator's §6 uses similar logic inline. |
| `scripts/ratio_xeval_quality_distribution_shift.py` | Distribution shift figure logic. Generator's §2.3 uses similar logic inline. |

### 3.4 Source reports the analysis builds on

| File | Provides |
|---|---|
| `reports/test-ratio-and-metrics.md` | Established the metric framework (prior-invariance, balanced ACC vs raw ACC, deriving raw_ACC(π) from per-class recalls) |
| `reports/ratio_xeval_7b.md` | The 7B ICLR-trained ratio cross-eval baseline numbers |
| `reports/balanced_eval_2026-05-02.md` | The arxiv-trained checkpoint sweep (3B + 7B, balanced + natrate) — source for §7 |
| `/scratch/gpfs/ZHUANGL/sk7524/Researcher/RESULTS_deepreviewer_balanced.md` | DeepReviewer-14B baseline results — source for §5 |

---

## 4. Data layout

### 4.1 Test/val sets (`data/<dataset_name>/data.json`)

Each is a JSON list of records. Each record has `conversations` (the
ShareGPT-format chat turns) plus `_metadata` (the labels and structured
fields). The generator reads `_metadata` only.

#### ICLR test sets

```
data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json
data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation/data.json
data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json
data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_validation/data.json
data/iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_test/data.json
data/iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_validation_DERIVED/data.json
data/iclr_2020_2023_2025_2026_30_70_original_vision_v7_filtered_test/data.json
data/iclr_2020_2023_2025_2026_30_70_original_vision_v7_filtered_validation_DERIVED/data.json
```

Naming: `iclr_<year_range>_<train_ratio>_<modality>_<filtering>_<split>`
- `train_ratio` = `85_5_10_balanced` (50/50 accept/reject) OR `30_70` (natural-prior)
- `modality` = `text` (paper text only) OR `vision` (text + page images)
- `<test, validation>` = the split
- `_filtered` / `_labelfix_v7` / `_filtered24480` are pipeline cleanup flags
- `_DERIVED` means the val split was constructed post-hoc from the test split's reject pool

ICLR `_metadata` fields:
```python
{
    "submission_id": "zpENPcQSj1",       # OpenReview submission ID
    "answer": "Accept",                   # binary label — what we use as gold
    "year": 2025,                         # conference year
    "ratings": [5, 6, 8],                 # per-reviewer scores
    "decision": "poster",                 # finer-grained decision (poster/spotlight/oral/accept/reject)
    "pct_rating": 0.7623,                 # percentile of avg rating within year
    "conference": "iclr",
    "authors": "[\"Changnan Xiao\", \"Bing Liu\"]",
    "citation": 0,                        # raw citation count
    "citation_normalized_by_year": 0.295  # percentile of citation_per_year within year
}
```

**Critical caveat**: `citation_normalized_by_year` is *degenerate for 2026
papers* because they have raw citation = 0 → all percentiles collapse to
0.500. Always year-filter to 2025 only when computing citation-quality
metrics on ICLR.

#### Arxiv test sets

```
data/arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test/data.json
data/arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation/data.json
data/arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_test/data.json
data/arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_validation/data.json
data/arxiv_natrate_21k_text_wmetadata_filtered24480_y24up_test/data.json
data/arxiv_natrate_21k_text_wmetadata_filtered24480_y24up_validation/data.json
data/arxiv_natrate_21k_vision_wmetadata_filtered24480_y24up_test/data.json
data/arxiv_natrate_21k_vision_wmetadata_filtered24480_y24up_validation/data.json
```

Naming: `arxiv_<train_prior>_21k_<modality>_wmetadata_<filtering>_y24up_<split>`
- `train_prior` = `50_50` (balanced) OR `natrate` (per-conference natural acceptance rate)
- `y24up` = filtered to conference_year ≥ 2024 (2024+2025+2026 papers)

Arxiv `_metadata` fields:
```python
{
    "arxiv_id": "2410.05317",
    "answer": "Accept",                   # binary label
    "venue": "iclr",                      # canonical venue lowercase
    "pl_venue": "iclr",                   # PaperLens-resolved venue (preferred)
    "conference_year": 2025,              # year of the conference (different from arxiv_year)
    "arxiv_year": 2024,
    "decision": "accept",                 # binary in arxiv set (collapsed from venue-specific tiers)
    "training_label": "accept",
    "title": "Accelerating Diffusion Transformers with Token-wise Feature Caching",
    "submission_date": ...,
    "body_pages": 12,
    "categories": ...,
    "pct_rating": 0.78,                   # SPARSE — only ~290/1414 have it
    "pct_citation": 0.62,                 # SPARSE — only ~340/1414 have it
    "pl_openreview": "https://openreview.net/forum?id=yYZbZGo4ei",  # SPARSE — only ICLR papers have it
    "pl_keywords": [...],
    "pl_primary_area": ...,
    "pl_year": 2025
}
```

Venues in the arxiv y24up balanced set: `iclr, neurips, cvpr, acl, aaai,
icml, iccv, eccv, colm, aistats, corl`. Per-(venue, year) cell sizes vary
from 21 to ~210 papers.

### 4.2 Inference jsonls (`results/.../<short>/<dataset_subdir>/finetuned-ckpt-<step>.jsonl`)

One row per evaluated paper. Critical fields:
```python
{
    "prompt": "<chat-template-formatted input>",
    "predict": "Outcome: \\boxed{Accept}",   # or \\boxed{Reject} — model generation
    "label": "Outcome: \\boxed{Accept}",     # gold answer
    "token_logprobs": [-0.12, -0.05, ..., -0.01],  # log-prob of EACH generated token (length 6-8)
    "logprob_accept": [-0.12, ..., -1.31],   # log-prob of "Accept" token at each position (only in newer evals)
    "logprob_reject": [-0.45, ..., -3.45]    # log-prob of "Reject" token at each position (only in newer evals)
}
```

The `Outcome: \boxed{<token>}` template means the Accept/Reject token sits at
**index 5** in `token_logprobs` (positions 0-4 are `Outcome`, `:`, ` `,
`\boxed`, `{`). Older jsonls only have `token_logprobs`; newer ones
additionally have `logprob_accept` and `logprob_reject` arrays of the same
length (giving you the counterfactual "what if the model had emitted
Accept/Reject at each position").

### 4.3 ICLR-trained checkpoint inference (`results/final_sweep_v7_datasweepv3/optim_search_2026/`)

The 7B 2nd-checkpoint models we use as the headline grid:

| Model | Cell directory | Ckpt | Notes |
|---|---|---:|---|
| 7B text 50/50 | `bz32_lr1e-6_text` | 1322 | balanced ICLR train |
| 7B text 30/70 | `bz32_lr1e-6_text_30_70` | 1322 | natural-prior ICLR train |
| 7B vision 50/50 | `bz16_lr1e-6_vision` | 2648 | balanced ICLR train |
| 7B vision 30/70 | `bz16_lr1e-6_vision_30_70` | 2642 | natural-prior ICLR train |
| 3B text 50/50 | `scaling/bz32_lr1e-6_text_3b` | 2644 | last ckpt |
| 3B vision 50/50 | `scaling/bz16_lr1e-6_vision_3b` | 5296 | last ckpt |

Per-cell inference paths (test/val × balanced/natural):
```
# Test, balanced (in-domain for 50/50 train, cross-domain for 30/70 train)
results/.../optim_search_2026/<cell>/finetuned-ckpt-<step>.jsonl                              # for 50/50 cells
results/.../optim_search_2026/ratio_sweep/<cell>/balanced/finetuned-ckpt-<step>.jsonl         # for 30/70 cells eval'd on balanced

# Test, natural (30/70 prior)
results/.../optim_search_2026/ratio_crossval_clean/<cell>/test_30_70/finetuned-ckpt-<step>.jsonl  # for 50/50 cells eval'd on natural
results/.../optim_search_2026/ratio_sweep/<cell>/30_70/finetuned-ckpt-<step>.jsonl                # for 30/70 cells

# Validation
results/iclr_val_calib/<cell>/val_balanced/finetuned-ckpt-<step>.jsonl
results/iclr_val_calib/<cell>/val_30_70/finetuned-ckpt-<step>.jsonl

# Cross-conference arxiv evaluation
results/cross_conference_arxiv_y24up/<cell>/arxiv_eval/finetuned-ckpt-<step>.jsonl
results/cross_conference_arxiv_y24up/<cell>/arxiv_val/finetuned-ckpt-<step>.jsonl
results/cross_conference_arxiv_natrate_y24up/<cell>/arxiv_eval/finetuned-ckpt-<step>.jsonl
results/cross_conference_arxiv_natrate_y24up/<cell>/arxiv_val/finetuned-ckpt-<step>.jsonl
```

The path resolution functions in the generator (`iclr_test_jsonl`,
`iclr_val_jsonl`, `arxiv_jsonl`) handle this — see lines 200-225 of
`generate_objective_analysis.py`.

### 4.4 Arxiv-trained checkpoint inference (`results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/`)

Per-epoch checkpoints from the source sweep. Eight datasets per cell
(arxiv/iclr × balanced/natrate × test/val):

| Cell | Modality | Train data | Sizes/checkpoints |
|---|---|---|---|
| `small/arxiv_21k_text` | text | 21k arxiv balanced | 7B; ckpts 656, 1312, 1968, 2624 (4 epochs) |
| `small/arxiv_21k_vision` | vision | 21k arxiv balanced | 7B; ckpts 1309, 2618 + partial 3927/5236 |
| `small_sachin/arxiv_21k_text_3b` | text | 21k arxiv balanced | 3B; ckpts 656, 1312, 1968, 2624 |
| `natrate_sachin/arxiv_natrate_21k_text` | text | 21k arxiv natrate | 7B; ckpts 656, 1312 + ckpts 1968, 2624 in flight |
| `natrate_sachin/arxiv_natrate_21k_text_3b` | text | 21k arxiv natrate | 3B; ckpts 656, 1312, 1968, 2624 |
| `natrate_sachin/arxiv_natrate_21k_vision` | vision | 21k arxiv natrate | not yet run |
| `large/arxiv_balanced_per_venue_text` | text | larger split | not yet evaluated |
| `large/arxiv_balanced_per_venue_vision` | vision | larger split | not yet evaluated |

Per-cell paths look like:
```
results/.../final_data_sweep_v3/arxiv_train/<cell>/<dataset_subdir>/finetuned-ckpt-<step>.jsonl
results/.../final_data_sweep_v3/arxiv_train/<cell>/<dataset_subdir>/finetuned-ckpt-<step>-gpu-test.jsonl
```

Where `<dataset_subdir>` is one of:
`arxiv_balanced_test`, `arxiv_balanced_val`, `arxiv_natrate_test`,
`arxiv_natrate_val`, `iclr_balanced_test`, `iclr_balanced_val`,
`iclr_natrate_test`, `iclr_natrate_val`.

The `-gpu-test` suffix exists because two parallel inference jobs (PLI
auto-watcher + gputest array) write to the same dir; the suffix prevents
overwrite. The loader (`load_arxiv_trained_jsonl`) tries the plain name first
then falls back to `-gpu-test`.

### 4.5 DeepReviewer-14B baseline (`/scratch/gpfs/ZHUANGL/sk7524/Researcher/`)

This is an external repo; **do not write into it**. We just consume its
outputs.

```
/scratch/gpfs/ZHUANGL/sk7524/Researcher/results/deepreviewer-14b-standard/
    ├── arxiv_balanced_test/deepreviewer-14b-standard.jsonl
    ├── arxiv_balanced_val/deepreviewer-14b-standard.jsonl
    ├── arxiv_natrate_test/deepreviewer-14b-standard.jsonl
    ├── arxiv_natrate_val/deepreviewer-14b-standard.jsonl
    ├── iclr_balanced_test/deepreviewer-14b-standard.jsonl
    ├── iclr_balanced_val/deepreviewer-14b-standard.jsonl
    ├── iclr_natrate_test/deepreviewer-14b-standard.jsonl
    └── iclr_natrate_val/deepreviewer-14b-standard.jsonl

/scratch/gpfs/ZHUANGL/sk7524/Researcher/subsamples/
    ├── arxiv_balanced_test_q4_seed42.json     # JSON list of integer row indices into our text data.json
    ├── arxiv_balanced_test_q4_seed42.json     # vision subsample (different paper set)
    ├── iclr_balanced_test_q4_seed42.json
    └── ...                                    # (val + natrate variants)
```

DeepReviewer was eval'd on a **stratified 1/4 subsample** of our balanced
test sets (stratified by `(year, label)` for ICLR, `(venue, label)` for
arxiv). The subsample indices are saved as JSON lists of row positions into
our `data/<dataset>/data.json` files. To compare apples-to-apples, restrict
our model predictions to the same row indices.

DeepReviewer jsonl row schema:
```python
{
    "idx": 3,                               # row index in source data.json
    "tag": "iclr_balanced_test",            # dataset
    "dataset": "iclr_2020_2023_..._test",   # full source dataset name
    "label": "Accept",                      # gold (parsed from \boxed{...})
    "label_raw": "Outcome: \\boxed{Accept}",
    "predict_decision": "Accept",           # "Accept" / "Reject" / "" if truncated
    "predict_meta_rating": 7.0,             # meta-reviewer rating, 1.0-10.0 — use as confidence
    "predict_reviewer_ratings": [6.0, 6.0, 8.0, 8.0],  # per-reviewer ratings
    "parsed": {...},                        # structured parse of the agent's output
    "metadata": {...}                       # full original _metadata (often serialized as repr-string)
}
```

`metadata` is sometimes a string (the repr of a dict) and sometimes a real
dict. Use `ast.literal_eval` to handle both:
```python
import ast
m = r["metadata"]
meta = ast.literal_eval(m) if isinstance(m, str) else (m or {})
```

The loader (`load_deepreviewer`) handles this and also drops rows where
`predict_decision` is empty (truncated outputs, ~5-10% per dataset).

---

## 5. Scoring conventions

The single-paper "score" is what we use as the continuous predictor for
AUC, Spearman ρ, and threshold calibration. **Two scoring functions exist**;
which one we use depends on which fields the jsonl has:

### 5.1 `score_logodds(row)` — used for ICLR-trained ckpts

```python
def score_logodds(row):
    chosen = 1 if "Accept" in row["predict"] else (0 if "Reject" in row["predict"] else None)
    lp = row["token_logprobs"][5]                         # decision token position
    p_chosen = exp(lp)                                    # prob the model assigned to its chosen token
    logit = log(p_chosen / (1 - p_chosen))                # always >= 0
    return +logit if chosen == 1 else -logit              # signed: + iff predicted Accept
```

This is **signed log-odds with sign from the prediction**. Score > 0 ↔
predicted Accept. The magnitude reflects model confidence on its chosen
class. **Limitation**: this is the marginal probability of the chosen token
(treats P(other) as 1 - P(chosen)), which conflates Accept/Reject vs *any
other token*.

### 5.2 `score_logodds_2class(row)` — used for arxiv-trained ckpts (newer jsonls)

```python
def score_logodds_2class(row):
    la = row["logprob_accept"]; lr = row["logprob_reject"]
    diffs = [abs(a - b) for a, b in zip(la, lr)]          # find decision position
    pos = argmax(diffs)
    return la[pos] - lr[pos]                              # log P(Accept) - log P(Reject) under 2-class softmax
```

This matches the source-report convention in
`reports/balanced_eval_2026-05-02.md`. It uses a **2-class softmax** between
Accept and Reject only (ignoring other tokens), which is more theoretically
correct for a binary classifier. Falls back to `score_logodds` if
`logprob_accept`/`reject` are absent.

The two scores agree to within ~1pp on bACC across the headline cells, but
`score_logodds_2class` is the right choice when the field is available.

---

## 6. Threshold calibration — two flavors

A "threshold" τ converts a continuous score into a binary prediction
(predict Accept iff score > τ). We compute τ on val and apply it to test.
Two objectives, both implemented:

### 6.1 `best_tau_raw(val_pairs)` — argmax raw accuracy

```python
def best_tau_raw(pairs):
    # Sort by score, sweep through unique scores, track raw accuracy
    # Returns the τ achieving max raw ACC on val
```

Used in §1.3 for the comparison and to demonstrate that on a balanced val,
this matches `best_tau_balanced` exactly.

### 6.2 `best_tau_balanced(val_pairs)` — argmax balanced accuracy

```python
def best_tau_balanced(pairs):
    # Same sweep but track (acc-recall + rej-recall)/2 instead
```

Used everywhere else in the doc when calibration is applied (§7 for
arxiv-trained ckpts). Preferred because it is robust to val/test prior
mismatch.

When val prior = test prior (both balanced 50/50), both τ functions return
essentially the same value. They diverge when val is at the natural prior
(30/70) — see §1.3 figure where the two curves clearly separate.

Both functions handle tied scores correctly via groupby. Verified against
brute-force sweep in `verify_objective_analysis.py`.

---

## 7. Metric implementations

Every metric is verified against sklearn/scipy reference implementations
in `verify_objective_analysis.py`. The implementations are pure-Python (no
sklearn/scipy dep at runtime):

### 7.1 AUC (`auc_score`)

Mann-Whitney U with **average-rank tie handling** (matches
`sklearn.metrics.roc_auc_score` exactly):

```python
def auc_score(scores, labels):
    n = len(scores); npos = sum(labels); nneg = n - npos
    # Average-rank for ties (rankdata-style)
    indexed = sorted(enumerate(scores), key=lambda p: p[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]: j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1): ranks[indexed[k][0]] = avg_rank
        i = j + 1
    rs = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (rs - npos * (npos + 1) / 2) / (npos * nneg)
```

This is the **second** version of `auc_score`; the first version did
positional ranking (no tie averaging) and produced AUCs that differed from
sklearn by 1e-5 to 1e-3 on cells with many tied scores. Fixed in commit
`2319ee52`.

### 7.2 Balanced ACC, raw ACC, per-class recall

Direct counting:

```python
def per_class_recall(pairs, tau):
    n_acc = sum(1 for _, g in pairs if g == 1)
    n_rej = sum(1 for _, g in pairs if g == 0)
    tp = sum(1 for s, g in pairs if g == 1 and s > tau)
    tn = sum(1 for s, g in pairs if g == 0 and s <= tau)
    return tp/n_acc * 100, tn/n_rej * 100

def bal_acc(pairs, tau):
    a, r = per_class_recall(pairs, tau)
    return (a + r) / 2

def raw_acc(pairs, tau):
    return sum(1 for s, g in pairs if (s > tau) == (g == 1)) / len(pairs) * 100
```

Verified to match `sklearn.metrics.{balanced_accuracy_score, accuracy_score,
recall_score}` to 1e-6.

### 7.3 Spearman ρ (`spearman`)

Average-rank correlation:

```python
def spearman(xs, ys):
    rx = _rank(xs); ry = _rank(ys)        # average-rank for ties
    mx = mean(rx); my = mean(ry)
    return sum((rx[i]-mx)*(ry[i]-my)) / (sqrt(sum_sq_dev_x) * sqrt(sum_sq_dev_y))
```

Verified to match `scipy.stats.spearmanr` to 1e-9 across hundreds of cells.

### 7.4 Bootstrap CI (`bootstrap_metric`)

```python
def bootstrap_metric(items, metric_fn, n=500, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(items), len(items))    # paper-level resampling with replacement
        sample = [items[i] for i in idx]
        v = metric_fn(sample)
        if v is not None: vals.append(v)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)
```

500 paper-resamples, percentile method, seed=42 for reproducibility. The
CIs are **paper-level** (not row-level) — this is the right grain for
"how reliable is this point estimate".

A separate `bootstrap_ci` exists with the same logic but takes (xs, ys)
instead of items + metric_fn — used for Spearman ρ CIs.

---

## 8. The doc structure (which functions produce which sections)

| Section | Topic | Generator function |
|---|---|---|
| TL;DR | 8-bullet summary | inline in `write_doc` |
| §1.1 | Prior-invariant metric inventory | inline (static table) |
| §1.2 | Why raw ACC misleads | inline; pulls from `compute_all_7b()` natural-prior cells |
| §1.3 | Dual-calibration thresholds | inline; uses `best_tau_raw` + `best_tau_balanced` on val/test |
| §1.4 | Natural-ACC formula derivation | inline; π·AccR + (1-π)·RejR vs empirical |
| §2.3 | Distribution shift figure | `figure_distribution_shift()` |
| §2.4 | Quality ρ — per-class decomposition | inline; uses `quality_corr` from `compute_all_7b()` |
| §2.5 | Bootstrap CIs on recommended config | uses `bootstrap_ci` |
| §3 | Conference acceptor headline tables | inline; uses `per_cell` from `compute_all_7b()` |
| §4 | Comprehensive recommendation | inline; uses `quality_corr` + `per_cell` for "best" detection |
| §5 | DeepReviewer comparison | `compute_deepreviewer()` + `figure_deepreviewer_comparison()` |
| §6 | Per-(venue, year) tracking | `compute_per_venue_year_arxiv()` + `compute_per_year_iclr()` + `figure_per_venue_year()` |
| §7 | Arxiv-trained checkpoint sweep | `compute_arxiv_trained_sweep()` + `figure_arxiv_trained_sweep()` |
| §8 | Dataset integrity case study | `compute_integrity_case_study()` |

---

## 9. Where to add things

### 9.1 New ICLR-trained checkpoint

Edit `MODELS_7B_ICLR` (or `MODELS_3B_ICLR`) in
`generate_objective_analysis.py`:

```python
MODELS_7B_ICLR = {
    ("text",   "50_50"): ("bz32_lr1e-6_text",         1322),
    # ... add new entry:
    ("vision", "30_70_v2"): ("bz16_lr1e-6_vision_30_70_v2", 2700),
}
```

The path resolvers (`iclr_test_jsonl`, `arxiv_jsonl`) use `train_ratio`
to decide the subdirectory. If your new ckpt uses a different convention,
extend those functions.

### 9.2 New arxiv-trained cell

Edit `ARXIV_TRAINED_CELLS` in `generate_objective_analysis.py`:

```python
ARXIV_TRAINED_CELLS = [
    # ... add new entry:
    ("3B large balanced text", "text", "balanced", "3B",
     "large/arxiv_balanced_per_venue_text",
     {1: 4431, 2: 8862, 3: 13293}),  # epoch -> ckpt step
]
```

The format is: `(display_name, modality, train_data, model_size,
cell_dir_relative_to_ARXIV_TRAIN_BASE, {epoch: ckpt_step})`.

### 9.3 New baseline (à la DeepReviewer)

Pattern from `compute_deepreviewer()`:
1. Add a `load_<name>` function that reads the new baseline's jsonl.
2. Add a `compute_<name>()` function that:
   - Loads test + val rows
   - Computes scoring (depends on what fields the baseline emits)
   - Calibrates if needed (`dr_calibrated_threshold` is the template)
   - Returns a metric stack
3. Wire it into `main()` and pass to `write_doc`.
4. Add a section to `write_doc` to display the comparison.

If the baseline was eval'd on a subsample, also create a JSON of row indices
into our `data/<dataset>/data.json` so we can subsample our own predictions
to match — see `Researcher/subsamples/*_q4_seed42.json` for the format.

### 9.4 New dataset (e.g., a different conference)

1. Add the data.json under `data/<new_dataset>/data.json` with the
   `_metadata` schema described in §4.1.
2. Add an entry to `ICLR_META` or `ARXIV_META` mapping
   `(modality, prior, split)` → path.
3. Add a path resolver function (mirror `iclr_test_jsonl` or `arxiv_jsonl`).
4. Run inference on your test/val splits using the existing inference
   pipeline (`scripts/vllm_infer.py` or the sbatch jobs in `sbatch/inference/`).

### 9.5 New section in the doc

Add a `compute_<section>()` function and pass its output to `write_doc`.
The doc is built by appending to a `lines` list — just add your section
between two existing sections, or before the appendix.

---

## 10. Verification protocol

`verify_objective_analysis.py` parses the published doc and re-derives every
number independently. Adding new content? Add a verification check too:

```python
section("§N my new metric — verify vs reference")
# Pull expected values from the doc table (parser handles "X.Y [lo, hi]" format)
row = extract_table_row("**My new section header**", f"| {model_label}")
bal_pt, bal_lo, bal_hi = parse_num(row[<col_idx>])

# Re-derive from raw data
pairs, _ = G.<your_loader>(...)
check_metric_stack(f"§N {model_label}", pairs, tau,
                   doc_bal=bal_pt, doc_auc=auc_pt, ...)
check_bootstrap(f"§N {model_label} CI", pairs, lambda p: G.bal_acc(p, tau),
                doc_lo=bal_lo, doc_hi=bal_hi)
```

If you don't add a verifier, your numbers are not actually verified.

---

## 11. Known caveats and open issues

These are flagged in the doc itself but worth a checklist for downstream work:

1. **2026 citation degeneracy** — `citation_normalized_by_year` for ICLR 2026
   papers is degenerately 0.500 because raw citations are all 0. Always
   year-filter to 2025 only when computing ρ to citation on ICLR.
2. **Arxiv quality fields are sparse** — `pct_rating` (n≈290) and
   `pct_citation` (n≈340) only cover ~20% of arxiv y24up. Bootstrap CIs on
   ρ here are wide; per-(venue, year) cells with n≥40 are limited to ~5.
3. **DeepReviewer subsample is small** — n=343 (arxiv) and n=399 (ICLR)
   after dropping truncated rows. Expect bACC CIs of ±5-10pp on this n.
4. **Arxiv-trained 7B vision iclr_balanced_test only has ckpt-1309** —
   ckpt-2618 (the best-arxiv ckpt) doesn't have iclr_test inference yet.
   §8 uses ckpt-1309 as the workaround.
5. **7B natrate text iclr OOD ep1 looks suspicious** in source report (very
   asymmetric per-class recall) — only ckpts 656+1312 done; re-evaluate when
   1968+2624 finish training.
6. **Paper-level integrity check between gold ICLR and arxiv-set ICLR-subset
   is impossible at scale** because only ~25 of the 87 arxiv ICLR-subset
   papers have `pl_openreview` filled in. §8 relies on aggregate-metric
   comparison instead.

---

## 12. Reproducibility checklist for downstream agents

Before publishing or extending:

- [ ] Run `uv run python scripts/verify_objective_analysis.py` — all 438
      checks must pass.
- [ ] If you modified the generator, re-run it and inspect the diff.
- [ ] If you added new tables, extend `verify_objective_analysis.py` to
      parse and verify them.
- [ ] All numbers in tables should be parseable (don't put tables in figures).
- [ ] Bootstrap CIs use seed=42 for reproducibility — don't change without
      reason.
- [ ] If you change `score_logodds` or `score_logodds_2class`, all numbers
      will shift slightly. Re-run the audit and update the doc.
- [ ] If you add a model trained on a new distribution, add a row to the
      "Where to add things" instructions and the §7-style sweep.

---

## 13. Canonical test datasets — one entry per dataset family

The repo contains **hundreds** of `data/<name>/data.json` directories from
many iterations of dataset construction. Most are dead. The current analysis
uses a small canonical set documented here.

### 13.1 ICLR canonical test/val sets

The ICLR datasets all derive from one source corpus: every ICLR submission
2020-2026 with parsed metadata + reviews from OpenReview. The named
suffixes describe how the raw corpus was filtered + balanced for training
and evaluation:

| Family | Path glob | What it is |
|---|---|---|
| **balanced (50/50) ICLR** | `data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_<text|vision>_*labelfix_v7_filtered_<test|validation>/data.json` | Training corpus split 85/5/10 train/val/test, then accepts and rejects subsampled to **exactly 50% accept rate within each year**. The test split is what we call "gold ICLR balanced" — 1,667 papers across years 2020-2026, with 836 from year ≥ 2025. |
| **natural (30/70) ICLR** | `data/iclr_2020_2023_2025_2026_30_70_original_<text|vision>_v7_filtered_<test|validation_DERIVED>/data.json` | Same source corpus but accept rate **forced to 30%** (matches ICLR's actual long-run acceptance rate). Test split is 1,594 papers; ~31.3% accept. The validation split was constructed post-hoc (`_DERIVED`) from the test split's reject pool. |
| **y25up subsamples** | `data/iclr_2020_2023_2025_2026_*_y25up_<test|validation>/data.json` | A pre-filtered version of the balanced/natural sets with year ≥ 2025 already applied. Same content as filtering the unrestricted version year-on-the-fly. We use y25up versions when running inference (saves time) but the in-place `year_keep={2025, 2026}` filter in our loaders works on either. |

**Key observation: balanced vs natural are different paper draws, not just
a relabeling**. Going from balanced (50/50) to natural (30/70) requires
*adding more rejects*, which means the natural set contains papers the
balanced set doesn't — and vice versa. The two are *not nested* (you can't
just discard accepts from balanced to get natural; the rejects come from a
different stratification). This matters when comparing per-class metrics
across the two: the per-class **distributions are essentially the same**
(both draw from the same accept-pool and reject-pool), but the **mixture
proportions differ**.

This is also why pooled metrics like Spearman ρ(score, pct_rating) shift
between balanced and natural: the marginal `pct_rating` distribution moves
because the accept/reject mixture moves. Per-class ρ is invariant; pooled
ρ is not.

### 13.2 Arxiv canonical test/val sets

The arxiv datasets come from a different pipeline: scraped arxiv papers
matched to peer-reviewed venues (ICLR, NeurIPS, CVPR, ACL, AAAI, ICML, ICCV,
ECCV, COLM, AISTATS, CORL). The accept/reject label comes from whether the
arxiv paper was eventually accepted to its target venue.

| Family | Path glob | What it is |
|---|---|---|
| **balanced (50/50) arxiv y24up** | `data/arxiv_50_50_21k_<text|vision>_wmetadata_filtered24480_y24up_<test|validation>/data.json` | 21k-paper training pool, balanced 50/50 within each (venue, year). y24up filter = conference_year ≥ 2024. Test n=1,414; val n=722. |
| **natural arxiv y24up** | `data/arxiv_natrate_21k_<text|vision>_wmetadata_filtered24480_y24up_<test|validation>/data.json` | Same y24up filter, but accepts/rejects sampled at each venue's **natural acceptance rate** (per openaccept.org 5-year averages). Different per-venue accept rate (e.g., NeurIPS ≈ 0.27, CVPR ≈ 0.25, ICLR ≈ 0.31). Pooled accept rate ≈ 23.8%. |

### 13.3 The `y25up` / `y24up` filter convention

Both families use a **year-cutoff filter** to focus on recent papers (the
test set we actually care about for current model performance):

- **`y25up`** — keep papers with `year ≥ 2025`. Used for ICLR (2025+2026 are
  the unseen-by-training years).
- **`y24up`** — keep papers with `conference_year ≥ 2024`. Used for arxiv
  (2024+2025+2026; arxiv has more uncertainty in publication dates so we go
  back one year).

**Why year-filter at all?** Training corpora include papers up to and
including 2023 (ICLR 2024 was the cutoff for some sweep iterations). To
measure generalization on truly held-out time periods, we evaluate on
`year ≥ 2025` only. The year filter is applied either:
- **Pre-filtered at the dataset level** (`_y25up_test` directory exists)
- **Applied on the fly in the loader** via `year_keep={2025, 2026}` argument

Both produce identical results.

### 13.4 ICLR balanced vs natural — when to use which

| Scenario | Use balanced | Use natural |
|---|---|---|
| Reporting headline bACC / AUC | ✅ (prior-invariant by construction) | ❌ (raw ACC misleading) |
| Reporting raw ACC under deployment prior | analytical: `π·AccR + (1-π)·RejR` from balanced | ✅ direct measurement |
| Validating the analytical formula | use both, compare predicted vs empirical | ✅ |
| Calibrating a threshold | depends on val/test prior match (see §6) | ✅ when val and test have same prior |
| Computing Spearman ρ to quality | use balanced for stable mixture | **not** for cross-prior comparison (mixture shifts) |
| Single-test-set deployment | ✅ (one set gives every metric we need) | optional, derivable from balanced |

The doc's §3 + §5 + §7 use **balanced** test as the canonical reporting
surface. **Natural** test is used only in §1.2 (to demonstrate the raw-ACC
exploit) and §1.4 (to verify the formula).

### 13.5 Other important datasets in `data/` (not used in objective_analysis)

These exist but are NOT used by `generate_objective_analysis.py`. Listed
for context if you're extending:

- `data/2017_2026_*_v8_*` — older v8 splits, deprecated
- `data/2017_2026_*split8*_v9_*` — v9 splits with different stratification
- `data/2020_2023_2025_train_2020_2026_valtest_*` — 2025 train + 2020-2026 val/test (used in early scaling experiments)
- `data/arxiv_*_y25up_*` — the y25up arxiv variant (drop year 2024)
- `data/arxiv_natrate_train_*` — training corpora for arxiv natrate cells
- `data/dataset_info.json` — registers dataset names → paths for LLaMA-Factory's `dataset` arg

The `data/dataset_info.json` is the canonical registry. Look there if you
need to find a dataset's full metadata or check what columns it exposes.

---

## 14. p(accept) — calculation and validity

The classifier's continuous prediction is `p(accept | paper)`. Three different
formulations exist in the codebase. They give nearly identical numbers in
practice but differ in correctness.

### 14.1 The three formulations

#### Formulation A: marginal-chosen-token (legacy)

```python
chosen = "Accept" if "Accept" in row["predict"] else "Reject"
p_chosen = exp(token_logprobs[5])           # confidence on chosen class
p_accept = p_chosen if chosen == "Accept" else 1 - p_chosen
```

This treats the **probability of the chosen token** as the marginal probability
under the binary classifier. It's what `score_logodds()` derives a
signed log-odds from. This is the **legacy convention** used by all earlier
calibration figures (`tmp_latex_dir/generate_calibration_*.py`).

**Why it's not perfectly valid**: the model has the entire vocabulary to
predict from. `exp(token_logprobs[5])` is `P(chosen_token | context)`,
not `P(Accept | {Accept, Reject})`. There may be other tokens with nonzero
probability that aren't Accept or Reject. But empirically, after SFT on the
"Outcome: \boxed{Accept|Reject}" template, those other tokens get
essentially-zero probability (~1e-4 cumulative), so the approximation is
tight.

**Why we still use it**: backward compatibility with older inference jsonls
that don't have `logprob_accept` / `logprob_reject` arrays.

#### Formulation B: 2-class softmax (current, theoretically correct)

```python
la = row["logprob_accept"]    # log P(Accept token at each position)
lr = row["logprob_reject"]    # log P(Reject token at each position)
pos = argmax(|la[i] - lr[i]|) # position with maximum class divergence
log_p_accept = la[pos] - logsumexp(la[pos], lr[pos])  # 2-class softmax
p_accept = exp(log_p_accept)
score = la[pos] - lr[pos]     # signed log-odds (used for AUC, threshold)
```

This is **conditional on the decision being either Accept or Reject** —
a true binary classifier. The `argmax` over positions handles cases where
the boxed{} template parsing is shifted (rare).

**Why it's correct**: explicitly normalizes over the two-class set, so
`p_accept + p_reject = 1` exactly.

**Used in**: `score_logodds_2class()` in
`generate_objective_analysis.py`. Source: `reports/balanced_eval_2026-05-02.md`.

#### Formulation C: greedy decode (pred_decision only)

```python
pred = "Accept" if "Accept" in row["predict"] else "Reject"
# No continuous score — just the binary decision the model emitted greedily
```

This is what DeepReviewer's `predict_decision` field gives us. It's
greedy decoding's output, no score. Useful for bACC/AR/RR but not for AUC
or Spearman.

### 14.2 Why the three agree to within ~0.5pp on bACC

For our SFT models on the boxed{} template:
1. Other tokens (anything besides `Accept` / `Reject`) get ~zero probability
   → P(chosen) ≈ P(Accept | {A, R}).
2. The model's chosen token matches the argmax under 2-class softmax in
   >99% of cases.
3. Greedy decode emits the same chosen token as argmax-softmax.

So the three formulations give:
- Same binary prediction in >99% of rows.
- Same score ranking (same AUC) in >99.9% of rows.
- Slightly different absolute scores (matters for calibration, not for
  rank-based metrics).

### 14.3 Validity checklist before using p(accept) for new analysis

If you're computing a new metric using p(accept):

- [ ] Confirm the model's `predict` field contains exactly one of `{Accept, Reject}`. Drop rows where it's neither (model emitted something else, ~0.1% of rows).
- [ ] Check for `logprob_accept`/`reject` arrays. If present, prefer Formulation B. If not, use Formulation A.
- [ ] If you need calibrated probabilities (not just ranking), apply Platt scaling on val (§17 below).
- [ ] If you're comparing across model checkpoints with different score scales, use rank-based metrics (AUC, Spearman) instead of absolute p(accept).

---

## 15. Legacy vs current logprob handling — token index conventions

Inference jsonls have evolved over time. The two regimes:

### 15.1 Legacy: `token_logprobs[5]` only

Old jsonls (pre-2026 sweeps, including ICLR-trained 7B models in
`results/final_sweep_v7_datasweepv3/optim_search_2026/`) only have
`token_logprobs` — the log-probability of *each generated token* in the
order they were produced.

For the chat template `"Outcome: \\boxed{Accept|Reject}"`:

| Token index | Token (concrete example) |
|---|---|
| 0 | `Outcome` |
| 1 | `:` |
| 2 | ` ` |
| 3 | `\\boxed` |
| 4 | `{` |
| **5** | **`Accept`** or **`Reject`** ← decision token |
| 6 | `}` |
| 7 | EOS |

The constant `DECISION_TOKEN_IDX = 5` is hardcoded across our codebase
(`scripts/calibration_posthoc.py`, `scripts/ratio_xeval_consolidated.py`,
all `tmp_latex_dir/generate_calibration_*.py`). **It is robust as long as
the chat template stays "Outcome: \boxed{...}"**. If the prompt template
changes (e.g., to "The decision is: Accept/Reject"), this index breaks.

How to compute p(accept) under legacy:
```python
DECISION_TOKEN_IDX = 5
lp = row["token_logprobs"][DECISION_TOKEN_IDX]  # log P(chosen token)
p_chosen = exp(lp)
chosen = "Accept" if "Accept" in row["predict"] else "Reject"
p_accept = p_chosen if chosen == "Accept" else 1 - p_chosen
```

### 15.2 Current: `logprob_accept` + `logprob_reject` arrays

Newer inference (added in `--save_logprobs` flag of `scripts/vllm_infer.py`,
used for arxiv-trained eval and DeepReviewer 2026 evals) saves **per-position
log-probability arrays** for both candidate tokens:

```python
{
    "token_logprobs": [-0.12, -0.05, ..., -1.31],          # 6-8 floats
    "logprob_accept": [-0.12, -0.05, ..., -1.31],          # same length, log P("Accept" | context_at_pos_i)
    "logprob_reject": [-0.45, -0.30, ..., -3.45]           # same length, log P("Reject" | context_at_pos_i)
}
```

Each entry is the conditional log-probability of emitting that *specific
token* at that position, given the prefix. So `logprob_accept[5]` =
log P(emit "Accept" at position 5 | "Outcome: \boxed{").

**Advantages over legacy**:
1. **Robust to template drift**: the decision position can be located
   dynamically by `argmax_i |logprob_accept[i] - logprob_reject[i]|` —
   the position with maximum class divergence.
2. **2-class softmax is well-defined**: `P(Accept | {A, R}) = sigmoid(la[pos] - lr[pos])`.
3. **None-entries OK**: vLLM emits `None` for positions where the candidate
   token wasn't in the top-k; just skip those positions.

How to compute p(accept) under the current convention:
```python
la = row["logprob_accept"]
lr = row["logprob_reject"]
diffs = [abs(a - b) if (a is not None and b is not None) else -1
         for a, b in zip(la, lr)]
pos = argmax(diffs)
score = la[pos] - lr[pos]                                # signed log-odds
p_accept = 1 / (1 + exp(-score))                         # 2-class sigmoid
```

`score_logodds_2class()` in `generate_objective_analysis.py` falls back to
the legacy formulation when `logprob_accept`/`reject` aren't present, so
the same generator works across both regimes.

### 15.3 When to expect each regime

| Inference run | Regime |
|---|---|
| ICLR-trained 7B/3B sweeps (everything in `optim_search_2026/`) | legacy |
| Arxiv-trained ckpt sweeps (everything in `final_data_sweep_v3/arxiv_train/`) | current (mixed: some have both, some have logprob arrays only) |
| DeepReviewer-14B baseline (external) | neither — DR emits `predict_decision` + `predict_meta_rating`, no logprobs |
| Anything you generate with `scripts/vllm_infer.py --save_logprobs` going forward | current |

---

## 16. Calibration procedures — three approaches

The doc mentions three calibration approaches. Here's how each works and
when to use it.

### 16.1 Threshold calibration on val (used in `objective_analysis.md`)

**Goal**: pick a threshold τ such that "predict Accept iff score > τ" maximizes
some objective on val, then apply to test.

**Two flavors of objective**:
- `τ*_raw` = argmax raw accuracy on val
- `τ*_bal` = argmax balanced accuracy on val

**Algorithm**: sweep τ over all unique val scores, pick the best:
```python
def best_tau_balanced(pairs):
    pairs = sorted(pairs)
    npos = sum(g for _, g in pairs)
    nneg = len(pairs) - npos
    tp = npos; tn = 0  # τ = -∞: predict all Accept
    best_b = (tp/npos + tn/nneg) / 2
    best_t = -∞
    for s, group in groupby(pairs, key=lambda p: p[0]):
        for _, g in group:
            if g == 1: tp -= 1   # this paper now predicted Reject
            else:      tn += 1
        b = (tp/npos + tn/nneg) / 2
        if b > best_b: best_b, best_t = b, s
    return best_t
```

**Use when**:
- You want a deployable binary classifier (not just a ranking).
- You have a labeled val set in the same prior as deployment.

**Properties**:
- `τ*_raw` is sensitive to val prior (favors majority class on imbalanced val).
- `τ*_bal` is val-prior-invariant (always picks the threshold maximizing balanced accuracy).
- When val is balanced 50/50, `τ*_raw == τ*_bal` (because raw accuracy = balanced accuracy at 50/50 prior).

**Where used**: `objective_analysis.md` §1.3 (dual calibration table), §7
(arxiv-trained ckpt sweep — uses `τ*_bal`).

### 16.2 Platt scaling (used in `tmp_latex_dir/generate_calibration_platt*.py`)

**Goal**: produce calibrated probabilities (not just a ranking) that match
the true accuracy in each confidence bin. Useful when downstream consumers
need `P(correct | confidence) ≈ confidence`.

**Algorithm**: fit a 2-parameter logistic on val log-odds:
```python
def fit_platt(val_logits, val_correct):
    # logit = log(p_chosen / (1 - p_chosen))
    # Calibrated p = sigmoid(a * logit + b)
    def nll(params):
        a, b = params
        p = sigmoid(a * val_logits + b)
        pc = where(val_correct, p, 1 - p)
        return -log(clip(pc, 1e-10, 1.0)).mean()
    result = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return result.x  # (a, b)

def apply_platt(test_logits, a, b):
    return sigmoid(abs(a * test_logits + b))  # |·| to map to chosen-class confidence
```

**Use when**:
- You want calibrated `P(correct)` for thresholding by confidence
  (e.g., "give me 80% accuracy on whatever fraction of papers I can").
- You want reliability diagrams to show points on the diagonal.

**Properties**:
- Doesn't change the score *ranking* → AUC, balanced ACC at any τ are unchanged.
- Only rescales the logit → maps raw confidence to calibrated confidence.
- Two parameters fit on val → small risk of overfitting on small val.

**Where used**:
- `scripts/tmp_latex_dir/generate_calibration_1x3.py` (the headline 1×3 figure with reliability + impact + 3D surface).
- `scripts/tmp_latex_dir/generate_calibration_platt*.py` (variants).
- `scripts/tmp_latex_dir/generate_calibration_val_correction.py` (compares uncalibrated vs Platt vs temperature vs isotonic).

### 16.3 Other post-hoc calibration methods (`scripts/calibration_posthoc.py`)

`calibration_posthoc.py` and `tmp_latex_dir/generate_calibration_val_correction.py`
implement 4 post-hoc methods:

| Method | Function | What it does |
|---|---|---|
| **Temperature scaling** | `fit_temperature` | 1-parameter `p_cal = sigmoid(logit / T)` — fits T on val NLL |
| **Platt scaling** | `fit_platt` | 2-parameter sigmoid (above) |
| **Isotonic regression** | `IsotonicRegression` from sklearn | Non-parametric monotonic mapping |
| **Histogram binning** | n/a in current code | Coarse calibration by binning |

These are compared in `generate_calibration_val_correction.py` to show that
**Platt is usually the best post-hoc method on our val** (lowest ECE), with
isotonic being competitive but more variable.

### 16.4 Which calibration to use when

| Goal | Method |
|---|---|
| Maximize balanced ACC on a deployment with known prior | `τ*_bal` threshold (§16.1) |
| Maximize raw ACC on a deployment with the val prior | `τ*_raw` threshold (§16.1) |
| Get reliable confidence values | Platt scaling (§16.2) |
| Reliability diagram for paper figure | Platt or isotonic (§16.3) |
| Threshold a confidence to trade coverage for accuracy ("80% acc on 30% of data") | Platt-calibrated confidence + threshold sweep |

For `objective_analysis.md`, all calibration is threshold-based (§16.1).
The `tmp_latex_dir/calibration_*.py` scripts use Platt (§16.2) because their
output is a paper figure that needs calibrated confidences for the 3D
surface and reliability diagram.

---

## 17. The `tmp_latex_dir/` script catalog

`scripts/tmp_latex_dir/` contains **41 figure-generation scripts** that
produce PDFs/PNGs for the lab's paper. Each script reads inference jsonls,
computes metrics, and writes to `tmp_latex_dir/figures/`.

Conventions used by all of them:
- Lab style: `"Arial"` font, sizes `labelsize=28, titlesize=34, legendsize=20, ticksize=20`
- Lab palette: `BLUE=#6098FF, ORANGE=#FECC81, GREEN=#77B25D, RED=#FF8988, PURPLE=#B28CFF`
- `LINEWIDTH=3.0, MARKERSIZE=10`
- Output: BOTH `.pdf` (for LaTeX) and `.png` (for markdown previews)
- `DECISION_TOKEN_IDX = 5` (legacy formulation A)

### 17.1 Calibration / reliability figures

| Script | Output | What it shows |
|---|---|---|
| `generate_calibration.py` | `calibration.{pdf,png}` | Original 1×3 figure (deprecated, use 1x3 below). |
| `generate_calibration_1x3.py` | `calibration_1x3.{pdf,png}` | **Headline figure**: (a) reliability + coverage (Platt), (b) impact correlation (text/vision p_accept vs human rating, both vs pctl_citation), (c) 3D accuracy surface vs (rating, confidence) |
| `generate_calibration_1x2.py` | `calibration_1x2.{pdf,png}` | Drops panel (b), keeps reliability + 3D surface |
| `generate_calibration_1x3_platt_surface.py` | variant of 1x3 with Platt-scaled 3D surface (alternate angle) |
| `generate_calibration_platt.py` | basic Platt-scaled reliability |
| `generate_calibration_platt_1x2.py` | 1×2 with Platt — reliability + 3D surface |
| `generate_calibration_platt_1x2_16x6.py` | wider 16×6 aspect ratio for paper layout |
| `generate_calibration_platt_1x2_30_70.py` | same but with natural 30/70 prior subsampling |
| `generate_calibration_extended.py` | 1×2 reliability + cumulative-threshold ("acc @ confidence ≥ τ") |
| `generate_calibration_extended_2025.py` | 1×2 with 2025-only year filter |
| `generate_calibration_extended_noise_em.py` | extended with the noise-EM model variant |
| `generate_calibration_balanced_vs_trainagreeing.py` | 2×2 grid: text/vision × reliability/cumulative, comparing balanced vs train-agreeing val splits |
| `generate_calibration_text_vs_vision_wd.py` | text vs vision side-by-side reliability (with weight decay) |
| `generate_calibration_human_corr.py` | scatter `p(accept) vs pct_rating` |
| `generate_calibration_human_corr_3d.py` | 3D version with reliability + scatter |
| `generate_calibration_ratio_sweep.py` | reliability comparison across train ratios (50/50, 40/60, 30/70) |
| `generate_calibration_posthoc.py` | 2×2 before/after for Platt/temperature/isotonic |
| `generate_calibration_val_correction.py` | per-bin reliability for uncalibrated vs all 4 post-hoc corrections |
| **`generate_objective_analysis.py`** | **`docs/objective_analysis.md` + 6 figures** | **The current headline analysis (§14 above)** |

### 17.2 Cross-conference / cross-distribution figures

| Script | Output | What it shows |
|---|---|---|
| `generate_cross_conference.py` | radar chart: SFT Text vs PT Base on balanced NeurIPS / ICML evals |
| `generate_cross_conference_acc.py` | compact cross-conference accuracy bar chart |
| `generate_cross_conference_combined.py` | 1×3: midtraining corpus pie + test-set pie + accuracy bars |
| `generate_data_dist.py` | dataset distribution histograms |
| `generate_data_mixture_combined.py` | data-mixture dot plots + cross-conference bars |
| `plot_calibration_story.py` | 3-panel calibration story: raw ACC, calibrated ACC (Δ-colored), AUC |

### 17.3 Scaling / comparison figures

| Script | Output | What it shows |
|---|---|---|
| `generate_3b_vs_7b.py` | 2 rows × 5 cols: 3B vs 7B head-to-head for text + vision |
| `generate_scaling_laws.py` | accuracy / accept-recall / reject-recall vs model size (3B/7B/14B text, 3B/7B vision) |
| `generate_scaling_flops.py` | metrics vs training FLOPs (Chinchilla approximation for text + vision) |
| `generate_32b_lora_train_curves.py` | 32B LoRA training loss curves (lr sweep) |
| `generate_ratio_train_curves.py` | 3×2: training curves per train ratio (50/50, 40/60, 30/70) |
| `generate_trainsize_vs_accuracy.py` | data-mixture ablation: paired dot plot |

### 17.4 Quality / impact figures

| Script | Output | What it shows |
|---|---|---|
| `generate_impact_correlation_1x2.py` | 1×2: p(accept) vs pctl_rating + p(accept) vs pctl_citation (both Platt-scaled) |
| `generate_impact_correlation_3x1.py` | 3×1 vertical version: human rating, text p(accept), vision p(accept) → all vs pctl_citation |
| `generate_citation_heatmap_3x1.py` | 3×1 compact heatmap: pctl_rating × pctl_citation → accuracy (human / text-SFT / vision-SFT) |

### 17.5 Bias / artifact figures

| Script | Output | What it shows |
|---|---|---|
| `generate_2024_bias.py` | 1×2: text vs vision accuracy by year (highlights 2024 artifact) + first-section-header y-position by accept/reject |
| `generate_2024_bias_13x3.py` / `_16x6.py` | wider/taller variants for paper layout |

### 17.6 Evidence / qualitative figures

| Script | Output | What it shows |
|---|---|---|
| `generate_evidence_cards.py` | 3-column visual: Human Review | Without SFT Prior | With SFT Prior |
| `generate_evidence_tex.py` | LaTeX-ready evidence display with thin colored borders + score dots |

### 17.7 Visualization helpers

| Script | Purpose |
|---|---|
| `visualize_paper.py` | Render a single paper's content + reviews + model decision (image-rich) |
| `visualize_paper_text.py` | Same but text-only pipeline |

### 17.8 LaTeX scaffolding

`tmp_latex_dir/base.latex` is the LaTeX preamble used by the lab's paper.
The figure files generated by the scripts above are referenced from there
via `\input{tmp_latex_dir/figures/<name>.pdf}`.

### 17.9 Which scripts are actively used in `objective_analysis.md`

**Only one**: `generate_objective_analysis.py`. It re-implements the
metric and scoring logic inline (doesn't import from the other figure
scripts) so it can have its own bootstrap CIs, integrity checks, etc.

If you're modifying `objective_analysis.md`, you don't need to touch any
other `tmp_latex_dir/` script. They serve a different consumer (the
LaTeX paper).

If you're adding a figure to the LaTeX paper, **don't** modify
`generate_objective_analysis.py` — add a new `generate_<name>.py` that
follows the conventions above (lab style, palette, both PDF + PNG output).

---

## 18. Glossary

- **bACC** — balanced accuracy = (accept-recall + reject-recall) / 2
- **AR / RR** — accept-recall / reject-recall (per-class recall)
- **AUC** — area under ROC curve, threshold-independent rank quality
- **ρ** — Spearman rank correlation
- **τ** — decision threshold on the continuous score
- **τ\*_raw** — argmax raw ACC threshold on val
- **τ\*_bal** — argmax balanced ACC threshold on val
- **π** — accept rate (prior); π_iclr_natural ≈ 0.313, π_arxiv_natural ≈ 0.238
- **OOD** — out-of-distribution (e.g., arxiv-trained model evaluated on ICLR)
- **In-domain / cross-domain** — train and test from same/different distributions
- **y25up / y24up** — year-cutoff filters: ICLR ≥ 2025 / arxiv conference_year ≥ 2024
- **balanced (50/50)** — accept rate forced to 50% by stratified subsampling
- **natural / natrate / 30/70** — accept rate matches the venue's actual rate
- **DR** — DeepReviewer-14B (Phi-3 14B, Standard Mode, 4-reviewer ensemble)
- **PaperLens** — our internal name for the 7B/3B Qwen-finetuned models
- **DECISION_TOKEN_IDX** — `5`; the position of the Accept/Reject token in `Outcome: \boxed{X}` template
- **Platt scaling** — 2-parameter `sigmoid(a·logit + b)` fit on val NLL → calibrated confidence
- **Temperature scaling** — 1-parameter `sigmoid(logit / T)` calibration
- **Isotonic regression** — non-parametric monotonic recalibration
- **ECE** — expected calibration error; mean absolute gap between confidence and accuracy across bins
- **Reliability diagram** — confidence-vs-accuracy plot, perfectly calibrated = on diagonal
- **Coverage** — fraction of papers above a confidence threshold
- **Logit / log-odds** — `log(p / (1-p))`, the linear scale where Platt scaling operates
- **Marginal vs 2-class softmax** — Formulation A vs B in §14
- **Legacy regime** — older inference jsonls with `token_logprobs` only, p(accept) via index 5
- **Current regime** — newer inference jsonls with `logprob_accept`/`logprob_reject` arrays, dynamic decision-position lookup

---

## 19. Quick reference card — most-used file paths

```
# Generator + verifier
scripts/tmp_latex_dir/generate_objective_analysis.py
scripts/verify_objective_analysis.py

# Output
docs/objective_analysis.md
docs/analysis_explain.md  ← this file
docs/objective_analysis_figures/*.png
tmp_latex_dir/figures/objective_*.{pdf,png}

# Source reports the analysis builds on
reports/test-ratio-and-metrics.md
reports/ratio_xeval_7b.md
reports/balanced_eval_2026-05-02.md
/scratch/gpfs/ZHUANGL/sk7524/Researcher/RESULTS_deepreviewer_balanced.md

# Canonical test datasets (ICLR balanced)
data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json
data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json

# Canonical test datasets (arxiv balanced y24up)
data/arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test/data.json
data/arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_test/data.json

# ICLR-trained 7B inference (the headline 4 cells)
results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/finetuned-ckpt-1322.jsonl                  # text 50/50
results/final_sweep_v7_datasweepv3/optim_search_2026/ratio_sweep/bz32_lr1e-6_text_30_70/balanced/finetuned-ckpt-1322.jsonl  # text 30/70
results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl                # vision 50/50
results/final_sweep_v7_datasweepv3/optim_search_2026/ratio_sweep/bz16_lr1e-6_vision_30_70/balanced/finetuned-ckpt-2642.jsonl  # vision 30/70

# Arxiv-trained ckpt sweep base
results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/<cell>/<dataset_subdir>/finetuned-ckpt-<step>{,-gpu-test}.jsonl

# DeepReviewer baseline (external repo)
/scratch/gpfs/ZHUANGL/sk7524/Researcher/results/deepreviewer-14b-standard/<tag>/deepreviewer-14b-standard.jsonl
/scratch/gpfs/ZHUANGL/sk7524/Researcher/subsamples/<tag>_q4_seed42.json

# This worktree
/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer-analysis-codex
  ├── data    → symlink to main/data
  ├── results → symlink to main/results
  └── (everything else is a real copy on branch `analysis-codex`)

# Re-run anything
uv run python scripts/tmp_latex_dir/generate_objective_analysis.py
uv run python scripts/verify_objective_analysis.py
```
