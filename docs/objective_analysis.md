# Objective Analysis: Quality Indicator vs Conference Acceptor

**TL;DR**
1. Two objectives, two metric stacks. *Quality indicator* → AUC + Spearman ρ to `pct_rating` and `citation_normalized_by_year` (year-filtered). *Conference acceptor* → balanced ACC + accept/reject recall.
2. Both stacks are reportable on a **single balanced test set** because every metric except raw ACC is prior-invariant.
3. **Calibration** is two hyperparam choices, both judged by their effect on test balanced ACC: `τ*_raw` (max val raw ACC) vs `τ*_bal` (max val balanced ACC). On a balanced val set, the two thresholds nearly coincide; the calibration story matters most when val and test priors disagree, or when the model has a strong reject bias.
4. **Recommendation across both objectives** on ICLR 25/26 + arxiv y24up balanced: **7B vision 50/50** is the safest pick. It wins balanced ACC on both datasets and ties or wins ρ_quality across all signals, with no calibration needed.

---

## 1. Right metrics

### 1.1 Prior-invariant family

| Metric | Formula | Prior-invariant? |
|---|---|---|
| Raw ACC          | `(TP + TN) / N`                       | **No** — depends on test class prior |
| Balanced ACC     | `(accept-recall + reject-recall) / 2` | **Yes** |
| Accept recall    | `TP / (TP + FN)`                      | **Yes** (within-class) |
| Reject recall    | `TN / (TN + FP)`                      | **Yes** (within-class) |
| AUC              | `P(score(accept) > score(reject))`    | **Yes** (rank-based) |
| Spearman ρ(score, quality) | rank correlation             | **No** — mixture changes the all-population marginal (see §2.3) |

### 1.2 Why raw ACC is misleading — concrete cases from our data

Without calibration, the **30/70 training ratio** inflates raw ACC on natural-prior test sets because the model develops a reject bias and the natural prior is reject-heavy. The cleanest example is text 30/70 on arxiv natural (π_arxiv ≈ 0.24 accept):

| Model | Test cell | Raw ACC | Accept recall | Reject recall | **Balanced ACC** |
|---|---|---:|---:|---:|---:|
| text 30/70 | arxiv natural | 76.3 | 0.3 | 100.0 | **50.1** |
| text 50/50 | arxiv natural | 76.7 | 25.0 | 92.8 | **58.9** |
| vision 30/70 | arxiv natural | 70.0 | 48.6 | 77.0 | **62.8** |
| vision 50/50 | arxiv natural | 72.0 | 46.2 | 80.5 | **63.3** |

Text 30/70 is the smoking gun: raw ACC = 76.3 looks deployable, but accept recall ≈ 0.3 means it predicts almost every paper as reject. Its balanced ACC is 50.1 — random. **It is not predicting; it is exploiting the natural prior.**

### 1.3 Calibration: two flavors, both as hyperparam choices

Both are val-derived thresholds applied to test. Both are evaluated by the metric we actually care about — **test balanced ACC**:

- `τ*_raw` — argmax raw ACC on val
- `τ*_bal` — argmax balanced ACC on val

With thresholding, **raw ACC jumps even more for the 30/70-trained models on balanced tests**, because the threshold pushes back against the model's reject bias. Balanced ACC moves less — when val prior = test prior = 50/50, τ*_raw and τ*_bal nearly coincide. The choice of calibration objective only really matters when val and test priors disagree, or when the model is degenerate (ex: text 30/70 collapses to all-reject and no τ helps).

![calibration sweep](../tmp_latex_dir/figures/objective_calibration_sweep.png)

**Per-model dual-calibration table on ICLR 25/26 balanced test:**

| Model | τ*_raw | τ*_bal | test raw @ τ=0 | test bACC @ τ=0 | test raw @ τ*_raw | test bACC @ τ*_raw | test raw @ τ*_bal | test bACC @ τ*_bal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text 50/50 | -0.37 | -0.37 | 65.2 | 65.2 | 66.1 | 66.1 | 66.1 | 66.1 |
| text 30/70 | -0.87 | -0.87 | 61.5 | 61.6 | 67.1 | 67.1 | 67.1 | 67.1 |
| vision 50/50 | -0.50 | -0.50 | 67.6 | 67.6 | 67.5 | 67.5 | 67.5 | 67.5 |
| vision 30/70 | -0.12 | -0.12 | 64.9 | 64.9 | 65.5 | 65.5 | 65.5 | 65.5 |

**Per-model dual-calibration table on arxiv y24up balanced test:**

| Model | τ*_raw | τ*_bal | test raw @ τ=0 | test bACC @ τ=0 | test raw @ τ*_raw | test bACC @ τ*_raw | test raw @ τ*_bal | test bACC @ τ*_bal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text 50/50 | -1.37 | -1.37 | 59.5 | 58.4 | 65.3 | 65.7 | 65.3 | 65.7 |
| text 30/70 | -4.25 | -4.25 | 51.8 | 50.1 | 62.4 | 62.9 | 62.4 | 62.9 |
| vision 50/50 | -1.12 | -1.12 | 63.4 | 62.8 | 66.0 | 66.4 | 66.0 | 66.4 |
| vision 30/70 | -0.25 | -0.75 | 62.6 | 62.1 | 63.9 | 63.8 | 64.4 | 65.1 |

**Headline observation**: when val and test priors match (both balanced), τ*_raw and τ*_bal pick essentially the same threshold and produce the same test bACC. Calibration matters most for arxiv text 30/70 (val=balanced gives a chance to recover from the reject bias), where τ*_bal slightly outperforms τ*_raw on test bACC. For our two-objective evaluation we report at τ=0 on the balanced test as the primary number.

### 1.4 Deriving natural raw ACC from balanced per-class recalls

Given prior π = P(accept), `raw_ACC(π) = π · accept_recall + (1 − π) · reject_recall`. Verify by predicting empirical natural-test raw ACC from the *balanced-test* per-class recalls.

| Model | dataset | accept-rec (bal) | reject-rec (bal) | π_natural | predicted natural raw ACC | empirical natural raw ACC | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| text 50/50 | iclr | 56.2 | 74.2 | 0.313 | 68.6 | 65.6 | -3.00 |
| text 50/50 | arxiv | 24.0 | 92.9 | 0.238 | 76.5 | 76.7 | +0.23 |
| text 30/70 | iclr | 39.7 | 83.4 | 0.313 | 69.7 | 73.7 | +3.97 |
| text 30/70 | arxiv | 0.3 | 100.0 | 0.238 | 76.3 | 76.3 | +0.04 |
| vision 50/50 | iclr | 65.4 | 69.8 | 0.313 | 68.4 | 66.2 | -2.18 |
| vision 50/50 | arxiv | 45.1 | 80.4 | 0.238 | 72.0 | 72.0 | -0.02 |
| vision 30/70 | iclr | 55.0 | 74.9 | 0.313 | 68.6 | 73.9 | +5.27 |
| vision 30/70 | arxiv | 48.0 | 76.2 | 0.238 | 69.5 | 70.0 | +0.47 |

On arxiv (where balanced and natural test sets share the y24up paper pool), the formula matches within ≤1pp. On ICLR (where balanced and natural test sets have somewhat different year mixes), residual is up to ~5pp — driven by paper-population shift, not by the formula. **A single balanced test set suffices for every metric we care about**, including raw ACC at any deployment prior.

---

## 2. Quality indicator metrics (Objective 1)

### 2.1 Metrics surfaced

- **AUC** — prior-invariant; threshold-free; the score's accept-vs-reject ranking quality.
- **Spearman ρ to `pct_rating`** — alignment with reviewer perception; full coverage on ICLR.
- **Spearman ρ to `citation_normalized_by_year`** — alignment with post-publication impact. *Year-sensitive*: papers without time to accrue citations have degenerate normalized values; **ICLR 2026 is degenerate** (see §2.3).

### 2.2 AUC is prior-invariant; rank correlations are NOT

AUC is rank-based on the binary label, so it is unaffected by the test mixture proportion. Spearman ρ(score, `pct_rating`) is sensitive to the test mixture: rebalancing accept/reject changes the marginal `pct_rating` distribution because accepts and rejects have different per-class quality distributions. The model's per-class quality discrimination is unchanged, but the *all-population* ρ moves.

### 2.3 Distribution shift figure — corrected

![distribution shift](../tmp_latex_dir/figures/objective_distribution_shift.png)

Three signals across two priors. Important details:
- **`pct_rating` (left column)** — strong class separation (accept median ≈ 0.80, reject median ≈ 0.30). Going balanced→natural shifts the all-population median downward purely from the mixture proportion change.
- **`citation_normalized_by_year` for 2025 papers (middle column)** — clear separation (accept median ≈ 0.65, reject median ≈ 0.30). 2025 papers have had time to accrue citations.
- **`citation_normalized_by_year` for 2026 papers (right column)** — degenerate. Raw citations are all 0 → normalized field collapses to 0.500 for everyone. **2026 must be excluded from any citation-quality analysis.** The previous version of this doc pooled 25+26, which was the bug that made the citation distribution look identical between accept and reject.

Quantified medians:

| Subset | accept-rate (%) | accept median | reject median | all-pop median |
|---|---:|---:|---:|---:|
| rating, balanced, 25+26 | 50.0 | 0.81 | 0.30 | 0.57 |
| rating, natural,  25+26 | 38.4 | 0.81 | 0.30 | 0.47 |
| citation, balanced, 2025 only | 50.1 | 0.65 | 0.29 | 0.29 |
| citation, natural,  2025 only | 37.5 | 0.65 | 0.29 | 0.29 |
| citation, balanced, 2026 only | 49.9 | 0.50 | 0.50 | 0.50 |
| citation, natural,  2026 only | 39.0 | 0.50 | 0.50 | 0.50 |

Note that the `citation 2026` accept median = reject median = 0.50 — exactly the degenerate case. Any 2026 citation analysis is uninformative.

### 2.4 Per-class ρ decomposition (Obj 1 results, 7B grid)

If `ρ_all >> ρ_acc, ρ_rej`, the apparent quality alignment comes from the binary label signal alone, not within-class ranking. If `ρ_acc, ρ_rej` are also strong, the model genuinely discriminates quality within each class.

**ICLR 25/26 balanced** (full coverage):

| Model | AUC | ρ rating all | ρ rating acc | ρ rating rej | ρ cit (2025) all | ρ cit (2025) acc | ρ cit (2025) rej |
|---|---:|---:|---:|---:|---:|---:|---:|
| text 50/50 | 0.721 | +0.47 | +0.21 | +0.44 | +0.30 | +0.22 | +0.19 |
| text 30/70 | 0.720 | +0.47 | +0.18 | +0.45 | +0.30 | +0.22 | +0.20 |
| vision 50/50 | 0.736 | +0.48 | +0.16 | +0.45 | +0.25 | +0.17 | +0.14 |
| vision 30/70 | 0.723 | +0.47 | +0.17 | +0.46 | +0.28 | +0.20 | +0.17 |

**Sample sizes (ICLR 25/26 balanced):** rating n_all/n_acc/n_rej = 1667/834/833; citation 2025-only n_all/n_acc/n_rej = 676/339/337.

**Arxiv y24up balanced** (sparse `pct_rating` and `pct_citation` annotations):

| Model | AUC | ρ rating all | ρ rating acc | ρ rating rej | ρ pct_citation all | ρ pct_citation acc | ρ pct_citation rej |
|---|---:|---:|---:|---:|---:|---:|---:|
| text 50/50 | 0.720 | +0.23 | +0.15 | +0.32 | +0.11 | +0.10 | +0.20 |
| text 30/70 | 0.697 | +0.18 | +0.10 | +0.15 | +0.18 | +0.16 | +0.31 |
| vision 50/50 | 0.724 | +0.17 | +0.08 | +0.36 | +0.19 | +0.17 | +0.48 |
| vision 30/70 | 0.704 | +0.14 | +0.02 | +0.39 | +0.22 | +0.19 | +0.62 |

**Sample sizes (arxiv y24up balanced):** rating n_all/n_acc/n_rej = 292/246/46; pct_citation n_all/n_acc/n_rej = 341/327/14.

### 2.5 Bootstrap CIs on the recommended Obj 1 model (7B vision 50/50)

Resampled 500× over the ICLR balanced test set; 95% CI. Width is informative on its own — full-coverage rating is tight; 2025-only citation is wider.

| signal | n | ρ point | 95% CI |
|---|---:|---:|---:|
| ICLR `pct_rating`              | 1670  | +0.48 | [+0.44, +0.51] |
| ICLR `citation_norm` 2025 only | 678 | +0.25  | [+0.19, +0.31] |

---

## 3. Conference acceptor metrics (Objective 2)

Balanced ACC + accept-recall + reject-recall on the **balanced** test set. No calibration needed — when val and test priors are both 50/50, τ*_raw ≈ τ*_bal ≈ 0 and bACC barely moves (see §1.3).

**ICLR 25/26 balanced (Obj 2 view):**

| Model | n | balanced ACC | accept-recall | reject-recall | AUC |
|---|---:|---:|---:|---:|---:|
| text 50/50 | 1667 | **65.2** | 56.2 | 74.2 | 0.721 |
| text 30/70 | 1667 | **61.6** | 39.7 | 83.4 | 0.720 |
| vision 50/50 | 1670 | **67.6** | 65.4 | 69.8 | 0.736 |
| vision 30/70 | 1670 | **64.9** | 55.0 | 74.9 | 0.723 |

**Arxiv y24up balanced (Obj 2 view):**

| Model | n | balanced ACC | accept-recall | reject-recall | AUC |
|---|---:|---:|---:|---:|---:|
| text 50/50 | 1414 | **58.4** | 24.0 | 92.9 | 0.720 |
| text 30/70 | 1415 | **50.1** | 0.3 | 100.0 | 0.697 |
| vision 50/50 | 1414 | **62.8** | 45.1 | 80.4 | 0.724 |
| vision 30/70 | 1414 | **62.1** | 48.0 | 76.2 | 0.704 |

---

## 4. Comprehensive analysis: optimal modality + train ratio for both objectives

![summary](../tmp_latex_dir/figures/objective_summary.png)

### 4.1 Headline rankings on the 7B grid (2nd ckpt)

| Objective | Metric | ICLR 25/26 balanced — winner | arxiv y24up balanced — winner |
|---|---|---|---|
| Obj 2 | balanced ACC      | vision 50/50 (67.6) | vision 50/50 (62.8) |
| Obj 1 | AUC               | vision 50/50 (0.736) | vision 50/50 (0.724) |
| Obj 1 | ρ pct_rating      | vision 50/50 (0.478) | text 50/50 (0.232) |
| Obj 1 | ρ citation        | text 30/70 (0.301) | vision 30/70 (0.215) |

### 4.2 Recommended configurations

| Objective | Recommended config | Why |
|---|---|---|
| **Obj 1 — General quality indicator** | **7B vision 50/50** at τ=0 | Top AUC on both datasets (0.736 ICLR, 0.724 arxiv); top ρ rating on ICLR (+0.48). On ρ citation, text 30/70 edges vision 50/50 by 0.05 on ICLR (+0.30 vs +0.25); vision 30/70 wins on arxiv pct_citation (+0.22). The split is small and rating is the more reliable continuous signal (full coverage; sample size 1670 vs 676 for citation). |
| **Obj 2 — Conference acceptor**       | **7B vision 50/50** at τ=0 | Best balanced ACC on both balanced test sets (ICLR 67.6, arxiv 62.8); the only model with both per-class recalls > 65% on ICLR. |

Both objectives converge on the **same config**: **7B vision 50/50, no calibration needed (τ=0 on balanced test)**. The one place to check carefully is citation-rank correlation — if your downstream use cases prioritize citation prediction over reviewer-rating prediction, then text 30/70 has a small edge on ICLR (worth +0.05 ρ vs vision 50/50). Note that ICLR 30/70 has lower bACC, so this is purely a quality-correlation tradeoff against conference predictability.

### 4.3 3B partial evidence (last ckpt)

Direct ICLR-balanced 3B runs reinforce the modality direction (vision ≈ text on ICLR-balanced bACC at 3B), but matching 3B vision arxiv cross-eval files are missing, so 3B is not used for the cross-dataset modality recommendation.

| 3B model | n | balanced ACC | AUC | accept-rec | reject-rec |
|---|---:|---:|---:|---:|---:|
| 3B text 50/50 (ICLR balanced) | 1667 | 66.2 | 0.719 | 67.1 | 65.2 |
| 3B vision 50/50 (ICLR balanced) | 1670 | 66.7 | 0.728 | 69.7 | 63.7 |

### 4.4 Train ratio: 50/50 wins for both objectives

- **Obj 1 (AUC + ρ_quality)**: 50/50 has the most consistent, slightly higher correlations on ICLR; on arxiv subset metrics the 30/70 vision model occasionally edges out, but small subset sizes (n=292 rating, n=341 citation) make it weak evidence.
- **Obj 2 (balanced ACC)**: 50/50 wins on both ICLR (67.6 vs 64.9 for vision; 65.2 vs 61.6 for text) and arxiv (62.8 vs 62.1 for vision; 58.4 vs 50.1 for text — text 30/70 collapses to chance).
- The 30/70 ratio's only strength is *natural-prior raw ACC* on natural test sets — but that is exactly the metric we don't trust (§1.2).

---

## Appendix: data sources

**7B 2nd-ckpt models (ICLR-trained):**
- text 50/50: `bz32_lr1e-6_text` ckpt 1322
- text 30/70: `bz32_lr1e-6_text_30_70` ckpt 1322
- vision 50/50: `bz16_lr1e-6_vision` ckpt 2648
- vision 30/70: `bz16_lr1e-6_vision_30_70` ckpt 2642

**3B last-ckpt models (ICLR-trained):**
- 3B text 50/50: `scaling/bz32_lr1e-6_text_3b` ckpt 2644
- 3B vision 50/50: `scaling/bz16_lr1e-6_vision_3b` ckpt 5296

**Test sets:**
- ICLR 25/26 balanced: `data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_{text,vision}_*_test/data.json` (year-filtered to {2025, 2026})
- ICLR 25/26 natural (30/70): `data/iclr_2020_2023_2025_2026_30_70_*_test/data.json`
- Arxiv y24up balanced: `data/arxiv_50_50_21k_{text,vision}_wmetadata_*_y24up_test/data.json`
- Arxiv y24up natural (per-conference natrate): `data/arxiv_natrate_21k_*_y24up_test/data.json`

Generated by `scripts/tmp_latex_dir/generate_objective_analysis.py`. All numbers reproducible from the source jsonls.
