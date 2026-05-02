# Test-Ratio and Metrics — what to measure, where to measure it

This document distills the metric + test-ratio question from the larger
`ratio_xeval_7b.md` analysis. **Goal: pick the right metric for each goal,
and stop debating which test set to evaluate on.**

## TL;DR

1. We're measuring two different things that are **not the same**:
   - **Universal quality indicator** — does the model's score track underlying paper quality?
   - **Conference accept/reject predictor** — does the binary decision match the venue's call?
2. The right metrics:
   - **Quality indicator** → `Spearman ρ(score, pct_rating)` (or any continuous reviewer signal)
   - **Conference predictor** → `balanced accuracy = avg(accept-recall, reject-recall)` (or `AUC` for threshold-independent ranking)
3. **Raw ACC is misleading** on imbalanced test sets — it rewards constant-class predictors. Don't report it as a primary metric on natural-prior cells.
4. **The test-ratio mostly doesn't matter** — `balanced ACC`, `AUC`, and `ρ_quality` are all approximately prior-invariant (Δ ≤ 5pp empirically). Only `raw ACC` is prior-dependent — and it can be reconstructed analytically as `raw_ACC(π) = π · AccR + (1-π) · RejR` from per-class recalls.

---

## 1. Two goals, two axes

| Axis | Goal | Right metric | What it measures |
|---|---|---|---|
| **A — Universal quality** | "Use the model's score to rank papers by quality" (review-assist, recommendation, citation prediction) | `Spearman ρ(score, pct_rating)` | Continuous quality alignment — does a higher score correspond to a higher reviewer percentile? |
| **B — Conference predictor** | "Use the model to decide accept/reject per the venue" (deployable classifier) | `balanced ACC = (AccR + RejR) / 2`  *or*  `AUC` | Binary classification quality, weighted equally across classes |

These two axes are **correlated but not identical** (ρ across our 16 cells ≈ +0.82 between balanced ACC and ρ_quality). They decouple in important cases — e.g., text on `arxiv-natural-iclr` reaches balanced ACC ≈ 64% but ρ_quality ≈ −0.04, meaning the model passes the binary classifier test while completely failing the quality test. Vision on the same papers is ≈ 64% balanced ACC AND ρ ≈ +0.24 — the universal-quality model.

---

## 2. Metric inventory

| Metric | What it measures | Prior-invariant? | When to use |
|---|---|---|---|
| **Raw ACC** | Fraction of correct predictions | **No** — depends on test prior | Only as "deployment readiness" on the natural-prior test |
| **Balanced ACC** | `(AccR + RejR) / 2` | **Yes** | Primary predictor metric |
| **AUC** | `P(score(accept) > score(reject))` | **Yes** | Threshold-independent ranking quality |
| **AccR / RejR** | Per-class recall | Yes (per-class) | Diagnostic for bias direction |
| **Spearman ρ** between score and `pct_rating` | Continuous quality alignment | Yes | Universal-quality metric |

---

## 3. Why raw ACC is misleading — concrete cases from our data

The cleanest evidence is the gap between raw ACC and balanced ACC for the same model.

| Model | Test cell | Raw ACC | AccR | RejR | **Balanced ACC** |
|---|---|---:|---:|---:|---:|
| **text 30/70** | arxiv natural | **76.3%** ✱ | 0% | 100% | **50.0%** ← random baseline |
| text 50/50 | arxiv natural | 76.7% | 25% | 93% | 59.0% |
| vision 50/50 | arxiv natural | 72.0% | 46% | 81% | **63.5%** ← actual best |
| vision 30/70 | arxiv natural | 70.0% | 49% | 77% | 63.0% |

**The text 30/70 row is the smoking gun.** Raw ACC says it's the best model on arxiv natural (76.3%). But it predicts *every paper as reject* (AccR = 0). Its balanced ACC is 50% — exactly chance. **It's not predicting; it's exploiting the natural prior.**

Why this happens: arxiv natrate has ~25% accept rate. A constant-reject classifier scores 75% raw ACC trivially. text 30/70's training distribution made it strongly reject-biased; combined with the natural prior, it gets a free 76% raw ACC without doing any work.

The same model on the **balanced** prior (50% accept) test set drops to **51.8% raw ACC** — a 24.5pp swing purely from the test prior change. **Same model, same papers (mostly), same evaluation pipeline. Just different prior.** No real classifier should swing this much from a re-weighting.

### The general pattern (from our 8 model×family pairs)

| Metric | Δ between balanced and natural test | Verdict |
|---|---:|---|
| Raw ACC | up to **+24.5 pp** (arxiv text 30/70) | Massive prior dependence |
| Balanced ACC | 0 to ~5 pp | Approximately prior-invariant |
| AUC | 0.003 to 0.084 | Approximately prior-invariant |

![Test-ratio invariance](../tmp_latex_dir/figures/ratio_xeval_test_ratio_invariance.png)

The 3-panel scatter above plots each metric's value on balanced vs natural test (one point per model × test family). **Balanced ACC and AUC sit on the diagonal; raw ACC scatters wildly off-diagonal.**

---

## 4. Why balanced ACC is the right predictor metric

**By construction**: `balanced ACC = (TP/(TP+FN) + TN/(TN+FP)) / 2`. Each per-class recall divides by the count of that class — so changing the relative class proportions doesn't change either recall.

**Intuitive interpretation**: balanced ACC = "what would the raw accuracy be if you had a 50/50 prior?" It answers the modeling question (how well does the model discriminate) independent of the deployment question (what prior will users see).

**Penalizes degenerate predictors**: a constant-class predictor gets balanced ACC = 50% no matter the test prior. Raw ACC for the same predictor = max(π, 1-π) which can look excellent on imbalanced data.

**Empirical stability**: the largest balanced ACC swing across our two test priors is ~5pp on ICLR (where the two test sets sample meaningfully different paper years), and ≤ 1pp on arxiv (where the paper population is the same y24up pool). The remaining variance comes from *paper-population differences*, not from the prior itself.

---

## 5. Which test ratio? It mostly doesn't matter.

| Metric | Where to measure | Why |
|---|---|---|
| Balanced ACC | **Either test set works**. Use balanced for cleaner numbers (where balanced ACC = raw ACC). | Prior-invariant by construction |
| AUC | Either | Prior-invariant by construction |
| ρ(score, pct_rating) | Either; use the larger n | Prior-invariant |
| **Raw ACC** | **Only natural** (matches deployment) | Depends on prior |

### Even simpler — you don't need a natural test set for raw ACC either

Raw ACC at *any* prior π can be reconstructed analytically from per-class recalls measured on the balanced test:

$$\text{raw\_ACC}(\pi) = \pi \cdot \text{AccR} + (1 - \pi) \cdot \text{RejR}$$

Empirically verified: on arxiv (where the balanced and natural test sets share the same y24up paper pool), the formula matches the empirical natural-test raw ACC within **≤ 1pp** across all 4 models. On ICLR (where the two test sets have somewhat different year mixes) the residual is up to ~5pp — driven by paper-population shift, not by the formula.

| Model | AccR (bal) | RejR (bal) | π (arxiv natrate ≈ 0.25) | predicted raw_ACC | empirical (natural test) | Δ |
|---|---:|---:|---:|---:|---:|---:|
| text 50/50 | 24 | 93 | 0.25 | 76.0 | 76.7 | +0.7 |
| text 30/70 | 0 | 100 | 0.25 | 75.3 | 76.3 | +1.0 |
| vision 50/50 | 45 | 80 | 0.25 | 71.4 | 72.0 | +0.6 |
| vision 30/70 | 48 | 76 | 0.25 | 69.1 | 70.0 | +0.9 |

So in principle, **a single balanced test set suffices for every metric we care about** — including raw ACC at any specified deployment prior.

---

## 5b. Aside — does AUC track quality better than balanced ACC?

A reasonable intuition: AUC is rank-based (`P(score(accept) > score(reject))`) and `ρ(score, pct_rating)` is also rank-based (Spearman). Both operate on the score ordering, so they *should* correlate strongly — perhaps more strongly than balanced ACC (which is threshold-based at τ=0).

**Empirically, in our data, balanced ACC tracks ρ_quality better than AUC does.** Apples-to-apples on the same 24 (model × test-population) subsets where pct_rating is available:

| Predictor metric | Spearman ρ vs ρ_quality | Pearson r vs ρ_quality |
|---|---:|---:|
| AUC | +0.50 | +0.52 |
| **balanced ACC** | **+0.66** | **+0.75** |

Why? Two reasons specific to our data:

1. **AUC compresses into a narrow band** (0.45 – 0.77 across 24 cells) — its discrimination between models is muted.
2. **balanced ACC spreads more** (39.7 – 70.6) and the spread aligns with the ρ_quality spread.

The cleanest illustration is the `arxiv-iclr-natural` cell (same papers, same labels):

| Model | AUC | bACC | ρ_quality |
|---|---:|---:|---:|
| text 50/50 | 0.712 | 55.2 | **−0.127** |
| text 30/70 | 0.719 | 50.0 | **−0.168** |
| vision 50/50 | 0.759 | 67.1 | **+0.455** |
| vision 30/70 | 0.747 | 66.3 | **+0.456** |

Vision's AUC is only 0.04–0.05 higher than text's — but vision's bACC is 12–17pp higher and ρ_quality is **0.6 higher**. **bACC's sharper discrimination tracks the ρ_quality split; AUC's narrower spread doesn't.**

**Deeper reason**: AUC measures whether the model *can* separate labeled accepts from labeled rejects at *some* threshold (an existence claim about the score's rank ordering). bACC measures whether the model's *natural* decision boundary (τ=0) aligns well with the labels. When a model's natural threshold is mis-placed (text 30/70's reject-bias), AUC stays high (the rank is fine) but bACC collapses (the threshold is wrong) — and that threshold-misalignment empirically turns out to correlate with quality-misalignment too.

**So the bottom line for Axis-A measurement**: don't proxy `ρ(score, pct_rating)` with anything — measure it directly when you have a continuous quality signal. AUC is a partial proxy; bACC is a better one in our data; but neither is a substitute for the direct measurement.

---

## 6. AUC consistency across train-ratio and modality

**Within-cell AUC range** (how much do the 4 model configs disagree on each test cell?):

| Test cell | min AUC | max AUC | range |
|---|---:|---:|---:|
| ICLR balanced | 0.720 | 0.736 | 0.016 (very tight) |
| ICLR natural  | 0.697 | **0.805** | **0.108** (large — see note) |
| arxiv balanced | 0.697 | 0.724 | 0.027 |
| arxiv natural  | 0.700 | 0.736 | 0.036 |

**Within-model AUC range** (does the same model give a stable AUC across populations?):

| Model | iclr-bal | iclr-nat | arxiv-bal | arxiv-nat | range |
|---|---:|---:|---:|---:|---:|
| text 50/50  | 0.721 | 0.697 | 0.720 | 0.725 | **0.028** |
| text 30/70  | 0.720 | **0.805** | 0.697 | 0.700 | **0.108** ← spike |
| vision 50/50 | 0.736 | 0.725 | 0.724 | 0.736 | **0.012** ← most consistent |
| vision 30/70 | 0.723 | **0.804** | 0.704 | 0.714 | 0.100 ← spike |

**Observations:**
- **Vision 50/50 has the most consistent AUC across populations (range 0.012)** — it ranks papers similarly regardless of the test set drawn.
- **30/70-trained models spike on ICLR-natural** (0.80+) but otherwise sit ~0.70-0.72. Likely because the ICLR natural test set's paper composition matches their training prior, exposing a real per-population AUC sensitivity.
- text 50/50 is the second-most-consistent (range 0.028). 30/70-train introduces population-dependent variance into AUC.

So AUC is **mostly consistent** for the well-trained models (text 50/50, vision 50/50), and somewhat population-dependent for the 30/70-trained ones. Vision 50/50's AUC stability across all 4 cells is a strong signal that it's learning a population-invariant ranking.

---

## 7. Recommended reporting protocol

For any future cross-eval:

1. **Run inference on the BALANCED test set.** That single test set gives you:
   - balanced ACC (prior-invariant predictor metric)
   - AUC (prior-invariant ranking metric)
   - ρ(score, `pct_rating`) (universal-quality metric)
   - Raw ACC at any deployment prior π via `π · AccR + (1-π) · RejR`
2. **Report the predictor and quality metrics on the balanced test:**
   - Axis A: `ρ(score, pct_rating)`
   - Axis B: `balanced ACC` (and optionally AUC)
3. **Report raw ACC at the natural prior** if you want a "what users see" deployment number — but compute it analytically rather than running a separate evaluation. Optionally cross-check by also running on a natural-prior test set, but the difference should be small (≤ 1pp on arxiv, ≤ 5pp on ICLR due to population mix).
4. **Skip these:**
   - Raw ACC on the balanced test (it's just balanced ACC under another name when prior is 50/50).
   - Balanced ACC on the natural test (gives the same answer as on balanced; only useful as a prior-invariance sanity check).
5. **If a model's per-class recalls are wildly asymmetric (one is < 10%)**, flag it as a degenerate / constant-class predictor — its raw ACC is meaningless even at the deployment prior.

This collapses the test-ratio question: **build one balanced test set, run inference once, report all metrics from there.** The natural-prior raw ACC for any deployment scenario is a one-line calculation, not a separate evaluation.
