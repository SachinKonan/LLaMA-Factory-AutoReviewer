# Predicting Paper Acceptance at Machine Learning Conferences via Fine-Tuned Large Language Models: A Multimodal Analysis

**Abstract.** We study the problem of predicting paper acceptance decisions at top machine learning conferences using only paper content, without access to reviewer scores, rebuttals, or meta-reviews. We fine-tune Qwen2.5-7B (text) and Qwen2.5-VL-7B (vision) models on ICLR papers spanning 2020--2026 and achieve up to 70.4% accuracy---substantially above the TF-IDF baseline (59.6%) and zero-shot LLM baselines (~52%). We provide theoretical grounding via the data processing inequality, showing that content-only prediction is fundamentally lossy, and we characterize the empirical ceiling imposed by reviewer inconsistency using the NeurIPS 2021 consistency study (~72--77% inter-committee agreement). Our multimodal analysis reveals that vision models achieve higher accept recall but lower reject recall than text models, and that the two modalities exhibit complementary error patterns amenable to ensembling. We further demonstrate that 2026 (ICLR 2026) presents a measurably harder prediction target due to compressed rating gaps and noisier labels---consistent with our theoretical predictions. Our dataset, spanning ~25,000 papers with binary accept/reject labels, is the largest content-only peer review prediction benchmark to date.

---

## 1. Introduction

The peer review system at machine learning conferences is under unprecedented strain. ICLR 2025 received over 11,000 submissions, and ICLR 2026 surpassed that figure. Simultaneously, AI-assisted paper writing tools have lowered the barrier to submission, further inflating the review workload. Area chairs must synthesize noisy, sometimes contradictory reviewer signals into binary accept/reject decisions under severe time pressure.

Several recent systems attempt to automate aspects of peer review. Sakana AI's "The AI Scientist" generates end-to-end papers and self-reviews; DeepReviewer and PaperDecision produce structured reviews from paper content. However, these systems overwhelmingly focus on *review generation* rather than *decision prediction*, and when they do predict decisions, they typically train on non-anonymized data (including author names, affiliations, and citation counts) that would be unavailable to a blind reviewer.

Our work departs from this paradigm in three ways. First, we focus exclusively on *acceptance prediction from paper content alone*, treating the problem as binary classification without access to any review-stage information. Second, we enforce rigorous content normalization: author names, affiliations, acknowledgments, and references are removed from all inputs, ensuring the model cannot exploit prestige signals. Third, we systematically compare text-only and vision-based (rendered PDF page) input modalities, revealing complementary strengths that motivate ensemble approaches.

We ground our analysis in information-theoretic principles. The data processing inequality guarantees that predicting the final decision from paper content alone cannot exceed the mutual information between content and decision---which is strictly less than the mutual information between the full review chain and the decision. The NeurIPS 2021 consistency experiment provides a direct empirical ceiling of ~72--77% agreement between independent review committees, establishing an upper bound for any single-pass prediction system. Our best models achieve 66.5%--70.9% depending on year and configuration, approaching this empirical ceiling.

**Our contributions are:**

1. **The largest content-only peer review prediction dataset**, spanning ~25,000 ICLR papers (2020--2026) with binary labels, cleaned text, and rendered PDF pages.
2. **Theoretical analysis** via the data processing inequality and Fano's inequality, establishing fundamental limits on content-only prediction accuracy.
3. **Systematic multimodal comparison** between text-only (Qwen2.5-7B) and vision-based (Qwen2.5-VL-7B) fine-tuned models, revealing complementary error patterns.
4. **Temporal generalization analysis** showing that models trained on 2020--2025 data can predict 2025 OOD papers at near in-distribution accuracy (70.9%), but 2026 accuracy drops to 66.5%.
5. **Confidence-based selective prediction analysis**, showing that restricting predictions to the most confident 50% of samples boosts accuracy from ~68% to ~77%.

---

## 2. Problem Formulation and Theoretical Limits

### 2.1 Problem Statement

Let $X$ denote the content of a submitted paper (text or rendered pages), and let $Z \in \{\text{accept}, \text{reject}\}$ denote the final decision. Our goal is to learn a classifier $f: X \to Z$ that maximizes $\Pr[f(X) = Z]$ using only content-level features.

Crucially, the actual decision process involves a chain of intermediate random variables:

$$X \to Y_1 \to Y_2 \to Y_3 \to Y_4 \to Z$$

where $Y_1$ denotes initial reviews, $Y_2$ the author rebuttal, $Y_3$ updated reviews post-discussion, and $Y_4$ the meta-review/AC decision. Each transition introduces additional information (reviewer expertise, rebuttal arguments, discussion dynamics) that is not accessible from $X$ alone.

### 2.2 Data Processing Inequality

By the data processing inequality, for any Markov chain $X \to Y \to Z$:

$$I(X; Z) \leq I(Y; Z)$$

This means the mutual information between paper content and the final decision is bounded above by the mutual information between the full review record and the decision. Since the review process aggregates information from multiple expert reviewers, their rebuttals, and meta-reviewer synthesis, $I(Y; Z) \gg I(X; Z)$ in general.

**ICLR 2026 special case.** The 2026 review cycle eliminated the traditional author rebuttal period and compressed the review timeline. Effectively, the chain simplifies to:

$$X \to Y_1 \to Z$$

where $Y_1$ is a single round of reviews without rebuttal updates. This makes $Y_1$ a noisier signal for $Z$, as authors cannot correct reviewer misunderstandings. Paradoxically, while this should make $I(Y_1; Z) < I(Y; Z)$ for the full chain, it does *not* necessarily increase $I(X; Z)$---the content-only prediction problem remains equally or more difficult because the decision itself becomes noisier.

### 2.3 Empirical Ceiling: NeurIPS 2021 Consistency Study

The NeurIPS 2021 consistency experiment provides a direct empirical estimate of the noise floor. In that experiment, ~10% of submissions were independently reviewed by two separate committees. The agreement rate between the two committees was approximately **72--77%**, establishing that even *two full review processes* disagree on roughly a quarter of papers.

This places an empirical upper bound on any single-pass prediction system, whether human or automated. Our best model achieves 70.4% accuracy, which is remarkably close to this ceiling, suggesting that content-based prediction is approaching the fundamental limit imposed by the stochasticity of the review process itself.

### 2.4 Rating Separability: pct_rating AUC

We can estimate a complementary, year-specific measure of class separability using the `pct_rating` variable---the average normalized reviewer rating for each paper. If we had access to `pct_rating` at test time, the optimal classifier would simply threshold it at the decision boundary. The AUC of `pct_rating` as a predictor of the accept/reject decision measures how well-separated the two classes are in rating space.

| Year | pct\_rating AUC | Rating-Based Ceiling | Model Acc |
|------|----------------|---------------------|-----------|
| 2020 | 0.951 | 93.5% | 71.7% |
| 2021 | 0.950 | 92.1% | 69.5% |
| 2022 | 0.941 | 91.3% | 66.5% |
| 2023 | 0.972 | 93.8% | 67.9% |
| 2025 | 0.939 | 89.1% | 70.9% |
| 2026 | 0.893 | 86.5% | 66.5% |

Note: The "Rating-Based Ceiling" represents the accuracy achievable with perfect access to aggregate reviewer ratings (not the Bayes-optimal content-only ceiling). The true content-only ceiling is better approximated by the NeurIPS 2021 consistency bound (~72--77%). The gap between our model accuracy and the rating-based ceiling quantifies the information lost by bypassing the review chain entirely.

### 2.5 Fano's Inequality

Fano's inequality provides a formal lower bound on the error probability of any classifier:

$$P_e \geq \frac{H(Z|X) - 1}{\log(|Z| - 1)}$$

For binary classification ($|Z| = 2$), this simplifies to $P_e \geq H(Z|X) - 1$. While $H(Z|X)$ is not directly observable, the NeurIPS 2021 consistency experiment suggests the Bayes error rate for a single evaluation is at least ~23--28% (the disagreement rate between two independent committees). This yields $H(Z|X) \approx h(0.25) \approx 0.81$ bits, implying a non-trivial irreducible error.

### 2.6 The Rating Gap as a Predictability Proxy

The gap between mean `pct_rating` for accepted and rejected papers provides a simple proxy for class separability:

| Year | Accept pct\_rating | Reject pct\_rating | Gap |
|------|-------------------|-------------------|------|
| 2020 | 0.8082 | 0.3337 | 0.47 |
| 2021 | 0.8063 | 0.3405 | 0.47 |
| 2022 | 0.7647 | 0.3417 | 0.42 |
| 2023 | 0.7904 | 0.2897 | 0.50 |
| 2025 | 0.7466 | 0.3117 | 0.43 |
| 2026 | 0.7382 | 0.3487 | 0.39 |

2026 has the smallest gap (0.39), indicating the greatest overlap between accept and reject rating distributions. This is consistent with the lower AUC (0.893) and suggests that the 2026 decision boundary is harder to learn from any signal, including content.

![Rating gap across years](../../figures/report_rating_gap.png)

---

## 3. Data Pipeline and Dataset Construction

### 3.1 Source Data

We collect papers from ICLR 2020, 2021, 2022, 2023, 2025, and 2026 via the OpenReview API. **We exclude ICLR 2024** because the 2024 review data was not fully publicly available with consistent decision labels at the time of dataset construction. For each paper, we obtain:
- The submitted PDF
- The final accept/reject decision
- Reviewer ratings and confidence scores (used for analysis, not as model input)
- Paper metadata (title, abstract, categories)

### 3.2 Text Extraction: MinerU Pipeline

We use MinerU, a high-fidelity PDF-to-text conversion tool, to extract clean text from each paper PDF. The pipeline performs:
1. **Layout analysis**: Identifies text blocks, figures, tables, equations, and captions
2. **OCR fallback**: For scanned or image-heavy PDFs
3. **Reference removal**: We strip the references/bibliography section to prevent the model from exploiting citation patterns as a prestige signal
4. **Author de-identification**: Author names, affiliations, and acknowledgment sections are removed

For the **vision modality**, we render each PDF page as an image and provide the first $k$ pages (typically 8--10) as visual input to the VLM. This preserves layout, figures, equations, and typographic quality that text extraction may lose.

For the **text+images** variant, we combine extracted text with embedded figure images, providing a middle ground between pure text and full vision input.

### 3.3 Input Modalities

We use three input representations throughout this work:

| Modality Name | Description |
|---------------|-------------|
| **Text** | Extracted clean text from MinerU (no figures) |
| **Text+Images** | Extracted text with embedded figure images |
| **Vision** | Rendered PDF pages as images (full visual input to VLM) |

### 3.4 Dataset Variants and Sizes

We construct multiple dataset variants along three axes:

| Axis | Options | Notes |
|------|---------|-------|
| Year range | 2020-2025, 2017-2025, 2024-2025 | Controls temporal coverage |
| Balancing | balanced, trainagreeing | balanced = 50/50 accept/reject; trainagreeing = separate filtered dataset |
| Modality | Text, Vision, Text+Images | Input representation |

**Dataset sizes (v7 splits):**

| Dataset Variant | Modality | Train | Val | Test | Best Acc |
|----------------|----------|-------|-----|------|----------|
| 2020-2025 Balanced | Text | 12,745 | ~1,000 | 2,024 | 66.2% |
| 2020-2025 Balanced | Text+Images | 12,451 | ~1,000 | 1,974 | 67.0% |
| 2020-2025 Balanced | Vision | 12,765 | ~1,000 | 2,026 | 69.8% |
| 2017-2025 Balanced | Text | 14,577 | ~1,100 | 2,234 | 65.7% |
| 2017-2025 Balanced | Text+Images | 14,211 | ~1,100 | 2,178 | 67.0% |
| 2017-2025 Balanced | Vision | 14,594 | ~1,100 | 2,236 | 69.5% |
| 2024-2025 Balanced | Text | 3,889 | ~500 | 1,189 | 63.8% |
| 2024-2025 Balanced | Text+Images | 3,772 | ~500 | 1,161 | 63.5% |
| 2024-2025 Balanced | Vision | 3,893 | ~500 | 1,190 | 70.9% |

**Trainagreeing dataset** (separate from balanced variants):

| Modality | Train | Val | Test | Best Acc |
|----------|-------|-----|------|----------|
| Text | 8,296 | ~1,000 | 2,024 | 66.9% |
| Text+Images | 8,093 | ~1,000 | 1,974 | 68.2% |
| Vision | 8,315 | ~1,000 | 2,026 | 70.4% |

The **trainagreeing** dataset is constructed by filtering the balanced training set to retain only examples where early-checkpoint predictions agree with the ground truth label. This produces a smaller but cleaner training set. Note: the val/test sets are identical to the balanced variant; only the training set differs.

![Dataset sizes and accuracy](../../figures/dataset_sizes.png)

### 3.5 Label Quality

Not all labels are equally clean. We measure label "cleanliness" using the pct_rating as a proxy: accepted papers with pct_rating >= 0.6 and rejected papers with pct_rating <= 0.4 are considered "clean" labels (i.e., the decision aligns with the aggregate reviewer sentiment).

| Year | % Accepts w/ pct >= 0.6 | % Rejects w/ pct <= 0.4 |
|------|------------------------|------------------------|
| 2020 | 95.7% | 60.9% |
| 2021 | 91.5% | 64.6% |
| 2022 | 87.2% | 62.4% |
| 2023 | 89.6% | 72.1% |
| 2025 | 82.6% | 67.0% |
| 2026 | 78.2% | 60.7% |

For 2026, 21.8% of accepted papers have pct_rating < 0.6, meaning over one-fifth of accepted papers were accepted *despite* receiving below-threshold aggregate ratings. This high label noise directly impacts model training and evaluation.

![Label cleanliness across years](../../figures/report_label_cleanliness.png)

![Rating distribution by decision](../../figures/latex/data/pct_rating_violin_by_decision.png)

![Year distribution in our dataset](../../figures/latex/data/year_distribution.png)

### 3.6 Comparison to Prior Datasets

| Dataset | Papers | Venues | Years | Content | Labels | Anonymized |
|---------|--------|--------|-------|---------|--------|------------|
| PeerRead | ~14K | ACL, NIPS, ICLR | 2013-2017 | Abstract only | Accept/Reject | No |
| MOPRD | ~6K | ICLR | 2017-2020 | Full text | Multi-class | No |
| **Ours** | **~25K** | **ICLR** | **2020-2026** | **Full text + vision** | **Binary** | **Yes** |

Our dataset is approximately 2x larger than PeerRead and 4x larger than MOPRD, covers more recent years (including 2025 and 2026), provides both text and vision modalities, and enforces author anonymization.

---

## 4. Zero-Shot and Baseline Results

### 4.1 TF-IDF Baseline

As a non-neural baseline, we train a TF-IDF + logistic regression classifier on the extracted text. Using unigram and bigram features with L2 regularization, this baseline achieves **59.6% accuracy** on the balanced test set.

![TF-IDF feature importance](../../figures/latex/baseline/tfidf_feature_importance.png)

![TF-IDF comparison across datasets](../../figures/latex/baseline/tfidf_comparison.png)

The TF-IDF model reveals interpretable but brittle signals: certain technical terms (e.g., "reinforcement", "generative") correlate weakly with acceptance, but the model largely captures superficial lexical patterns rather than deep content understanding.

### 4.2 Zero-Shot LLM: Qwen 3.5-122B

We evaluate Qwen 3.5-122B (the largest available Qwen model at evaluation time) in a zero-shot, blind review setting. The model is prompted to act as an expert ML reviewer and predict accept/reject for each paper based on its content.

**Overall results:**
- **Accuracy**: 52.1% (21,960 papers evaluated, barely above random)
- **Precision (Accept)**: 74.96%
- **Recall (Accept)**: 4.67%
- **F1 (Accept)**: 8.78%

**Confusion matrix:**

| | Predicted Accept | Predicted Reject |
|---|---|---|
| **GT Accept** | 506 (TP) | 10,339 (FN) |
| **GT Reject** | 169 (FP) | 10,946 (TN) |

The model is *extremely* conservative: it predicts accept for only 3.1% of papers (675 out of 21,960), compared to the ground truth accept rate of 49.4% (10,845 out of 21,960). This extreme reject bias means the model achieves near-perfect reject recall (98.5%) but catastrophic accept recall (4.67%).

**Per-year breakdown:**

| Year | Accuracy | N |
|------|----------|------|
| 2020 | 54.5% | 1,293 |
| 2021 | 54.9% | 1,529 |
| 2022 | 53.3% | 2,040 |
| 2023 | 53.1% | 2,847 |
| 2025 | 50.9% | 6,247 |
| 2026 | 51.6% | 8,004 |

Performance is near-random across all years, with a slight degradation on more recent years.

**Per rating bin breakdown:**

| Rating Bin | Accuracy | N |
|------------|----------|------|
| Strong Reject (0--0.3) | 97.7% | 4,892 |
| Lean Reject (0.3--0.45) | 85.7% | 3,453 |
| Borderline (0.45--0.55) | 68.6% | 1,960 |
| Lean Accept (0.55--0.7) | 37.3% | 5,226 |
| Strong Accept (0.7+) | 11.5% | 6,429 |

The zero-shot model is accurate only on clear rejects. For papers in the "lean accept" and "strong accept" rating bins, accuracy plummets because the model systematically predicts reject for these papers. The zero-shot LLM lacks *venue calibration*: it does not know where the acceptance threshold lies for ICLR, and defaults to an overly pessimistic assessment.

### 4.3 Zero-Shot LLM: Gemini

We also evaluate Gemini in a zero-shot setting for comparison with our fine-tuned models.

![Gemini vs fine-tuned model comparison](../../figures/latex/baseline/gemini_vs_finetuned.png)

![Gemini detailed comparison](../../figures/latex/baseline/gemini_detailed_comparison.png)

### 4.4 Discussion: Why Zero-Shot Fails

The failure of zero-shot LLMs to predict acceptance reveals a fundamental mismatch: LLMs can assess paper quality in an absolute sense (identifying clear strengths and weaknesses), but they cannot calibrate to a specific venue's acceptance threshold. The threshold is an emergent property of the reviewer pool, the submission pool, and the AC's decision function---information that is not accessible from any single paper's content. Fine-tuning provides this calibration by exposing the model to the empirical accept/reject boundary.

---

## 5. Supervised Fine-Tuning Experiments

### 5.1 Setup

We fine-tune two model families:
- **Qwen2.5-7B** for text input (extracted and cleaned paper text)
- **Qwen2.5-VL-7B** for vision input (rendered PDF pages)

Training configuration:
- **Epochs**: 4
- **Learning rate schedule**: cosine decay with warm-up, then constant (cosine_then_constant)
- **Batch sizes**: 16 or 32 (swept)
- **Learning rates**: 1e-6, 2e-6, 5e-6 (swept)
- **LoRA**: rank 64, alpha 128
- **Precision**: bf16

The optimal configuration is **batch size 16, learning rate 1e-6** for vision models and **batch size 32, learning rate 1e-6** for text models.

### 5.2 Year Range Ablation

A key design decision is which years to include in training. We compare three year ranges across all three modalities:

| Year Range | Text | Text+Images | Vision |
|------------|------|-------------|--------|
| 2020-2025 (balanced) | 66.2% | 67.0% | 69.8% |
| 2017-2025 (balanced) | 65.7% | 67.0% | 69.5% |
| 2024-2025 (balanced) | 63.8% | 63.5% | **70.9%** |

![Year range ablation](../../figures/year_range_ablation.png)

**Key findings:**
- **Vision consistently outperforms text and text+images** across all year ranges, with the best vision model reaching 70.9% (2024-2025 balanced).
- **Adding pre-2020 data provides marginal benefit** for text (65.7% vs 66.2%) and is roughly neutral for vision (69.5% vs 69.8%).
- **Training on 2024+2025 only** yields the best vision accuracy (70.9%) but the worst text accuracy (63.8%), suggesting vision models are more robust to small training sets.
- The **text+images** modality performs between pure text and pure vision, suggesting figures carry meaningful signal.

### 5.3 Trainagreeing Dataset Results

The **trainagreeing** dataset is a separate, filtered training set that retains only examples where early-checkpoint predictions agree with the ground truth label. This produces a smaller but higher-quality training set (8.3K vs 12.7K examples).

| Modality | Balanced Acc | Trainagreeing Acc | Delta |
|----------|-------------|-------------------|-------|
| Text | 66.2% | **66.9%** | +0.7 |
| Text+Images | 67.0% | **68.2%** | +1.2 |
| Vision | 69.8% | **70.4%** | +0.6 |

![Balanced vs Trainagreeing](../../figures/year_range_ablation_trainagreeing.png)

Trainagreeing consistently improves overall accuracy despite using ~35% fewer training examples. However, it shifts the recall balance toward accepts:

| Config | Modality | Accuracy | Accept Recall | Reject Recall |
|--------|----------|----------|---------------|---------------|
| Balanced | Vision | 69.8% | 68.9% | 70.8% |
| Trainagreeing | Vision | **70.4%** | 80.3% | 60.5% |
| Balanced | Text | 66.2% | 65.2% | 67.2% |
| Trainagreeing | Text | **66.9%** | 78.5% | 55.4% |

The trainagreeing vision model achieves our best overall accuracy of **70.4%** but with a 20-point gap between accept recall (80.3%) and reject recall (60.5%).

### 5.4 Summary: Best Configurations

| Config | Modality | Accuracy | Accept Recall | Reject Recall | Pred Accept Rate |
|--------|----------|----------|---------------|---------------|------------------|
| Trainagreeing | Vision | **70.4%** | 80.3% | 60.5% | 59.9% |
| 2024-2025 Balanced | Vision | **70.9%** | 63.2% | 78.7% | 42.3% |
| 2020-2025 Balanced | Vision | 69.8% | 68.9% | 70.8% | 49.1% |
| Trainagreeing | Text+Images | 68.2% | 71.7% | 64.7% | 53.4% |
| 2020-2025 Balanced | Text+Images | 67.0% | 77.1% | 57.0% | 60.0% |
| Trainagreeing | Text | 66.9% | 78.5% | 55.4% | 61.6% |
| 2020-2025 Balanced | Text | 66.2% | 65.2% | 67.2% | 49.0% |

The modality ranking is consistent: **Vision > Text+Images > Text**. The trainagreeing dataset provides a small but consistent boost. The best overall accuracy (70.9%) comes from the 2024-2025 balanced vision model, which achieves high reject recall (78.7%) but lower accept recall (63.2%).

![Train size vs accuracy](../../figures/train_size_vs_accuracy.png)

---

## 6. Modality Analysis: Text vs Vision (2020--2025)

This section presents detailed modality comparison results from the `text_vs_vision_v7` analysis, trained on 2020--2023+2025 data (excluding 2026).

### 6.1 Overall Metrics

| Metric | SFT Text | SFT Vision | RL Text |
|--------|----------|------------|---------|
| Training Size | 12,745 | 12,765 | N/A |
| Testing Size | 1,504 | 1,506 | 1,504 |
| Overall Accuracy | **68.9%** | 67.0% | 62.5% |
| IKF Accuracy (2020+2023) | 68.6% | **69.7%** | 63.9% |
| OOKF Accuracy (2025) | **68.5%** | 66.1% | 61.7% |
| Accept Recall | 72.6% | **80.9%** | 69.2% |
| Reject Recall | **65.1%** | 53.1% | 55.8% |
| Pred Accept Rate | 53.8% | 63.9% | 56.7% |

In this configuration (without 2026 data), text slightly outperforms vision overall (68.9% vs 67.0%), but the pattern reverses on in-distribution years (69.7% vision vs 68.6% text for 2020+2023). The two modalities exhibit strikingly different recall profiles:
- **Vision** has much higher accept recall (80.9% vs 72.6%) but much lower reject recall (53.1% vs 65.1%)
- **Text** is more balanced and produces a lower predicted accept rate (53.8% vs 63.9%)

The RL-trained text model underperforms both SFT variants (62.5%), suggesting that RLHF-style optimization does not help for this task.

### 6.2 Per-Year Accuracy

![Accuracy by year --- text vs vision (v7)](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/accuracy_by_year.png)

### 6.3 Agreement and Disagreement Analysis

The two modalities agree on 73.8% of predictions but diverge on 26.2%---the disagreement set is where ensemble methods can potentially improve.

![Agreement breakdown](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/agreement_breakdown.png)

![Prediction agreement Venn diagram](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/prediction_agreement_venn.png)

![Disagreement analysis](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/disagreement_analysis.png)

### 6.4 What Drives Disagreement?

We analyze the features of papers where text and vision disagree using feature importance analysis.

![Disagreement feature importance](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/disagreement_feature_importance.png)

Papers with high pct_rating (strong accepts) and short length tend to have vision predict accept while text predicts reject. Conversely, papers with low pct_rating and complex mathematical content tend to have text predict reject while vision predicts accept, possibly because the vision model cannot fully parse dense equations.

### 6.5 Factor Analysis

We decompose paper features into latent factors to understand what each modality captures.

![Factor analysis](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/factor_analysis.png)

![Factor prediction R-squared](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/factor_prediction_r2.png)

### 6.6 Rating Interval Analysis

![Rating interval analysis](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/rating_interval_analysis.png)

Both models achieve near-perfect accuracy on extreme rating intervals (pct_rating < 0.2 or > 0.8) but diverge significantly in the borderline zone (0.4--0.6). In this zone, text achieves ~60% accuracy while vision drops to ~55%, reflecting vision's tendency to optimistically predict accept for borderline papers.

### 6.7 Recall Analysis

![Recall bars](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/recall_bars.png)

### 6.8 Ensemble Results

Combining text and vision predictions via simple ensembling (majority vote, weighted average, or learned combiner) can exploit their complementary error patterns.

![Ensemble accuracy bars (val)](../../results/summarized_investigation/text_vs_vision_v7/ensemble_val/accuracy_bars.png)

![Ensemble accuracy by year (val)](../../results/summarized_investigation/text_vs_vision_v7/ensemble_val/accuracy_by_year.png)

![Ensemble feature importance (val)](../../results/summarized_investigation/text_vs_vision_v7/ensemble_val/feature_importance.png)

With additional training data for the ensemble combiner:

![Ensemble accuracy bars (val+train2k)](../../results/summarized_investigation/text_vs_vision_v7/ensemble_val+train2k/accuracy_bars.png)

![Ensemble accuracy by year (val+train2k)](../../results/summarized_investigation/text_vs_vision_v7/ensemble_val+train2k/accuracy_by_year.png)

---

## 7. Modality Analysis: Including 2026

This section extends the modality analysis to include 2026 data in both training and evaluation, using the `text_vs_vision_2026` configuration.

### 7.1 Overall Metrics

| Metric | SFT Text (bz32) | SFT Vision (bz16) |
|--------|-----------------|-------------------|
| Training Size | 21,141 | 21,174 |
| Testing Size | 2,495 | 2,498 |
| Overall Accuracy | 67.0% | **68.4%** |
| IKF Accuracy (2020+2023) | **69.3%** | 69.1% |
| OOKF Accuracy (2025+2026) | 66.0% | **68.3%** |
| Accept Recall | 66.0% | **71.0%** |
| Reject Recall | **67.9%** | 65.7% |
| Pred Accept Rate | 49.1% | 52.6% |

When 2026 data is included, the modality ranking reverses: **vision now outperforms text** (68.4% vs 67.0%), driven by a large gap on OOD data (68.3% vs 66.0% on 2025+2026). This suggests vision models are more robust to the distribution shift introduced by 2026.

The recall asymmetry persists: vision has higher accept recall (71.0% vs 66.0%) and lower reject recall (65.7% vs 67.9%), though the gap is smaller than in the v7 analysis.

### 7.2 Per-Year Accuracy

![Accuracy by year --- text vs vision (2026)](../../results/summarized_investigation/text_vs_vision_2026/modality_analysis/accuracy_by_year.png)

### 7.3 Agreement and Disagreement

![Agreement breakdown (2026)](../../results/summarized_investigation/text_vs_vision_2026/modality_analysis/agreement_breakdown.png)

![Prediction agreement Venn (2026)](../../results/summarized_investigation/text_vs_vision_2026/modality_analysis/prediction_agreement_venn.png)

![Disagreement analysis (2026)](../../results/summarized_investigation/text_vs_vision_2026/modality_analysis/disagreement_analysis.png)

### 7.4 What Drives Disagreement?

![Disagreement feature importance (2026)](../../results/summarized_investigation/text_vs_vision_2026/modality_analysis/disagreement_feature_importance.png)

### 7.5 Factor Analysis

![Factor analysis (2026)](../../results/summarized_investigation/text_vs_vision_2026/modality_analysis/factor_analysis.png)

![Factor prediction R-squared (2026)](../../results/summarized_investigation/text_vs_vision_2026/modality_analysis/factor_prediction_r2.png)

### 7.6 Rating Interval and Recall

![Rating interval analysis (2026)](../../results/summarized_investigation/text_vs_vision_2026/modality_analysis/rating_interval_analysis.png)

![Recall bars (2026)](../../results/summarized_investigation/text_vs_vision_2026/modality_analysis/recall_bars.png)

### 7.7 Key Comparison: v7 vs 2026-Inclusive

| Metric | v7 Text | v7 Vision | 2026 Text | 2026 Vision |
|--------|---------|-----------|-----------|-------------|
| Overall Accuracy | **68.9%** | 67.0% | 67.0% | **68.4%** |
| OOD Accuracy | **68.5%** | 66.1% | 66.0% | **68.3%** |
| Accept Recall | 72.6% | **80.9%** | 66.0% | **71.0%** |
| Reject Recall | **65.1%** | 53.1% | **67.9%** | 65.7% |

Including 2026 data reduces text accuracy (-1.9 pp) more than vision accuracy (+1.4 pp), and vision becomes the dominant modality. This suggests that the visual features of papers (layout quality, figure density, typographic professionalism) are more robust to the temporal distribution shift than text content features, which may be more sensitive to evolving topic distributions.

---

## 8. Why 2026 Performance Declines

### 8.1 The 2026 Distribution Shift

ICLR 2026 introduced several changes to the review process that fundamentally altered the decision-making dynamics:
1. **Compressed review timeline**: shorter rebuttal/discussion period
2. **Larger submission pool**: increased number of submissions straining reviewer bandwidth
3. **Modified scoring rubrics**: changes in how reviewers assign ratings

These changes manifest in our data as measurable shifts in the rating distribution and label quality.

### 8.2 Rating Gap Compression

The gap between mean accept and reject pct_rating is smallest for 2026 (0.39), compared to 0.42--0.50 for other years. This means the two classes are less separable in rating space, making the decision boundary harder to learn.

![Rating gap across years](../../figures/report_rating_gap.png)

### 8.3 Rating Separability and AUC

| Year | pct\_rating AUC | Rating-Based Ceiling | Model Acc |
|------|----------------|---------------------|-----------|
| 2020 | 0.951 | 93.5% | 71.7% |
| 2021 | 0.950 | 92.1% | 69.5% |
| 2022 | 0.941 | 91.3% | 66.5% |
| 2023 | 0.972 | 93.8% | 67.9% |
| 2025 | 0.939 | 89.1% | 70.9% |
| 2026 | **0.893** | **86.5%** | **66.5%** |

2026 stands out with the lowest AUC (0.893) and lowest rating-based ceiling (86.5%). Ratings are least predictive of decisions in 2026, meaning even with perfect access to reviewer ratings, 13.5% of papers would be misclassified.

### 8.4 Label Noise

As shown in Section 3.5, 21.8% of 2026 accepted papers have pct_rating < 0.6, compared to 4.3% for 2020 and 10.4% for 2025. These are papers accepted despite lukewarm or negative reviewer sentiment, representing cases where the AC overrode the aggregate rating. Training on these "noisy accepts" teaches the model conflicting signals.

![Label cleanliness](../../figures/report_label_cleanliness.png)

### 8.5 Model Performance Decomposition

![Model performance by year](../../figures/report_model_performance.png)

The per-year accuracy breakdown shows that 2026 is consistently the hardest year across all model configurations, even when 2026 data is included in training. The model achieves 66.5% on 2026 versus 70.9% on 2025 (a 4.4 pp gap), despite 2026 comprising the largest cohort in the training set.

### 8.6 Threshold Sensitivity

![Threshold accuracy analysis](../../figures/report_threshold_accuracy.png)

The threshold analysis reveals that the optimal decision threshold shifts for 2026 relative to other years, further confirming that the accept/reject boundary has moved.

### 8.7 Summary Dashboard

![Summary dashboard](../../figures/report_summary_dashboard.png)

---

## 9. Deeper Analysis

### 9.1 Confidence and Calibration Analysis

We extract per-sample prediction confidence from the model's token log-probabilities. The confidence is the probability the model assigns to its predicted token (Accept or Reject) at the decision position.

![Confidence and calibration analysis](../../figures/confidence_calibration.png)

**Key findings from calibration analysis:**

1. **The models are moderately well-calibrated**: When the model predicts with 80% confidence, it is correct roughly 75--80% of the time. Both text and vision models show slight overconfidence at the high end.

2. **Vision models are more confident overall**: The vision model produces a higher proportion of high-confidence predictions compared to the text model.

3. **Correct predictions cluster at high confidence**: The confidence distribution for correct predictions is strongly right-skewed, while incorrect predictions have a flatter, lower confidence profile. This separation enables confidence-based selective prediction.

### 9.2 Coverage vs Accuracy Tradeoff

A key practical question: if we only make predictions on samples where the model is sufficiently confident, what accuracy can we achieve, and how many samples must we abstain on?

![Coverage vs accuracy tradeoff](../../figures/coverage_vs_accuracy.png)

| Coverage | Text Accuracy | Vision Accuracy |
|----------|---------------|-----------------|
| 100% (all) | 67.6% | 67.6% |
| 90% | 70.4% | 70.0% |
| 75% | 73.4% | 72.5% |
| 50% | 76.2% | 76.9% |
| 25% | ~84% | ~84% |

At 50% coverage (keeping only the most confident half of predictions), both models exceed 76% accuracy. At 25% coverage, accuracy reaches ~84%. This has practical implications: a system that only flags papers above a confidence threshold could provide highly reliable predictions on a meaningful fraction of submissions.

### 9.3 Structural Features by Decision

We extract structural features from each paper (number of pages, figures, tables, equations, sections) and analyze their correlation with acceptance decisions.

![Structural features by decision](../../figures/latex/analysis/structural_features_by_decision.png)

Accepted papers tend to have more figures, more tables, and longer page counts, but these features alone are weakly predictive (the distributions overlap substantially).

### 9.4 Feature Correlation with Modality

![Feature correlation with modality](../../figures/latex/analysis/feature_correlation_with_modality.png)

Text and vision models weight structural features differently. Vision models are more sensitive to figure quality and layout, while text models respond more to equation density and section organization.

### 9.5 Mediation Analysis

We perform a mediation analysis to understand the causal pathway from paper features to model predictions, mediated by underlying paper quality (proxied by pct_rating).

![Mediation path coefficients](../../figures/latex/analysis/mediation_path_coefficients.png)

![Mediation indirect effects](../../figures/latex/analysis/mediation_indirect_effects.png)

![Mediation proportion](../../figures/latex/analysis/mediation_proportion.png)

![Mediation scatter grid](../../figures/latex/analysis/mediation_scatter_grid.png)

The mediation analysis reveals that structural features influence predictions both directly and indirectly through their correlation with quality. The indirect (mediated) path accounts for approximately 40--60% of the total effect for features like figure count and page length, suggesting the model partially learns a quality proxy.

### 9.6 Performance Stratification

![Accuracy vs rating standard deviation](../../figures/latex/ablations/acc_vs_rating_std.png)

![Accuracy heatmap (2D)](../../figures/latex/ablations/acc_heatmap_2d.png)

![Accuracy vs pct\_rating (violin)](../../figures/latex/analysis/acc_vs_pct_rating_violin.png)

![Borderline analysis](../../figures/latex/analysis/borderline_analysis.png)

Model accuracy drops sharply for papers with pct_rating near 0.5 (the decision boundary) and with high rating standard deviation (high reviewer disagreement). The 2D heatmap reveals that the hardest papers combine borderline ratings with high reviewer variance---exactly the papers where the decision is most influenced by factors beyond the paper content (reviewer assignment, discussion dynamics, AC judgment).

### 9.7 Category Analysis

![Accuracy by category](../../figures/latex/analysis/accuracy_by_category.png)

![Category difficulty ranking](../../figures/latex/analysis/category_difficulty_ranking.png)

![Category-modality interaction](../../figures/latex/analysis/category_modality_interaction.png)

Model accuracy varies substantially across paper categories. Reinforcement learning and generative model papers are easier to predict, while theory and optimization papers are harder. The category-modality interaction plot reveals that vision models have an advantage on empirically-heavy papers (with many figures and tables) while text models perform better on theoretically-dense papers.

### 9.8 Best Model Decomposition

![Best model confusion matrix](../../figures/latex/ablations/best_model_confusion.png)

![Best model --- stratified performance](../../figures/latex/ablations/best_model_stratified.png)

![Best model --- by year](../../figures/latex/ablations/best_model_by_year.png)

![Best model --- error analysis](../../figures/latex/ablations/best_model_error_analysis.png)

Error analysis of the best model (trainagreeing vision, 70.4%) reveals that false negatives (missed accepts) tend to be papers with unconventional structure or sparse figures, while false positives (incorrectly predicted accepts) tend to be well-formatted papers with superficially strong presentation but weak technical content.

### 9.9 Modality Wins Analysis

![Modality wins analysis](../../figures/latex/ablations/modality_wins_analysis.png)

We identify papers where one modality is correct and the other is wrong. Vision "wins" (correct when text is wrong) on papers with more figures and better visual presentation. Text "wins" on papers with dense mathematical content and minimal figures.

### 9.10 Disagreement Features

![Disagreement features](../../figures/latex/ablations/disagreement_features.png)

The features most predictive of text-vision disagreement are: number of figures (vision-favoring), equation density (text-favoring), page count, and paper category.

---

## 10. Discussion

### 10.1 Fundamental vs Modeling Limitations

Our analysis distinguishes two sources of prediction error:

1. **Fundamental limitations**: The data processing inequality guarantees that bypassing the review chain loses information. The NeurIPS 2021 consistency experiment establishes that ~23--28% of decisions are inherently stochastic with respect to any single evaluation, placing an empirical ceiling of ~72--77% on single-pass prediction accuracy. Our best model (70.4%) is remarkably close to this ceiling.

2. **Modeling limitations**: While our models approach the consistency ceiling, they are well below the rating-based ceiling (86.5%--93.8%), suggesting that there exists signal in the review process that content-only models cannot capture. Key gaps include:
   - Limited context window (current models see at most 8--10 pages)
   - No multi-paper comparison (reviewers benchmark papers against each other)
   - No awareness of submission pool composition (acceptance rates vary by year)

### 10.2 Practical Applications via Selective Prediction

Our confidence analysis reveals that the model can achieve much higher accuracy by abstaining on uncertain predictions. At 50% coverage, accuracy reaches ~77%; at 25% coverage, ~84%. This enables practical deployment scenarios:

- **Desk reject screening**: Predicting rejection with high confidence can identify clearly below-threshold papers (the model achieves 78.7% reject recall in one configuration)
- **Confidence-weighted triage**: Area chairs could prioritize review of papers where the model is uncertain, allocating reviewer effort where it matters most
- **Pre-submission quality feedback**: Providing authors with model confidence alongside predictions gives calibrated quality estimates

### 10.3 Ethical Considerations

Automated acceptance prediction raises several ethical concerns:
- **Bias amplification**: If the model learns biases present in historical decisions (topic bias, methodology bias), it perpetuates them
- **Gaming**: If prediction signals are known, authors could optimize for predicted acceptance rather than genuine scientific contribution
- **Fairness**: Content-only prediction may disadvantage papers from non-traditional research paradigms or underrepresented communities
- **Transparency**: Any deployment should be transparent about the model's limitations and the role of human judgment

We emphasize that our system is designed as a research tool for understanding the peer review process, not as a replacement for human reviewers.

---

## 11. Related Work

### 11.1 Peer Review Prediction

**PeerRead** (Kang et al., 2018) was the first large-scale study of peer review prediction, achieving ~65% accuracy on ICLR 2017 using hand-crafted features. Subsequent work by Li et al. (2020) extended this to multi-task learning with review generation. **MOPRD** (Gao et al., 2019) focused on multi-outcome prediction (accept/revise/reject) using attention-based models.

**DeepReviewer** (2023) and **PaperDecision** (2024) use LLMs to generate structured reviews, but their acceptance prediction accuracy remains near random (~52--55%), consistent with our zero-shot baseline findings. Sakana AI's "The AI Scientist" (2024) focuses on end-to-end paper generation rather than review prediction.

### 11.2 LLMs for Scientific Text Understanding

Large language models have been applied to scientific text for summarization (SciBERT, SPECTER), claim verification (SciFact), and review generation (ReviewerGPT). Our work differs by focusing on binary classification rather than generation, and by systematically comparing text-only and vision-based input modalities.

### 11.3 Vision-Language Models for Document Understanding

Recent VLMs (Qwen-VL, InternVL, GPT-4V) have shown strong performance on document understanding benchmarks (DocVQA, ChartQA). Our work applies VLMs to a novel domain---scientific paper quality assessment---where the visual modality (layout, figures, typography) carries complementary information to text content.

### 11.4 Automated Review Systems

Several systems attempt to automate the review process: AIDER (automated individual document evaluation for review), ReviewAdvisor, and various LLM-based review generators. Our work is complementary, focusing on the acceptance decision rather than the review text, and providing empirical evidence for the fundamental limits of content-only prediction.

---

## 12. Conclusion

We present a comprehensive study of paper acceptance prediction at ICLR using fine-tuned large language models, yielding five key findings:

1. **Fine-tuning dramatically improves over zero-shot prediction.** Our best fine-tuned model (70.4%) outperforms zero-shot Qwen 3.5-122B (52.1%) by 18.3 percentage points, demonstrating that venue-specific calibration is essential.

2. **Vision models capture complementary signals to text models.** While text models achieve higher overall accuracy on 2020--2025 data (68.9% vs 67.0%), vision models outperform on 2026-inclusive data (68.4% vs 67.0%) and exhibit different recall profiles (higher accept recall, lower reject recall).

3. **Content-only prediction approaches the empirical consistency ceiling.** The NeurIPS 2021 consistency study shows that independent review committees agree only ~72--77% of the time. Our best model (70.4%) is within a few percentage points of this ceiling, suggesting we are near the fundamental limit of content-only prediction.

4. **2026 presents a measurably harder prediction target.** The combination of compressed rating gaps (0.39 vs 0.42--0.50) and noisier labels (21.8% noisy accepts) explains the 4.4 pp accuracy drop from 2025 to 2026.

5. **Selective prediction enables high-accuracy deployment.** By restricting predictions to the most confident 50% of samples, accuracy increases from ~68% to ~77%; at 25% coverage, accuracy reaches ~84%, making the system practically useful for desk reject screening and confidence-weighted triage.

**Limitations.** Our study is limited to ICLR and may not generalize to other venues with different review cultures. The binary accept/reject framing ignores the rich structure of review scores and discussions. Our models are constrained by the context window (8--10 pages for vision, ~32K tokens for text), potentially missing important information in appendices.

**Future work.** Promising directions include: (1) longer-context models that process full papers including appendices, (2) multi-venue training to learn venue-invariant quality signals, (3) ensemble methods that formally combine text and vision predictions with learned confidence weighting, and (4) incorporating structural metadata (number of authors, submission timing) as auxiliary signals.

---

## Appendix A: Full Modality Comparison Table (modality_v7)

| Dataset Group | Modality | Best Checkpoint | Accuracy | Accept Recall | Reject Recall | Pred Accept Rate | N |
|---------------|----------|-----------------|----------|---------------|---------------|------------------|------|
| 2020-2025 Balanced | Text | ckpt-1069 | 66.2% | 65.2% | 67.2% | 49.0% | 2024 |
| 2020-2025 Balanced | Text+Images | ckpt-4152 | 67.0% | 77.1% | 57.0% | 60.0% | 1974 |
| 2020-2025 Balanced | Vision | ckpt-4284 | 69.8% | 68.9% | 70.8% | 49.1% | 2026 |
| 2017-2025 Balanced | Text | ckpt-2354 | 65.7% | 73.5% | 57.8% | 57.9% | 2234 |
| 2017-2025 Balanced | Text+Images | ckpt-5715 | 67.0% | 67.3% | 66.7% | 50.3% | 2178 |
| 2017-2025 Balanced | Vision | finetuned | 69.5% | 72.0% | 66.9% | 52.5% | 2236 |
| 2024-2025 Balanced | Text | ckpt-3145 | 63.8% | 60.7% | 67.0% | 46.8% | 1189 |
| 2024-2025 Balanced | Text+Images | ckpt-1833 | 63.5% | 56.8% | 70.2% | 43.4% | 1161 |
| 2024-2025 Balanced | Vision | ckpt-2520 | 70.9% | 63.2% | 78.7% | 42.3% | 1190 |
| Trainagreeing | Text | ckpt-1816 | 66.9% | 78.5% | 55.4% | 61.6% | 2024 |
| Trainagreeing | Text+Images | ckpt-4400 | 68.2% | 71.7% | 64.7% | 53.4% | 1974 |
| Trainagreeing | Vision | ckpt-1818 | 70.4% | 80.3% | 60.5% | 59.9% | 2026 |

## Appendix B: Qwen 3.5-122B Zero-Shot Detailed Results

**Error characteristics (average pct_rating by quadrant):**

| Quadrant | Mean pct\_rating |
|----------|-----------------|
| True Positive (correct accept) | 0.815 |
| True Negative (correct reject) | 0.325 |
| False Positive (incorrect accept) | 0.444 |
| False Negative (missed accept) | 0.757 |

**Content similarity analysis (500-paper sample):**

| Metric | Strengths | Weaknesses |
|--------|-----------|------------|
| Word Jaccard | 0.284 | 0.181 |
| 3-gram Overlap | 0.437 | 0.336 |

The low similarity between independently generated reviews (Jaccard ~0.2--0.3) suggests the model produces diverse assessments, though there is moderate 3-gram overlap in common phrases like "well-written" and "lacks novelty."

## Appendix C: Hyperparameter Sweep Results

![Hyperparameter accuracy heatmap](../../figures/latex/ablations/hyperparam_accuracy_heatmap.png)

![Hyperparameter comparison](../../figures/latex/ablations/hyperparam_comparison.png)

## Appendix D: Additional Data Analysis

![pct\_rating by year (violin)](../../figures/latex/data/pct_rating_by_year_violin.png)

![pct\_rating distribution](../../figures/latex/data/pct_rating_distribution.png)

![Rating KDE](../../figures/latex/data/rating_kde.png)

![Category distribution](../../figures/latex/data/category_distribution.png)

![Category accept rate](../../figures/latex/data/category_accept_rate.png)

![Correlation heatmap](../../figures/latex/data/correlation_heatmap.png)
