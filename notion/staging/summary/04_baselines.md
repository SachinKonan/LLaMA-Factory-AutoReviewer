## 4. Baselines

We establish several baselines spanning random chance, classical ML on engineered features, and zero-shot large language models. All results are measured on the v7 test set (ICLR 2020+2023+2025, balanced).

### 4.1 TF-IDF Baseline

As a non-neural baseline, we train a TF-IDF + logistic regression classifier on the extracted text. Using unigram and bigram features with L2 regularization, this baseline achieves **59.6% accuracy** on the balanced test set.

![TF-IDF feature importance](../../figures/latex/baseline/tfidf_feature_importance.png)

**Figure 5.** Top TF-IDF features by logistic regression weight.

**Key findings.** The TF-IDF model captures superficial lexical patterns rather than deep content understanding. Certain technical terms correlate weakly with acceptance, but the model's 59.6% accuracy is only modestly above random chance (50%), indicating that bag-of-words features are insufficient for this task.

### 4.2 Logistic Regression on Paper Statistics

We extract 187 engineered features from each paper, including structural features (page count, figure count, equation count, reference count), writing quality metrics (Flesch-Kincaid readability, vocabulary richness), rhetorical features (hedge/confidence word ratios), topic indicators, and citation patterns. We train logistic regression, gradient-boosted trees (HistGBM), and random forest classifiers.

| Model | Features | Test Accuracy |
|-------|----------|---------------|
| Logistic Regression | 187 | 64.0% |
| HistGradientBoosting | 187 | 64.8% |
| Random Forest | 187 | 65.0% |

<!-- TODO: Update with final results from results/paperstats_baseline/ once srun completes -->

**Top logistic regression feature weights:**

| Feature | Weight | Direction |
|---------|--------|-----------|
| appendix_ratio | +0.310 | Accept |
| year | -0.305 | Reject |
| log_total_words | +0.226 | Accept |
| pct_recent_cites | +0.178 | Accept |
| vision_token_ratio | -0.153 | Reject |
| log_num_cited_references | +0.153 | Accept |
| cross_ref_density | +0.137 | Accept |
| num_inline_math | +0.132 | Accept |

**Key findings.** Even with 187 engineered features, the best traditional ML model (Random Forest) achieves 65.0%---5.4 pp below our SFT Vision model (70.4%). The feature weights are interpretable: longer papers with more appendix material, more references, and more mathematical content tend toward acceptance. The negative weight on `year` reflects increasing difficulty of prediction for more recent years. The modest ceiling of classical ML approaches confirms that paper acceptance depends on semantic content signals that hand-crafted features cannot capture.

### 4.3 Zero-Shot LLM: Qwen 3.5-122B

We evaluate Qwen 3.5-122B (the largest available Qwen model at evaluation time) in a zero-shot, blind review setting. The model is prompted to act as an expert ML reviewer and predict accept/reject.

*Evaluated on the full dataset (~22K papers, not the v7 test set).*

- **Accuracy**: 52.1%
- **Accept Recall**: 4.67%
- **Reject Recall**: 98.5%

The model is extremely conservative: it predicts accept for only 3.1% of papers, compared to the ground truth accept rate of 49.4%. This extreme reject bias yields near-perfect reject recall but catastrophic accept recall.

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

### 4.4 Zero-Shot LLM: Gemini

We also evaluate Gemini in a zero-shot setting for comparison.

![Gemini vs fine-tuned model comparison](../../figures/latex/baseline/gemini_vs_finetuned.png)

**Figure 6.** Zero-shot Gemini vs fine-tuned SFT models.

**Key findings.** Gemini performs similarly to zero-shot Qwen, confirming that the failure of zero-shot prediction is not model-specific but reflects a fundamental mismatch between LLM quality assessment and venue-specific acceptance thresholds.

### 4.5 Why Zero-Shot Fails

The failure of zero-shot LLMs reveals a fundamental mismatch: LLMs can assess paper quality in an absolute sense (identifying clear strengths and weaknesses), but they cannot calibrate to a specific venue's acceptance threshold. The threshold is an emergent property of the reviewer pool, the submission pool, and the AC's decision function---information not accessible from any single paper's content. Fine-tuning provides this calibration by exposing the model to the empirical accept/reject boundary.

### 4.6 Summary: All Baselines vs SFT

<!-- TODO: Insert baseline comparison bar chart from scripts/plot_baseline_comparison.py -->
<!-- ![Baseline comparison](../../figures/baseline_comparison.png) -->

| Method | Accuracy | Notes |
|--------|----------|-------|
| Random | 50.0% | |
| Zero-shot Qwen 3.5-122B | 52.1% | Full dataset |
| TF-IDF + LogReg | 59.6% | v7 test |
| LogReg (187 features) | 64.0% | v7 test |
| Random Forest (187 features) | 65.0% | v7 test |
| **SFT Text (Trainagreeing)** | **66.9%** | v7 test |
| **SFT Vision (Trainagreeing)** | **70.4%** | v7 test |
| Bayes-optimal ceiling | ~80.2% | NeurIPS 2021 |

**Figure 7.** Comparison of all methods. SFT Vision achieves 70.4%, capturing ~88% of the Bayes-optimal ceiling and exceeding the best feature-engineered baseline by 5.4 pp.

**Key findings.** The progression from random (50%) to TF-IDF (59.6%) to engineered features (65.0%) to SFT (70.4%) demonstrates that each level of representation captures additional signal. The 5.4 pp gap between the best classical ML and SFT confirms that the LLM captures semantic content signals---argument structure, technical depth, experimental rigor---that cannot be reduced to surface statistics.
