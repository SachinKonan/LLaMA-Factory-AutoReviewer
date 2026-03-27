## Appendix

### Appendix A: Full Modality Comparison Table

*All results on the v7 test set (ICLR 2020+2023+2025, balanced).*

| Dataset Group | Modality | Best Checkpoint | Accuracy | Accept Recall | Reject Recall | Pred Accept Rate | N |
|---------------|----------|-----------------|----------|---------------|---------------|------------------|------|
| 2020-2025 Balanced | Text | ckpt-1069 | 66.2% | 65.2% | 67.2% | 49.0% | 2,024 |
| 2020-2025 Balanced | Vision | ckpt-4284 | 69.8% | 68.9% | 70.8% | 49.1% | 2,026 |
| 2017-2025 Balanced | Text | ckpt-2354 | 65.7% | 73.5% | 57.8% | 57.9% | 2,234 |
| 2017-2025 Balanced | Vision | finetuned | 69.5% | 72.0% | 66.9% | 52.5% | 2,236 |
| 2024-2025 Balanced | Text | ckpt-3145 | 63.8% | 60.7% | 67.0% | 46.8% | 1,189 |
| 2024-2025 Balanced | Vision | ckpt-2520 | 70.9% | 63.2% | 78.7% | 42.3% | 1,190 |
| Trainagreeing | Text | ckpt-1816 | 66.9% | 78.5% | 55.4% | 61.6% | 2,024 |
| Trainagreeing | Vision | ckpt-1818 | 70.4% | 80.3% | 60.5% | 59.9% | 2,026 |

### Appendix B: Qwen 3.5-122B Zero-Shot Details

**Error characteristics (average pct_rating by prediction quadrant):**

| Quadrant | Mean pct\_rating |
|----------|-----------------|
| True Positive (correct accept) | 0.815 |
| True Negative (correct reject) | 0.325 |
| False Positive (incorrect accept) | 0.444 |
| False Negative (missed accept) | 0.757 |

The high mean pct_rating for false negatives (0.757) confirms that the zero-shot model systematically rejects papers that received strong reviewer endorsement. The model's quality assessment does not align with ICLR's acceptance threshold.

### Appendix C: Weight Decay Details

![Hyperparameter accuracy heatmap](../../figures/latex/ablations/hyperparam_accuracy_heatmap.png)

**Figure A1.** Accuracy across hyperparameter configurations.

![Hyperparameter comparison](../../figures/latex/ablations/hyperparam_comparison.png)

**Figure A2.** Detailed hyperparameter comparison.

### Appendix D: Additional Data Analysis

![pct_rating by year](../../figures/latex/data/pct_rating_by_year_violin.png)

**Figure A3.** Distribution of pct_rating by year.

![pct_rating distribution](../../figures/latex/data/pct_rating_distribution.png)

**Figure A4.** Overall pct_rating distribution.

![Category distribution](../../figures/latex/data/category_distribution.png)

**Figure A5.** Distribution of papers by category.

![Category accept rate](../../figures/latex/data/category_accept_rate.png)

**Figure A6.** Accept rate by category.

![Correlation heatmap](../../figures/latex/data/correlation_heatmap.png)

**Figure A7.** Correlation heatmap of paper features.

### Appendix E: Paper Statistics Baseline Details

**Logistic regression feature weights** (top 30, from 187 engineered features):

| Feature | Weight | Direction |
|---------|--------|-----------|
| appendix_ratio | +0.310 | Accept |
| year | -0.305 | Reject |
| log_total_words | +0.226 | Accept |
| pct_recent_cites | +0.178 | Accept |
| vision_token_ratio | -0.153 | Reject |
| log_number_of_cited_references | +0.153 | Accept |
| cross_ref_density | +0.137 | Accept |
| num_inline_math | +0.132 | Accept |
| num_text_image_tokens | +0.126 | Accept |
| mentions_ablation | +0.108 | Accept |
| title_chars | -0.106 | Reject |
| avg_name_parts | -0.102 | Reject |
| pages_per_author | +0.093 | Accept |
| tokens_per_page | -0.092 | Reject |
| abstract_sentences | +0.090 | Accept |
| total_words | +0.088 | Accept |
| abstract_has_numbers | +0.084 | Accept |
| long_words_ratio | -0.081 | Reject |
| year_x_refs | -0.079 | Reject |
| refs_per_page | +0.078 | Accept |

**HGB permutation importance** (top 10):

| Feature | Importance |
|---------|-----------|
| original_total_pages | 0.0163 |
| pct_recent_cites | 0.0147 |
| log_total_words | 0.0103 |
| year | 0.0102 |
| num_text_image_tokens | 0.0074 |
| num_figure_mentions | 0.0052 |
| length_x_refs | 0.0050 |
| log_num_text_tokens | 0.0032 |
| hedge_ratio | 0.0031 |
| negative_count | 0.0030 |

<!-- TODO: Add SFT vs paperstats correlation analysis results and figure once logreg results are saved to disk -->
