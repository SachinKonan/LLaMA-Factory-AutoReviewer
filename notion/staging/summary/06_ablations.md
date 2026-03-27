## 6. Ablations

> **Test set specification.** All results are on the v7 test set (ICLR 2020+2023+2025, balanced) unless otherwise noted.

### 6.1 Year Range Ablation

A key design decision is which years to include in training. We compare three year ranges:

| Year Range | Text | Vision |
|------------|------|--------|
| 2020-2025 (balanced) | 66.2% | 69.8% |
| 2017-2025 (balanced) | 65.7% | 69.5% |
| 2024-2025 (balanced) | 63.8% | **70.9%** |

![Year range ablation](../../figures/year_range_ablation.png)

**Figure 10.** Accuracy by year range and modality.

**Key findings.** Adding pre-2020 data provides no benefit (65.7% vs 66.2% for text; 69.5% vs 69.8% for vision), suggesting that older papers from different reviewing eras introduce noise rather than signal. Training on 2024+2025 only yields the best vision accuracy (70.9%) but the worst text accuracy (63.8%), revealing an asymmetry: vision models are more robust to small training sets, while text models benefit from larger and more diverse training data. This is consistent with the hypothesis that visual quality signals (layout, figures, typography) are more year-invariant than textual content patterns.

### 6.2 Trainagreeing: Data Quality vs Quantity

The trainagreeing dataset retains only examples where early-checkpoint predictions agree with the ground truth, producing a smaller but cleaner training set (~8.3K vs ~12.7K examples, a 35% reduction).

| Modality | Balanced Acc | Trainagreeing Acc | Delta |
|----------|-------------|-------------------|-------|
| Text | 66.2% | **66.9%** | +0.7 |
| Vision | 69.8% | **70.4%** | +0.6 |

![Balanced vs Trainagreeing](../../figures/year_range_ablation_trainagreeing.png)

**Figure 11.** Balanced vs trainagreeing accuracy by modality.

**Key findings.** Trainagreeing consistently improves accuracy despite using 35% fewer training examples, suggesting that label noise in the full balanced dataset actively harms learning. However, the improvement is modest (+0.6--0.7 pp), and the recall balance shifts substantially: trainagreeing vision achieves 80.3% accept recall but only 60.5% reject recall, compared to the more symmetric 68.9%/70.8% for balanced. This tradeoff---higher overall accuracy at the cost of recall imbalance---reflects the filtering's bias toward "easy" examples that are disproportionately clear accepts.

### 6.3 Weight Decay Sweep

We sweep weight decay values to investigate the effect of regularization on generalization.

<!-- TODO: Insert wd_sweep figures -->

**Key findings.** Moderate weight decay (0.001--0.002) provides a small benefit, with the best vision accuracy of 69.1% at wd=0.002 (on the 1506-paper no-2026 test subset). Higher weight decay (>0.004) degrades performance. The effect is modest compared to the impact of data selection (trainagreeing) and modality choice.

### 6.4 Train Size vs Accuracy

![Train size vs accuracy](../../figures/train_size_vs_accuracy.png)

**Figure 12.** Accuracy as a function of training set size for Text and Vision modalities.

**Key findings.** Vision models achieve higher accuracy than text models at every training set size. Notably, vision accuracy plateaus earlier, suggesting that visual quality signals can be learned from fewer examples. Text accuracy continues to improve with more data but remains 3--4 pp below vision, indicating that the advantage is not merely a data efficiency effect.

### 6.5 Generalization to ICLR 2026

*Results in this section compare the best v7 models (trainagreeing) against models retrained with 2026 data included, evaluated on a separate test set containing ICLR 2026 papers.*

**Best v7 models (trainagreeing, evaluated on v7 test: 2020+2023+2025):**

| Metric | Text | Vision |
|--------|------|--------|
| Overall Accuracy | 66.9% | **70.4%** |
| Accept Recall | 78.5% | **80.3%** |
| Reject Recall | 55.4% | 60.5% |

**2026-inclusive models (retrained with 2026 data, evaluated on 2026-inclusive test):**

| Metric | Text | Vision |
|--------|------|--------|
| Overall Accuracy | 67.0% | **68.4%** |
| 2025+2026 OOD Accuracy | 66.0% | **68.3%** |
| Accept Recall | 66.0% | **71.0%** |
| Reject Recall | **67.9%** | 65.7% |

**Key comparison:**

| Metric | v7 Text | v7 Vision | 2026-incl Text | 2026-incl Vision |
|--------|---------|-----------|----------------|------------------|
| Overall Accuracy | 66.9% | **70.4%** | 67.0% | 68.4% |
| Accept Recall | 78.5% | **80.3%** | 66.0% | 71.0% |
| Reject Recall | 55.4% | 60.5% | **67.9%** | 65.7% |

The accuracy drop from the best v7 vision (70.4%) to the 2026-inclusive vision (68.4%) reflects measurable changes in the 2026 data:
- **Rating gap compression**: 2026 accept-reject gap is 0.39 vs 0.43 for 2025
- **Lower AUC**: pct_rating AUC drops from 0.939 (2025) to 0.893 (2026)
- **Higher label noise**: 21.8% of 2026 accepts have below-threshold aggregate ratings

Vision models are more robust to the distribution shift: vision drops 2.0 pp (70.4% to 68.4%) while the recall profiles shift substantially---the 2026-inclusive models produce more balanced recall, likely because the noisier 2026 labels discourage the accept-biased predictions seen in the trainagreeing v7 models.

### 6.6 Rating Standard Deviation Analysis (Per-Year)

Because ICLR has changed its rating scale across years, cross-year comparison of raw rating standard deviation is not meaningful. We analyze the relationship between within-paper reviewer disagreement and model accuracy separately for each year.

![Accuracy vs rating std](../../figures/latex/ablations/acc_vs_rating_std.png)

**Figure 13.** Accuracy vs within-paper rating standard deviation, analyzed per year.

**Key findings.** Within each year, higher reviewer disagreement (larger rating std) consistently predicts lower model accuracy. Papers in the top quartile of rating std are 8--12 pp harder to predict than papers in the bottom quartile. This is expected: when reviewers disagree strongly, the final decision depends on discussion dynamics and AC judgment---factors invisible to content-only models.
