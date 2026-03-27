## 7. Analysis

### 7.1 Confidence and Calibration

We extract per-sample prediction confidence from the model's token log-probabilities.

![Confidence and calibration analysis](../../figures/confidence_calibration.png)

**Figure 14.** Calibration plots and confidence distributions for text and vision models.

**Key findings.** Both models are moderately well-calibrated: when predicting with 80% confidence, the model is correct roughly 75--80% of the time, with slight overconfidence at the high end. Vision models produce a higher proportion of high-confidence predictions. Correct predictions cluster at high confidence, while incorrect predictions have a flatter profile---this separation enables confidence-based selective prediction.

### 7.2 Coverage vs Accuracy Tradeoff

A key practical question: if we restrict predictions to samples where the model is sufficiently confident, what accuracy can we achieve?

![Coverage vs accuracy tradeoff](../../figures/coverage_vs_accuracy.png)

**Figure 15.** Accuracy as a function of coverage (fraction of test set on which predictions are made), ordered by model confidence.

| Coverage | Text Accuracy | Vision Accuracy |
|----------|---------------|-----------------|
| 100% | 67.6% | 67.6% |
| 90% | 70.4% | 70.0% |
| 75% | 73.4% | 72.5% |
| 50% | 76.2% | 76.9% |
| 25% | ~84% | ~84% |

*Table 2. Accuracy at various coverage levels on the v7 test set.*

**Key findings.** At 50% coverage, both models exceed 76% accuracy---approaching the Bayes-optimal ceiling of 80.2%. At 25% coverage, accuracy reaches ~84%, exceeding the theoretical ceiling on the retained subset. This has practical implications: a system that flags papers above a confidence threshold could provide highly reliable predictions on a meaningful fraction of submissions. The convergence of text and vision accuracy at low coverage suggests that both modalities identify the same "easy" papers with high confidence.

### 7.3 Agreement and Disagreement Analysis (Text vs Vision)

The two modalities agree on ~74% of predictions but diverge on ~26%---the disagreement set is where ensemble methods can potentially improve.

![Agreement breakdown](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/agreement_breakdown.png)

**Figure 16.** Breakdown of text-vision agreement on the v7 test set.

![Prediction agreement Venn diagram](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/prediction_agreement_venn.png)

**Figure 17.** Venn diagram of correct predictions by modality.

**Key findings.** Of the ~26% disagreement cases, vision is correct more often than text, but text uniquely captures some papers that vision misses (particularly theory-heavy papers with minimal figures). The error overlap is only ~50%, confirming substantial complementarity.

### 7.4 What Drives Modality Disagreement?

![Disagreement feature importance](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/disagreement_feature_importance.png)

**Figure 18.** Features most predictive of text-vision disagreement.

**Key findings.** Papers with high pct_rating (strong accepts) and shorter length tend to have vision predict accept while text predicts reject. Conversely, papers with complex mathematical content and dense equations tend to favor text. The features most predictive of disagreement are: number of figures (vision-favoring), equation density (text-favoring), page count, and paper category.

### 7.5 Category Analysis

![Accuracy by category](../../figures/latex/analysis/accuracy_by_category.png)

**Figure 19.** Model accuracy by paper category.

![Category-modality interaction](../../figures/latex/analysis/category_modality_interaction.png)

**Figure 20.** Category-modality interaction: relative advantage of vision vs text by paper category.

**Key findings.** Model accuracy varies substantially across categories. Reinforcement learning and generative model papers are easier to predict; theory and optimization papers are harder. Vision models have an advantage on empirically-heavy papers (with many figures and tables) while text models perform better on theoretically-dense papers, consistent with the modality-specific information each captures.

### 7.6 Structural Features by Decision

![Structural features by decision](../../figures/latex/analysis/structural_features_by_decision.png)

**Figure 21.** Distribution of structural features for accepted vs rejected papers.

**Key findings.** Accepted papers tend to have more figures, more tables, and longer page counts, but these features alone are weakly predictive (the distributions overlap substantially). This confirms that surface statistics are insufficient and the SFT model captures deeper semantic signals.

### 7.7 Mediation Analysis

We perform a mediation analysis to understand the causal pathway from paper features to model predictions, mediated by underlying paper quality (proxied by pct_rating).

![Mediation path coefficients](../../figures/latex/analysis/mediation_path_coefficients.png)

**Figure 22.** Mediation path coefficients.

![Mediation proportion](../../figures/latex/analysis/mediation_proportion.png)

**Figure 23.** Proportion of total effect mediated through paper quality proxy.

**Key findings.** The indirect (mediated) path accounts for approximately 40--60% of the total effect for features like figure count and page length, suggesting the model partially learns a quality proxy. The remaining 40--60% represents direct effects---the model uses structural features as independent signals beyond their correlation with reviewer-assessed quality.

### 7.8 Modality Wins Analysis

![Modality wins analysis](../../figures/latex/ablations/modality_wins_analysis.png)

**Figure 24.** Papers where one modality is correct and the other is wrong.

**Key findings.** Vision "wins" on papers with more figures, better visual presentation, and clear experimental results. Text "wins" on papers with dense mathematical content and minimal figures. This complementarity motivates ensemble approaches.

### 7.9 Rating Interval Analysis

![Rating interval analysis](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/rating_interval_analysis.png)

**Figure 25.** Model accuracy by pct_rating interval.

**Key findings.** Both models achieve near-perfect accuracy on extreme rating intervals (pct_rating < 0.2 or > 0.8) but diverge significantly in the borderline zone (0.4--0.6). In this zone, text achieves ~60% accuracy while vision drops to ~55%, reflecting vision's tendency to optimistically predict accept for borderline papers.

![Accuracy heatmap](../../figures/latex/ablations/acc_heatmap_2d.png)

**Figure 26.** 2D heatmap: accuracy as a function of pct_rating and rating standard deviation.

**Key findings.** The hardest papers combine borderline ratings with high reviewer variance---exactly the papers where the decision depends most on factors beyond paper content.
