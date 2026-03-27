## 1. Introduction

The peer review system at machine learning conferences is under unprecedented strain. ICLR 2025 received over 11,000 submissions, and ICLR 2026 surpassed that figure. Simultaneously, AI-assisted paper writing tools have lowered the barrier to submission, further inflating the review workload. Area chairs must synthesize noisy, sometimes contradictory reviewer signals into binary accept/reject decisions under severe time pressure.

Several recent systems attempt to automate aspects of peer review. Sakana AI's "The AI Scientist" generates end-to-end papers and self-reviews; DeepReviewer and PaperDecision produce structured reviews from paper content. However, these systems overwhelmingly focus on *review generation* rather than *decision prediction*, and when they do predict decisions, they typically train on non-anonymized data (including author names, affiliations, and citation counts) that would be unavailable to a blind reviewer.

Our work departs from this paradigm in three ways. First, we focus exclusively on *acceptance prediction from paper content alone*, treating the problem as binary classification without access to any review-stage information. Second, we enforce rigorous content normalization: author names, affiliations, acknowledgments, and references are removed from all inputs, ensuring the model cannot exploit prestige signals. Third, we systematically compare text-only and vision-based (rendered PDF page) input modalities, revealing complementary strengths.

We ground our analysis in information-theoretic principles. The data processing inequality guarantees that predicting the final decision from paper content alone cannot exceed the mutual information between content and decision---which is strictly less than the mutual information between the full review chain and the decision. Using the NeurIPS 2021 consistency experiment---where independent committees disagreed on 23% of papers---we fit a Bayesian model and derive a Bayes-optimal accuracy of **~80.2%** on balanced evaluation sets. Our best models achieve 70.4%, capturing ~88% of this theoretical ceiling with a 9.8 pp gap still to close. Notably, ICLR 2025 is out-of-distribution for the base model (Qwen2.5-7B, trained on data up to early 2025), making our 70.4% accuracy on 2025 papers a genuine test of the model's ability to generalize beyond its pretraining data.

### Contributions

1. **The largest anonymized content-only peer review prediction dataset**, spanning ~25,000 ICLR papers (2020--2026) with binary labels, cleaned text, and rendered PDF pages---approximately 2x larger than PeerRead and 4x larger than MOPRD.

2. **Theoretical ceiling analysis.** We derive a Bayes-optimal accuracy of ~80.2% from the NeurIPS 2021 consistency experiment and establish, via the data processing inequality, that content-only prediction is fundamentally bounded by information loss when bypassing the review chain.

3. **70.4% accuracy on ICLR 2025 (OOD) via vision-based SFT.** This substantially exceeds feature-engineered baselines (logistic regression on 187 paper statistics: 64.0%; TF-IDF: 59.6%) and zero-shot LLMs (~52%), capturing ~88% of the theoretical ceiling.

4. **Systematic ablations** across year range, input modality, training data filtering (trainagreeing), and regularization (weight decay). Vision-based models show a consistent advantage of 3--4 pp over text, with complementary recall profiles: vision achieves higher accept recall (80.3% vs 78.5%) while text achieves more balanced recall.

5. **Temporal generalization analysis.** Accuracy on ICLR 2026 drops to ~68%, which we attribute to measurably compressed rating separability (AUC 0.893 vs 0.939 for 2025) and increased label noise (21.8% of 2026 accepts have below-threshold aggregate ratings).

6. **Selective prediction.** Restricting to the most confident 50% of predictions yields ~77% accuracy, approaching the Bayes-optimal ceiling and enabling practical deployment for desk reject screening and confidence-weighted triage.

We emphasize that the modality comparison is not a simple "Vision > Text" ranking. Rather, vision-based input yields a consistent accuracy advantage of 3--4 pp across configurations, with notably higher accept recall but lower reject recall---suggesting the two modalities capture complementary aspects of paper quality that could be exploited via ensembling.
