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

**ICLR 2026 special case.** The 2026 review cycle eliminated the traditional author rebuttal period and compressed the review timeline. Effectively, the chain simplifies to $X \to Y_1 \to Z$, where $Y_1$ is a single round of reviews without rebuttal updates. This makes $Y_1$ a noisier signal for $Z$, as authors cannot correct reviewer misunderstandings. Paradoxically, while this should make $I(Y_1; Z) < I(Y; Z)$ for the full chain, it does *not* necessarily increase $I(X; Z)$---the content-only prediction problem remains equally or more difficult because the decision itself becomes noisier.

### 2.3 Bayes-Optimal Ceiling: NeurIPS 2021 Consistency Study

The NeurIPS 2021 consistency experiment provides the empirical foundation for estimating the fundamental ceiling. In that experiment, ~10% of submissions were independently reviewed by two separate committees. The two committees disagreed on **23%** of papers (raw agreement ~77%).

We model each paper $i$ as having a latent acceptance probability $p_i \sim \text{Beta}(\alpha, \beta)$. Matching the observed acceptance rate (26%) and disagreement rate (23%) yields $\alpha \approx 0.39$, $\beta \approx 1.10$. The Bayes-optimal predictor---one that knows each paper's latent $p_i$ but not which specific reviewers are assigned---achieves:

- **83.3% accuracy** on the original distribution (26% accept rate)
- **80.2% accuracy** on a balanced (50/50) evaluation set

Since we evaluate on balanced test sets, **~80% is the Bayes-optimal ceiling** for any predictor operating under reviewer noise. This represents the accuracy achievable by a perfect model that knows everything about a paper *except* the randomness of reviewer assignment. The gap between 80% and 100% reflects irreducible noise in peer review.

Our best model achieves 70.4%, leaving a **9.8 pp gap** to the Bayes-optimal ceiling. This gap represents the combined effect of (1) information lost by bypassing the review chain (the DPI bound) and (2) modeling limitations.

### 2.4 Rating Separability: pct_rating AUC

We estimate a complementary, year-specific measure of class separability using the `pct_rating` variable---the average normalized reviewer rating for each paper. If we had access to `pct_rating` at test time, the optimal classifier would simply threshold it at the decision boundary. The AUC of `pct_rating` as a predictor of the accept/reject decision measures how well-separated the two classes are in rating space.

*All accuracy figures below are measured on the v7 test set (ICLR 2020+2023+2025).*

| Year | pct\_rating AUC | Rating-Based Ceiling | Best Model Acc |
|------|----------------|---------------------|-----------|
| 2020 | 0.951 | 93.5% | 71.7% |
| 2021 | 0.950 | 92.1% | 69.5% |
| 2022 | 0.941 | 91.3% | 66.5% |
| 2023 | 0.972 | 93.8% | 67.9% |
| 2025 | 0.939 | 89.1% | 70.9% |
| 2026 | 0.893 | 86.5% | 66.5% |

The "Rating-Based Ceiling" represents the accuracy achievable with perfect access to aggregate reviewer ratings---a tighter, year-specific bound. The Bayes-optimal ceiling from Section 2.3 (~80% on balanced sets) accounts for the inherent stochasticity of reviewer assignment across all years. The rating-based ceiling is higher because it conditions on the actual ratings received, which resolve much of the reviewer-assignment noise.

### 2.5 The Rating Gap as a Predictability Proxy

The gap between mean `pct_rating` for accepted and rejected papers provides a simple proxy for class separability:

| Year | Accept pct\_rating | Reject pct\_rating | Gap |
|------|-------------------|-------------------|------|
| 2020 | 0.808 | 0.334 | 0.47 |
| 2021 | 0.806 | 0.341 | 0.47 |
| 2022 | 0.765 | 0.342 | 0.42 |
| 2023 | 0.790 | 0.290 | 0.50 |
| 2025 | 0.747 | 0.312 | 0.43 |
| 2026 | 0.738 | 0.349 | 0.39 |

2026 has the smallest gap (0.39), indicating the greatest overlap between accept and reject rating distributions. This is consistent with the lower AUC (0.893) and suggests that the 2026 decision boundary is harder to learn from any signal, including content.

![Rating gap across years](../../figures/report_rating_gap.png)

**Figure 1.** Rating gap between accepted and rejected papers across ICLR years. The gap measures the difference in mean normalized reviewer rating (pct_rating) between the two classes.

**Key findings.** The rating gap is smallest for 2026 (0.39) and largest for 2023 (0.50). Years with smaller rating gaps correspond to lower model accuracy, consistent with the interpretation that compressed rating distributions reflect noisier or more borderline decisions. The monotonic decline from 2023 to 2026 suggests an increasing difficulty in distinguishing accepted from rejected papers, potentially driven by growing submission volume and evolving review standards.

### 2.6 Rating Standard Deviation (Per-Year)

We analyze within-paper reviewer disagreement via the standard deviation of individual reviewer ratings. Because ICLR has changed its rating scale across years (e.g., 1--10 in earlier years, 1--5 in some years), **cross-year comparisons of raw standard deviation are not meaningful**. We therefore present per-year statistics only.

Within each year, higher rating standard deviation indicates greater reviewer disagreement, which we expect to correlate with noisier labels and lower model accuracy on those papers.

![Accuracy vs rating standard deviation](../../figures/latex/ablations/acc_vs_rating_std.png)

**Figure 2.** Model accuracy as a function of within-paper rating standard deviation, analyzed separately per year.

**Key findings.** Within each year, papers with high rating standard deviation (high reviewer disagreement) are substantially harder to predict. This is consistent with the theoretical expectation: when reviewers disagree strongly, the final decision depends heavily on discussion dynamics and AC judgment---information inaccessible from paper content alone.
