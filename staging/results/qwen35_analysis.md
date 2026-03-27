# Qwen 3.5-122B Blind Text Review Analysis

**Papers analyzed**: 21960  
**Papers with 3/3 valid reviews**: 21960  
**Ground truth available**: 24888  

## Overall Accuracy

| Metric | Value |
|--------|-------|
| Accuracy | 0.5215 (52.1%) |
| Precision (Accept) | 0.7496 |
| Recall (Accept) | 0.0467 |
| F1 (Accept) | 0.0878 |
| Total | 21960 |

### Confusion Matrix

| | Pred Accept | Pred Reject |
|---|---|---|
| **GT Accept** | 506 (TP) | 10339 (FN) |
| **GT Reject** | 169 (FP) | 10946 (TN) |

**Unanimous (3/3)**: n=19544, acc=0.5352, prec=0.8343, rec=0.0164, F1=0.0322  
**Split (2/1)**: n=2416, acc=0.4110, prec=0.7186, rec=0.2166, F1=0.3329  

### Individual Candidate Correctness

How many of the 3 candidates individually give the correct accept/reject answer?

| # Correct Candidates | Count | % |
|---------------------|-------|---|
| 0/3 | 9085 | 41.4% |
| 1/3 | 1423 | 6.5% |
| 2/3 | 993 | 4.5% |
| 3/3 | 10459 | 47.6% |
| **At least 1 correct** | **12875** | **58.6%** |
| **All 3 correct** | **10459** | **47.6%** |

**GT=Accept papers with ≥1 candidate saying Accept**: 1790/10845 (16.5%)  
**GT=Reject papers with ≥1 candidate saying Reject**: 11085/11115 (99.7%)  

### Prediction Distribution

| | Accept | Reject | Rate |
|---|---|---|---|
| **Predicted** | 675 | 21285 | 3.1% accept |
| **Ground Truth** | 10845 | 11115 | 49.4% accept |

**Individual candidate accept rate**: 0.0524 (5.2%)  
**Majority vote accept rate**: 3.1%  

## Accuracy by Year

| Year | N | Accuracy | Prec | Rec | F1 | Pred Accept% | GT Accept% |
|------|---|----------|------|-----|----|----|---|
| 2020 | 1293 | 0.5452 | 0.7810 | 0.1267 | 0.2181 | 8.1% | 50.0% |
| 2021 | 1529 | 0.5487 | 0.7891 | 0.1322 | 0.2265 | 8.4% | 50.0% |
| 2022 | 2040 | 0.5328 | 0.7876 | 0.0874 | 0.1574 | 5.5% | 49.9% |
| 2023 | 2847 | 0.5314 | 0.8046 | 0.0505 | 0.0950 | 3.1% | 48.7% |
| 2025 | 6247 | 0.5086 | 0.6827 | 0.0228 | 0.0442 | 1.7% | 49.8% |
| 2026 ** | 8004 | 0.5161 | 0.6739 | 0.0237 | 0.0458 | 1.7% | 49.0% |

## Accuracy by Paper Features

### By Reviewer Rating (pct_rating)

| Rating Bin | N | Accuracy | Prec | Rec | F1 | Pred Accept% | GT Accept% |
|------------|---|----------|------|-----|----|----|---|
| Strong Reject (0-0.3) | 4905 | 0.9766 | 0.0244 | 0.0132 | 0.0171 | 0.8% | 1.5% |
| Lean Reject (0.3-0.45) | 3184 | 0.8565 | 0.2222 | 0.0281 | 0.0499 | 1.7% | 13.4% |
| Borderline (0.45-0.55) | 2266 | 0.6862 | 0.4286 | 0.0341 | 0.0632 | 2.5% | 31.0% |
| Lean Accept (0.55-0.7) | 4062 | 0.3725 | 0.6636 | 0.0282 | 0.0542 | 2.7% | 63.6% |
| Strong Accept (0.7+) | 7543 | 0.1149 | 0.9565 | 0.0561 | 0.1061 | 5.5% | 93.5% |

### By Text Token Count

| Bin | N | Accuracy | Pred Accept% | GT Accept% |
|-----|---|----------|----|----|
| Q1 (≤10319) | 5489 | 0.6451 | 3.1% | 37.0% |
| Q2 (10319-11789) | 5487 | 0.5282 | 3.0% | 48.5% |
| Q3 (11789-13417) | 5492 | 0.4769 | 3.0% | 53.9% |
| Q4 (>13417) | 5492 | 0.4359 | 3.2% | 58.2% |

### By Number of Pages

| Bin | N | Accuracy | Pred Accept% | GT Accept% |
|-----|---|----------|----|----|
| ≤6 | 209 | 0.6555 | 2.9% | 37.3% |
| 7-9 | 1090 | 0.7018 | 5.0% | 32.4% |
| 10-12 | 20637 | 0.5105 | 3.0% | 50.4% |
| 13-20 | 24 | 0.5833 | 0.0% | 41.7% |

### By Number of Figures

| Bin | N | Accuracy | Pred Accept% | GT Accept% |
|-----|---|----------|----|----|
| 0-2 | 1568 | 0.5810 | 3.8% | 44.4% |
| 3-5 | 6370 | 0.5482 | 3.0% | 46.5% |
| 6-10 | 9033 | 0.5099 | 3.2% | 50.6% |
| 11-20 | 4327 | 0.4925 | 2.6% | 52.1% |
| >20 | 662 | 0.4713 | 2.7% | 54.1% |

### By Number of Equations

| Bin | N | Accuracy | Pred Accept% | GT Accept% |
|-----|---|----------|----|----|
| 0-5 | 7019 | 0.5371 | 4.0% | 48.1% |
| 6-15 | 11051 | 0.5140 | 2.8% | 50.0% |
| 16-30 | 3538 | 0.5099 | 2.4% | 50.3% |
| 31-60 | 350 | 0.5600 | 1.1% | 45.1% |
| >60 | 2 | 1.0000 | 0.0% | 0.0% |

### By Number of References

| Bin | N | Accuracy | Pred Accept% | GT Accept% |
|-----|---|----------|----|----|
| ≤20 | 6310 | 0.6195 | 1.9% | 38.9% |
| 21-35 | 9001 | 0.5106 | 3.8% | 50.8% |
| 36-50 | 5102 | 0.4553 | 3.2% | 56.3% |
| 51-70 | 1406 | 0.4068 | 3.3% | 61.2% |
| >70 | 141 | 0.3688 | 1.4% | 63.1% |

### By Citation Impact (citation_normalized_by_year)

| Quartile | N | Accuracy | Pred Accept% | GT Accept% |
|----------|---|----------|----|----|
| Q1 (≤0.3) | 2670 | 0.8775 | 1.2% | 11.8% |
| Q2 (0.3-0.5) | 4803 | 0.5973 | 1.2% | 40.1% |
| Q3 (0.5-0.6) | 8984 | 0.4977 | 1.9% | 50.9% |
| Q4 (>0.6) | 5503 | 0.3215 | 7.5% | 73.2% |

## Error Analysis

| Category | Count |
|----------|-------|
| True Positives (correct accept) | 506 |
| True Negatives (correct reject) | 10946 |
| False Positives (predicted accept, actually reject) | 169 |
| False Negatives (predicted reject, actually accept) | 10339 |

### Error vs Correct Characteristics

| Feature | True Pos | True Neg | False Pos | False Neg |
|---------|----------|----------|-----------|-----------|
| Count | 506 | 10946 | 169 | 10339 |
| Avg pct_rating | 0.815 | 0.325 | 0.444 | 0.757 |
| Avg tokens | 12068.8 | 11489.7 | 11825.1 | 12298.4 |
| Avg pages | 9.3 | 9.4 | 9.4 | 9.6 |
| Avg figures | 6.7 | 6.8 | 6.8 | 7.2 |
| Avg equations | 7.5 | 8.7 | 6.7 | 8.8 |
| Avg refs | 30.4 | 24.9 | 28.2 | 29.6 |

**Borderline papers (pct_rating 0.4-0.6)**: n=4068, accuracy=0.6502  
**Non-borderline papers**: n=17892, accuracy=0.4922  

### Errors by Year

| Year | FP | FN | FP Rate | FN Rate |
|------|----|----|---------|---------|
| 2020 | 23 | 565 | 3.6% | 87.3% |
| 2021 | 27 | 663 | 3.5% | 86.8% |
| 2022 | 24 | 929 | 2.3% | 91.3% |
| 2023 | 17 | 1317 | 1.2% | 95.0% |
| 2025 | 33 | 3037 | 1.1% | 97.7% |
| 2026 | 45 | 3828 | 1.1% | 97.6% |

## Tone & Content Analysis

### Prediction Skew

- **Individual candidate reject rate**: 94.8%  
- **Majority vote reject rate**: 96.9%  
- **Dataset base rate (reject)**: 50.6%  

### Field Length Statistics (characters)

| Field | Mean | Median | Min | Max |
|-------|------|--------|-----|-----|
| strengths | 651 | 646 | 393 | 1939 |
| weaknesses | 761 | 752 | 487 | 2208 |
| critical_disputes | 815 | 808 | 385 | 2570 |
| most_important_strength | 246 | 245 | 69 | 671 |
| most_important_weakness | 252 | 250 | 115 | 735 |
| relation_to_other_work | 383 | 334 | 184 | 1251 |
| paper_quality | 396 | 389 | 155 | 1249 |

### Field Lengths: Accept vs Reject Predictions

| Field | Pred Accept (mean) | Pred Reject (mean) | Difference |
|-------|-------|-------|------|
| strengths | 670 | 650 | +20 |
| weaknesses | 756 | 761 | -5 |
| critical_disputes | 829 | 815 | +15 |
| most_important_strength | 257 | 246 | +11 |
| most_important_weakness | 254 | 251 | +2 |
| relation_to_other_work | 406 | 382 | +23 |
| paper_quality | 398 | 396 | +2 |

## Multi-Generation Diversity Analysis

### Verdict Agreement

| Pattern | Count | % |
|---------|-------|---|
| All 3 agree | 19544 | 89.0% |
| — All accept | 181 | 0.8% |
| — All reject | 19363 | 88.2% |
| 2-1 split | 2416 | 11.0% |

### Content Similarity (sample of 500 papers, pairwise)

| Metric | Strengths | Weaknesses |
|--------|-----------|------------|
| Word Jaccard (mean) | 0.2835 | 0.1813 |
| 3-gram overlap (mean) | 0.4372 | 0.3361 |

### Concrete Examples

#### Example 1: Split Decision (3 candidates side-by-side)

**Paper**: `EKjUnoX-7M0` (year=2022, pct_rating=0.3934)  
**Ground truth**: Reject  
**Majority prediction**: Reject  
**Individual votes**: ['Reject', 'Reject', 'Accept']  

<details><summary>Candidate 0 (Reject)</summary>

**strengths**:
The paper introduces a novel penalization framework for fairness in stochastic multi-armed bandits that allows for a tunable trade-off between cumulative reward and fairness constraints, moving beyond hard constraints. The theoretical analysis is comprehensive, establishing both gap-dependent and gap-independent upper bounds for the proposed hard-threshold UCB algorithm alongside corresponding lower bounds that demonstrate tightness in the gap-dependent regime. The experimental results effective...

**weaknesses**:
The gap-independent regret bound contains a suboptimal $O(T^{2/3})$ term arising from the penalization component, which the authors explicitly acknowledge as a significant theoretical gap compared to the standard $O(\sqrt{T})$ lower bound for bandits. The analysis relies on a 'hard-threshold' mechanism that introduces discontinuities in the index function, creating significant technical hurdles and preventing a clean decomposition of regret terms that are standard in classical UCB analysis. The ...

**critical_disputes**:
The core dispute lies in the gap-independent upper bound where the authors achieve $O(T^{2/3})$ while admitting the lower bound is $O(\sqrt{T})$, leaving a substantial open question on whether the proposed algorithm can ever match standard bandit rates in the worst-case scenario. The definition of the 'optimal arm' fluctuates dynamically based on the penalty term, which the authors argue creates unique challenges, yet this dynamic nature suggests the regret definition might not be the most natur...

**most_important_strength**:
The formulation of a unified penalized regret framework that rigorously quantifies the trade-off between exploration, exploitation, and fairness, enabling a flexible 'soft' constraint approach that adapts to different penalty rates.

**most_important_weakness**:
The gap-independent regret upper bound is suboptimal with an $O(T^{2/3})$ term, creating a significant gap with the known $\Omega(\sqrt{T})$ lower bound and undermining the claim of near-optimality in the worst-case setting.

**relation_to_other_work**:
The work extends classical fairness MAB literature by shifting from hard constraints or queue-based balancing to a penalized optimization framework, distinguishing itself from Joseph et al. (2016) and Patil et al. (2020) by allowing variable penalty rates. It shares conceptual similarities with LASSO regularization in statistics but applies this to the sequential decision-making context of bandits, offering a new theoretical perspective distinct from the queue-based approaches of Li et al. (2019...

**paper_quality**:
The paper is well-written with clear notation and a logical structure, but the theoretical contribution is marred by the loose gap-independent bound and the lack of comparison with state-of-the-art methods from 2020-2022. The experimental section is solid but limited in scope, and the technical complexity of the hard-threshold analysis introduces assumptions that weaken the general robustness of the proposed algorithm.

**accept**: False

</details>

<details><summary>Candidate 1 (Reject)</summary>

**strengths**:
The paper introduces a novel penalization framework for stochastic multi-armed bandits that explicitly balances cumulative reward against fairness constraints, moving beyond hard constraints. The proposed hard-threshold UCB algorithm is a distinct modification of classical indices that theoretically guarantees asymptotic fairness under specific penalty conditions. The theoretical analysis is comprehensive, providing rigorous gap-dependent and gap-independent upper bounds alongside matching lower...

**weaknesses**:
The gap-independent upper bound derived in Theorem 4 is significantly loose, scaling as O(T^{2/3}), while the corresponding lower bound is O(T^{1/2}), leaving a substantial gap in the theoretical understanding of the algorithm's optimality. The experimental evaluation is weak and insufficient, as it relies exclusively on synthetic Gaussian data with fixed parameters rather than real-world datasets or more diverse distributional assumptions. The comparison baselines are limited to two specific al...

**critical_disputes**:
The claim of 'nearly optimal regret' is factually incorrect for the gap-independent setting given the O(T^{2/3}) upper bound versus the O(T^{1/2}) lower bound. The assertion that the proposed method offers a 'better tradeoff' is not rigorously substantiated because the experimental design lacks statistical significance testing or analysis on diverse problem instances. The theoretical derivation of the gap-independent bound relies on a trivial lower bound of zero for the penalty term, which the a...

**most_important_strength**:
The formulation of a penalized regret metric that unifies reward maximization and fairness constraints into a single objective function allows for a flexible trade-off controlled by penalty rates, a perspective not previously formalized in the MAB literature.

**most_important_weakness**:
The inability to establish a tight O(T^{1/2}) gap-independent regret bound renders the theoretical contribution incomplete and raises questions about the algorithm's fundamental efficiency in the worst-case scenario.

**relation_to_other_work**:
The paper correctly identifies the gap in existing literature regarding penalized fairness MABs but fails to position itself adequately against the broader landscape of constrained optimization and resource allocation in bandits, limiting its impact to a specific niche of fairness definitions.

**paper_quality**:
The paper suffers from a significant disconnect between its ambitious theoretical claims and the actual results, particularly the loose gap-independent bound and the narrow experimental scope. While the problem formulation is interesting, the execution of the analysis and evaluation falls short of the standards required for a top-tier ML conference.

**accept**: False

</details>

<details><summary>Candidate 2 (Accept)</summary>

**strengths**:
The paper proposes a novel penalization framework for fairness in stochastic multi-armed bandits, explicitly defining a penalized regret metric that balances cumulative reward against fairness violations. The theoretical analysis is comprehensive, providing both gap-dependent and gap-independent upper bounds, alongside matching lower bounds that demonstrate the near-optimality of the proposed hard-threshold UCB algorithm in the gap-dependent regime. The experimental evaluation includes compariso...

**weaknesses**:
The gap-independent regret bound is sub-optimal, exhibiting a $O(T^{2/3})$ term that creates a significant gap with the information-theoretic lower bound of $O(\sqrt{T})$, leaving a major theoretical open question unresolved. The reliance on a hard-threshold mechanism introduces discontinuities that complicate the analysis and may lead to unstable behavior in practice, yet the paper provides no empirical evidence of such instability or discussion on the robustness of the algorithm to parameter m...

**critical_disputes**:
The central claim of achieving a 'complete' theory is undermined by the admission in Section 4.2 that the gap-independent upper bound is substantially loose ($T^{2/3}$ vs $\sqrt{T}$), which suggests the method is theoretically incomplete for finite-time guarantees in the worst case. The definition of the 'prophet' optimum assumes a static allocation strategy that ignores the sequential nature of learning, potentially inflating the perceived performance of the algorithm relative to a more dynamic...

**most_important_strength**:
The formulation of a rigorous penalized regret metric and the derivation of tight gap-dependent bounds that recover classical results when fairness penalties are zero provide a solid theoretical foundation for balancing fairness and reward in bandit settings.

**most_important_weakness**:
The failure to close the gap between the $O(T^{2/3})$ upper bound and the $O(\sqrt{T})$ lower bound for gap-independent regret indicates a significant limitation in the theoretical understanding of the algorithm's worst-case performance.

**relation_to_other_work**:
The work extends the specific lines of inquiry by Joseph et al. (2016) and Patil et al. (2020) by moving from hard constraints to a soft penalization framework, distinguishing it from the queue-based approach of Li et al. (2019) which struggles with asymptotic fairness guarantees.

**paper_quality**:
The paper demonstrates high technical competence with detailed mathematical derivations and clear experimental protocols, but the presence of a substantial unresolved gap in the theoretical bounds prevents it from being considered a definitive solution to the fairness MAB problem.

**accept**: True

</details>

#### Example 2: Correct Accept Prediction

**Paper**: `_eTZBs-yedr` (year=2023, pct_rating=0.7705)  
**Ground truth**: Accept  
**Majority prediction**: Accept  
**Individual votes**: ['Accept', 'Reject', 'Accept']  

<details><summary>Candidate 0 (Accept)</summary>

**strengths**:
The paper proposes a novel and well-motivated optimization framework, CrAM, that directly addresses the computational inefficiency of training separate models for specific compression targets by producing a single dense model robust to one-shot pruning. The empirical results are strong and extensive, demonstrating that CrAM-trained ResNet and BERT models can achieve high sparsity levels (up to 90-95%) with minimal accuracy loss, often outperforming or matching state-of-the-art gradual pruning me...

**weaknesses**:
The proposed method incurs a significant computational overhead, requiring two forward-backward passes per iteration (similar to SAM), which effectively doubles the training time compared to standard optimizers like SGD or Adam without providing a mechanism to reduce this cost to parity in the main experimental setup. The experimental evaluation relies heavily on BatchNorm Tuning (BNT) using a small calibration set to recover accuracy after pruning, yet the paper fails to rigorously disentangle ...

**critical_disputes**:
The claim that CrAM produces models that are 'stable' under compression is potentially misleading because the reported high accuracy after one-shot pruning is contingent upon an additional, albeit cheap, BatchNorm Tuning step; without this step, the performance drop would likely be severe, which undermines the 'one-shot' nature of the claim if the tuning is considered part of the deployment pipeline. The paper asserts that CrAM is the first to apply sharpness-aware concepts to compression, yet i...

**most_important_strength**:
CrAM successfully bridges the gap between dense training efficiency and sparse deployment flexibility, demonstrating that a single training run can yield a model competitive with specialized gradual pruning methods at high sparsity levels across both vision and language domains.

**most_important_weakness**:
The doubling of computational cost per iteration without a clear, demonstrated strategy to achieve the same final accuracy with standard training budgets or a reduced number of steps makes the method impractical for resource-constrained settings where efficiency is the primary motivation for compression.

**relation_to_other_work**:
CrAM extends the Sharpness-Aware Minimization (SAM) framework by replacing the loss sharpness objective with a compression-robustness objective, positioning itself as a specialized variant for model compression rather than generalization. It significantly advances over prior one-shot methods like Stochastic Frank-Wolfe by achieving comparable or superior results on large-scale datasets (ImageNet, BERT) where those methods previously failed. While it shares the goal of training prunable networks ...

**paper_quality**:
The paper is of high quality in terms of writing clarity, experimental breadth, and logical flow, presenting a compelling narrative supported by diverse benchmarks. However, the quality is marred by the lack of a fair baseline comparison regarding training wall-clock time and the insufficient ablation of the BatchNorm Tuning contribution, which casts doubt on the absolute robustness of the raw compressed model.

**accept**: True

</details>

<details><summary>Candidate 1 (Reject)</summary>

**strengths**:
The paper introduces CrAM, a principled optimization framework that extends Sharpness-Aware Minimization to explicitly target robustness against one-shot model compression, offering a novel perspective on training prunable networks. The empirical evaluation is extensive and compelling, demonstrating that CrAM-trained models achieve competitive or superior accuracy after one-shot pruning at high sparsity levels (up to 95% on CIFAR and 80-90% on ImageNet) compared to state-of-the-art gradual pruni...

**weaknesses**:
The primary limitation is the significant computational overhead, as the method requires two forward-backward passes per iteration, effectively doubling the training cost compared to standard SGD or Adam, which severely hinders scalability for massive models. The theoretical justification relies on Danskin's theorem and a straight-through gradient estimator for the non-differentiable Top-K operator, yet the paper lacks rigorous convergence guarantees or bounds on the quality of the stationary po...

**critical_disputes**:
The claim that CrAM produces models "compressible in one-shot" is partially undermined by the consistent inclusion of BatchNorm Tuning (BNT) in the reported results for ImageNet; without this cheap but necessary post-processing step, the accuracy drop at high sparsity would likely be much more severe, contradicting the narrative of a pure one-shot solution. The comparison against gradual pruning methods like PLATON or Movement Pruning is slightly apples-to-oranges, as CrAM requires double the co...

**most_important_strength**:
The ability to train a single dense model that remains highly accurate after aggressive one-shot pruning (up to 90-95% sparsity) and achieves performance competitive with complex, multi-stage gradual pruning pipelines represents a major practical advancement in deployment flexibility.

**most_important_weakness**:
The computational cost of requiring two full backward passes per training iteration makes the method prohibitively expensive for large-scale pre-training scenarios, potentially negating the efficiency gains achieved during the subsequent deployment phase.

**relation_to_other_work**:
CrAM directly builds upon Sharpness-Aware Minimization (SAM) by replacing the loss sharpness objective with a compression-robustness objective, distinguishing itself from Stochastic Frank-Wolfe methods like SFW by optimizing a non-convex loss landscape rather than forcing weights into a convex hull. While it shares the goal of training prunable networks with frameworks like Once-For-All (OFA) and slimmable networks, CrAM avoids the complexity of multi-width training and architecture search by mo...

**paper_quality**:
The paper is well-written with clear algorithmic descriptions and a logical flow from motivation to theory and extensive experiments. However, the experimental setup requires careful scrutiny regarding the normalization statistics correction (BNT), and the theoretical section, while referencing established optimization theorems, relies heavily on heuristic approximations for the non-differentiable operators without providing robust convergence analysis.

**accept**: False

</details>

<details><summary>Candidate 2 (Accept)</summary>

**strengths**:
The proposed CrAM method effectively addresses the practical bottleneck of model compression by enabling one-shot pruning without the need for iterative retraining or task-specific fine-tuning. The empirical results are strong, demonstrating that CrAM-trained models maintain high accuracy at 80-90% sparsity levels, directly competing with or surpassing complex gradual pruning baselines like PLATON and Movement Pruning on both ImageNet and BERT benchmarks. The experimental scope is comprehensive,...

**weaknesses**:
The method fundamentally inherits the computational inefficiency of Sharpness-Aware Minimization (SAM), requiring two full forward-backward passes per iteration, which doubles the training cost and negates the benefits of reduced hyperparameter search for compression targets. The theoretical justification relies on a straight-through gradient estimator through a non-differentiable Top-K operator, yet the paper fails to provide a rigorous convergence analysis or stability guarantees for this spec...

**critical_disputes**:
The core claim that CrAM produces models compressible in a 'single step' is misleading because the reported accuracy metrics consistently depend on an immediate post-hoc BatchNorm Tuning step using 1000 samples; without this step, the performance of pruned models degrades significantly, meaning the process is not truly training-time autonomous. The comparison against gradual pruning methods is slightly inequitable because the baselines often involve extensive fine-tuning schedules specifically t...

**most_important_strength**:
The ability to produce dense models that are robust to aggressive one-shot pruning (up to 90% sparsity) with accuracy competitive to state-of-the-art gradual pruning methods, thereby eliminating the need for multiple training runs for different deployment constraints.

**most_important_weakness**:
The doubling of training computational cost due to the SAM-like double backpropagation requirement, which significantly reduces the overall efficiency of the training pipeline compared to standard optimizers.

**relation_to_other_work**:
CrAM extends Sharpness-Aware Minimization (SAM) by replacing the loss sharpness objective with a compression robustness objective, distinguishing itself from Stochastic Frank-Wolfe approaches like SFW by using a direct perturbation-based update rather than projecting onto a convex hull of sparse vectors. It advances beyond existing one-shot pruning methods by achieving performance on large-scale datasets (ImageNet, BERT) where previous attempts failed, and it improves upon gradual pruning by rem...

**paper_quality**:
The paper is well-written with a clear narrative and logical structure, providing extensive ablation studies and comparisons against a wide range of strong baselines across multiple domains. However, the lack of a formal convergence proof for the straight-through estimator in this specific min-max setting and the omission of a detailed analysis of the BNT dependency slightly detract from the theoretical rigor. The experimental section is thorough, though the computational overhead is acknowledge...

**accept**: True

</details>

#### Example 3: Correct Reject Prediction

**Paper**: `Cy8Qa4gAst` (year=2026, pct_rating=0.1789)  
**Ground truth**: Reject  
**Majority prediction**: Reject  
**Individual votes**: ['Reject', 'Reject', 'Reject']  

<details><summary>Candidate 0 (Reject)</summary>

**strengths**:
The paper addresses a critical and timely problem: the severe performance degradation of large reasoning models (LRMs) after structured pruning, proposing a novel recovery strategy using reinforcement learning with verifiable rewards (RLVR). The empirical investigation into pass@k performance effectively demonstrates that pruning does not eliminate underlying reasoning capabilities, providing a strong theoretical justification for the proposed self-healing approach. The experimental results are ...

**weaknesses**:
The paper lacks a rigorous ablation study on the specific contributions of the temperature warm-up schedule versus the core RLVR algorithm, leaving the exact source of performance gains ambiguous. The comparison baselines are limited to SFT, failing to benchmark against other state-of-the-art recovery methods such as knowledge distillation, mask-aware fine-tuning, or alternative reinforcement learning algorithms like PPO which might yield different efficiency-accuracy trade-offs. The claims rega...

**critical_disputes**:
The assertion that this is the 'first' work to introduce RLVR for recovery training is factually weak, as applying reinforcement learning to post-training or recovery tasks in LLMs is a well-established technique in the 2024-2025 literature, even if not specifically framed as 'self-healing' for pruned LRMs. The paper conflates 'recovery' with 'training from scratch,' as the cold-start phase relies on SFT with reasoning trajectories, effectively making the method a two-stage training pipeline rat...

**most_important_strength**:
The comprehensive empirical demonstration that RLVR can recover over 95% of mathematical reasoning performance in heavily pruned LRMs, significantly outperforming the saturation limits of SFT-based recovery methods.

**most_important_weakness**:
The lack of comparative baselines beyond SFT and the failure to isolate the contribution of the proposed temperature warm-up schedule from the core RL optimization process.

**relation_to_other_work**:
While the paper correctly identifies the limitations of SFT in the specific context of pruned reasoning models, it neglects the broader landscape of RL-based fine-tuning and structured recovery techniques that have matured rapidly by 2026, making its claims of novelty regarding the use of RLVR appear incremental rather than groundbreaking.

**paper_quality**:
The paper is well-structured with clear experiments and a compelling narrative, but the lack of rigorous ablation studies and limited baseline comparisons significantly undermines the robustness of its claims and the depth of its technical contribution.

**accept**: False

</details>

<details><summary>Candidate 1 (Reject)</summary>

**strengths**:
The paper provides a timely and systematic empirical study on the impact of structured pruning on Large Reasoning Models (LRMs), a critical gap in the 2026 literature where most work focuses on standard LLMs. The central finding that pruning degrades reasoning quality but preserves underlying capability, measurable via pass@k, is a robust and insightful observation that fundamentally justifies the recovery approach. The methodological shift from imitation-based SFT to RLVR for recovery training ...

**weaknesses**:
The claim of 'first comprehensive study' and 'first to introduce RLVR' is factually incorrect given the 2026 landscape where RL-based pruning recovery has likely been explored by other groups by now, indicating a severe lack of awareness of the immediate prior art. The experimental design relies on a single, relatively narrow pruning algorithm (LLM-Pruner) and LayerDrop, failing to evaluate state-of-the-art 2026 structured pruning techniques like magnitude-based adaptive pruning or gradient-awar...

**critical_disputes**:
The assertion that this work introduces the first use of RLVR for pruning recovery is demonstrably false in a 2026 context, as the convergence of RLHF and model compression was a mainstream research track by 2025. The paper's dismissal of existing recovery methods as merely 'imitating reasoning trajectories' ignores significant advances in 2024-2025 that combined SFT with lightweight RL or contrastive learning to mitigate exactly these issues. The conclusion that SFT is fundamentally limited is ...

**most_important_strength**:
The empirical demonstration that structured pruning does not destroy the latent mathematical reasoning capability of LRMs, evidenced by the strong pass@k performance of the pruned models, provides a theoretically grounded foundation for the proposed recovery strategy.

**most_important_weakness**:
The paper's claim of novelty regarding the 'first' application of RLVR to pruning recovery is invalid and misleading, reflecting a critical failure to conduct a thorough literature review of the rapidly evolving 2025-2026 research landscape.

**relation_to_other_work**:
The paper positions itself against 2023-2024 pruning works like Sheared LLaMA and LLM-Pruner but completely fails to engage with the 2025-2026 wave of RL-based model compression and recovery papers, rendering its contribution claims obsolete. It treats SFT as a static baseline without acknowledging recent 2025 advancements in data-centric SFT for recovery that likely outperform the implementation used here.

**paper_quality**:
The paper is technically sound in its execution of RLVR but suffers from poor scholarship regarding the state of the field in 2026. The writing is clear, but the repeated and inaccurate claims of novelty significantly undermine the paper's credibility and contribution value. The experimental scope is too narrow to support broad claims about the effectiveness of the method across the entire field of LRM compression.

**accept**: False

</details>

<details><summary>Candidate 2 (Reject)</summary>

**strengths**:
The paper presents a timely and practically significant investigation into recovering pruned Large Reasoning Models (LRMs), a niche that is largely unexplored compared to standard LLM pruning. The empirical finding that pruning degrades but does not eliminate reasoning capabilities, supported by pass@k analysis, provides a strong theoretical basis for the proposed recovery strategy. The experimental results are compelling, showing that RLVR with verifiable rewards can overcome the performance sa...

**weaknesses**:
The claim of being the 'first' work to apply RLVR for pruning recovery is likely false given the rapid proliferation of RL-based fine-tuning papers in 2025-2026, suggesting a failure to conduct a comprehensive related work review. The experimental setup relies on a single rule-based reward function (exact answer matching), which ignores the quality of reasoning chains and risks overfitting to correct answers without genuine reasoning improvement. The evaluation of 'efficiency' is superficial; th...

**critical_disputes**:
The authors assert their method is the first to introduce RLVR for recovery, yet the landscape of 2026 likely includes numerous studies on RL for pruning or distillation that this paper ignores, making this novelty claim invalid. The comparison between SFT and RLVR is skewed because SFT is trained on high-quality CoT trajectories while RLVR is trained on simple math problems without trajectories, confounding the recovery strategy with the training data distribution. The assertion that pruning 'a...

**most_important_strength**:
The demonstration that RLVR can break the performance ceiling of SFT in the specific context of recovering pruned LRMs offers a novel and effective solution to a critical bottleneck in deploying efficient reasoning models.

**most_important_weakness**:
The paper ignores the computational cost of RLVR training (rollout generation) when claiming efficiency, rendering the 'self-healing' argument practically weak for resource-constrained scenarios compared to simpler SFT approaches.

**relation_to_other_work**:
The paper fails to adequately position itself against the extensive body of work on RL-based instruction tuning and pruning recovery published between 2024 and 2026, leading to unjustified novelty claims regarding the first application of RLVR to this problem.

**paper_quality**:
The paper is well-structured with clear empirical motivations, but it suffers from significant flaws in literature review, questionable novelty claims, and an incomplete analysis of the true computational costs involved in the proposed method.

**accept**: False

</details>

#### Example 4: False Positive (predicted Accept, actually Reject)

**Paper**: `ToWi1RjuEr8` (year=2021, pct_rating=0.0724)  
**Ground truth**: Reject  
**Majority prediction**: Accept  
**Individual votes**: ['Accept', 'Reject', 'Accept']  

<details><summary>Candidate 0 (Accept)</summary>

**strengths**:
The primary strength of this paper is the proposal of Advantage-Weighted Regression (AWR), a remarkably simple algorithm that achieves competitive performance with state-of-the-art methods like SAC and TD3 while using only standard supervised learning primitives. The method's ability to seamlessly handle fully off-policy learning from static datasets without the complex stabilization tricks required by Q-learning baselines is a significant and practical advantage. Furthermore, the ablation studi...

**weaknesses**:
The sample efficiency of AWR is demonstrably inferior to the strongest off-policy baselines like TD3 and SAC, as evidenced by the learning curves where AWR requires significantly more environment interactions to reach comparable asymptotic performance. The theoretical analysis, while grounded in constrained optimization, relies heavily on the assumption that the replay buffer accurately models a mixture of policies, yet it fails to rigorously address the distributional shift and potential diverg...

**critical_disputes**:
The paper claims that AWR is a general and scalable solution for off-policy RL, but the empirical results show it is strictly outperformed by TD3 and SAC in terms of sample efficiency on standard continuous control tasks, calling into question its 'scalability' for data-constrained scenarios. The derivation posits that the replay buffer defines a valid sampling policy for the constrained optimization, yet the analysis glosses over the fact that the value function fit to mixed data represents a w...

**most_important_strength**:
The algorithm's elegance and simplicity allow it to achieve strong performance using only standard supervised learning libraries, while uniquely avoiding the divergence issues common in off-policy Q-learning when trained on static datasets.

**most_important_weakness**:
Significantly lower sample efficiency compared to top-tier off-policy algorithms like TD3 and SAC, making it less practical for environments where data collection is expensive or time-limited.

**relation_to_other_work**:
AWR directly extends Reward-Weighted Regression by incorporating a value baseline and experience replay, differentiating itself from REPS and MPO by using simple supervised regression for value fitting rather than complex dual optimization, and from DDPG/SAC by avoiding Q-function bootstrapping entirely.

**paper_quality**:
The paper is well-written and clearly motivated, with a solid empirical evaluation on standard benchmarks and ablation studies that justify the proposed design choices. However, the theoretical analysis is somewhat hand-wavy regarding the off-policy convergence guarantees, and the results show a clear performance gap in sample efficiency compared to the strongest baselines, limiting the claims of broad superiority.

**accept**: True

</details>

<details><summary>Candidate 1 (Reject)</summary>

**strengths**:
The paper introduces a remarkably simple and modular algorithm that reduces reinforcement learning to standard supervised regression steps, significantly lowering implementation barriers compared to complex actor-critic or Q-learning methods. The theoretical derivation effectively frames the method as a constrained policy optimization problem, justifying the use of exponentiated advantages and providing a principled interpretation of experience replay as a mixture of past policies. Empirically, ...

**weaknesses**:
The algorithm fails to match the sample efficiency of state-of-the-art off-policy methods like SAC and TD3 on standard online control benchmarks, often requiring significantly more environment interactions to reach comparable asymptotic performance. The theoretical analysis relies on first-order approximations and assumes specific properties of the replay buffer that may not hold in practice, offering limited guarantees regarding convergence stability in high-dimensional or non-Markovian setting...

**critical_disputes**:
The claim that AWR is 'simple' and 'scalable' is undermined by its inferior sample efficiency, which forces practitioners to pay a steep data cost that negates the benefits of reduced algorithmic complexity in sample-constrained real-world domains. The assertion that the method handles off-policy data robustly is contradicted by the fact that performance degrades significantly when the replay buffer size is small, indicating a fragility to data distribution shifts that the theoretical derivation...

**most_important_strength**:
The ability to leverage fully off-policy static datasets and achieve performance competitive with specialized offline reinforcement learning algorithms using a unified, simple regression framework without complex constraint mechanisms.

**most_important_weakness**:
Significantly lower sample efficiency compared to leading off-policy algorithms like SAC and TD3, rendering the method impractical for data-limited applications despite its algorithmic simplicity.

**relation_to_other_work**:
AWR acts as a modern, deep learning adaptation of classical Reward-Weighted Regression and Relative Entropy Policy Search, integrating experience replay to bridge the gap between on-policy EM methods and modern off-policy actor-critic algorithms. It occupies a niche between the instability of pure Q-learning and the sample inefficiency of on-policy methods, though it ultimately trails behind TD3 and SAC in raw sample efficiency. The approach shares conceptual similarities with MPO but simplifies...

**paper_quality**:
The paper is well-written with clear mathematical derivations and comprehensive experiments across online, offline, and motion imitation settings, but the core contribution suffers from a lack of novelty in the fundamental update rule which is essentially a weighted version of standard policy regression. While the experimental evaluation is thorough, the results do not convincingly demonstrate that the simplicity of the method justifies its lower sample efficiency compared to existing state-of-t...

**accept**: False

</details>

<details><summary>Candidate 2 (Accept)</summary>

**strengths**:
The proposed Advantage-Weighted Regression (AWR) offers a remarkably simple architecture that reduces off-policy reinforcement learning to two standard supervised learning steps, significantly lowering implementation complexity compared to TD3 or SAC. The theoretical derivation convincingly connects the method to constrained policy optimization via expected improvement, justifying the use of advantage-weighted regression and the experience replay mechanism as a mixture of past policies. Empirica...

**weaknesses**:
The algorithm exhibits significantly lower sample efficiency than current state-of-the-art off-policy methods like SAC and TD3, often requiring substantially more environment interactions to reach comparable asymptotic performance. The experimental evaluation lacks rigorous statistical significance testing, such as confidence intervals across seeds for the learning curves, and relies heavily on mean and standard deviation in tables which may obscure variance issues. The method relies on heuristi...

**critical_disputes**:
The claim that AWR is 'substantially simpler' than MPO or SAC is overstated because the algorithm's performance critically depends on non-trivial engineering decisions like weight clipping, TD(lambda) bootstrapping, and specific replay buffer sizing that introduce their own complexity and tuning burdens. The theoretical analysis assumes the sampling policy is a mixture of past policies modeled by the replay buffer, yet it glosses over the distribution shift errors that inevitably arise when the ...

**most_important_strength**:
AWR uniquely bridges the gap between simple supervised learning and effective off-policy reinforcement learning, achieving competitive asymptotic performance on complex continuous control tasks while naturally handling fully off-policy static datasets without the instability and complexity inherent in Q-function-based methods.

**most_important_weakness**:
The method suffers from poor sample efficiency compared to leading off-policy algorithms, often failing to match the data utilization rates of SAC or TD3, which severely limits its practical applicability in sample-constrained real-world scenarios.

**relation_to_other_work**:
AWR extends Reward-Weighted Regression (RWR) by introducing advantage-weighting and experience replay to achieve off-policy learning, positioning itself as a simpler alternative to Maximum a Posteriori Policy Optimization (MPO) and REPS by avoiding complex dual optimization or Bellman error minimization. It shares conceptual similarities with soft actor-critic in its use of entropy regularization but avoids the dual Q-network architecture, and it relates to Batch-Constrained Q-learning (BCQ) by ...

**paper_quality**:
The paper presents a clear and well-motivated contribution with a solid theoretical foundation and comprehensive empirical evaluation across diverse tasks, though the writing occasionally lacks rigor in statistical reporting and fails to deeply address the sample efficiency gap with state-of-the-art baselines. The experimental setup is generally sound, but the omission of detailed hyperparameter sensitivity analysis and the reliance on heuristic temperature selection weaken the robustness of the...

**accept**: True

</details>

#### Example 5: False Negative (predicted Reject, actually Accept)

**Paper**: `5p9uled7JM` (year=2026, pct_rating=0.9623)  
**Ground truth**: Accept  
**Majority prediction**: Reject  
**Individual votes**: ['Reject', 'Reject', 'Reject']  

<details><summary>Candidate 0 (Reject)</summary>

**strengths**:
Kanzi achieves state-of-the-art reconstruction metrics with significantly fewer parameters and training data than major baselines like ESM3 and DPLM-2, demonstrating exceptional parameter efficiency. The proposed simplification of replacing complex SE(3)-invariant components with standard attention and a single flow matching loss creates a more scalable and easier-to-optimize architecture. The paper introduces the rFPSD metric to address the limitations of point-wise reconstruction scores like R...

**weaknesses**:
The generated structures exhibit a severe bias toward alpha-helical content, with Table 3 showing alpha percentages exceeding 88% compared to the natural distribution, indicating a fundamental failure to model diverse protein folds. The autoregressive generator fails to match the designability and diversity of strong continuous diffusion baselines, and the authors admit the model does not yet achieve state-of-the-art generative performance despite the superior tokenizer. The reliance on a small ...

**critical_disputes**:
The claim that non-invariant encoders are superior is contradicted by the authors' own findings that invariant encoders lead to codebook collapse, suggesting the model may simply be overfitting to the training distribution's lack of rotational augmentation rather than learning a robust generalizable representation. The assertion that Kanzi is the first tokenized model to produce designable structures without massive pretraining is misleading given the poor designability scores (0.328-0.617) and ...

**most_important_strength**:
Kanzi demonstrates that a simplified, non-invariant flow-based tokenizer can outperform large, specialized SE(3)-equivariant models on reconstruction metrics while using a fraction of the computational resources.

**most_important_weakness**:
The generative model trained on Kanzi tokens exhibits a catastrophic bias toward alpha-helices and fails to generate diverse, designable structures, rendering the tokenizer ineffective for its primary downstream application despite strong reconstruction scores.

**relation_to_other_work**:
The work positions itself as a simplification of the complex SE(3)-invariant tokenizers used in ESM3 and DPLM-2 by leveraging flow matching, but it ultimately underperforms the generative capabilities of the very continuous diffusion models it attempts to complement.

**paper_quality**:
The paper suffers from significant issues in experimental rigor, including the use of synthetic training data for benchmarking against models trained on the same data, a failure to control for secondary structure bias in generation, and an overstatement of claims regarding designability. While the architectural simplification is interesting, the poor generative outcomes and the reliance on a limited dataset make the contribution incremental at best for the broader goal of protein design.

**accept**: False

</details>

<details><summary>Candidate 1 (Reject)</summary>

**strengths**:
The paper presents Kanzi, a flow-based protein structure tokenizer that achieves state-of-the-art reconstruction metrics with significantly fewer parameters (30M vs 600M+ for ESM3) and less training data than existing methods. The architectural simplification of replacing complex SE(3)-invariant modules with standard attention and a single flow matching loss offers a compelling alternative to current inductive bias-heavy approaches, demonstrating that non-invariant models can learn spatial symme...

**weaknesses**:
The generative results reveal a severe failure mode where the autoregressive model overpredicts alpha helices (88.7% alpha content vs typical biological distributions), indicating a fundamental distributional shift that undermines the claim of producing 'designable' structures comparable to state-of-the-art baselines. The comparison to diffusion-based generative models is weak; the paper admits Kanzi does not match state-of-the-art continuous diffusion performance, yet claims superiority over to...

**critical_disputes**:
The authors claim Kanzi is the 'first tokenized model that produces designable structures without massive pretraining,' yet Table 3 shows their best designability score (0.617) is comparable to or lower than ESM3-AR (0.520) and DPLM2 (0.486) depending on the metric, and the extreme alpha-helix bias (88.7% vs ~40-45% in natural proteins) suggests the generated structures are physically unrealistic despite favorable TM scores. The argument that non-invariant encoders are superior is contradicted b...

**most_important_strength**:
Kanzi achieves reconstruction performance matching or exceeding much larger, data-hungry tokenizers (like ESM3) while using a drastically simplified architecture that relies on standard attention and a single flow matching loss, proving that explicit SE(3) invariance is not strictly necessary for high-quality protein structure tokenization.

**most_important_weakness**:
The autoregressive generation pipeline produces structurally plausible but biologically unrealistic proteins characterized by a massive overrepresentation of alpha helices, indicating the model fails to capture the true distribution of natural protein folds despite achieving good reconstruction metrics.

**relation_to_other_work**:
The work directly challenges the prevailing paradigm of SE(3)-invariant architectures established by AlphaFold2 and subsequent tokenizers like ESM3, aligning instead with recent findings (e.g., Genie2, Proteina) that suggest learnable invariance is sufficient. It extends the application of flow autoencoders from computer vision (FlowMo, DiTo) to the biological domain, while distinguishing itself from diffusion-based protein generators by prioritizing autoregressive flexibility over pure generati...

**paper_quality**:
The paper is well-written with a clear narrative and comprehensive experimental setup, including multiple datasets and ablation studies. However, the interpretation of results is overly optimistic regarding generative capabilities, particularly when ignoring the severe structural bias in generated samples. The inclusion of a novel metric (rFPSD) adds value, but the failure to adequately address the biological implausibility of the generated structures weakens the overall impact of the contributi...

**accept**: False

</details>

<details><summary>Candidate 2 (Reject)</summary>

**strengths**:
The paper presents a compelling empirical argument that standard attention mechanisms and flow matching can replace complex SE(3)-invariant architectures for protein structure tokenization, simplifying the training pipeline significantly. The proposed Kanzi tokenizer achieves state-of-the-art reconstruction metrics on multiple benchmarks (CAMEO, CASP, CATH) while utilizing a fraction of the parameters and training data required by competitors like ESM3 and DPLM2. The introduction of the rFPSD me...

**weaknesses**:
The autoregressive generation model suffers from a severe secondary structure imbalance, overpredicting alpha helices and completely collapsing the beta strand distribution, which fundamentally limits its utility for generating diverse, naturalistic proteins. The evaluation of generative capabilities is inconsistent, as the model relies on best-of-N sampling and heavy guidance tuning to achieve competitive scores, masking the poor performance of the unconditional distribution. The claim that non...

**critical_disputes**:
The central claim that removing SE(3) inductive biases is superior is contradicted by the model's failure to generate stable beta sheets without explicit conditioning or massive pre-training, suggesting that geometric invariance remains critical for modeling specific folding motifs. The reported efficiency gains are misleading because the model was trained on a relatively small, filtered dataset (approx. 500k structures) compared to the hundreds of millions used by state-of-the-art baselines, ma...

**most_important_strength**:
The demonstration that a simple flow-matching loss and standard attention mechanisms can achieve superior reconstruction accuracy with significantly smaller models, effectively challenging the dogma that explicit SE(3) invariance is strictly necessary for tokenization.

**most_important_weakness**:
The catastrophic failure in generating diverse secondary structures, specifically the near-total absence of beta strands in unconditional sampling, which invalidates the model's utility as a general-purpose protein generator.

**relation_to_other_work**:
The work directly challenges the architectural paradigms of ESM3, DPLM-2, and AlphaFold3 by removing geometric invariance, yet it fails to match their generative diversity and stability, suggesting that while tokenization can be simplified, generation may still require stronger inductive biases.

**paper_quality**:
The paper is well-written with clear motivation and a logical flow, but the experimental evaluation is incomplete and the conclusions are overstated given the model's inability to generate balanced protein structures. The presentation of results often highlights best-case scenarios while downplaying the significant distributional failures of the generative model.

**accept**: False

</details>

## Key Findings

1. **Overall accuracy**: 52.1% across 21960 papers
2. **Massive rejection bias**: 94.8% of individual candidates predict reject vs 50.6% actual reject rate
3. **Unanimous agreement**: 89.0% of papers have all 3 candidates agree
4. **Borderline difficulty**: Accuracy on borderline papers (pct_rating 0.4-0.6) is 65.0% vs 49.2% on non-borderline
5. **2026 (unseen year)**: Accuracy 51.6% on 8004 papers
6. **Content similarity**: Pairwise word Jaccard for strengths=0.284, weaknesses=0.181
