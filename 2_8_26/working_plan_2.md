# Working Plan 2: AutoReviewer Bias Mitigation & Scaling

**Status**: Draft for Review
**Date**: 2026-02-09
**Refines**: `rough_plan_2.md`

## 1. Objective
Understand the "optimism" bias (95-100% acceptance rate in current models) and improve overall accuracy/recall for ICLR paper review prediction.

## 2. Research Directions

### A. Bias Investigation
**Hypothesis 1: Instruct Finetuning Artifact**
*   **Theory**: Instruct models are RLHF-tuned to be helpful/agreeable, causing extreme optimism.
*   **Experiment**: Run **Qwen 2.5 Base** (non-instruct) model.
    *   *Challenge*: Base models don't follow complex JSON schemas easily. May need simple completion prompts or few-shot examples.
*   **Analysis**:
    *   Compare acceptance rates (Base vs. Instruct).
    *   Identify "disagreement papers" (Accepted by Base, Rejected by Instruct and vice versa).
    *   Plot breakdowns of novelty/presentation/soundness scores for these papers.

**Hypothesis 2: Data Contamination**
*   **Theory**: Papers, especially famous ones, are in the pre-training corpus, leading to memorized positive sentiment.
*   **Experiment**: **Leaking Check** via Title Query.
    *   *Method*: Feed *only* the title to the model. Ask: "Do you know this paper? Summarize it." or try to predict acceptance solely from the title.
*   **Ablations**:
    1.  Title only.
    2.  Title + Abstract.
    3.  Title + Intro.
    4.  Title + Conclusion.
    5.  Full Paper (Baseline).
*   **Metric**: Embedding similarity (using Qwen-Embedding) between generated summary/abstract and actual abstract.
*   **Correlation**: Check if high "recognition" (embedding similarity) correlates with high acceptance rate or citation count.

### B. Prompt Engineering & Ensembling (Inference Scaling)
**1. Parallel Distill Response (PDR)**
*   **Reference**: "Rethinking Thinking Tokens: LLMs as Improvement Operators".
*   **Method**:
    *   **Phase 1 (Parallel)**: Generate $N$ (e.g., 5) diverse reviews in parallel.
    *   **Phase 2 (Distill)**: Use a "Summarizer" model to synthesize these into a single "meta-response" or decision, acting as an improvement operator.
*   **Differentiation**: Unlike simple majority voting, this allows the model to *reason* over the diverse perspectives.

**2. Role-Playing / Adversarial Prompts**
*   **Method**: Create **new datasets** where the system prompt explicitly sets a persona.
    *   *Strategy A*: "Bad Cop" / "Critical Reviewer" (Bias towards rejection/fault-finding).
    *   *Strategy B*: "Standard Reviewer".
*   **Goal**: Determine if "Bad Cop" reduces False Positive Rate on (High Presentation, Low Soundness) papers.

### C. Noise Modeling & Bayesian Optimization
**Proposed Model**:
*   Estimated Confusion Matrix (Priors):
    *   $P(\text{Pred Reject} | \text{True Accept}) \approx 50\%$ (Harshness/Noise)
    *   $P(\text{Pred Accept} | \text{True Reject}) \approx 20\%$ (Optimism)
*   **Bayesian Decision**:
    *   Use these priors to calculate $P(\text{TrueLabel} | \text{Prediction})$.
    *   Find the decision threshold that maximizes accuracy under these noise conditions.
*   **Finetuning Experiment**:
    *   Finetune a smaller model (**3B**) to predict "fuzzy" scores (e.g., probability of acceptance) instead of binary labels.
    *   **Loss Function**: Compare MSE (Mean Squared Error) vs. BCE (Binary Cross Entropy) on these soft targets.

## 3. Implementation Plan

### Phase 1: Infrastructure & New Datasets
-   [ ] **Gemini Title-Only Script**:
    -   Create `gemini_title_infer.py`.
    -   Create `generate_title_dataset.py`: Extracts only titles.
    -   New System Prompt: "You are checking for paper familiarity..."
-   [ ] **Role-Based Datasets**:
    -   Modify `generate_datasets.py` to accept a `system_prompt_type` argument ("standard", "critical").
    -   Generate `clean_critical` datasets.

### Phase 2: Experiments (Running Inference)
-   [ ] **Base Model Inference**:
    -   Adapt `vllm_infer.py` to support `Qwen/Qwen2.5-7B` (base).
    -   *Note*: Might need a custom "completion" loop if the chat template doesn't work well.
-   [ ] **Contamination Ablations**:
    -   Run the Title/Abstract/Intro sequence.
    -   Script to compute embedding similarity (Qwen-Embedding).
-   [ ] **PDR Pipeline**:
    -   Run `n=5` generation.
    -   Implement the "Distill/Summarizer" step (distinct from simple meta-review).

### Phase 3: Analysis
-   [ ] **Noise Modeling**:
    -   Script to calculate empirical confusion matrices from Validation set (2020-2024).
    -   Apply Bayesian correction to Test set (2025).
-   [ ] **Finetuning 3B**:
    -   Setup training script for Qwen-2.5-3B.
    -   Implement MSE vs BCE loss comparison.

## 4. Immediate Next Steps / User Review
*   Confirm the "Leaking Check" methodology (Title -> Summary -> Embedding Sim).
*   Approve the creation of `gemini_title_infer.py` and `generate_title_dataset.py`.
