# Working Plan 3: AutoReviewer Bias Mitigation & Scaling

**Status**: Draft for Review
**Date**: 2026-02-09
**Refines**: `rough_plan_3.md`

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
*   **Analysis**: Correlation between "recognition" (embedding similarity) and acceptance rate/citation count.

### B. Prompt Engineering & Ensembling (Inference Scaling)
**1. Parallel Distill Response (PDR)**
*   **Reference**: "Rethinking Thinking Tokens: LLMs as Improvement Operators".
*   **Method**:
    *   **Phase 1 (Parallel)**: Generate $N$ (e.g., 5) diverse reviews in parallel.
    *   **Phase 2 (Distill)**: Use a "Summarizer" model to synthesize these into a single "meta-response" or decision.
*   **Differentiation**: Unlike simple majority voting, this allows the model to *reason* over the diverse perspectives.

**2. Role-Playing / Adversarial Prompts**
*   **Method**: Create **new datasets** where the system prompt explicitly sets a persona.
    *   *Strategy A*: "Bad Cop" / "Critical Reviewer" (Bias towards rejection/fault-finding).
    *   *Strategy B*: "Standard Reviewer".
    *   *Strategy C*: "Enthusiastic Reviewer" (Bias towards acceptance).
    *   *Strategy D*: **Multi-Perspective Meta-Review**.
        *   Step 1: Generate "Critical" review (only weaknesses).
        *   Step 2: Generate "Enthusiastic" review (only strengths).
        *   Step 3: Meta-Reviewer aggregates both to make a final decision.
*   **Goal**: Determine if explicitly separating concerns (strengths vs weaknesses) reduces the "halo effect" of good presentation.
*   **Re-Analysis**: Redo [High/Low Presentation] x [High/Low Soundness] plots with native acceptance rates and proper normalization.

### C. Noise Modeling & Bayesian Optimization
**Proposed Model**:
*   Estimated Confusion Matrix (Priors):
    *   $P(\text{Pred Reject} | \text{True Accept}) \approx 50\%$ (Harshness/Noise)
    *   $P(\text{Pred Accept} | \text{True Reject}) \approx 20\%$ (Optimism)
*   **Bayesian Decision**:
    *   Use these priors to calculate $P(\text{TrueLabel} | \text{Prediction})$.
    *   Find the decision threshold that maximizes accuracy.
*   **Finetuning Experiment**:
    *   Finetune a smaller model (**Qwen 2.5 3B**) to predict **soft labels** (e.g. 0.1, 0.9) to represent uncertainty.
    *   **Loss Function**: Compare predictions trained with MSE vs. BCE.

## 3. New Questions & Proposed Experiments

**Q1: Reviewer Style Modeling & Archetypes**
*   *Question*: Can we model specific reviewer biases and "archetypes" to better understand the gap between model and human reviewers?
*   *Data*: We have the full text of **ground truth reviews**.
*   *Experiment*:
    1.  **Archetype Clustering**: Use unsupervised clustering (e.g., on embeddings of review text) to categorize reviews into styles (e.g., "Pedantic/Nitpicky", "Big Picture", "Methodology-Focused", "Tone-Policing").
        *   Cluster *ground truth* reviews to define the search space.
        *   Classify *generated* reviews into these clusters.
    2.  **Distribution Shift**: Compare the distribution of archetypes. Does the model exclusively generate "Big Picture" reviews while humans are "Nitpicky"?
    3.  **Performance Correlation**: Analyze which archetypes (when generated) correlate best with the final ground truth decision.
    4.  **Action**: If "Nitpicky" reviews are more accurate but under-represented, we can instruct-tune or prompt for that specific persona (linked to Strategy D).

## 4. Implementation Plan

### Phase 1: Infrastructure & New Datasets
-   [ ] **Gemini Title-Only Script**: `gemini_title_infer.py` + `generate_title_dataset.py`.
-   [ ] **Role-Based Datasets**:
    -   Modify `generate_datasets.py` to support `system_prompt_type` ("critical", "enthusiastic", "standard").
    -   Generate `clean_critical` and `clean_enthusiastic` datasets.

### Phase 2: Experiments (Running Inference)
-   [ ] **Specific Datasets & Models**:
    -   **Text-Only**: `iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test`
        -   Model: `Qwen/Qwen2.5-7B-Instruct`
    -   **Text + Images**: `iclr_2024_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7_test`
        -   Model: `Qwen/Qwen2.5-VL-7B-Instruct`
    -   **Vision**: `iclr_2020_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7_test`
        -   Model: `Qwen/Qwen2.5-VL-7B-Instruct`

-   [ ] **Base Model Inference**: Adapt `vllm_infer.py` for Qwen-Base on the Text-Only dataset.
-   [ ] **Contamination Ablations**: Run Title/Abstract/Intro sequence.
-   [ ] **Strategy D Implementation**:
    -   Pipeline: Critical Generation -> Enthusiastic Generation -> Meta-Review.

### Phase 3: Analysis
-   [ ] **Noise Modeling**: Bayesian correction script.
-   [ ] **Finetuning 3B**: Soft-label training (MSE vs BCE).
-   [ ] **Reviewer Style Analysis**: Cluster ground truth vs. generated reviews.

## 5. Immediate Next Steps / User Review
*   Approve `gemini_title_infer.py` creation.
*   Confirm the "Strategy D" pipeline logic.
*   Feedback on Q1 (Reviewer Style) proposal.
