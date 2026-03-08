# Working Plan: AutoReviewer Bias Mitigation & Scaling

**Status**: Draft for Review
**Date**: 2026-02-09

## 1. Objective
Reduce "optimism" bias (high false positive rate on well-presented but low-soundness papers) and improve overall accuracy/recall for ICLR paper review prediction.

## 2. Research Directions

### A. Bias Investigation
**Problem**: Models exhibit an "accept bias" (optimism), especially for well-written papers.
-   **Hypothesis 1**: Presentation masks poor soundness (confirmed by `report_update.md`).
-   **Hypothesis 2 (Contamination)**: Are these papers in the training data?
    -   *Action*: Identify papers likely in the training set (e.g., highly cited, older years) and check performance difference.
-   **Hypothesis 3 (Instruction Tuning)**: Instruct models are trained to be "helpful" and "nice".
    -   *Action*: Test **non-instruct (base)** models to see if they are less biased.

### B. Prompt Engineering & Ensembling Strategies
Move beyond simple "predict decision" to more complex reasoning pipelines.

1.  **Parallel Distill Response (PDR) / Metareview**:
    -   *Concept*: Generate $N$ reviews (e.g., $N=5$), then use a "Summarizer" model to aggregate them into a final decision.
    -   *Current State*: We have `vllm_infer.py` with `n_generations` and `run_metareview.py`.
    -   *Refinement*: Experiment with the "Summarizer" prompt to be more critical.

2.  **Role-Playing / Adversarial Prompts**:
    -   *Concept*: Explicitly prompt the model to adopt a specific persona.
    -   *Strategy 1*: "Role: Bad Cop" / "Critical Reviewer" - bias towards finding faults.
    -   *Strategy 2*: Standard/Balanced Reviewer.
    -   *Goal*: See if "Bad Cop" reduces the FPR on (High Presentation, Low Soundness) papers.

3.  **Prompt Variations**:
    -   Simple Prompt vs. New (Detailed) Prompt.
    -   Few-Shot: Add examples (1 accept, 1 reject).
    -   *Result Extraction*: Compare directly predicting probability vs. generating text.

### C. Noise Modeling & Bayesian Decision
**Observation**: We can estimate the "noise" or "confusion matrix" of the model (e.g., $P(\text{Predicted Accept} | \text{True Reject})$).
-   *Proposed Model*:
    -   $P(R_{\text{pred}} | A_{\text{true}}) \approx 50\%$ (High noise / harshness?)
    -   $P(A_{\text{pred}} | R_{\text{true}}) \approx 20\%$ (Optimism?)
-   *Action*:
    -   Estimate these probabilities empirically from the validation set (2020-2024).
    -   Use these priors to compute a **Bayesian Optimal** decision threshold or re-weight predictions during inference.
    -   Feature: "Knowing the noise, can we find the maximum likelihood label?"

### D. Model Variations
-   **Gemini Experiments**:
    -   Test `Gemini 2.5 Flash`.
    -   *Title-Only Test*: Does the model predict accept/reject based *only* on the title? (Checks for popularity bias / lookup).
-   **Vision Models**: Continue testing Qwen-VL to see if visual grounding reduces text-based hallucinations (initial results suggest yes).

## 3. Implementation Plan (This Week)

### Phase 1: Infrastructure & Baselines
-   [ ] **Contamination Check**: Script to correlate paper age/citations with model accuracy.
-   [ ] **Non-Instruct Model Setup**: Configure `vllm_infer.py` to run Qwen base model (requires different chat template or raw completion interface).

### Phase 2: Advanced Strategies
-   [ ] **Implement "Role-Playing" Prompts**: 
    -   Add `strategy` argument to `dataset_generation` to inject "You are a critical reviewer..." system prompts.
-   [ ] **Scale PDR**:
    -   Run `n=5` generation for "Bad Cop" and "Standard" prompts.
    -   Run `metareview` aggregation on these outputs.

### Phase 3: Analysis
-   [ ] **Noise Modeling Script**: 
    -   Compute $P(\text{Pred}|\text{Truth})$ confusion matrices from existing results.
    -   Implement a post-processing script to apply Bayesian correction.

## 4. Open Questions / Decisions Needed

1.  **Noise Model Numbers**: The rough plan mentions $P(R|A) = 50\%$ and $P(A|R) = 20\%$. Are these hypothetical values to test the math, or estimated from a specific run?
2.  **Parallel Distill Response**: Is this different from the existing `metareview` strategy? (Line 11 in rough_plan says "Parallel Distill Response... Summarizer across them").
3.  **Non-Instruct Models**: Do we have specific base model checkpoints in mind (e.g., `Qwen/Qwen2.5-7B`)?
4.  **"Role Bad Review"**: Should this be a separate dataset generation or just a prompt override at inference time?
5.  **Gemini Title-only**: Is there a script for this, or should we create a quick `gemini_title_infer.py`?
