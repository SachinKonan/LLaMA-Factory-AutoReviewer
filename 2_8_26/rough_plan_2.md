# Working Plan: AutoReviewer Bias Mitigation & Scaling

**Status**: Draft for Review
**Date**: 2026-02-09

## 1. Objective
Understand "optimism" bias (between 95-100% acceptance rate from models) and improve overall accuracy/recall for ICLR paper review prediction.

## 2. Research Directions

### A. Bias Investigation
**Hypothesis 1**: Instruct finetuning is causing this phenomenon. 
-   **Experiment**: Use the qwen2.5 base model to predict accept/reject. 
-   **Analysis**: Compare the acceptance rates of the base model vs the instruct model. 
-   **Follow up**: Which papers are accepted/rejected by the base model but not the instruct model? Provide plots that break down their average novelty, presentation, and soundness, rating scores. 


**Hypothesis 2**: Contamination. Some papers are much more likely to occur in training data multiple times
-   **Experiment**: Using the title of each paper, query the model to see if it knows the contents. 
-   **Ablations**: Ask for 1) abstract, 2) intro, 3) conclusion, 4) full paper. Compare the embedding similarity of abstracts using qwen embedding model. 
-   **Analysis**: Which papers have high embedding similarity? Are they highly cited papers? Do they have higher acceptance rates? 

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


### C. Noise Modeling & Bayesian Decision
**Observation**: We can estimate the "noise" or "confusion matrix" of the model (e.g., $P(\text{Predicted Accept} | \text{True Reject})$).
-   *Proposed Model*:
    -   $P(R_{\text{pred}} | A_{\text{true}}) \approx 50\%$ (High noise / harshness?)
    -   $P(A_{\text{pred}} | R_{\text{true}}) \approx 20\%$ (Optimism?)
-   *Action*:
    -   Estimate these probabilities empirically from the validation set (2020-2024).
    -   Use these priors to compute a **Bayesian Optimal** decision threshold or re-weight predictions during inference.
    -   Feature: "Knowing the noise, can we find the maximum likelihood label?"
    -   Finetune with 3B model. If we use fuzzy accept/reject numbers, can we improve the performance? Evaluate MSE versus BCE loss. Do we use different loss values for correctness? 


## 4. Discussion of Open Questions / Decisions Needed

1.  **Noise Model Numbers**: The rough plan mentions $P(R|A) = 50\%$ and $P(A|R) = 20\%$. Are these hypothetical values to test the math, or estimated from a specific run? 
-   These are numbers that are roughly correct. These can be used.
2.  **Parallel Distill Response**: Is this different from the existing `metareview` strategy? (Line 11 in rough_plan says "Parallel Distill Response... Summarizer across them").
-   Yes, it is different. Search for the paper "Rethinking Thinking Tokens: LLMs as Improvement Operators" and its implementation there.
3.  **Non-Instruct Models**: Do we have specific base model checkpoints in mind (e.g., `Qwen/Qwen2.5-7B`)?
-   Yes. 
4.  **"Role Bad Review"**: Should this be a separate dataset generation or just a prompt override at inference time?
-   Generate new datasets.
5.  **Gemini Title-only**: Is there a script for this, or should we create a quick `gemini_title_infer.py`?
-   This requires a new script. Be sure to make a new dataset with just titles and a new system and user prompt to reflect this task. 
