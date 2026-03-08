# H1 Experiment: Base vs Instruct Model Evaluation

**Date**: 2026-02-11
**Hypothesis**: The base model (Qwen2.5-7B) is less optimistic than the instruct-tuned model (Qwen2.5-7B-Instruct)

---

## Executive Summary

❌ **The hypothesis is REJECTED** — but not in the expected way.

**Key Finding**: BOTH models exhibit extreme optimism bias, predicting "Accept" for nearly all papers (~97-99%). The base model is only marginally less optimistic (97.3% vs 99.4%), but this 2.1 percentage point difference is negligible compared to the massive systematic bias in both models.

---

## Data Verification

### Timestamps ✅
- **Base model results**: Generated Feb 10 13:09
- **Instruct model results**: Generated Feb 10 12:42
- **Analysis plots**: Regenerated Feb 11 (current)

All data is from the same timeframe and consistent.

### Model Configuration ✅

| Aspect | Base Model | Instruct Model |
|--------|-----------|----------------|
| Model | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-7B-Instruct` |
| Template | `default` (no chat format) | `qwen` (chat format) |
| Prompt Style | Completion-based | Instruction-following |
| Dataset | test_base_model | clean/standard |

Models are **correctly configured** — not swapped.

---

## Results

### Ground Truth Distribution
- **Accept**: 1,013 papers (50.0%)
- **Reject**: 1,011 papers (50.0%)
- **Total**: 2,024 papers

The dataset is **perfectly balanced**.

### Prediction Distribution

| Model | Accept Predictions | Reject Predictions | Acceptance Rate |
|-------|-------------------|-------------------|-----------------|
| **Base** | 1,956 (97.3%) | 55 (2.7%) | 97.3% |
| **Instruct** | 1,975 (99.4%) | 11 (0.6%) | 99.4% |

### Performance Metrics

| Metric | Base Model | Instruct Model |
|--------|-----------|----------------|
| **Accuracy** | 49.5% | 50.4% |
| **Accept Recall** | 96.2% | 98.1% |
| **Reject Recall** | 2.0% ❌ | 0.7% ❌ |
| **Accept Precision** | 49.8% | 50.3% |
| **Reject Precision** | 36.4% | 63.6% |

### Confusion Matrices

**Base Model:**
```
                 Predicted
              Accept  Reject
    Accept      975     35
    Reject      981     20   ← Only 20 TN!
```

**Instruct Model:**
```
                 Predicted
              Accept  Reject
    Accept      994      4
    Reject      981      7   ← Only 7 TN!
```

---

## Analysis

### 1. Both Models Are Broken

- **Base model** correctly predicts "Reject" only **2.0%** of the time
- **Instruct model** correctly predicts "Reject" only **0.7%** of the time
- Both models have **catastrophically low reject recall**

### 2. The Difference Is Negligible

- Base model: 97.3% accept → predicts "Accept" for 1,956/2,011 papers
- Instruct model: 99.4% accept → predicts "Accept" for 1,975/1,986 papers
- Difference: **2.1 percentage points** (within noise)

### 3. Agreement Analysis

Out of 1,978 aligned papers:
- **Both correct**: 958 (48.4%)
- **Both wrong**: 956 (48.3%)
- **Disagree**: 64 (3.2%)
  - Base correct, Instruct wrong: 23 (1.2%)
  - Base wrong, Instruct correct: 41 (2.1%)

**Agreement Rate**: 96.8% — both models make nearly identical predictions

---

## Root Cause Analysis

### Why Are Both Models So Optimistic?

1. **Prompt Bias**: The system prompt mentions "~30% acceptance rate" but doesn't explicitly instruct models to maintain this rate
2. **Base Model Limitation**: Base models (non-instruct) may not follow implicit constraints well
3. **Instruct Model Learned Bias**: Instruction-tuning may amplify helpfulness/positivity bias
4. **Task Framing**: Binary classification without calibration or examples

### Specific Issues

**Base Model**:
```python
BASE_COMPLETION_SYSTEM_PROMPT = (
    "You are an expert academic reviewer. "
    "Read the following paper and predict whether it was accepted or rejected at ICLR. "
    "ICLR generally has a ~30% acceptance rate. "
    "Your answer must start with: \\boxed{Accept} or \\boxed{Reject}"
)
```
- No explicit instruction to be critical
- No examples or calibration
- Completion-style prompting may not work well for base models

**Instruct Model**:
- Uses chat formatting with standard prompt
- Still shows extreme optimism bias
- Only predicts "Reject" for 11 papers out of 1,986

---

## Conclusions

### Hypothesis Status: ❌ REJECTED

The hypothesis that "the base model is less optimistic than the instruct model" is technically true (97.3% vs 99.4%), but **meaningless** because:

1. Both models are **catastrophically biased** toward "Accept"
2. The 2.1% difference is negligible compared to the 47-49% deviation from ground truth
3. Neither model is usable for paper review prediction

### Recommendations

1. **Add Calibration Examples**: Include few-shot examples with balanced Accept/Reject
2. **Strengthen Critical Instructions**: Explicitly instruct models to be skeptical
3. **Use Temperature Sampling**: Current temp=0.7 may not be sufficient
4. **Consider Logit Bias**: Manually bias toward "Reject" predictions
5. **Try Different Models**: Both Qwen models may have intrinsic optimism bias
6. **Add Reasoning Steps**: Chain-of-thought may improve critical thinking

### Next Steps

Consider testing:
- Critical role prompts (B2 experiment)
- Few-shot calibration examples
- Different model families (Llama, Mistral, etc.)
- Ensemble methods with explicit balancing

---

## File Manifest

- `analyze.py` - Analysis script
- `generate_dataset.py` - Dataset generation
- `run_inference.sbatch` - SLURM inference job
- `metrics/` - All plots and metrics
  - `acceptance_rate_base_vs_instruct.png`
  - `accuracy_base_vs_instruct.png`
  - `confusion_base.png`, `confusion_instruct.png`
  - `venn_agreement.png`, `disagreement_analysis.png`
  - `h1_metrics.json`
- `results/` - Raw predictions and extracted results
  - `predictions.jsonl` (87MB)
  - `results_single.jsonl` (6.5MB)
