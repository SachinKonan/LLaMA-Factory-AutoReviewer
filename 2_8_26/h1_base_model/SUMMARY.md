# H1 Experiment Summary: Base vs Instruct Model

**Hypothesis**: Base model (Qwen2.5-7B) is less optimistic than instruct model (Qwen2.5-7B-Instruct)

**Status**: ❌ **HYPOTHESIS REJECTED**

---

## What We Tested

- **Base Model**: `Qwen/Qwen2.5-7B` (non-instruct)
- **Instruct Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Dataset**: ICLR 2020-2025 test set (2,024 papers, 50% Accept / 50% Reject)
- **Modality**: Text-only (clean)

---

## Key Findings

### Both Models Are Extremely Optimistic

| Model | Acceptance Rate | Reject Recall | Accuracy |
|-------|----------------|---------------|----------|
| **Ground Truth** | 50.0% | - | - |
| **Base** | **97.3%** | 2.0% | 49.5% |
| **Instruct** | **99.4%** | 0.7% | 50.4% |

### The Problem

Both models predict "Accept" for almost all papers:
- **Base**: Predicts "Accept" for 1,956/2,024 papers (97.3%)
- **Instruct**: Predicts "Accept" for 1,975/2,024 papers (99.4%)

The base model is only marginally less optimistic (2.1 percentage points), but both are catastrophically biased.

### Confusion Matrices

**Base Model:**
```
                 Predicted
              Accept  Reject
GT Accept      975     35
GT Reject      981     20   ← Only 20 correct!
```

**Instruct Model:**
```
                 Predicted
              Accept  Reject
GT Accept      994      4
GT Reject      981      7   ← Only 7 correct!
```

Both models almost never predict "Reject" even when papers were actually rejected.

---

## Why Both Models Fail

1. **Weak Prompting**: Simply mentioning "30% acceptance rate" doesn't enforce it
2. **No Calibration**: No few-shot examples showing balanced predictions
3. **Base Model Limitations**: Non-instruct models may not follow implicit constraints
4. **Instruct Tuning Bias**: May amplify "helpful"/optimistic behavior

---

## Files

### Results
- Base model: `2_8_26/h1_base_model/results/`
- Instruct model: `2_8_26/b2_role_prompts/results/clean/standard/`

### Analysis
- Script: `analyze.py`
- Plots: `metrics/*.png`
- Metrics: `metrics/h1_metrics.json`
- Full Report: `metrics/evaluation_report.md`

---

## Next Steps

Consider testing:
1. **Critical role prompts** (B2 experiment) - explicit instructions to be skeptical
2. **Few-shot calibration** - include balanced Accept/Reject examples
3. **Different models** - Llama, Mistral, Claude, etc.
4. **Logit bias** - manually increase "Reject" token probability
5. **Temperature tuning** - test higher temperatures

---

## Conclusion

The hypothesis that base models are less optimistic is **technically true** (97.3% vs 99.4%), but **practically meaningless** because:

- Both models are ~50 percentage points above the ground truth (50%)
- The 2.1% difference is negligible compared to the overall bias
- Neither model is usable for paper review prediction in its current form

**Bottom line**: Instruction tuning is NOT the root cause of optimism bias. Both base and instruct models exhibit extreme optimism, suggesting the issue lies in prompting strategy, not model architecture.
