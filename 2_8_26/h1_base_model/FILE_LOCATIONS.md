# H1 Experiment File Locations

## Results Files

### Base Model (Qwen2.5-7B, non-instruct)
**Location**: `2_8_26/h1_base_model/results/`

- `predictions.jsonl` (87MB) - Raw vLLM predictions with prompts
- `results_single.jsonl` (6.5MB) - Extracted Accept/Reject predictions
- Generated: Feb 10 13:09

**Configuration**:
- Model: `Qwen/Qwen2.5-7B` (base, non-instruct)
- Template: `default` (completion-style, no chat format)
- Script: `run_inference.sbatch`

### Instruct Model (Qwen2.5-7B-Instruct)
**Location**: `2_8_26/b2_role_prompts/results/clean/standard/`

- `predictions.jsonl` (88MB) - Raw vLLM predictions with prompts  
- `results_single.jsonl` (6.1MB) - Extracted Accept/Reject predictions
- Generated: Feb 10 12:42

**Configuration**:
- Model: `Qwen/Qwen2.5-7B-Instruct` (instruction-tuned)
- Template: `qwen` (chat format)
- Script: `2_8_26/b2_role_prompts/run_role_inference.sbatch`
- Role: "standard" (no critical/enthusiastic modifier)

## Analysis Files

### Analysis Script
**Location**: `2_8_26/h1_base_model/analyze.py`

**Default paths** (lines 42, 254-259):
```python
DEFAULT_INSTRUCT = "2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl"

--base_results "2_8_26/h1_base_model/results/results_single.jsonl"
--instruct_results "2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl"
```

### Metrics & Plots
**Location**: `2_8_26/h1_base_model/metrics/`

- `h1_metrics.json` - All computed metrics
- `acceptance_rate_base_vs_instruct.png` - Acceptance rate comparison
- `accuracy_base_vs_instruct.png` - Accuracy comparison  
- `confusion_base.png` - Base model confusion matrix
- `confusion_instruct.png` - Instruct model confusion matrix
- `venn_agreement.png` - Agreement visualization
- `disagreement_analysis.png` - Disagreement breakdown
- `evaluation_report.md` - Full analysis report

## Quick Verification

```bash
# Check base model results
wc -l 2_8_26/h1_base_model/results/results_single.jsonl
# Output: 2024 results

# Check instruct model results  
wc -l 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl
# Output: 2024 results

# Count accepts from base model
grep -c '"prediction": "Accept"' 2_8_26/h1_base_model/results/results_single.jsonl
# Output: 1956 (96.6%)

# Count accepts from instruct model
grep -c '"prediction": "Accept"' 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl  
# Output: 1975 (97.6%)
```

## Why Two Separate Directories?

The H1 experiment tests **only the base model** to see if non-instruct models are less biased.

For comparison, the analysis script references the B2 experiment's **"standard" role**, which:
- Uses the **instruct model** (`Qwen2.5-7B-Instruct`)
- Has **no role modifier** (not critical, not enthusiastic)
- Uses the **same test set** (split7, balanced, clean, v7)

This allows fair comparison: base vs instruct on identical data.
