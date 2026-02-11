# Checklist Optimization for Review Evaluation (C2)

**Experiment ID**: C2
**Date**: 2026-02-11
**Status**: Implementation complete, ready for execution

## Overview

This experiment develops a binary checklist system for evaluating **real ICLR reviews** (from OpenReview). The goal is to identify the minimal set of yes/no questions that maximally correlates with ground truth accept/reject decisions.

**Key Hypothesis**: A small, optimized set of binary checklist questions applied to real reviews can predict paper outcomes, providing interpretable evaluation criteria.

**Data Source**: Real ICLR reviews from HuggingFace Arrow dataset (2,024 test papers, 4,682 individual reviews).

**Model**: `Qwen/Qwen3-30B-A3B-Thinking-2507` via vLLM (local inference, 2x L40 GPUs).

## Success Criteria

- **Primary**: Checklist-based decisions achieve accuracy >= raw LLM decisions from B2 standard/clean (baseline: ~70%)
- **Secondary**: Final checklist has <=15 questions (interpretable, not overwhelming)
- **Tertiary**: Point-biserial correlation between individual questions and outcomes > 0.15

## Pipeline Overview

```
Stage 1: Generate Candidates (38 ICLR seed + LLM expansion to ~100 questions)
    |
Stage 2: Filter by Enforceability (~60-80 questions)
    |
Stage 3: Semantic Deduplication (~40-60 questions)
    |
Stage 4: Beam Search Optimization (10-15 questions)
    |
Stage 5: Evaluation & Analysis
```

## Files

### Core Scripts
- `utils.py` - Shared utilities (vLLM inference, HF dataset loading, metrics)
- `stage1_generate_candidates.py` - ICLR-criteria seed questions + LLM expansion
- `stage2_filter_questions.py` - Filter by enforceability and answerability
- `stage3_deduplicate.py` - Semantic clustering and deduplication
- `stage4_beam_search.py` - Beam search optimization on real reviews
- `stage5_evaluate.py` - Comprehensive evaluation
- `analyze.py` - Generate plots and visualizations
- `run_pipeline.sbatch` - SLURM job array (2x L40 GPUs)

## Quick Start

### Test Individual Stages (Debug Mode)

```bash
# Stage 1: Generate questions from ICLR criteria + LLM expansion
python 2_8_26/checklist_optimization/stage1_generate_candidates.py \
    --n_questions 50 --sample_size 10 --debug

# Stage 2: Filter with relaxed thresholds
python 2_8_26/checklist_optimization/stage2_filter_questions.py \
    --input_questions 2_8_26/checklist_optimization/data/candidate_questions.jsonl \
    --output 2_8_26/checklist_optimization/data/filtered_questions.jsonl \
    --consistency_threshold 0.60 --n_test_reviews 2 --n_repeats 2 --debug

# Stage 3: Deduplicate (skip LLM selection for speed)
python 2_8_26/checklist_optimization/stage3_deduplicate.py \
    --input 2_8_26/checklist_optimization/data/filtered_questions.jsonl \
    --output 2_8_26/checklist_optimization/data/deduplicated_questions.jsonl \
    --similarity_threshold 0.7 --skip_llm_selection --debug

# Stage 4: Beam search on subset of reviews
python 2_8_26/checklist_optimization/stage4_beam_search.py \
    --input_questions 2_8_26/checklist_optimization/data/deduplicated_questions.jsonl \
    --output 2_8_26/checklist_optimization/data/optimal_checklist.json \
    --beam_width 3 --max_questions 5 --n_reviews 100 --debug

# Stage 5: Evaluate
python 2_8_26/checklist_optimization/stage5_evaluate.py \
    --checklist 2_8_26/checklist_optimization/data/optimal_checklist.json \
    --evaluations 2_8_26/checklist_optimization/data/paper_evaluations.jsonl \
    --output_metrics 2_8_26/checklist_optimization/metrics/checklist_metrics.json \
    --output_predictions 2_8_26/checklist_optimization/results/final_predictions.jsonl --debug

# Generate plots
python 2_8_26/checklist_optimization/analyze.py
```

### Run Full Pipeline on SLURM

```bash
# Submit stages sequentially with dependencies
STAGE1=$(sbatch --parsable --array=0 2_8_26/checklist_optimization/run_pipeline.sbatch)
STAGE2=$(sbatch --parsable --dependency=afterok:$STAGE1 --array=1 2_8_26/checklist_optimization/run_pipeline.sbatch)
STAGE3=$(sbatch --parsable --dependency=afterok:$STAGE2 --array=2 2_8_26/checklist_optimization/run_pipeline.sbatch)
STAGE4=$(sbatch --parsable --dependency=afterok:$STAGE3 --array=3 2_8_26/checklist_optimization/run_pipeline.sbatch)
STAGE5=$(sbatch --parsable --dependency=afterok:$STAGE4 --array=4 2_8_26/checklist_optimization/run_pipeline.sbatch)
```

## Expected Outputs

### Data Files
- `data/candidate_questions.jsonl` - ~100 questions (38 ICLR seed + LLM generated)
- `data/filtered_questions.jsonl` - ~60-80 enforceable questions
- `data/deduplicated_questions.jsonl` - ~40-60 unique questions
- `data/checklist_evaluations.jsonl` - Review-level answers (4682 reviews)
- `data/paper_evaluations.jsonl` - Paper-level aggregated answers (2024 papers)
- `data/optimal_checklist.json` - Final 10-15 questions

### Plots
- `metrics/optimization_curve.png` - Beam search progress
- `metrics/question_importance.png` - Top questions by correlation
- `metrics/correlation_heatmap.png` - Question x outcome correlation
- `metrics/comparison_plot.png` - Checklist vs LLM baseline

## Expected Runtime

**Model**: `Qwen/Qwen3-30B-A3B-Thinking-2507` on 2x L40 (48GB each)

| Stage | Runtime (Full) | Runtime (Debug) | Notes |
|-------|----------------|-----------------|-------|
| Stage 1 | ~15 min | ~5 min | 38 ICLR seed + LLM expansion |
| Stage 2 | ~1 hour | ~10 min | 100 questions x 5 reviews x 3 repeats (batched) |
| Stage 3 | ~30 min | ~5 min | Embedding + LLM cluster selection |
| Stage 4 | ~6 hours | ~30 min | ~50 questions x 4682 reviews (batched via vLLM) |
| Stage 5 | ~10 min | ~2 min | Pure computation |
| **Total** | **~8 hours** | **~1 hour** | All local inference, no API costs |

## Dependencies

All dependencies should be available in `.venv_vllm_inf`:
- `vllm` - Local LLM inference
- `transformers` - Model loading
- `sentence-transformers` - Embedding model
- `scikit-learn` - Clustering
- `scipy` - Statistical tests
- `datasets` - HuggingFace Arrow dataset loading
- `numpy`, `matplotlib`, `seaborn` - Data processing and visualization

## Troubleshooting

### Issue: Stage 2 filters out too many questions
**Solution**: Lower `--consistency_threshold` to 0.70 or reduce `--n_repeats` to 2

### Issue: Stage 3 produces too few clusters
**Solution**: Increase `--similarity_threshold` to 0.8 (stricter clustering)

### Issue: Stage 4 beam search doesn't converge
**Solution**: Increase `--beam_width` to 15 or adjust composite score weights in code

### Issue: Model doesn't fit on 2x L40s
**Solution**: Increase `--tensor_parallel_size` to 4, or reduce max_model_len in utils.py

### Issue: Checklist accuracy below baseline
**Solution**: Stage 5 optimizes threshold automatically; check per-question correlations

## References

- EMNLP 2025 Industry Track #104: Checklist-based evaluation framework
- Q1 experiment: Reviewer archetypes clustering (HF dataset loading patterns)
- B2 experiment: Role prompts and LLM-generated prediction baseline
- vLLM inference patterns: `inference_scaling/scripts/vllm_infer_ensemble.py`
