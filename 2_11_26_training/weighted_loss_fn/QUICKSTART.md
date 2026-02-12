# Quick Start Guide - Weighted Loss Experiments

## TL;DR - Run Everything

```bash
cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer

# Step 1: Validate implementation
python 2_11_26_training/weighted_loss_fn/scripts/validate_implementation.py

# Step 2: Generate full datasets (~30 minutes)
python 2_11_26_training/weighted_loss_fn/scripts/stage1_generate_datasets.py

# Step 3: Submit all training jobs (~48 hours)
sbatch 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch

# Step 4: After training, run inference (~4 hours)
sbatch 2_11_26_training/weighted_loss_fn/scripts/stage3_run_inference.sbatch

# Step 5: Evaluate results
python 2_11_26_training/weighted_loss_fn/scripts/stage4_evaluate.py

# Step 6: Generate plots
python 2_11_26_training/weighted_loss_fn/scripts/stage5_visualize.py
```

## Step-by-Step Guide

### Step 1: Validate Implementation

Verify all files are in place:
```bash
python 2_11_26_training/weighted_loss_fn/scripts/validate_implementation.py
```

Expected output: "✓ ALL VALIDATION CHECKS PASSED"

### Step 2: Generate Training Datasets

**Debug mode (100 samples, fast test)**:
```bash
python 2_11_26_training/weighted_loss_fn/scripts/stage1_generate_datasets.py --debug
```

**Full mode (17,101 samples per dataset)**:
```bash
python 2_11_26_training/weighted_loss_fn/scripts/stage1_generate_datasets.py
```

Verify outputs:
```bash
ls -lh 2_11_26_training/weighted_loss_fn/data/*/data.json
# Should show 4 files, ~500MB each
```

### Step 3: Training

**Test with baseline only (index 0)**:
```bash
sbatch --array=0 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch
```

**Run balanced dataset experiments (indices 0-6)**:
```bash
sbatch --array=0-6 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch
```

**Run all 19 experiments**:
```bash
sbatch 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch
```

Monitor progress:
```bash
squeue -u $USER
tail -f 2_11_26_training/weighted_loss_fn/logs/train_*.out
```

Check saved models:
```bash
ls -lh saves/weighted_loss/
```

### Step 4: Inference

**After training completes**, run inference on all models:
```bash
sbatch 2_11_26_training/weighted_loss_fn/scripts/stage3_run_inference.sbatch
```

Or test single experiment:
```bash
sbatch --array=0 2_11_26_training/weighted_loss_fn/scripts/stage3_run_inference.sbatch
```

Monitor:
```bash
tail -f 2_11_26_training/weighted_loss_fn/logs/infer_*.out
```

Verify predictions:
```bash
ls -lh 2_11_26_training/weighted_loss_fn/results/*/predictions.jsonl
wc -l 2_11_26_training/weighted_loss_fn/results/*/predictions.jsonl
# Each should have 2024 lines
```

### Step 5: Evaluation

**Evaluate all experiments**:
```bash
python 2_11_26_training/weighted_loss_fn/scripts/stage4_evaluate.py
```

**Evaluate specific experiment**:
```bash
python 2_11_26_training/weighted_loss_fn/scripts/stage4_evaluate.py \
    --experiment baseline_gamma1.0_prop1_2
```

View results:
```bash
cat 2_11_26_training/weighted_loss_fn/metrics/all_metrics.json | jq
```

### Step 6: Visualization

Generate all plots:
```bash
python 2_11_26_training/weighted_loss_fn/scripts/stage5_visualize.py
```

View plots:
```bash
ls -lh 2_11_26_training/weighted_loss_fn/metrics/plots/
# acceptance_rate_vs_gamma.png
# accuracy_vs_gamma.png
# precision_recall.png
# heatmap_*.png
```

## Experiment Index Reference

| Index | Config | Description |
|-------|--------|-------------|
| 0 | baseline_gamma1.0_prop1_2 | Standard BCE (baseline) |
| 1-3 | accept_gamma[2,4,8]_prop1_2 | Weight accepts, balanced |
| 4-6 | reject_gamma[2,4,8]_prop1_2 | Weight rejects, balanced |
| 7-8 | accept_gamma[2,4]_prop1_3 | Weight accepts, 33% accept |
| 9-10 | reject_gamma[2,4]_prop1_3 | Weight rejects, 33% accept |
| 11-12 | accept_gamma[2,4]_prop1_4 | Weight accepts, 25% accept |
| 13-14 | reject_gamma[2,4]_prop1_4 | Weight rejects, 25% accept |
| 15-16 | accept_gamma[2,4]_prop1_8 | Weight accepts, 12.5% accept |
| 17-18 | reject_gamma[2,4]_prop1_8 | Weight rejects, 12.5% accept |

## Troubleshooting

### Training job fails
```bash
# Check error log
cat 2_11_26_training/weighted_loss_fn/logs/train_*_INDEX.err

# Verify config was generated
cat 2_11_26_training/weighted_loss_fn/configs/generated/*.yaml

# Test with smaller dataset (debug mode)
# Edit sbatch script to use debug datasets
```

### Inference fails
```bash
# Check if model checkpoint exists
ls saves/weighted_loss/EXPERIMENT_NAME/

# Check error log
cat 2_11_26_training/weighted_loss_fn/logs/infer_*_INDEX.err

# Verify vLLM environment
source .vllm/bin/activate
python -c "import vllm; print(vllm.__version__)"
```

### Evaluation produces unexpected results
```bash
# Check number of predictions
wc -l 2_11_26_training/weighted_loss_fn/results/*/predictions.jsonl

# Inspect sample predictions
head -1 2_11_26_training/weighted_loss_fn/results/EXPERIMENT_NAME/predictions.jsonl | jq

# Run evaluation with debug output
python 2_11_26_training/weighted_loss_fn/scripts/stage4_evaluate.py \
    --experiment EXPERIMENT_NAME
```

## Resource Usage

- **Training**: 3,648 GPU-hours (19 jobs × 4 GPUs × 48h)
- **Inference**: 76 GPU-hours (19 jobs × 1 GPU × 4h)
- **Storage**: ~287GB (19 × 15GB checkpoints + 2GB datasets)
- **Wall Time**: ~2.4 days (if all jobs run in parallel)

## Key Files

| File | Purpose |
|------|---------|
| `experiment_implementation.md` | Detailed implementation plan |
| `README.md` | Project overview and documentation |
| `IMPLEMENTATION_SUMMARY.md` | Implementation checklist |
| `QUICKSTART.md` | This file - quick reference |

## Support

For issues or questions:
1. Check `experiment_implementation.md` for detailed specifications
2. Run `validate_implementation.py` to verify setup
3. Review logs in `2_11_26_training/weighted_loss_fn/logs/`
4. Check SLURM status with `squeue -u $USER`

## Expected Results

**Success indicators**:
- All 19 training jobs complete
- All 19 inference outputs have 2,024 predictions
- Metrics show monotonic trends with gamma
- Predicted acceptance rates vary across conditions
- Visualizations show clear patterns

**Key plots to check**:
- `acceptance_rate_vs_gamma.png` - Should show increasing/decreasing trends
- `accuracy_vs_gamma.png` - Performance trade-offs
- `heatmap_predicted_accept_rate.png` - Full experimental matrix
