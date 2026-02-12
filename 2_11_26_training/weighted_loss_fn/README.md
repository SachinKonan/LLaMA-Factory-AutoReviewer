# Weighted Loss Function Experiments

This directory contains the implementation of weighted loss function experiments to investigate how loss weighting and dataset composition affect model acceptance rate predictions for ICLR paper reviews.

## Quick Start

```bash
# 1. Generate datasets (run once)
python scripts/stage1_generate_datasets.py

# 2. Submit training jobs (19 experiments)
sbatch scripts/stage2_train_models.sbatch

# 3. Run inference (after training completes)
sbatch scripts/stage3_run_inference.sbatch

# 4. Evaluate results
python scripts/stage4_evaluate.py

# 5. Generate visualizations
python scripts/stage5_visualize.py
```

## Experiment Overview

**Total Experiments**: 19
- 1 baseline (gamma=1.0, standard BCE)
- 6 balanced dataset experiments (gammas: 2, 4, 8)
- 12 imbalanced dataset experiments (gammas: 2, 4)

**Experiment Dimensions**:
- **Loss weighting gamma**: 1.0 (baseline), 2.0, 4.0, 8.0
- **Loss variants**: accept (weight false negatives), reject (weight false positives)
- **Dataset proportions**: 1:2 (50%), 1:3 (33%), 1:4 (25%), 1:8 (12.5%)

## Directory Structure

```
weighted_loss_fn/
├── README.md                           # This file
├── experiment_implementation.md        # Detailed implementation plan
├── data/                               # Generated training datasets
│   ├── iclr_weighted_loss_train_1_2/   # 50:50 (balanced)
│   ├── iclr_weighted_loss_train_1_3/   # 33:67
│   ├── iclr_weighted_loss_train_1_4/   # 25:75
│   └── iclr_weighted_loss_train_1_8/   # 12.5:87.5
├── configs/
│   ├── base_config.yaml                # Reference config
│   └── generated/                      # Auto-generated per experiment
├── scripts/
│   ├── stage1_generate_datasets.py     # Create training datasets
│   ├── stage2_train_models.sbatch      # SLURM training array (19 jobs)
│   ├── stage3_run_inference.sbatch     # SLURM inference array (19 jobs)
│   ├── stage4_evaluate.py              # Compute metrics
│   └── stage5_visualize.py             # Generate plots
├── logs/                               # SLURM output logs
├── results/                            # Inference predictions
└── metrics/                            # Evaluation outputs
    ├── all_metrics.json
    └── plots/
```

## Experiment Index Mapping

| Index | Proportion | Variant | Gamma | Experiment Name |
|-------|------------|---------|-------|-----------------|
| 0     | 1/2        | baseline| 1.0   | baseline_gamma1.0_prop1_2 |
| 1     | 1/2        | accept  | 2.0   | accept_gamma2.0_prop1_2 |
| 2     | 1/2        | accept  | 4.0   | accept_gamma4.0_prop1_2 |
| 3     | 1/2        | accept  | 8.0   | accept_gamma8.0_prop1_2 |
| 4     | 1/2        | reject  | 2.0   | reject_gamma2.0_prop1_2 |
| 5     | 1/2        | reject  | 4.0   | reject_gamma4.0_prop1_2 |
| 6     | 1/2        | reject  | 8.0   | reject_gamma8.0_prop1_2 |
| 7     | 1/3        | accept  | 2.0   | accept_gamma2.0_prop1_3 |
| 8     | 1/3        | accept  | 4.0   | accept_gamma4.0_prop1_3 |
| 9     | 1/3        | reject  | 2.0   | reject_gamma2.0_prop1_3 |
| 10    | 1/3        | reject  | 4.0   | reject_gamma4.0_prop1_3 |
| 11    | 1/4        | accept  | 2.0   | accept_gamma2.0_prop1_4 |
| 12    | 1/4        | accept  | 4.0   | accept_gamma4.0_prop1_4 |
| 13    | 1/4        | reject  | 2.0   | reject_gamma2.0_prop1_4 |
| 14    | 1/4        | reject  | 4.0   | reject_gamma4.0_prop1_4 |
| 15    | 1/8        | accept  | 2.0   | accept_gamma2.0_prop1_8 |
| 16    | 1/8        | accept  | 4.0   | accept_gamma4.0_prop1_8 |
| 17    | 1/8        | reject  | 2.0   | reject_gamma2.0_prop1_8 |
| 18    | 1/8        | reject  | 4.0   | reject_gamma4.0_prop1_8 |

## Code Modifications

The following LLaMA Factory source files were modified:

1. **src/llamafactory/train/trainer_utils.py**:
   - Added `weighted_bce_loss_accept()` - weights false negatives
   - Added `weighted_bce_loss_reject()` - weights false positives
   - Added helper functions `_weighted_cross_entropy_accept()` and `_weighted_cross_entropy_reject()`

2. **src/llamafactory/hparams/finetuning_args.py**:
   - Added `use_weighted_loss` field
   - Added `weighted_loss_variant` field (accept/reject)
   - Added `weighted_loss_gamma` field (weighting factor)

3. **src/llamafactory/train/sft/trainer.py**:
   - Added weighted loss function selection in `CustomSeq2SeqTrainer.__init__()`

4. **data/dataset_info.json**:
   - Registered 4 new training datasets

## Usage Examples

### Test Dataset Generation (Debug Mode)
```bash
python scripts/stage1_generate_datasets.py --debug
```

### Generate Full Datasets
```bash
python scripts/stage1_generate_datasets.py
```

### Submit Specific Experiments
```bash
# Baseline only
sbatch --array=0 scripts/stage2_train_models.sbatch

# Balanced dataset (indices 0-6)
sbatch --array=0-6 scripts/stage2_train_models.sbatch

# One imbalanced dataset (1/3 proportion)
sbatch --array=7-10 scripts/stage2_train_models.sbatch

# All experiments
sbatch scripts/stage2_train_models.sbatch
```

### Monitor Training
```bash
squeue -u $USER
tail -f logs/train_*.out
```

### Evaluate Single Experiment
```bash
python scripts/stage4_evaluate.py --experiment baseline_gamma1.0_prop1_2
```

## Expected Outcomes

**Hypothesis 1: Loss Weighting Effect**
- Accept variant → higher predicted acceptance rates as gamma increases
- Reject variant → lower predicted acceptance rates as gamma increases

**Hypothesis 2: Dataset Proportion Effect**
- Reject-heavy datasets → lower predicted acceptance rates
- Effect should be observable across both loss variants

**Hypothesis 3: Interaction Effects**
- Balanced + gamma=8 → maximum effect size
- Imbalanced + moderate gamma → constrained by data distribution

## Resource Estimates

- **Training**: 19 jobs × 4 GPUs × 48 hours = 3,648 GPU-hours (~$7,296)
- **Inference**: 19 jobs × 1 GPU × 4 hours = 76 GPU-hours (~$152)
- **Storage**: ~287GB (19 model checkpoints + datasets)
- **Wall Time**: ~2.4 days (parallelized)

## Output Metrics

For each experiment:
- Accuracy
- Precision (Accept class)
- Recall (Accept class)
- F1 score
- Predicted acceptance rate
- Ground truth acceptance rate

## Visualizations

Generated plots:
- `acceptance_rate_vs_gamma.png` - Main result
- `accuracy_vs_gamma.png` - Performance trade-off
- `precision_recall.png` - Classification quality
- `heatmap_*.png` - Metric heatmaps across dimensions

## References

- Detailed plan: `experiment_implementation.md`
- Base model: Qwen/Qwen2.5-7B-Instruct
- Test set: iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test (2,024 samples)
- Original training set: 17,101 samples (50:50 balanced)
