# Weighted Loss Experiment - Quick Reference Summary

## Experiment Configuration (19 Total)

### Design Matrix

```
BASELINE:           Gamma [1.0] (standard BCE)                    = 1 experiment
Balanced (1/2):     Gammas [2, 4, 8] × Variants [accept, reject]  = 6 experiments
Imbalanced (1/3):   Gammas [2, 4]    × Variants [accept, reject]  = 4 experiments
Imbalanced (1/4):   Gammas [2, 4]    × Variants [accept, reject]  = 4 experiments
Imbalanced (1/8):   Gammas [2, 4]    × Variants [accept, reject]  = 4 experiments
                                                        TOTAL:      19 experiments
```

### SLURM Array Index Mapping

```
Index 0:     BASELINE (gamma=1.0, standard BCE)
Indices 1-6:   Balanced dataset (1/2) with weighted loss
Indices 7-10:  Proportion 1/3
Indices 11-14: Proportion 1/4
Indices 15-18: Proportion 1/8
```

## Key Features

**Index 0 is BASELINE** - Standard BCE (gamma=1.0) on balanced dataset
- Provides reference point for all comparisons
- Essential for measuring weighted loss effect sizes
- All other experiments can be compared against this baseline

## Complete Experiment List

| Index | Proportion | Variant | Gamma | Name                      | Notes |
|-------|------------|---------|-------|---------------------------|-------|
| **0** | **1/2**    | **baseline** | **1.0** | **baseline_gamma1.0_prop1_2** | **Standard BCE** |
| 1     | 1/2        | accept  | 2.0   | accept_gamma2.0_prop1_2   | |
| 2     | 1/2        | accept  | 4.0   | accept_gamma4.0_prop1_2   | |
| 3     | 1/2        | accept  | 8.0   | accept_gamma8.0_prop1_2   | |
| 4     | 1/2        | reject  | 2.0   | reject_gamma2.0_prop1_2   | |
| 5     | 1/2        | reject  | 4.0   | reject_gamma4.0_prop1_2   | |
| 6     | 1/2        | reject  | 8.0   | reject_gamma8.0_prop1_2   | |
| 7     | 1/3        | accept  | 2.0   | accept_gamma2.0_prop1_3   | |
| 8     | 1/3        | accept  | 4.0   | accept_gamma4.0_prop1_3   | |
| 9     | 1/3        | reject  | 2.0   | reject_gamma2.0_prop1_3   | |
| 10    | 1/3        | reject  | 4.0   | reject_gamma4.0_prop1_3   | |
| 11    | 1/4        | accept  | 2.0   | accept_gamma2.0_prop1_4   | |
| 12    | 1/4        | accept  | 4.0   | accept_gamma4.0_prop1_4   | |
| 13    | 1/4        | reject  | 2.0   | reject_gamma2.0_prop1_4   | |
| 14    | 1/4        | reject  | 4.0   | reject_gamma4.0_prop1_4   | |
| 15    | 1/8        | accept  | 2.0   | accept_gamma2.0_prop1_8   | |
| 16    | 1/8        | accept  | 4.0   | accept_gamma4.0_prop1_8   | |
| 17    | 1/8        | reject  | 2.0   | reject_gamma2.0_prop1_8   | |
| 18    | 1/8        | reject  | 4.0   | reject_gamma4.0_prop1_8   | |

## Resource Summary

| Resource | Amount |
|----------|--------|
| Training GPU-hours | 3,648 (19 jobs × 4 GPUs × 48h) |
| Inference GPU-hours | 76 (19 jobs × 1 GPU × 4h) |
| Estimated cost | ~$7,448 |
| Storage | 287GB |
| Wall time | ~2.4 days (parallelized) |

**For full implementation details, see `experiment_implementation.md`**
