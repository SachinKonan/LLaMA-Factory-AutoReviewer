# Weighted Loss Experiment - Changelog

## Version 2 (19 experiments) - Current

### Changes from Version 1
- **Added baseline experiment at index 0**
  - Gamma: 1.0 (standard BCE, no weighting)
  - Dataset: Balanced (1/2)
  - Variant: baseline
  - Purpose: Provides reference point for all weighted loss comparisons

### Updated Index Mapping
- **Index 0**: NEW - Baseline (gamma=1.0)
- **Indices 1-6**: Balanced weighted (previously 0-5) - SHIFTED BY 1
- **Indices 7-10**: Proportion 1/3 (previously 6-9) - SHIFTED BY 1
- **Indices 11-14**: Proportion 1/4 (previously 10-13) - SHIFTED BY 1
- **Indices 15-18**: Proportion 1/8 (previously 14-17) - SHIFTED BY 1

### Resource Changes
- **Experiments**: 18 → 19 (+1)
- **Training GPU-hours**: 3,456 → 3,648 (+192)
- **Inference GPU-hours**: 72 → 76 (+4)
- **Estimated cost**: ~$7,056 → ~$7,448 (+$392)
- **Storage**: 272GB → 287GB (+15GB)

### Updated Files
1. `experiment_implementation.md` - All sections updated
2. `EXPERIMENT_SUMMARY.md` - Completely rewritten
3. `validate_indices.py` - Index decoding logic updated
4. SLURM scripts (in implementation doc) - Array range 0-17 → 0-18

### Key Benefits
- ✅ All weighted experiments can be compared against standard BCE baseline
- ✅ Enables measurement of weighted loss effect sizes
- ✅ Provides expected accuracy reference (~70%)
- ✅ Validates that baseline performs similarly to existing models

---

## Version 1 (18 experiments) - Superseded

### Original Design
- No baseline (no gamma=1.0)
- 18 experiments total:
  - Balanced (1/2): 6 experiments (gammas 2, 4, 8)
  - Imbalanced (1/3, 1/4, 1/8): 12 experiments (gammas 2, 4)
- Array indices: 0-17
- Cost: ~$7,056

### Rationale for Update
- User requested baseline for comparison purposes
- Standard practice in ML experiments to include unmodified baseline
- Minimal additional cost (~5% increase)
