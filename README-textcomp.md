# Text Model: Accept/Reject Training Ratio Comparison

## Setup

All models use the same architecture (Qwen2.5-7B) and hyperparameters (`bz32`, `lr=1e-6`), varying only the accept:reject ratio in training data. Evaluated on the ICLR 2025+2026 test set with `pct_rating` buckets to measure performance across paper quality tiers.

- **50/50**: `saves/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/checkpoint-1322`
- **40/60**: `saves/final_sweep_v7_datasweepv3/optim_search_2026/ratio_sweep/bz32_lr1e-6_text_40_60/checkpoint-1983`  
- **30/70**: `saves/final_sweep_v7_datasweepv3/optim_search_2026/ratio_sweep/bz32_lr1e-6_text_30_70/checkpoint-1322`

Each is the checkpoint with the best combined 2025+2026 accuracy for its ratio.

Dataset: `iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered` (ratio sweep), `..._text_v7_filtered` (50/50 baseline).

Analysis script: `scripts/analyze_v7_ds3.py`

## Overall Results (2025+2026)

| Ratio | Accuracy | Accept Recall | Reject Recall |
|-------|----------|---------------|---------------|
| 50/50 | **66.0%** | 62.7% | **69.3%** |
| 40/60 | 65.2% | **75.5%** | 54.9% |
| 30/70 | 64.8% | 72.1% | 57.5% |

50/50 has the best overall accuracy and is reject-biased. Skewing training toward more reject examples counterintuitively increases accept recall and decreases reject recall.

## Per-Year Breakdown (best 2025+2026 checkpoint)

| Ratio | 2025 Acc | 2026 Acc | 2025 AccR | 2025 RejR | 2026 AccR | 2026 RejR |
|-------|----------|----------|-----------|-----------|-----------|-----------|
| 50/50 | **69.4%** | 63.7% | 63.7% | **75.1%** | 62.0% | **65.3%** |
| 40/60 | 67.8% | 63.5% | **74.0%** | 61.4% | **76.6%** | 50.4% |
| 30/70 | 66.6% | **63.6%** | 69.3% | 63.8% | 73.9% | 53.2% |

## Accuracy by Paper Quality (`pct_rating` buckets)

Confidence is raw `exp(token_logprobs[5])`, no Platt scaling.

### Low pct_rating [0, 0.4] — clear rejects (n=575)

| Ratio | Overall | 2025 | 2026 | Conf |
|-------|---------|------|------|------|
| 50/50 | **75.0%** | **79.9%** | **71.3%** | 71.1% |
| 40/60 | 65.7% | 68.4% | 63.7% | 74.1% |
| 30/70 | 66.4% | 68.4% | 65.0% | 71.4% |

### Mid pct_rating (0.4, 0.6] — borderline (n=311)

| Ratio | Overall | 2025 | 2026 | Conf |
|-------|---------|------|------|------|
| 50/50 | **57.9%** | **58.5%** | **57.5%** | 63.6% |
| 40/60 | 52.7% | 55.1% | 51.3% | 70.8% |
| 30/70 | 51.1% | 55.1% | 48.7% | 66.8% |

### High pct_rating (0.6, 1.0] — clear accepts (n=781)

| Ratio | Overall | 2025 | 2026 | Conf |
|-------|---------|------|------|------|
| 50/50 | 62.6% | 65.3% | 60.8% | 63.6% |
| 40/60 | **69.8%** | **72.0%** | **68.3%** | 72.8% |
| 30/70 | 69.0% | 69.4% | 68.7% | 68.4% |

## Key Findings

1. **50/50 is great at rejecting weak papers but over-rejects strong ones.** It gets 75% on easy rejects but only 62.6% on easy accepts, and it's appropriately uncertain (63-71% confidence).

2. **40/60 is the opposite.** It's better at recognizing strong papers (69.8% on easy accepts) but loses 10 points on easy rejects (65.7%). It's also the most overconfident — 70-74% confidence while being less accurate.

3. **30/70 splits the difference** without excelling at either end.

4. **All models are near coin-flip on borderline papers** (51-58%), which makes sense — those are genuinely hard calls.

5. **Skewing toward more reject training examples does not make the model better at rejecting.** It actually makes it worse at rejecting and better at accepting. The model appears to learn "what reject looks like" and becomes pickier about calling something a reject, which helps on strong papers but hurts on weak ones.

6. **2026 accuracy is consistently lower** across all buckets and ratios, but the gap narrows in the high bucket for skewed ratios (30/70 gets 68.7% on '26 vs 69.4% on '25).
