# Ratio-Sweep Training-Curve Panels — Summary

This document explains the figures `tmp_latex_dir/figures/ratio_train_curves_text.{pdf,png}` and `..._vision.{pdf,png}`. The panels visualize how the choice of accept-vs-reject **training ratio** affects an LLM's behavior as a paper reviewer.

## 1. Data

### Source
- **Anonymized OpenReview submissions** for ICLR years 2017–2026.
- For each paper we have: full PDF text, page renderings (vision), per-reviewer ratings, the area-chair decision (Accept / Reject), and metadata (submission ID, year, conference, percentile rating).
- Two parallel modalities are derived from the same papers:
  - **text**: serialized paper content (no reviews, no references) prepared with the Qwen tokenizer (cutoff 24,480 tokens).
  - **vision**: page renderings as image sequences (with `image_min_pixels=784`, `image_max_pixels=1003520`), processed with Qwen2.5-VL.
- Models train to emit `\boxed{Accept}` or `\boxed{Reject}` at the end of an instruction-style prompt.

### Splits used in this work
We focus on `iclr_2020_2023_2025_2026_85_5_10_balanced_original_<text|vision>_labelfix_v7_filtered`:
- 25k papers total (12,442 Accept / 12,446 Reject) — naturally **balanced 50/50** by sampling.
- 85% train / 5% validation / 10% test split, stratified by (year, label).
- "labelfix" = a corrected version of decision labels after fixing a labeling bug.
- "filtered" = papers truncated to 24,480 tokens (vision: also `_filtered24480` to ensure they fit Qwen2.5-VL's context).

### Year coverage in the test set
- Years 2020–2026 are all represented in train/val/test.
- **All evaluations in these panels filter test samples to year ∈ {2025, 2026}** — about 1,667 papers for text, 1,670 for vision. We chose ICLR 2025/26 because (a) recent papers are the most relevant proxy for "future ICLR submissions," and (b) the 2017–2019 era used a different review process and is left out of this analysis.

### Ratio-modified ICLR test sets
For the cross-evaluation matrix we additionally use:
- `iclr_2020_2023_2025_2026_40_60_original_<text|vision>_v7_filtered_test`: same papers as above, but accepts subsampled so the test set has 40% Accept / 60% Reject.
- `iclr_2020_2023_2025_2026_30_70_original_<text|vision>_v7_filtered_test`: 30% / 70%.

These let us measure how each model behaves when the deployment-time class distribution differs from training.

### Cross-conference (out-of-distribution) test sets
- **NeurIPS 2021–2025 eval500**: 500-paper stratified sample. Heavily accept-skewed because OpenReview only publishes accepted papers + a handful of rejects. Distribution: 471 Accept / 29 Reject (94% accept). Per-year: 64/3 in 2021, 70/6 in 2022, 86/7 in 2023, 110/5 in 2024, 141/8 in 2025.
- **ICML+COLM 2024–2025 eval350**: 344 Accept / 6 Reject (98% accept). All 6 rejects are from ICML 2025; ICML 2024 and COLM 2024+2025 have zero rejects in the eval split. Per-conference: 140/0 (ICML 2024), 168/6 (ICML 2025), 13/0 (COLM 2024), 23/0 (COLM 2025).

These exist to measure **out-of-distribution generalization**: take an ICLR-trained model and ask "does it still rank papers correctly on a different conference?"

## 2. Training experiments

We trained the same Qwen base (Qwen2.5-7B-Instruct for text, Qwen2.5-VL-7B-Instruct for vision) on three different accept/reject training ratios:

| Train ratio | Text dataset | Vision dataset |
|---|---|---|
| 50/50 (balanced baseline) | `..._labelfix_v7_filtered_train` | same with `_filtered24480` |
| 40/60 | `iclr_..._40_60_original_text_v7_filtered_train` | `..._40_60_original_vision_v7_filtered_train` |
| 30/70 | `..._30_70_original_text_v7_filtered_train` | `..._30_70_original_vision_v7_filtered_train` |

The 40/60 and 30/70 sets are constructed by **subsampling accepts** (per year) from the 50/50 set — same papers, fewer accepts, same rejects. So the rejects are identical across the three datasets; the accepts are a strict subset for the more reject-heavy variants.

Hyperparameters: SGD-equivalent batch 32 (text) / 16 (vision), learning rate 1e-6, cosine LR schedule with `min_lr_rate=0.001`, 4 epochs, full fine-tuning (no LoRA), bf16, FSDP across 4 GPUs. Save one checkpoint per epoch.

Each model thus produces 3 checkpoints (40/60 and 30/70 training, 4 epochs) or 4 checkpoints (50/50 baseline, ran for 4 epochs as well — actually all are 4 epochs but the last 50/50 ckpt happens at a slightly higher step due to a slightly different total step count; the labeling is by step number, not epoch number).

Step → epoch mapping at evaluated checkpoints:
- Text 50/50 ckpts: 661, 1322, 1983, 2644 (epochs ≈1, 2, 3, 4)
- Text 40/60 and 30/70 ckpts: 661, 1322, 1983 (epochs ≈1, 2, 3) — early time-out left epoch 4 unsaved
- Vision 50/50 ckpts: 1324, 2648, 3972, 5296
- Vision 40/60 and 30/70 ckpts: 1321, 2642, 3963, 5284

## 3. The figure layout

Each modality figure is a **3 rows × 6 columns** grid. Rows correspond to the three training ratios (50/50 top, then 40/60, then 30/70). Columns:

### Col 1 — Train Loss (log scale)
Loss at every 10 training steps from `trainer_log.jsonl`, with a 10-step moving-average smoothing overlay. X-axis = epoch (so all three rows share a comparable horizontal axis). Drops from ~4.0 to ~0.04 over training.

### Col 2 — Test Accuracy (ICLR 2025+26)
For every saved checkpoint (epoch 1, 2, 3, [4]), we run greedy-decoding inference on three different ICLR 2025+26 test slices:
- **Test 50/50** (blue) — the original balanced labelfix test set.
- **Test 40/60** (green) — the accept-subsampled 40/60 variant.
- **Test 30/70** (purple) — the 30/70 variant.

Y-axis is accuracy on the 25/26 subset of each test set. Star markers denote single-point cross-evals: the 50/50-trained model was only evaluated on 40/60 and 30/70 test sets at its **best 25+26 checkpoint** (text=1322, vision=2648). The ratio-trained models were evaluated on all three test sets at every checkpoint.

### Col 3 — Accept/Reject Recall mini-stack (ICLR 2025+26)
Each row's col-3 cell contains **three vertically-stacked sub-panels** (one per test ratio, labeled in-panel). Each sub-panel plots two lines vs epoch:
- Accept Recall (green ■): TP / (TP + FN), the fraction of true accepts the model labels Accept.
- Reject Recall (red ▲): TN / (TN + FP), the fraction of true rejects the model labels Reject.

The **gap between the two lines** within a sub-panel is the model's *direction of bias*. A balanced model has the lines on top of each other; an accept-biased model has Accept Recall > Reject Recall; a reject-biased model the other way around. The shape of how the gap evolves over training tells you whether bias is increasing, decreasing, or stable.

### Col 4 — ROC-AUC (ICLR 2025+26)
Per checkpoint, AUC of `lp_accept - lp_reject` (the model's log-odds for Accept) versus the ground-truth label, restricted to 2025+26. Three lines, one per test ratio (same colors as col 2). Star markers for the 50/50 best-only points.

The score is the difference of two log-probabilities at the decision step where the model emitted Accept or Reject — so it's a true continuous signal even though the deployed model is greedy-decoded.

### Cols 5–6 — Cross-Conference (epoch-2 only)
Single-checkpoint bar charts (epoch-2 ckpt: text=1322, vision=2648/2642). Three bars per cell:
- ICLR 25/26 (blue) — model evaluated on its **own training ratio** test set, filtered to 2025+26.
- NeurIPS eval500 (orange) — full 500 papers, no year filter.
- ICML+COLM eval350 (purple) — full 350 papers, no year filter.

**Col 5: ROC-AUC** — bars drawn from a baseline of 0.5 (random-classifier line shown dashed). Y-axis fixed [0.45, 1.0].

**Col 6: Accept Recall (%)** — bars drawn from 0. Y-axis fixed [0, 105]. This metric is the most informative on the OOD splits because both NeurIPS and ICML/COLM are accept-dominated, so reject recall on those splits is statistically unstable (only 29 and 6 rejects respectively).

## 4. Metric semantics

| Metric | Tells you | Best when |
|---|---|---|
| Accuracy | How often greedy decoding picks the correct class. Threshold is implicit (the argmax). | You're deploying with greedy decoding and the test distribution matches the deployment distribution. |
| Accept Recall | TP/(TP+FN). Of all true accepts, how many did we label Accept. | Sensitive when missing a paper is costly. |
| Reject Recall | TN/(TN+FP). Of all true rejects, how many did we label Reject. | Sensitive when accepting a bad paper is costly. |
| ROC-AUC | Probability that a random Accept gets a higher Accept-score than a random Reject. Threshold-independent. | You care about ranking quality and might re-tune the threshold at deployment time. |

The Accept Recall vs Reject Recall pair is the **bias diagnostic**: their gap reveals which direction the model leans, regardless of accuracy. AUC is the **deployment-flexible quality metric**: a high-AUC + bias-skewed model can be recalibrated at test time, whereas a low-AUC model can't be saved by recalibration.

## 5. Key takeaways from the panels

1. **Training-data ratio inverts the deployed bias direction**: 50/50 and 40/60 trained models are reject-biased (Reject Recall > Accept Recall); the 30/70 model overshoots the bias correction and ends up *accept*-biased. More reject training data does not produce more reject-biased models — the model becomes pickier about what counts as reject.

2. **Test accuracy peaks early**, usually epoch 2, then plateaus or slightly drops. AUC follows the same pattern. The recall *gap* keeps shrinking past the accuracy peak — the model becomes more balanced as training proceeds, even though raw accuracy isn't improving.

3. **Cross-distribution generalization (col 6)**: the 50/50-trained text model maintains ~57% Accept Recall on both NIPS and ICML/COLM. The 40/60 and 30/70 text models collapse to **0% Accept Recall on ICML+COLM** — they reject every paper in a 98%-accept distribution. They have learned ICLR's accept/reject decision boundary but it doesn't transfer.

4. **Vision is more robust to OOD shift than text**: vision 40/60 actually gets the highest NIPS AUC (0.816) and highest NIPS Accept Recall (82%). Text generalizes worse across conferences.

5. **There is no universally best training ratio** — the answer depends on the deployment scenario:
   - Building a deployable ICLR triage tool: train 30/70 (matches the natural ~30% acceptance rate) and you'll see the best in-distribution accuracy on the realistic distribution.
   - Building a flexible reviewer that might be deployed at any threshold: train 50/50 (balanced gradient signal) — best AUC and most robust under distribution shift.
   - For vision, the picture is shifted: 40/60 vision actually has the best cross-distribution AUC, suggesting the visual signal generalizes better when the ratio is moderately accept-skewed.

## 6. File pointers

- Plot script: `scripts/tmp_latex_dir/generate_ratio_train_curves.py`
- Inference results (in-distribution): `results/final_sweep_v7_datasweepv3/optim_search_2026/{,ratio_sweep/}<short_name>/<test_ratio>/finetuned-ckpt-<step>.jsonl`
- Inference results (cross-conference): `results/cross_conference/<short_name>/<conference>_eval/finetuned-ckpt-<step>.jsonl`
- Train logs: `saves/.../trainer_log.jsonl`
- Cross-conference inference sbatch: `sbatch/inference/cross_conference.sbatch`
- Each output JSONL has fields: `prompt`, `predict`, `label`, `token_logprobs` (per generated token), `logprob_accept` (logprob of Accept token at every step), `logprob_reject` (logprob of Reject token). The decision step is index 5 in the generated sequence (the slot inside `\boxed{...}`).
