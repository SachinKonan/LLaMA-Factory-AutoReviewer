# Reverse-Label Ablation

## Purpose

Sanity-check whether the reviewer model is learning genuine paper-quality signal or just memorizing surface features / shortcut cues. If the model only picks up real signal, then **training on flipped labels** (Accept ↔ Reject swapped) should produce a model that:

- Achieves comparable accuracy when tested on a similarly **flipped** test set (proving the architecture/optimizer can fit flipped supervision).
- Performs **inversely** on the regular test set (predicts Reject for genuinely-strong papers and Accept for weak ones).

This rules out the trivial worry that the model is just learning "long paper = accept" or some other cheap proxy that's symmetric to label flipping.

## What was tried

### Forward ("regular") runs — already documented elsewhere, included here for comparison
| Modality | Save dir | Train dataset |
|---|---|---|
| Text  | `saves/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/`        | `iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train` |
| Vision | `saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/`     | `iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train` |

Hyperparameters: 4 epochs, learning rate 1e-6 (cosine), bz=32 (text) / bz=16 (vision), full FT, bf16, FSDP across 4 GPUs. Saves a checkpoint per epoch (text steps: 661 / 1322 / 1983 / 2644; vision steps: 1324 / 2648 / 3972 / 5296).

### Reversed runs

Same hyperparameters and same papers, but the dataset's `Accept` and `Reject` labels are **swapped before training**. Models thus learn to emit `\boxed{Reject}` on what humans called Accept and vice versa.

| Modality | Save dir | Train / Eval / Test datasets |
|---|---|---|
| Text  | `saves/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text_reversed/`        | `..._text_labelfix_v7_filtered_{train,validation,test}_reversed` |
| Vision | `saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision_reversed/`     | `..._vision_labelfix_..._{train,validation,test}_reversed` |

Sbatch launchers:
- `sbatch/final_sweep_v7/datasweep_v3/optim_search_2026/text_bz32_reversed_ailab.sbatch`
- `sbatch/final_sweep_v7/datasweep_v3/optim_search_2026/vision_bz16_reversed_ailab.sbatch`

Both arrays sweep two LRs (`1e-6`, `2e-6`); the analyses below use the `1e-6` variant for an apples-to-apples comparison with the forward 1e-6 baseline. `_reversed` suffix lives at the dataset level, not the model level — i.e., the reversed datasets were pre-built (see `scripts/build_reversed_datasets.py`) by swapping the answer field of every entry in the matching forward dataset.

Reversed-dataset construction script: `scripts/build_reversed_datasets.py`
- Reads `data/<base>_train/data.json` (and `_validation`, `_test`)
- For each sample, flips the gold label in `_metadata.answer` (Accept ↔ Reject) and rewrites the `\boxed{...}` token in the assistant turn of `conversations`
- Writes to `data/<base>_train_reversed/data.json` (and validation/test)

## Where the results live

### Inference output
| Run | Path |
|---|---|
| Text, forward, regular test    | `results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/finetuned-ckpt-{661,1322,1983,2644}.jsonl` |
| Text, forward, reversed test   | `results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/validation-ckpt-1322-reversed.jsonl` (validation set only — best ckpt) |
| Text, **reversed**, reversed test | `results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text_reversed/finetuned-ckpt-{661,1322,1983,2644}.jsonl` |
| Vision, forward, regular test     | `results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-{1324,2648,3972,5296}.jsonl` |
| Vision, forward, reversed test    | `results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/validation-ckpt-2648-reversed.jsonl` |
| Vision, **reversed**, reversed test | `results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision_reversed/finetuned-ckpt-{1324,2648,3972,5296}.jsonl` |

### Training logs
- Forward: `saves/.../{bz32_lr1e-6_text,bz16_lr1e-6_vision}/trainer_log.jsonl`
- Reversed: `saves/.../{bz32_lr1e-6_text_reversed,bz16_lr1e-6_vision_reversed}/trainer_log.jsonl`

## Quick numbers (epoch-2, full balanced labelfix test set)

| Run | Test set | Accuracy | n |
|---|---|---|---|
| Forward text   | regular  | **67.0%** | 2495 |
| Reverse text   | reversed | **65.7%** | 2495 |
| Forward vision | regular  | **68.4%** | 2498 |
| Reverse vision | reversed | **67.2%** | 2498 |

Both modalities reach essentially equivalent accuracy under flipped supervision — within ~1pt. **Conclusion: the model can fit either label polarity equally well, which is consistent with the model learning a real (signed) paper-quality signal rather than a label-symmetric shortcut.**

## What's *not* in this doc but would close the loop

- **Forward model on reversed test** (and **reverse model on forward test**) — the symmetric ablation. We have a single best-ckpt validation-set point (`validation-ckpt-1322-reversed.jsonl`, `validation-ckpt-2648-reversed.jsonl`); a full test-set sweep across all 4 ckpts hasn't been run. If desired, that's a 2-modality × 4-ckpt vLLM job sweep, ~15 min each on PLI.
- **Per-year breakdown** of the reverse run on the reversed test set (filtering to 2025+26 to match the standard panels).
