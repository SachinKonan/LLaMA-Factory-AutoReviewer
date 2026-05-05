# Reverse-Order (Sequential-Cue) Ablation

## Purpose

Test whether the reviewer model is relying on **positional / sequential cues** in the input (e.g., "if the introduction is unusually short, I tend to predict accept"; "the model effectively just looks at the abstract because it's first") vs. genuinely consuming the full paper content.

The ablation: train and evaluate on inputs where the order of the paper's *content blocks* is reversed, while keeping the **labels unchanged**.

- **Text**: the human-turn prompt is split on `# HEADER` markers (the paper section breaks: `# INTRODUCTION`, `# METHOD`, `# RESULTS`, `# CONCLUSION`, etc.). Headers are then reordered last-to-first. The pre-header preamble (instructions / system text) and the assistant turn (the gold label) are untouched.
- **Vision**: the `images` list (page renderings) is reversed, so the model sees the last page first.

If the model relies heavily on positional cues (e.g., "I expect the abstract first"), the reversed condition should hurt accuracy meaningfully. If the model is content-driven, the drop should be small.

## What was tried

### Forward ("regular") runs
| Modality | Save dir | Train dataset |
|---|---|---|
| Text  | `saves/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/`        | `iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train` |
| Vision | `saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/`     | `iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train` |

Hyperparameters: 4 epochs, LR 1e-6 (cosine), bz=32 text / bz=16 vision, full FT, bf16, FSDP × 4 GPUs, one checkpoint per epoch.

### Reversed-input runs
Same hyperparameters, **same gold labels, same papers**, but every (train / validation / test) split's input has its section order (text) or page order (vision) reversed before training and evaluation.

| Modality | Save dir | Train / Eval / Test datasets |
|---|---|---|
| Text  | `saves/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text_reversed/`        | `..._text_labelfix_v7_filtered_{train,validation,test}_reversed` |
| Vision | `saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision_reversed/`     | `..._vision_labelfix_..._{train,validation,test}_reversed` |

Sbatch launchers (each sweeps two LRs, `1e-6` and `2e-6`; analyses below use `1e-6` for parity with the forward 1e-6 baseline):
- `sbatch/final_sweep_v7/datasweep_v3/optim_search_2026/text_bz32_reversed_ailab.sbatch`
- `sbatch/final_sweep_v7/datasweep_v3/optim_search_2026/vision_bz16_reversed_ailab.sbatch`

Reversed-dataset construction script: **`scripts/build_reversed_datasets.py`**
- Text: splits the human-turn value on `\n(?=# )`, reverses the section list, rejoins. Preamble before the first `# HEADER` is preserved. Assistant turn (label) is left alone.
- Vision: reverses `entry["images"]` so the page sequence is flipped. The prompt's `<image>` token order is positional, so this effectively shows the model the last page first.
- Output: `data/<base>_train_reversed/`, `..._validation_reversed/`, `..._test_reversed/`

## Where the results live

### Inference output (greedy decoding via vLLM)
| Run | Path |
|---|---|
| Text forward, regular test    | `results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/finetuned-ckpt-{661,1322,1983,2644}.jsonl` |
| Text forward, reversed valid. | `results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/validation-ckpt-1322-reversed.jsonl` (best ckpt only) |
| Text **reversed**, reversed test | `results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text_reversed/finetuned-ckpt-{661,1322,1983,2644}.jsonl` |
| Vision forward, regular test     | `results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-{1324,2648,3972,5296}.jsonl` |
| Vision forward, reversed valid.  | `results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/validation-ckpt-2648-reversed.jsonl` |
| Vision **reversed**, reversed test | `results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision_reversed/finetuned-ckpt-{1324,2648,3972,5296}.jsonl` |

### Training logs
- Forward: `saves/.../{bz32_lr1e-6_text,bz16_lr1e-6_vision}/trainer_log.jsonl`
- Reversed-input: `saves/.../{bz32_lr1e-6_text_reversed,bz16_lr1e-6_vision_reversed}/trainer_log.jsonl`

## Quick numbers (filtered to ICLR 2025+26 to match the standard panel reporting)

### Train+test in same condition (epoch-2 ckpt, test set, year ∈ {2025, 2026})
| Run | Test set | Accuracy | n |
|---|---|---|---|
| Forward text   | regular  | **66.0%** | 1667 |
| Reverse text   | reversed | **65.2%** | 1667 |
| Forward vision | regular  | **68.3%** | 1670 |
| Reverse vision | reversed | **66.8%** | 1670 |

Both modalities lose only ~1pt when both training and inference run on order-reversed inputs.

### Forward model on reversed input (cross-domain test, validation set, year ∈ {2025, 2026})
| Model | Eval split | Accuracy | n |
|---|---|---|---|
| Forward text   ckpt-1322 | regular validation  | 64.9% | 844 |
| Forward text   ckpt-1322 | **reversed validation** | **63.2%** | 844 |
| Forward vision ckpt-2648 | regular validation  | 65.4% | 845 |
| Forward vision ckpt-2648 | **reversed validation** | **65.9%** | 845 |

The **forward-trained model still scores ~63-66% on inputs whose section / page order has been scrambled** — despite never seeing reversed inputs in training. Drops:
- Text: -1.7pt (regular 64.9 → reversed 63.2)
- Vision: **+0.5pt** (regular 65.4 → reversed 65.9 — within noise; effectively unchanged)

For reference, full-test-set numbers (all years 2020–2026) are slightly higher: forward-text 67.0% / reverse-text 65.7% / forward-vision 68.4% / reverse-vision 67.2%.

## Interpretation

If the model relied heavily on positional cues, the forward-on-reversed numbers would crater. They don't — text loses ~1.7pt, vision is essentially unchanged. The model **mostly uses content semantics rather than positional / sequential layout**. Vision is even more order-invariant than text, which makes sense: pages are processed largely independently and aggregated, while text has stronger sequential semantic structure (preceding sections set context for later ones).

## What's not in this doc but would close the loop
- **Full test-set sweep across all 4 ckpts** for forward-on-reversed (we currently only have validation @ best ckpt).
- **Per-year breakdown on 2025+26** for the cross-domain comparison.
