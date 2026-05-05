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

## Quick numbers (filtered to ICLR 2025+26)

### Train+test in same condition (epoch-2 ckpt, test set, year ∈ {2025, 2026})
| Run | Eval set | Accuracy | AccR | RejR | n |
|---|---|---|---|---|---|
| Forward text   | regular  | **66.0%** | 62.7% | 69.3% | 1667 |
| Reverse text   | reversed | **65.2%** | 59.8% | 70.6% | 1667 |
| Forward vision | regular  | **68.3%** | 69.2% | 67.4% | 1670 |
| Reverse vision | reversed | **66.8%** | 66.3% | 67.3% | 1670 |

Both modalities lose only ~1pt overall when both training and inference run on reversed inputs.

### **The interesting story: Forward model on reversed input (validation 25/26)**

This is where the recall pattern reveals what's actually happening.

| Model | Eval split | Accuracy | AccR | RejR | n |
|---|---|---|---|---|---|
| Forward text   ckpt-1322 | regular validation   | 64.9% | **62.2%** | 67.6% | 844 |
| Forward text   ckpt-1322 | **reversed validation**  | 63.2% | **46.1%** | **80.1%** | 844 |
| Forward vision ckpt-2648 | regular validation   | 65.4% | **67.1%** | 63.8% | 845 |
| Forward vision ckpt-2648 | **reversed validation**  | 65.9% | **54.7%** | **77.1%** | 845 |

**Overall accuracy barely changes** (text -1.7pt, vision +0.5pt) — but the per-class behavior shifts dramatically:

| Modality | ΔAccept Recall | ΔReject Recall |
|---|---|---|
| Text   | **−16.1 pt** | +12.5 pt |
| Vision | **−12.4 pt** | +13.3 pt |

When input order is scrambled, the forward model's confidence in saying "Accept" collapses while it becomes much more willing to say "Reject". On a balanced 50/50 eval set the gains and losses cancel out, so accuracy looks unaffected — but on a real (~30% accept) deployment distribution, the same shift would translate into a sharp drop in true-positive rate.

For reference, full-test-set numbers (all years 2020–2026) are slightly higher: forward-text 67.0% / reverse-text 65.7% / forward-vision 68.4% / reverse-vision 67.2%.

## Interpretation

If we just looked at overall accuracy, the conclusion would be "no effect — model is content-driven." But the recall breakdown tells a different story:

- **The model's positive (Accept) decisions specifically depend on coherent input order.** When sections (text) or pages (vision) come in the "wrong" order, Accept Recall drops by 12–16 pt.
- **The model's negative (Reject) decisions are much more robust to scrambling** — Reject Recall actually *rises* by 12–13 pt because the model now refuses to commit to Accept and falls back to Reject as the safer default.
- **On a balanced eval set, these changes cancel out**, masking the bias shift. On the real ICLR deployment distribution (~30% accept), the same model would deliver many more false rejects.

This implies the model has learned an asymmetric inference pattern: predicting Accept requires the model to track structural coherence (intro → method → results → conclusion in text; cover page → figures → references in vision). Predicting Reject can be triggered by lots of cheap signals that don't depend on order. So scrambling pushes the model toward its "default" reject prediction.

Vision and text both show this pattern with similar magnitudes, even though earlier we saw vision is more **content-tolerant** (overall accuracy unchanged, even slightly up on reversed val). The Accept-vs-Reject asymmetry isn't a content vs position story — it's about which class the model defaults to when its preferred signals are unreliable.

## What's not in this doc but would close the loop
- **Full test-set sweep across all 4 ckpts** for forward-on-reversed (we currently only have validation @ best ckpt).
- **Per-year breakdown on 2025+26** for the cross-domain comparison.
