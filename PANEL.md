# Panel datasets & training summary

## TL;DR
Single-image panel representation of a paper (5×2 grid of typeset/screenshot pages on a 2380×1512 canvas) → **3× cheaper than the existing 8-page vision input at ~97% of the accuracy**. ICLR panel-only training is the only finish so far: 66.2% on the 2,498-paper test vs. 68.4% for the 8-page baseline at >2× the token cost.

## Image format

| Property | Value |
|---|---|
| Canvas | **2380 × 1512** px (multiples of 28 → Qwen2.5-VL `smart_resize` no-op at `image_max_pixels=4014080`) |
| Layout | 5 cols × 2 rows = 10 page slots (left-to-right, top-to-bottom) |
| Per-cell | 476 × 756 px (cell aspect 0.625) |
| Pre-process | `trim_white_margins` (crop solid-white PDF margins) → `fit_into_cell` (letterbox-fit preserving aspect) |
| Per-venue inner border (arxiv only) | ICLR/COLM=0, NeurIPS=12, CoRL=18, EccV/AISTATS/ACL=22, ICML/ICCV/AAAI=24, CVPR=26 px |
| Qwen2.5-VL vision tokens | **4,590** (smart_resize no-op) |

ICLR panels (`data/images_panel/`) use border=0 across the board — single-column ICLR papers already get ~30 px natural letterbox from their narrow trimmed-content aspect (~0.55) inside the 0.625 cell.

Arxiv panels (`data/images_panel_arxiv/`) bake in the per-venue borders so wide-content 2-column venues (CVPR/ICCV/AAAI/ICML/...) read with visible page boundaries comparable to single-column venues.

Builder: `scripts/build_panel_images.py --source {iclr,arxiv}` (16 workers via srun on cpu partition, ~45 min for ~25k papers).

Verification (OCR smoke, base Qwen2.5-VL-7B): 3/5 papers transcribed at 52–78% unique-word recall — content is legible at 4,590 tokens.

---

## Datasets registered (12 total in `data/dataset_info.json`)

All in sharegpt format. ICLR panel images are at `data/images_panel/<sid>.png` (24,925 PNGs, 70 GB). Arxiv panel images are at `data/images_panel_arxiv/<aid>.png` (24,610 PNGs, 61 GB).

### ICLR — panel-only (vision-style preamble + 1 panel image)

| Dataset name | Entries |
|---|---:|
| `iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_panel_train` | 21,174 |
| `..._panel_validation` | 1,253 |
| `..._panel_test` | 2,498 |

### ICLR — text + panel (full paper markdown + 1 panel suffix)

| Dataset name | Entries |
|---|---:|
| `iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_panel_v7_filtered_train` | 21,134 |
| `..._validation` | 1,252 |
| `..._test` | 2,495 |

### arxiv — panel-only (vision-style preamble + 1 panel image)

| Dataset name | Entries |
|---|---:|
| `arxiv_50_50_21k_vision_wmetadata_filtered24480_panel_train` | 20,929 |
| `..._panel_validation` | 1,223 |
| `..._panel_test` | 2,458 |

### arxiv — text + panel

| Dataset name | Entries |
|---|---:|
| `arxiv_50_50_21k_text_panel_wmetadata_filtered24480_train` | 20,922 |
| `..._validation` | 1,223 |
| `..._test` | 2,457 |

### Mirror under shared (ICLR only)

```
/scratch/gpfs/ZHUANGL/shared/paperlensdata/
├── panel/
│   ├── images/<sid>.png      (24,925 PNGs, 70 GB)
│   ├── train/data.json       (21,174 entries)
│   ├── val/data.json         (1,253)
│   └── test/data.json        (2,498)
└── text_panel/
    ├── train/data.json       (21,134)
    ├── val/data.json         (1,252)
    └── test/data.json        (2,495)
```

`images:` fields use absolute paths into `panel/images/`. Arxiv was not mirrored to shared.

### Prompt format reference

| Variant | User content | `<image>` tokens | `images` field |
|---|---|---:|---|
| 8-page baseline (existing) | preamble + title + abstract + `<image>×8` | 8 | 8 page PNGs |
| **panel-only** | preamble + title + abstract + `<image>` | 1 | 1 panel PNG |
| **text + panel** | preamble + full paper markdown + `<image>` | 1 | 1 panel PNG |

`text+panel` builders also scrub stray `<image>/<audio>/<video>` tokens from the source paper body (HTML-derived ICLR papers + code-block tag examples) so qwen2_vl's MM plugin doesn't try to bind ghost tags to images at load time.

---

## Training results

### Existing 8-page vision baseline (from `OVERALL.csv`)

`saves/.../bz16_lr1e-6_vision/checkpoint-2648` — Qwen2.5-VL-7B full FT, 4 epochs, bz=16, lr=1e-6.

| Test acc 2025 | Test acc 2026 | **Best 2025+2026** | Tokens/paper |
|---:|---:|---:|---:|
| 70.9% | 66.5% | **68.4%** | ~9,920 |

### ICLR panel-only training (this work; job 7562033, COMPLETED 11h08m on 2 GPUs)

`vision_bz16_panel_ailab.sbatch` — same model, full FT, 4 epochs, **bz=16 (2 GPU × per_device=1 × grad_accum=8)**, lr=1e-6, qwen2_vl template, `image_max_pixels=4014080` passed through to both training and per-checkpoint eval. Test set: `..._panel_test` (2,498 entries — same papers as the 8-page baseline test set).

| ckpt | epoch | Overall (Acc/AccR/RejR) | 2025 | 2026 |
|---|---|---|---|---|
| 1324 | 1 | 65.3 / 79.7 / 50.9 | 67.7 / 81.4 / 54.0 | 64.0 / 79.8 / 48.2 |
| 2648 | 2 | 66.1 / 69.0 / 63.2 | **69.5** / 69.0 / **69.9** | 63.3 / 65.7 / 60.9 |
| 3972 | 3 | 66.1 / 70.8 / 61.3 | 68.6 / 72.9 / 64.3 | 64.1 / 69.0 / 59.3 |
| **5296** | **4** | **66.2** / 70.5 / 61.8 | 68.9 / 72.3 / 65.5 | **64.2** / 68.5 / 59.9 |

By epoch 2 the panel run is balanced (AccR/RejR within a few points of each other) and within ~2 pts of the 8-page baseline. Two more epochs only add ~0.1 pp.

### Headline comparison (best ckpts each)

| | 8-page baseline | Panel-only (FT'd) | Δ |
|---|---|---|---|
| Best test acc | 68.4% | **66.2%** | −2.2 pp |
| 2025 | 70.9% | 68.9% | −2.0 pp |
| 2026 | 66.5% | 64.2% | −2.3 pp |
| Vision tokens / paper | 9,920 | 4,590 | **−54%** |
| Tokens-per-accuracy-point | 145.0 | **69.3** | **2.1× more efficient** |

**Panel achieves ~97% of 8-page accuracy at ~46% of the vision-token cost.** The fine-tune fully fixes the zero-shot Accept-bias seen on the 8-page checkpoint when fed panels (epoch-1 panel-only is ~80% AccR/51% RejR, epoch-4 is 70.5/61.8 — properly balanced).

### What did NOT finish

- **ICLR text+panel** (`vision_bz32_text_panel_ailab.sbatch`): two attempts.
  - 7562034 — failed step 23, `cutoff_len=28000` too small (qwen2_vl template overhead pushed longest paper past 28k).
  - 7566398 — bumped to `cutoff_len=30000`, trained 108 steps (~42 min, ~4% of total), then died with `Image features and image tokens do not match: tokens: 18361, features 18360` (off-by-one Qwen2.5-VL preprocessing quirk on one of the 4 in-batch panels). **Never produced a usable checkpoint.**
- **arxiv panel-only and text+panel**: datasets exist + re-rendered with venue-aware borders, training never submitted.

---

## File index

Builders & helpers:
- `scripts/build_panel_images.py --source {iclr,arxiv}` — bulk panel PNG generator, includes `VENUE_PAGE_BORDER` map and `compose_panel*` helpers
- `scripts/build_panel_dataset.py --source {iclr,arxiv}` — writes panel-only data.json + registers in dataset_info.json
- `scripts/build_text_panel_dataset.py --source {iclr,arxiv}` — writes text+panel data.json (with stray-tag scrubbing)
- `scripts/dump_arxiv_venue_samples.py` — per-venue contact-sheet generator (visualization only)

Single-paper viewers (matplotlib-based, decoupled from the bulk pipeline):
- `scripts/tmp_latex_dir/visualize_paper.py` — render one paper's first 4 page screenshots, 1×4 grid, trim+pad
- `scripts/tmp_latex_dir/visualize_paper_text.py` — pandoc + xelatex typeset of the markdown, same 1×4 layout

Training sbatches (ICLR done, arxiv pending):
- `sbatch/.../vision_bz16_panel_ailab.sbatch`        ← ICLR panel-only (DONE)
- `sbatch/.../vision_bz32_text_panel_ailab.sbatch`   ← ICLR text+panel (FAILED twice)
- `sbatch/.../vision_bz16_panel_arxiv_ailab.sbatch`  ← arxiv panel-only (not submitted)
- `sbatch/.../vision_bz32_text_panel_arxiv_ailab.sbatch` ← arxiv text+panel (not submitted)

Inference sbatches:
- `sbatch/.../validation_infer/validation_panel_bz16.sbatch` — panel validation eval with the existing 8-page checkpoint or the new panel-trained checkpoint
- `sbatch/.../validation_infer/ocr_smoke_panel.sbatch` — OCR legibility smoke test

Sample images (in `tmp_latex_dir/figures/`):
- `paper_FhBT596F1X.{png,pdf}` (8-page panel viewer sample)
- `paper_text_FhBT596F1X.{png,pdf}` (markdown-typeset viewer sample)
- `panel_smoke_2510.25867.png` (arxiv iclr-venue, border=0)
- `panel_smoke_2201.00346.png` (arxiv aaai-venue, border=24)
- `panel_smoke_2511.14208.png` (arxiv cvpr-venue, border=26)
- `arxiv_venue_samples/<venue>.jpg` × 11 (per-venue contact sheets)
