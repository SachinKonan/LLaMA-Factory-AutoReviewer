# Panel datasets, training & cross-domain evaluation

## TL;DR
A single-image **panel** representation of a paper (5×2 grid of 10 page screenshots on a 2380×1512 canvas, **4,590 Qwen2.5-VL vision tokens / paper**):
- gets within ~2 pp of the existing 8-page vision baseline on ICLR (66.2% vs 68.4%) at **~2× cheaper FLOPs**,
- **beats** 8-page out-of-domain on arxiv (67.5% vs 66.4%, same 2,458 papers / same prompt / same ckpt — also at ~2× cheaper FLOPs) and stays balanced (65.9 AccR / 69.1 RejR) while 8-page collapses to a reject-biased classifier (51.9 / 80.8),
- and matches the 7B text-only baseline (66.0%) at **~1/5 the input-token cost**.

---

## 1. Motivation

Going into this we had three baselines for paper-classification on ICLR test:

| Variant | Model | Input | Tokens / paper | Best test acc (2025+2026) |
|---|---|---|---:|---:|
| Text 7B | Qwen2.5-7B (LLM only) | full paper markdown | ~24,480 | 66.0% |
| Vision 7B (8-page) | Qwen2.5-VL-7B | 8 page screenshots | ~9,920 | **68.4%** |
| Vision 7B (panel — this work) | Qwen2.5-VL-7B | 1 panel image | **4,590** | **66.2%** |

**Question:** can we get most of the 8-page vision quality at a fraction of the input cost by collapsing the 8 separate page images into one composite image?

Why this should work:
- Qwen2.5-VL's vision tower scales as O(W·H / 28²) for a single image, so one big image with the same total pixels as 8 separate images costs the same number of vision tokens — but with fewer per-image overheads (position embeddings, image-start/end markers, etc.).
- Compressing to a smaller composite (e.g., 5×2 tiled at 2380×1512) cuts vision tokens further while keeping per-page content readable to the model.
- Single image = simpler global attention pattern across the whole paper, instead of 8 independently-encoded image chunks.

Cost target: aim for under 5,000 vision tokens (smart_resize no-op at `image_max_pixels=4014080`), comparable to existing baseline costs at ~half the FLOPs.

---

## 2. How the data was created

### Image format

| Property | Value |
|---|---|
| Canvas | **2380 × 1512** px (both dims multiples of 28 → Qwen2.5-VL `smart_resize` is a no-op at `image_max_pixels=4014080`) |
| Layout | 5 cols × 2 rows = 10 page slots (left-to-right, top-to-bottom) |
| Per-cell | 476 × 756 px (cell aspect 0.625) |
| Per-cell processing | `trim_white_margins` (crop solid-white PDF margins) → `fit_into_cell` (letterbox-fit preserving aspect) → paste at `(border, border)` inside the cell |
| Per-venue inner border (px) | iclr/colm=0, neurips=12, corl=18, eccv/aistats/acl=22, icml/iccv/aaai=24, cvpr=26 |
| **Qwen2.5-VL vision tokens / paper** | **4,590** |

Per-venue borders compensate for source-PDF margin differences. Single-column venues (ICLR/COLM) already get ~30 px natural letterbox from their narrow trimmed-content aspect (~0.55) inside the 0.625 cell. Two-column venues (CVPR/ICCV/AAAI/...) trim flush with cell edges, so we inject explicit padding for visible page boundaries.

### Build pipeline

- `scripts/build_panel_images.py --source {iclr,arxiv}` — bulk panel PNG generator (16 workers via srun on cpu partition; ~45 min for ~25k papers, 0 failures).
- `scripts/build_panel_dataset.py --source {iclr,arxiv}` — emits panel-only data.json files + registers in `data/dataset_info.json`. Collapses the 8 `<image>` placeholders in the source vision dataset to a single `<image>` to match the new image count.
- `scripts/build_text_panel_dataset.py --source {iclr,arxiv}` — emits text+panel data.json (full markdown + `<image>` suffix). Scrubs stray `<image>/<audio>/<video>` tags from paper bodies before appending the terminal `<image>` (some HTML-derived papers + code-block tag examples confuse qwen2_vl's MM plugin otherwise).
- Single-paper viewers `scripts/tmp_latex_dir/visualize_paper.py` and `visualize_paper_text.py` produce 1×4 inspection panels (matplotlib subplots, decoupled from the bulk pipeline).
- `scripts/dump_arxiv_venue_samples.py` — per-venue contact-sheet generator (visualization only; same `VENUE_PAGE_BORDER` map as the builder).

### Verification

- **OCR smoke** (base Qwen2.5-VL-7B on 5 sampled panels): 3/5 papers transcribed at 52–78% unique-word recall — content is legibly extractable at 4,590 tokens.
- **Stray-tag scrub** (text+panel only, all splits): 0 stray `<image>/<audio>/<video>` tags after build; exactly 1 terminal `<image>` per row.
- **All `images:` fields verified to resolve** before training begins.

### Where the data is stored

#### Panel image directories

| Source | Path | Files | Size |
|---|---|---:|---:|
| ICLR | `data/images_panel/<sid>.png` | 24,925 | 70 GB |
| arxiv | `data/images_panel_arxiv/<aid>.png` | 24,610 | 61 GB |

#### Sharegpt-format dataset.json files (16 entries registered in `data/dataset_info.json`)

ICLR — panel-only:
| Dataset | Entries |
|---|---:|
| `iclr_..._original_vision_labelfix_v7_filtered_filtered24480_panel_train` | 21,174 |
| `..._panel_validation` | 1,253 |
| `..._panel_test` | 2,498 |
| `..._panel_y24up_test` (year ≥ 2024) | **1,670** |

ICLR — text+panel:
| Dataset | Entries |
|---|---:|
| `iclr_..._text_panel_v7_filtered_train` | 21,134 |
| `..._validation` | 1,252 |
| `..._test` | 2,495 |
| `..._y24up_test` (year ≥ 2024) | **1,667** |

arxiv — panel-only:
| Dataset | Entries |
|---|---:|
| `arxiv_50_50_21k_vision_wmetadata_filtered24480_panel_train` | 20,929 |
| `..._panel_validation` | 1,223 |
| `..._panel_test` | 2,458 |
| `..._panel_y25up_test` (conference_year ≥ 2025) | **923** |

arxiv — text+panel:
| Dataset | Entries |
|---|---:|
| `arxiv_50_50_21k_text_panel_wmetadata_filtered24480_train` | 20,922 |
| `..._validation` | 1,223 |
| `..._test` | 2,457 |
| `..._y25up_test` (conference_year ≥ 2025) | **923** |

#### Shared mirror (ICLR only)
```
/scratch/gpfs/ZHUANGL/shared/paperlensdata/
├── panel/
│   ├── images/<sid>.png       (24,925 PNGs, 70 GB)
│   ├── train/data.json        (21,174 entries)
│   ├── val/data.json          (1,253)
│   ├── test/data.json         (2,498)
│   └── y24up_test/data.json   (1,670)
└── text_panel/
    ├── train/data.json        (21,134)
    ├── val/data.json          (1,252)
    ├── test/data.json         (2,495)
    └── y24up_test/data.json   (1,667)
```
Absolute image paths used in shared. Arxiv was not mirrored to shared.

#### Prompt format
| Variant | User content | `<image>` count | `images` field |
|---|---|---:|---|
| 8-page baseline | preamble + title + abstract + `<image>×8` | 8 | 8 page PNGs |
| **Panel-only** | preamble + title + abstract + `<image>` | 1 | 1 panel PNG |
| **Text + panel** | preamble + full paper markdown + `<image>` | 1 | 1 panel PNG |

---

## 3. Results

### 3a. ICLR in-domain (panel-only fine-tune, 2,498-paper test set)

Job 7562033 — Qwen2.5-VL-7B-Instruct, full FT, 4 epochs, bz=16 (2 GPU × per_device=1 × grad_accum=8), lr=1e-6, qwen2_vl template, `image_max_pixels=4014080` passed through to both training and per-checkpoint eval. 11 h 08 min, 0 failures.

| ckpt | epoch | Overall (Acc/AccR/RejR) | 2025 | 2026 |
|---|---|---|---|---|
| 1324 | 1 | 65.3 / 79.7 / 50.9 | 67.7 / 81.4 / 54.0 | 64.0 / 79.8 / 48.2 |
| 2648 | 2 | 66.1 / 69.0 / 63.2 | **69.5** / 69.0 / **69.9** | 63.3 / 65.7 / 60.9 |
| 3972 | 3 | 66.1 / 70.8 / 61.3 | 68.6 / 72.9 / 64.3 | 64.1 / 69.0 / 59.3 |
| **5296** | **4** | **66.2** / 70.5 / 61.8 | 68.9 / 72.3 / 65.5 | **64.2** / 68.5 / 59.9 |

By epoch 2 the panel run is balanced (AccR/RejR within a few points of each other) and within ~2 pp of the 8-page baseline. Two more epochs add ~0.1 pp.

### 3b. Cross-domain: ICLR-trained panel → arxiv panel test (2,458 entries)

Same 4 checkpoints, evaluated zero-shot on the arxiv panel test set. All 4 finished (jobs 7723541_[1,3], 7746451_[0,2]; ~28 min each on gpu-test).

| ckpt | epoch | Overall (Acc/AccR/RejR) | y25up (n=923) |
|---|---|---|---|
| 1324 | 1 | 67.2 / 76.6 / 58.0 | 65.5 / 80.1 / 52.2 |
| **2648** | **2** | **67.5** / 65.9 / 69.1 | **67.2** / 67.4 / 66.9 |
| 3972 | 3 | 66.3 / 67.1 / 65.4 | 64.6 / 67.9 / 61.5 |
| 5296 | 4 | 66.3 / 66.5 / 66.2 | 64.5 / 66.7 / 62.4 |

**Per-venue accuracy (best ckpt 2648):**

| venue | n | Acc | | venue | n | Acc |
|---|---:|---:|---|---|---:|---:|
| neurips | 679 | **71.3%** | | iclr | 176 | **72.7%** |
| cvpr | 493 | 66.3% | | icml | 174 | **72.4%** |
| acl | 444 | 66.0% | | iccv | 145 | 65.5% |
| aaai | 214 | 57.9% | | eccv | 75 | 58.7% |
| corl | 14 | 92.9% | | colm | 24 | 62.5% |
| aistats | 20 | 50.0% | | | | |

Headline observations:
- **Cross-domain transfer is real.** ICLR-trained panel hits **67.5% on arxiv test**, actually higher than its 66.2% in-domain best (arxiv mix is easier: lots of NeurIPS, which looks the most ICLR-like).
- **Epoch 2 generalizes better than epoch 4** on out-of-domain arxiv. Later checkpoints over-fit to ICLR-specific layout/language. In-domain the best is epoch 4; out-of-domain it's epoch 2.
- **Per-venue gradient:** NeurIPS/ICLR/ICML (single-column, similar formatting) at 71–73% → CVPR/ACL/ICCV at 65–66% → AAAI/EccV (compact 2-column) at 58%. Visual distribution-shift maps cleanly onto layout family.

#### 3b.i Head-to-head: panel vs 8-page cross-domain (same 2,458 papers, same prompt)

Job 9054850 — ran the existing **8-page** ICLR vision checkpoint (`bz16_lr1e-6_vision/checkpoint-2648`) on `arxiv_50_50_21k_vision_wmetadata_filtered24480_test` (same paper IDs as `..._panel_test`, just 8-page format). Apples-to-apples with the panel cross-domain at the same epoch-2 ckpt:

| | 8-page (~9,920 tok) | Panel (4,590 tok) | Δ |
|---|---:|---:|---:|
| **Overall Acc** | 66.4 | **67.5** | **+1.1 pp** |
| Overall AccR / RejR | 51.9 / **80.8** | **65.9** / 69.1 | balance shift |
| **y25up Acc (n=923)** | 66.1 | **67.2** | **+1.1 pp** |

Per-venue (8-page → panel): neurips 67.6 → **71.3**, cvpr 66.1 → 66.3, acl 61.5 → **66.0**, aaai 58.9 → 57.9, iclr 74.4 → 72.7, icml 74.1 → 72.4, iccv 69.0 → 65.5, eccv 66.7 → 58.7, colm 58.3 → **62.5**, aistats 65.0 → 50.0, corl 85.7 → **92.9**.

**Two findings:**
1. **Panel is slightly *better* than 8-page cross-domain (+1.1 pp overall, +1.1 pp on y25up) at ~½ the vision tokens.** The "panel saves FLOPs at a small accuracy cost" story (true in-domain on ICLR: −2.2 pp) flips cleanly out-of-domain — panel saves FLOPs *and* generalizes better.
2. **8-page transfers as a severely reject-biased classifier on arxiv (AccR 51.9 / RejR 80.8 — under-accepts everything).** Panel is far more balanced (65.9 / 69.1). The single-image input appears to be a more transferable feature for this task.

### 3c. 7B comparison + FLOPs

All three baselines use Qwen2.5-(VL-)7B fine-tuned on the same train set; only the input representation differs.

| Variant | Tokens/paper | Best ICLR test | Cross-domain arxiv test (e2 ckpt) | LLM FLOPs / paper (≈) | Vision-tower FLOPs (≈) | **Total** |
|---|---:|---:|---:|---:|---:|---:|
| Text 7B (Qwen2.5-7B) | ~24,480 | 66.0% | n/a | ~343 TF | 0 | **343 TF** |
| Vision 7B 8-page | ~9,920 | **68.4%** | 66.4% | ~143 TF | ~13 TF | **~156 TF** |
| **Panel 7B (this work)** | **4,590** | **66.2%** | **67.5%** | ~69 TF | ~6 TF | **~75 TF** |

(FLOPs estimate: `2 · N_params · tokens` per forward pass. Vision-tower N≈675M per visual token; LLM N≈7B per total token.)

**Per-accuracy efficiency:**
- Tokens / accuracy-point: text 7B = 371, vision 8-page = 145, panel = **69 (5.4× / 2.1× more efficient)**
- FLOPs / accuracy-point: text 7B = 5.2 TF, vision 8-page = 2.3 TF, panel = **1.1 TF (4.7× / 2.0× more efficient)**

So:
- Vs **text 7B**: panel matches accuracy (66.2% vs 66.0%) at **~1/5 the input tokens** and **~1/5 the FLOPs**.
- Vs **vision 7B 8-page**: panel gives up 2.2 pp on accuracy (66.2% vs 68.4%) for **~½ the FLOPs**.
- Plus: panel transfers to arxiv out-of-domain at 67.5% with zero arxiv training — better than its in-domain ICLR result.

---

## 4. What didn't finish

### ICLR text+panel training — never produced a usable checkpoint

`vision_bz32_text_panel_ailab.sbatch` — full text (24,480-token budget) + 1 panel image (4,590 vision tokens). Effective context ~29k tokens per sample. Two attempts:

- **7562034** — failed at step 23 (~15 min in) with `cutoff_len=28000` too small. qwen2_vl template overhead pushed the longest paper past 28k.
- **7566398** — bumped `cutoff_len` to 30,000, trained 108 steps (~42 min, ~4% of total), then died with `Image features and image tokens do not match: tokens: 18361, features 18360` (off-by-one Qwen2.5-VL preprocessing quirk on one of the 4 in-batch panels). Likely fixable by either (a) excluding ~handful of samples with edge-case panel resize behavior, or (b) preprocessing-disable workaround.

Net: the question "does adding full text on top of the panel close the −2.2 pp gap vs the 8-page baseline?" remains **unanswered**.

### arxiv training — datasets ready, never submitted

Per call earlier in the session: cancelled both arxiv training jobs (`ax_panel`, `ax_txtpan`). Datasets exist + re-rendered today with venue-aware borders, sbatches written (`vision_bz16_panel_arxiv_ailab.sbatch`, `vision_bz32_text_panel_arxiv_ailab.sbatch`), inference path verified via the cross-domain run. Just need to re-submit if/when wanted.

---

## File index

Builders & helpers:
- `scripts/build_panel_images.py --source {iclr,arxiv}` — bulk panel PNG generator, `VENUE_PAGE_BORDER` map, `compose_panel*` helpers
- `scripts/build_panel_dataset.py --source {iclr,arxiv}` — panel-only data.json + register
- `scripts/build_text_panel_dataset.py --source {iclr,arxiv}` — text+panel data.json (with stray-tag scrub)
- `scripts/dump_arxiv_venue_samples.py` — per-venue contact-sheet generator
- `scripts/tmp_latex_dir/visualize_paper.py` / `visualize_paper_text.py` — single-paper inspection viewers (1×4 layout)

Training sbatches:
- `vision_bz16_panel_ailab.sbatch` — ICLR panel-only ✅ DONE
- `vision_bz32_text_panel_ailab.sbatch` — ICLR text+panel ❌ failed twice
- `vision_bz16_panel_arxiv_ailab.sbatch` — arxiv panel-only (not submitted)
- `vision_bz32_text_panel_arxiv_ailab.sbatch` — arxiv text+panel (not submitted)

Inference sbatches:
- `validation_panel_bz16.sbatch` — panel validation eval
- `iclr_panel_to_arxiv_test.sbatch` — cross-domain 4-task array (this work)
- `ocr_smoke_panel.sbatch` — OCR legibility smoke

Result JSONLs:
- ICLR in-domain: `results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_panel/finetuned-ckpt-{1324,2648,3972,5296}.jsonl`
- Cross-domain arxiv: `results/iclr_panel_to_arxiv/arxiv_test-ckpt-{1324,2648,3972,5296}.jsonl`

Sample images (`tmp_latex_dir/figures/`):
- `paper_FhBT596F1X.{png,pdf}` — ICLR single-paper panel viewer
- `paper_text_FhBT596F1X.{png,pdf}` — pandoc-typeset text viewer
- `panel_smoke_{2510.25867,2201.00346,2511.14208}.png` — arxiv iclr/aaai/cvpr smoke samples
- `arxiv_venue_samples/<venue>.jpg` × 11 — per-venue contact sheets
