# Page 7: Optim Search 2026 — Labelfix (4-Epoch)

**12 variants (6 text + 6 vision), 4 epochs each**

- **Base results path:** `results/final_sweep_v7_datasweepv3/optim_search_2026/{variant}/`
- **Base saves path:** `saves/final_sweep_v7_datasweepv3/optim_search_2026/{variant}/`
- **Sbatch path:** `sbatch/final_sweep_v7/datasweep_v3/optim_search_2026/`
- **Text dataset:** `iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered`
- **Vision dataset:** `iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480`
- **Config:** cosine LR schedule, 4 epochs, WD=0

**Variant grid:**
| Batch Size | Text LRs | Vision LRs |
|-----------|----------|------------|
| 16 | 0.5e-6, 1e-6 | 1e-6, 2e-6 |
| 32 | 1e-6, 2e-6 | 2e-6, 4e-6 |
| 64 | 2e-6, 4e-6 | 4e-6, 5.5e-6 |

Steps/epoch: text bz16 = 1322 (ep1=1322, ep2=2644, ep3=3966, ep4=5288) | vision bz16 = 1324 (ep1=1324, ep2=2648, ep3=3972, ep4=5296) | bz32 = half, bz64 = quarter

**Current status:** 5/6 vision variants fully done (train + all 4 epoch inference). bz32_lr4e-6_vision still pending (never launched). 2 text bz64 variants fully done. bz16_lr0.5e-6_text fully done. bz16_lr1e-6_text running (~ep2.5). bz32 text variants running (~ep1.6).

<table fit-page-width="true" header-row="true">
<tr>
<td>Variant</td>
<td>Train</td>
<td>PROOF</td>
<td>Ep1 TrainInf</td>
<td>PROOF</td>
<td>Ep1 TestInf</td>
<td>PROOF</td>
<td>Ep2 TrainInf</td>
<td>PROOF</td>
<td>Ep2 TestInf</td>
<td>PROOF</td>
<td>Ep3 TrainInf</td>
<td>PROOF</td>
<td>Ep3 TestInf</td>
<td>PROOF</td>
<td>Ep4 TrainInf</td>
<td>PROOF</td>
<td>Ep4 TestInf</td>
<td>PROOF</td>
</tr>
<tr>
<td>bz16_lr0.5e-6_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr0.5e-6_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr0.5e-6_text/train-ckpt-1322.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr0.5e-6_text/finetuned-ckpt-1322.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr0.5e-6_text/train-ckpt-2644.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr0.5e-6_text/finetuned-ckpt-2644.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr0.5e-6_text/train-ckpt-3966.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr0.5e-6_text/finetuned-ckpt-3966.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr0.5e-6_text/train-ckpt-5288.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr0.5e-6_text/finetuned-ckpt-5288.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_text</td>
<td>running</td>
<td>saves/.../optim_search_2026/bz16_lr1e-6_text/ (step 3300/5288, ~ep2.5)</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_text/train-ckpt-1322.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_text/finetuned-ckpt-1322.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_text/train-ckpt-2644.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_text/finetuned-ckpt-2644.jsonl</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>bz32_lr1e-6_text</td>
<td>running</td>
<td>saves/.../optim_search_2026/bz32_lr1e-6_text/ (step 1080/2644, ~ep1.6)</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/train-ckpt-661.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/finetuned-ckpt-661.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/train-ckpt-1322.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/finetuned-ckpt-1322.jsonl</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>bz32_lr2e-6_text</td>
<td>running</td>
<td>saves/.../optim_search_2026/bz32_lr2e-6_text/ (step 1060/2644, ~ep1.6)</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_text/train-ckpt-661.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_text/finetuned-ckpt-661.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_text/train-ckpt-1322.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_text/finetuned-ckpt-1322.jsonl</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>bz64_lr2e-6_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr2e-6_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr2e-6_text/train-ckpt-331.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr2e-6_text/finetuned-ckpt-331.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr2e-6_text/train-ckpt-662.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr2e-6_text/finetuned-ckpt-662.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr2e-6_text/train-ckpt-993.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr2e-6_text/finetuned-ckpt-993.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr2e-6_text/train-ckpt-1324.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr2e-6_text/finetuned-ckpt-1324.jsonl</td>
</tr>
<tr>
<td>bz64_lr4e-6_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_text/train-ckpt-331.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_text/finetuned-ckpt-331.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_text/train-ckpt-662.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_text/finetuned-ckpt-662.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_text/train-ckpt-993.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_text/finetuned-ckpt-993.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_text/train-ckpt-1324.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_text/finetuned-ckpt-1324.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/train-ckpt-1324.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-1324.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/train-ckpt-2648.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/train-ckpt-3972.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-3972.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/train-ckpt-5296.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-5296.jsonl</td>
</tr>
<tr>
<td>bz16_lr2e-6_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr2e-6_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr2e-6_vision/train-ckpt-1324.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr2e-6_vision/finetuned-ckpt-1324.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr2e-6_vision/train-ckpt-2648.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr2e-6_vision/finetuned-ckpt-2648.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr2e-6_vision/train-ckpt-3972.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr2e-6_vision/finetuned-ckpt-3972.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr2e-6_vision/train-ckpt-5296.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr2e-6_vision/finetuned-ckpt-5296.jsonl</td>
</tr>
<tr>
<td>bz32_lr2e-6_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_vision/train-ckpt-662.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_vision/finetuned-ckpt-662.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_vision/train-ckpt-1324.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_vision/finetuned-ckpt-1324.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_vision/train-ckpt-1986.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_vision/finetuned-ckpt-1986.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_vision/train-ckpt-2648.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_vision/finetuned-ckpt-2648.jsonl</td>
</tr>
<tr>
<td>bz32_lr4e-6_vision</td>
<td>pending</td>
<td>(not launched — save dir does not exist)</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>bz64_lr4e-6_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_vision/train-ckpt-331.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_vision/finetuned-ckpt-331.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_vision/train-ckpt-662.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_vision/finetuned-ckpt-662.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_vision/train-ckpt-993.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_vision/finetuned-ckpt-993.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_vision/train-ckpt-1324.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr4e-6_vision/finetuned-ckpt-1324.jsonl</td>
</tr>
<tr>
<td>bz64_lr5.5e-6_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr5.5e-6_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr5.5e-6_vision/train-ckpt-331.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr5.5e-6_vision/finetuned-ckpt-331.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr5.5e-6_vision/train-ckpt-662.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr5.5e-6_vision/finetuned-ckpt-662.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr5.5e-6_vision/train-ckpt-993.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr5.5e-6_vision/finetuned-ckpt-993.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr5.5e-6_vision/train-ckpt-1324.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search_2026/bz64_lr5.5e-6_vision/finetuned-ckpt-1324.jsonl</td>
</tr>
</table>

## Train Accuracy from Inference (train-ckpt JSON)

| Variant | Ep1 | Ep2 | Ep3 | Ep4 |
|---------|-----|-----|-----|-----|
| bz16_lr0.5e-6_text | 74.55% | 76.05% | 79.90% | 80.35% |
| bz16_lr1e-6_text | 74.70% | 79.85% | - | - |
| bz32_lr1e-6_text | 73.50% | 79.40% | - | - |
| bz32_lr2e-6_text | 74.20% | 78.70% | - | - |
| bz64_lr2e-6_text | 50.20% | 77.35% | 83.55% | 83.70% |
| bz64_lr4e-6_text | 49.10% | 80.05% | 92.20% | 92.30% |
| bz16_lr1e-6_vision | 69.25% | 74.90% | 82.20% | 82.10% |
| bz16_lr2e-6_vision | 71.80% | 83.95% | 93.40% | 93.35% |
| bz32_lr2e-6_vision | 44.75% | 74.60% | 87.55% | 87.10% |
| bz64_lr4e-6_vision | 73.00% | 76.00% | 87.95% | 88.05% |
| bz64_lr5.5e-6_vision | 68.95% | 77.10% | 89.50% | 89.35% |

## Notion Properties
- **Task name:** Optim Search 2026 — Labelfix (4-Epoch)
- **Status:** In progress
- **Priority:** High
- **Epochs:** 4
- **LR Scheduler:** cosine
- **Modality:** Text, Vision
- **Description:** Optimizer/batch-size search on 2026-labelfix dataset (2020+2023+2025+2026 years). 12 variants: 3 batch sizes x 2 LRs x 2 modalities. 4 epochs, cosine LR, WD=0.

## Notes
- Earlier text batch (bz16/bz32) completed ep1+ep2 inference from a prior failed training run. Those results are valid and included as ✅.
- bz32_lr4e-6_vision was never launched (save directory does not exist).
- Top performers: bz16_lr2e-6_vision (93.4% train acc), bz64_lr4e-6_text (92.3% train acc).

---

## Diff Summary (vs. previous staging)

- **bz16_lr0.5e-6_text**: Train `pending` → `✅` (all_results.json exists). Ep3+Ep4 TrainInf/TestInf → `✅` (all files on disk).
- **bz16_lr1e-6_text**: Train `pending` → `running` (step 3300/5288, ~ep2.5). Ep1+Ep2 unchanged (already ✅).
- **bz32_lr1e-6_text**: Train `pending` → `running` (step 1080/2644, ~ep1.6). Ep1 TrainInf proof corrected (train-ckpt-661.json). Ep1+Ep2 ✅.
- **bz32_lr2e-6_text**: Train `pending` → `running` (step 1060/2644, ~ep1.6). Ep1 TrainInf proof corrected (train-ckpt-661.json). Ep1+Ep2 ✅.
- **bz64_lr2e-6_text**: Train `pending` → `✅`. All 4 epochs TrainInf/TestInf → `✅`.
- **bz64_lr4e-6_text**: Train `pending` → `✅`. All 4 epochs TrainInf/TestInf → `✅`.
- **bz16_lr1e-6_vision**: Train `running` → `✅`. All 4 epochs TrainInf/TestInf → `✅`.
- **bz16_lr2e-6_vision**: Train `running` → `✅`. All 4 epochs TrainInf/TestInf → `✅`.
- **bz32_lr2e-6_vision**: Train `running` → `✅`. All 4 epochs TrainInf/TestInf → `✅`.
- **bz32_lr4e-6_vision**: Unchanged — still pending (never launched).
- **bz64_lr4e-6_vision**: Train `running` → `✅`. All 4 epochs TrainInf/TestInf → `✅`.
- **bz64_lr5.5e-6_vision**: Train `pending` → `✅`. All 4 epochs TrainInf/TestInf → `✅`.
