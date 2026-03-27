# Optim Search (3 Epochs) -- Experiment Status

12 variants (6 text, 6 vision), 3 epochs each.
Base paths:
- Results: `results/final_sweep_v7_datasweepv3/optim_search/{variant}/`
- Saves: `saves/final_sweep_v7_datasweepv3/optim_search/{variant}/`

All 12 variants have training complete (`all_results.json` present).

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
</tr>
<tr>
<td>bz16_lr0.5e-6_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search/bz16_lr0.5e-6_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr0.5e-6_text/train-ckpt-797.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr0.5e-6_text/finetuned-ckpt-797.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr0.5e-6_text/train-ckpt-1594.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr0.5e-6_text/finetuned-ckpt-1594.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr0.5e-6_text/train-ckpt-2391.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr0.5e-6_text/finetuned.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_text/train-ckpt-797.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_text/finetuned-ckpt-797.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_text/train-ckpt-1594.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_text/finetuned-ckpt-1594.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_text/train-ckpt-2391.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_text/finetuned-ckpt-2391.jsonl</td>
</tr>
<tr>
<td>bz32_lr1e-6_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search/bz32_lr1e-6_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr1e-6_text/train-ckpt-399.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr1e-6_text/finetuned-ckpt-399.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr1e-6_text/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr1e-6_text/finetuned-ckpt-798.jsonl</td>
<td>not run</td>
<td>-</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr1e-6_text/finetuned.jsonl</td>
</tr>
<tr>
<td>bz32_lr2e-6_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search/bz32_lr2e-6_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr2e-6_text/train-ckpt-399.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr2e-6_text/finetuned-ckpt-399.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr2e-6_text/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr2e-6_text/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr2e-6_text/train-ckpt-1197.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr2e-6_text/finetuned.jsonl</td>
</tr>
<tr>
<td>bz64_lr2e-6_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search/bz64_lr2e-6_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr2e-6_text/train-ckpt-200.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr2e-6_text/finetuned-ckpt-200.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr2e-6_text/train-ckpt-400.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr2e-6_text/finetuned-ckpt-400.jsonl</td>
<td>not run</td>
<td>-</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr2e-6_text/finetuned.jsonl</td>
</tr>
<tr>
<td>bz64_lr4e-6_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search/bz64_lr4e-6_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr4e-6_text/train-ckpt-200.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr4e-6_text/finetuned-ckpt-200.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr4e-6_text/train-ckpt-400.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr4e-6_text/finetuned-ckpt-400.jsonl</td>
<td>not run</td>
<td>-</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr4e-6_text/finetuned.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_vision/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_vision/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_vision/train-ckpt-1596.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_vision/finetuned-ckpt-1596.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_vision/train-ckpt-2394.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr1e-6_vision/finetuned.jsonl</td>
</tr>
<tr>
<td>bz16_lr2e-6_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search/bz16_lr2e-6_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr2e-6_vision/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr2e-6_vision/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr2e-6_vision/train-ckpt-1596.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr2e-6_vision/finetuned-ckpt-1596.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr2e-6_vision/train-ckpt-2394.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz16_lr2e-6_vision/finetuned.jsonl</td>
</tr>
<tr>
<td>bz32_lr2e-6_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search/bz32_lr2e-6_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr2e-6_vision/train-ckpt-399.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr2e-6_vision/finetuned-ckpt-399.jsonl</td>
<td>ckpt cleaned</td>
<td>-</td>
<td>ckpt cleaned</td>
<td>-</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr2e-6_vision/train-ckpt-1197.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr2e-6_vision/finetuned.jsonl</td>
</tr>
<tr>
<td>bz32_lr4e-6_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search/bz32_lr4e-6_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr4e-6_vision/train-ckpt-399.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr4e-6_vision/finetuned-ckpt-399.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr4e-6_vision/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr4e-6_vision/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr4e-6_vision/train-ckpt-1197.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz32_lr4e-6_vision/finetuned-ckpt-1197.jsonl</td>
</tr>
<tr>
<td>bz64_lr4e-6_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search/bz64_lr4e-6_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr4e-6_vision/train-ckpt-200.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr4e-6_vision/finetuned-ckpt-200.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr4e-6_vision/train-ckpt-400.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr4e-6_vision/finetuned-ckpt-400.jsonl</td>
<td>not run</td>
<td>-</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr4e-6_vision/finetuned.jsonl</td>
</tr>
<tr>
<td>bz64_lr5.5e-6_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/optim_search/bz64_lr5.5e-6_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr5.5e-6_vision/train-ckpt-200.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr5.5e-6_vision/finetuned-ckpt-200.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr5.5e-6_vision/train-ckpt-400.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr5.5e-6_vision/finetuned-ckpt-400.jsonl</td>
<td>not run</td>
<td>-</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/optim_search/bz64_lr5.5e-6_vision/finetuned.jsonl</td>
</tr>
</table>

---

## Diff Summary (vs. current Notion page)

**Format changes:**
- Removed per-epoch Train columns (Train Ep1, Train Ep2, Train Ep3) -- single Train column now covers all epochs
- Removed Train Job column
- Added PROOF columns next to every status cell with relative file paths from project root

**Status corrections (11+ entries):**
- `bz16_lr0.5e-6_text`: Multiple entries corrected from non-complete to ✅ (all inference files confirmed present)
- `bz16_lr1e-6_text`: Multiple entries corrected from non-complete to ✅
- `bz32_lr1e-6_text`: Ep3 TrainInf confirmed `not run` (was showing as pending); other entries corrected to ✅
- `bz32_lr2e-6_text`: Entries corrected from non-complete to ✅
- `bz64_lr2e-6_text`: Ep3 TrainInf confirmed `not run` (was showing as pending); other entries corrected to ✅
- `bz64_lr4e-6_text`: Ep3 TrainInf confirmed `not run` (was showing as pending); other entries corrected to ✅
- `bz16_lr1e-6_vision`: Multiple entries corrected to ✅
- `bz16_lr2e-6_vision`: Multiple entries corrected to ✅
- `bz32_lr2e-6_vision`: Ep2 TrainInf and Ep2 TestInf changed to `ckpt cleaned` (checkpoint deleted before post-loop inference); was showing as pending
- `bz32_lr4e-6_vision`: Multiple entries corrected to ✅
- `bz64_lr4e-6_vision`: Ep3 TrainInf confirmed `not run`; other entries corrected to ✅
- `bz64_lr5.5e-6_vision`: Ep3 TrainInf confirmed `not run`; other entries corrected to ✅

**Net effect:** 11+ cells changed from pending/error icons to ✅; 5 Ep3 TrainInf cells confirmed as `not run`; 2 cells marked `ckpt cleaned`.
