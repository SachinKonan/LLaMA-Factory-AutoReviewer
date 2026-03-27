# Page 4: Weight Decay Sweep (Exponential Decay Scheduler)

**6 variants (3 vision + 3 text), 6 epochs each**
Base results path: `results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/{variant}/`
Base saves path: `saves/final_sweep_v7_datasweepv3/wd_sweep_expdecay/{variant}/`

Steps/epoch: text = 797 (ep1=797, ep2=1594, ep3=2391, ep4=3188, ep5=3985, ep6=4782) | vision = 798 (ep1=798, ep2=1596, ep3=2394, ep4=3192, ep5=3990, ep6=4788)

All 6 training runs complete. All inference complete (72/72 files).

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
<td>Ep5 TrainInf</td>
<td>PROOF</td>
<td>Ep5 TestInf</td>
<td>PROOF</td>
<td>Ep6 TrainInf</td>
<td>PROOF</td>
<td>Ep6 TestInf</td>
<td>PROOF</td>
</tr>
<tr>
<td>bz16_lr1e-6_wd0.001_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/train-ckpt-1596.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-1596.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/train-ckpt-2394.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-2394.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/train-ckpt-3192.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-3192.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/train-ckpt-3990.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-3990.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/train-ckpt-4788.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-4788.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_wd0.002_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/train-ckpt-1596.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/finetuned-ckpt-1596.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/train-ckpt-2394.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/finetuned-ckpt-2394.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/train-ckpt-3192.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/finetuned-ckpt-3192.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/train-ckpt-3990.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/finetuned-ckpt-3990.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/train-ckpt-4788.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/finetuned-ckpt-4788.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_wd0.004_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/train-ckpt-1596.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/finetuned-ckpt-1596.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/train-ckpt-2394.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/finetuned-ckpt-2394.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/train-ckpt-3192.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/finetuned-ckpt-3192.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/train-ckpt-3990.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/finetuned-ckpt-3990.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/train-ckpt-4788.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_vision/finetuned-ckpt-4788.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_wd0.001_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/train-ckpt-797.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-797.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/train-ckpt-1594.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-1594.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/train-ckpt-2391.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-2391.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/train-ckpt-3188.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-3188.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/train-ckpt-3985.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-3985.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/train-ckpt-4782.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-4782.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_wd0.002_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/train-ckpt-797.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/finetuned-ckpt-797.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/train-ckpt-1594.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/finetuned-ckpt-1594.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/train-ckpt-2391.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/finetuned-ckpt-2391.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/train-ckpt-3188.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/finetuned-ckpt-3188.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/train-ckpt-3985.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/finetuned-ckpt-3985.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/train-ckpt-4782.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_text/finetuned-ckpt-4782.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_wd0.004_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/train-ckpt-797.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/finetuned-ckpt-797.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/train-ckpt-1594.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/finetuned-ckpt-1594.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/train-ckpt-2391.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/finetuned-ckpt-2391.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/train-ckpt-3188.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/finetuned-ckpt-3188.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/train-ckpt-3985.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/finetuned-ckpt-3985.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/train-ckpt-4782.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.004_text/finetuned-ckpt-4782.jsonl</td>
</tr>
</table>

## Train Accuracy from Inference (train-ckpt JSON)

**Vision variants:**

| Variant | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 | Ep6 |
|---------|-----|-----|-----|-----|-----|-----|
| wd0.001_vision | 69.55% | 69.50% | 68.80% | 68.95% | 68.95% | 68.95% |
| wd0.002_vision | 70.85% | 69.05% | 69.00% | 68.40% | 68.40% | 68.40% |
| wd0.004_vision | 70.05% | 69.60% | 69.10% | 68.95% | 68.95% | 68.95% |

**Text variants:**

| Variant | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 | Ep6 |
|---------|-----|-----|-----|-----|-----|-----|
| wd0.001_text | 67.90% | 70.50% | 72.55% | 72.10% | 72.10% | 72.10% |
| wd0.002_text | 65.65% | 69.40% | 71.00% | 71.15% | 71.15% | 71.15% |
| wd0.004_text | 67.40% | 70.60% | 72.00% | 71.50% | 71.50% | 71.50% |

Note: Expdecay results plateau by ep4 and stagnate at ep5-6, suggesting the exponential decay schedule killed learning too aggressively. Vision peaks ~70% vs ~81% for base scheduler; text peaks ~72.5% vs ~87% for base.

---

## Diff Summary (vs. previous staging)

- **ALL 6 variants**: Every remaining `pending`/`running` → `✅`. All 72 files (6 variants × 6 epochs × 2 files) confirmed on disk.
- **Status → Done**
