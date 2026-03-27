# Page 3: Weight Decay Sweep (Base Scheduler)

**9 variants (3 text + 6 vision), 6 epochs each**
Base results path: `results/final_sweep_v7_datasweepv3/wd_sweep/{variant}/`
Base saves path: `saves/final_sweep_v7_datasweepv3/wd_sweep/{variant}/`

Steps/epoch: text = 797 (ep1=797, ep2=1594, ep3=2391, ep4=3188, ep5=3985, ep6=4782) | vision = 798 (ep1=798, ep2=1596, ep3=2394, ep4=3192, ep5=3990, ep6=4788)

6/9 complete, 3/9 timed out (higher-WD vision variants).

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
<td>bz16_lr1e-6_wd0_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/train-ckpt-797.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/finetuned-ckpt-797.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/train-ckpt-1594.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/finetuned-ckpt-1594.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/train-ckpt-2391.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/finetuned-ckpt-2391.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/train-ckpt-3188.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/finetuned-ckpt-3188.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/train-ckpt-3985.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/finetuned-ckpt-3985.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/train-ckpt-4782.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_text/finetuned-ckpt-4782.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_wd0.001_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/train-ckpt-797.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-797.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/train-ckpt-1594.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-1594.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/train-ckpt-2391.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-2391.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/train-ckpt-3188.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-3188.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/train-ckpt-3985.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-3985.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/train-ckpt-4782.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-4782.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_wd0.01_text</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/train-ckpt-797.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/finetuned-ckpt-797.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/train-ckpt-1594.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/finetuned-ckpt-1594.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/train-ckpt-2391.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/finetuned-ckpt-2391.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/train-ckpt-3188.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/finetuned-ckpt-3188.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/train-ckpt-3985.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/finetuned-ckpt-3985.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/train-ckpt-4782.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_text/finetuned-ckpt-4782.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_wd0_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/train-ckpt-1596.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/finetuned-ckpt-1596.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/train-ckpt-2394.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/finetuned-ckpt-2394.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/train-ckpt-3192.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/finetuned-ckpt-3192.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/train-ckpt-3990.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/finetuned-ckpt-3990.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/train-ckpt-4788.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0_vision/finetuned-ckpt-4788.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_wd0.001_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/train-ckpt-1596.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-1596.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/train-ckpt-2394.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-2394.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/train-ckpt-3192.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-3192.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/train-ckpt-3990.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-3990.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/train-ckpt-4788.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-4788.jsonl</td>
</tr>
<tr>
<td>bz16_lr1e-6_wd0.002_vision</td>
<td>timed out</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.002_vision/trainer_log.jsonl (last ep: 5.0)</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.002_vision/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.002_vision/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.002_vision/train-ckpt-1596.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.002_vision/finetuned-ckpt-1596.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.002_vision/train-ckpt-2394.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.002_vision/finetuned-ckpt-2394.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.002_vision/train-ckpt-3192.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.002_vision/finetuned-ckpt-3192.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.002_vision/train-ckpt-3990.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.002_vision/finetuned-ckpt-3990.jsonl</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>bz16_lr1e-6_wd0.004_vision</td>
<td>timed out</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.004_vision/trainer_log.jsonl (last ep: 4.93)</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.004_vision/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.004_vision/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.004_vision/train-ckpt-1596.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.004_vision/finetuned-ckpt-1596.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.004_vision/train-ckpt-2394.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.004_vision/finetuned-ckpt-2394.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.004_vision/train-ckpt-3192.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.004_vision/finetuned-ckpt-3192.jsonl</td>
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
<td>bz16_lr1e-6_wd0.008_vision</td>
<td>timed out</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.008_vision/trainer_log.jsonl (last ep: 4.91)</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.008_vision/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.008_vision/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.008_vision/train-ckpt-1596.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.008_vision/finetuned-ckpt-1596.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.008_vision/train-ckpt-2394.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.008_vision/finetuned-ckpt-2394.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.008_vision/train-ckpt-3192.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.008_vision/finetuned-ckpt-3192.jsonl</td>
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
<td>bz16_lr1e-6_wd0.01_vision</td>
<td>✅</td>
<td>saves/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/all_results.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/train-ckpt-798.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/finetuned-ckpt-798.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/train-ckpt-1596.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/finetuned-ckpt-1596.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/train-ckpt-2394.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/finetuned-ckpt-2394.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/train-ckpt-3192.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/finetuned-ckpt-3192.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/train-ckpt-3990.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/finetuned-ckpt-3990.jsonl</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/train-ckpt-4788.json</td>
<td>✅</td>
<td>results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.01_vision/finetuned-ckpt-4788.jsonl</td>
</tr>
</table>

## Train Accuracy from Newly Completed Inference (train-ckpt JSON)

| Variant | Epoch | Train Accuracy |
|---------|-------|---------------|
| wd0.002_vision | 2 | 76.60% |
| wd0.002_vision | 3 | 81.90% |
| wd0.002_vision | 4 | 81.50% |
| wd0.002_vision | 5 | 80.95% |
| wd0.004_vision | 2 | 77.40% |
| wd0.004_vision | 3 | 80.40% |
| wd0.004_vision | 4 | 80.35% |
| wd0.008_vision | 2 | 76.70% |
| wd0.008_vision | 3 | 80.00% |
| wd0.008_vision | 4 | 79.75% |

Note: Per-year test accuracy table needs updating via analysis script once all inference completes.

---

## Diff Summary (vs. previous staging)

- **wd0.002_vision**: Ep5 TrainInf/TestInf changed `pending PLI 4796668` → `✅` (train-ckpt-3990.json + finetuned-ckpt-3990.jsonl on disk). No ep6 (training timed out at ep 5.0).
- **wd0.004_vision**: Unchanged. No ep5/6 (training timed out at ep 4.93).
- **wd0.008_vision**: Ep4 TestInf changed `running PLI 4792538` → `✅` (finetuned-ckpt-3192.jsonl on disk). No ep5/6 (training timed out at ep 4.91).
- **Notion property**: Status remains `In progress` (not all higher-WD inference done yet — wd0.004 stops at ep4, wd0.008 stops at ep4).
