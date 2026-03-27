# Page 8: Optim 2020-2025 Origin — Corrected (4-Epoch)

**12 variants (6 text + 6 vision), 4 epochs each**

- **Base results path:** `results/final_sweep_v7_datasweepv3/optim_2020_2025_origin/{variant}/`
- **Base saves path:** `saves/final_sweep_v7_datasweepv3/optim_2020_2025_origin/{variant}/`
- **Sbatch path:** `sbatch/final_sweep_v7/datasweep_v3/optim_2020_2025_origin/`
- **Text dataset:** `iclr_2020_2025_85_5_10_balanced_original_text_corrected_v7_filtered`
- **Vision dataset:** `iclr_2020_2025_85_5_10_balanced_original_vision_corrected_v7_filtered`
- **Config:** cosine LR schedule, 4 epochs, WD=0

**Variant grid:**
| Batch Size | Text LRs | Vision LRs |
|-----------|----------|------------|
| 16 | 0.5e-6, 1e-6 | 1e-6, 2e-6 |
| 32 | 1e-6, 2e-6 | 2e-6, 4e-6 |
| 64 | 2e-6, 4e-6 | 4e-6, 5.5e-6 |

**Current status:** All 12 variants NOT STARTED. Sbatch files exist but no training submitted yet. All text and vision jobs pending in queue.

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
<td>pending</td>
<td>PLI 4771311</td>
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
<td>bz16_lr1e-6_text</td>
<td>pending</td>
<td>PLI 4771311</td>
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
<td>bz32_lr1e-6_text</td>
<td>pending</td>
<td>PLI 4771312</td>
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
<td>bz32_lr2e-6_text</td>
<td>pending</td>
<td>PLI 4771312</td>
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
<td>bz64_lr2e-6_text</td>
<td>pending</td>
<td>PLI 4771313</td>
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
<td>bz64_lr4e-6_text</td>
<td>pending</td>
<td>PLI 4771313</td>
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
<td>bz16_lr1e-6_vision</td>
<td>pending</td>
<td>PLI 4771314</td>
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
<td>bz16_lr2e-6_vision</td>
<td>pending</td>
<td>PLI 4771314</td>
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
<td>bz32_lr2e-6_vision</td>
<td>pending</td>
<td>PLI 4834053</td>
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
<td>bz32_lr4e-6_vision</td>
<td>pending</td>
<td>PLI 4834053</td>
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
<td>pending</td>
<td>PLI 4771316</td>
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
<td>bz64_lr5.5e-6_vision</td>
<td>pending</td>
<td>PLI 4771316</td>
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
</table>

## Notion Properties
- **Task name:** Optim 2020-2025 Origin — Corrected (4-Epoch)
- **Status:** Not started
- **Priority:** High
- **Epochs:** 4
- **LR Scheduler:** cosine
- **Modality:** Text, Vision
- **Description:** Optimizer/batch-size search on corrected 2020-2025 origin dataset (2020+2021+2022+2023+2025 years, with 2024 correction). 12 variants: 3 batch sizes x 2 LRs x 2 modalities. 4 epochs, cosine LR, WD=0.

## Notes
- All 12 training jobs submitted and pending in SLURM queue (Priority reason).
- bz32 vision was resubmitted (PLI 4834053 replaces original).
- No saves or results directories created yet.
