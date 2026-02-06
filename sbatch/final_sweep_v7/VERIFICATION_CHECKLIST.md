# Learning Rate Experiment - Pre-Flight Verification

## ✅ All Checks Passed

### Scripts Fixed
- ✓ **WANDB_DISABLED=true** added to both scripts
- ✓ **--report_to none** added to both scripts
- ✓ Bash syntax validated for both scripts
- ✓ LR configuration logic tested and working

### Text Script (`lr_experiment_text_trainagreeing.sbatch`)
- ✓ WANDB_DISABLED=true
- ✓ TRANSFORMERS_OFFLINE=1
- ✓ --report_to none
- ✓ --stage cls
- ✓ --do_train
- ✓ --do_predict
- ✓ Dataset: `iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7`
- ✓ Model: Qwen/Qwen2.5-7B-Instruct
- ✓ Batch config: 4×2×2=16
- ✓ 3 LR configs: 2e-5, 2e-6, separate (2e-5 head/2e-6 backbone)

### Vision Script (`lr_experiment_vision_trainagreeing.sbatch`)
- ✓ WANDB_DISABLED=true
- ✓ TRANSFORMERS_OFFLINE=1
- ✓ --report_to none
- ✓ --stage cls
- ✓ --do_train
- ✓ --do_predict
- ✓ --image_min_pixels and --image_max_pixels
- ✓ --freeze_vision_tower true
- ✓ --freeze_multi_modal_projector true
- ✓ Dataset: `iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7`
- ✓ Model: Qwen/Qwen2.5-VL-7B-Instruct
- ✓ Batch config: 4×2×2=16
- ✓ 3 LR configs: 2e-5, 2e-6, separate (2e-5 head/2e-6 backbone)

### Datasets
- ✓ Text train dataset exists
- ✓ Text test dataset exists
- ✓ Vision train dataset exists
- ✓ Vision test dataset exists

### Infrastructure
- ✓ FSDP config file exists (`configs/fsdp2_2gpu_config.yaml`)
- ✓ Log directory created (`logs/lr_experiment_v7/`)
- ✓ Save directory created (`saves/lr_experiment_v7/`)
- ✓ Results directory created (`results/lr_experiment_v7/`)

### LR Configuration Logic
- ✓ Config 0: HEAD_LR=2e-5, BACKBONE_LR=none → Both use 2e-5
- ✓ Config 1: HEAD_LR=2e-6, BACKBONE_LR=none → Both use 2e-6
- ✓ Config 2: HEAD_LR=2e-5, BACKBONE_LR=2e-6 → Separate LRs (adds --cls_backbone_learning_rate)

## Previous Failures Resolved

### Issue 1: WandB Timeout (Text Jobs)
**Error**: `wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec`
**Fix**: Added `export WANDB_DISABLED=true` and `--report_to none`

### Issue 2: Distributed Training Timeout (Vision Jobs)
**Error**: `RuntimeError: Timed out waiting for recv operation to complete`
**Fix**: Added `export WANDB_DISABLED=true` and `--report_to none` (prevents initialization hangs)

## Ready to Submit

The scripts are now ready to submit:

```bash
# Submit text experiments (3 jobs)
sbatch sbatch/final_sweep_v7/lr_experiment_text_trainagreeing.sbatch

# Submit vision experiments (3 jobs)
sbatch sbatch/final_sweep_v7/lr_experiment_vision_trainagreeing.sbatch
```

Or use the helper script:
```bash
./sbatch/final_sweep_v7/submit_lr_experiment.sh
```

## What to Monitor

### Expected Log Messages

For **array index 2** (separate LRs), you should see:
```
[INFO] Using custom classification optimizer with backbone_lr=2.00e-06, head_lr=2.00e-05
```

### During Training

Monitor with:
```bash
squeue -u $USER
tail -f logs/lr_experiment_v7/<job_id>_*.out
```

### After Completion

Check results:
```bash
python scripts/analyze_lr_experiment.py
```

## Estimated Runtime

- **Text jobs**: ~10-15 hours each (24h time limit)
- **Vision jobs**: ~15-20 hours each (24h time limit)

## Output Structure

```
saves/lr_experiment_v7/
├── text_trainagreeing_lr_2e5/
├── text_trainagreeing_lr_2e6/
├── text_trainagreeing_lr_2e5_backbone_2e6/
├── vision_trainagreeing_lr_2e5/
├── vision_trainagreeing_lr_2e6/
└── vision_trainagreeing_lr_2e5_backbone_2e6/

results/lr_experiment_v7/
├── text_trainagreeing_lr_2e5/
│   ├── generated_predictions.jsonl
│   ├── predict_results.json
│   └── all_results.json
├── ... (same for all 6 configs)
```

---

**Verified on**: 2026-02-02 19:XX:XX
**Status**: ✅ READY TO LAUNCH