# Learning Rate Experiment Documentation

## Overview

This experiment tests 3 different learning rate configurations on the trainagreeing dataset for both text and vision models to identify optimal learning rate strategies for classification training.

## Experiment Design

### Datasets
- **Text**: `iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7`
- **Vision**: `iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7`

### Models
- **Text**: Qwen/Qwen2.5-7B-Instruct
- **Vision**: Qwen/Qwen2.5-VL-7B-Instruct

### Learning Rate Configurations (3 total)
1. **LR=2e-5 (uniform)**: Both backbone and classification head use 2e-5
2. **LR=2e-6 (uniform)**: Both backbone and classification head use 2e-6
3. **LR=2e-5/2e-6 (separate)**: Head uses 2e-5, backbone uses 2e-6

### Training Configuration
- **Stage**: `cls` (classification, not SFT)
- **Batch size**: 16 (effective across all GPUs and gradient accumulation)
  - Text: 2 GPUs × 2 per_device × 4 grad_accum = 16
  - Vision: 2 GPUs × 4 per_device × 2 grad_accum = 16
- **Epochs**: 6
- **Scheduler**: Cosine with 5-step warmup
- **Evaluation**: Every epoch on test set
- **Precision**: BF16

### Total Experiments
6 jobs total (2 model types × 3 LR configs)

## Directory Structure

```
sbatch/final_sweep_v7/
├── lr_experiment_text_trainagreeing.sbatch    # Text experiments (array 0-2)
├── lr_experiment_vision_trainagreeing.sbatch  # Vision experiments (array 0-2)
├── submit_lr_experiment.sh                    # Convenience submission script
└── LR_EXPERIMENT_README.md                    # This file

logs/lr_experiment_v7/
├── <text_job_id>_0.out     # Text LR=2e-5 stdout
├── <text_job_id>_0.err     # Text LR=2e-5 stderr
├── <text_job_id>_1.out     # Text LR=2e-6 stdout
├── <text_job_id>_1.err     # Text LR=2e-6 stderr
├── <text_job_id>_2.out     # Text separate LRs stdout
├── <text_job_id>_2.err     # Text separate LRs stderr
└── ... (vision logs)

saves/lr_experiment_v7/
├── text_trainagreeing_lr_2e5/
│   ├── checkpoint-epoch1/
│   ├── checkpoint-epoch2/
│   ├── ...
│   └── cls_head.safetensors
├── text_trainagreeing_lr_2e6/
├── text_trainagreeing_lr_2e5_backbone_2e6/
├── vision_trainagreeing_lr_2e5/
├── vision_trainagreeing_lr_2e6/
└── vision_trainagreeing_lr_2e5_backbone_2e6/

results/lr_experiment_v7/
├── text_trainagreeing_lr_2e5/
│   ├── generated_predictions.jsonl   # Predictions with logits and probabilities
│   ├── predict_results.json          # Test metrics (accuracy, F1, etc.)
│   └── all_results.json              # All epoch results
├── text_trainagreeing_lr_2e6/
├── text_trainagreeing_lr_2e5_backbone_2e6/
├── vision_trainagreeing_lr_2e5/
├── vision_trainagreeing_lr_2e6/
└── vision_trainagreeing_lr_2e5_backbone_2e6/
```

## Submission

### Option 1: Use the submission script (recommended)
```bash
./sbatch/final_sweep_v7/submit_lr_experiment.sh
```

### Option 2: Submit individually
```bash
# Submit text experiments
sbatch sbatch/final_sweep_v7/lr_experiment_text_trainagreeing.sbatch

# Submit vision experiments
sbatch sbatch/final_sweep_v7/lr_experiment_vision_trainagreeing.sbatch
```

## Monitoring

### Check job status
```bash
squeue -u $USER
```

### View live logs
```bash
# Text experiments
tail -f logs/lr_experiment_v7/<job_id>_0.out  # LR=2e-5
tail -f logs/lr_experiment_v7/<job_id>_1.out  # LR=2e-6
tail -f logs/lr_experiment_v7/<job_id>_2.out  # Separate LRs

# Vision experiments (similar)
tail -f logs/lr_experiment_v7/<job_id>_*.out
```

### Monitor for separate LR usage
For array index 2 (separate LRs), you should see in the logs:
```
[INFO] Using custom classification optimizer with backbone_lr=2.00e-06, head_lr=2.00e-05
```

## Results Analysis

### Quick comparison of test accuracy
```bash
for dir in results/lr_experiment_v7/*/; do
    echo "$(basename $dir):"
    jq '.predict_accuracy' "$dir/predict_results.json" 2>/dev/null || echo "Not available yet"
done
```

### Extract all metrics
```bash
# View complete results for a specific configuration
jq '.' results/lr_experiment_v7/text_trainagreeing_lr_2e5/predict_results.json

# Compare F1 scores
for dir in results/lr_experiment_v7/*/; do
    echo "$(basename $dir):"
    jq '.predict_f1' "$dir/predict_results.json" 2>/dev/null || echo "Not available yet"
done
```

### Analyze predictions
Each `generated_predictions.jsonl` contains per-sample predictions with:
- `logits`: Raw model outputs
- `probabilities`: Softmax probabilities
- `predicted_label`: Predicted class
- `true_label`: Ground truth
- All metadata from the dataset

## Resource Allocation

### Text Experiments
- **Time limit**: 24 hours per job
- **Memory**: 200GB
- **GPUs**: 2 × H200 (ailab partition)
- **CPUs**: 10 cores

### Vision Experiments
- **Time limit**: 36 hours per job
- **Memory**: 350GB
- **GPUs**: 2 × H200 (ailab partition)
- **CPUs**: 10 cores

## Key Implementation Details

### Separate Learning Rate Feature
- Implemented via `--cls_backbone_learning_rate` parameter
- Only valid for `--stage cls`
- When specified:
  - Classification head uses `--learning_rate`
  - Backbone (language model) uses `--cls_backbone_learning_rate`
- When not specified:
  - Both use `--learning_rate` (uniform)

### Vision Model Freezing
- Vision tower: **Frozen** (`--freeze_vision_tower true`)
- Multimodal projector: **Frozen** (`--freeze_multi_modal_projector true`)
- Only language model and classification head are trained

### Evaluation Strategy
- **During training**: Evaluate on test set every epoch (via `--eval_dataset`)
- **After training**: Generate predictions on test set (`--do_predict`)
- **Checkpointing**: Save every epoch, keep last 3 checkpoints

## Expected Outcomes

This experiment will help determine:
1. Whether a slower learning rate (2e-6) improves stability/performance
2. Whether a faster learning rate (2e-5) converges faster without overfitting
3. Whether separate learning rates (faster head, slower backbone) provides the best balance

Results will inform the optimal learning rate strategy for future classification experiments.

## Troubleshooting

### Common Issues

**Dataset not found**
```bash
# Verify datasets exist
ls data/iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7_train/
ls data/iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7_train/
```

**Out of memory**
- Text: 200GB should be sufficient
- Vision: 350GB should be sufficient
- If OOM occurs, reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps` proportionally

**Job timeout**
- Text: 24 hours should be sufficient for 6 epochs
- Vision: 36 hours should be sufficient for 6 epochs
- If timeout occurs, reduce `num_train_epochs` or request more time

**Port conflicts**
- Each array task uses a different port: 29700 + array_index
- This prevents conflicts when multiple jobs run simultaneously

## Contact

For questions or issues, check the main project documentation or logs.
