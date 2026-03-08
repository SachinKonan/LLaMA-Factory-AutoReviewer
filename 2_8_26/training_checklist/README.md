# Training Checklist: SFT and DPO with Checklist-Optimized Reviews

This directory contains scripts and configurations for training Qwen2.5-3B models on ICLR review data filtered by checklist optimization scores.

## Overview

**Goal**: Train models that generate high-quality reviews by learning from reviews that meet key evaluation criteria.

**Approach**:
1. Apply checklist to ALL individual reviews (4,682 reviews from test set)
2. **SFT**: Train on top 20% of reviews by checkmark score
3. **DPO**: Train on contrastive pairs (high vs low quality reviews) from accepted papers

## Directory Structure

```
2_8_26/training_checklist/
├── scripts/
│   ├── evaluate_reviews.py         # Apply checklist to reviews
│   ├── generate_training_data.py   # Generate SFT/DPO datasets
│   └── validate_datasets.py        # Validate dataset formats
├── configs/
│   ├── sft_base.yaml               # Base SFT configuration
│   ├── dpo_base.yaml               # Base DPO configuration
│   └── generated/                  # Auto-generated configs (by SLURM)
├── sbatch/
│   └── train_pipeline.sbatch       # SLURM job array (6 jobs)
├── data/                            # Generated datasets
│   ├── review_evaluations.jsonl    # Checklist scores per review
│   ├── sft_{clean,clean_images,vision}_{train,val}/
│   └── dpo_{clean,clean_images,vision}_{train,val}/
├── checkpoints/                     # Model checkpoints
│   ├── sft_{clean,clean_images,vision}/
│   └── dpo_{clean,clean_images,vision}/
└── logs/                            # Training logs
```

## Prerequisites

1. **Checklist optimization complete**: You need `optimal_checklist.json` from the checklist_optimization experiment
2. **Virtual environment**: Activate `.venv` for training (has LLaMA Factory dependencies)
3. **vLLM environment**: Use `.venv_vllm_inf` or `.vllm` for review evaluation (has vLLM)
4. **GPU resources**: 8× A100/H100 80GB GPUs for full fine-tuning with DeepSpeed ZeRO-3

## Usage

### Step 1: Run Checklist Optimization (if not complete)

```bash
cd 2_8_26/checklist_optimization
sbatch --array=3-4 run_pipeline.sbatch  # Stages 4-5
```

Wait for `optimal_checklist.json` to be generated.

### Step 2: Evaluate All Reviews with Checklist

Apply the optimal checklist to all 4,682 individual reviews:

```bash
# Activate vLLM environment
source .vllm/bin/activate

# Run evaluation (requires 2× L40 48GB GPUs)
python 2_8_26/training_checklist/scripts/evaluate_reviews.py \
    --checklist 2_8_26/checklist_optimization/data/optimal_checklist.json \
    --output 2_8_26/training_checklist/data/review_evaluations.jsonl \
    --batch_size 64

# Expected runtime: 4-6 hours
```

**Output**: `review_evaluations.jsonl` with checkmark scores for each review

**Debug mode** (test with 100 reviews):
```bash
python 2_8_26/training_checklist/scripts/evaluate_reviews.py \
    --checklist 2_8_26/checklist_optimization/data/optimal_checklist.json \
    --output 2_8_26/training_checklist/data/review_evaluations.jsonl \
    --debug
```

### Step 3: Generate Training Datasets

Create SFT (top 20%) and DPO (contrastive pairs) datasets:

```bash
python 2_8_26/training_checklist/scripts/generate_training_data.py \
    --review_evaluations 2_8_26/training_checklist/data/review_evaluations.jsonl \
    --output_dir 2_8_26/training_checklist/data
```

**Outputs**:
- SFT: ~842 train, ~94 val samples per modality
- DPO: ~360-720 train, ~40-80 val pairs per modality
- All 3 modalities: clean, clean_images, vision

### Step 4: Validate Datasets

```bash
python 2_8_26/training_checklist/scripts/validate_datasets.py \
    --data_dir 2_8_26/training_checklist/data \
    --stage all  # or sft or dpo
```

**Checks**:
- ✅ Format correctness (conversations, chosen/rejected)
- ✅ Metadata completeness
- ✅ Score distributions
- ✅ DPO contrast requirements (rating_diff >= 2, checkmark_diff >= 2)

### Step 5: Submit Training Jobs

```bash
# Switch to training environment
source .venv/bin/activate

# Submit all 6 jobs (3 SFT + 3 DPO)
sbatch --array=0-5 2_8_26/training_checklist/sbatch/train_pipeline.sbatch
```

**Job Array**:
- Job 0: SFT clean (text-only)
- Job 1: SFT clean_images (text + images)
- Job 2: SFT vision (images-focused)
- Job 3: DPO clean (waits for job 0)
- Job 4: DPO clean_images (waits for job 1)
- Job 5: DPO vision (waits for job 2)

**Expected Timeline**:
- SFT: 12-16 hours per modality (parallel)
- DPO: 8-12 hours per modality (sequential after SFT)
- **Total**: ~24-32 hours

## Monitoring

### Check job status
```bash
squeue -u $USER
```

### Monitor training logs
```bash
tail -f 2_8_26/training_checklist/logs/train_*.out
```

### Check checkpoints
```bash
ls -lh 2_8_26/training_checklist/checkpoints/*/config.json
```

## Model Details

### Text Model (clean modality)
- **Model**: Qwen/Qwen2.5-3B-Instruct
- **Template**: qwen
- **Context length**: 28,000 tokens

### Vision Models (clean_images, vision modalities)
- **Model**: Qwen/Qwen2.5-VL-3B-Instruct
- **Template**: qwen2_vl
- **Context length**: 16,384 tokens (reduced for image tokens)
- **Image config**: max_pixels=250880, min_pixels=196

## Training Configurations

### SFT (Supervised Fine-Tuning)
- **Learning rate**: 1e-5
- **Epochs**: 3
- **Batch size**: 1 per device × 16 gradient accumulation = 16 effective
- **Optimizer**: AdamW with cosine schedule
- **Precision**: BF16

### DPO (Direct Preference Optimization)
- **Learning rate**: 5e-6 (lower than SFT)
- **Epochs**: 2 (fewer than SFT)
- **Beta**: 0.1
- **Loss**: sigmoid (standard DPO)
- **Reference model**: Auto-created from SFT checkpoint

## Dataset Registration

All 12 datasets are registered in `data/dataset_info.json`:

**SFT datasets**:
- `training_checklist_sft_{clean,clean_images,vision}_{train,val}`

**DPO datasets**:
- `training_checklist_dpo_{clean,clean_images,vision}_{train,val}`

## Troubleshooting

### Issue: review_evaluations.jsonl not found
**Solution**: Run Step 2 (evaluate_reviews.py) first

### Issue: DPO job waiting too long
**Solution**: Check SFT checkpoint exists at `checkpoints/sft_{modality}/config.json`

### Issue: Out of memory during training
**Solution**:
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Ensure DeepSpeed ZeRO-3 is enabled

### Issue: Dataset validation errors
**Solution**: Check review_evaluations.jsonl format and re-run generate_training_data.py

## Next Steps

After training completes:

1. **Evaluate models**: Run inference on test set with trained checkpoints
2. **Compare baselines**: Compare SFT vs DPO vs base model performance
3. **Analyze improvements**: Measure accuracy, F1, and review quality metrics

## Files Created

This implementation created:
1. ✅ `scripts/evaluate_reviews.py` - Apply checklist to reviews
2. ✅ `scripts/generate_training_data.py` - Generate SFT/DPO datasets
3. ✅ `scripts/validate_datasets.py` - Validate dataset formats
4. ✅ `configs/sft_base.yaml` - SFT configuration template
5. ✅ `configs/dpo_base.yaml` - DPO configuration template
6. ✅ `sbatch/train_pipeline.sbatch` - SLURM training script
7. ✅ Updated `data/dataset_info.json` - Registered 12 new datasets

## References

- **Checklist optimization**: `2_8_26/checklist_optimization/README.md`
- **LLaMA Factory docs**: https://github.com/hiyouga/LLaMA-Factory
- **DPO paper**: https://arxiv.org/abs/2305.18290
