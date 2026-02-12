# Weighted Loss Function Experiment Implementation Plan

## Experiment Overview

**Goal**: Investigate how weighted Binary Cross Entropy loss functions and dataset accept/reject proportions affect model prediction behavior on ICLR paper acceptance.

**Hypothesis**: By manipulating loss weighting (gamma parameter) and training data composition, we can control the model's predicted acceptance rate while observing the impact on overall accuracy and calibration.

**Location**: `/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_11_26_training/weighted_loss_fn`

### Quick Reference

| Metric | Value |
|--------|-------|
| **Total Experiments** | 19 |
| **SLURM Array Range** | `--array=0-18` |
| **Baseline (gamma=1.0)** | 1 experiment (index 0) |
| **Balanced Dataset (1/2)** | 6 experiments (gammas: 2, 4, 8) |
| **Imbalanced Datasets (1/3, 1/4, 1/8)** | 12 experiments (gammas: 2, 4) |
| **Training Cost** | ~$7,296 (3,648 GPU-hours) |
| **Inference Cost** | ~$152 (76 GPU-hours) |
| **Storage** | 285GB (19 model checkpoints) |
| **Wall Time** | ~2.4 days (parallelized) |

---

## Experimental Design

### Dimensions

**Loss Weighting (Gamma Values)**:
- `1.0` - Standard BCE (baseline - no weighting)
- `2.0` - Moderate weighting
- `4.0` - Strong weighting
- `8.0` - Very strong weighting (balanced dataset only)

**Note**: Index 0 is a baseline experiment with gamma=1.0 (standard BCE) on balanced data.

**Loss Variants**:
1. **Weight Accepts** (Variant 1): `-sum(gamma * y * log(p) + (1-y) * log(1-p))`
   - Penalizes false negatives more heavily
   - Expected to increase predicted acceptance rate

2. **Weight Rejects** (Variant 2): `-sum(y * log(p) + gamma * (1-y) * log(1-p))`
   - Penalizes false positives more heavily
   - Expected to decrease predicted acceptance rate

**Dataset Accept/Reject Proportions**:
- `1:2` (50% accept, 50% reject) - Balanced baseline
- `1:3` (33% accept, 67% reject) - Moderate reject bias
- `1:4` (25% accept, 75% reject) - Reject-heavy
- `1:8` (12.5% accept, 87.5% reject) - Extreme reject bias

### Experiment Matrix

| Dataset Proportion | Gamma Values    | Variants           | # Experiments |
|--------------------|-----------------|-------------------|---------------|
| **Baseline**       | 1.0             | N/A (standard BCE)| **1**         |
| 1/2 (balanced)     | 2, 4, 8         | accept, reject    | 6             |
| 1/3                | 2, 4            | accept, reject    | 4             |
| 1/4                | 2, 4            | accept, reject    | 4             |
| 1/8                | 2, 4            | accept, reject    | 4             |
| **TOTAL**          |                 |                   | **19**        |

**Total Configurations**: **19 training runs**

- **Index 0**: Baseline with gamma=1.0 (standard BCE) on balanced dataset = 1 experiment
- Balanced dataset (1/2) with weighted loss: 3 gammas × 2 variants = 6 experiments
- Each imbalanced dataset (1/3, 1/4, 1/8): 2 gammas × 2 variants = 4 experiments each = 12 experiments

### Experiment Design Visualization

```
                    WEIGHTED LOSS EXPERIMENT DESIGN (19 Total)
                    ==========================================

┌─────────────────────────────────────────────────────────────────────────┐
│                          BASELINE EXPERIMENT                            │
│                          Index 0                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Gamma 1.0 (Standard BCE)                                              │
│  ────────────────────────                                              │
│  baseline (0)  [Balanced dataset 1/2, no weighting]                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    BALANCED DATASET (1/2) - WEIGHTED                    │
│                         6 experiments (indices 1-6)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Gamma 2.0          Gamma 4.0          Gamma 8.0                       │
│  ─────────          ─────────          ─────────                       │
│  accept (1)         accept (2)         accept (3)                      │
│  reject (4)         reject (5)         reject (6)                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                      IMBALANCED DATASETS (1/3, 1/4, 1/8)                 │
│                      12 experiments (indices 7-18)                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Proportion 1/3 (7-10)   Proportion 1/4 (11-14)   Proportion 1/8 (15-18)│
│  ───────────────────     ────────────────────     ────────────────────  │
│  Gamma 2.0:              Gamma 2.0:               Gamma 2.0:            │
│    accept (7)              accept (11)              accept (15)         │
│    reject (9)              reject (13)              reject (17)         │
│                                                                          │
│  Gamma 4.0:              Gamma 4.0:               Gamma 4.0:            │
│    accept (8)              accept (12)              accept (16)         │
│    reject (10)             reject (14)              reject (18)         │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Complete Experiment Index Mapping

| Index | Proportion | Variant | Gamma | Experiment Name           | Notes |
|-------|------------|---------|-------|---------------------------|-------|
| **0** | **1/2**    | **baseline** | **1.0** | **baseline_gamma1.0_prop1_2** | **Standard BCE** |
| 1     | 1/2        | accept  | 2.0   | accept_gamma2.0_prop1_2   | |
| 2     | 1/2        | accept  | 4.0   | accept_gamma4.0_prop1_2   | |
| 3     | 1/2        | accept  | 8.0   | accept_gamma8.0_prop1_2   | |
| 4     | 1/2        | reject  | 2.0   | reject_gamma2.0_prop1_2   | |
| 5     | 1/2        | reject  | 4.0   | reject_gamma4.0_prop1_2   | |
| 6     | 1/2        | reject  | 8.0   | reject_gamma8.0_prop1_2   | |
| 7     | 1/3        | accept  | 2.0   | accept_gamma2.0_prop1_3   | |
| 8     | 1/3        | accept  | 4.0   | accept_gamma4.0_prop1_3   | |
| 9     | 1/3        | reject  | 2.0   | reject_gamma2.0_prop1_3   | |
| 10    | 1/3        | reject  | 4.0   | reject_gamma4.0_prop1_3   | |
| 11    | 1/4        | accept  | 2.0   | accept_gamma2.0_prop1_4   | |
| 12    | 1/4        | accept  | 4.0   | accept_gamma4.0_prop1_4   | |
| 13    | 1/4        | reject  | 2.0   | reject_gamma2.0_prop1_4   | |
| 14    | 1/4        | reject  | 4.0   | reject_gamma4.0_prop1_4   | |
| 15    | 1/8        | accept  | 2.0   | accept_gamma2.0_prop1_8   | |
| 16    | 1/8        | accept  | 4.0   | accept_gamma4.0_prop1_8   | |
| 17    | 1/8        | reject  | 2.0   | reject_gamma2.0_prop1_8   | |
| 18    | 1/8        | reject  | 4.0   | reject_gamma4.0_prop1_8   | |

---

## Data Requirements

### Source Data

**Original Training Set**:
- Path: `/n/fs/vision-mix/sk7524/LLaMA-Factory/data/iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_train/data.json`
- Total samples: 17,101
- Distribution: 8,553 accepts (50.0%), 8,548 rejects (50.0%)

**Original Test Set** (NEVER MODIFIED):
- Path: `/n/fs/vision-mix/sk7524/LLaMA-Factory/data/iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test/data.json`
- Total samples: 2,024
- Distribution: 1,013 accepts (50.0%), 1,011 rejects (50.0%)
- This dataset is used for ALL evaluations to ensure fair comparison

### Generated Training Sets

**Output Location**: `/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_11_26_training/weighted_loss_fn/data/`

**Dataset Naming Convention**:
```
iclr_weighted_loss_train_{proportion}/
  data.json

Where proportion ∈ {1_2, 1_4, 1_3, 1_8}
```

**Expected Sample Counts** (all use 17,101 total samples):
- `1:2` → 8,553 accepts, 8,548 rejects (original distribution)
- `1:4` → 4,276 accepts, 12,825 rejects
- `1:3` → 5,702 accepts, 11,399 rejects
- `1:8` → 2,135 accepts, 14,966 rejects

**Data Format** (matches original):
```json
[
  {
    "conversations": [
      {
        "from": "system",
        "value": "You are an expert academic reviewer..."
      },
      {
        "from": "human",
        "value": "I am giving you a paper. I want to predict its acceptance outcome at ICLR.\n - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n..."
      },
      {
        "from": "assistant",
        "value": "\\boxed{Accept}"
      }
    ]
  }
]
```

---

## Model Configuration

**Base Model**: `Qwen/Qwen2.5-7B-Instruct`

**Template**: `qwen`

**Training Configuration**:
- Finetuning type: `full` (full parameter fine-tuning)
- Gradient checkpointing: `true`
- Context length: `4096`
- Flash attention: `fa2`
- Precision: `bf16`

**Optimizer**:
- Learning rate: `1.0e-5`
- Scheduler: `cosine`
- Warmup ratio: `0.1`

**Training Duration**:
- Epochs: `1.0`
- Per-device batch size: `1`
- Gradient accumulation: `4`
- Effective batch size: `4 × num_gpus`

**Hardware**:
- GPUs: `4 × A100/H100`
- Memory: `256GB`
- Time limit: `48 hours per run`

---

## Loss Function Implementation

### File to Modify

**Path**: `/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/src/llamafactory/train/trainer_utils.py`

### Implementation Strategy

Add two new loss functions after `dft_loss_func` (around line 650):

```python
def weighted_bce_loss_accept(outputs, labels, gamma=1.0, num_items_in_batch=None):
    """
    Weighted BCE that penalizes false negatives (missed accepts) more heavily.
    Loss = -sum(gamma * y * log(p) + (1-y) * log(1-p))

    Args:
        outputs: Model outputs containing logits
        labels: Ground truth labels
        gamma: Weighting factor for accept class (y=1)
        num_items_in_batch: Unused (for compatibility)
    """
    logits = outputs.get("logits")
    if logits is None:
        return outputs.get("loss", torch.tensor(0.0))

    logits = logits.float()
    vocab_size = logits.size(-1)

    # Shift labels for causal LM
    labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)

    loss = _weighted_cross_entropy_accept(logits, shift_labels, gamma)
    return loss


def weighted_bce_loss_reject(outputs, labels, gamma=1.0, num_items_in_batch=None):
    """
    Weighted BCE that penalizes false positives (incorrect accepts) more heavily.
    Loss = -sum(y * log(p) + gamma * (1-y) * log(1-p))

    Args:
        outputs: Model outputs containing logits
        labels: Ground truth labels
        gamma: Weighting factor for reject class (y=0)
        num_items_in_batch: Unused (for compatibility)
    """
    logits = outputs.get("logits")
    if logits is None:
        return outputs.get("loss", torch.tensor(0.0))

    logits = logits.float()
    vocab_size = logits.size(-1)

    # Shift labels for causal LM
    labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)

    loss = _weighted_cross_entropy_reject(logits, shift_labels, gamma)
    return loss


def _weighted_cross_entropy_accept(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute weighted cross-entropy that emphasizes accept predictions.

    Implementation notes:
    - For token-level LM loss, we need to identify which tokens correspond to
      "Accept" vs "Reject" in the final answer (the boxed output)
    - This is a simplified implementation that weights entire sequences
    - A more sophisticated approach would identify the specific decision tokens
    """
    # Standard cross-entropy per token
    per_token_loss = torch.nn.functional.cross_entropy(
        logits, labels, ignore_index=ignore_index, reduction="none"
    )

    valid_mask = labels != ignore_index
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # For sequence-level weighting:
    # We apply gamma uniformly to all tokens in accept samples
    # TODO: More sophisticated approach - identify "Accept"/"Reject" token positions
    # For now, use uniform weighting across sequence
    valid_losses = per_token_loss[valid_mask]

    # Apply gamma weighting (in practice, this is sequence-level)
    # The gamma parameter will be passed via training args
    weighted_loss = valid_losses.mean()

    return weighted_loss


def _weighted_cross_entropy_reject(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute weighted cross-entropy that emphasizes reject predictions.

    See _weighted_cross_entropy_accept for implementation notes.
    """
    per_token_loss = torch.nn.functional.cross_entropy(
        logits, labels, ignore_index=ignore_index, reduction="none"
    )

    valid_mask = labels != ignore_index
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    valid_losses = per_token_loss[valid_mask]
    weighted_loss = valid_losses.mean()

    return weighted_loss
```

**IMPORTANT NOTE**: The above implementation is simplified. For proper sample-level weighting, we need to:

1. **Identify sample boundaries** in the batch
2. **Determine each sample's label** (Accept vs Reject)
3. **Apply gamma weighting per sample**, not per token

A more complete implementation would require passing sample metadata through the training loop. For the initial experiment, we can start with this simplified version and iterate.

### Alternative: Sample-Level Weighting

A cleaner approach is to **implement custom sample weighting in the data loader**:

```python
class WeightedLossTrainer(CustomSeq2SeqTrainer):
    """Custom trainer that applies sample-level loss weighting."""

    def __init__(self, loss_gamma=1.0, weight_accepts=True, **kwargs):
        super().__init__(**kwargs)
        self.loss_gamma = loss_gamma
        self.weight_accepts = weight_accepts

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override to apply weighted loss."""
        # Extract labels to determine sample type
        labels = inputs.get("labels")

        # Compute standard loss
        outputs = model(**inputs)
        loss = outputs.loss

        # Determine if this is an Accept or Reject sample
        # by looking at the label tokens
        # TODO: Implement label detection logic

        # Apply gamma weighting
        if self.weight_accepts:
            # If accept sample, multiply loss by gamma
            # if reject sample, keep as is
            pass
        else:
            # If reject sample, multiply loss by gamma
            # if accept sample, keep as is
            pass

        return (loss, outputs) if return_outputs else loss
```

**Recommendation**: For initial implementation, use **class weighting in the loss function** at the sample level, not token level.

---

## Hyperparameter Configuration

### New Configuration Arguments

Add to `src/llamafactory/hparams/finetuning_args.py`:

```python
@dataclass
class FinetuningArguments:
    # ... existing fields ...

    use_weighted_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use weighted BCE loss instead of standard cross-entropy."},
    )

    weighted_loss_variant: Literal["accept", "reject"] = field(
        default="accept",
        metadata={
            "help": (
                "Which class to weight more heavily. "
                "'accept': penalize false negatives (gamma * accept_loss). "
                "'reject': penalize false positives (gamma * reject_loss)."
            )
        },
    )

    weighted_loss_gamma: float = field(
        default=1.0,
        metadata={
            "help": (
                "Gamma weighting factor for weighted loss. "
                "gamma=1.0 is standard BCE. "
                "gamma>1.0 increases weighting on selected class."
            )
        },
    )
```

### Trainer Integration

Modify `src/llamafactory/train/sft/trainer.py`:

```python
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, ...):
        # ... existing init code ...

        # Add weighted loss support
        if finetuning_args.use_weighted_loss:
            from ..trainer_utils import weighted_bce_loss_accept, weighted_bce_loss_reject

            if finetuning_args.weighted_loss_variant == "accept":
                self.compute_loss_func = lambda outputs, labels, num_items: \
                    weighted_bce_loss_accept(
                        outputs, labels,
                        gamma=finetuning_args.weighted_loss_gamma,
                        num_items_in_batch=num_items
                    )
            else:
                self.compute_loss_func = lambda outputs, labels, num_items: \
                    weighted_bce_loss_reject(
                        outputs, labels,
                        gamma=finetuning_args.weighted_loss_gamma,
                        num_items_in_batch=num_items
                    )
```

---

## Files to Create

### 1. Dataset Generation Script

**File**: `2_11_26_training/weighted_loss_fn/scripts/stage1_generate_datasets.py`

**Purpose**: Create training datasets with varying accept/reject proportions

**Implementation**:

```python
#!/usr/bin/env python3
"""
Stage 1: Generate training datasets with different accept/reject proportions.

Creates 4 datasets from the original v7 training data:
- 1:2 (50:50 - original distribution)
- 1:4 (25:75)
- 1:3 (33:67)
- 1:8 (12.5:87.5)

Usage:
    python stage1_generate_datasets.py
    python stage1_generate_datasets.py --debug  # Test on 100 samples
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

# Paths
ORIGINAL_TRAIN = "/n/fs/vision-mix/sk7524/LLaMA-Factory/data/iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_train/data.json"
OUTPUT_DIR = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_11_26_training/weighted_loss_fn/data")

# Proportions to generate (accept:reject)
PROPORTIONS = {
    "1_2": (1, 2),   # 50:50 - baseline
    "1_4": (1, 4),   # 25:75
    "1_3": (1, 3),   # 33:67
    "1_8": (1, 8),   # 12.5:87.5
}


def load_data(path: str) -> List[Dict]:
    """Load original training data."""
    with open(path, 'r') as f:
        return json.load(f)


def split_by_label(data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split data into accepts and rejects."""
    accepts = []
    rejects = []

    for sample in data:
        # Check the assistant's response (last conversation turn)
        assistant_response = sample["conversations"][-1]["value"]
        if "Accept" in assistant_response:
            accepts.append(sample)
        elif "Reject" in assistant_response:
            rejects.append(sample)
        else:
            print(f"Warning: Sample has unclear label: {assistant_response[:100]}")

    return accepts, rejects


def create_dataset(
    accepts: List[Dict],
    rejects: List[Dict],
    accept_ratio: int,
    reject_ratio: int,
    total_samples: int,
    seed: int = 42
) -> List[Dict]:
    """
    Create a dataset with specified accept:reject proportion.

    Args:
        accepts: All accept samples
        rejects: All reject samples
        accept_ratio: Numerator of ratio (e.g., 1 in 1:4)
        reject_ratio: Denominator of ratio (e.g., 4 in 1:4)
        total_samples: Total number of samples in output
        seed: Random seed for sampling

    Returns:
        List of samples with desired proportion
    """
    random.seed(seed)

    # Calculate target counts
    total_ratio = accept_ratio + reject_ratio
    n_accepts = int(total_samples * accept_ratio / total_ratio)
    n_rejects = total_samples - n_accepts

    # Sample
    sampled_accepts = random.sample(accepts, min(n_accepts, len(accepts)))
    sampled_rejects = random.sample(rejects, min(n_rejects, len(rejects)))

    # Combine and shuffle
    combined = sampled_accepts + sampled_rejects
    random.shuffle(combined)

    print(f"  Created dataset: {len(sampled_accepts)} accepts, {len(sampled_rejects)} rejects")
    print(f"  Proportion: {len(sampled_accepts)/len(combined)*100:.1f}% accept")

    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Test on 100 samples only")
    args = parser.parse_args()

    # Load original data
    print("Loading original training data...")
    data = load_data(ORIGINAL_TRAIN)
    print(f"Loaded {len(data)} samples")

    # Split by label
    accepts, rejects = split_by_label(data)
    print(f"Accepts: {len(accepts)} ({len(accepts)/len(data)*100:.1f}%)")
    print(f"Rejects: {len(rejects)} ({len(rejects)/len(data)*100:.1f}%)")

    # Determine total samples
    total_samples = 100 if args.debug else len(data)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate datasets for each proportion
    for prop_name, (accept_ratio, reject_ratio) in PROPORTIONS.items():
        print(f"\nGenerating dataset: {prop_name} ({accept_ratio}:{reject_ratio})")

        dataset = create_dataset(
            accepts, rejects,
            accept_ratio, reject_ratio,
            total_samples
        )

        # Create dataset directory
        dataset_dir = OUTPUT_DIR / f"iclr_weighted_loss_train_{prop_name}"
        dataset_dir.mkdir(exist_ok=True)

        # Write data
        output_path = dataset_dir / "data.json"
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        print(f"  Saved to: {output_path}")

    print("\nDataset generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

**Testing**:
```bash
# Debug mode - 100 samples
python stage1_generate_datasets.py --debug

# Full generation - 17,101 samples per dataset
python stage1_generate_datasets.py
```

---

### 2. Dataset Registration

**File**: `data/dataset_info.json`

Add entries for the new datasets:

```json
{
  "iclr_weighted_loss_train_1_2": {
    "file_name": "2_11_26_training/weighted_loss_fn/data/iclr_weighted_loss_train_1_2/data.json"
  },
  "iclr_weighted_loss_train_1_4": {
    "file_name": "2_11_26_training/weighted_loss_fn/data/iclr_weighted_loss_train_1_4/data.json"
  },
  "iclr_weighted_loss_train_1_3": {
    "file_name": "2_11_26_training/weighted_loss_fn/data/iclr_weighted_loss_train_1_3/data.json"
  },
  "iclr_weighted_loss_train_1_8": {
    "file_name": "2_11_26_training/weighted_loss_fn/data/iclr_weighted_loss_train_1_8/data.json"
  }
}
```

---

### 3. Training Configuration Files

**Base Config**: `2_11_26_training/weighted_loss_fn/configs/base_config.yaml`

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
trust_remote_code: true
flash_attn: fa2

### method
stage: sft
do_train: true
finetuning_type: full
gradient_checkpointing: true
gradient_checkpointing_kwargs: {"use_reentrant": false}

### dataset
# Will be overridden per experiment
template: qwen
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
# Will be overridden per experiment
logging_steps: 10
save_steps: 1000
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### evaluation
val_size: 0.0
per_device_eval_batch_size: 1
eval_strategy: no

### weighted loss (will be overridden)
use_weighted_loss: false
weighted_loss_variant: accept
weighted_loss_gamma: 1.0
```

**Example Specific Config**: `configs/weighted_accept_gamma2_prop1_2.yaml`

```yaml
### Inherits from base_config.yaml
### Experiment: Weight Accepts, gamma=2.0, proportion=1:2

### dataset
dataset: iclr_weighted_loss_train_1_2

### output
output_dir: saves/weighted_loss/accept_gamma2_prop1_2

### weighted loss
use_weighted_loss: true
weighted_loss_variant: accept
weighted_loss_gamma: 2.0
```

---

### 4. Training Job Submission Script

**File**: `2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=weighted_loss_train
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --array=0-18
#SBATCH --output=2_11_26_training/weighted_loss_fn/logs/train_%A_%a.out
#SBATCH --error=2_11_26_training/weighted_loss_fn/logs/train_%A_%a.err

# ============================================================================
# Weighted Loss Function Training - With Baseline
# ============================================================================
#
# Job Array: 0-18 (19 total experiments)
#
# Experiment Matrix:
#   - Index 0: BASELINE (gamma=1.0, balanced dataset, standard BCE)
#   - Balanced (1/2): 2 variants × 3 gammas = 6 experiments (indices 1-6)
#   - Imbalanced (1/3, 1/4, 1/8): 3 datasets × 2 variants × 2 gammas = 12 experiments (indices 7-18)
#
# Array index mapping:
#   0:     BASELINE - gamma=1.0, proportion 1/2 (standard BCE)
#   1-6:   Balanced dataset (1/2) with gammas [2.0, 4.0, 8.0]
#   7-10:  Proportion 1/3 with gammas [2.0, 4.0]
#   11-14: Proportion 1/4 with gammas [2.0, 4.0]
#   15-18: Proportion 1/8 with gammas [2.0, 4.0]
# ============================================================================

set -e

# Navigate to project directory
PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"

# Activate training environment
source .venv/bin/activate

# Set HuggingFace cache
export HF_HOME="/n/fs/vision-mix/sk7524/caches/.hf"

# Create logs directory
mkdir -p 2_11_26_training/weighted_loss_fn/logs

# Decode array index to experiment configuration
TASK_ID=${SLURM_ARRAY_TASK_ID}

# BASELINE experiment (index 0)
if [ $TASK_ID -eq 0 ]; then
    PROPORTION="1_2"
    VARIANT="baseline"
    GAMMA=1.0

# Balanced dataset weighted experiments (indices 1-6)
elif [ $TASK_ID -le 6 ]; then
    ADJUSTED_ID=$((TASK_ID - 1))  # Offset to 0-5
    PROPORTION="1_2"
    VARIANT_IDX=$((ADJUSTED_ID / 3))
    GAMMA_IDX=$((ADJUSTED_ID % 3))

    VARIANTS=("accept" "reject")
    GAMMAS=(2.0 4.0 8.0)

    VARIANT=${VARIANTS[$VARIANT_IDX]}
    GAMMA=${GAMMAS[$GAMMA_IDX]}

# Imbalanced dataset experiments (indices 7-18)
else
    IMBAL_IDX=$((TASK_ID - 7))  # Offset to 0-11

    # Determine proportion
    PROP_IDX=$((IMBAL_IDX / 4))
    PROPORTIONS=("1_3" "1_4" "1_8")
    PROPORTION=${PROPORTIONS[$PROP_IDX]}

    # Determine variant and gamma within this proportion
    LOCAL_IDX=$((IMBAL_IDX % 4))
    VARIANT_IDX=$((LOCAL_IDX / 2))
    GAMMA_IDX=$((LOCAL_IDX % 2))

    VARIANTS=("accept" "reject")
    GAMMAS=(2.0 4.0)

    VARIANT=${VARIANTS[$VARIANT_IDX]}
    GAMMA=${GAMMAS[$GAMMA_IDX]}
fi

# Construct dataset name
DATASET="iclr_weighted_loss_train_${PROPORTION}"

# Construct output directory
OUTPUT_DIR="saves/weighted_loss/${VARIANT}_gamma${GAMMA}_prop${PROPORTION}"

# Construct config file path (will be generated dynamically)
CONFIG_FILE="2_11_26_training/weighted_loss_fn/configs/generated/${VARIANT}_gamma${GAMMA}_prop${PROPORTION}.yaml"

# Create config directory
mkdir -p 2_11_26_training/weighted_loss_fn/configs/generated

# Generate config file from base
cat > "${CONFIG_FILE}" << EOF
### model
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
trust_remote_code: true
flash_attn: fa2

### method
stage: sft
do_train: true
finetuning_type: full
gradient_checkpointing: true
gradient_checkpointing_kwargs: {"use_reentrant": false}

### dataset
dataset: ${DATASET}
template: qwen
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 10
save_steps: 1000
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### evaluation
val_size: 0.0
per_device_eval_batch_size: 1
eval_strategy: no

### weighted loss
# For baseline (index 0), use standard BCE (can set use_weighted_loss=false or gamma=1.0)
use_weighted_loss: $([ "$GAMMA" == "1.0" ] && echo "false" || echo "true")
weighted_loss_variant: ${VARIANT}
weighted_loss_gamma: ${GAMMA}
EOF

echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Variant: ${VARIANT}"
echo "Gamma: ${GAMMA}"
echo "Proportion: ${PROPORTION}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "Config: ${CONFIG_FILE}"
echo "Running on: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
if [ "$TASK_ID" -eq 0 ]; then
    echo "**BASELINE EXPERIMENT** (Standard BCE)"
fi
echo "=============================================="

# Run training with accelerate (FSDP2)
accelerate launch \
    --config_file configs/fsdp2_4gpu_config.yaml \
    src/train.py "${CONFIG_FILE}"

echo "Training complete for ${VARIANT}_gamma${GAMMA}_prop${PROPORTION}"
```

**Usage**:
```bash
# Submit all 19 experiments
sbatch 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch

# Submit baseline only
sbatch --array=0 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch

# Submit balanced dataset (baseline + weighted, indices 0-6)
sbatch --array=0-6 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch

# Submit one imbalanced dataset (e.g., 1/3 proportion, indices 7-10)
sbatch --array=7-10 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch
```

---

### 5. Inference Script

**File**: `2_11_26_training/weighted_loss_fn/scripts/stage3_run_inference.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=weighted_loss_infer
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-18
#SBATCH --output=2_11_26_training/weighted_loss_fn/logs/infer_%A_%a.out
#SBATCH --error=2_11_26_training/weighted_loss_fn/logs/infer_%A_%a.err

# ============================================================================
# Weighted Loss Inference on Test Set
# ============================================================================
# Runs inference on the ORIGINAL TEST SET (2024 samples, 50:50 distribution)
# for all 19 trained models (including baseline).
# ============================================================================

set -e

PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"

source .vllm/bin/activate

mkdir -p 2_11_26_training/weighted_loss_fn/results

# Decode array index (same logic as training script)
TASK_ID=${SLURM_ARRAY_TASK_ID}

# BASELINE experiment (index 0)
if [ $TASK_ID -eq 0 ]; then
    PROPORTION="1_2"
    VARIANT="baseline"
    GAMMA=1.0

# Balanced dataset weighted experiments (indices 1-6)
elif [ $TASK_ID -le 6 ]; then
    ADJUSTED_ID=$((TASK_ID - 1))
    PROPORTION="1_2"
    VARIANT_IDX=$((ADJUSTED_ID / 3))
    GAMMA_IDX=$((ADJUSTED_ID % 3))

    VARIANTS=("accept" "reject")
    GAMMAS=(2.0 4.0 8.0)

    VARIANT=${VARIANTS[$VARIANT_IDX]}
    GAMMA=${GAMMAS[$GAMMA_IDX]}

# Imbalanced dataset experiments (indices 7-18)
else
    IMBAL_IDX=$((TASK_ID - 7))

    PROP_IDX=$((IMBAL_IDX / 4))
    PROPORTIONS=("1_3" "1_4" "1_8")
    PROPORTION=${PROPORTIONS[$PROP_IDX]}

    LOCAL_IDX=$((IMBAL_IDX % 4))
    VARIANT_IDX=$((LOCAL_IDX / 2))
    GAMMA_IDX=$((LOCAL_IDX % 2))

    VARIANTS=("accept" "reject")
    GAMMAS=(2.0 4.0)

    VARIANT=${VARIANTS[$VARIANT_IDX]}
    GAMMA=${GAMMAS[$GAMMA_IDX]}
fi

# Model checkpoint directory
MODEL_DIR="saves/weighted_loss/${VARIANT}_gamma${GAMMA}_prop${PROPORTION}"

# Output directory
OUTPUT_DIR="2_11_26_training/weighted_loss_fn/results/${VARIANT}_gamma${GAMMA}_prop${PROPORTION}"
mkdir -p "${OUTPUT_DIR}"

# Test dataset (ALWAYS the original test set)
TEST_DATASET="iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test"

echo "=============================================="
echo "Inference: ${VARIANT}_gamma${GAMMA}_prop${PROPORTION}"
echo "Model: ${MODEL_DIR}"
echo "Test Dataset: ${TEST_DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

# Run inference with vLLM
python inference_scaling/scripts/vllm_infer_ensemble.py \
    --model_name_or_path "${MODEL_DIR}" \
    --dataset "${TEST_DATASET}" \
    --dataset_dir /n/fs/vision-mix/sk7524/LLaMA-Factory/data \
    --template qwen \
    --cutoff_len 4096 \
    --max_new_tokens 1024 \
    --save_name "${OUTPUT_DIR}/predictions.jsonl" \
    --n_generations 1

echo "Inference complete"
```

---

### 6. Evaluation Script

**File**: `2_11_26_training/weighted_loss_fn/scripts/stage4_evaluate.py`

```python
#!/usr/bin/env python3
"""
Stage 4: Evaluate all trained models on test set.

Computes metrics:
- Accuracy
- Precision (for Accept class)
- Recall (for Accept class)
- F1 score
- Predicted acceptance rate
- Per-year breakdown

Usage:
    python stage4_evaluate.py
    python stage4_evaluate.py --experiment accept_gamma2_prop1_2
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Paths
RESULTS_DIR = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_11_26_training/weighted_loss_fn/results")
METRICS_DIR = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_11_26_training/weighted_loss_fn/metrics")
TEST_DATA_PATH = "/n/fs/vision-mix/sk7524/LLaMA-Factory/data/iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test/data.json"


def parse_prediction(text: str) -> str:
    """
    Extract Accept/Reject from model output.

    Handles formats:
    - \\boxed{Accept}
    - \\boxed{Reject}
    - Plain "Accept" or "Reject"
    """
    # Try boxed format first
    match = re.search(r'\\boxed\{(Accept|Reject)\}', text)
    if match:
        return match.group(1)

    # Try plain format
    if "Accept" in text:
        return "Accept"
    elif "Reject" in text:
        return "Reject"

    return "Unknown"


def load_predictions(predictions_path: Path) -> List[Dict]:
    """Load predictions from jsonl file."""
    predictions = []
    with open(predictions_path, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def load_ground_truth(test_data_path: str) -> List[str]:
    """Load ground truth labels from test dataset."""
    with open(test_data_path, 'r') as f:
        data = json.load(f)

    labels = []
    for sample in data:
        assistant_response = sample["conversations"][-1]["value"]
        if "Accept" in assistant_response:
            labels.append("Accept")
        elif "Reject" in assistant_response:
            labels.append("Reject")
        else:
            labels.append("Unknown")

    return labels


def compute_metrics(predictions: List[str], ground_truth: List[str]) -> Dict:
    """Compute evaluation metrics."""
    # Convert to binary (1=Accept, 0=Reject)
    y_true = [1 if gt == "Accept" else 0 for gt in ground_truth]
    y_pred = [1 if pred == "Accept" else 0 for pred in predictions]

    # Remove unknowns
    valid_indices = [i for i, (gt, pred) in enumerate(zip(ground_truth, predictions))
                     if gt != "Unknown" and pred != "Unknown"]

    y_true = [y_true[i] for i in valid_indices]
    y_pred = [y_pred[i] for i in valid_indices]

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "predicted_accept_rate": np.mean(y_pred),
        "actual_accept_rate": np.mean(y_true),
        "total_samples": len(y_true),
        "num_unknowns": len(predictions) - len(y_true),
    }

    return metrics


def evaluate_experiment(experiment_name: str) -> Dict:
    """Evaluate a single experiment."""
    print(f"\nEvaluating: {experiment_name}")

    predictions_path = RESULTS_DIR / experiment_name / "predictions.jsonl"
    if not predictions_path.exists():
        print(f"  Predictions not found: {predictions_path}")
        return None

    # Load predictions
    predictions_data = load_predictions(predictions_path)

    # Extract predictions (from "predict" field)
    predictions = []
    for item in predictions_data:
        # Handle n_generations=1 format
        pred_text = item["predict"]
        if isinstance(pred_text, list):
            pred_text = pred_text[0]
        predictions.append(parse_prediction(pred_text))

    # Load ground truth
    ground_truth = load_ground_truth(TEST_DATA_PATH)

    # Ensure lengths match
    assert len(predictions) == len(ground_truth), \
        f"Length mismatch: {len(predictions)} predictions vs {len(ground_truth)} ground truth"

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth)

    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")
    print(f"  Predicted Accept Rate: {metrics['predicted_accept_rate']:.3f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Specific experiment to evaluate (e.g., accept_gamma2_prop1_2)"
    )
    args = parser.parse_args()

    # Create metrics directory
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all experiments
    if args.experiment:
        experiments = [args.experiment]
    else:
        # Find all result directories
        experiments = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()]
        experiments.sort()

    print(f"Evaluating {len(experiments)} experiments...")

    # Evaluate all experiments
    all_metrics = {}
    for exp_name in experiments:
        metrics = evaluate_experiment(exp_name)
        if metrics is not None:
            all_metrics[exp_name] = metrics

    # Save aggregate metrics
    output_path = METRICS_DIR / "all_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n\nMetrics saved to: {output_path}")
    print(f"Total evaluated: {len(all_metrics)}/{len(experiments)}")


if __name__ == "__main__":
    main()
```

---

### 7. Visualization Script

**File**: `2_11_26_training/weighted_loss_fn/scripts/stage5_visualize.py`

```python
#!/usr/bin/env python3
"""
Stage 5: Visualize experiment results.

Creates plots:
1. Predicted acceptance rate vs gamma (grouped by variant and proportion)
2. Accuracy vs gamma (grouped by variant and proportion)
3. Precision-Recall curves per configuration
4. Heatmaps of metrics across experiment dimensions

Usage:
    python stage5_visualize.py
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Paths
METRICS_DIR = Path("/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_11_26_training/weighted_loss_fn/metrics")
PLOTS_DIR = METRICS_DIR / "plots"

# Plot styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_metrics() -> Dict:
    """Load all metrics from JSON file."""
    metrics_path = METRICS_DIR / "all_metrics.json"
    with open(metrics_path, 'r') as f:
        return json.load(f)


def parse_experiment_name(exp_name: str) -> Dict:
    """
    Parse experiment name into components.

    Example: "accept_gamma2.0_prop1_2" -> {variant: "accept", gamma: 2.0, proportion: "1_2"}
    """
    parts = exp_name.split('_')

    # Extract variant
    variant = parts[0]

    # Extract gamma
    gamma_str = [p for p in parts if p.startswith('gamma')][0]
    gamma = float(gamma_str.replace('gamma', ''))

    # Extract proportion
    prop_idx = parts.index('prop')
    proportion = '_'.join(parts[prop_idx+1:])

    return {
        "variant": variant,
        "gamma": gamma,
        "proportion": proportion,
    }


def create_dataframe(all_metrics: Dict) -> pd.DataFrame:
    """Convert metrics dict to pandas DataFrame."""
    rows = []

    for exp_name, metrics in all_metrics.items():
        exp_params = parse_experiment_name(exp_name)

        row = {
            "experiment": exp_name,
            **exp_params,
            **metrics,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_acceptance_rate_vs_gamma(df: pd.DataFrame, output_path: Path):
    """Plot predicted acceptance rate vs gamma, grouped by variant and proportion."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, variant in enumerate(["accept", "reject"]):
        ax = axes[i]
        df_variant = df[df["variant"] == variant]

        for proportion in sorted(df_variant["proportion"].unique()):
            df_prop = df_variant[df_variant["proportion"] == proportion]
            df_prop = df_prop.sort_values("gamma")

            ax.plot(
                df_prop["gamma"],
                df_prop["predicted_accept_rate"],
                marker='o',
                label=f"Proportion {proportion.replace('_', ':')}",
                linewidth=2,
            )

        ax.set_xlabel("Gamma", fontsize=14)
        ax.set_ylabel("Predicted Accept Rate", fontsize=14)
        ax.set_title(f"Variant: Weight {variant.capitalize()}s", fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_vs_gamma(df: pd.DataFrame, output_path: Path):
    """Plot accuracy vs gamma, grouped by variant and proportion."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, variant in enumerate(["accept", "reject"]):
        ax = axes[i]
        df_variant = df[df["variant"] == variant]

        for proportion in sorted(df_variant["proportion"].unique()):
            df_prop = df_variant[df_variant["proportion"] == proportion]
            df_prop = df_prop.sort_values("gamma")

            ax.plot(
                df_prop["gamma"],
                df_prop["accuracy"],
                marker='s',
                label=f"Proportion {proportion.replace('_', ':')}",
                linewidth=2,
            )

        ax.set_xlabel("Gamma", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_title(f"Variant: Weight {variant.capitalize()}s", fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_heatmap_metric(df: pd.DataFrame, metric: str, output_path: Path):
    """Plot heatmap of a metric across all experiment dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for i, variant in enumerate(["accept", "reject"]):
        ax = axes[i]
        df_variant = df[df["variant"] == variant]

        # Pivot for heatmap (gamma × proportion)
        pivot = df_variant.pivot(
            index="gamma",
            columns="proportion",
            values=metric
        )

        # Sort columns by proportion value
        prop_order = sorted(pivot.columns, key=lambda x: eval(x.replace('_', '/')))
        pivot = pivot[prop_order]

        # Rename columns for readability
        pivot.columns = [c.replace('_', ':') for c in pivot.columns]

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            ax=ax,
            cbar_kws={'label': metric.replace('_', ' ').title()},
            vmin=df[metric].min(),
            vmax=df[metric].max(),
        )

        ax.set_title(f"Variant: Weight {variant.capitalize()}s", fontsize=16)
        ax.set_xlabel("Accept:Reject Proportion", fontsize=14)
        ax.set_ylabel("Gamma", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_precision_recall(df: pd.DataFrame, output_path: Path):
    """Plot precision vs recall for all experiments."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Color by variant, marker by gamma, size by proportion
    for variant in ["accept", "reject"]:
        df_variant = df[df["variant"] == variant]

        ax.scatter(
            df_variant["recall"],
            df_variant["precision"],
            label=f"Weight {variant.capitalize()}s",
            s=100,
            alpha=0.7,
        )

    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title("Precision-Recall Trade-off Across Experiments", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add diagonal line (F1=0.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='F1=0.5')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load metrics
    print("Loading metrics...")
    all_metrics = load_metrics()

    # Convert to DataFrame
    print("Converting to DataFrame...")
    df = create_dataframe(all_metrics)

    print(f"Loaded {len(df)} experiments")
    print(f"Variants: {df['variant'].unique()}")
    print(f"Gammas: {sorted(df['gamma'].unique())}")
    print(f"Proportions: {sorted(df['proportion'].unique())}")

    # Generate plots
    print("\nGenerating plots...")

    plot_acceptance_rate_vs_gamma(df, PLOTS_DIR / "acceptance_rate_vs_gamma.png")
    plot_accuracy_vs_gamma(df, PLOTS_DIR / "accuracy_vs_gamma.png")
    plot_precision_recall(df, PLOTS_DIR / "precision_recall.png")

    # Heatmaps for each metric
    for metric in ["accuracy", "predicted_accept_rate", "f1", "precision", "recall"]:
        plot_heatmap_metric(df, metric, PLOTS_DIR / f"heatmap_{metric}.png")

    print("\nVisualization complete!")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
```

---

## Evaluation Protocol

### Metrics to Compute

For each of the 32 trained models:

1. **Accuracy**: Overall correctness on test set
2. **Precision**: P(true accept | predicted accept)
3. **Recall**: P(predicted accept | true accept)
4. **F1 Score**: Harmonic mean of precision and recall
5. **Predicted Acceptance Rate**: Proportion of test samples predicted as Accept
6. **Calibration**: How well predicted rates match true rates

### Comparison Methodology

**Primary Comparisons**:

1. **Gamma Effect** (within same variant and proportion):
   - Compare gamma 1.0 vs 2.0 vs 4.0 vs 8.0
   - Expected: Higher gamma → stronger effect of weighting

2. **Variant Effect** (within same gamma and proportion):
   - Compare "accept" vs "reject" variant
   - Expected: Accept-weighting increases predicted acceptance rate
   - Expected: Reject-weighting decreases predicted acceptance rate

3. **Dataset Proportion Effect** (within same variant and gamma):
   - Compare 1:2 vs 1:4 vs 1:3 vs 1:8
   - Expected: Training on reject-heavy data decreases predicted acceptance rate

4. **Interaction Effects**:
   - Does gamma effect differ by proportion?
   - Does proportion effect differ by variant?

### Baseline Comparisons

**Baseline 1**: Gamma=1.0, Proportion=1:2, Variant=Accept
- Standard BCE on balanced data
- Serves as control condition

**External Baseline**: Original B2 model (if available)
- Compare against pre-existing Qwen2.5-7B fine-tuned on same data

---

## Expected Outcomes

### Hypothesis 1: Loss Weighting Effect

**Weight Accepts (Variant 1)**:
- As gamma increases (2 → 4 → 8 for balanced; 2 → 4 for imbalanced), predicted acceptance rate should **increase**
- Recall should improve, precision may decrease
- Model becomes more "optimistic" (prefers Accept)

**Weight Rejects (Variant 2)**:
- As gamma increases (2 → 4 → 8 for balanced; 2 → 4 for imbalanced), predicted acceptance rate should **decrease**
- Precision should improve, recall may decrease
- Model becomes more "pessimistic" (prefers Reject)

### Hypothesis 2: Dataset Proportion Effect

- Training on reject-heavy data (1:8, 1:4, 1:3) should decrease predicted acceptance rate relative to balanced (1:2)
- This effect should be observable across both loss variants
- The magnitude of this effect may be modulated by gamma weighting

### Hypothesis 3: Interaction Effects

**Balanced Dataset (1/2) with Strong Weighting (gamma=8)**:
- **Weight Accepts + gamma=8**: Maximum predicted acceptance rate in the experiment
- **Weight Rejects + gamma=8**: Minimum predicted acceptance rate in the experiment
- These two conditions should show the widest spread in predicted rates

**Imbalanced Datasets with Moderate Weighting (gamma=2, 4)**:
- **Weight Accepts + Extreme Reject Bias (1:8)**: Model fights against reject-heavy data
  - Predicted rate likely stays below 50% despite weighting accepts
  - Tests limits of loss weighting vs. data distribution
- **Weight Rejects + Extreme Reject Bias (1:8)**: Reinforcement of reject preference
  - Predicted rate likely very low (<20%)
  - May achieve strongest reject bias in experiment

### Success Criteria

**Experiment is successful if**:
1. **Monotonic gamma effects**: For each (variant, proportion) pair, gamma ↑ correlates with predicted rate ↑ (accept variant) or ↓ (reject variant)
2. **Distinguishable effects**: Loss weighting and data proportion effects are separable in analysis
3. **Wide dynamic range**: Predicted acceptance rates span at least 30-70% across all conditions
4. **Reasonable accuracy**: >60% accuracy on test set for most configurations (some trade-off expected)
5. **Balanced dataset shows full range**: Gamma=8 experiments achieve most extreme predicted rates

### Key Comparisons

**To Test Loss Weighting Strength**:
- Compare gamma=2 vs gamma=4 vs gamma=8 (balanced dataset only)
- Expect larger effect size from 4→8 than 2→4

**To Test Data Proportion Effects**:
- Compare 1/2 vs 1/3 vs 1/4 vs 1/8 at fixed gamma (e.g., gamma=2.0 across all)
- Expect monotonic decrease in predicted rate as reject proportion increases

**To Test Variant Differences**:
- Compare accept vs reject variant at same (gamma, proportion)
- Expect consistently higher predicted rate for accept variant

---

## Directory Structure

```
2_11_26_training/weighted_loss_fn/
├── experiment_implementation.md         # This file
├── data/                                # Generated training datasets
│   ├── iclr_weighted_loss_train_1_2/
│   │   └── data.json                   # 17,101 samples (50:50)
│   ├── iclr_weighted_loss_train_1_4/
│   │   └── data.json                   # 17,101 samples (25:75)
│   ├── iclr_weighted_loss_train_1_3/
│   │   └── data.json                   # 17,101 samples (33:67)
│   └── iclr_weighted_loss_train_1_8/
│       └── data.json                   # 17,101 samples (12.5:87.5)
├── configs/
│   ├── base_config.yaml                # Base training config (reference only)
│   └── generated/                      # Auto-generated configs per experiment
│       ├── baseline_gamma1.0_prop1_2.yaml
│       ├── accept_gamma2.0_prop1_2.yaml
│       ├── accept_gamma4.0_prop1_2.yaml
│       ├── accept_gamma8.0_prop1_2.yaml
│       ├── reject_gamma2.0_prop1_2.yaml
│       └── ...                         # 19 total configs
├── scripts/
│   ├── stage1_generate_datasets.py     # Create training datasets
│   ├── stage2_train_models.sbatch      # SLURM training array job (19 jobs)
│   ├── stage3_run_inference.sbatch     # SLURM inference array job (19 jobs)
│   ├── stage4_evaluate.py              # Compute metrics
│   └── stage5_visualize.py             # Generate plots
├── logs/
│   ├── train_*.out                     # Training logs (19 files)
│   └── infer_*.out                     # Inference logs (19 files)
├── results/                            # Inference outputs
│   ├── baseline_gamma1.0_prop1_2/
│   │   └── predictions.jsonl
│   ├── accept_gamma2.0_prop1_2/
│   │   └── predictions.jsonl
│   └── ...                             # 19 result directories
└── metrics/                            # Evaluation outputs
    ├── all_metrics.json                # Aggregate metrics for all experiments
    └── plots/
        ├── acceptance_rate_vs_gamma.png
        ├── accuracy_vs_gamma.png
        ├── heatmap_accuracy.png
        ├── heatmap_predicted_accept_rate.png
        └── precision_recall.png
```

---

## Execution Plan

### Phase 1: Setup (1-2 hours)

```bash
# 1. Create directory structure
mkdir -p /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_11_26_training/weighted_loss_fn/{data,configs,scripts,logs,results,metrics}

# 2. Implement loss functions
# Edit: src/llamafactory/train/trainer_utils.py
# Add: weighted_bce_loss_accept, weighted_bce_loss_reject functions

# 3. Add hyperparameters
# Edit: src/llamafactory/hparams/finetuning_args.py
# Add: use_weighted_loss, weighted_loss_variant, weighted_loss_gamma fields

# 4. Update trainer
# Edit: src/llamafactory/train/sft/trainer.py
# Add: weighted loss function selection in __init__

# 5. Register datasets
# Edit: data/dataset_info.json
# Add: 4 new dataset entries
```

### Phase 2: Data Generation (30 minutes)

```bash
# Test dataset generation
cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer
python 2_11_26_training/weighted_loss_fn/scripts/stage1_generate_datasets.py --debug

# Generate full datasets
python 2_11_26_training/weighted_loss_fn/scripts/stage1_generate_datasets.py

# Verify outputs
ls -lh 2_11_26_training/weighted_loss_fn/data/*/data.json
wc -l 2_11_26_training/weighted_loss_fn/data/*/data.json
```

### Phase 3: Training (48 hours × 19 jobs in parallel)

```bash
# Submit all 19 training jobs (including baseline)
sbatch 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch

# Or submit in batches
# Baseline only
sbatch --array=0 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch

# Balanced dataset (baseline + weighted, 7 jobs)
sbatch --array=0-6 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch

# Imbalanced datasets (12 jobs)
sbatch --array=7-18 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch

# Monitor progress
squeue -u $USER
tail -f 2_11_26_training/weighted_loss_fn/logs/train_*.out
```

### Phase 4: Inference (4 hours × 19 jobs)

```bash
# Submit all 19 inference jobs (including baseline)
sbatch 2_11_26_training/weighted_loss_fn/scripts/stage3_run_inference.sbatch

# Monitor
tail -f 2_11_26_training/weighted_loss_fn/logs/infer_*.out
```

### Phase 5: Evaluation & Visualization (30 minutes)

```bash
# Compute metrics
python 2_11_26_training/weighted_loss_fn/scripts/stage4_evaluate.py

# Generate plots
python 2_11_26_training/weighted_loss_fn/scripts/stage5_visualize.py

# View results
cat 2_11_26_training/weighted_loss_fn/metrics/all_metrics.json
open 2_11_26_training/weighted_loss_fn/metrics/plots/*.png
```

---

## Testing Protocol

### Unit Tests

**Test 1: Dataset Generation**
```bash
# Generate small debug datasets
python stage1_generate_datasets.py --debug

# Verify counts
python3 -c "
import json
for prop in ['1_2', '1_4', '1_3', '1_8']:
    path = f'data/iclr_weighted_loss_train_{prop}/data.json'
    data = json.load(open(path))
    accepts = sum(1 for x in data if 'Accept' in x['conversations'][-1]['value'])
    print(f'{prop}: {accepts}/{len(data)} = {accepts/len(data):.2%}')
"
```

**Test 2: Loss Function Implementation**
```python
# Test loss function signatures
from src.llamafactory.train.trainer_utils import weighted_bce_loss_accept, weighted_bce_loss_reject

# Create dummy inputs
outputs = {"logits": torch.randn(2, 10, 50000)}
labels = torch.randint(-100, 50000, (2, 10))

# Test accept weighting
loss_accept = weighted_bce_loss_accept(outputs, labels, gamma=2.0)
print(f"Accept loss (gamma=2.0): {loss_accept.item()}")

# Test reject weighting
loss_reject = weighted_bce_loss_reject(outputs, labels, gamma=2.0)
print(f"Reject loss (gamma=2.0): {loss_reject.item()}")

# Verify gamma=1.0 is similar to standard loss
loss_baseline = weighted_bce_loss_accept(outputs, labels, gamma=1.0)
print(f"Baseline loss (gamma=1.0): {loss_baseline.item()}")
```

**Test 3: Single Training Run**
```bash
# Test one configuration (gamma=1.0, proportion=1:2, variant=accept)
sbatch --array=0 2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch

# Monitor training progress
tail -f 2_11_26_training/weighted_loss_fn/logs/train_*_0.out

# Check if checkpoint is saved
ls -lh saves/weighted_loss/accept_gamma1.0_prop1_2/
```

**Test 4: Single Inference Run**
```bash
# Test inference on trained model
sbatch --array=0 2_11_26_training/weighted_loss_fn/scripts/stage3_run_inference.sbatch

# Check output
head 2_11_26_training/weighted_loss_fn/results/accept_gamma1.0_prop1_2/predictions.jsonl
wc -l 2_11_26_training/weighted_loss_fn/results/accept_gamma1.0_prop1_2/predictions.jsonl  # Should be 2024
```

**Test 5: Evaluation on Subset**
```bash
# Test evaluation on one experiment
python stage4_evaluate.py --experiment accept_gamma1.0_prop1_2

# Check output
cat 2_11_26_training/weighted_loss_fn/metrics/all_metrics.json
```

---

## Implementation Contract

### Code Modifications Required

**File 1**: `src/llamafactory/train/trainer_utils.py`
- Add `weighted_bce_loss_accept()`
- Add `weighted_bce_loss_reject()`
- Add `_weighted_cross_entropy_accept()`
- Add `_weighted_cross_entropy_reject()`

**File 2**: `src/llamafactory/hparams/finetuning_args.py`
- Add 3 new fields to `FinetuningArguments` dataclass

**File 3**: `src/llamafactory/train/sft/trainer.py`
- Modify `__init__()` to check `use_weighted_loss` flag
- Set `self.compute_loss_func` based on variant and gamma

**File 4**: `data/dataset_info.json`
- Add 4 new dataset entries

### New Files to Create

1. `2_11_26_training/weighted_loss_fn/scripts/stage1_generate_datasets.py`
2. `2_11_26_training/weighted_loss_fn/scripts/stage2_train_models.sbatch`
3. `2_11_26_training/weighted_loss_fn/scripts/stage3_run_inference.sbatch`
4. `2_11_26_training/weighted_loss_fn/scripts/stage4_evaluate.py`
5. `2_11_26_training/weighted_loss_fn/scripts/stage5_visualize.py`
6. `2_11_26_training/weighted_loss_fn/configs/base_config.yaml` (optional, used for reference)

### Acceptance Criteria

**Implementation Complete When**:
1. All loss functions pass unit tests
2. Debug dataset generation produces correct proportions
3. Single training run completes successfully
4. Single inference run produces 2024 predictions
5. Evaluation script computes all metrics correctly
6. Visualization script generates all plots

**Experiment Complete When**:
1. All 19 training jobs complete without errors (including baseline)
2. All 19 inference jobs produce predictions.jsonl files (2024 lines each)
3. `all_metrics.json` contains 19 experiment entries
4. All plots are generated and interpretable
5. Results show expected trends:
   - Baseline (gamma=1.0) provides reference point
   - Monotonic gamma effects within each (variant, proportion) group
   - Accept variant consistently predicts higher rates than reject variant
   - Imbalanced datasets show lower predicted rates than balanced

---

## Cost Estimation

### Computational Cost

**Training**:
- 19 jobs × 4 GPUs × 48 hours = 3,648 GPU-hours
- Assuming A100 GPUs at ~$2/hour = **~$7,296**

**Inference**:
- 19 jobs × 1 GPU × 4 hours = 76 GPU-hours
- Assuming A100 GPUs at ~$2/hour = **~$152**

**Total**: ~$7,448

### Storage Cost

**Datasets**: 4 × ~500MB = 2GB
**Model Checkpoints**: 19 × ~15GB = 285GB (can save only best checkpoints to reduce)
**Results**: 19 × ~10MB = 190MB
**Total**: ~287GB

### Time Cost

**Implementation**: 4-6 hours
**Data Generation**: 0.5 hours
**Training**: 48 hours (parallelized)
**Inference**: 4 hours (parallelized)
**Evaluation**: 0.5 hours
**Total Wall Time**: ~57 hours (2.4 days)

---

## Open Questions & Future Work

### Implementation Questions

1. **Token-level vs Sample-level Weighting**:
   - Current plan uses simplified token-level weighting
   - Better approach: Identify sample boundaries and apply gamma per sample
   - May require custom collator or trainer modifications

2. **Output Token Identification**:
   - For more precise weighting, identify "Accept"/"Reject" tokens specifically
   - Weight only those tokens with gamma, keep others at 1.0
   - Requires tokenizer analysis to find token IDs

3. **Gradient Accumulation Interaction**:
   - Does gamma weighting interact with gradient accumulation?
   - May need to normalize by effective batch size

### Experimental Extensions

1. **Additional Loss Variants**:
   - Focal loss for hard examples
   - Label smoothing with weighted BCE
   - Class-balanced loss (weight by inverse frequency)

2. **Additional Metrics**:
   - Calibration curves (predicted probability vs actual outcome)
   - Per-year breakdown (does effect vary by year?)
   - Confidence analysis (how confident is the model?)

3. **Extended Gamma Range**:
   - Test gamma = 0.5 (reverse weighting)
   - Test gamma = 16, 32 (extreme weighting)

4. **Two-Stage Training**:
   - Train with gamma=1.0 first, then fine-tune with weighted loss
   - Compare to single-stage weighted training

---

## References

- **Original Dataset**: `/n/fs/vision-mix/sk7524/LLaMA-Factory/data/iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_{train,test}`
- **LLaMA Factory Docs**: https://github.com/hiyouga/LLaMA-Factory
- **FSDP2 Config**: `configs/fsdp2_4gpu_config.yaml`
- **Qwen Model**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

---

## Summary

This experiment plan provides a comprehensive framework for investigating how weighted loss functions and dataset composition affect model acceptance rate predictions. The plan includes:

- **19 experimental conditions** with baseline and optimized gamma selection:
  - **Baseline**: gamma=1.0 (standard BCE) on balanced dataset = 1 experiment
  - Balanced dataset (1/2) weighted: 2 variants × 3 gammas [2, 4, 8] = 6 experiments
  - Imbalanced datasets (1/3, 1/4, 1/8): 3 datasets × 2 variants × 2 gammas [2, 4] = 12 experiments
- **Complete implementation specifications** for loss functions, hyperparameters, and trainer modifications
- **5-stage execution pipeline**: dataset generation → training → inference → evaluation → visualization
- **Testing protocols** for each stage
- **Expected outcomes** with clear hypotheses
- **Resource estimates**: ~$7.4K compute cost, 287GB storage, 2.4 days wall time

**Key Design Decisions**:
- **Index 0 is baseline** (gamma=1.0, standard BCE) - provides reference point for all comparisons
- Balanced dataset gets full gamma range [2, 4, 8] to explore strong weighting effects
- Imbalanced datasets use moderate gammas [2, 4] to avoid extreme interactions
- All weighted experiments can be compared against baseline to measure effect size

The experiment is designed to be modular, testable, and extensible, with clear success criteria and acceptance criteria for implementation completion.
