# Plan: Token-Level Accept/Reject Probability Tracking During Training

## Goal
Track the probabilities of "accept" and "reject" tokens during fine-tuning to:
1. Monitor model confidence (accept% + reject%) approaching 100%
2. Extract fine-grained calibration metrics
3. Observe how decision confidence evolves during training

## User Preferences
- **When**: During training via callback (real-time monitoring)
- **Detail**: Aggregate stats during training + standalone script for per-sample analysis

## Current Understanding
- **Training**: Uses LLaMA-Factory's `CustomSeq2SeqTrainer` (SFT stage)
- **Data format**: Answers in `\boxed{Accept}` or `\boxed{Reject}` format
- **Inference**: Currently uses vLLM for batch inference, then `analyze.py` extracts `\boxed{}` answers
- **No existing token probability tracking** during training

## Implementation Approach

### Step 1: Create Token Probability Callback
Create a new callback `AcceptRejectProbabilityCallback` in `src/llamafactory/train/callbacks.py`:

```python
class AcceptRejectProbabilityCallback(TrainerCallback):
    """
    Callback to track accept/reject token probabilities during training.

    Logs:
    - P(accept), P(reject) at the answer token position
    - Total probability mass (should approach 1.0)
    - Confidence = max(P(accept), P(reject))
    """
```

Key implementation details:
1. Store tokenizer reference and precompute token IDs for "Accept", "Reject", "accept", "reject"
2. On `on_evaluate` or at specified step intervals, run forward pass on eval samples
3. Find the position where the answer token should be (after `\boxed{`)
4. Extract logits, apply softmax, get probabilities for accept/reject tokens
5. Log to trainer's log history and optionally to a separate file

### Step 2: Modify SFT Workflow to Include Callback
Update `src/llamafactory/train/sft/workflow.py` to optionally include the new callback:
- Add a flag like `track_decision_probs: bool` to finetuning args
- When enabled, add `AcceptRejectProbabilityCallback` to callbacks list

### Step 3: Create Standalone Probability Extraction Script
Create `scripts/extract_token_probs.py` for:
- Loading a checkpoint at any training step
- Running forward pass on a dataset
- Saving detailed per-sample probabilities to JSONL

This is useful for:
- Post-hoc analysis of saved checkpoints
- Detailed analysis beyond what's logged during training

### Step 4: Visualization/Analysis
Create `scripts/analyze_probs.py` or extend `scripts/analyze.py` to:
- Plot probability evolution over training steps
- Compute calibration curves
- Show confidence distribution histograms

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/llamafactory/train/callbacks.py` | Modify | Add `AcceptRejectProbabilityCallback` class (~100 lines) |
| `src/llamafactory/hparams/finetuning_args.py` | Modify | Add `track_decision_probs` flag (~5 lines) |
| `src/llamafactory/train/sft/workflow.py` | Modify | Integrate callback when flag is set (~8 lines) |
| `scripts/extract_token_probs.py` | Create | Standalone script for per-sample probability extraction (~180 lines) |
| `configs/qwen2_5_3bvl_full_sft_ds3.yaml` | Modify | Add `track_decision_probs: true` |

## Technical Details

### Token ID Resolution
For Qwen tokenizer:
- Precompute token IDs: `tokenizer.encode("Accept", add_special_tokens=False)` etc.
- Handle variations: "Accept"/"accept"/"Reject"/"reject"
- Sum probabilities across all variants

### Finding Answer Position
The label sequence ends with `\boxed{Accept}` or `\boxed{Reject}`:
1. Search backwards from end of non-IGNORE_INDEX labels
2. Find the token ID that matches "Accept" or "Reject"
3. The position before that token is where we extract logits

### Probability Computation
```python
# At position where model predicts Accept/Reject
logits = outputs.logits[batch_idx, answer_pos-1, :]  # (vocab_size,)
probs = F.softmax(logits, dim=-1)
p_accept = probs[accept_token_ids].sum()
p_reject = probs[reject_token_ids].sum()
decision_prob_mass = p_accept + p_reject
confidence = max(p_accept, p_reject)
```

### Logged Metrics (during training)
```python
{
    "eval_avg_p_accept": 0.45,
    "eval_avg_p_reject": 0.35,
    "eval_decision_prob_mass": 0.80,  # Should approach 1.0
    "eval_decision_confidence": 0.55,
    "eval_decision_accuracy": 0.72    # Based on argmax
}
```

### Per-sample Output (standalone script)
```jsonl
{"sample_id": 0, "p_accept": 0.72, "p_reject": 0.18, "gt": "accept", "pred": "accept", "correct": true}
{"sample_id": 1, "p_accept": 0.35, "p_reject": 0.45, "gt": "reject", "pred": "reject", "correct": true}
```

## Implementation Sketch

### In `finetuning_args.py` (line ~585, after `plot_loss`):
```python
track_decision_probs: bool = field(
    default=False,
    metadata={"help": "Track accept/reject token probabilities during evaluation."},
)
```

### In `callbacks.py` (new class at end of file):
```python
class AcceptRejectProbabilityCallback(TrainerCallback):
    """Track accept/reject probabilities during evaluation."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Precompute token IDs for accept/reject variants
        self.accept_ids = self._get_token_ids(["Accept", "accept"])
        self.reject_ids = self._get_token_ids(["Reject", "reject"])

    def _get_token_ids(self, words):
        ids = set()
        for word in words:
            ids.update(self.tokenizer.encode(word, add_special_tokens=False))
        return list(ids)

    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        # Iterate through eval_dataloader, compute forward pass, extract probs
        # Log aggregate stats
```

### In `workflow.py` (after line ~80, metric_module setup):
```python
if finetuning_args.track_decision_probs:
    from ..callbacks import AcceptRejectProbabilityCallback
    callbacks.append(AcceptRejectProbabilityCallback(tokenizer=tokenizer))
```

## Verification
1. Add `track_decision_probs: true` to training config
2. Run training: `torchrun --nproc_per_node=8 src/train.py configs/qwen2_5_3bvl_full_sft_ds3.yaml`
3. Check logs for `eval_decision_prob_mass` metric at each eval step
4. Verify prob mass increases from ~0.5 toward 1.0 over training epochs
5. Run standalone script on checkpoint:
   ```bash
   python scripts/extract_token_probs.py \
       --checkpoint saves/qwen2.5vl-3b/full/sft_ds3/checkpoint-500 \
       --dataset iclr_2020_2025_85_5_10_split6_original_vision_binary_noreviews_v6_validation \
       --output results/probs_ckpt500.jsonl
   ```
6. Analyze output: check per-sample calibration, create confidence histograms
