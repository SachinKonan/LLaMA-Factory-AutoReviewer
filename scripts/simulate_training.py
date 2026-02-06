#!/usr/bin/env python3
"""
Simulate Training for Testing Auto-Inference Watcher

Creates fake checkpoints at intervals to test the auto_infer_watcher.py script.

Usage:
    python scripts/simulate_training.py \
        --save_dir /tmp/test_training \
        --num_checkpoints 5 \
        --checkpoint_interval 10 \
        --steps_per_checkpoint 100
"""

import argparse
import json
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate training by creating fake checkpoints")
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory to create fake checkpoints in",
    )
    parser.add_argument(
        "--num_checkpoints",
        type=int,
        default=5,
        help="Number of checkpoints to create (default: 5)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Seconds between checkpoint creation (default: 10)",
    )
    parser.add_argument(
        "--steps_per_checkpoint",
        type=int,
        default=100,
        help="Step increment between checkpoints (default: 100)",
    )
    return parser.parse_args()


def create_fake_checkpoint(save_dir: Path, step: int) -> Path:
    """Create a fake checkpoint directory with required files."""
    ckpt_dir = save_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create fake model files (just empty files with right names)
    (ckpt_dir / "model-00001-of-00001.safetensors").write_text("fake model weights")
    (ckpt_dir / "config.json").write_text(json.dumps({"model_type": "fake", "step": step}))
    (ckpt_dir / "tokenizer.json").write_text("{}")
    (ckpt_dir / "trainer_state.json").write_text(json.dumps({"global_step": step}))

    return ckpt_dir


def create_all_results(save_dir: Path, total_steps: int):
    """Create all_results.json to signal training completion."""
    results = {
        "epoch": 3.0,
        "total_flos": 0,
        "train_loss": 0.5,
        "train_runtime": 3600,
        "train_samples": 1000,
        "train_samples_per_second": 0.28,
        "train_steps_per_second": 0.01,
        "global_step": total_steps,
    }
    (save_dir / "all_results.json").write_text(json.dumps(results, indent=2))


def main():
    args = parse_args()

    # Create save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Simulator Started")
    print("=" * 60)
    print(f"Save directory: {args.save_dir}")
    print(f"Checkpoints to create: {args.num_checkpoints}")
    print(f"Interval between checkpoints: {args.checkpoint_interval}s")
    print(f"Steps per checkpoint: {args.steps_per_checkpoint}")
    print("=" * 60)

    current_step = 0

    for i in range(args.num_checkpoints):
        current_step += args.steps_per_checkpoint

        print(f"\n[Simulator] Creating checkpoint-{current_step}...")
        ckpt_dir = create_fake_checkpoint(args.save_dir, current_step)
        print(f"[Simulator] Created: {ckpt_dir}")

        if i < args.num_checkpoints - 1:
            print(f"[Simulator] Waiting {args.checkpoint_interval}s before next checkpoint...")
            time.sleep(args.checkpoint_interval)

    # Create all_results.json to signal completion
    print(f"\n[Simulator] Training complete! Creating all_results.json...")
    create_all_results(args.save_dir, current_step)

    print("\n" + "=" * 60)
    print("Training Simulation Complete")
    print("=" * 60)
    print(f"Created {args.num_checkpoints} checkpoints (step {args.steps_per_checkpoint} to {current_step})")
    print(f"Final checkpoint: checkpoint-{current_step}")
    print("=" * 60)


if __name__ == "__main__":
    main()
