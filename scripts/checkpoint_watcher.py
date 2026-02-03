#!/usr/bin/env python3
"""
Checkpoint Watcher for Auto-Inference

Monitors saves directories for new checkpoints and automatically spawns inference jobs.

Supported experiments:
- saves/final_sweep_v7/ and saves/final_sweep_v7_pli/ (v7 experiments)
- saves/testing_v8/ (v8 experiments)

Usage:
    python scripts/checkpoint_watcher.py              # Run once
    python scripts/checkpoint_watcher.py --dry-run    # Show what would be launched
    watch -n 300 python scripts/checkpoint_watcher.py # Run every 5 minutes
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Base directories to monitor
BASE_DIR = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")

# Save directories and their corresponding sbatch scripts
EXPERIMENT_CONFIGS = {
    "final_sweep_v7": {
        "save_dirs": [
            BASE_DIR / "saves" / "final_sweep_v7",
            BASE_DIR / "saves" / "final_sweep_v7_pli",
        ],
        "sbatch_script": BASE_DIR / "sbatch" / "final_sweep_v7" / "infer_checkpoint.sbatch",
        "image_max_pixels": 1003520,
    },
    "testing_v8": {
        "save_dirs": [
            BASE_DIR / "saves" / "testing_v8",
        ],
        "sbatch_script": BASE_DIR / "sbatch" / "testing_v8" / "infer_checkpoint.sbatch",
        "image_max_pixels": 5000000,  # 5e6 for v8 vision
    },
}

# Inference parameters (common)
CUTOFF_LEN = 24480
MAX_NEW_TOKENS = 1280
IMAGE_MIN_PIXELS = 784

# Dataset mappings
# Text-only datasets (template: qwen)
TEXT_DATASETS = {
    "balanced_clean": "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7",
    "balanced_title_abstract": "iclr_2020_2025_85_5_10_split7_balanced_clean_title_abstract_binary_noreviews_v7",
    "balanced_trainagreeing": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_binary_noreviews_v7",
    "balanced_2024_2025": "iclr_2024_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7",
    "balanced_2017_2025": "iclr_2017_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7",
}

# Vision datasets (template: qwen2_vl, with image params)
VISION_DATASETS = {
    "balanced_vision": "iclr_2020_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_trainagreeing_vision": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
    "balanced_2024_2025_vision": "iclr_2024_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_2017_2025_vision": "iclr_2017_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_no2024_vision": "iclr_2020_2023_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7",
    "balanced_trainagreeing_no2024_vision": "iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7",
}

# Images datasets (template: qwen2_vl, with image params)
IMAGES_DATASETS = {
    "balanced_clean_images": "iclr_2020_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7",
    "balanced_trainagreeing_images": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_clean_images_binary_noreviews_v7",
    "balanced_2024_2025_images": "iclr_2024_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7",
    "balanced_2017_2025_images": "iclr_2017_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7",
}

# ============================================================================
# V8 Dataset Mappings (testing_v8)
# ============================================================================

# V8 Text datasets (template: qwen)
V8_TEXT_DATASETS = {
    # Binary experiments
    "26ood_text": "2020_2025_train_2020_2026_valtest_clean_binary_noreviews_v8",
    "26ood_rm24_text": "2020_2023_2025_train_2020_2026_valtest_clean_binary_noreviews_v8",
    "balanced_text": "2020_2026_balanced_clean_binary_noreviews_v8",
    "26ood_text_bz32": "2020_2025_train_2020_2026_valtest_clean_binary_noreviews_v8",
    # Ratingbinary experiments
    "ood_text_rating": "2020_2025_train_2020_2026_valtest_clean_ratingbinary_noreviews_v8",
    "balanced_text_rating": "2020_2026_balanced_clean_ratingbinary_noreviews_v8",
}

# V8 Vision datasets (template: qwen2_vl, with image params)
V8_VISION_DATASETS = {
    # Binary experiments
    "26ood_vision": "2020_2025_train_2020_2026_valtest_vision_binary_noreviews_v8",
    "26ood_rm24_vision": "2020_2023_2025_train_2020_2026_valtest_vision_binary_noreviews_v8",
    "balanced_vision": "2020_2026_balanced_vision_binary_noreviews_v8",
    "26ood_vision_bz32": "2020_2025_train_2020_2026_valtest_vision_binary_noreviews_v8",
    # Ratingbinary experiments
    "ood_vision_rating": "2020_2025_train_2020_2026_valtest_vision_ratingbinary_noreviews_v8",
    "balanced_vision_rating": "2020_2026_balanced_vision_ratingbinary_noreviews_v8",
}


def get_model_type(short_name: str) -> str:
    """Determine model type from short_name suffix."""
    if short_name.endswith("_vision"):
        return "vision"
    elif short_name.endswith("_images"):
        return "images"
    else:
        return "text"


def get_dataset_and_template(short_name: str, experiment_type: str = "final_sweep_v7") -> Optional[tuple[str, str, bool]]:
    """
    Get dataset name, template, and whether image params are needed.
    Returns: (dataset_name, template, needs_image_params) or None if not found.
    """
    if experiment_type == "testing_v8":
        # V8 datasets
        if short_name in V8_TEXT_DATASETS:
            return V8_TEXT_DATASETS[short_name], "qwen", False
        elif short_name in V8_VISION_DATASETS:
            return V8_VISION_DATASETS[short_name], "qwen2_vl", True
    else:
        # V7 datasets (final_sweep_v7)
        if short_name in TEXT_DATASETS:
            return TEXT_DATASETS[short_name], "qwen", False
        elif short_name in VISION_DATASETS:
            return VISION_DATASETS[short_name], "qwen2_vl", True
        elif short_name in IMAGES_DATASETS:
            return IMAGES_DATASETS[short_name], "qwen2_vl", True
    return None


def is_checkpoint_complete(checkpoint_dir: Path) -> bool:
    """Check if a checkpoint has all required model files."""
    # Check for common checkpoint files
    required_patterns = [
        "model*.safetensors",  # Model weights
        "config.json",  # Config file
    ]

    for pattern in required_patterns:
        matches = list(checkpoint_dir.glob(pattern))
        if not matches:
            return False
    return True


def get_checkpoint_step(checkpoint_dir: Path) -> Optional[int]:
    """Extract the step number from checkpoint-XXXX directory name."""
    name = checkpoint_dir.name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-")[1])
        except (IndexError, ValueError):
            return None
    return None


def get_all_checkpoints(model_dir: Path) -> list[Path]:
    """Get all checkpoint directories in a model directory, sorted by step."""
    checkpoints = []
    for ckpt in model_dir.glob("checkpoint-*"):
        if ckpt.is_dir():
            step = get_checkpoint_step(ckpt)
            if step is not None:
                checkpoints.append((step, ckpt))

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return [ckpt for _, ckpt in checkpoints]


def is_final_checkpoint(checkpoint_dir: Path, model_dir: Path) -> bool:
    """
    Check if this is the final (6th) checkpoint that training script handles.
    Also checks if all_results.json exists (training completed).
    """
    # Check if training completed (all_results.json exists)
    if (model_dir / "all_results.json").exists():
        all_checkpoints = get_all_checkpoints(model_dir)
        if all_checkpoints and checkpoint_dir == all_checkpoints[-1]:
            return True

    # Alternative: count checkpoints and check if this is the max
    all_checkpoints = get_all_checkpoints(model_dir)
    if len(all_checkpoints) >= 6:  # Expected 6 epochs
        if checkpoint_dir == all_checkpoints[-1]:
            return True

    return False


def get_results_dir(save_dir: Path, short_name: str) -> Path:
    """Get the results directory for a given save directory and short name."""
    # Map saves/final_sweep_v7 -> results/final_sweep_v7
    # Map saves/final_sweep_v7_pli -> results/final_sweep_v7_pli
    results_base = save_dir.name  # final_sweep_v7 or final_sweep_v7_pli
    return BASE_DIR / "results" / results_base / short_name


def submit_inference_job(
    checkpoint_dir: Path,
    dataset: str,
    template: str,
    short_name: str,
    ckpt_step: int,
    needs_image_params: bool,
    results_dir: Path,
    sbatch_script: Path,
    image_max_pixels: int,
) -> Optional[str]:
    """Submit an sbatch job for inference on a checkpoint.

    Returns the job ID if successful, None otherwise.
    """
    # Prepare environment variables for sbatch
    env = os.environ.copy()
    env["CHECKPOINT_DIR"] = str(checkpoint_dir)
    env["DATASET"] = dataset
    env["TEMPLATE"] = template
    env["SHORT_NAME"] = short_name
    env["CKPT_STEP"] = str(ckpt_step)
    env["RESULTS_DIR"] = str(results_dir)
    env["CUTOFF_LEN"] = str(CUTOFF_LEN)
    env["MAX_NEW_TOKENS"] = str(MAX_NEW_TOKENS)

    if needs_image_params:
        env["IMAGE_PARAMS"] = f"--image_min_pixels {IMAGE_MIN_PIXELS} --image_max_pixels {image_max_pixels}"
    else:
        env["IMAGE_PARAMS"] = ""

    # Submit the job
    try:
        result = subprocess.run(
            ["sbatch", str(sbatch_script)],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )

        if result.returncode == 0:
            # Parse job ID from "Submitted batch job 12345"
            job_output = result.stdout.strip()
            print(f"  Submitted job: {job_output}")
            # Extract just the job ID number
            job_id = job_output.split()[-1] if job_output else None
            return job_id
        else:
            print(f"  ERROR submitting job: {result.stderr}")
            return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def process_checkpoint(
    checkpoint_dir: Path,
    model_dir: Path,
    short_name: str,
    save_dir: Path,
    experiment_type: str,
    sbatch_script: Path,
    image_max_pixels: int,
    dry_run: bool = False,
) -> bool:
    """
    Process a single checkpoint.
    Returns True if a job was submitted (or would be in dry-run), False otherwise.
    """
    touch_file = checkpoint_dir / ".infer.touch"

    # Skip if already processed
    if touch_file.exists():
        return False

    # Skip if checkpoint is incomplete
    if not is_checkpoint_complete(checkpoint_dir):
        print(f"  Skipping (incomplete): {checkpoint_dir}")
        return False

    # Skip if it's the final checkpoint
    if is_final_checkpoint(checkpoint_dir, model_dir):
        print(f"  Skipping (final checkpoint, handled by training script): {checkpoint_dir}")
        return False

    # Get dataset info
    dataset_info = get_dataset_and_template(short_name, experiment_type)
    if dataset_info is None:
        print(f"  WARNING: Unknown short_name '{short_name}' for {experiment_type}, skipping")
        return False

    dataset, template, needs_image_params = dataset_info
    ckpt_step = get_checkpoint_step(checkpoint_dir)

    if ckpt_step is None:
        print(f"  WARNING: Could not parse step from {checkpoint_dir}")
        return False

    results_dir = get_results_dir(save_dir, short_name)

    if dry_run:
        print(f"  [DRY-RUN] Would submit: {checkpoint_dir.name}")
        print(f"    Dataset: {dataset}")
        print(f"    Template: {template}")
        print(f"    Step: {ckpt_step}")
        print(f"    Output: {results_dir}/finetuned-ckpt-{ckpt_step}.jsonl")
        return True

    print(f"  Processing: {checkpoint_dir.name}")
    print(f"    Dataset: {dataset}")
    print(f"    Template: {template}")
    print(f"    Step: {ckpt_step}")
    print(f"    Results: {results_dir}")

    # Submit inference job
    job_id = submit_inference_job(
        checkpoint_dir=checkpoint_dir,
        dataset=dataset,
        template=template,
        short_name=short_name,
        ckpt_step=ckpt_step,
        needs_image_params=needs_image_params,
        results_dir=results_dir,
        sbatch_script=sbatch_script,
        image_max_pixels=image_max_pixels,
    )

    if job_id is None:
        return False

    # Write touch file with job ID to prevent duplicate submissions
    touch_file.write_text(f"{job_id}\n")

    return True


def scan_and_process(dry_run: bool = False):
    """Scan all save directories and process new checkpoints."""
    total_submitted = 0

    for experiment_type, config in EXPERIMENT_CONFIGS.items():
        sbatch_script = config["sbatch_script"]
        image_max_pixels = config["image_max_pixels"]

        for save_dir in config["save_dirs"]:
            if not save_dir.exists():
                print(f"Save directory does not exist: {save_dir}")
                continue

            print(f"\nScanning: {save_dir} ({experiment_type})")

            # Find all model directories (short_name directories)
            for model_dir in save_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                short_name = model_dir.name

                # Skip if not a known short_name for this experiment type
                if get_dataset_and_template(short_name, experiment_type) is None:
                    continue

                print(f"\n  Model: {short_name}")

                # Get all checkpoints
                checkpoints = get_all_checkpoints(model_dir)

                if not checkpoints:
                    print(f"    No checkpoints found")
                    continue

                print(f"    Found {len(checkpoints)} checkpoint(s)")

                # Process each checkpoint
                for ckpt_dir in checkpoints:
                    if process_checkpoint(
                        ckpt_dir,
                        model_dir,
                        short_name,
                        save_dir,
                        experiment_type=experiment_type,
                        sbatch_script=sbatch_script,
                        image_max_pixels=image_max_pixels,
                        dry_run=dry_run,
                    ):
                        total_submitted += 1

    print(f"\n{'='*60}")
    if dry_run:
        print(f"Total jobs that would be submitted: {total_submitted}")
    else:
        print(f"Total jobs submitted: {total_submitted}")
    print(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor checkpoints and spawn inference jobs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what jobs would be submitted without actually submitting them",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Checkpoint Watcher for Auto-Inference")
    if args.dry_run:
        print("[DRY-RUN MODE - No jobs will be submitted]")
    print("=" * 60)

    # Verify sbatch scripts exist
    for experiment_type, config in EXPERIMENT_CONFIGS.items():
        sbatch_script = config["sbatch_script"]
        if not sbatch_script.exists():
            print(f"ERROR: Sbatch script not found for {experiment_type}: {sbatch_script}")
            sys.exit(1)

    scan_and_process(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
