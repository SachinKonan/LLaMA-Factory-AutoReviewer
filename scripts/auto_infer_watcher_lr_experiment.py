#!/usr/bin/env python3
"""
Background Checkpoint Watcher for LR Experiment Original Trainagreeing Jobs

Monitors 6 training directories for new checkpoints and auto-submits inference jobs:
- text_sft (4524593)
- vision_sft (4524596)
- text_cls (4524591)
- text_cls_rating (4524592)
- vision_cls (4524594)
- vision_cls_rating (4524595)

Usage:
    # Start in background
    nohup python scripts/auto_infer_watcher_lr_experiment.py \
        --poll_interval 60 \
        >> logs/lr_experiment_watcher.log 2>&1 &

    # Dry-run mode
    python scripts/auto_infer_watcher_lr_experiment.py --dry_run

    # Stop watcher
    pkill -f auto_infer_watcher_lr_experiment
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Base directories
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
SBATCH_DIR = REPO_ROOT / "sbatch" / "inference"


def setup_logging(log_file: Optional[Path] = None) -> None:
    """Configure logging to both console and a file."""
    log_format = "[%(asctime)s] [LR-Watcher] %(levelname)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        root_logger.addHandler(file_handler)


# Job configurations for the 6 original trainagreeing jobs
JOBS = [
    {
        'name': 'text_sft',
        'job_id': '4524593',
        'save_dir': 'saves/lr_experiment_v7/trainagreeing_original_no2024_text_sft',
        'results_dir': 'results/lr_experiment_v7/trainagreeing_original_no2024_text_sft',
        'dataset': 'iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_original_text_binary_noreviews_v7',
        'template': 'qwen',
        'cutoff_len': 24480,
        'max_new_tokens': 1280,
        'sbatch_script': 'sbatch/inference/text_inference.sbatch',
    },
    {
        'name': 'vision_sft',
        'job_id': '4524596',
        'save_dir': 'saves/lr_experiment_v7/trainagreeing_original_no2024_vision_sft',
        'results_dir': 'results/lr_experiment_v7/trainagreeing_original_no2024_vision_sft',
        'dataset': 'iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_original_vision_binary_noreviews_v7',
        'template': 'qwen2_vl',
        'cutoff_len': 24480,
        'max_new_tokens': 1280,
        'image_min_pixels': 784,
        'image_max_pixels': 1003520,
        'sbatch_script': 'sbatch/inference/vision_inference.sbatch',
    }
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor LR experiment training directories and submit inference jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--poll_interval",
        type=int,
        default=60,
        help="Seconds between polling for new checkpoints (default: 60)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without actually submitting jobs",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Log file path (default: logs/lr_experiment_watcher.log)",
    )

    return parser.parse_args()


def is_checkpoint_complete(checkpoint_dir: Path) -> bool:
    """Check if a checkpoint has all required model files."""
    required_patterns = [
        "model*.safetensors",
        "config.json",
    ]

    for pattern in required_patterns:
        matches = list(checkpoint_dir.glob(pattern))
        if not matches:
            return False
    return True


def get_checkpoint_step(checkpoint_dir: Path) -> Optional[int]:
    """Extract step number from checkpoint-XXXX directory name."""
    name = checkpoint_dir.name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-")[1])
        except (IndexError, ValueError):
            return None
    return None


def is_training_complete(save_dir: Path) -> bool:
    """Check if training has completed (all_results.json exists)."""
    return (save_dir / "all_results.json").exists()


def submit_sbatch(
    sbatch_script: Path,
    env: dict[str, str],
    dry_run: bool = False,
) -> Optional[str]:
    """Submit sbatch job with environment variables. Returns job ID if successful."""
    if dry_run:
        logger.info(f"[DRY-RUN] Would submit: {sbatch_script}")
        logger.info(f"[DRY-RUN] Environment: {env}")
        return "DRY-RUN-JOB-ID"

    try:
        full_env = os.environ.copy()
        full_env.update(env)

        result = subprocess.run(
            ["sbatch", str(sbatch_script)],
            env=full_env,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )

        if result.returncode == 0:
            # Parse job ID from "Submitted batch job 12345"
            job_output = result.stdout.strip()
            job_id = job_output.split()[-1] if job_output else None
            return job_id
        else:
            logger.warning(f"sbatch failed: {result.stderr}")
            return None
    except Exception as e:
        logger.warning(f"Error submitting sbatch: {e}")
        return None


def process_job(job_config: dict, dry_run: bool = False) -> int:
    """
    Process a single job, checking for new checkpoints and submitting inference.
    Returns number of new jobs submitted.
    """
    save_dir = REPO_ROOT / job_config['save_dir']
    results_dir = REPO_ROOT / job_config['results_dir']
    sbatch_script = REPO_ROOT / job_config['sbatch_script']

    if not save_dir.exists():
        return 0

    if not sbatch_script.exists():
        logger.error(f"{job_config['name']}: sbatch script not found: {sbatch_script}")
        return 0

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    # Add _test suffix to dataset
    test_dataset = job_config['dataset']
    if not test_dataset.endswith('_test'):
        test_dataset = f"{test_dataset}_test"

    # Build image params for vision models
    image_params = ""
    if 'qwen2_vl' in job_config['template'].lower():
        if 'image_min_pixels' in job_config and 'image_max_pixels' in job_config:
            image_params = f"--image_min_pixels {job_config['image_min_pixels']} --image_max_pixels {job_config['image_max_pixels']}"

    submitted_count = 0

    # Scan for unprocessed checkpoints
    for ckpt_dir in save_dir.glob("checkpoint-*"):
        try:
            if not ckpt_dir.is_dir():
                continue

            touch_file = ckpt_dir / ".infer.touch"
            if touch_file.exists():
                continue  # Already submitted

            if not is_checkpoint_complete(ckpt_dir):
                continue  # Still being written

            ckpt_step = get_checkpoint_step(ckpt_dir)
            if ckpt_step is None:
                continue

            # Prepare environment variables
            env = {
                "CHECKPOINT_DIR": str(ckpt_dir),
                "DATASET": test_dataset,
                "TEMPLATE": job_config['template'],
                "CKPT_STEP": str(ckpt_step),
                "RESULTS_DIR": str(results_dir),
                "CUTOFF_LEN": str(job_config['cutoff_len']),
                "MAX_NEW_TOKENS": str(job_config['max_new_tokens']),
            }

            if image_params:
                env["IMAGE_PARAMS"] = image_params

            # Submit job
            job_id = submit_sbatch(sbatch_script, env, dry_run=dry_run)

            if job_id:
                # Write touch file
                touch_file.write_text(f"{job_id}\n")
                logger.info(f"{job_config['name']}: Submitted job {job_id} for {ckpt_dir.name}")
                submitted_count += 1
            else:
                logger.warning(f"{job_config['name']}: Failed to submit job for {ckpt_dir.name}")

        except Exception as e:
            logger.warning(f"{job_config['name']}: Error processing {ckpt_dir}: {e}")
            continue

    return submitted_count


def main():
    """Main entry point for the LR experiment watcher."""
    args = parse_args()

    # Setup logging
    log_file = args.log_file if args.log_file else REPO_ROOT / "logs" / "lr_experiment_watcher.log"
    setup_logging(log_file)

    # Validate sbatch scripts
    missing_scripts = []
    for job in JOBS:
        sbatch_script = REPO_ROOT / job['sbatch_script']
        if not sbatch_script.exists():
            missing_scripts.append(str(sbatch_script))

    if missing_scripts:
        logger.error("Missing sbatch scripts:")
        for script in missing_scripts:
            logger.error(f"  {script}")
        sys.exit(1)

    # Log startup info
    logger.info("=" * 80)
    logger.info("LR Experiment Checkpoint Watcher Started")
    logger.info("=" * 80)
    logger.info(f"Monitoring {len(JOBS)} jobs:")
    for job in JOBS:
        logger.info(f"  - {job['name']} (job {job['job_id']}): {job['save_dir']}")
    logger.info(f"Poll interval: {args.poll_interval}s")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    # Track completion status
    completed_jobs = set()

    # Main polling loop
    while True:
        try:
            all_complete = True
            total_submitted = 0

            for job_config in JOBS:
                job_name = job_config['name']
                save_dir = REPO_ROOT / job_config['save_dir']

                # Skip if already marked complete
                if job_name in completed_jobs:
                    continue

                # Check if training complete
                if is_training_complete(save_dir):
                    if job_name not in completed_jobs:
                        logger.info(f"{job_name}: Training complete (all_results.json found)")
                        completed_jobs.add(job_name)
                    continue

                # Training still ongoing
                all_complete = False

                # Process checkpoints for this job
                submitted = process_job(job_config, dry_run=args.dry_run)
                total_submitted += submitted

            # Exit if all jobs complete
            if all_complete:
                logger.info("=" * 80)
                logger.info("All jobs completed, exiting watcher")
                logger.info("=" * 80)
                break

            if total_submitted > 0:
                logger.info(f"Submitted {total_submitted} inference jobs this cycle")

        except Exception as e:
            logger.warning(f"Watcher loop error (will retry): {e}")

        # Wait for next poll
        time.sleep(args.poll_interval)

    logger.info("LR Experiment Checkpoint Watcher stopped")


if __name__ == "__main__":
    main()
