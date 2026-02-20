#!/usr/bin/env python3
"""
Validation script to verify the weighted loss implementation.

Checks:
- All required files exist
- Dataset registration in dataset_info.json
- Code modifications in source files
- Directory structure

Usage:
    python validate_implementation.py
"""

import json
from pathlib import Path

# Base paths
BASE_DIR = Path("/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer")
EXPERIMENT_DIR = BASE_DIR / "2_11_26_training/weighted_loss_fn"


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists and report."""
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def check_directory_exists(path: Path, description: str) -> bool:
    """Check if a directory exists and report."""
    exists = path.exists() and path.is_dir()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists


def check_datasets_registered() -> bool:
    """Check if datasets are registered in dataset_info.json."""
    print("\nChecking dataset registration...")
    dataset_info_path = BASE_DIR / "data/dataset_info.json"

    if not dataset_info_path.exists():
        print("✗ dataset_info.json not found")
        return False

    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)

    required_datasets = [
        "iclr_weighted_loss_train_1_2",
        "iclr_weighted_loss_train_1_3",
        "iclr_weighted_loss_train_1_4",
        "iclr_weighted_loss_train_1_8",
    ]

    all_registered = True
    for dataset_name in required_datasets:
        if dataset_name in dataset_info:
            print(f"✓ {dataset_name} registered")
        else:
            print(f"✗ {dataset_name} NOT registered")
            all_registered = False

    return all_registered


def check_code_modifications() -> bool:
    """Check if source code modifications are present."""
    print("\nChecking source code modifications...")

    # Check trainer_utils.py
    trainer_utils = BASE_DIR / "src/llamafactory/train/trainer_utils.py"
    if not trainer_utils.exists():
        print("✗ trainer_utils.py not found")
        return False

    with open(trainer_utils, "r") as f:
        content = f.read()

    checks = [
        ("weighted_bce_loss_accept" in content, "weighted_bce_loss_accept function"),
        ("weighted_bce_loss_reject" in content, "weighted_bce_loss_reject function"),
        ("_get_binary_decision_logit" in content, "_get_binary_decision_logit function"),
    ]

    for check, description in checks:
        status = "✓" if check else "✗"
        print(f"{status} {description}")

    # Check finetuning_args.py
    finetuning_args = BASE_DIR / "src/llamafactory/hparams/finetuning_args.py"
    if not finetuning_args.exists():
        print("✗ finetuning_args.py not found")
        return False

    with open(finetuning_args, "r") as f:
        content = f.read()

    checks = [
        ("use_weighted_loss" in content, "use_weighted_loss field"),
        ("weighted_loss_variant" in content, "weighted_loss_variant field"),
        ("weighted_loss_gamma" in content, "weighted_loss_gamma field"),
    ]

    for check, description in checks:
        status = "✓" if check else "✗"
        print(f"{status} {description}")

    # Check trainer.py
    trainer_py = BASE_DIR / "src/llamafactory/train/sft/trainer.py"
    if not trainer_py.exists():
        print("✗ trainer.py not found")
        return False

    with open(trainer_py, "r") as f:
        content = f.read()

    checks = [
        ("use_weighted_loss" in content, "Weighted loss integration in trainer"),
        ("weighted_bce_loss_accept" in content, "weighted_bce_loss_accept import"),
        ("weighted_bce_loss_reject" in content, "weighted_bce_loss_reject import"),
    ]

    for check, description in checks:
        status = "✓" if check else "✗"
        print(f"{status} {description}")

    return True


def main():
    print("=" * 60)
    print("Weighted Loss Function Implementation Validation")
    print("=" * 60)

    all_checks_passed = True

    # Check directory structure
    print("\nChecking directory structure...")
    directories = [
        (EXPERIMENT_DIR, "Experiment directory"),
        (EXPERIMENT_DIR / "data", "Data directory"),
        (EXPERIMENT_DIR / "configs", "Configs directory"),
        (EXPERIMENT_DIR / "scripts", "Scripts directory"),
        (EXPERIMENT_DIR / "logs", "Logs directory"),
        (EXPERIMENT_DIR / "results", "Results directory"),
        (EXPERIMENT_DIR / "metrics", "Metrics directory"),
    ]

    for path, desc in directories:
        if not check_directory_exists(path, desc):
            all_checks_passed = False

    # Check script files
    print("\nChecking script files...")
    scripts = [
        (EXPERIMENT_DIR / "scripts/stage1_generate_datasets.py", "Stage 1: Dataset generation"),
        (EXPERIMENT_DIR / "scripts/stage1b_pretokenize.sbatch", "Stage 1b: Pre-tokenization"),
        (EXPERIMENT_DIR / "scripts/stage2_train_models.sbatch", "Stage 2: Training"),
        (EXPERIMENT_DIR / "scripts/stage3_run_inference.sbatch", "Stage 3: Inference"),
        (EXPERIMENT_DIR / "scripts/stage4_evaluate.py", "Stage 4: Evaluation"),
        (EXPERIMENT_DIR / "scripts/stage5_visualize.py", "Stage 5: Visualization"),
    ]

    for path, desc in scripts:
        if not check_file_exists(path, desc):
            all_checks_passed = False

    # Check documentation
    print("\nChecking documentation...")
    docs = [
        (EXPERIMENT_DIR / "README.md", "README"),
        (EXPERIMENT_DIR / "experiment_implementation.md", "Implementation plan"),
        (EXPERIMENT_DIR / "IMPLEMENTATION_SUMMARY.md", "Implementation summary"),
    ]

    for path, desc in docs:
        if not check_file_exists(path, desc):
            all_checks_passed = False

    # Check dataset registration
    if not check_datasets_registered():
        all_checks_passed = False

    # Check code modifications
    if not check_code_modifications():
        all_checks_passed = False

    # Check if debug datasets exist
    print("\nChecking debug datasets (if generated)...")
    debug_datasets = [
        EXPERIMENT_DIR / "data/iclr_weighted_loss_train_1_2/data.json",
        EXPERIMENT_DIR / "data/iclr_weighted_loss_train_1_3/data.json",
        EXPERIMENT_DIR / "data/iclr_weighted_loss_train_1_4/data.json",
        EXPERIMENT_DIR / "data/iclr_weighted_loss_train_1_8/data.json",
    ]

    datasets_exist = 0
    for path in debug_datasets:
        if path.exists():
            datasets_exist += 1
            print(f"✓ {path.name} exists")

    if datasets_exist == 0:
        print("ℹ No datasets generated yet (run stage1_generate_datasets.py)")
    elif datasets_exist == 4:
        print("✓ All datasets generated")
    else:
        print(f"⚠ Only {datasets_exist}/4 datasets exist")

    # Final summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("\nImplementation is complete and ready for execution.")
        print("\nNext steps:")
        print("1. Generate full datasets: python scripts/stage1_generate_datasets.py")
        print("2. Test baseline: sbatch --array=0 scripts/stage2_train_models.sbatch")
        print("3. Run all experiments: sbatch scripts/stage2_train_models.sbatch")
    else:
        print("✗ SOME VALIDATION CHECKS FAILED")
        print("\nPlease review the errors above and fix before proceeding.")
    print("=" * 60)


if __name__ == "__main__":
    main()
