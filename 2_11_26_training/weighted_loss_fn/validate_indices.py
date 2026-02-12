#!/usr/bin/env python3
"""
Validate SLURM array index mapping for weighted loss experiment.

This script demonstrates the index decoding logic used in the SLURM training
and inference scripts. Use it to verify experiment configurations before
submitting jobs.

Usage:
    python validate_indices.py
    python validate_indices.py --index 5
"""

import argparse


def decode_index(task_id: int) -> dict:
    """
    Decode SLURM array index to experiment configuration.

    Matches the logic in stage2_train_models.sbatch and stage3_run_inference.sbatch.
    """
    # BASELINE experiment (index 0)
    if task_id == 0:
        proportion = "1_2"
        variant = "baseline"
        gamma = 1.0

    # Balanced dataset weighted experiments (indices 1-6)
    elif task_id <= 6:
        adjusted_id = task_id - 1  # Offset to 0-5
        proportion = "1_2"
        variant_idx = adjusted_id // 3
        gamma_idx = adjusted_id % 3

        variants = ["accept", "reject"]
        gammas = [2.0, 4.0, 8.0]

        variant = variants[variant_idx]
        gamma = gammas[gamma_idx]

    # Imbalanced dataset experiments (indices 7-18)
    else:
        imbal_idx = task_id - 7  # Offset to 0-11

        # Determine proportion
        prop_idx = imbal_idx // 4
        proportions = ["1_3", "1_4", "1_8"]
        proportion = proportions[prop_idx]

        # Determine variant and gamma within this proportion
        local_idx = imbal_idx % 4
        variant_idx = local_idx // 2
        gamma_idx = local_idx % 2

        variants = ["accept", "reject"]
        gammas = [2.0, 4.0]

        variant = variants[variant_idx]
        gamma = gammas[gamma_idx]

    return {
        "index": task_id,
        "proportion": proportion,
        "variant": variant,
        "gamma": gamma,
        "experiment_name": f"{variant}_gamma{gamma}_prop{proportion}",
        "dataset": f"iclr_weighted_loss_train_{proportion}",
        "output_dir": f"saves/weighted_loss/{variant}_gamma{gamma}_prop{proportion}",
    }


def main():
    parser = argparse.ArgumentParser(description="Validate experiment index mapping")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Specific index to decode (0-18). If not provided, shows all.",
    )
    args = parser.parse_args()

    if args.index is not None:
        if args.index < 0 or args.index > 18:
            print(f"ERROR: Index must be 0-18, got {args.index}")
            return

        config = decode_index(args.index)
        print(f"\nIndex {args.index}:")
        print(f"  Proportion:      {config['proportion'].replace('_', '/')}")
        print(f"  Variant:         {config['variant']}")
        print(f"  Gamma:           {config['gamma']}")
        print(f"  Experiment Name: {config['experiment_name']}")
        print(f"  Dataset:         {config['dataset']}")
        print(f"  Output Dir:      {config['output_dir']}")
        if args.index == 0:
            print(f"  ** BASELINE ** (Standard BCE)")

    else:
        print("\nComplete Experiment Index Mapping (19 total):")
        print("=" * 90)

        # Baseline
        print("\nBASELINE:")
        print("-" * 90)
        config = decode_index(0)
        print(f"  [ 0] {config['variant']:8s} | gamma={config['gamma']:.1f} | {config['experiment_name']}")
        print("       ** STANDARD BCE (no weighting) **")

        # Group by proportion for readability
        for prop_name, prop_range in [
            ("Balanced (1/2) - Weighted", range(1, 7)),
            ("Proportion 1/3", range(7, 11)),
            ("Proportion 1/4", range(11, 15)),
            ("Proportion 1/8", range(15, 19)),
        ]:
            print(f"\n{prop_name}:")
            print("-" * 90)

            for idx in prop_range:
                config = decode_index(idx)
                print(
                    f"  [{idx:2d}] {config['variant']:6s} | gamma={config['gamma']:.1f} | {config['experiment_name']}"
                )

        print("\n" + "=" * 90)
        print(f"Total: 19 experiments")

        # Summary stats
        all_configs = [decode_index(i) for i in range(19)]

        print("\nSummary:")
        print(f"  Proportions: {sorted(set(c['proportion'] for c in all_configs))}")
        print(f"  Variants: {sorted(set(c['variant'] for c in all_configs))}")

        gamma_counts = {}
        for c in all_configs:
            gamma_counts[c["gamma"]] = gamma_counts.get(c["gamma"], 0) + 1
        print(f"  Gamma distribution: {dict(sorted(gamma_counts.items()))}")

        print("\nSubmission commands:")
        print("  All experiments:         sbatch --array=0-18 stage2_train_models.sbatch")
        print("  Baseline only:           sbatch --array=0 stage2_train_models.sbatch")
        print("  Balanced (w/ baseline):  sbatch --array=0-6 stage2_train_models.sbatch")
        print("  Imbalanced only:         sbatch --array=7-18 stage2_train_models.sbatch")
        print("  One proportion (e.g. 1/3): sbatch --array=7-10 stage2_train_models.sbatch")


if __name__ == "__main__":
    main()
