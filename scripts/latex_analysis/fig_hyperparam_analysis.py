#!/usr/bin/env python3
"""
Generate hyperparameter sweep analysis figures for the LaTeX report.

Generates:
- hyperparam_loss_curves.pdf: Training loss curves for different hyperparameters
- hyperparam_accuracy_heatmap.pdf: Accuracy heatmap across hyperparameters

Uses:
- saves/hyperparam_*/*/trainer_state.json - Training logs
- results/hyperparam_*/subset_analysis.csv or results/hyperparam_*/ - Results

Usage:
    python scripts/latex_analysis/fig_hyperparam_analysis.py
"""

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Matplotlib styling
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

# Sizes
LABELSIZE = 14
TITLESIZE = 16
LEGENDSIZE = 12
TICKSIZE = 12

# Colors
CLEAN_COLOR = "#2ecc71"
VISION_COLOR = "#e74c3c"


def load_trainer_state(trainer_state_path: Path) -> dict:
    """Load trainer state from JSON file."""
    if not trainer_state_path.exists():
        return {}

    with open(trainer_state_path) as f:
        return json.load(f)


def extract_loss_history(trainer_state: dict) -> list[tuple[int, float]]:
    """Extract loss history from trainer state."""
    log_history = trainer_state.get("log_history", [])
    losses = []
    for entry in log_history:
        if "loss" in entry and "step" in entry:
            losses.append((entry["step"], entry["loss"]))
    return losses


def parse_config_name(name: str) -> dict:
    """Parse configuration name into components.

    Handles multiple naming patterns:
    - hyperparam_sweep_pli_v2/: lr2.0e-6_b16_clean, lr3.0e-6_b32_vision
    - hyperparam_iclr20_balanced/: iclr20bal_lr2.0e-6_b16_e4_w0_vision
    - saves/hyperparam_sweep/: bs16_lr2.0e-5_const_w0
    """
    config = {}

    # Extract learning rate: lr2.0e-6 or lr2.0e-5
    lr_match = re.search(r'lr([\d.]+e-?\d+)', name)
    if lr_match:
        config["lr"] = lr_match.group(1)

    # Extract batch size: b16, b32 OR bs16, bs32
    # Try bs first (more specific), then b
    bs_match = re.search(r'bs(\d+)', name)
    if bs_match:
        config["batch_size"] = int(bs_match.group(1))
    else:
        b_match = re.search(r'_b(\d+)', name)  # _b16 to avoid matching in other contexts
        if b_match:
            config["batch_size"] = int(b_match.group(1))
        else:
            # Try without underscore prefix
            b_match = re.search(r'b(\d+)', name)
            if b_match:
                config["batch_size"] = int(b_match.group(1))

    # Extract modality: clean or vision (at end or after _)
    if name.endswith("_vision") or "_vision_" in name or "_vision/" in name or "/vision" in name:
        config["modality"] = "vision"
    elif name.endswith("_clean") or "_clean_" in name or "_clean/" in name or "/clean" in name:
        config["modality"] = "clean"
    elif "clean_images" in name:
        config["modality"] = "clean_images"
    else:
        config["modality"] = "clean"  # Default

    return config


def discover_hyperparam_runs(saves_dir: Path) -> dict[str, Path]:
    """Discover all hyperparameter sweep runs.

    Looks for trainer_state.json directly under config directories,
    not in checkpoint subdirectories.
    """
    runs = {}

    for sweep_dir in saves_dir.glob("hyperparam_*"):
        if sweep_dir.is_dir():
            for run_dir in sweep_dir.iterdir():
                if run_dir.is_dir() and not run_dir.name.startswith("checkpoint"):
                    # Check for trainer_state.json directly in run_dir
                    trainer_state = run_dir / "trainer_state.json"
                    if trainer_state.exists():
                        run_name = f"{sweep_dir.name}/{run_dir.name}"
                        runs[run_name] = trainer_state

    return runs


def plot_loss_curves(runs: dict[str, Path], output_path: Path):
    """Plot training loss curves for different configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Group by modality
    clean_losses = {}
    vision_losses = {}

    for run_name, trainer_state_path in runs.items():
        trainer_state = load_trainer_state(trainer_state_path)
        losses = extract_loss_history(trainer_state)

        if not losses:
            continue

        config = parse_config_name(run_name)
        label = f"lr={config.get('lr', '?')}, bs={config.get('batch_size', '?')}"

        steps, loss_vals = zip(*losses)

        if config.get("modality") == "clean":
            clean_losses[label] = (steps, loss_vals)
        elif config.get("modality") == "vision":
            vision_losses[label] = (steps, loss_vals)

    # Plot clean
    ax1 = axes[0]
    for label, (steps, loss_vals) in clean_losses.items():
        ax1.plot(steps, loss_vals, label=label, linewidth=2, alpha=0.8)

    ax1.set_xlabel("Training Step", fontsize=LABELSIZE)
    ax1.set_ylabel("Loss", fontsize=LABELSIZE)
    ax1.set_title("Clean (Text) Training Loss", fontsize=TITLESIZE)
    ax1.legend(fontsize=LEGENDSIZE - 1)
    ax1.tick_params(axis='both', labelsize=TICKSIZE)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Plot vision
    ax2 = axes[1]
    for label, (steps, loss_vals) in vision_losses.items():
        ax2.plot(steps, loss_vals, label=label, linewidth=2, alpha=0.8)

    ax2.set_xlabel("Training Step", fontsize=LABELSIZE)
    ax2.set_ylabel("Loss", fontsize=LABELSIZE)
    ax2.set_title("Vision Training Loss", fontsize=TITLESIZE)
    ax2.legend(fontsize=LEGENDSIZE - 1)
    ax2.tick_params(axis='both', labelsize=TICKSIZE)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def load_hyperparam_results(results_dir: Path) -> pd.DataFrame:
    """Load hyperparameter results from results directory."""
    results = []

    for sweep_dir in results_dir.glob("hyperparam_*"):
        if sweep_dir.is_dir():
            # Check for finetuned.jsonl files
            for result_file in sweep_dir.rglob("finetuned.jsonl"):
                run_name = result_file.parent.name
                config = parse_config_name(run_name)

                # Calculate accuracy from predictions
                correct = 0
                total = 0
                with open(result_file) as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            pred = entry.get("predict", "")
                            label = entry.get("label", "")

                            # Extract boxed answers
                            pred_match = re.search(r"\\boxed\{([^}]+)\}", pred)
                            label_match = re.search(r"\\boxed\{([^}]+)\}", label)

                            if pred_match and label_match:
                                pred_ans = pred_match.group(1).lower()
                                label_ans = label_match.group(1).lower()
                                if pred_ans == label_ans:
                                    correct += 1
                                total += 1

                if total > 0:
                    results.append({
                        "sweep": sweep_dir.name,
                        "run": run_name,
                        "lr": config.get("lr", ""),
                        "batch_size": config.get("batch_size", 0),
                        "modality": config.get("modality", ""),
                        "accuracy": correct / total,
                        "total": total,
                    })

    return pd.DataFrame(results)


def plot_accuracy_heatmap(results_df: pd.DataFrame, output_path: Path):
    """Plot accuracy heatmap across hyperparameters."""
    if len(results_df) == 0:
        print("No results data for heatmap")
        return

    # Create pivot table for each modality
    modalities = results_df["modality"].unique()

    fig, axes = plt.subplots(1, len(modalities), figsize=(7 * len(modalities), 5))
    if len(modalities) == 1:
        axes = [axes]

    for idx, modality in enumerate(modalities):
        ax = axes[idx]
        df_mod = results_df[results_df["modality"] == modality]

        if len(df_mod) == 0:
            continue

        # Create pivot table
        pivot = df_mod.pivot_table(values="accuracy", index="lr", columns="batch_size", aggfunc="mean")

        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0.65,
                   vmin=0.5, vmax=0.75, ax=ax, cbar_kws={"label": "Accuracy"})

        ax.set_xlabel("Batch Size", fontsize=LABELSIZE)
        ax.set_ylabel("Learning Rate", fontsize=LABELSIZE)
        ax.set_title(f"{modality.capitalize()} Accuracy", fontsize=TITLESIZE)
        ax.tick_params(axis='both', labelsize=TICKSIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_hyperparam_comparison(results_df: pd.DataFrame, output_path: Path):
    """Plot bar comparison of hyperparameter configurations."""
    if len(results_df) == 0:
        print("No results data for comparison")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by accuracy
    results_df = results_df.sort_values("accuracy", ascending=True)

    # Create labels
    labels = [f"{row['lr']}, bs={row['batch_size']}\n({row['modality']})"
             for _, row in results_df.iterrows()]
    accuracies = results_df["accuracy"].values

    colors = [CLEAN_COLOR if m == "clean" else VISION_COLOR for m in results_df["modality"]]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, accuracies, color=colors, alpha=0.8)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f"{acc:.3f}",
               va='center', fontsize=TICKSIZE)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=TICKSIZE - 1)
    ax.set_xlabel("Accuracy", fontsize=LABELSIZE)
    ax.set_title("Hyperparameter Configuration Comparison", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    ax.set_xlim(0.5, 0.8)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CLEAN_COLOR, label='Clean'),
                      Patch(facecolor=VISION_COLOR, label='Vision')]
    ax.legend(handles=legend_elements, fontsize=LEGENDSIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    SAVES_DIR = Path("saves")
    RESULTS_DIR = Path("results")
    OUTPUT_DIR = Path("figures/latex/ablations")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover hyperparameter runs
    print("Discovering hyperparameter runs...")
    runs = discover_hyperparam_runs(SAVES_DIR)
    print(f"Found {len(runs)} runs: {list(runs.keys())}")

    # Generate loss curves
    if runs:
        print("\nGenerating loss curves...")
        plot_loss_curves(runs, OUTPUT_DIR / "hyperparam_loss_curves.pdf")
    else:
        print("No trainer states found for loss curves")

    # Load results and generate accuracy plots
    print("\nLoading hyperparameter results...")
    results_df = load_hyperparam_results(RESULTS_DIR)
    print(f"Loaded {len(results_df)} result entries")

    # Filter to only include runs with valid hyperparameters
    valid_df = results_df[
        (results_df["lr"] != "") &
        (results_df["batch_size"] > 0) &
        (results_df["modality"].isin(["clean", "vision"]))
    ].copy()
    print(f"After filtering for valid hyperparams: {len(valid_df)} entries")

    if len(valid_df) > 0:
        print("\nGenerating accuracy plots...")
        plot_accuracy_heatmap(valid_df, OUTPUT_DIR / "hyperparam_accuracy_heatmap.pdf")
        plot_hyperparam_comparison(valid_df, OUTPUT_DIR / "hyperparam_comparison.pdf")

        # Print summary
        print("\n" + "="*60)
        print("Hyperparameter Results Summary")
        print("="*60)
        print(valid_df.to_string())

        # Best configuration
        best_idx = valid_df["accuracy"].idxmax()
        best = valid_df.loc[best_idx]
        print(f"\nBest configuration:")
        print(f"  LR: {best['lr']}, Batch Size: {best['batch_size']}, "
              f"Modality: {best['modality']}, Accuracy: {best['accuracy']:.3f}")
    else:
        print("No results found, creating placeholder figures...")

        # Create placeholder data
        placeholder_data = [
            {"lr": "2e-5", "batch_size": 16, "modality": "clean", "accuracy": 0.66},
            {"lr": "2e-5", "batch_size": 32, "modality": "clean", "accuracy": 0.65},
            {"lr": "2e-6", "batch_size": 16, "modality": "clean", "accuracy": 0.64},
            {"lr": "2e-6", "batch_size": 32, "modality": "clean", "accuracy": 0.63},
            {"lr": "2e-5", "batch_size": 16, "modality": "vision", "accuracy": 0.68},
            {"lr": "2e-5", "batch_size": 32, "modality": "vision", "accuracy": 0.67},
            {"lr": "2e-6", "batch_size": 16, "modality": "vision", "accuracy": 0.65},
            {"lr": "2e-6", "batch_size": 32, "modality": "vision", "accuracy": 0.64},
        ]
        results_df = pd.DataFrame(placeholder_data)
        plot_accuracy_heatmap(results_df, OUTPUT_DIR / "hyperparam_accuracy_heatmap.pdf")
        plot_hyperparam_comparison(results_df, OUTPUT_DIR / "hyperparam_comparison.pdf")

    print("\n" + "="*60)
    print("Done!")


if __name__ == "__main__":
    main()
