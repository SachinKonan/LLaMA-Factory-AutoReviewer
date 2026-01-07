#!/usr/bin/env python3
"""
Plot training loss curves for data sweep experiments.

Usage:
    python scripts/plot_datasweep.py
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Matplotlib styling
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "helvetica",
})

# Sizes
labelsize = 16
titlesize = 18
legendsize = 11
ticksize = 14

# Color palette for data sweep variants
COLORS = {
    "balanced": "#FF6B6B",
    "balanced_deepreview": "#4ECDC4",
    "balanced_deepreview_agreeing": "#45B7D1",
    "balanced_deepreview_seed10": "#96CEB4",
    "balanced_deepreview_seed20": "#FFEAA7",
    "original": "#DDA0DD",
    "original_max20k": "#F39C12",
}

# Display names for legend
DISPLAY_NAMES = {
    "balanced": "Balanced",
    "balanced_deepreview": "Balanced + DeepReview",
    "balanced_deepreview_agreeing": "Balanced + DeepReview (Agreeing)",
    "balanced_deepreview_seed10": "Balanced + DeepReview (Seed 10)",
    "balanced_deepreview_seed20": "Balanced + DeepReview (Seed 20)",
    "original": "Original",
    "original_max20k": "Original (Max 20k)",
}


def load_trainer_state(path: Path) -> list:
    """Load trainer_state.json and return log_history."""
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    return data.get("log_history", [])


def load_trainer_log(path: Path) -> list:
    """Load trainer_log.jsonl (line-delimited JSON during training)."""
    if not path.exists():
        return []

    data = []
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # Convert format to match trainer_state.json
                if "current_steps" in entry:
                    entry["step"] = entry["current_steps"]
                data.append(entry)
            except json.JSONDecodeError:
                continue

    return data


def discover_configs(saves_dir: Path) -> dict:
    """
    Discover all dataset configs in the data_sweep directory.

    Returns dict: {short_name: {"path": Path, "log_type": str}}
    """
    configs = {}

    if not saves_dir.exists():
        return configs

    for subdir in sorted(saves_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Check for completed training (trainer_state.json) or in-progress (trainer_log.jsonl)
        state_path = subdir / "trainer_state.json"
        log_path = subdir / "trainer_log.jsonl"

        if state_path.exists():
            configs[subdir.name] = {
                "path": state_path,
                "log_type": "state",
            }
        elif log_path.exists():
            configs[subdir.name] = {
                "path": log_path,
                "log_type": "log",
            }

    return configs


def plot_data_sweep_losses(ax, saves_dir: Path):
    """Plot training loss curves for all dataset variants."""
    configs = discover_configs(saves_dir)

    if not configs:
        ax.text(0.5, 0.5, "No data found",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=labelsize, color='gray')
        return

    for short_name, info in sorted(configs.items()):
        # Use appropriate loader based on log type
        if info.get("log_type") == "log":
            log_history = load_trainer_log(info["path"])
        else:
            log_history = load_trainer_state(info["path"])

        if not log_history:
            continue

        # Extract steps and losses
        steps = []
        losses = []
        for entry in log_history:
            if "loss" in entry and "step" in entry:
                steps.append(entry["step"])
                losses.append(entry["loss"])

        if not steps:
            continue

        # Convert steps to progress percentage
        max_step = max(steps)
        progress = [100.0 * s / max_step for s in steps]

        # Get color
        color = COLORS.get(short_name, "#888888")

        # Calculate min loss for legend
        min_loss = min(losses)
        display_name = DISPLAY_NAMES.get(short_name, short_name)
        label = f"{display_name} $\\rightarrow$ Min: {min_loss:.3f}"

        ax.plot(progress, losses, label=label, color=color,
                linestyle="-", linewidth=2.0, alpha=0.9)

    ax.set_xlabel(r"Progress (\%)", fontsize=labelsize)
    ax.set_ylabel(r"Loss", fontsize=labelsize)
    ax.set_title(r"Data Sweep: Training Loss Curves", fontsize=titlesize, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', labelsize=ticksize)
    ax.legend(fontsize=legendsize, loc='upper right')
    ax.set_xlim(0, 100)


def main():
    SAVES_DIR = Path("saves/qwen2.5-7b/full/data_sweep")
    OUTPUT_DIR = Path("results/data_sweep")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    plot_data_sweep_losses(ax, SAVES_DIR)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "training_loss.pdf"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save PNG for quick viewing
    output_path_png = OUTPUT_DIR / "training_loss.png"
    plt.savefig(output_path_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path_png}")

    plt.close()


if __name__ == "__main__":
    main()
