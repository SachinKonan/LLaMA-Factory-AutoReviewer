#!/usr/bin/env python3
"""
Plot training loss curves for vision sweep experiments.

Usage:
    python scripts/plot_vision_hyperparam_variants.py --sweep vision_sweep --dataset vision
    python scripts/plot_vision_hyperparam_variants.py --sweep hyperparam_vision_sweep --dataset clean_images
    python scripts/plot_vision_hyperparam_variants.py --sweep all --dataset vision
"""

import argparse
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

# Color palette for different configs
COLORS = {
    # vision_sweep (768x768 cap)
    "vision_sweep_clean_images": "#2E86AB",
    "vision_sweep_vision": "#A23B72",
    # hyperparam_vision_sweep batch16
    "bs16_proj_frozen_vision": "#F18F01",
    "bs16_proj_unfrozen_vision": "#C73E1D",
    "bs16_proj_frozen_clean_images": "#3A86A8",
    "bs16_proj_unfrozen_clean_images": "#1B4965",
    # hyperparam_vision_sweep batch32
    "bs32_proj_frozen_vision": "#99C24D",
    "bs32_proj_unfrozen_vision": "#2E4057",
    "bs32_proj_frozen_clean_images": "#048A81",
    "bs32_proj_unfrozen_clean_images": "#54C6EB",
}

# Display names for legend
DISPLAY_NAMES = {
    # vision_sweep
    "vision_sweep_clean_images": "768² cap, bs16, frozen (clean+img)",
    "vision_sweep_vision": "768² cap, bs16, frozen (vision)",
    # hyperparam_vision_sweep batch16
    "bs16_proj_frozen_vision": "1280×28² cap, bs16, frozen (vision)",
    "bs16_proj_unfrozen_vision": "1280×28² cap, bs16, unfrozen (vision)",
    "bs16_proj_frozen_clean_images": "1280×28² cap, bs16, frozen (clean+img)",
    "bs16_proj_unfrozen_clean_images": "1280×28² cap, bs16, unfrozen (clean+img)",
    # hyperparam_vision_sweep batch32
    "bs32_proj_frozen_vision": "1280×28² cap, bs32, frozen (vision)",
    "bs32_proj_unfrozen_vision": "1280×28² cap, bs32, unfrozen (vision)",
    "bs32_proj_frozen_clean_images": "1280×28² cap, bs32, frozen (clean+img)",
    "bs32_proj_unfrozen_clean_images": "1280×28² cap, bs32, unfrozen (clean+img)",
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
                if "current_steps" in entry:
                    entry["step"] = entry["current_steps"]
                data.append(entry)
            except json.JSONDecodeError:
                continue

    return data


def get_dataset_type(name: str) -> str:
    """Extract dataset type from config name."""
    if "clean_images" in name or "clean+images" in name:
        return "clean_images"
    elif "vision" in name:
        return "vision"
    return "unknown"


def discover_vision_sweep_configs(saves_dir: Path) -> dict:
    """Discover configs in vision_sweep directory."""
    configs = {}
    sweep_dir = saves_dir / "vision_sweep"

    if not sweep_dir.exists():
        return configs

    for subdir in sorted(sweep_dir.iterdir()):
        if not subdir.is_dir():
            continue

        state_path = subdir / "trainer_state.json"
        log_path = subdir / "trainer_log.jsonl"

        config_key = f"vision_sweep_{subdir.name}"

        if state_path.exists():
            configs[config_key] = {
                "path": state_path,
                "log_type": "state",
                "dataset_type": get_dataset_type(subdir.name),
            }
        elif log_path.exists():
            configs[config_key] = {
                "path": log_path,
                "log_type": "log",
                "dataset_type": get_dataset_type(subdir.name),
            }

    return configs


def discover_hyperparam_vision_sweep_configs(saves_dir: Path) -> dict:
    """Discover configs in hyperparam_vision_sweep directory."""
    configs = {}
    sweep_dir = saves_dir / "hyperparam_vision_sweep"

    if not sweep_dir.exists():
        return configs

    for subdir in sorted(sweep_dir.iterdir()):
        if not subdir.is_dir():
            continue

        state_path = subdir / "trainer_state.json"
        log_path = subdir / "trainer_log.jsonl"

        if state_path.exists():
            configs[subdir.name] = {
                "path": state_path,
                "log_type": "state",
                "dataset_type": get_dataset_type(subdir.name),
            }
        elif log_path.exists():
            configs[subdir.name] = {
                "path": log_path,
                "log_type": "log",
                "dataset_type": get_dataset_type(subdir.name),
            }

    return configs


def plot_vision_losses(ax, saves_dir: Path, sweep: str, dataset_filter: str, max_y_val: float | None = None):
    """Plot training loss curves for vision sweep variants."""
    configs = {}

    # Discover configs based on sweep type
    if sweep in ["vision_sweep", "all"]:
        configs.update(discover_vision_sweep_configs(saves_dir))

    if sweep in ["hyperparam_vision_sweep", "all"]:
        configs.update(discover_hyperparam_vision_sweep_configs(saves_dir))

    # Filter by dataset type
    if dataset_filter != "all":
        configs = {k: v for k, v in configs.items() if v["dataset_type"] == dataset_filter}

    if not configs:
        ax.text(0.5, 0.5, "No data found",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=labelsize, color='gray')
        return

    for config_name, info in sorted(configs.items()):
        # Load data
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

        # Get color and display name
        color = COLORS.get(config_name, "#888888")
        display_name = DISPLAY_NAMES.get(config_name, config_name)

        # Calculate min loss for legend
        min_loss = min(losses)
        label = f"{display_name} $\\rightarrow$ {min_loss:.3f}"

        ax.plot(progress, losses, label=label, color=color,
                linestyle="-", linewidth=2.0, alpha=0.9)

    ax.set_xlabel(r"Progress (\%)", fontsize=labelsize)
    ax.set_ylabel(r"Loss", fontsize=labelsize)

    # Title based on sweep and dataset
    title_parts = ["Vision Sweep"]
    if sweep != "all":
        title_parts.append(f"({sweep})")
    if dataset_filter != "all":
        title_parts.append(f"- {dataset_filter}")
    ax.set_title(" ".join(title_parts), fontsize=titlesize, fontweight='bold')

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', labelsize=ticksize)
    ax.legend(fontsize=legendsize, loc='upper right')
    ax.set_xlim(0, 100)

    if max_y_val is not None:
        ax.set_ylim(top=max_y_val)


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves for vision sweep experiments.")
    parser.add_argument(
        "--sweep",
        type=str,
        default="all",
        choices=["vision_sweep", "hyperparam_vision_sweep", "all"],
        help="Which sweep to plot (vision_sweep, hyperparam_vision_sweep, or all)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["vision", "clean_images", "all"],
        help="Filter by dataset type (vision, clean_images, or all)"
    )
    parser.add_argument(
        "--max-y-val",
        type=float,
        default=None,
        help="Maximum y-axis value for the plot"
    )
    args = parser.parse_args()

    SAVES_DIR = Path("saves/qwen2.5-vl-7b/full")
    OUTPUT_DIR = Path("results/hyperparam_vision_sweep")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    plot_vision_losses(ax, SAVES_DIR, args.sweep, args.dataset, max_y_val=args.max_y_val)

    plt.tight_layout()

    # Output filename includes dataset filter if not 'all'
    suffix = f"_{args.dataset}" if args.dataset != "all" else ""
    output_path = OUTPUT_DIR / f"training_loss{suffix}.pdf"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save PNG
    output_path_png = OUTPUT_DIR / f"training_loss{suffix}.png"
    plt.savefig(output_path_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path_png}")

    plt.close()


if __name__ == "__main__":
    main()
