#!/usr/bin/env python3
"""
Plot training loss curves for hyperparameter sweep experiments.

Usage:
    python scripts/plot_hyperparamsweep.py
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
legendsize = 12
ticksize = 14

# Color palette - distinct colors for different configs
COLORS = {
    "lr2.0e-5_cos_w0": "#FF6B6B",
    "lr2.0e-5_cos_w0.03": "#FF8E8E",
    "lr2.0e-5_const_w0": "#4ECDC4",
    "lr2.0e-5_const_w0.03": "#7EDDD6",
    "lr2.0e-6_cos_w0": "#45B7D1",
    "lr2.0e-6_cos_w0.03": "#74CAE0",
    "lr2.0e-6_const_w0": "#96CEB4",
    "lr2.0e-6_const_w0.03": "#B4DCC8",
}

# Line styles for warmup distinction
LINESTYLES = {
    "w0": "-",
    "w0.03": "--",
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


def parse_config_name(name: str) -> dict:
    """
    Parse config directory name like 'bs16_lr2.0e-6_cos_w0.03'.

    Returns dict with batch_size, lr, scheduler, warmup.
    """
    parts = name.split("_")
    config = {}

    for part in parts:
        if part.startswith("bs"):
            config["batch_size"] = int(part[2:])
        elif part.startswith("lr"):
            config["lr"] = part[2:]
        elif part in ["cos", "const"]:
            config["scheduler"] = "cosine" if part == "cos" else "constant"
        elif part.startswith("w"):
            config["warmup"] = part

    return config


def get_legend_label(config: dict, min_loss: float = None) -> str:
    """Create legend label from config with min loss."""
    sched = "cos" if config.get("scheduler") == "cosine" else "const"
    warmup = config.get("warmup", "w0")
    lr = config.get("lr", "?")
    base = f"lr={lr}, {sched}, {warmup}"
    if min_loss is not None:
        return f"{base} $\\rightarrow$ Min: {min_loss:.3f}"
    return base


def discover_configs(saves_dir: Path, batch_size: int) -> dict:
    """
    Discover all configs for a given batch size.

    Returns dict: {config_name: {"path": Path, "config": dict, "log_type": str}}
    """
    configs = {}
    prefix = f"bs{batch_size}_"

    if not saves_dir.exists():
        return configs

    for subdir in sorted(saves_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith(prefix):
            continue

        # Check for completed training (trainer_state.json) or in-progress (trainer_log.jsonl)
        state_path = subdir / "trainer_state.json"
        log_path = subdir / "trainer_log.jsonl"

        if state_path.exists():
            config = parse_config_name(subdir.name)
            configs[subdir.name] = {
                "path": state_path,
                "config": config,
                "log_type": "state",
            }
        elif log_path.exists():
            config = parse_config_name(subdir.name)
            configs[subdir.name] = {
                "path": log_path,
                "config": config,
                "log_type": "log",
            }

    return configs


def plot_batch_losses(ax, saves_dir: Path, batch_size: int):
    """Plot training loss curves for a given batch size."""
    configs = discover_configs(saves_dir, batch_size)

    if not configs:
        ax.text(0.5, 0.5, f"No data for batch size {batch_size}",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=labelsize, color='gray')
        return

    for config_name, info in sorted(configs.items()):
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

        # Get color and style
        config = info["config"]
        color_key = f"lr{config.get('lr', '')}_{config.get('scheduler', 'cos')[:3]}_{config.get('warmup', 'w0')}"
        color = COLORS.get(color_key, "#888888")
        warmup = config.get("warmup", "w0")
        linestyle = LINESTYLES.get(warmup, "-")

        # Calculate min loss for legend
        min_loss = min(losses)
        label = get_legend_label(config, min_loss)
        ax.plot(progress, losses, label=label, color=color,
                linestyle=linestyle, linewidth=2.0, alpha=0.9)

    ax.set_xlabel(r"Progress (\%)", fontsize=labelsize)
    ax.set_ylabel(r"Loss", fontsize=labelsize)
    ax.set_title(f"Batch Size {batch_size}", fontsize=titlesize, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', labelsize=ticksize)
    ax.legend(fontsize=legendsize, loc='upper right')
    ax.set_xlim(0, 100)


def main():
    SAVES_DIR = Path("saves/qwen2.5-7b/full/hyperparam_sweep")
    OUTPUT_DIR = Path("results/hyperparam_sweep")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create 2x1 figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    fig.suptitle(r"Hyperparameter Sweep: Training Loss Curves",
                 fontsize=titlesize + 2, fontweight='bold')

    # Top: Batch 16
    plot_batch_losses(axes[0], SAVES_DIR, batch_size=16)

    # Bottom: Batch 32
    plot_batch_losses(axes[1], SAVES_DIR, batch_size=32)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "training_loss.pdf"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
