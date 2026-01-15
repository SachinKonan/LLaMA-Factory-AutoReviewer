#!/usr/bin/env python3
"""
Plot training loss curves and test accuracy for grid_searchv2 experiments.

Usage:
    python scripts/plot_gridsearchv2_training.py
    python scripts/plot_gridsearchv2_training.py --max-y-val 1.5
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# Matplotlib styling
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "helvetica",
})

# Sizes
labelsize = 20
titlesize = 20
legendsize = 20
ticksize = 16

# Color palette
red = "#FF8988"
orange = "#FECC81"
blue = "#6098FF"
green = "#77B25D"
purple = "#B28CFF"
colors = [red, orange, blue, green, purple]

# Target types for columns
TARGET_TYPES = [
    ("no_reviews", "No Reviews"),
    ("normalized_normalizedmeta", "Meta Only"),
    ("normalized_3normalizedreviews_and_normalizedmeta", "Reviews + Meta"),
]

# Indicator types for rows
INDICATOR_TYPES = [
    ("binary", "Binary"),
    ("multiclass", "Multiclass"),
    ("citation", "Citation"),
]


def load_trainer_log(path: Path) -> pd.DataFrame:
    """Load trainer_log.jsonl as dataframe."""
    if not path.exists():
        return pd.DataFrame()

    data = []
    with open(path) as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def parse_dataset_name(name: str) -> tuple[str, str, str]:
    """
    Parse dataset name to extract text_type, indicator, and target.

    Returns (text_type, indicator, target)
    """
    # Remove iclr_ prefix and _v2 suffix
    name = name.replace("iclr_", "")
    if name.endswith("_v2"):
        name = name[:-3]

    # Determine text_type
    if "clean-title+abstract" in name:
        text_type = "clean-title+abstract"
        name = name.replace("clean-title+abstract_", "")
    elif name.startswith("clean_"):
        text_type = "clean"
        name = name.replace("clean_", "")
    elif name.startswith("text_"):
        text_type = "text"
        # Handle text_binary_20480_... pattern
        parts = name.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            text_type = f"text_{parts[2]}"
            name = "_".join([parts[1]] + parts[3:])
        else:
            name = "_".join(parts[1:])
    else:
        text_type = "unknown"

    # Determine indicator
    for indicator in ["binary", "multiclass", "citation"]:
        if name.startswith(f"{indicator}_"):
            name = name[len(indicator) + 1:]
            break
    else:
        indicator = "unknown"

    # Rest is target
    target = name

    return text_type, indicator, target


def discover_training_logs(saves_dir: Path, text_type_filter: str = None) -> dict:
    """
    Discover all trainer_log.jsonl files.

    Returns dict: {(indicator, target): {"path": Path, "name": str}}
    """
    logs = {}

    if not saves_dir.exists():
        print(f"Warning: {saves_dir} does not exist")
        return logs

    for subdir in sorted(saves_dir.iterdir()):
        if not subdir.is_dir():
            continue

        log_path = subdir / "trainer_log.jsonl"
        if not log_path.exists():
            continue

        text_type, indicator, target = parse_dataset_name(subdir.name)

        # Apply text_type filter if specified
        if text_type_filter and text_type != text_type_filter:
            continue

        key = (indicator, target)
        if key not in logs:
            logs[key] = {"path": log_path, "name": subdir.name, "text_type": text_type}

    return logs


def plot_training_curves(saves_dir: Path, output_path: Path, text_type_filter: str = None, max_y_val: float | None = None):
    """Create 3x3 grid of training loss curves."""

    # Discover logs
    logs = discover_training_logs(saves_dir, text_type_filter)

    if not logs:
        print("No training logs found!")
        return

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    title_text = r"Training Loss Curves"
    if text_type_filter:
        title_text += f" ({text_type_filter})"
    fig.suptitle(title_text, fontsize=titlesize, fontweight='bold')

    for row_idx, (indicator, indicator_label) in enumerate(INDICATOR_TYPES):
        for col_idx, (target, target_label) in enumerate(TARGET_TYPES):
            ax = axes[row_idx, col_idx]

            key = (indicator, target)

            if key in logs:
                log_info = logs[key]
                df = load_trainer_log(log_info["path"])

                if not df.empty and "percentage" in df.columns and "loss" in df.columns:
                    # Filter out NaN values
                    df_clean = df[df["loss"].notna()].copy()

                    if len(df_clean) > 0:
                        # Use color based on row
                        color = colors[row_idx % len(colors)]

                        ax.plot(df_clean["percentage"], df_clean["loss"],
                               linewidth=3.0, marker="o", markersize=4,
                               alpha=0.9, color=color, markevery=max(1, len(df_clean)//20))

                        # Get start and final loss (non-NaN)
                        start_loss = df_clean["loss"].iloc[0]
                        final_loss = df_clean["loss"].iloc[-1]

                        # Add start/final loss annotation
                        ax.annotate(f"Start: {start_loss:.3f}\nFinal: {final_loss:.3f}",
                                   xy=(0.95, 0.95), xycoords='axes fraction',
                                   ha='right', va='top', fontsize=ticksize-2,
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                    else:
                        ax.text(0.5, 0.5, r"No valid data", ha='center', va='center',
                               transform=ax.transAxes, fontsize=labelsize)
                else:
                    ax.text(0.5, 0.5, r"No data", ha='center', va='center',
                           transform=ax.transAxes, fontsize=labelsize)
            else:
                ax.text(0.5, 0.5, r"Not found", ha='center', va='center',
                       transform=ax.transAxes, color='gray', fontsize=labelsize)

            # Labels
            ax.set_xlabel(r"Progress (\%)", fontsize=labelsize)
            ax.set_ylabel(r"Loss", fontsize=labelsize)

            # Set column title (only top row)
            if row_idx == 0:
                ax.set_title(target_label, fontsize=titlesize, fontweight='bold')

            # Set row label (only left column)
            if col_idx == 0:
                ax.set_ylabel(f"{indicator_label}\n" + r"Loss", fontsize=labelsize)

            # Styling
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', labelsize=ticksize)
            if max_y_val is not None:
                ax.set_ylim(top=max_y_val)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', transparent=False)
    print(f"Saved: {output_path}")
    plt.close()


def load_results_from_csv(results_dir: Path, text_type_filter: str = None) -> dict:
    """
    Load test results from CSV files generated by analyze.py.

    Returns dict: {(indicator, target): [(iteration, metric_value), ...]}
    """
    results = {}

    # Load each CSV file
    csv_files = {
        "binary": results_dir / "binary_results.csv",
        "multiclass": results_dir / "multiclass_results.csv",
        "citation": results_dir / "citation_results.csv",
    }

    for indicator, csv_path in csv_files.items():
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            dataset = row["dataset"]
            run = row["run"]

            # Parse dataset name
            text_type, _, target = parse_dataset_name(dataset)

            # Apply text_type filter
            if text_type_filter and text_type != text_type_filter:
                continue

            # Parse iteration from run name
            if run == "base":
                iteration = 0
            elif run.startswith("finetuned"):
                try:
                    iteration = int(run.replace("finetuned", ""))
                except ValueError:
                    continue
            else:
                continue

            # Get metric value
            if indicator == "citation":
                # Use 1 - MAE for citation
                metric_value = 1.0 - row["mae"] if pd.notna(row["mae"]) else None
            else:
                # Use accuracy for binary/multiclass
                metric_value = row["accuracy"] if pd.notna(row["accuracy"]) else None

            if metric_value is None:
                continue

            key = (indicator, target)
            if key not in results:
                results[key] = []

            results[key].append((iteration, metric_value))

    # Sort each by iteration
    for key in results:
        results[key].sort(key=lambda x: x[0])

    return results


def plot_test_accuracy(results_dir: Path, output_path: Path, text_type_filter: str = None, max_y_val: float | None = None):
    """Create 3x3 grid of test accuracy vs iterations."""

    # Load results from CSV files
    results = load_results_from_csv(results_dir, text_type_filter)

    if not results:
        print("No test results found!")
        return

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    title_text = r"Test Accuracy vs Training Iterations"
    if text_type_filter:
        title_text += f" ({text_type_filter})"
    fig.suptitle(title_text, fontsize=titlesize, fontweight='bold')

    for row_idx, (indicator, indicator_label) in enumerate(INDICATOR_TYPES):
        for col_idx, (target, target_label) in enumerate(TARGET_TYPES):
            ax = axes[row_idx, col_idx]

            key = (indicator, target)

            if key in results and len(results[key]) > 0:
                data = results[key]
                iterations = [d[0] for d in data]
                accuracies = [d[1] for d in data]

                # Use color based on row
                color = colors[row_idx % len(colors)]

                ax.plot(iterations, accuracies,
                       linewidth=3.0, marker="o", markersize=8,
                       alpha=0.9, color=color)

                # Add start/final accuracy annotation
                start_acc = accuracies[0]
                final_acc = accuracies[-1]
                final_iter = iterations[-1]

                y_label = "1-MAE" if indicator == "citation" else "Accuracy"

                ax.annotate(f"Base: {start_acc:.3f}\nFinal ({final_iter}): {final_acc:.3f}",
                           xy=(0.95, 0.05), xycoords='axes fraction',
                           ha='right', va='bottom', fontsize=ticksize-2,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            else:
                ax.text(0.5, 0.5, r"Not found", ha='center', va='center',
                       transform=ax.transAxes, color='gray', fontsize=labelsize)

            # Labels
            ax.set_xlabel(r"Training Iterations", fontsize=labelsize)
            y_label = "1-MAE" if indicator == "citation" else "Accuracy"
            ax.set_ylabel(y_label, fontsize=labelsize)

            # Set column title (only top row)
            if row_idx == 0:
                ax.set_title(target_label, fontsize=titlesize, fontweight='bold')

            # Set row label (only left column)
            if col_idx == 0:
                ax.set_ylabel(f"{indicator_label}\n{y_label}", fontsize=labelsize)

            # Styling
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', labelsize=ticksize)
            if max_y_val is not None:
                ax.set_ylim(top=max_y_val)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', transparent=False)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves and test accuracy for grid_searchv2 experiments.")
    parser.add_argument(
        "--max-y-val",
        type=float,
        default=None,
        help="Maximum y-axis value for the plot"
    )
    args = parser.parse_args()

    # Constants for grid_searchv2
    DIR = "grid_searchv2"
    SAVES_DIR = Path("saves/qwen2.5-7b/full/grid_searchv2")
    RESULTS_DIR = Path("results") / DIR
    OUTPUT_DIR = Path("results") / DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate plots for each text type
    for text_type in ["clean", "clean-title+abstract"]:
        # Training loss curves
        output_path = OUTPUT_DIR / f"training_loss_{text_type}.pdf"
        plot_training_curves(SAVES_DIR, output_path, text_type, max_y_val=args.max_y_val)

        # Test accuracy vs iterations
        output_path = OUTPUT_DIR / f"test_accuracy_{text_type}.pdf"
        plot_test_accuracy(RESULTS_DIR, output_path, text_type, max_y_val=args.max_y_val)


if __name__ == "__main__":
    main()
