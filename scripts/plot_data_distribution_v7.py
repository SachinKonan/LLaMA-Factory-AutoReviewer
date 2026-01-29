#!/usr/bin/env python3
"""
Plot data distribution for v7 balanced dataset showing:
- Top row: Combined (train + validation + test)
- Middle row: Train only
- Bottom row: Test only

Usage:
    python scripts/plot_data_distribution_v7.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Matplotlib styling
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "helvetica",
})

# Sizes
labelsize = 14
titlesize = 16
legendsize = 12
ticksize = 12

# Colors
ACCEPT_COLOR = "#4CAF50"  # Green
REJECT_COLOR = "#F44336"  # Red


def load_dataset(data_dir: Path, dataset_name: str, split: str) -> list[dict]:
    """Load dataset and return list of entries with metadata."""
    path = data_dir / f"{dataset_name}_{split}" / "data.json"
    if not path.exists():
        print(f"Warning: {path} not found")
        return []

    with open(path) as f:
        data = json.load(f)

    return data


def extract_year_stats(entries: list[dict]) -> dict:
    """Extract year counts from entries."""
    year_accept = defaultdict(int)
    year_reject = defaultdict(int)

    for entry in entries:
        metadata = entry.get("_metadata", {})
        year = metadata.get("year", "unknown")
        answer = metadata.get("answer", "").lower()

        if answer == "accept":
            year_accept[year] += 1
        elif answer == "reject":
            year_reject[year] += 1

    return {
        "year_accept": dict(year_accept),
        "year_reject": dict(year_reject),
    }


def plot_year_distribution(ax, stats: dict, title: str):
    """Plot stacked bar chart of submissions per year."""
    # Get all years and sort
    all_years = sorted(set(stats["year_accept"].keys()) | set(stats["year_reject"].keys()))

    if not all_years:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return

    # Get counts
    accepts = [stats["year_accept"].get(y, 0) for y in all_years]
    rejects = [stats["year_reject"].get(y, 0) for y in all_years]

    x = np.arange(len(all_years))
    width = 0.6

    # Stacked bar chart
    ax.bar(x, accepts, width, label='Accept', color=ACCEPT_COLOR)
    ax.bar(x, rejects, width, bottom=accepts, label='Reject', color=REJECT_COLOR)

    # Add count labels on bars
    for i, (a, r) in enumerate(zip(accepts, rejects)):
        if a > 0:
            ax.text(i, a / 2, str(a), ha='center', va='center', fontsize=ticksize - 2, fontweight='bold', color='white')
        if r > 0:
            ax.text(i, a + r / 2, str(r), ha='center', va='center', fontsize=ticksize - 2, fontweight='bold', color='white')

    # Calculate overall acceptance rate
    total_accept = sum(accepts)
    total_reject = sum(rejects)
    total = total_accept + total_reject
    acc_rate = 100.0 * total_accept / total if total > 0 else 0

    ax.set_xlabel("Year", fontsize=labelsize)
    ax.set_ylabel("Count", fontsize=labelsize)
    ax.set_title(f"{title}\nAcceptance Rate: {acc_rate:.1f}\\% ({total_accept}/{total})", fontsize=titlesize)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in all_years])
    ax.tick_params(axis='both', labelsize=ticksize)
    ax.legend(fontsize=legendsize, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')


def merge_year_stats(*stats_list) -> dict:
    """Merge multiple year stats into one."""
    merged_accept = defaultdict(int)
    merged_reject = defaultdict(int)

    for stats in stats_list:
        for year, count in stats["year_accept"].items():
            merged_accept[year] += count
        for year, count in stats["year_reject"].items():
            merged_reject[year] += count

    return {
        "year_accept": dict(merged_accept),
        "year_reject": dict(merged_reject),
    }


def main():
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("results/data_distribution")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_name = "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7"

    # Load all splits
    print("Loading datasets...")
    train_data = load_dataset(DATA_DIR, dataset_name, "train")
    val_data = load_dataset(DATA_DIR, dataset_name, "validation")
    test_data = load_dataset(DATA_DIR, dataset_name, "test")

    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Total: {len(train_data) + len(val_data) + len(test_data)} samples")

    # Extract stats
    print("Extracting statistics...")
    train_stats = extract_year_stats(train_data)
    val_stats = extract_year_stats(val_data)
    test_stats = extract_year_stats(test_data)

    # Merge all stats for combined view
    all_stats = merge_year_stats(train_stats, val_stats, test_stats)

    # Create 3x1 figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))

    # Row 1: Combined (All)
    plot_year_distribution(axes[0], all_stats, "All Data (Train + Validation + Test)")

    # Row 2: Train
    plot_year_distribution(axes[1], train_stats, "Train")

    # Row 3: Test
    plot_year_distribution(axes[2], test_stats, "Test")

    plt.tight_layout()

    # Save
    output_base = OUTPUT_DIR / "v7_balanced_distribution"
    plt.savefig(f"{output_base}.pdf", dpi=200, bbox_inches='tight')
    print(f"Saved: {output_base}.pdf")

    plt.savefig(f"{output_base}.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_base}.png")

    plt.close()


if __name__ == "__main__":
    main()
