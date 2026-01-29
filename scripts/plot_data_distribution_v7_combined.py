#!/usr/bin/env python3
"""
Plot combined data distribution for v7 balanced dataset (single figure).
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

    total = len(train_data) + len(val_data) + len(test_data)
    print(f"  Total: {total} samples")

    # Extract and merge stats
    train_stats = extract_year_stats(train_data)
    val_stats = extract_year_stats(val_data)
    test_stats = extract_year_stats(test_data)
    all_stats = merge_year_stats(train_stats, val_stats, test_stats)

    # Get all years and sort
    all_years = sorted(set(all_stats["year_accept"].keys()) | set(all_stats["year_reject"].keys()))
    accepts = [all_stats["year_accept"].get(y, 0) for y in all_years]
    rejects = [all_stats["year_reject"].get(y, 0) for y in all_years]

    # Create single figure
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(all_years))
    width = 0.6

    # Stacked bar chart
    ax.bar(x, accepts, width, label='Accept', color=ACCEPT_COLOR)
    ax.bar(x, rejects, width, bottom=accepts, label='Reject', color=REJECT_COLOR)

    # Add count labels on bars
    for i, (a, r) in enumerate(zip(accepts, rejects)):
        if a > 0:
            ax.text(i, a / 2, str(a), ha='center', va='center', fontsize=ticksize - 1, fontweight='bold', color='white')
        if r > 0:
            ax.text(i, a + r / 2, str(r), ha='center', va='center', fontsize=ticksize - 1, fontweight='bold', color='white')

    # Calculate overall acceptance rate
    total_accept = sum(accepts)
    total_reject = sum(rejects)
    total = total_accept + total_reject
    acc_rate = 100.0 * total_accept / total if total > 0 else 0

    ax.set_xlabel("Year", fontsize=labelsize)
    ax.set_ylabel("Count", fontsize=labelsize)
    ax.set_title(f"Dataset Distribution by Year\nAcceptance Rate: {acc_rate:.1f}\\% ({total_accept}/{total})", fontsize=titlesize)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in all_years])
    ax.tick_params(axis='both', labelsize=ticksize)
    ax.legend(fontsize=legendsize, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_base = OUTPUT_DIR / "v7_balanced_combined"
    plt.savefig(f"{output_base}.pdf", dpi=200, bbox_inches='tight')
    print(f"Saved: {output_base}.pdf")
    plt.close()


if __name__ == "__main__":
    main()
