#!/usr/bin/env python3
"""
Plot data distribution for datasets showing year breakdown and token/page/image statistics.

Usage:
    python scripts/plot_data_distribution.py \\
        --clean iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3 \\
        --clean_images iclr_2020_2025_80_20_split5_balanced_deepreview_clean+images_binary_no_reviews_titleabs_corrected_v3 \\
        --vision iclr_2020_2025_80_20_split5_balanced_deepreview_vision_binary_no_reviews_titleabs_corrected_v3
"""

import argparse
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


def extract_token_stats(entries: list[dict]) -> dict:
    """Extract token counts from clean dataset (text tokens)."""
    accept_tokens = []
    reject_tokens = []

    for entry in entries:
        metadata = entry.get("_metadata", {})
        answer = metadata.get("answer", "").lower()

        conversations = entry.get("conversations", [])
        human_content = ""
        for msg in conversations:
            if msg.get("from") == "human":
                human_content = msg.get("value", "")
                break

        # Count tokens (approximate by character count / 4)
        tokens = len(human_content) // 4

        if answer == "accept":
            accept_tokens.append(tokens)
        elif answer == "reject":
            reject_tokens.append(tokens)

    return {
        "accept_tokens": accept_tokens,
        "reject_tokens": reject_tokens,
    }


def extract_image_stats(entries: list[dict]) -> dict:
    """Extract image counts from clean+images dataset."""
    accept_images = []
    reject_images = []

    for entry in entries:
        metadata = entry.get("_metadata", {})
        answer = metadata.get("answer", "").lower()

        conversations = entry.get("conversations", [])
        human_content = ""
        for msg in conversations:
            if msg.get("from") == "human":
                human_content = msg.get("value", "")
                break

        # Count images (number of <image> tags)
        images = human_content.count("<image>")

        if answer == "accept":
            accept_images.append(images)
        elif answer == "reject":
            reject_images.append(images)

    return {
        "accept_images": accept_images,
        "reject_images": reject_images,
    }


def extract_page_stats(entries: list[dict]) -> dict:
    """Extract page counts from vision dataset (pages = images in vision)."""
    accept_pages = []
    reject_pages = []

    for entry in entries:
        metadata = entry.get("_metadata", {})
        answer = metadata.get("answer", "").lower()

        conversations = entry.get("conversations", [])
        human_content = ""
        for msg in conversations:
            if msg.get("from") == "human":
                human_content = msg.get("value", "")
                break

        # Count pages (number of <image> tags in vision = pages)
        pages = human_content.count("<image>")

        if answer == "accept":
            accept_pages.append(pages)
        elif answer == "reject":
            reject_pages.append(pages)

    return {
        "accept_pages": accept_pages,
        "reject_pages": reject_pages,
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
    bars_accept = ax.bar(x, accepts, width, label='Accept', color=ACCEPT_COLOR)
    bars_reject = ax.bar(x, rejects, width, bottom=accepts, label='Reject', color=REJECT_COLOR)

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
    ax.legend(fontsize=legendsize, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')


def plot_metrics_distribution(ax, token_stats: dict, image_stats: dict, page_stats: dict, title: str):
    """Plot diverging bar chart of tokens, images, and pages for accepts vs rejects."""
    # Calculate means
    accept_tokens_mean = np.mean(token_stats["accept_tokens"]) if token_stats["accept_tokens"] else 0
    reject_tokens_mean = np.mean(token_stats["reject_tokens"]) if token_stats["reject_tokens"] else 0
    accept_images_mean = np.mean(image_stats["accept_images"]) if image_stats["accept_images"] else 0
    reject_images_mean = np.mean(image_stats["reject_images"]) if image_stats["reject_images"] else 0
    accept_pages_mean = np.mean(page_stats["accept_pages"]) if page_stats["accept_pages"] else 0
    reject_pages_mean = np.mean(page_stats["reject_pages"]) if page_stats["reject_pages"] else 0

    categories = ["Tokens (k)", "Images", "Pages"]
    x = np.arange(len(categories))
    width = 0.35

    # Scale tokens to thousands
    accept_vals = [accept_tokens_mean / 1000, accept_images_mean, accept_pages_mean]
    reject_vals = [-reject_tokens_mean / 1000, -reject_images_mean, -reject_pages_mean]

    ax.barh(x - width/2, accept_vals, width, label='Accept', color=ACCEPT_COLOR)
    ax.barh(x + width/2, reject_vals, width, label='Reject', color=REJECT_COLOR)

    # Add value labels
    for i, (a, r) in enumerate(zip(accept_vals, reject_vals)):
        if i == 0:  # Tokens
            ax.text(a + 0.2, i - width/2, f"{abs(a):.1f}k", ha='left', va='center', fontsize=ticksize - 1)
            ax.text(r - 0.2, i + width/2, f"{abs(r):.1f}k", ha='right', va='center', fontsize=ticksize - 1)
        else:  # Images/Pages
            ax.text(a + 0.2, i - width/2, f"{abs(a):.1f}", ha='left', va='center', fontsize=ticksize - 1)
            ax.text(r - 0.2, i + width/2, f"{abs(r):.1f}", ha='right', va='center', fontsize=ticksize - 1)

    ax.set_xlabel("Mean Value (Accept $\\rightarrow$ | $\\leftarrow$ Reject)", fontsize=labelsize)
    ax.set_title(title, fontsize=titlesize)
    ax.set_yticks(x)
    ax.set_yticklabels(categories)
    ax.tick_params(axis='both', labelsize=ticksize)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.legend(fontsize=legendsize, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')


def main():
    parser = argparse.ArgumentParser(description="Plot data distribution for datasets.")
    parser.add_argument(
        "--clean",
        type=str,
        required=True,
        help="Clean dataset name (for text tokens)"
    )
    parser.add_argument(
        "--clean_images",
        type=str,
        required=True,
        help="Clean+images dataset name (for image counts)"
    )
    parser.add_argument(
        "--vision",
        type=str,
        required=True,
        help="Vision dataset name (for page counts)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_distribution",
        help="Output filename (without extension)"
    )
    args = parser.parse_args()

    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("results/data_distribution")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all datasets
    print("Loading datasets...")
    clean_train = load_dataset(DATA_DIR, args.clean, "train")
    clean_test = load_dataset(DATA_DIR, args.clean, "test")
    clean_images_train = load_dataset(DATA_DIR, args.clean_images, "train")
    clean_images_test = load_dataset(DATA_DIR, args.clean_images, "test")
    vision_train = load_dataset(DATA_DIR, args.vision, "train")
    vision_test = load_dataset(DATA_DIR, args.vision, "test")

    if not clean_train:
        print(f"Error: No data found for clean dataset {args.clean}")
        return

    # Extract stats
    print("Extracting statistics...")
    # Year stats from clean dataset (should be same across all)
    train_year_stats = extract_year_stats(clean_train)
    test_year_stats = extract_year_stats(clean_test)

    # Token stats from clean dataset
    train_token_stats = extract_token_stats(clean_train)
    test_token_stats = extract_token_stats(clean_test)

    # Image stats from clean+images dataset
    train_image_stats = extract_image_stats(clean_images_train)
    test_image_stats = extract_image_stats(clean_images_test)

    # Page stats from vision dataset
    train_page_stats = extract_page_stats(vision_train)
    test_page_stats = extract_page_stats(vision_test)

    # Create 2x2 figure with larger size
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Row 1: Train
    plot_year_distribution(axes[0, 0], train_year_stats, "Train: Submissions per Year")
    plot_metrics_distribution(axes[0, 1], train_token_stats, train_image_stats, train_page_stats,
                              "Train: Tokens, Images \\& Pages (Mean)")

    # Row 2: Test
    plot_year_distribution(axes[1, 0], test_year_stats, "Test: Submissions per Year")
    plot_metrics_distribution(axes[1, 1], test_token_stats, test_image_stats, test_page_stats,
                              "Test: Tokens, Images \\& Pages (Mean)")

    plt.tight_layout()

    # Save
    output_base = OUTPUT_DIR / args.output
    plt.savefig(f"{output_base}.pdf", dpi=200, bbox_inches='tight')
    print(f"Saved: {output_base}.pdf")

    plt.savefig(f"{output_base}.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_base}.png")

    plt.close()


if __name__ == "__main__":
    main()
