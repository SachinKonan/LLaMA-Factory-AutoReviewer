"""Dataset composition plots: samples per year, accept/reject balance, train/val/test sizes.

Usage:
    uv run python scripts/plot_dataset_composition.py
"""

import json, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Key dataset variants to show
DATASETS = {
    "2020-2025 Balanced (Vision)": {
        "train": "iclr_2020_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7_train",
        "val": "iclr_2020_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7_validation",
        "test": "iclr_2020_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7_test",
    },
    "Trainagreeing (Vision)": {
        "train": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7_train",
        "val": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7_validation",
        "test": "iclr_2020_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7_test",
    },
    "2020-2025 Balanced (Text)": {
        "train": "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_train",
        "val": "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_validation",
        "test": "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test",
    },
}


def load_dataset_stats(dataset_name):
    """Load a dataset and return per-year accept/reject counts."""
    path = os.path.join(DATA_DIR, dataset_name, "data.json")
    if not os.path.exists(path):
        return None

    with open(path) as f:
        data = json.load(f)

    year_counts = defaultdict(lambda: {"accept": 0, "reject": 0})
    for entry in data:
        meta = entry.get("_metadata", {})
        year = meta.get("year", "unknown")
        label = None
        for c in entry["conversations"]:
            if c["from"] == "gpt":
                label = "accept" if "Accept" in c["value"] else "reject"
                break
        if label:
            year_counts[year][label] += 1

    return dict(year_counts), len(data)


def main():
    print("Loading dataset statistics...")

    all_stats = {}
    for name, splits in DATASETS.items():
        all_stats[name] = {}
        for split, dataset_name in splits.items():
            stats = load_dataset_stats(dataset_name)
            if stats:
                year_counts, total = stats
                all_stats[name][split] = {"year_counts": year_counts, "total": total}
                print(f"  {name} / {split}: {total} papers")
            else:
                print(f"  {name} / {split}: NOT FOUND ({dataset_name})")

    # ---- Plot 1: Train/Val/Test sizes by dataset variant ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1a: Split sizes
    ax = axes[0]
    dataset_names = list(DATASETS.keys())
    split_names = ["train", "val", "test"]
    x = np.arange(len(dataset_names))
    width = 0.25
    colors = {"train": "#4CAF50", "val": "#FF9800", "test": "#2196F3"}

    for i, split in enumerate(split_names):
        sizes = []
        for name in dataset_names:
            if split in all_stats.get(name, {}):
                sizes.append(all_stats[name][split]["total"])
            else:
                sizes.append(0)
        bars = ax.bar(x + i * width, sizes, width, label=split.capitalize(), color=colors[split], alpha=0.8)
        for bar, size in zip(bars, sizes):
            if size > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                        f"{size:,}", ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels([n.replace(" (Vision)", "\n(Vision)").replace(" (Text)", "\n(Text)") for n in dataset_names], fontsize=8)
    ax.set_ylabel("Number of Papers")
    ax.set_title("Split Sizes by Dataset Variant")
    ax.legend(fontsize=9)

    # Plot 1b: Per-year distribution (train set of balanced vision)
    ax = axes[1]
    main_ds = "2020-2025 Balanced (Vision)"
    if "train" in all_stats.get(main_ds, {}):
        year_counts = all_stats[main_ds]["train"]["year_counts"]
        years = sorted(year_counts.keys())
        accepts = [year_counts[y]["accept"] for y in years]
        rejects = [year_counts[y]["reject"] for y in years]
        x = np.arange(len(years))
        ax.bar(x - 0.15, accepts, 0.3, label="Accept", color="#4CAF50", alpha=0.8)
        ax.bar(x + 0.15, rejects, 0.3, label="Reject", color="#F44336", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_ylabel("Count")
        ax.set_title(f"Per-Year Distribution\n({main_ds}, Train)")
        ax.legend(fontsize=9)

    # Plot 1c: Accept/reject balance across splits
    ax = axes[2]
    for i, (name, splits_data) in enumerate(all_stats.items()):
        for j, split in enumerate(["train", "test"]):
            if split not in splits_data:
                continue
            yc = splits_data[split]["year_counts"]
            total_accept = sum(yc[y]["accept"] for y in yc)
            total_reject = sum(yc[y]["reject"] for y in yc)
            total = total_accept + total_reject
            accept_pct = total_accept / total * 100 if total > 0 else 0
            bar_x = i * 2 + j
            ax.bar(bar_x, accept_pct, color="#4CAF50" if split == "train" else "#2196F3", alpha=0.8)
            ax.text(bar_x, accept_pct + 1, f"{accept_pct:.0f}%", ha='center', fontsize=8)

    labels = []
    for name in all_stats:
        short = name.split("(")[0].strip()[:15]
        labels.extend([f"{short}\ntrain", f"{short}\ntest"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Accept %")
    ax.set_title("Accept/Reject Balance")
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 65)

    plt.suptitle("Dataset Composition", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "dataset_composition.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
