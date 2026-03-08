#!/usr/bin/env python3
"""
Q1 Analysis: Visualize reviewer archetype clusters.

Generates:
- review_embeddings_umap.png: UMAP visualization of review clusters
- review_embeddings_tsne.png: t-SNE visualization of review clusters
- archetype_distribution.png: Bar chart of cluster frequencies
- cluster_characteristics.png: Heatmap of cluster characteristics

Usage:
    python 2_8_26/q1_reviewer_archetypes/analyze.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.analysis_utils import save_metrics_json, setup_plot_style

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")


def plot_umap(embeddings, labels, cluster_info, save_path):
    """Plot UMAP visualization of review embeddings."""
    try:
        import umap
    except ImportError:
        print("Warning: umap-learn not installed. Skipping UMAP plot.")
        print("Install with: pip install umap-learn")
        return

    setup_plot_style()

    print(f"Running UMAP on {len(embeddings)} embeddings...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))

    n_clusters = len(cluster_info)
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        info = cluster_info[str(cluster_id)]
        label = f"C{cluster_id}: n={info['size']}, rating={info.get('avg_rating', 0):.1f}"

        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=[colors[cluster_id]],
                  label=label,
                  s=20,
                  alpha=0.6,
                  edgecolors='black',
                  linewidth=0.3)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("Q1: Reviewer Archetypes (UMAP Projection)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_tsne(embeddings, labels, cluster_info, save_path):
    """Plot t-SNE visualization of review embeddings."""
    from sklearn.manifold import TSNE

    setup_plot_style()

    print(f"Running t-SNE on {len(embeddings)} embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))

    n_clusters = len(cluster_info)
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        info = cluster_info[str(cluster_id)]
        label = f"C{cluster_id}: n={info['size']}, rating={info.get('avg_rating', 0):.1f}"

        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=[colors[cluster_id]],
                  label=label,
                  s=20,
                  alpha=0.6,
                  edgecolors='black',
                  linewidth=0.3)

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("Q1: Reviewer Archetypes (t-SNE Projection)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_distribution(reviews, cluster_info, save_path):
    """Plot archetype distribution bar chart."""
    setup_plot_style()

    n_clusters = len(cluster_info)
    clusters = [r["cluster"] for r in reviews]

    counts = np.zeros(n_clusters)
    for c in clusters:
        counts[c] += 1

    freq = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(n_clusters)

    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    bars = ax.bar(x, freq, color=colors, edgecolor="black", linewidth=1.5, alpha=0.8)

    # Add percentage labels on bars
    for bar, val, count in zip(bars, freq, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                f"{val:.1%}\n(n={int(count)})",
                ha="center", va="bottom", fontsize=11, fontweight='bold')

    # Add cluster descriptions
    descriptions = []
    for i in range(n_clusters):
        info = cluster_info[str(i)]
        keywords = ', '.join(info['top_keywords'][:3])
        rating = info.get('avg_rating', 0)
        ratio = info.get('weakness_to_strength_ratio', 0)
        desc = f"C{i}: {keywords}\nRating: {rating:.1f}, W/S: {ratio:.1f}"
        descriptions.append(desc)

    ax.set_xticks(x)
    ax.set_xticklabels(descriptions, fontsize=9)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Q1: Reviewer Archetype Distribution", fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(freq) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_cluster_characteristics(cluster_info, save_path):
    """Plot heatmap of cluster characteristics."""
    setup_plot_style()

    n_clusters = len(cluster_info)

    # Extract characteristics
    characteristics = ['avg_rating', 'avg_length', 'weakness_to_strength_ratio',
                      'avg_strengths_len', 'avg_weaknesses_len']
    char_labels = ['Avg Rating', 'Avg Length', 'W/S Ratio',
                   'Strengths Len', 'Weaknesses Len']

    data = np.zeros((len(characteristics), n_clusters))
    for i, char in enumerate(characteristics):
        for j in range(n_clusters):
            info = cluster_info[str(j)]
            if char in info:
                data[i, j] = info[char]

    # Normalize each row for better visualization
    data_norm = np.zeros_like(data)
    for i in range(len(characteristics)):
        row_min = data[i].min()
        row_max = data[i].max()
        if row_max > row_min:
            data_norm[i] = (data[i] - row_min) / (row_max - row_min)
        else:
            data_norm[i] = 0.5

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Add actual values as text
    for i in range(len(characteristics)):
        for j in range(n_clusters):
            text = ax.text(j, i, f"{data[i, j]:.0f}" if i > 0 else f"{data[i, j]:.1f}",
                         ha="center", va="center", color="black", fontsize=11, fontweight='bold')

    ax.set_xticks(np.arange(n_clusters))
    ax.set_yticks(np.arange(len(characteristics)))
    ax.set_xticklabels([f"Cluster {i}" for i in range(n_clusters)], fontsize=12)
    ax.set_yticklabels(char_labels, fontsize=12)
    ax.set_title("Q1: Cluster Characteristics (Normalized Heatmap)", fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Normalized Value (0=min, 1=max)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_cluster_summary(cluster_info):
    """Print detailed cluster summary."""
    print("\n" + "="*80)
    print("CLUSTER SUMMARY")
    print("="*80)

    for cluster_id, info in sorted(cluster_info.items(), key=lambda x: int(x[0])):
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {info['size']} reviews ({info['size']/sum(c['size'] for c in cluster_info.values()):.1%})")
        print(f"  Top keywords: {', '.join(info['top_keywords'][:8])}")
        if 'avg_rating' in info:
            print(f"  Avg rating: {info['avg_rating']:.2f} ± {info.get('std_rating', 0):.2f}")
        if 'weakness_to_strength_ratio' in info:
            print(f"  Weakness/Strength ratio: {info['weakness_to_strength_ratio']:.2f}")
            print(f"    → Avg strengths length: {info['avg_strengths_len']:.0f} chars")
            print(f"    → Avg weaknesses length: {info['avg_weaknesses_len']:.0f} chars")
        print(f"  Avg review length: {info['avg_length']:.0f} chars")

    print("\n" + "="*80)


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Load data
    gt_path = os.path.join(RESULTS_DIR, "gt_reviews_clustered.json")
    cluster_info_path = os.path.join(RESULTS_DIR, "cluster_info.json")
    embeddings_path = os.path.join(RESULTS_DIR, "gt_embeddings.npy")

    if not os.path.exists(gt_path):
        print(f"Ground truth reviews not found: {gt_path}")
        print("Run cluster_reviews.py first.")
        return

    with open(gt_path, "r") as f:
        gt_reviews = json.load(f)
    print(f"Loaded {len(gt_reviews)} ground truth reviews")

    with open(cluster_info_path, "r") as f:
        cluster_info = json.load(f)
    n_clusters = len(cluster_info)

    embeddings = np.load(embeddings_path)
    labels = np.array([r["cluster"] for r in gt_reviews])

    print(f"\nGenerating visualizations for {n_clusters} clusters...")

    # Generate all plots
    plot_umap(embeddings, labels, cluster_info,
              os.path.join(METRICS_DIR, "review_embeddings_umap.png"))

    plot_tsne(embeddings, labels, cluster_info,
              os.path.join(METRICS_DIR, "review_embeddings_tsne.png"))

    plot_distribution(gt_reviews, cluster_info,
                     os.path.join(METRICS_DIR, "archetype_distribution.png"))

    plot_cluster_characteristics(cluster_info,
                                os.path.join(METRICS_DIR, "cluster_characteristics.png"))

    # Print summary
    print_cluster_summary(cluster_info)

    # Save metrics
    gt_counts = np.zeros(n_clusters)
    for r in gt_reviews:
        gt_counts[r["cluster"]] += 1
    gt_freq = gt_counts / gt_counts.sum()

    metrics = {
        "n_clusters": n_clusters,
        "n_reviews": len(gt_reviews),
        "distribution": {str(k): float(v) for k, v in enumerate(gt_freq)},
        "cluster_info": cluster_info,
    }

    save_metrics_json(metrics, os.path.join(METRICS_DIR, "q1_metrics.json"))
    print(f"\nAll plots saved to: {METRICS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
