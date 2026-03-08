#!/usr/bin/env python3
"""
H2 Analysis: Progressive content ablation for contamination detection.

Generates:
- accuracy_vs_content_level.png: Line plot of accuracy at each content level
- acceptance_rate_by_level.png: Acceptance rate at each level
- abstract_similarity_distribution.png: Distribution of cosine similarity between
  generated and true abstracts (title-only experiment)
- similarity_vs_citations.png: Similarity correlated with citation percentile
- similarity_vs_rating.png: Similarity correlated with avg reviewer score

Usage:
    python 2_8_26/h2_contamination/analyze.py
"""

import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.analysis_utils import (
    compute_acceptance_rate,
    compute_accuracy,
    compute_metrics_summary,
    load_results,
    save_metrics_json,
    setup_plot_style,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Ordered content levels (increasing information)
LEVELS = ["title_abstract", "title_intro", "title_conclusion", "full_paper"]
LEVEL_LABELS = ["Title +\nAbstract", "Title +\nIntro", "Title +\nConclusion", "Full\nPaper"]

# Test dataset with metadata
DEFAULT_TEST_DATASET = (
    "/n/fs/vision-mix/sk7524/LLaMA-Factory/data/"
    "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test/data.json"
)

# Title-abstract dataset (contains true abstracts in human message)
TITLE_ABSTRACT_DATASET_PATTERN = (
    "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test_title_abstract"
)


def load_all_level_results():
    """Load results for all content levels."""
    level_results = {}
    for level in LEVELS:
        path = os.path.join(RESULTS_DIR, level, "results_single.jsonl")
        if os.path.exists(path):
            level_results[level] = load_results(path)
            print(f"  Loaded {level}: {len(level_results[level])} results")
        else:
            print(f"  Missing: {level} (no results at {path})")
    return level_results


def extract_abstract_from_paper(human_text):
    """Extract abstract from the paper markdown in the human message."""
    # Look for "# Abstract\n..." pattern
    match = re.search(r'#\s*Abstract\s*\n(.*?)(?=\n#|\Z)', human_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: look for "Abstract\n..." without #
    match = re.search(r'Abstract\s*\n(.*?)(?=\n[A-Z#]|\Z)', human_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def load_title_only_data():
    """Load title-only predictions and corresponding true abstracts.

    Returns:
        List of dicts with: generated_abstract, true_abstract, metadata
    """
    title_only_pred_path = os.path.join(RESULTS_DIR, "title_only", "predictions.jsonl")
    if not os.path.exists(title_only_pred_path):
        return []

    # Load generated abstracts from predictions
    predictions = []
    with open(title_only_pred_path) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line.strip()))

    # Load true abstracts from title_abstract dataset (has abstract in human message)
    ta_data_path = os.path.join(DATA_DIR, TITLE_ABSTRACT_DATASET_PATTERN, "data.json")
    if not os.path.exists(ta_data_path):
        print(f"  Title-abstract dataset not found at {ta_data_path}")
        return []

    with open(ta_data_path) as f:
        ta_data = json.load(f)

    # Load metadata from test dataset
    if os.path.exists(DEFAULT_TEST_DATASET):
        with open(DEFAULT_TEST_DATASET) as f:
            test_data = json.load(f)
    else:
        test_data = ta_data  # fallback

    if len(predictions) != len(ta_data):
        print(f"  Warning: prediction count ({len(predictions)}) != dataset count ({len(ta_data)})")

    samples = []
    for i in range(min(len(predictions), len(ta_data), len(test_data))):
        pred = predictions[i]
        ta_sample = ta_data[i]
        test_sample = test_data[i] if i < len(test_data) else ta_sample
        meta = test_sample.get("_metadata", ta_sample.get("_metadata", {}))

        # Extract true abstract and title from title_abstract dataset's human message
        true_abstract = ""
        title = ""
        for conv in ta_sample["conversations"]:
            if conv["from"] == "human":
                true_abstract = extract_abstract_from_paper(conv["value"])
                # Extract title (first line after "# Title" or first non-empty line)
                title_match = re.search(r'#\s*Title\s*\n(.*?)(?=\n#|\n\n|\Z)', conv["value"], re.DOTALL)
                if title_match:
                    title = title_match.group(1).strip()
                else:
                    # Fallback: first non-empty line
                    for tl in conv["value"].split("\n"):
                        tl = tl.strip().lstrip("#").strip()
                        if tl:
                            title = tl
                            break
                break

        # Extract generated abstract from predictions
        generated = pred.get("predict", "")
        if isinstance(generated, list):
            generated = generated[0] if generated else ""
        # Strip "Abstract: " prefix if present
        generated = re.sub(r'^Abstract:\s*', '', generated, flags=re.IGNORECASE).strip()

        if not true_abstract or not generated:
            continue

        samples.append({
            "true_abstract": true_abstract,
            "generated_abstract": generated,
            "title": title,
            "submission_id": meta.get("submission_id", ""),
            "year": meta.get("year", 0),
            "citation": meta.get("citation", 0),
            "citation_normalized": meta.get("citation_normalized_by_year", 0),
            "pct_rating": meta.get("pct_rating", 0),
            "ratings": meta.get("ratings", []),
            "answer": meta.get("answer", ""),
        })

    return samples


def compute_similarities(samples, model_name="all-MiniLM-L6-v2"):
    """Compute cosine similarities between generated and true abstracts."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        return []

    model = SentenceTransformer(model_name)

    true_texts = [s["true_abstract"][:1500] for s in samples]
    gen_texts = [s["generated_abstract"][:1500] for s in samples]

    print(f"  Embedding {len(true_texts)} abstract pairs with {model_name}...")
    true_embs = model.encode(true_texts, batch_size=64, show_progress_bar=True)
    gen_embs = model.encode(gen_texts, batch_size=64, show_progress_bar=True)

    # Cosine similarity
    similarities = []
    for t, g in zip(true_embs, gen_embs):
        cos_sim = float(np.dot(t, g) / (np.linalg.norm(t) * np.linalg.norm(g) + 1e-8))
        similarities.append(cos_sim)

    return similarities


def _wrap_text(text, width=100):
    """Wrap text to given width, preserving words."""
    import textwrap
    return textwrap.fill(text.replace("\n", " ").strip(), width=width)


def plot_similarity_distribution(similarities, samples, save_path):
    """Plot distribution of abstract similarities with full example abstracts."""
    setup_plot_style()

    sims = np.array(similarities)

    # Pick examples from low, medium, high bins
    sorted_indices = np.argsort(sims)
    n_samples = len(sorted_indices)
    example_indices = {
        "Low": sorted_indices[n_samples // 10],
        "Medium": sorted_indices[n_samples // 2],
        "High": sorted_indices[-n_samples // 10],
    }

    # Build example text blocks to estimate height
    example_blocks = []
    for label, idx in example_indices.items():
        sim = sims[idx]
        s = samples[idx]
        title = s.get("title", "N/A")
        true_abs = _wrap_text(s["true_abstract"], width=110)
        gen_abs = _wrap_text(s["generated_abstract"], width=110)
        block = (
            f"{'='*120}\n"
            f"[{label} similarity = {sim:.3f}]  Title: {title}\n"
            f"{'='*120}\n"
            f"TRUE ABSTRACT:\n{true_abs}\n\n"
            f"GENERATED ABSTRACT:\n{gen_abs}\n"
        )
        example_blocks.append(block)

    full_text = "\n".join(example_blocks)
    n_text_lines = full_text.count("\n") + 1

    # Dynamic figure height: histogram (5 inches) + examples (scaled to text)
    text_height = max(8, n_text_lines * 0.16)
    fig_height = 5 + text_height
    fig, axes = plt.subplots(2, 1, figsize=(16, fig_height),
                             gridspec_kw={"height_ratios": [5, text_height]})

    # Top: Histogram
    ax = axes[0]
    n, bins, patches_hist = ax.hist(sims, bins=30, color="#3498db", edgecolor="black",
                                     linewidth=0.5, alpha=0.8)

    # Color code bins
    for i, (patch, left, right) in enumerate(zip(patches_hist, bins[:-1], bins[1:])):
        mid = (left + right) / 2
        if mid > 0.8:
            patch.set_facecolor("#e74c3c")  # High similarity (potential contamination)
        elif mid > 0.5:
            patch.set_facecolor("#f39c12")  # Moderate
        else:
            patch.set_facecolor("#3498db")  # Low

    ax.axvline(x=np.mean(sims), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(sims):.3f}")
    ax.axvline(x=np.median(sims), color="green", linestyle="--", linewidth=2,
               label=f"Median: {np.median(sims):.3f}")

    ax.set_xlabel("Cosine Similarity (Generated vs True Abstract)")
    ax.set_ylabel("Count")
    ax.set_title("H2: Semantic Similarity of Title-Only Generated Abstracts\n"
                 "(Higher = more similar to real abstract = potential contamination)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom: Full example abstracts at different similarity levels
    ax2 = axes[1]
    ax2.axis("off")

    ax2.text(0.01, 0.99, full_text, transform=ax2.transAxes,
             fontsize=7, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_similarity_vs_citations(similarities, samples, save_path):
    """Plot similarity vs citation percentile."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    sims = np.array(similarities)
    citations_norm = np.array([s["citation_normalized"] for s in samples])

    # Scatter plot
    colors = ["#e74c3c" if s["answer"] == "Accept" else "#3498db" for s in samples]
    ax.scatter(citations_norm, sims, c=colors, alpha=0.3, s=10, edgecolors="none")

    # Add trend line
    valid = ~np.isnan(citations_norm) & ~np.isnan(sims) & (citations_norm > 0)
    if valid.sum() > 10:
        z = np.polyfit(citations_norm[valid], sims[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(citations_norm[valid].min(), citations_norm[valid].max(), 100)
        ax.plot(x_line, p(x_line), "k--", linewidth=2, label=f"Trend (slope={z[0]:.3f})")

        # Correlation
        corr = np.corrcoef(citations_norm[valid], sims[valid])[0, 1]
        ax.set_title(f"H2: Abstract Similarity vs Citation Percentile (r={corr:.3f})")
    else:
        ax.set_title("H2: Abstract Similarity vs Citation Percentile")

    # Add legend for accept/reject
    ax.scatter([], [], c="#e74c3c", label="Accept", s=30)
    ax.scatter([], [], c="#3498db", label="Reject", s=30)

    ax.set_xlabel("Citation Percentile (normalized by year)")
    ax.set_ylabel("Cosine Similarity (Generated vs True Abstract)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_similarity_vs_rating(similarities, samples, save_path):
    """Plot similarity vs average reviewer score."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    sims = np.array(similarities)
    avg_ratings = []
    for s in samples:
        ratings = s.get("ratings", [])
        if ratings:
            avg_ratings.append(np.mean(ratings))
        else:
            avg_ratings.append(s.get("pct_rating", 0) * 10)  # fallback: scale pct_rating
    avg_ratings = np.array(avg_ratings)

    colors = ["#e74c3c" if s["answer"] == "Accept" else "#3498db" for s in samples]
    ax.scatter(avg_ratings, sims, c=colors, alpha=0.3, s=10, edgecolors="none")

    # Trend line
    valid = ~np.isnan(avg_ratings) & ~np.isnan(sims) & (avg_ratings > 0)
    if valid.sum() > 10:
        z = np.polyfit(avg_ratings[valid], sims[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(avg_ratings[valid].min(), avg_ratings[valid].max(), 100)
        ax.plot(x_line, p(x_line), "k--", linewidth=2, label=f"Trend (slope={z[0]:.3f})")

        corr = np.corrcoef(avg_ratings[valid], sims[valid])[0, 1]
        ax.set_title(f"H2: Abstract Similarity vs Avg Reviewer Score (r={corr:.3f})")
    else:
        ax.set_title("H2: Abstract Similarity vs Avg Reviewer Score")

    ax.scatter([], [], c="#e74c3c", label="Accept", s=30)
    ax.scatter([], [], c="#3498db", label="Reject", s=30)

    ax.set_xlabel("Average Reviewer Score")
    ax.set_ylabel("Cosine Similarity (Generated vs True Abstract)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_similarity_vs_title_length(similarities, samples, save_path):
    """Plot similarity vs title length (number of words)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    sims = np.array(similarities)
    title_lengths = np.array([len(s.get("title", "").split()) for s in samples])

    colors = ["#e74c3c" if s["answer"] == "Accept" else "#3498db" for s in samples]
    ax.scatter(title_lengths, sims, c=colors, alpha=0.3, s=10, edgecolors="none")

    # Trend line
    valid = (title_lengths > 0) & ~np.isnan(sims)
    if valid.sum() > 10:
        z = np.polyfit(title_lengths[valid], sims[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(title_lengths[valid].min(), title_lengths[valid].max(), 100)
        ax.plot(x_line, p(x_line), "k--", linewidth=2, label=f"Trend (slope={z[0]:.4f})")

        corr = np.corrcoef(title_lengths[valid], sims[valid])[0, 1]
        ax.set_title(f"H2: Abstract Similarity vs Title Length (r={corr:.3f})")
    else:
        ax.set_title("H2: Abstract Similarity vs Title Length")

    ax.scatter([], [], c="#e74c3c", label="Accept", s=30)
    ax.scatter([], [], c="#3498db", label="Reject", s=30)

    ax.set_xlabel("Title Length (number of words)")
    ax.set_ylabel("Cosine Similarity (Generated vs True Abstract)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_accuracy_vs_content(level_results, save_path):
    """Plot accuracy as a function of content level."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    accuracies = []

    for level, label in zip(LEVELS, LEVEL_LABELS):
        if level in level_results:
            labels.append(label)
            accuracies.append(compute_accuracy(level_results[level]))

    x = np.arange(len(labels))
    bars = ax.bar(x, accuracies, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(labels))),
                  edgecolor="black", linewidth=0.5, width=0.6)

    ax.plot(x, accuracies, "o-", color="red", linewidth=2, markersize=8, zorder=5)

    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("H2: Accuracy vs Content Level (Contamination Check)")
    ax.set_ylim(0, max(accuracies) * 1.15 if accuracies else 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_acceptance_rate_by_level(level_results, save_path):
    """Plot acceptance rate at each content level."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    accept_rates = []

    for level, label in zip(LEVELS, LEVEL_LABELS):
        if level in level_results:
            labels.append(label)
            accept_rates.append(compute_acceptance_rate(level_results[level]))

    x = np.arange(len(labels))
    colors = ["#e74c3c" if ar > 0.7 else "#f39c12" if ar > 0.5 else "#2ecc71" for ar in accept_rates]
    bars = ax.bar(x, accept_rates, color=colors, edgecolor="black", linewidth=0.5, width=0.6)

    for bar, val in zip(bars, accept_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("H2: Acceptance Rate by Content Level")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.3, color="blue", linestyle="--", alpha=0.5, label="ICLR baseline (~30%)")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Balanced (50%)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="H2: Contamination ablation analysis")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--skip_similarity", action="store_true",
                        help="Skip abstract similarity analysis (requires sentence-transformers)")
    args = parser.parse_args()

    os.makedirs(METRICS_DIR, exist_ok=True)

    print("Loading results for all content levels...")
    level_results = load_all_level_results()

    if not level_results:
        print("No results found. Run inference first.")
        return

    # Compute metrics for each level
    all_metrics = {}
    print(f"\n{'='*60}")
    print("Metrics by Content Level")
    print(f"{'='*60}")

    for level in LEVELS:
        if level in level_results:
            metrics = compute_metrics_summary(level_results[level])
            all_metrics[level] = metrics
            print(f"\n{level}:")
            print(f"  Accuracy:        {metrics['accuracy']:.1%}")
            print(f"  Acceptance Rate: {metrics['acceptance_rate']:.1%}")
            print(f"  Accept Recall:   {metrics['accept_recall']:.1%}")
            print(f"  Reject Recall:   {metrics['reject_recall']:.1%}")

    # Generate content level plots
    print(f"\n{'='*60}")
    print("Generating plots...")

    plot_accuracy_vs_content(level_results,
                             os.path.join(METRICS_DIR, "accuracy_vs_content_level.png"))
    plot_acceptance_rate_by_level(level_results,
                                  os.path.join(METRICS_DIR, "acceptance_rate_by_level.png"))

    # Abstract similarity analysis (title-only)
    if not args.skip_similarity:
        print(f"\n{'='*60}")
        print("Abstract Similarity Analysis (title-only)")
        print(f"{'='*60}")

        samples = load_title_only_data()
        if samples:
            print(f"  Loaded {len(samples)} title-only samples with true abstracts")

            similarities = compute_similarities(samples, args.embedding_model)
            if similarities:
                sims = np.array(similarities)
                print(f"\n  Similarity Statistics:")
                print(f"    Mean:   {np.mean(sims):.4f}")
                print(f"    Median: {np.median(sims):.4f}")
                print(f"    Std:    {np.std(sims):.4f}")
                print(f"    Min:    {np.min(sims):.4f}")
                print(f"    Max:    {np.max(sims):.4f}")
                print(f"    >0.8:   {np.sum(sims > 0.8)} ({np.mean(sims > 0.8):.1%})")
                print(f"    >0.9:   {np.sum(sims > 0.9)} ({np.mean(sims > 0.9):.1%})")

                all_metrics["title_only_similarity"] = {
                    "mean": float(np.mean(sims)),
                    "median": float(np.median(sims)),
                    "std": float(np.std(sims)),
                    "min": float(np.min(sims)),
                    "max": float(np.max(sims)),
                    "pct_above_0.8": float(np.mean(sims > 0.8)),
                    "pct_above_0.9": float(np.mean(sims > 0.9)),
                    "n_samples": len(sims),
                }

                # Plot similarity distribution with examples
                plot_similarity_distribution(similarities, samples,
                                              os.path.join(METRICS_DIR, "abstract_similarity_distribution.png"))

                # Plot correlation with citation percentile
                plot_similarity_vs_citations(similarities, samples,
                                              os.path.join(METRICS_DIR, "similarity_vs_citations.png"))

                # Plot correlation with avg reviewer score
                plot_similarity_vs_rating(similarities, samples,
                                           os.path.join(METRICS_DIR, "similarity_vs_rating.png"))

                # Plot correlation with title length
                plot_similarity_vs_title_length(similarities, samples,
                                                os.path.join(METRICS_DIR, "similarity_vs_title_length.png"))
        else:
            print("  No title-only predictions found or missing true abstracts.")
            print("  Ensure both title_only/predictions.jsonl and title_abstract dataset exist.")

    # Save metrics
    save_metrics_json(all_metrics, os.path.join(METRICS_DIR, "h2_metrics.json"))

    print("\nDone!")


if __name__ == "__main__":
    main()
