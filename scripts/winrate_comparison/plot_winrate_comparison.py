#!/usr/bin/env python3
"""
Win-Rate Comparison Across Methods

Computes and visualizes win-rates for SFT and RL methods.
A method "wins" on a paper if it predicted correctly AND >50% of other methods were wrong.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import re

# Paths
TEST_DATA = "data/iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6_test/data.json"
METADATA_PATH = "data/massive_metadata.csv"
OUTPUT_DIR = Path("results/summarized_investigation/winrate")

# Methods configuration
METHODS = {
    # SFT Modality
    "Full Paper": {
        "results": "results/data_sweep_v2/iclr20_balanced_clean/finetuned.jsonl",
        "type": "sft",
    },
    "Full Paper w/ Figures": {
        "results": "results/data_sweep_clean_images/balanced_clean_images/finetuned.jsonl",
        "type": "sft",
    },
    "Full Paper as Images": {
        "results": "results/data_sweep_v2/iclr20_balanced_vision/finetuned.jsonl",
        "type": "sft",
    },
    # RL ArXiv
    "Qwen3-8B": {
        "results": "/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/results/arxiv_qwen3_8b_step40_ailab.jsonl",
        "type": "rl",
    },
    "Qwen3-4B": {
        "results": "/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/results/arxiv_qwen3_4b_step50_ailab.jsonl",
        "type": "rl",
    },
    "Qwen2.5-7B": {
        "results": "/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/results/arxiv_qwen2.5_7b_step45_ailab.jsonl",
        "type": "rl",
    },
    # RL NoArXiv
    "Qwen3-4B (NoArXiv)": {
        "results": "/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/results/noarxiv_qwen3_4b_step40_ailab.jsonl",
        "type": "rl",
    },
    "Qwen2.5-7B (NoArXiv)": {
        "results": "/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/results/noarxiv_qwen2.5_7b_step55_ailab.jsonl",
        "type": "rl",
    },
    "Qwen3-8B-RP (NoArXiv)": {
        "results": "/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/results/noarxiv_qwen3_8b_reviewprocess_step65_ailab.jsonl",
        "type": "rl",
    },
}

CATEGORIES = {
    "SFT Modality": ["Full Paper", "Full Paper w/ Figures", "Full Paper as Images"],
    "RL ArXiv": ["Qwen3-8B", "Qwen3-4B", "Qwen2.5-7B"],
    "RL NoArXiv": ["Qwen3-4B (NoArXiv)", "Qwen2.5-7B (NoArXiv)", "Qwen3-8B-RP (NoArXiv)"],
}

CATEGORY_COLORS = {
    "SFT Modality": "#1f77b4",     # Blue
    "RL ArXiv": "#2ca02c",         # Green
    "RL NoArXiv": "#d62728",       # Red
}

# Factors from massive_metadata.csv
BASE_FACTORS = [
    ('num_authors', 'Authors'),
    ('num_figures', 'Figures'),
    ('num_pages', 'Pages'),
    ('num_text_tokens', 'Text Tokens'),
    ('num_text_image_tokens', 'Text+Img Tokens'),
    ('num_vision_tokens', 'Vision Tokens'),
    ('number_of_cited_references', 'Citations'),
    ('number_of_bib_items', 'Bib Items'),
    ('num_equations', 'Equations'),
]

METADATA_FACTORS = [
    ('removed_before_intro_count', 'Pre-Intro Removed'),
    ('removed_after_refs_pages', 'Appendix Pages'),
    ('removed_reproducibility_count', 'Has Reproducibility'),
    ('removed_acknowledgments_count', 'Has Acknowledgments'),
    ('removed_aside_text_count', 'Aside Text Removed'),
]


def extract_prediction(text):
    """Extract Accept/Reject from prediction text."""
    if isinstance(text, list):
        # RL format: majority vote
        votes = []
        for t in text:
            if "Accept" in t and "Reject" not in t:
                votes.append("Accept")
            elif "Reject" in t:
                votes.append("Reject")
        if not votes:
            return None
        return "Accept" if votes.count("Accept") > votes.count("Reject") else "Reject"
    else:
        # SFT format: single string
        if "Accept" in text and "Reject" not in text:
            return "Accept"
        elif "Reject" in text:
            return "Reject"
        return None


def load_predictions(method_name, method_config):
    """Load predictions from a JSONL file."""
    results_path = method_config["results"]
    predictions = []

    with open(results_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            pred = extract_prediction(data.get("predict", ""))
            label = extract_prediction(data.get("label", ""))
            metadata = data.get("_metadata", {})

            predictions.append({
                "prediction": pred,
                "ground_truth": label,
                "is_correct": pred == label if pred and label else False,
                "submission_id": metadata.get("submission_id"),
                "year": metadata.get("year"),
                "pct_rating": metadata.get("pct_rating"),
                "pct_citation": metadata.get("citation_normalized_by_year"),
            })

    return predictions


def load_test_metadata():
    """Load metadata from test data."""
    with open(TEST_DATA, 'r') as f:
        data = json.load(f)

    metadata_list = []
    for item in data:
        meta = item.get("_metadata", {})
        metadata_list.append({
            "submission_id": meta.get("submission_id"),
            "year": meta.get("year"),
            "pct_rating": meta.get("pct_rating"),
            "pct_citation": meta.get("citation_normalized_by_year"),
        })

    return metadata_list


def load_massive_metadata():
    """Load additional metadata from massive_metadata.csv."""
    df = pd.read_csv(METADATA_PATH)

    # Parse metadata_of_changes JSON
    def parse_changes(row):
        try:
            if pd.isna(row.get('metadata_of_changes')):
                return {}
            return json.loads(row['metadata_of_changes'])
        except:
            return {}

    metadata_dict = {}
    for _, row in df.iterrows():
        sid = row['submission_id']
        changes = parse_changes(row)

        metadata_dict[sid] = {
            'num_authors': row.get('num_authors'),
            'num_figures': row.get('num_figures'),
            'num_pages': row.get('num_pages'),
            'num_text_tokens': row.get('num_text_tokens'),
            'num_text_image_tokens': row.get('num_text_image_tokens'),
            'num_vision_tokens': row.get('num_vision_tokens'),
            'number_of_cited_references': row.get('number_of_cited_references'),
            'number_of_bib_items': row.get('number_of_bib_items'),
            'num_equations': row.get('num_equations'),
            'removed_before_intro_count': changes.get('removed_before_intro_count'),
            'removed_after_refs_pages': changes.get('removed_after_refs_pages'),
            'removed_reproducibility_count': changes.get('removed_reproducibility_count'),
            'removed_acknowledgments_count': changes.get('removed_acknowledgments_count'),
            'removed_aside_text_count': changes.get('removed_aside_text_count'),
        }

    return metadata_dict


def compute_winrate(paper_results, methods):
    """
    Compute win-rate for each method.
    A method wins if it's correct AND >50% of others are wrong.
    """
    n_methods = len(methods)
    threshold = n_methods / 2  # >50% of others wrong means incorrect_count > n_methods/2

    wins = {m: 0 for m in methods}
    total_papers = len(paper_results)

    for paper in paper_results:
        correct_methods = [m for m in methods if paper.get(m, False)]
        incorrect_count = n_methods - len(correct_methods)

        # A method wins if correct AND >50% of others are wrong
        if incorrect_count > threshold:
            for m in correct_methods:
                wins[m] += 1

    # Convert to percentages
    winrates = {m: (wins[m] / total_papers * 100) if total_papers > 0 else 0 for m in methods}
    return winrates


def compute_factorized_winrate(paper_results, factor_values, methods, bins=None):
    """Compute win-rate grouped by factor values."""
    if bins is not None:
        # Use provided bins (for continuous factors like pct_rating)
        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        factor_bins = pd.cut(factor_values, bins=bins, labels=bin_labels, include_lowest=True)
    else:
        # Use quartiles for numeric factors
        try:
            factor_bins = pd.qcut(factor_values, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        except ValueError:
            # Not enough unique values for quartiles
            factor_bins = pd.Series(factor_values).astype(str)

    # Group papers by factor bins
    grouped_results = defaultdict(list)
    for i, (paper, bin_val) in enumerate(zip(paper_results, factor_bins)):
        if pd.notna(bin_val):
            grouped_results[bin_val].append(paper)

    # Compute win-rate for each group
    factorized_winrates = {}
    for bin_val, papers in grouped_results.items():
        if len(papers) > 0:
            factorized_winrates[bin_val] = compute_winrate(papers, methods)

    return factorized_winrates


def get_method_color(method_name):
    """Get color for a method based on its category."""
    for category, method_list in CATEGORIES.items():
        if method_name in method_list:
            return CATEGORY_COLORS[category]
    return "#7f7f7f"  # Default gray


def get_method_category(method_name):
    """Get category for a method."""
    for category, method_list in CATEGORIES.items():
        if method_name in method_list:
            return category
    return "Unknown"


def plot_overall_winrate(winrates, output_path):
    """Plot overall win-rate bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = list(winrates.keys())
    rates = [winrates[m] for m in methods]
    colors = [get_method_color(m) for m in methods]

    bars = ax.bar(range(len(methods)), rates, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Overall Win-Rate Comparison Across Methods', fontsize=14)

    # Add legend for categories
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in CATEGORY_COLORS.values()]
    ax.legend(handles, CATEGORY_COLORS.keys(), loc='upper right')

    ax.set_ylim(0, max(rates) * 1.2 if rates else 10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_factorized_winrate_year(factorized_winrates, methods, output_path):
    """Plot win-rate by year."""
    fig, ax = plt.subplots(figsize=(12, 6))

    years = sorted([y for y in factorized_winrates.keys()])
    x = np.arange(len(years))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        rates = [factorized_winrates[y].get(method, 0) for y in years]
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=method,
                     color=get_method_color(method), edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=10)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Win-Rate by Year', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_factorized_winrate_pct(factorized_pct_rating, factorized_pct_citation, methods, output_path):
    """Plot win-rate by pct_rating and pct_citation."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # pct_rating
    ax = axes[0]
    bins_order = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5',
                  '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    available_bins = [b for b in bins_order if b in factorized_pct_rating]

    for method in methods:
        rates = [factorized_pct_rating.get(b, {}).get(method, 0) for b in available_bins]
        ax.plot(range(len(available_bins)), rates, marker='o', label=method,
               color=get_method_color(method), linewidth=2, markersize=4)

    ax.set_xticks(range(len(available_bins)))
    ax.set_xticklabels(available_bins, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('pct_rating', fontsize=11)
    ax.set_ylabel('Win Rate (%)', fontsize=11)
    ax.set_title('Win-Rate by Rating Percentile', fontsize=12)
    ax.grid(alpha=0.3)

    # pct_citation
    ax = axes[1]
    available_bins = [b for b in bins_order if b in factorized_pct_citation]

    for method in methods:
        rates = [factorized_pct_citation.get(b, {}).get(method, 0) for b in available_bins]
        ax.plot(range(len(available_bins)), rates, marker='o', label=method,
               color=get_method_color(method), linewidth=2, markersize=4)

    ax.set_xticks(range(len(available_bins)))
    ax.set_xticklabels(available_bins, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('pct_citation', fontsize=11)
    ax.set_ylabel('Win Rate (%)', fontsize=11)
    ax.set_title('Win-Rate by Citation Percentile', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_factorized_winrate_metadata(all_factor_winrates, methods, output_path):
    """Plot win-rate by paper metadata factors in a grid."""
    n_factors = len(all_factor_winrates)
    ncols = 3
    nrows = (n_factors + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten() if n_factors > 1 else [axes]

    for idx, (factor_name, factor_data) in enumerate(all_factor_winrates.items()):
        ax = axes[idx]

        # Sort bins
        bins = list(factor_data.keys())
        if all(b.startswith('Q') for b in bins):
            bins = sorted(bins)
        else:
            try:
                bins = sorted(bins, key=lambda x: float(x.split('-')[0]) if '-' in x else float(x))
            except:
                bins = sorted(bins)

        x = np.arange(len(bins))
        width = 0.8 / len(methods)

        for i, method in enumerate(methods):
            rates = [factor_data.get(b, {}).get(method, 0) for b in bins]
            offset = (i - len(methods)/2 + 0.5) * width
            ax.bar(x + offset, rates, width, label=method if idx == 0 else "",
                  color=get_method_color(method), edgecolor='black', linewidth=0.2)

        ax.set_xticks(x)
        ax.set_xticklabels(bins, fontsize=8)
        ax.set_xlabel(factor_name, fontsize=10)
        ax.set_ylabel('Win Rate (%)', fontsize=9)
        ax.set_title(f'Win-Rate by {factor_name}', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for idx in range(n_factors, len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=get_method_color(m)) for m in methods]
    fig.legend(handles, methods, loc='upper center', bbox_to_anchor=(0.5, 1.02),
              ncol=min(4, len(methods)), fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test metadata...")
    test_metadata = load_test_metadata()

    print("Loading massive metadata...")
    massive_metadata = load_massive_metadata()

    print(f"\nLoading predictions for {len(METHODS)} methods...")
    all_predictions = {}
    for method_name, method_config in METHODS.items():
        print(f"  Loading {method_name}...")
        try:
            all_predictions[method_name] = load_predictions(method_name, method_config)
            print(f"    Loaded {len(all_predictions[method_name])} predictions")
        except FileNotFoundError as e:
            print(f"    ERROR: File not found - {e}")
            continue

    if not all_predictions:
        print("ERROR: No predictions loaded!")
        return

    methods = list(all_predictions.keys())
    n_papers = len(test_metadata)

    print(f"\nBuilding paper results for {n_papers} papers...")

    # Build paper-level results
    paper_results = []
    paper_metadata = []

    for i in range(n_papers):
        paper = {}
        for method in methods:
            if i < len(all_predictions[method]):
                paper[method] = all_predictions[method][i]["is_correct"]
            else:
                paper[method] = False
        paper_results.append(paper)

        # Get metadata for this paper
        meta = test_metadata[i]
        submission_id = meta.get("submission_id")

        # Merge with massive metadata
        extra_meta = massive_metadata.get(submission_id, {})
        combined_meta = {**meta, **extra_meta}
        paper_metadata.append(combined_meta)

    # Compute overall win-rate
    print("\nComputing overall win-rate...")
    overall_winrates = compute_winrate(paper_results, methods)
    print("Overall Win-Rates:")
    for m, r in overall_winrates.items():
        print(f"  {m}: {r:.2f}%")

    # Plot overall win-rate
    plot_overall_winrate(overall_winrates, OUTPUT_DIR / "overall_winrate.png")

    # Compute factorized win-rate by year
    print("\nComputing win-rate by year...")
    years = [m.get("year") for m in paper_metadata]
    year_groups = defaultdict(list)
    for paper, year in zip(paper_results, years):
        if year:
            year_groups[year].append(paper)

    factorized_year = {}
    for year, papers in sorted(year_groups.items()):
        factorized_year[year] = compute_winrate(papers, methods)

    plot_factorized_winrate_year(factorized_year, methods, OUTPUT_DIR / "factorized_winrate_year.png")

    # Compute factorized win-rate by pct_rating and pct_citation
    print("\nComputing win-rate by pct_rating and pct_citation...")
    pct_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    pct_ratings = pd.Series([m.get("pct_rating") for m in paper_metadata])
    pct_citations = pd.Series([m.get("pct_citation") for m in paper_metadata])

    factorized_pct_rating = compute_factorized_winrate(paper_results, pct_ratings, methods, bins=pct_bins)
    factorized_pct_citation = compute_factorized_winrate(paper_results, pct_citations, methods, bins=pct_bins)

    plot_factorized_winrate_pct(factorized_pct_rating, factorized_pct_citation, methods,
                                OUTPUT_DIR / "factorized_winrate_pct.png")

    # Compute factorized win-rate by metadata factors
    print("\nComputing win-rate by metadata factors...")
    all_factor_winrates = {}

    for factor_col, factor_name in BASE_FACTORS + METADATA_FACTORS:
        factor_values = pd.Series([m.get(factor_col) for m in paper_metadata])
        valid_count = factor_values.notna().sum()

        if valid_count < 100:
            print(f"  Skipping {factor_name}: only {valid_count} valid values")
            continue

        print(f"  Processing {factor_name}...")
        factorized = compute_factorized_winrate(paper_results, factor_values, methods)
        if factorized:
            all_factor_winrates[factor_name] = factorized

    if all_factor_winrates:
        plot_factorized_winrate_metadata(all_factor_winrates, methods, OUTPUT_DIR / "factorized_winrate_metadata.png")

    # Save raw data to CSV
    print("\nSaving raw data to CSV...")
    csv_data = []
    for i in range(n_papers):
        row = {
            "paper_idx": i,
            "submission_id": paper_metadata[i].get("submission_id"),
            "year": paper_metadata[i].get("year"),
            "pct_rating": paper_metadata[i].get("pct_rating"),
            "pct_citation": paper_metadata[i].get("pct_citation"),
        }
        for method in methods:
            row[f"{method}_correct"] = paper_results[i].get(method, False)

        # Determine winners for this paper
        correct_methods = [m for m in methods if paper_results[i].get(m, False)]
        incorrect_count = len(methods) - len(correct_methods)
        threshold = len(methods) / 2

        winners = []
        if incorrect_count > threshold:
            winners = correct_methods
        row["winners"] = "|".join(winners) if winners else ""

        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(OUTPUT_DIR / "winrate_data.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'winrate_data.csv'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
