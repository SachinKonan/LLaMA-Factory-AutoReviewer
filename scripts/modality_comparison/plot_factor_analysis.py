#!/usr/bin/env python3
"""
Factor analysis: How do paper features correlate with model accuracy?

For each modality, creates a bar chart showing correlation between
paper features and model correctness, split by ground truth label.
"""

import json
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.ticker import PercentFormatter

# Configuration - same as plot_modality_investigation.py
VARIANTS = {
    "Full Paper": {
        "results": "results/data_sweep_v2/iclr20_balanced_clean/finetuned.jsonl",
        "test_data": "data/iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6_test/data.json",
    },
    "Full Paper w/ Figures": {
        "results": "results/data_sweep_clean_images/balanced_clean_images/finetuned.jsonl",
        "test_data": "data/iclr_2020_2025_85_5_10_split6_balanced_clean_images_binary_noreviews_v6_test/data.json",
    },
    "Full Paper as Images": {
        "results": "results/data_sweep_v2/iclr20_balanced_vision/finetuned.jsonl",
        "test_data": "data/iclr_2020_2025_85_5_10_split6_balanced_vision_binary_noreviews_v6_test/data.json",
    },
}

# Factors to analyze - split into two groups
BASE_FACTORS = [
    # Direct columns from massive_metadata.csv
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
    # From metadata_of_changes (JSON field)
    ('removed_before_intro_count', 'Pre-Intro Removed'),
    ('removed_after_refs_pages', 'Appendix Pages'),
    ('removed_reproducibility_count', 'Has Reproducibility'),
    ('removed_acknowledgments_count', 'Has Acknowledgments'),
    ('removed_aside_text_count', 'Aside Text Removed'),
]

RATING_FACTORS = [
    # From test.json _metadata
    ('pct_rating', 'Pct Rating'),
    ('pct_citation', 'Pct Citation'),
]

# Combined for loading
FACTORS = BASE_FACTORS + METADATA_FACTORS + RATING_FACTORS

OUTPUT_DIR = Path("results/summarized_investigation/modality")
METADATA_PATH = Path("data/massive_metadata.csv")


def extract_boxed_answer(text: str) -> str:
    """Parse \\boxed{} format to extract Accept/Reject."""
    if text is None:
        return None
    match = re.search(r'\\boxed\{(\w+)\}', text)
    if match:
        return match.group(1).lower()
    if 'Accept' in text:
        return 'accept'
    if 'Reject' in text:
        return 'reject'
    return None


def load_predictions(results_path: str, test_data_path: str) -> pd.DataFrame:
    """Load predictions and join with test data metadata."""
    base_dir = Path(__file__).parent.parent.parent

    # Load test data to get submission_id and ground truth
    with open(base_dir / test_data_path, 'r') as f:
        test_data = json.load(f)

    metadata_list = []
    for i, item in enumerate(test_data):
        meta = item.get('_metadata', {})
        metadata_list.append({
            'index': i,
            'submission_id': meta.get('submission_id'),
            'ground_truth': meta.get('answer', '').lower(),
            'pct_rating': meta.get('pct_rating'),
            'pct_citation': meta.get('citation_normalized_by_year'),  # This is pct_citation
        })
    metadata_df = pd.DataFrame(metadata_list)

    # Load predictions
    predictions = []
    with open(base_dir / results_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            pred = extract_boxed_answer(data.get('predict', ''))
            predictions.append({
                'index': i,
                'prediction': pred,
            })
    predictions_df = pd.DataFrame(predictions)

    # Merge
    df = pd.merge(metadata_df, predictions_df, on='index')
    df['correct'] = (df['prediction'] == df['ground_truth']).astype(int)

    return df


def load_massive_metadata() -> pd.DataFrame:
    """Load and parse massive_metadata.csv."""
    base_dir = Path(__file__).parent.parent.parent
    df = pd.read_csv(base_dir / METADATA_PATH)

    # Parse metadata_of_changes JSON column
    def parse_metadata_changes(row):
        if pd.isna(row['metadata_of_changes']):
            return row
        try:
            meta = json.loads(row['metadata_of_changes'])
            row['removed_before_intro_count'] = meta.get('removed_before_intro_count', 0)
            row['removed_after_refs_pages'] = meta.get('removed_after_refs_pages', 0)
            row['removed_reproducibility_count'] = meta.get('removed_reproducibility_count', 0)
            row['removed_acknowledgments_count'] = meta.get('removed_acknowledgments_count', 0)
            row['removed_aside_text_count'] = meta.get('removed_aside_text_count', 0)
        except:
            pass
        return row

    df = df.apply(parse_metadata_changes, axis=1)

    return df


def compute_correlations(pred_df: pd.DataFrame, meta_df: pd.DataFrame) -> dict:
    """Compute correlations between factors and model predictions/accuracy."""
    # Join predictions with metadata
    merged = pd.merge(pred_df, meta_df, on='submission_id', how='inner')

    # Add binary prediction column (Accept=1, Reject=0)
    merged['pred_binary'] = (merged['prediction'] == 'accept').astype(int)

    results = {}
    for factor_col, factor_label in FACTORS:
        if factor_col not in merged.columns:
            print(f"  Warning: {factor_col} not found")
            continue

        # Get valid data
        valid = merged[[factor_col, 'correct', 'ground_truth', 'pred_binary']].dropna()
        if len(valid) < 10:
            continue

        # Split by ground truth
        accepts = valid[valid['ground_truth'] == 'accept']
        rejects = valid[valid['ground_truth'] == 'reject']

        try:
            # 1. Correlation with correctness (original analysis)
            if len(accepts) > 5:
                r_accept_correct, p_accept_correct = stats.pointbiserialr(accepts['correct'], accepts[factor_col])
            else:
                r_accept_correct, p_accept_correct = np.nan, np.nan

            if len(rejects) > 5:
                r_reject_correct, p_reject_correct = stats.pointbiserialr(rejects['correct'], rejects[factor_col])
            else:
                r_reject_correct, p_reject_correct = np.nan, np.nan

            # 2. Correlation with binary prediction (factor vs Accept/Reject prediction)
            # Using Pearson correlation between factor and pred_binary
            r_pred, p_pred = stats.pearsonr(valid[factor_col], valid['pred_binary'])

            # 3. R² for accepts vs rejects (how predictive is factor of ground truth)
            # Correlation between factor and ground truth being accept
            valid['gt_binary'] = (valid['ground_truth'] == 'accept').astype(int)
            r_gt, p_gt = stats.pearsonr(valid[factor_col], valid['gt_binary'])

            results[factor_label] = {
                # Original: correlation with correctness
                'accept_corr': r_accept_correct,
                'accept_p': p_accept_correct,
                'reject_corr': r_reject_correct,
                'reject_p': p_reject_correct,
                # New: correlation with binary prediction
                'pred_r': r_pred,
                'pred_r2': r_pred ** 2,
                'pred_p': p_pred,
                # New: correlation with ground truth (for ratio)
                'gt_r': r_gt,
                'gt_r2': r_gt ** 2,
                'gt_p': p_gt,
                # Sample sizes
                'n_accept': len(accepts),
                'n_reject': len(rejects),
                'n_total': len(valid),
            }
        except Exception as e:
            print(f"  Error computing correlation for {factor_col}: {e}")

    return results


def plot_factor_analysis(all_correlations: dict, output_dir: Path):
    """Create 3x3 grid: 3 modalities x 3 factor groups."""

    variants = list(all_correlations.keys())
    n_variants = len(variants)

    # Factor groups with labels
    base_factor_labels = [label for _, label in BASE_FACTORS]
    meta_factor_labels = [label for _, label in METADATA_FACTORS]
    rating_factor_labels = [label for _, label in RATING_FACTORS]

    fig, axes = plt.subplots(n_variants, 3, figsize=(18, 4 * n_variants))

    for row_idx, variant in enumerate(variants):
        correlations = all_correlations[variant]

        for col_idx, (factor_labels, col_title) in enumerate([
            (base_factor_labels, 'Paper Features'),
            (meta_factor_labels, 'Structural Changes'),
            (rating_factor_labels, 'Rating & Citation'),
        ]):
            ax = axes[row_idx, col_idx]

            # Filter to factors in this group
            factors = [f for f in factor_labels if f in correlations]
            if not factors:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                continue

            x = np.arange(len(factors))
            width = 0.35

            accept_corrs = [correlations[f]['accept_corr'] for f in factors]
            reject_corrs = [correlations[f]['reject_corr'] for f in factors]

            # Create horizontal bars
            bars1 = ax.barh(x - width/2, accept_corrs, width, label='Accept', color='#2ecc71', edgecolor='black')
            bars2 = ax.barh(x + width/2, reject_corrs, width, label='Reject', color='#e74c3c', edgecolor='black')

            # Add significance markers
            for i, f in enumerate(factors):
                if correlations[f]['accept_p'] < 0.05:
                    xpos = accept_corrs[i] + 0.01 if accept_corrs[i] >= 0 else accept_corrs[i] - 0.03
                    ax.text(xpos, i - width/2, '*', fontsize=14, va='center', fontweight='bold')
                if correlations[f]['reject_p'] < 0.05:
                    xpos = reject_corrs[i] + 0.01 if reject_corrs[i] >= 0 else reject_corrs[i] - 0.03
                    ax.text(xpos, i + width/2, '*', fontsize=14, va='center', fontweight='bold')

            ax.set_yticks(x)
            ax.set_yticklabels(factors, fontsize=12)
            ax.set_xlabel('Correlation with Model Accuracy', fontsize=12)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_xlim(-0.25, 0.25)
            ax.tick_params(labelsize=12)
            ax.grid(axis='x', alpha=0.3)

            # Title for top row only
            if row_idx == 0:
                ax.set_title(col_title, fontsize=14, fontweight='bold')

            # Variant label on left column
            if col_idx == 0:
                ax.set_ylabel(variant, fontsize=13, fontweight='bold')

            # Legend only on first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='lower right', fontsize=11)

    plt.suptitle('Factor Correlation with Model Accuracy (* = p < 0.05)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    png_path = output_dir / 'factor_analysis.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_prediction_r2(all_correlations: dict, output_dir: Path):
    """Create 3x3 grid showing R² between factors and binary prediction."""

    variants = list(all_correlations.keys())
    n_variants = len(variants)

    # Factor groups with labels
    base_factor_labels = [label for _, label in BASE_FACTORS]
    meta_factor_labels = [label for _, label in METADATA_FACTORS]
    rating_factor_labels = [label for _, label in RATING_FACTORS]

    fig, axes = plt.subplots(n_variants, 3, figsize=(18, 4 * n_variants))

    for row_idx, variant in enumerate(variants):
        correlations = all_correlations[variant]

        for col_idx, (factor_labels, col_title) in enumerate([
            (base_factor_labels, 'Paper Features'),
            (meta_factor_labels, 'Structural Changes'),
            (rating_factor_labels, 'Rating & Citation'),
        ]):
            ax = axes[row_idx, col_idx]

            # Filter to factors in this group
            factors = [f for f in factor_labels if f in correlations]
            if not factors:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                continue

            x = np.arange(len(factors))

            # Get R² values (as percentages) and determine bar color based on correlation sign
            r2_values_pct = []
            colors = []
            for f in factors:
                r2 = correlations[f]['pred_r2'] * 100  # Convert to percentage
                r = correlations[f]['pred_r']
                r2_values_pct.append(r2)
                # Green if positive correlation (higher factor → more Accept predictions)
                # Red if negative correlation (higher factor → more Reject predictions)
                colors.append('#2ecc71' if r >= 0 else '#e74c3c')

            # Create horizontal bars
            bars = ax.barh(x, r2_values_pct, color=colors, edgecolor='black')

            # Add significance markers and ratio labels for Vision
            for i, f in enumerate(factors):
                label_parts = []
                if correlations[f]['pred_p'] < 0.05:
                    label_parts.append('*')

                # For Vision model, show ratio vs other models
                if variant == "Full Paper as Images":
                    vision_r2 = correlations[f]['pred_r2']
                    text_r2 = all_correlations.get("Full Paper", {}).get(f, {}).get('pred_r2', 0)
                    text_fig_r2 = all_correlations.get("Full Paper w/ Figures", {}).get(f, {}).get('pred_r2', 0)

                    ratio_strs = []
                    if text_r2 > 0.001:
                        ratio_text = vision_r2 / text_r2
                        ratio_strs.append(f'{ratio_text:.1f}x')
                    if text_fig_r2 > 0.001:
                        ratio_fig = vision_r2 / text_fig_r2
                        ratio_strs.append(f'{ratio_fig:.1f}x')

                    if ratio_strs:
                        label_parts.append(f'[{",".join(ratio_strs)}]')

                if label_parts:
                    ax.text(r2_values_pct[i] + 0.2, i, ' '.join(label_parts),
                           fontsize=9, va='center', fontweight='bold')

            ax.set_yticks(x)
            ax.set_yticklabels(factors, fontsize=12)
            ax.set_xlabel('R² % (Variance Explained)', fontsize=12)
            ax.set_xlim(0, 15)  # 0-15%
            ax.tick_params(labelsize=12)
            ax.grid(axis='x', alpha=0.3)

            # Format x-axis as percentage
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

            # Title for top row only
            if row_idx == 0:
                ax.set_title(col_title, fontsize=14, fontweight='bold')

            # Variant label on left column
            if col_idx == 0:
                ax.set_ylabel(variant, fontsize=13, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Higher → Accept'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Higher → Reject'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11, bbox_to_anchor=(0.98, 0.98))

    plt.suptitle('R² Between Factors and Model Prediction (* = p < 0.05)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    png_path = output_dir / 'factor_prediction_r2.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_prediction_ratio(all_correlations: dict, output_dir: Path):
    """Create 3x3 grid showing log ratio of prediction R² vs ground truth R²."""

    variants = list(all_correlations.keys())
    n_variants = len(variants)

    # Factor groups with labels
    base_factor_labels = [label for _, label in BASE_FACTORS]
    meta_factor_labels = [label for _, label in METADATA_FACTORS]
    rating_factor_labels = [label for _, label in RATING_FACTORS]

    fig, axes = plt.subplots(n_variants, 3, figsize=(18, 4 * n_variants))

    for row_idx, variant in enumerate(variants):
        correlations = all_correlations[variant]

        for col_idx, (factor_labels, col_title) in enumerate([
            (base_factor_labels, 'Paper Features'),
            (meta_factor_labels, 'Structural Changes'),
            (rating_factor_labels, 'Rating & Citation'),
        ]):
            ax = axes[row_idx, col_idx]

            # Filter to factors in this group
            factors = [f for f in factor_labels if f in correlations]
            if not factors:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                continue

            x = np.arange(len(factors))

            # Compute ratio: pred_r2 / gt_r2
            # If ratio > 1, model relies on this factor MORE than ground truth does
            # If ratio < 1, model relies on this factor LESS than ground truth does
            ratios = []
            for f in factors:
                pred_r2 = correlations[f]['pred_r2']
                gt_r2 = correlations[f]['gt_r2']
                # Use log ratio for better visualization, add small epsilon to avoid div by zero
                epsilon = 1e-6
                ratio = np.log2((pred_r2 + epsilon) / (gt_r2 + epsilon))
                ratios.append(ratio)

            # Color based on whether model over-relies or under-relies
            colors = ['#e74c3c' if r > 0 else '#3498db' for r in ratios]

            # Create horizontal bars
            bars = ax.barh(x, ratios, color=colors, edgecolor='black')

            ax.set_yticks(x)
            ax.set_yticklabels(factors, fontsize=12)
            ax.set_xlabel('log₂(Pred R² / GT R²)', fontsize=12)
            ax.axvline(x=0, color='black', linewidth=1)
            ax.set_xlim(-5, 5)
            ax.tick_params(labelsize=12)
            ax.grid(axis='x', alpha=0.3)

            # Title for top row only
            if row_idx == 0:
                ax.set_title(col_title, fontsize=14, fontweight='bold')

            # Variant label on left column
            if col_idx == 0:
                ax.set_ylabel(variant, fontsize=13, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='black', label='Model over-relies'),
        Patch(facecolor='#3498db', edgecolor='black', label='Model under-relies'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11, bbox_to_anchor=(0.98, 0.98))

    plt.suptitle('Model Reliance on Factors vs Ground Truth\n(log₂ ratio of R² values)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    png_path = output_dir / 'factor_ratio.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def plot_factor_by_year(meta_df: pd.DataFrame, output_dir: Path):
    """Create 6xN grid showing factor distributions by year, split by accept/reject."""

    years = [2020, 2021, 2022, 2023, 2024, 2025]
    n_years = len(years)

    # Specific factors to show
    factors_to_plot = [
        ('num_text_tokens', 'Text Tokens', False),
        ('num_text_image_tokens', 'Text+Img Tokens', False),
        ('num_vision_tokens', 'Vision Tokens', False),
        ('number_of_bib_items', 'Bib Items', False),
        ('removed_acknowledgments_count', 'Has Acknowledgments', True),  # Binary
        ('removed_reproducibility_count', 'Has Reproducibility', True),  # Binary
        ('removed_aside_text_count', 'Aside Text Removed', True),  # Binary
    ]
    n_factors = len(factors_to_plot)

    # Parse technical_indicators to get accept/reject
    def get_decision(row):
        try:
            if pd.notna(row.get('technical_indicators')):
                ti = json.loads(row['technical_indicators'])
                return ti.get('binary_decision', None)
        except:
            pass
        return None

    meta_df = meta_df.copy()
    meta_df['decision'] = meta_df.apply(get_decision, axis=1)

    fig, axes = plt.subplots(n_years, n_factors, figsize=(3.5 * n_factors, 2.5 * n_years))

    for row_idx, year in enumerate(years):
        year_df = meta_df[meta_df['year'] == year]
        accepts = year_df[year_df['decision'] == 'accept']
        rejects = year_df[year_df['decision'] == 'reject']

        for col_idx, (factor_col, factor_label, is_binary) in enumerate(factors_to_plot):
            ax = axes[row_idx, col_idx]

            if factor_col not in year_df.columns:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                continue

            accept_vals = accepts[factor_col].dropna()
            reject_vals = rejects[factor_col].dropna()

            if is_binary:
                # Bar chart for binary factors (convert count > 0 to "has")
                x = [0, 1]
                accept_counts = [sum(accept_vals == 0), sum(accept_vals > 0)]
                reject_counts = [sum(reject_vals == 0), sum(reject_vals > 0)]

                # Normalize to percentages
                accept_pct = [c / len(accept_vals) * 100 if len(accept_vals) > 0 else 0 for c in accept_counts]
                reject_pct = [c / len(reject_vals) * 100 if len(reject_vals) > 0 else 0 for c in reject_counts]

                width = 0.35
                ax.bar([i - width/2 for i in x], accept_pct, width, color='#2ecc71', alpha=0.7, label='Accept')
                ax.bar([i + width/2 for i in x], reject_pct, width, color='#e74c3c', alpha=0.7, label='Reject')
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['No', 'Yes'], fontsize=10)
                ax.set_ylabel('% of papers', fontsize=10)
            else:
                # Histogram for continuous factors
                bins = 30
                # Get common range
                all_vals = pd.concat([accept_vals, reject_vals])
                if len(all_vals) > 0:
                    vmin, vmax = all_vals.quantile(0.01), all_vals.quantile(0.99)
                    bins = np.linspace(vmin, vmax, 31)

                ax.hist(accept_vals, bins=bins, color='#2ecc71', alpha=0.6, label='Accept', density=True)
                ax.hist(reject_vals, bins=bins, color='#e74c3c', alpha=0.6, label='Reject', density=True)
                ax.set_ylabel('Density', fontsize=10)

            ax.tick_params(labelsize=10)
            ax.grid(axis='y', alpha=0.3)

            # Title for top row only
            if row_idx == 0:
                ax.set_title(factor_label, fontsize=12, fontweight='bold')

            # Year label on left column
            if col_idx == 0:
                ax.set_ylabel(f'{year}\n(n={len(year_df):,})', fontsize=11, fontweight='bold')

            # Legend only on first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('Factor Distributions by Year (Accept vs Reject)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    png_path = output_dir / 'factor_by_year.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")


def main():
    """Run factor analysis for all modalities."""
    base_dir = Path(__file__).parent.parent.parent
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading massive_metadata.csv...")
    meta_df = load_massive_metadata()
    print(f"  Loaded {len(meta_df)} papers")

    all_correlations = {}

    for variant, config in VARIANTS.items():
        print(f"\nProcessing: {variant}")

        # Load predictions
        pred_df = load_predictions(config['results'], config['test_data'])
        print(f"  Predictions: {len(pred_df)}")

        # Compute correlations
        correlations = compute_correlations(pred_df, meta_df)
        all_correlations[variant] = correlations

        # Print summary
        print(f"  Factors analyzed: {len(correlations)}")
        for factor, corr in correlations.items():
            sig_accept = '*' if corr['accept_p'] < 0.05 else ''
            sig_reject = '*' if corr['reject_p'] < 0.05 else ''
            print(f"    {factor}: Accept r={corr['accept_corr']:.3f}{sig_accept}, Reject r={corr['reject_corr']:.3f}{sig_reject}")

    # Generate plots
    print("\n" + "="*50)
    print("Generating factor analysis plots...")
    plot_factor_analysis(all_correlations, OUTPUT_DIR)
    plot_prediction_r2(all_correlations, OUTPUT_DIR)
    plot_prediction_ratio(all_correlations, OUTPUT_DIR)
    plot_factor_by_year(meta_df, OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
