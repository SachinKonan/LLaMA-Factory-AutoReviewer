#!/usr/bin/env python3
"""
Correlative analysis: Analyze correlation between model predictions and structural features.

This script reveals if models are overfitting to document characteristics (tokens, pages, images)
rather than making decisions based on content quality.

Usage:
    # Analyze Gemini clean results
    python scripts/analyze_correlative.py \
        --results_dir results/gemini/clean \
        --dataset_type clean \
        --output results/correlative/gemini_clean

    # Analyze hyperparam vision sweep variants
    python scripts/analyze_correlative.py \
        --results_dir results/hyperparam_vision_data_sweep \
        --dataset_type vision \
        --variants "ci_bd,vis_bd"
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Matplotlib styling
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "helvetica",
})

# Sizes
LABELSIZE = 14
TITLESIZE = 16
LEGENDSIZE = 12
TICKSIZE = 12

# Colors
ACCEPT_COLOR = "#4CAF50"  # Green
REJECT_COLOR = "#F44336"  # Red

# Dataset mapping
DATASETS = {
    "clean": "iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3",
    "clean_images": "iclr_2020_2025_80_20_split5_balanced_deepreview_clean+images_binary_no_reviews_titleabs_corrected_v3",
    "vision": "iclr_2020_2025_80_20_split5_balanced_deepreview_vision_binary_no_reviews_titleabs_corrected_v3",
}

# Features to plot per dataset type
FEATURES_TO_PLOT = {
    "clean": ["tokens"],
    "clean_images": ["tokens", "images"],
    "vision": ["pages"],
}


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def extract_gemini_answer(text: str) -> str | None:
    """Extract answer from Gemini output (simple string matching)."""
    if not text:
        return None
    text_lower = text.lower()
    is_reject = "reject" in text_lower
    is_accept = "accept" in text_lower
    if is_accept and not is_reject:
        return "Accept"
    if is_reject and not is_accept:
        return "Reject"
    return None


def load_predictions(results_path: Path) -> list[dict]:
    """Load predictions from JSONL file."""
    # Try different possible filenames
    for filename in ["full.jsonl", "finetuned.jsonl"]:
        path = results_path / filename
        if path.exists():
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]

    # Try direct file
    if results_path.suffix == ".jsonl" and results_path.exists():
        with open(results_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    raise FileNotFoundError(f"No prediction file found in {results_path}")


def load_dataset(data_dir: Path, dataset_name: str, split: str) -> list[dict]:
    """Load dataset and return list of entries."""
    path = data_dir / f"{dataset_name}_{split}" / "data.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path) as f:
        return json.load(f)


def extract_features(entries: list[dict], dataset_type: str) -> pd.DataFrame:
    """Extract structural features from dataset entries."""
    features = []

    for entry in entries:
        metadata = entry.get("_metadata", {})
        answer = metadata.get("answer", "").lower()

        # Get human message content
        conversations = entry.get("conversations", [])
        human_content = ""
        for msg in conversations:
            if msg.get("from") == "human":
                human_content = msg.get("value", "")
                break

        # Calculate features
        tokens = len(human_content) // 4  # Approximate token count
        images = human_content.count("<image>")  # Image/page count

        features.append({
            "label": 1 if answer == "accept" else 0,
            "tokens": tokens,
            "images": images,
            "pages": images,  # For vision, images = pages
        })

    return pd.DataFrame(features)


def parse_predictions(predictions: list[dict], is_gemini: bool = False) -> pd.DataFrame:
    """Parse predictions and labels from prediction results."""
    results = []

    for pred in predictions:
        predict_text = pred.get("predict", "")
        label_text = pred.get("label", "")

        if is_gemini:
            pred_answer = extract_gemini_answer(predict_text)
            label_answer = extract_gemini_answer(label_text)
            if label_answer is None:
                label_answer = extract_boxed_answer(label_text)
        else:
            pred_answer = extract_boxed_answer(predict_text)
            label_answer = extract_boxed_answer(label_text)

        results.append({
            "pred": 1 if pred_answer and pred_answer.lower() == "accept" else 0,
            "label": 1 if label_answer and label_answer.lower() == "accept" else 0,
        })

    return pd.DataFrame(results)


def plot_correlation_matrix(df: pd.DataFrame, features: list[str], output_path: Path, title: str = ""):
    """Plot correlation matrix heatmap."""
    # Select columns for correlation
    cols = ["pred", "label"] + features
    corr_df = df[cols].corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        annot_kws={"size": TICKSIZE},
    )

    ax.set_title(f"Correlation Matrix{': ' + title if title else ''}", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved correlation matrix: {output_path}")

    return corr_df


def plot_logistic_regression(
    df: pd.DataFrame,
    features: list[str],
    output_path: Path,
    title: str = "",
):
    """Plot density distributions comparing GT vs Predictions for each feature.

    Creates a 1x2 subplot for each feature: Accepts on left, Rejects on right.
    """
    from scipy.stats import gaussian_kde

    # Colors
    GT_COLOR = "#3498db"  # Blue for ground truth
    PRED_COLOR = "#e67e22"  # Orange for predictions

    for feature in features:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        X = df[feature].values
        y_true = df["label"].values
        y_pred = df["pred"].values

        # Fit logistic regression on feature -> ground truth
        lr = LogisticRegression(random_state=42)
        lr.fit(X.reshape(-1, 1), y_true)
        lr_accuracy = lr.score(X.reshape(-1, 1), y_true)

        # Get feature values for each group
        gt_accept_x = X[y_true == 1]
        gt_reject_x = X[y_true == 0]
        pred_accept_x = X[y_pred == 1]
        pred_reject_x = X[y_pred == 0]

        # Create x range for density curves (same for both subplots)
        x_min, x_max = X.min(), X.max()
        x_range = np.linspace(x_min, x_max, 300)

        feature_label = feature.replace("_", " ").title()

        # Helper function to plot density
        def plot_density(ax, data, color, linestyle, label, fill_alpha=0.15):
            if len(data) > 1:
                kde = gaussian_kde(data, bw_method='scott')
                density = kde(x_range)
                ax.plot(x_range, density, color=color, linestyle=linestyle,
                       linewidth=2.5, label=label)
                ax.fill_between(x_range, density, alpha=fill_alpha, color=color)

        # Left subplot: Accepts
        ax_accept = axes[0]
        plot_density(ax_accept, gt_accept_x, GT_COLOR, '-', f'GT Accept (n={len(gt_accept_x)})')
        plot_density(ax_accept, pred_accept_x, PRED_COLOR, '--', f'Pred Accept (n={len(pred_accept_x)})')

        # Vertical lines for means
        gt_acc_mean = gt_accept_x.mean()
        pred_acc_mean = pred_accept_x.mean()
        ax_accept.axvline(gt_acc_mean, color=GT_COLOR, linestyle='-', alpha=0.7, linewidth=1)
        ax_accept.axvline(pred_acc_mean, color=PRED_COLOR, linestyle='--', alpha=0.7, linewidth=1)

        # Secondary x-axis for Accept subplot
        ax_accept2 = ax_accept.twiny()
        ax_accept2.set_xlim(ax_accept.get_xlim())
        ax_accept2.set_xticks([gt_acc_mean, pred_acc_mean])
        ax_accept2.set_xticklabels([f'{gt_acc_mean:.1f}', f'{pred_acc_mean:.1f}'],
                                    fontsize=TICKSIZE - 2, rotation=45)
        ax_accept2.tick_params(axis='x', colors='gray', length=8, width=1)

        ax_accept.set_xlabel(feature_label, fontsize=LABELSIZE)
        ax_accept.set_ylabel("Density", fontsize=LABELSIZE)
        ax_accept.set_title(f"Accepts - {feature_label}", fontsize=TITLESIZE)
        ax_accept.tick_params(axis='both', labelsize=TICKSIZE)
        ax_accept.legend(fontsize=LEGENDSIZE, loc='upper right')
        ax_accept.grid(True, alpha=0.3)

        # Right subplot: Rejects
        ax_reject = axes[1]
        plot_density(ax_reject, gt_reject_x, GT_COLOR, '-', f'GT Reject (n={len(gt_reject_x)})')
        plot_density(ax_reject, pred_reject_x, PRED_COLOR, '--', f'Pred Reject (n={len(pred_reject_x)})')

        # Vertical lines for means
        gt_rej_mean = gt_reject_x.mean()
        pred_rej_mean = pred_reject_x.mean()
        ax_reject.axvline(gt_rej_mean, color=GT_COLOR, linestyle='-', alpha=0.7, linewidth=1)
        ax_reject.axvline(pred_rej_mean, color=PRED_COLOR, linestyle='--', alpha=0.7, linewidth=1)

        # Secondary x-axis for Reject subplot
        ax_reject2 = ax_reject.twiny()
        ax_reject2.set_xlim(ax_reject.get_xlim())
        ax_reject2.set_xticks([gt_rej_mean, pred_rej_mean])
        ax_reject2.set_xticklabels([f'{gt_rej_mean:.1f}', f'{pred_rej_mean:.1f}'],
                                    fontsize=TICKSIZE - 2, rotation=45)
        ax_reject2.tick_params(axis='x', colors='gray', length=8, width=1)

        ax_reject.set_xlabel(feature_label, fontsize=LABELSIZE)
        ax_reject.set_ylabel("Density", fontsize=LABELSIZE)
        ax_reject.set_title(f"Rejects - {feature_label}", fontsize=TITLESIZE)
        ax_reject.tick_params(axis='both', labelsize=TICKSIZE)
        ax_reject.legend(fontsize=LEGENDSIZE, loc='upper right')
        ax_reject.grid(True, alpha=0.3)

        # Overall title
        plot_title = f"{feature_label} Distribution (LogReg acc={lr_accuracy:.2f})"
        if title:
            plot_title = f"{title} - {plot_title}"
        fig.suptitle(plot_title, fontsize=TITLESIZE + 2, y=1.02)

        plt.tight_layout()

        # Save with feature name in filename
        feature_output = output_path.parent / f"{output_path.stem}_{feature}{output_path.suffix}"
        plt.savefig(feature_output, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved density distribution plot: {feature_output}")


# =============================================================================
# Cross-Run Comparison Functions
# =============================================================================

def load_with_submission_ids(
    results_path: Path,
    dataset_name: str,
    split: str,
    is_gemini: bool = False,
) -> pd.DataFrame:
    """Load predictions and match with submission IDs from original dataset."""
    # Load predictions
    predictions = load_predictions(results_path)
    pred_df = parse_predictions(predictions, is_gemini=is_gemini)

    # Load original dataset
    data_dir = Path("data")
    dataset = load_dataset(data_dir, dataset_name, split)

    # Extract submission IDs
    submission_ids = []
    for entry in dataset:
        metadata = entry.get("_metadata", {})
        submission_ids.append(metadata.get("submission_id", ""))

    # Match sizes
    min_size = min(len(pred_df), len(submission_ids))
    pred_df = pred_df.iloc[:min_size].copy()
    pred_df["submission_id"] = submission_ids[:min_size]

    return pred_df


def find_intersection(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find intersection of two dataframes by submission_id."""
    # Find common submission IDs
    common_ids = set(df1["submission_id"]) & set(df2["submission_id"])
    print(f"Found {len(common_ids)} common submissions")

    # Filter to common submissions
    df1_filtered = df1[df1["submission_id"].isin(common_ids)].copy()
    df2_filtered = df2[df2["submission_id"].isin(common_ids)].copy()

    # Sort by submission_id to align
    df1_filtered = df1_filtered.sort_values("submission_id").reset_index(drop=True)
    df2_filtered = df2_filtered.sort_values("submission_id").reset_index(drop=True)

    return df1_filtered, df2_filtered


def plot_cross_correlation_matrix(
    df: pd.DataFrame,
    run1_name: str,
    run2_name: str,
    output_path: Path,
):
    """Plot correlation matrix between two runs."""
    # Rename columns for clarity
    corr_df = df[["run1_pred", "run2_pred", "label"]].copy()
    corr_df.columns = [run1_name, run2_name, "Ground Truth"]

    corr_matrix = corr_df.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        annot_kws={"size": TITLESIZE},
    )

    ax.set_title(f"Cross-Run Correlation: {run1_name} vs {run2_name}", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE + 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved cross-correlation matrix: {output_path}")
    return corr_matrix


def plot_agreement_heatmap(
    df: pd.DataFrame,
    run1_name: str,
    run2_name: str,
    output_path: Path,
):
    """Plot 2x2 agreement confusion matrix between runs."""
    # Create confusion matrix
    # Rows: Run1 prediction, Cols: Run2 prediction
    confusion = np.zeros((2, 2), dtype=int)
    gt_breakdown = [[{"accept": 0, "reject": 0} for _ in range(2)] for _ in range(2)]

    for _, row in df.iterrows():
        r1, r2, gt = int(row["run1_pred"]), int(row["run2_pred"]), int(row["label"])
        confusion[1 - r1, r2] += 1  # Flip rows so Accept is on top
        if gt == 1:
            gt_breakdown[1 - r1][r2]["accept"] += 1
        else:
            gt_breakdown[1 - r1][r2]["reject"] += 1

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    sns.heatmap(
        confusion,
        annot=False,
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    # Add annotations with counts and GT breakdown
    for i in range(2):
        for j in range(2):
            count = confusion[i, j]
            pct = 100 * count / len(df)
            gt_acc = gt_breakdown[i][j]["accept"]
            gt_rej = gt_breakdown[i][j]["reject"]
            text = f"{count}\n({pct:.1f}\\%)\n\nGT: {gt_acc}A / {gt_rej}R"
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                   fontsize=LABELSIZE, fontweight='bold')

    # Labels
    ax.set_xticklabels(["Reject", "Accept"], fontsize=LABELSIZE)
    ax.set_yticklabels(["Accept", "Reject"], fontsize=LABELSIZE)
    ax.set_xlabel(f"{run2_name} Prediction", fontsize=TITLESIZE)
    ax.set_ylabel(f"{run1_name} Prediction", fontsize=TITLESIZE)
    ax.set_title(f"Prediction Agreement: {run1_name} vs {run2_name}", fontsize=TITLESIZE + 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved agreement heatmap: {output_path}")

    # Print summary
    # With indexing confusion[1-r1, r2]: agree is [0,1] (both Accept) + [1,0] (both Reject)
    agree = confusion[0, 1] + confusion[1, 0]
    disagree = confusion[0, 0] + confusion[1, 1]
    print(f"Agreement: {agree} ({100*agree/len(df):.1f}%), Disagreement: {disagree} ({100*disagree/len(df):.1f}%)")


def plot_3d_hyperplane(
    df: pd.DataFrame,
    run1_name: str,
    run2_name: str,
    output_path: Path,
):
    """Plot 3D scatter with hyperplane showing relationship between runs and ground truth."""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Get data
    x = df["run1_pred"].values
    y = df["run2_pred"].values
    z = df["label"].values

    # Add jitter for visualization
    jitter = 0.08
    x_jit = x + np.random.uniform(-jitter, jitter, len(x))
    y_jit = y + np.random.uniform(-jitter, jitter, len(y))
    z_jit = z + np.random.uniform(-jitter, jitter, len(z))

    # Colors based on ground truth
    colors = [ACCEPT_COLOR if gt == 1 else REJECT_COLOR for gt in z]

    # Plot scatter points
    ax.scatter(x_jit, y_jit, z_jit, c=colors, alpha=0.5, s=30)

    # Fit hyperplane: z = a*x + b*y + c
    X_features = np.column_stack([x, y])
    reg = LinearRegression()
    reg.fit(X_features, z)
    r2_score = reg.score(X_features, z)

    # Create mesh for hyperplane
    xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 20), np.linspace(-0.2, 1.2, 20))
    zz = reg.coef_[0] * xx + reg.coef_[1] * yy + reg.intercept_

    # Clip z to reasonable range
    zz = np.clip(zz, -0.3, 1.3)

    # Plot hyperplane
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='blue', edgecolor='none')

    # Labels
    ax.set_xlabel(f"{run1_name}\n(0=Reject, 1=Accept)", fontsize=LABELSIZE, labelpad=10)
    ax.set_ylabel(f"{run2_name}\n(0=Reject, 1=Accept)", fontsize=LABELSIZE, labelpad=10)
    ax.set_zlabel("Ground Truth\n(0=Reject, 1=Accept)", fontsize=LABELSIZE, labelpad=10)

    # Title with hyperplane equation
    eq = f"z = {reg.coef_[0]:.3f}x + {reg.coef_[1]:.3f}y + {reg.intercept_:.3f}"
    ax.set_title(f"3D Hyperplane: {run1_name} vs {run2_name}\n{eq} (R²={r2_score:.3f})",
                fontsize=TITLESIZE)

    # Set axis limits
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_zlim(-0.2, 1.2)

    # Set viewing angle: rotate right and down to see hyperplane better
    ax.view_init(elev=25, azim=45)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ACCEPT_COLOR,
               markersize=10, label='GT Accept'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=REJECT_COLOR,
               markersize=10, label='GT Reject'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=LEGENDSIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved 3D hyperplane plot: {output_path}")
    print(f"Hyperplane: z = {reg.coef_[0]:.3f}*{run1_name} + {reg.coef_[1]:.3f}*{run2_name} + {reg.intercept_:.3f}")
    print(f"R² score: {r2_score:.3f}")

    # Also create interactive plotly version
    try:
        import plotly.graph_objects as go

        # Create scatter trace
        scatter = go.Scatter3d(
            x=x_jit, y=y_jit, z=z_jit,
            mode='markers',
            marker=dict(
                size=4,
                color=z,
                colorscale=[[0, REJECT_COLOR], [1, ACCEPT_COLOR]],
                opacity=0.6,
            ),
            name='Predictions',
            hovertemplate=(
                f'{run1_name}: %{{x:.2f}}<br>'
                f'{run2_name}: %{{y:.2f}}<br>'
                'Ground Truth: %{z:.2f}<extra></extra>'
            ),
        )

        # Create hyperplane surface
        surface = go.Surface(
            x=xx, y=yy, z=zz,
            opacity=0.4,
            colorscale=[[0, 'blue'], [1, 'blue']],
            showscale=False,
            name='Hyperplane',
        )

        # Create figure
        fig = go.Figure(data=[scatter, surface])

        eq = f"z = {reg.coef_[0]:.3f}x + {reg.coef_[1]:.3f}y + {reg.intercept_:.3f}"
        fig.update_layout(
            title=f"3D Hyperplane: {run1_name} vs {run2_name}<br>{eq} (R²={r2_score:.3f})",
            scene=dict(
                xaxis_title=f"{run1_name} (0=Reject, 1=Accept)",
                yaxis_title=f"{run2_name} (0=Reject, 1=Accept)",
                zaxis_title="Ground Truth (0=Reject, 1=Accept)",
                xaxis=dict(range=[-0.2, 1.2]),
                yaxis=dict(range=[-0.2, 1.2]),
                zaxis=dict(range=[-0.2, 1.2]),
            ),
            width=1000,
            height=800,
        )

        # Save interactive HTML
        html_path = output_path.with_suffix('.html')
        fig.write_html(str(html_path))
        print(f"Saved interactive 3D plot: {html_path}")

    except ImportError:
        print("Note: Install plotly for interactive 3D plots (pip install plotly)")


def compare_runs(
    run1_results: Path,
    run1_dataset: str,
    run1_name: str,
    run2_results: Path,
    run2_dataset: str,
    run2_name: str,
    output_dir: Path,
    split: str = "test",
    is_gemini: bool = False,
):
    """Compare predictions between two runs."""
    print(f"\n{'='*60}")
    print(f"Cross-Run Comparison")
    print(f"Run 1: {run1_name} ({run1_results})")
    print(f"Run 2: {run2_name} ({run2_results})")
    print(f"{'='*60}")

    # Load both runs with submission IDs
    print(f"\nLoading {run1_name}...")
    df1 = load_with_submission_ids(run1_results, run1_dataset, split, is_gemini)
    print(f"  Loaded {len(df1)} predictions")

    print(f"\nLoading {run2_name}...")
    df2 = load_with_submission_ids(run2_results, run2_dataset, split, is_gemini)
    print(f"  Loaded {len(df2)} predictions")

    # Find intersection
    print(f"\nFinding intersection...")
    df1_common, df2_common = find_intersection(df1, df2)

    # Combine into single dataframe
    df = pd.DataFrame({
        "submission_id": df1_common["submission_id"],
        "run1_pred": df1_common["pred"],
        "run2_pred": df2_common["pred"],
        "label": df1_common["label"],  # Should be same as df2_common["label"]
    })

    # Verify labels match
    label_match = (df1_common["label"] == df2_common["label"]).all()
    if not label_match:
        print("Warning: Labels don't match between runs!")
        mismatches = (df1_common["label"] != df2_common["label"]).sum()
        print(f"  {mismatches} mismatches found")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Cross-correlation matrix
    corr_path = output_dir / "cross_correlation.png"
    corr_matrix = plot_cross_correlation_matrix(df, run1_name, run2_name, corr_path)
    print(f"\nCorrelation Matrix:")
    print(corr_matrix.to_string())

    # Plot 2: Agreement heatmap
    agreement_path = output_dir / "agreement_heatmap.png"
    plot_agreement_heatmap(df, run1_name, run2_name, agreement_path)

    # Plot 3: 3D hyperplane
    hyperplane_path = output_dir / "3d_hyperplane.png"
    plot_3d_hyperplane(df, run1_name, run2_name, hyperplane_path)

    print(f"\n{'='*60}")
    print("Cross-run comparison complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    return df, corr_matrix


def analyze_single(
    results_path: Path,
    dataset_type: str,
    split: str,
    output_dir: Path,
    name: str = "",
    is_gemini: bool = False,
):
    """Analyze a single results directory."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {results_path}")
    print(f"Dataset type: {dataset_type}")
    print(f"{'='*60}")

    # Load predictions
    predictions = load_predictions(results_path)
    print(f"Loaded {len(predictions)} predictions")

    # Parse predictions
    pred_df = parse_predictions(predictions, is_gemini=is_gemini)

    # Load original dataset for features
    data_dir = Path("data")
    dataset_name = DATASETS[dataset_type]
    dataset = load_dataset(data_dir, dataset_name, split)

    # Match sizes
    if len(dataset) != len(predictions):
        print(f"Warning: Dataset size ({len(dataset)}) != predictions ({len(predictions)})")
        min_size = min(len(dataset), len(predictions))
        dataset = dataset[:min_size]
        pred_df = pred_df.iloc[:min_size]

    # Extract features
    features_df = extract_features(dataset, dataset_type)

    # Combine predictions with features
    df = pd.concat([pred_df, features_df[["tokens", "images", "pages"]]], axis=1)

    # Get features to plot
    features = FEATURES_TO_PLOT[dataset_type]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot correlation matrix
    suffix = f"_{name}" if name else ""
    corr_path = output_dir / f"correlation{suffix}.png"
    corr_df = plot_correlation_matrix(df, features, corr_path, title=name)

    # Print correlation values
    print(f"\nCorrelation Matrix:")
    print(corr_df.to_string())

    # Key correlations
    print(f"\nKey Correlations:")
    for feat in features:
        pred_corr = corr_df.loc["pred", feat]
        label_corr = corr_df.loc["label", feat]
        print(f"  {feat}: pred->{pred_corr:.3f}, label->{label_corr:.3f}")

    # Plot logistic regression
    logreg_path = output_dir / f"logistic_regression{suffix}.png"
    plot_logistic_regression(df, features, logreg_path, title=name)

    return df, corr_df


def main():
    parser = argparse.ArgumentParser(
        description="Correlative analysis of model predictions vs structural features"
    )

    # Original single-run analysis arguments
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Results directory (e.g., results/gemini/clean)",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        choices=["clean", "clean_images", "vision"],
        help="Dataset type for feature extraction",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variant names (e.g., 'ci_bd,vis_bd')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--is_gemini",
        action="store_true",
        help="Use Gemini-style answer parsing (simple string matching)",
    )

    # Cross-run comparison arguments
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Enable cross-run comparison mode",
    )
    parser.add_argument(
        "--run1_results",
        type=str,
        default=None,
        help="Run 1 results directory",
    )
    parser.add_argument(
        "--run1_dataset",
        type=str,
        default=None,
        help="Run 1 dataset name (full name without _test suffix)",
    )
    parser.add_argument(
        "--run1_name",
        type=str,
        default="Run1",
        help="Short name for Run 1",
    )
    parser.add_argument(
        "--run2_results",
        type=str,
        default=None,
        help="Run 2 results directory",
    )
    parser.add_argument(
        "--run2_dataset",
        type=str,
        default=None,
        help="Run 2 dataset name (full name without _test suffix)",
    )
    parser.add_argument(
        "--run2_name",
        type=str,
        default="Run2",
        help="Short name for Run 2",
    )

    args = parser.parse_args()

    # Cross-run comparison mode
    if args.compare:
        if not all([args.run1_results, args.run1_dataset, args.run2_results, args.run2_dataset]):
            parser.error("--compare requires --run1_results, --run1_dataset, --run2_results, --run2_dataset")

        output_dir = Path(args.output) if args.output else Path("results/correlative/cross_run")
        compare_runs(
            run1_results=Path(args.run1_results),
            run1_dataset=args.run1_dataset,
            run1_name=args.run1_name,
            run2_results=Path(args.run2_results),
            run2_dataset=args.run2_dataset,
            run2_name=args.run2_name,
            output_dir=output_dir,
            split=args.split,
            is_gemini=args.is_gemini,
        )
        return

    # Original single-run analysis mode
    if not args.results_dir or not args.dataset_type:
        parser.error("--results_dir and --dataset_type are required for single-run analysis")

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output) if args.output else Path("results/correlative") / results_dir.name

    if args.variants:
        # Process multiple variants
        variants = [v.strip() for v in args.variants.split(",")]
        for variant in variants:
            variant_path = results_dir / variant
            if variant_path.exists():
                analyze_single(
                    variant_path,
                    args.dataset_type,
                    args.split,
                    output_dir,
                    name=variant,
                    is_gemini=args.is_gemini,
                )
            else:
                print(f"Warning: Variant path not found: {variant_path}")
    else:
        # Process single directory
        analyze_single(
            results_dir,
            args.dataset_type,
            args.split,
            output_dir,
            name=results_dir.name,
            is_gemini=args.is_gemini,
        )

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
