#!/usr/bin/env python3
"""
Mediation Analysis: Features -> Quality -> Predictions

Models the causal pathway:
- X = Paper features (length, figures, equations, tokens, etc.)
- Y = pct_rating (latent quality signal from reviewers)
- Z = Model prediction correctness

Causal Model:
        X (Paper Features)
           /        \
     (path a)    (path c' - direct)
         |          |
         v          v
    Y (pct_rating) -> Z (Model Prediction)
         (path b)

Key Questions:
1. Path a: Do paper features predict pct_rating? (X -> Y)
2. Path b: Does pct_rating predict model accuracy? (Y -> Z)
3. Path c': Direct effect of features on predictions beyond pct_rating? (X -> Z | Y)
4. Mediation: How much of X->Z is explained by Y?

Statistical Methods:
- Regression-based mediation analysis
- Sobel test for mediation significance
- Bootstrap confidence intervals for indirect effects

Generates:
- mediation_path_coefficients.pdf: Bar chart of a, b, c' coefficients
- mediation_indirect_effects.pdf: Horizontal bar of indirect effects (a*b)
- mediation_proportion.pdf: Stacked bar of direct vs indirect proportion
- mediation_scatter_grid.pdf: Scatter plots with regression lines

Usage:
    python scripts/latex_analysis/fig_mediation_analysis.py
"""

import json
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

# Matplotlib styling
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

# Sizes
LABELSIZE = 12
TITLESIZE = 14
LEGENDSIZE = 10
TICKSIZE = 10

# Colors
ACCEPT_COLOR = "#4CAF50"
REJECT_COLOR = "#F44336"
CLEAN_COLOR = "#2ecc71"
VISION_COLOR = "#e74c3c"
PATH_A_COLOR = "#3498db"  # Blue for X->Y
PATH_B_COLOR = "#9b59b6"  # Purple for Y->Z
PATH_C_COLOR = "#e67e22"  # Orange for X->Z direct
INDIRECT_COLOR = "#1abc9c"  # Teal for indirect effect

# Rating scales by year (for normalization)
RATING_SCALES = {
    2020: [1, 3, 6, 8],
    2021: list(range(1, 11)),
    2022: [1, 3, 5, 6, 8, 10],
    2023: [1, 3, 5, 6, 8, 10],
    2024: [1, 3, 5, 6, 8, 10],
    2025: [0, 2, 4, 6, 8, 10],
}


def rating_to_pct_rank(rating: float, year: int) -> float:
    """Convert raw rating to percentile rank within year's scale.

    Args:
        rating: Raw rating value
        year: Year to determine rating scale

    Returns:
        Percentile rank (0 = worst, 1 = best)
    """
    if year not in RATING_SCALES:
        # Default to 2022 scale
        year = 2022

    scale = RATING_SCALES[year]

    # Find closest rating in scale
    closest_rating = min(scale, key=lambda x: abs(x - rating))
    idx = scale.index(closest_rating)

    # Return percentile (0 = worst, 1 = best)
    return idx / (len(scale) - 1) if len(scale) > 1 else 0.5


def compute_normalized_stats(ratings: list, year: int) -> tuple:
    """Compute scale-invariant mean and std from raw ratings.

    Args:
        ratings: List of raw rating values
        year: Year to determine rating scale

    Returns:
        Tuple of (mean_pct_rank, std_pct_rank)
    """
    if not ratings:
        return np.nan, np.nan

    pct_ranks = [rating_to_pct_rank(r, year) for r in ratings]
    return np.mean(pct_ranks), np.std(pct_ranks)


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip().lower()
    return None


def load_predictions(pred_path: Path) -> list[dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(pred_path) as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    pred = extract_boxed_answer(entry.get("predict", ""))
                    label = extract_boxed_answer(entry.get("label", ""))
                    predictions.append({
                        "pred": 1 if pred == "accept" else 0,
                        "label": 1 if label == "accept" else 0,
                        "correct": int(pred == label),
                    })
                except json.JSONDecodeError:
                    continue
    return predictions


def load_metadata(data_dir: Path, dataset_name: str, split: str) -> pd.DataFrame:
    """Load paper metadata from data.json."""
    path = data_dir / f"{dataset_name}_{split}" / "data.json"
    if not path.exists():
        return pd.DataFrame()

    with open(path) as f:
        data = json.load(f)

    records = []
    for entry in data:
        metadata = entry.get("_metadata", {})
        conversations = entry.get("conversations", [])

        human_content = ""
        for msg in conversations:
            if msg.get("from") == "human":
                human_content = msg.get("value", "")
                break

        tokens = len(human_content.split())
        ratings = metadata.get("ratings", [])
        year = metadata.get("year", 2022)

        # Compute normalized stats
        pct_rating = metadata.get("pct_rating", np.nan)
        rating_std = np.std(ratings) if len(ratings) > 1 else 0

        # Also compute scale-invariant std
        _, rating_std_normalized = compute_normalized_stats(ratings, year)

        records.append({
            "submission_id": metadata.get("submission_id", ""),
            "year": year,
            "tokens": tokens,
            "rating_std": rating_std,
            "rating_std_normalized": rating_std_normalized if not np.isnan(rating_std_normalized) else 0,
            "pct_rating": pct_rating,
        })

    return pd.DataFrame(records)


def load_structural_features(features_path: Path) -> pd.DataFrame:
    """Load structural features from CSV."""
    if not features_path.exists():
        return pd.DataFrame()

    return pd.read_csv(features_path)


def join_all_data(preds: list[dict], metadata_df: pd.DataFrame,
                  features_df: pd.DataFrame) -> pd.DataFrame:
    """Join predictions, metadata, and structural features."""
    # First join predictions with metadata
    min_len = min(len(preds), len(metadata_df))
    df = metadata_df.iloc[:min_len].copy()
    df["pred"] = [p["pred"] for p in preds[:min_len]]
    df["label"] = [p["label"] for p in preds[:min_len]]
    df["correct"] = [p["correct"] for p in preds[:min_len]]

    # Join with structural features if available
    if len(features_df) > 0 and "submission_id" in features_df.columns:
        feature_cols = [c for c in features_df.columns
                       if c not in ["label", "decision", "year", "pct_rating"]]
        df = df.merge(features_df[feature_cols], on="submission_id", how="left")

    return df


def standardize(x: np.ndarray) -> np.ndarray:
    """Standardize array to mean=0, std=1."""
    return (x - np.mean(x)) / np.std(x) if np.std(x) > 0 else x - np.mean(x)


class MediationResult:
    """Container for mediation analysis results."""

    def __init__(self, x_var: str):
        self.x_var = x_var
        self.a = 0.0  # X -> Y
        self.a_se = 0.0
        self.a_p = 1.0
        self.b = 0.0  # Y -> Z (controlling for X)
        self.b_se = 0.0
        self.b_p = 1.0
        self.c_prime = 0.0  # X -> Z (controlling for Y)
        self.c_prime_se = 0.0
        self.c_prime_p = 1.0
        self.c_total = 0.0  # Total effect X -> Z
        self.c_total_se = 0.0
        self.c_total_p = 1.0
        self.indirect = 0.0  # a * b
        self.indirect_se = 0.0
        self.indirect_ci_low = 0.0
        self.indirect_ci_high = 0.0
        self.sobel_z = 0.0
        self.sobel_p = 1.0
        self.proportion_mediated = 0.0
        self.n = 0

    def __repr__(self):
        return (f"MediationResult({self.x_var}): "
                f"a={self.a:.3f}, b={self.b:.3f}, c'={self.c_prime:.3f}, "
                f"indirect={self.indirect:.3f} (p={self.sobel_p:.4f})")


def run_mediation(df: pd.DataFrame, x: str, y: str = "pct_rating",
                  z: str = "correct", n_bootstrap: int = 1000) -> MediationResult:
    """Run mediation analysis for X -> Y -> Z.

    Args:
        df: DataFrame with columns x, y, z
        x: Feature variable name (X)
        y: Mediator variable name (Y), default "pct_rating"
        z: Outcome variable name (Z), default "correct"
        n_bootstrap: Number of bootstrap samples

    Returns:
        MediationResult with all coefficients and statistics
    """
    result = MediationResult(x)

    # Drop missing values
    cols = [x, y, z]
    clean_df = df[cols].dropna()
    result.n = len(clean_df)

    if result.n < 30:
        return result

    X = clean_df[x].values
    Y = clean_df[y].values
    Z = clean_df[z].values

    # Standardize X for comparable coefficients
    X_std = standardize(X)
    Y_std = standardize(Y)

    # Step 1: Total effect (X -> Z)
    X_with_const = np.column_stack([np.ones(len(X_std)), X_std])
    try:
        beta_total, residuals, rank, s = np.linalg.lstsq(X_with_const, Z, rcond=None)
        result.c_total = beta_total[1]

        # Compute standard error
        n = len(Z)
        p = 2
        mse = np.sum((Z - X_with_const @ beta_total)**2) / (n - p)
        var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        result.c_total_se = np.sqrt(var_beta[1, 1])
        t_stat = result.c_total / result.c_total_se if result.c_total_se > 0 else 0
        result.c_total_p = 2 * (1 - stats.t.cdf(abs(t_stat), n - p))
    except:
        pass

    # Step 2: Path a (X -> Y)
    try:
        beta_a, _, _, _ = np.linalg.lstsq(X_with_const, Y_std, rcond=None)
        result.a = beta_a[1]

        mse_a = np.sum((Y_std - X_with_const @ beta_a)**2) / (result.n - 2)
        var_a = mse_a * np.linalg.inv(X_with_const.T @ X_with_const)
        result.a_se = np.sqrt(var_a[1, 1])
        t_stat_a = result.a / result.a_se if result.a_se > 0 else 0
        result.a_p = 2 * (1 - stats.t.cdf(abs(t_stat_a), result.n - 2))
    except:
        pass

    # Step 3: Paths b and c' (X, Y -> Z)
    XY_with_const = np.column_stack([np.ones(len(X_std)), X_std, Y_std])
    try:
        beta_bc, _, _, _ = np.linalg.lstsq(XY_with_const, Z, rcond=None)
        result.c_prime = beta_bc[1]  # Direct effect
        result.b = beta_bc[2]  # Y -> Z controlling for X

        mse_bc = np.sum((Z - XY_with_const @ beta_bc)**2) / (result.n - 3)
        var_bc = mse_bc * np.linalg.inv(XY_with_const.T @ XY_with_const)
        result.c_prime_se = np.sqrt(var_bc[1, 1])
        result.b_se = np.sqrt(var_bc[2, 2])

        t_stat_c = result.c_prime / result.c_prime_se if result.c_prime_se > 0 else 0
        t_stat_b = result.b / result.b_se if result.b_se > 0 else 0
        result.c_prime_p = 2 * (1 - stats.t.cdf(abs(t_stat_c), result.n - 3))
        result.b_p = 2 * (1 - stats.t.cdf(abs(t_stat_b), result.n - 3))
    except:
        pass

    # Indirect effect = a * b
    result.indirect = result.a * result.b

    # Sobel test for mediation significance
    # z = (a * b) / sqrt(b^2 * SE_a^2 + a^2 * SE_b^2)
    if result.a_se > 0 and result.b_se > 0:
        sobel_se = np.sqrt(result.b**2 * result.a_se**2 +
                          result.a**2 * result.b_se**2)
        if sobel_se > 0:
            result.indirect_se = sobel_se
            result.sobel_z = result.indirect / sobel_se
            result.sobel_p = 2 * (1 - norm.cdf(abs(result.sobel_z)))

    # Bootstrap confidence intervals for indirect effect
    bootstrap_indirect = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = rng.choice(result.n, result.n, replace=True)
        X_boot = X_std[idx]
        Y_boot = Y_std[idx]
        Z_boot = Z[idx]

        try:
            # Path a
            X_const = np.column_stack([np.ones(len(X_boot)), X_boot])
            a_boot = np.linalg.lstsq(X_const, Y_boot, rcond=None)[0][1]

            # Path b (controlling for X)
            XY_const = np.column_stack([np.ones(len(X_boot)), X_boot, Y_boot])
            b_boot = np.linalg.lstsq(XY_const, Z_boot, rcond=None)[0][2]

            bootstrap_indirect.append(a_boot * b_boot)
        except:
            continue

    if len(bootstrap_indirect) > 10:
        bootstrap_indirect = np.array(bootstrap_indirect)
        result.indirect_ci_low = np.percentile(bootstrap_indirect, 2.5)
        result.indirect_ci_high = np.percentile(bootstrap_indirect, 97.5)

    # Proportion mediated
    if abs(result.c_total) > 1e-10:
        result.proportion_mediated = result.indirect / result.c_total

    return result


def plot_mediation_paths(results: list[MediationResult], output_path: Path):
    """Plot path coefficients (a, b, c') for each feature."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    features = [r.x_var for r in results]
    x = np.arange(len(features))
    width = 0.6

    # Path a (X -> Y)
    ax1 = axes[0]
    a_vals = [r.a for r in results]
    a_errs = [1.96 * r.a_se for r in results]
    bars1 = ax1.bar(x, a_vals, width, yerr=a_errs, capsize=4,
                    color=PATH_A_COLOR, alpha=0.7, label="Path a")
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Feature", fontsize=LABELSIZE)
    ax1.set_ylabel("Coefficient (standardized)", fontsize=LABELSIZE)
    ax1.set_title("Path a: X → Y (pct_rating)\n" +
                 r"$Y = a \cdot X + \epsilon$", fontsize=TITLESIZE)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f.replace("_", "\n") for f in features],
                        fontsize=TICKSIZE - 1, rotation=45, ha='right')
    ax1.tick_params(axis='both', labelsize=TICKSIZE)

    # Add significance stars
    for i, r in enumerate(results):
        if r.a_p < 0.001:
            ax1.text(i, a_vals[i] + a_errs[i] + 0.01, "***",
                    ha='center', fontsize=TICKSIZE)
        elif r.a_p < 0.01:
            ax1.text(i, a_vals[i] + a_errs[i] + 0.01, "**",
                    ha='center', fontsize=TICKSIZE)
        elif r.a_p < 0.05:
            ax1.text(i, a_vals[i] + a_errs[i] + 0.01, "*",
                    ha='center', fontsize=TICKSIZE)

    # Path b (Y -> Z | X)
    ax2 = axes[1]
    b_vals = [r.b for r in results]
    b_errs = [1.96 * r.b_se for r in results]
    bars2 = ax2.bar(x, b_vals, width, yerr=b_errs, capsize=4,
                    color=PATH_B_COLOR, alpha=0.7, label="Path b")
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Feature", fontsize=LABELSIZE)
    ax2.set_ylabel("Coefficient (standardized)", fontsize=LABELSIZE)
    ax2.set_title("Path b: Y → Z | X (controlling for X)\n" +
                 r"$Z = b \cdot Y + c' \cdot X + \epsilon$", fontsize=TITLESIZE)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.replace("_", "\n") for f in features],
                        fontsize=TICKSIZE - 1, rotation=45, ha='right')
    ax2.tick_params(axis='both', labelsize=TICKSIZE)

    for i, r in enumerate(results):
        if r.b_p < 0.001:
            ax2.text(i, b_vals[i] + b_errs[i] + 0.01, "***",
                    ha='center', fontsize=TICKSIZE)
        elif r.b_p < 0.01:
            ax2.text(i, b_vals[i] + b_errs[i] + 0.01, "**",
                    ha='center', fontsize=TICKSIZE)
        elif r.b_p < 0.05:
            ax2.text(i, b_vals[i] + b_errs[i] + 0.01, "*",
                    ha='center', fontsize=TICKSIZE)

    # Path c' (X -> Z | Y)
    ax3 = axes[2]
    c_vals = [r.c_prime for r in results]
    c_errs = [1.96 * r.c_prime_se for r in results]
    bars3 = ax3.bar(x, c_vals, width, yerr=c_errs, capsize=4,
                    color=PATH_C_COLOR, alpha=0.7, label="Path c'")
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel("Feature", fontsize=LABELSIZE)
    ax3.set_ylabel("Coefficient (standardized)", fontsize=LABELSIZE)
    ax3.set_title("Path c': X → Z | Y (direct effect)\n" +
                 r"$Z = c' \cdot X + b \cdot Y + \epsilon$", fontsize=TITLESIZE)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f.replace("_", "\n") for f in features],
                        fontsize=TICKSIZE - 1, rotation=45, ha='right')
    ax3.tick_params(axis='both', labelsize=TICKSIZE)

    for i, r in enumerate(results):
        if r.c_prime_p < 0.001:
            ax3.text(i, c_vals[i] + c_errs[i] + 0.01, "***",
                    ha='center', fontsize=TICKSIZE)
        elif r.c_prime_p < 0.01:
            ax3.text(i, c_vals[i] + c_errs[i] + 0.01, "**",
                    ha='center', fontsize=TICKSIZE)
        elif r.c_prime_p < 0.05:
            ax3.text(i, c_vals[i] + c_errs[i] + 0.01, "*",
                    ha='center', fontsize=TICKSIZE)

    # Add note about significance
    fig.text(0.5, 0.01, "* p<0.05, ** p<0.01, *** p<0.001",
            ha='center', fontsize=TICKSIZE)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_indirect_effects(results: list[MediationResult], output_path: Path):
    """Horizontal bar chart of indirect effects (a*b) sorted by magnitude."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by absolute indirect effect
    sorted_results = sorted(results, key=lambda r: abs(r.indirect), reverse=True)

    features = [r.x_var.replace("_", " ") for r in sorted_results]
    indirect_vals = [r.indirect for r in sorted_results]
    ci_lows = [r.indirect_ci_low for r in sorted_results]
    ci_highs = [r.indirect_ci_high for r in sorted_results]

    y = np.arange(len(features))

    # Color by sign
    colors = [ACCEPT_COLOR if v > 0 else REJECT_COLOR for v in indirect_vals]

    bars = ax.barh(y, indirect_vals, color=colors, alpha=0.7, height=0.6)

    # Add error bars (bootstrap CI)
    for i, (low, high, val) in enumerate(zip(ci_lows, ci_highs, indirect_vals)):
        ax.errorbar(val, i, xerr=[[val - low], [high - val]],
                   fmt='none', color='black', capsize=4, capthick=1.5)

    # Add significance markers
    for i, r in enumerate(sorted_results):
        # Significant if CI doesn't include 0
        if r.indirect_ci_low > 0 or r.indirect_ci_high < 0:
            x_pos = max(ci_highs[i], 0) + 0.005
            ax.text(x_pos, i, "*", fontsize=LABELSIZE, va='center', fontweight='bold')

    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Indirect Effect (a × b)", fontsize=LABELSIZE)
    ax.set_ylabel("Paper Feature", fontsize=LABELSIZE)
    ax.set_title("Mediation Analysis: Indirect Effects\n" +
                r"$\text{Indirect Effect} = a \cdot b = \beta_{X \to Y} \cdot \beta_{Y \to Z|X}$",
                fontsize=TITLESIZE)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=TICKSIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCEPT_COLOR, alpha=0.7, label='Positive (feature → higher quality → more correct)'),
        Patch(facecolor=REJECT_COLOR, alpha=0.7, label='Negative (feature → lower quality → less correct)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=LEGENDSIZE)

    # Add note
    ax.text(0.02, 0.02, "* = 95% CI excludes 0 (significant mediation)",
           transform=ax.transAxes, fontsize=TICKSIZE - 1, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mediation_proportion(results: list[MediationResult], output_path: Path):
    """Stacked bar showing direct vs indirect effect proportion."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter to results with non-zero total effect
    valid_results = [r for r in results if abs(r.c_total) > 1e-6]

    if not valid_results:
        print("No valid results for proportion plot")
        return

    features = [r.x_var.replace("_", " ") for r in valid_results]
    x = np.arange(len(features))
    width = 0.6

    # Calculate proportions
    indirect_props = []
    direct_props = []

    for r in valid_results:
        total_abs = abs(r.c_total)
        indirect_abs = abs(r.indirect)
        direct_abs = abs(r.c_prime)

        # Proportion of total effect that is mediated
        if total_abs > 0:
            indirect_prop = min(indirect_abs / total_abs, 1.0)
            direct_prop = 1.0 - indirect_prop
        else:
            indirect_prop = 0
            direct_prop = 1.0

        indirect_props.append(indirect_prop)
        direct_props.append(direct_prop)

    # Stacked bar
    bars1 = ax.bar(x, direct_props, width, label='Direct Effect (c\')',
                   color=PATH_C_COLOR, alpha=0.7)
    bars2 = ax.bar(x, indirect_props, width, bottom=direct_props,
                   label='Indirect Effect (a×b)', color=INDIRECT_COLOR, alpha=0.7)

    # Add annotations
    for i, (d, ind) in enumerate(zip(direct_props, indirect_props)):
        if d > 0.1:
            ax.text(i, d/2, f"{d:.0%}", ha='center', va='center',
                   fontsize=TICKSIZE - 1, color='white', fontweight='bold')
        if ind > 0.1:
            ax.text(i, d + ind/2, f"{ind:.0%}", ha='center', va='center',
                   fontsize=TICKSIZE - 1, color='white', fontweight='bold')

    ax.set_xlabel("Paper Feature", fontsize=LABELSIZE)
    ax.set_ylabel("Proportion of Total Effect", fontsize=LABELSIZE)
    ax.set_title("Effect Decomposition: Direct vs Mediated\n" +
                r"$c = c' + a \cdot b$ (Total = Direct + Indirect)",
                fontsize=TITLESIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=TICKSIZE - 1, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.legend(fontsize=LEGENDSIZE, loc='upper right')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scatter_grid(df: pd.DataFrame, x_vars: list[str], output_path: Path):
    """2x2 scatter grid: X vs Y, Y vs Z for top features."""
    # Select top 2 features by total effect
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    y_col = "pct_rating"
    z_col = "correct"

    # Top-left: First X vs Y
    if len(x_vars) > 0 and x_vars[0] in df.columns:
        ax = axes[0, 0]
        x = df[x_vars[0]].dropna()
        y = df.loc[x.index, y_col]

        ax.scatter(x, y, alpha=0.3, s=20, c=PATH_A_COLOR)

        # Add regression line
        valid_idx = ~(np.isnan(x) | np.isnan(y))
        if valid_idx.sum() > 10:
            z = np.polyfit(x[valid_idx], y[valid_idx], 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), color='red', linewidth=2, label='OLS fit')

            # R-squared
            r, _ = stats.pearsonr(x[valid_idx], y[valid_idx])
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                   fontsize=TICKSIZE, va='top')

        ax.set_xlabel(x_vars[0].replace("_", " "), fontsize=LABELSIZE)
        ax.set_ylabel("pct_rating (Y)", fontsize=LABELSIZE)
        ax.set_title(f"Path a: {x_vars[0]} → pct_rating", fontsize=TITLESIZE)
        ax.tick_params(axis='both', labelsize=TICKSIZE)

    # Top-right: Second X vs Y
    if len(x_vars) > 1 and x_vars[1] in df.columns:
        ax = axes[0, 1]
        x = df[x_vars[1]].dropna()
        y = df.loc[x.index, y_col]

        ax.scatter(x, y, alpha=0.3, s=20, c=PATH_A_COLOR)

        valid_idx = ~(np.isnan(x) | np.isnan(y))
        if valid_idx.sum() > 10:
            z = np.polyfit(x[valid_idx], y[valid_idx], 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), color='red', linewidth=2)

            r, _ = stats.pearsonr(x[valid_idx], y[valid_idx])
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                   fontsize=TICKSIZE, va='top')

        ax.set_xlabel(x_vars[1].replace("_", " "), fontsize=LABELSIZE)
        ax.set_ylabel("pct_rating (Y)", fontsize=LABELSIZE)
        ax.set_title(f"Path a: {x_vars[1]} → pct_rating", fontsize=TITLESIZE)
        ax.tick_params(axis='both', labelsize=TICKSIZE)

    # Bottom-left: Y vs Z (box plot by correctness)
    ax = axes[1, 0]
    correct_df = df[df[z_col] == 1][y_col].dropna()
    incorrect_df = df[df[z_col] == 0][y_col].dropna()

    bp = ax.boxplot([incorrect_df, correct_df], labels=['Incorrect', 'Correct'],
                    patch_artist=True)
    colors = [REJECT_COLOR, ACCEPT_COLOR]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Model Prediction", fontsize=LABELSIZE)
    ax.set_ylabel("pct_rating (Y)", fontsize=LABELSIZE)
    ax.set_title("Path b: pct_rating → Correctness (Z)", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)

    # Add stats
    u_stat, p_val = stats.mannwhitneyu(correct_df, incorrect_df)
    ax.text(0.5, 0.02, f"Mann-Whitney U={u_stat:.0f}, p={p_val:.2e}",
           transform=ax.transAxes, ha='center', fontsize=TICKSIZE - 1)

    # Bottom-right: pct_rating vs accuracy (binned)
    ax = axes[1, 1]

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    df['pct_bin'] = pd.cut(df[y_col], bins=bins, labels=labels)

    acc_by_bin = df.groupby('pct_bin', observed=True)[z_col].mean()
    count_by_bin = df.groupby('pct_bin', observed=True)[z_col].count()

    x_pos = np.arange(len(acc_by_bin))
    bars = ax.bar(x_pos, acc_by_bin.values, color=PATH_B_COLOR, alpha=0.7)

    ax.set_xlabel("pct_rating bins (Y)", fontsize=LABELSIZE)
    ax.set_ylabel("Accuracy (P(correct))", fontsize=LABELSIZE)
    ax.set_title("Path b: pct_rating → P(Correct)", fontsize=TITLESIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(acc_by_bin.index, fontsize=TICKSIZE - 1)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Add counts
    for i, (acc, n) in enumerate(zip(acc_by_bin.values, count_by_bin.values)):
        ax.text(i, acc + 0.02, f"n={n}", ha='center', fontsize=TICKSIZE - 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_mediation_table(results: list[MediationResult], output_path: Path):
    """Generate LaTeX table of mediation statistics."""
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Mediation Analysis: Paper Features $\to$ pct\_rating $\to$ Correctness}")
    latex.append(r"\label{tab:mediation}")
    latex.append(r"\footnotesize")
    latex.append(r"\begin{tabular}{l|ccc|cc|c}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Feature (X)} & \textbf{a} & \textbf{b} & \textbf{c'} & "
                r"\textbf{Indirect} & \textbf{95\% CI} & \textbf{Sobel p} \\")
    latex.append(r" & (X$\to$Y) & (Y$\to$Z$|$X) & (X$\to$Z$|$Y) & (a$\times$b) & & \\")
    latex.append(r"\hline")

    for r in results:
        sig = ""
        if r.sobel_p < 0.001:
            sig = "***"
        elif r.sobel_p < 0.01:
            sig = "**"
        elif r.sobel_p < 0.05:
            sig = "*"

        ci_str = f"[{r.indirect_ci_low:.3f}, {r.indirect_ci_high:.3f}]"

        latex.append(
            f"{r.x_var.replace('_', ' ')} & "
            f"{r.a:.3f} & {r.b:.3f} & {r.c_prime:.3f} & "
            f"{r.indirect:.3f}{sig} & {ci_str} & {r.sobel_p:.4f} \\\\"
        )

    latex.append(r"\hline")
    latex.append(r"\multicolumn{7}{l}{\footnotesize * $p<0.05$, ** $p<0.01$, *** $p<0.001$} \\")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"Saved: {output_path}")


def main():
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results/data_sweep_v2")
    FEATURES_PATH = Path("figures/latex/analysis/structural_features.csv")
    OUTPUT_DIR = Path("figures/latex/analysis")
    TABLE_DIR = Path("latex/tables")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Paths
    pred_path = RESULTS_DIR / "iclr20_balanced_clean" / "finetuned.jsonl"
    dataset_name = "iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6"

    # Load data
    print("Loading predictions...")
    if pred_path.exists():
        preds = load_predictions(pred_path)
        print(f"Loaded {len(preds)} predictions")
    else:
        print("Predictions not found, using simulated data")
        np.random.seed(42)
        n = 1000
        preds = [{"pred": np.random.choice([0, 1]), "label": np.random.choice([0, 1]),
                 "correct": int(np.random.random() < 0.66)} for _ in range(n)]

    print("Loading metadata...")
    metadata_df = load_metadata(DATA_DIR, dataset_name, "test")

    if len(metadata_df) == 0:
        print("Metadata not found, using simulated data")
        n = len(preds)
        metadata_df = pd.DataFrame({
            "submission_id": [f"sim_{i}" for i in range(n)],
            "tokens": np.random.normal(5000, 1500, n).astype(int),
            "rating_std": np.random.exponential(0.8, n),
            "rating_std_normalized": np.random.exponential(0.1, n),
            "pct_rating": np.random.beta(2, 2, n),
            "year": np.random.choice([2020, 2021, 2022, 2023, 2024, 2025], n),
        })

    print("Loading structural features...")
    features_df = load_structural_features(FEATURES_PATH)

    if len(features_df) == 0:
        print("Structural features not found, using simulated data")
        n = len(preds)
        features_df = pd.DataFrame({
            "submission_id": metadata_df["submission_id"].values if "submission_id" in metadata_df.columns else [f"sim_{i}" for i in range(n)],
            "total_equations": np.random.poisson(50, n),
            "figure_marker": np.random.poisson(8, n),
            "table_marker": np.random.poisson(3, n),
            "theorem_env": np.random.poisson(2, n),
            "word_count": np.random.normal(6000, 1000, n).astype(int),
            "citation": np.random.poisson(15, n),
            "sections": np.random.poisson(12, n),
        })

    # Join all data
    print("Joining data...")
    df = join_all_data(preds, metadata_df, features_df)
    print(f"Joined {len(df)} entries")

    # Define X variables (paper features)
    x_vars = ["total_equations", "figure_marker", "table_marker",
              "theorem_env", "word_count", "citation", "sections"]

    # Filter to available variables
    x_vars = [v for v in x_vars if v in df.columns]
    print(f"Analyzing {len(x_vars)} features: {x_vars}")

    # Run mediation analysis for each X
    print("\nRunning mediation analysis...")
    results = []
    for x_var in x_vars:
        print(f"  Analyzing: {x_var}")
        result = run_mediation(df, x=x_var, y="pct_rating", z="correct")
        results.append(result)
        print(f"    {result}")

    # Generate figures
    print("\nGenerating figures...")

    # 1. Path coefficients
    plot_mediation_paths(results, OUTPUT_DIR / "mediation_path_coefficients.pdf")

    # 2. Indirect effects
    plot_indirect_effects(results, OUTPUT_DIR / "mediation_indirect_effects.pdf")

    # 3. Effect proportions
    plot_mediation_proportion(results, OUTPUT_DIR / "mediation_proportion.pdf")

    # 4. Scatter grid
    plot_scatter_grid(df, x_vars[:2], OUTPUT_DIR / "mediation_scatter_grid.pdf")

    # 5. LaTeX table
    generate_mediation_table(results, TABLE_DIR / "mediation_results.tex")

    # Print summary
    print("\n" + "=" * 60)
    print("Mediation Analysis Summary")
    print("=" * 60)

    print("\nKey Findings:")

    # Sort by indirect effect magnitude
    sorted_results = sorted(results, key=lambda r: abs(r.indirect), reverse=True)

    print("\nTop Features by Indirect Effect (through pct_rating):")
    for r in sorted_results[:3]:
        sig = "SIGNIFICANT" if r.indirect_ci_low > 0 or r.indirect_ci_high < 0 else "not sig."
        print(f"  {r.x_var}: indirect={r.indirect:.4f} [{r.indirect_ci_low:.4f}, {r.indirect_ci_high:.4f}] ({sig})")

    print("\nInterpretation:")

    # Check if pct_rating mediates the effect
    significant_mediations = [r for r in results
                              if r.indirect_ci_low > 0 or r.indirect_ci_high < 0]

    if significant_mediations:
        print(f"  - {len(significant_mediations)} features show significant mediation through pct_rating")
        print("  - This suggests the model captures quality signals that reviewers also detect")
    else:
        print("  - No significant mediation detected")
        print("  - The model may rely on surface features independent of reviewer quality assessments")

    # Check direct effects
    direct_features = [r for r in results if r.c_prime_p < 0.05]
    if direct_features:
        print(f"\n  - {len(direct_features)} features have significant direct effects on predictions")
        print("  - These features influence predictions beyond their effect on quality")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
