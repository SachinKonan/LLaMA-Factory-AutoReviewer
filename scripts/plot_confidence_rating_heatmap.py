#!/usr/bin/env python3
"""
Continuous contour heatmap: model confidence (x) vs reviewer rating (y), colored by accuracy.

Uses Gaussian kernel smoothing + filled contours for a smooth surface with bold isoclines.
Uses best optim_search_2026 checkpoints.

Usage:
    python scripts/plot_confidence_rating_heatmap.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "Text (bz32, ep2)": {
        "pred_path": PROJECT_ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/finetuned-ckpt-1322.jsonl",
        "dataset": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered",
    },
    "Vision (bz16, ep2)": {
        "pred_path": PROJECT_ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl",
        "dataset": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480",
    },
}

DECISION_TOKEN_IDX = 5
DPI = 150

# Smoothing params
GRID_RES = 40
SMOOTH_SIGMA = 3.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_prediction(text: str) -> str:
    text_lower = text.lower().strip()
    if "\\boxed{accept}" in text_lower or "boxed{accept}" in text_lower:
        return "accept"
    if "\\boxed{reject}" in text_lower or "boxed{reject}" in text_lower:
        return "reject"
    if "accept" in text_lower:
        return "accept"
    if "reject" in text_lower:
        return "reject"
    return "unknown"


def load_data(pred_path, dataset_name):
    """Load predictions + metadata."""
    preds_data = []
    with open(pred_path) as f:
        for line in f:
            preds_data.append(json.loads(line))

    test_path = PROJECT_ROOT / "data" / f"{dataset_name}_test" / "data.json"
    with open(test_path) as f:
        test_data = json.load(f)

    confidences, correct_arr, ratings, years = [], [], [], []
    pred_labels, true_labels = [], []

    for i, rec in enumerate(preds_data):
        if i >= len(test_data):
            break
        meta = test_data[i].get("_metadata", {})
        pct_rating = meta.get("pct_rating")
        year = meta.get("year")
        if pct_rating is None:
            continue

        pred = extract_prediction(rec["predict"])
        label = extract_prediction(rec["label"])
        logprobs = rec["token_logprobs"]
        conf = math.exp(logprobs[DECISION_TOKEN_IDX])

        confidences.append(conf)
        correct_arr.append(pred == label and pred != "unknown")
        ratings.append(pct_rating)
        years.append(year)
        pred_labels.append(pred)
        true_labels.append(label)

    return (
        np.array(confidences),
        np.array(correct_arr, dtype=float),
        np.array(ratings),
        np.array(years),
        pred_labels,
        true_labels,
    )


def make_smooth_surface(conf, correct, rating, grid_res=GRID_RES, sigma=SMOOTH_SIGMA):
    """Build smooth accuracy surface via Gaussian-weighted binning."""
    conf_centers = np.linspace(0.5, 1.0, grid_res)
    rat_centers = np.linspace(0.0, 1.0, grid_res)

    dc = (conf_centers[1] - conf_centers[0]) / 2
    dr = (rat_centers[1] - rat_centers[0]) / 2
    conf_edges = np.concatenate([[conf_centers[0] - dc], conf_centers + dc])
    rat_edges = np.concatenate([[rat_centers[0] - dr], rat_centers + dr])

    correct_grid = np.zeros((grid_res, grid_res))
    total_grid = np.zeros((grid_res, grid_res))

    ci = np.clip(np.digitize(conf, conf_edges) - 1, 0, grid_res - 1)
    ri = np.clip(np.digitize(rating, rat_edges) - 1, 0, grid_res - 1)

    for k in range(len(conf)):
        total_grid[ri[k], ci[k]] += 1
        correct_grid[ri[k], ci[k]] += correct[k]

    smooth_total = gaussian_filter(total_grid.astype(float), sigma=sigma)
    smooth_correct = gaussian_filter(correct_grid.astype(float), sigma=sigma)

    # Where we have any smoothed data, compute accuracy
    # Use a very low threshold — the Gaussian already handles smoothing
    mask = smooth_total > 0.1
    acc_surface = np.full((grid_res, grid_res), np.nan)
    acc_surface[mask] = smooth_correct[mask] / smooth_total[mask] * 100

    return acc_surface, smooth_total, conf_centers, rat_centers


def plot_contour_heatmap(ax, acc_surface, density, conf_centers, rat_centers,
                          title, show_ylabel=True, show_cbar=True):
    """Plot filled contour heatmap with bold isoclines."""
    # Fill NaN with nearest valid for contourf (it can't handle NaN)
    from scipy.ndimage import distance_transform_edt
    nan_mask = np.isnan(acc_surface)
    if nan_mask.any() and not nan_mask.all():
        # Fill NaN with nearest-neighbor interpolation
        ind = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        acc_filled = acc_surface[tuple(ind)]
    else:
        acc_filled = acc_surface.copy()

    # Filled contour — smooth continuous background
    levels_fill = np.linspace(45, 90, 40)
    cf = ax.contourf(
        conf_centers, rat_centers, acc_filled,
        levels=levels_fill, cmap="RdYlGn", extend="both",
    )

    # Bold isoclines with labels
    iso_levels = [50, 55, 60, 65, 70, 75, 80, 85]
    cs = ax.contour(
        conf_centers, rat_centers, acc_filled,
        levels=iso_levels,
        colors="black", linewidths=2.0, alpha=0.8,
    )
    clabels = ax.clabel(cs, inline=True, fontsize=11, fmt="%.0f%%", inline_spacing=10)
    for txt in clabels:
        txt.set_fontweight("bold")
        txt.set_bbox(dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.75, edgecolor="none"))

    # Lighter sub-isoclines
    sub_levels = [52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5]
    ax.contour(
        conf_centers, rat_centers, acc_filled,
        levels=sub_levels,
        colors="black", linewidths=0.6, linestyles="--", alpha=0.3,
    )

    # Data density contours (white dotted) to show where samples live
    if density.max() > 0:
        nonzero = density[density > 0]
        if len(nonzero) > 10:
            d_levels = np.percentile(nonzero, [60, 85, 95])
            ax.contour(
                conf_centers, rat_centers, density,
                levels=d_levels,
                colors="white", linewidths=0.8, linestyles=":", alpha=0.5,
            )

    ax.set_xlabel("Model Confidence", fontsize=12, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Reviewer Rating Percentile", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(labelsize=10)

    return cf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading data ...")

all_data = {}
for name, cfg in RUNS.items():
    conf, corr, rat, yr, pl, tl = load_data(cfg["pred_path"], cfg["dataset"])
    all_data[name] = {
        "conf": conf, "correct": corr, "rating": rat, "year": yr,
        "pred_labels": pl, "true_labels": tl,
    }
    print(f"  {name}: {len(conf)} samples, acc={corr.mean():.4f}")

# ---------------------------------------------------------------------------
# Figure 1: Side-by-side continuous contour heatmaps
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 7))
gs = GridSpec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.2)

for idx, (name, data) in enumerate(all_data.items()):
    ax = fig.add_subplot(gs[0, idx])
    acc_surf, density, cc, rc = make_smooth_surface(
        data["conf"], data["correct"], data["rating"]
    )
    cf = plot_contour_heatmap(ax, acc_surf, density, cc, rc, name,
                               show_ylabel=(idx == 0))

cax = fig.add_subplot(gs[0, 2])
cbar = fig.colorbar(cf, cax=cax)
cbar.set_label("Accuracy (%)", fontsize=12, fontweight="bold")
cbar.ax.tick_params(labelsize=10)

fig.suptitle("Accuracy by Model Confidence × Reviewer Rating",
             fontsize=16, fontweight="bold", y=1.02)
out1 = FIGURES_DIR / "confidence_rating_heatmap.png"
fig.savefig(out1, dpi=DPI, bbox_inches="tight")
print(f"Saved {out1}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Individual heatmaps with marginal distributions
# ---------------------------------------------------------------------------
for name, data in all_data.items():
    fig = plt.figure(figsize=(10, 10))
    gs_inner = GridSpec(
        2, 2, width_ratios=[0.18, 1], height_ratios=[0.18, 1],
        wspace=0.05, hspace=0.05,
    )

    # Main heatmap
    ax_main = fig.add_subplot(gs_inner[1, 1])
    acc_surf, density, cc, rc = make_smooth_surface(
        data["conf"], data["correct"], data["rating"]
    )
    cf = plot_contour_heatmap(ax_main, acc_surf, density, cc, rc, "")

    cbar = fig.colorbar(cf, ax=ax_main, fraction=0.03, pad=0.01)
    cbar.set_label("Accuracy (%)", fontsize=10, fontweight="bold")

    # Top marginal
    ax_top = fig.add_subplot(gs_inner[0, 1], sharex=ax_main)
    n_marg = 20
    ce = np.linspace(0.5, 1.0, n_marg + 1)
    cm = (ce[:-1] + ce[1:]) / 2
    ccounts = np.array([((data["conf"] >= ce[i]) & (data["conf"] < ce[i + 1])).sum()
                         for i in range(n_marg)])
    caccs = np.array([
        data["correct"][(data["conf"] >= ce[i]) & (data["conf"] < ce[i + 1])].mean() * 100
        if ccounts[i] > 0 else np.nan for i in range(n_marg)
    ])

    ax_top.bar(cm, ccounts, width=0.023, alpha=0.5, color="#1f77b4")
    ax_top_acc = ax_top.twinx()
    v = ~np.isnan(caccs)
    ax_top_acc.plot(cm[v], caccs[v], "ro-", markersize=4, linewidth=2, zorder=5)
    ax_top_acc.set_ylim(40, 100)
    ax_top_acc.set_ylabel("Acc %", fontsize=9, color="red", fontweight="bold")
    ax_top_acc.tick_params(labelsize=8, colors="red")
    ax_top.set_ylabel("Count", fontsize=9, fontweight="bold")
    ax_top.tick_params(labelsize=8)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_title(f"{name}: Confidence × Rating → Accuracy",
                      fontsize=14, fontweight="bold")

    # Left marginal
    ax_left = fig.add_subplot(gs_inner[1, 0], sharey=ax_main)
    re = np.linspace(0.0, 1.0, n_marg + 1)
    rm = (re[:-1] + re[1:]) / 2
    rcounts = np.array([((data["rating"] >= re[i]) & (data["rating"] < re[i + 1])).sum()
                         for i in range(n_marg)])
    raccs = np.array([
        data["correct"][(data["rating"] >= re[i]) & (data["rating"] < re[i + 1])].mean() * 100
        if rcounts[i] > 0 else np.nan for i in range(n_marg)
    ])

    ax_left.barh(rm, rcounts, height=0.045, alpha=0.5, color="#ff7f0e")
    ax_left_acc = ax_left.twiny()
    vr = ~np.isnan(raccs)
    ax_left_acc.plot(raccs[vr], rm[vr], "go-", markersize=4, linewidth=2, zorder=5)
    ax_left_acc.set_xlim(40, 100)
    ax_left_acc.set_xlabel("Acc %", fontsize=9, color="green", fontweight="bold")
    ax_left_acc.tick_params(labelsize=8, colors="green")
    ax_left.set_xlabel("Count", fontsize=9, fontweight="bold")
    ax_left.tick_params(labelsize=8)
    ax_left.invert_xaxis()
    plt.setp(ax_left.get_yticklabels(), visible=False)

    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    out = FIGURES_DIR / f"confidence_rating_heatmap_{safe_name}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: Scatter with quadrant accuracy
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (name, data) in enumerate(all_data.items()):
    ax = axes[idx]
    conf = data["conf"]
    rating = data["rating"]
    correct = data["correct"].astype(bool)

    high_conf = conf >= 0.7
    high_rating = rating >= 0.5

    ax.scatter(conf[correct], rating[correct],
               c="green", alpha=0.15, s=10, label=f"Correct ({correct.sum()})",
               rasterized=True)
    ax.scatter(conf[~correct], rating[~correct],
               c="red", alpha=0.25, s=10, label=f"Wrong ({(~correct).sum()})",
               rasterized=True)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.axvline(0.7, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)

    quadrants = [
        (~high_conf & ~high_rating),
        (~high_conf & high_rating),
        (high_conf & ~high_rating),
        (high_conf & high_rating),
    ]
    positions = [(0.55, 0.25), (0.55, 0.75), (0.85, 0.25), (0.85, 0.75)]

    for qmask, (px, py) in zip(quadrants, positions):
        n = qmask.sum()
        if n > 0:
            qacc = correct[qmask].mean() * 100
            ax.text(px, py, f"{qacc:.0f}%\nn={n}", ha="center", va="center",
                    fontsize=10, fontweight="bold", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.85))

    ax.set_xlabel("Model Confidence", fontsize=12, fontweight="bold")
    ax.set_ylabel("Reviewer Rating Percentile", fontsize=12, fontweight="bold")
    ax.set_title(name, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=10)

fig.suptitle("Model Confidence vs Reviewer Rating: Correct/Incorrect",
             fontsize=15, fontweight="bold")
plt.tight_layout()
out3 = FIGURES_DIR / "confidence_rating_scatter.png"
fig.savefig(out3, dpi=DPI, bbox_inches="tight")
print(f"Saved {out3}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4 (Vision only): 3D joint density surface
#   Confidence (x) × Rating (y) → sample density (z)
# ---------------------------------------------------------------------------
VIS_KEY = "Vision (bz16, ep2)"
vdata = all_data[VIS_KEY]

GRID_3D = 30
SIGMA_3D = 2.5

def build_density_grid(conf, rating, grid_res=GRID_3D, sigma=SIGMA_3D):
    """KDE-like smoothed 2D histogram → density surface."""
    conf_c = np.linspace(0.5, 1.0, grid_res)
    rat_c = np.linspace(0.0, 1.0, grid_res)
    dc = (conf_c[1] - conf_c[0]) / 2
    dr = (rat_c[1] - rat_c[0]) / 2
    conf_e = np.concatenate([[conf_c[0] - dc], conf_c + dc])
    rat_e = np.concatenate([[rat_c[0] - dr], rat_c + dr])

    grid = np.zeros((grid_res, grid_res))
    ci = np.clip(np.digitize(conf, conf_e) - 1, 0, grid_res - 1)
    ri = np.clip(np.digitize(rating, rat_e) - 1, 0, grid_res - 1)
    for k in range(len(conf)):
        grid[ri[k], ci[k]] += 1

    smooth = gaussian_filter(grid.astype(float), sigma=sigma)
    return smooth, conf_c, rat_c

density_all, cc3, rc3 = build_density_grid(vdata["conf"], vdata["rating"])
# Normalise to percentage of total samples
density_pct = density_all / density_all.sum() * 100

CC, RC = np.meshgrid(cc3, rc3)

from mpl_toolkits.mplot3d import Axes3D  # noqa: E402
from matplotlib import cm  # noqa: E402

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(CC, RC, density_pct, cmap="viridis", edgecolor="none", alpha=0.85)
ax.set_xlabel("Model Confidence", fontsize=11, fontweight="bold", labelpad=10)
ax.set_ylabel("Rating Percentile", fontsize=11, fontweight="bold", labelpad=10)
ax.set_zlabel("% of Samples", fontsize=11, fontweight="bold", labelpad=8)
ax.set_title(f"{VIS_KEY}: Joint Density of Confidence × Rating",
             fontsize=14, fontweight="bold")
ax.view_init(elev=30, azim=-120)
fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1, label="% of Samples")
out4 = FIGURES_DIR / "confidence_rating_3d_density_vision.png"
fig.savefig(out4, dpi=DPI, bbox_inches="tight")
print(f"Saved {out4}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5 (Vision only): 3D density split by Accept vs Reject (true label)
#   Two semi-transparent surfaces on the same axes
# ---------------------------------------------------------------------------
accept_mask = np.array([l == "accept" for l in vdata["true_labels"]])
reject_mask = ~accept_mask

density_acc, _, _ = build_density_grid(
    vdata["conf"][accept_mask], vdata["rating"][accept_mask]
)
density_rej, _, _ = build_density_grid(
    vdata["conf"][reject_mask], vdata["rating"][reject_mask]
)
# Normalise each to % of ALL samples so heights are directly comparable
total_cells = density_all.sum()
density_acc_pct = density_acc / total_cells * 100
density_rej_pct = density_rej / total_cells * 100

from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(CC, RC, density_acc_pct, color="green", alpha=0.45,
                edgecolor="green", linewidth=0.3)
ax.plot_surface(CC, RC, density_rej_pct, color="red", alpha=0.45,
                edgecolor="red", linewidth=0.3)

# Manual legend
ax.legend(handles=[Patch(facecolor="green", alpha=0.5, label="Accept (true)"),
                   Patch(facecolor="red", alpha=0.5, label="Reject (true)")],
          fontsize=11, loc="upper left")

ax.set_xlabel("Model Confidence", fontsize=11, fontweight="bold", labelpad=10)
ax.set_ylabel("Rating Percentile", fontsize=11, fontweight="bold", labelpad=10)
ax.set_zlabel("% of Samples", fontsize=11, fontweight="bold", labelpad=8)
ax.set_title(f"{VIS_KEY}: Accept vs Reject Density",
             fontsize=14, fontweight="bold")
ax.view_init(elev=33, azim=-120)
out5 = FIGURES_DIR / "confidence_rating_3d_accept_reject_vision.png"
fig.savefig(out5, dpi=DPI, bbox_inches="tight")
print(f"Saved {out5}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 6 (Vision only): 2D density with annotated region percentages
#   Overlays the accuracy isoclines with data-% and accept/reject breakdown
#   per region defined by confidence & rating thresholds
# ---------------------------------------------------------------------------
conf_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rat_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]

fig, ax = plt.subplots(figsize=(11, 9))

# Background: smoothed density as filled contours
density_bg, cc_bg, rc_bg = build_density_grid(
    vdata["conf"], vdata["rating"], grid_res=GRID_RES, sigma=SMOOTH_SIGMA
)
density_bg_pct = density_bg / density_bg.sum() * 100
levels_d = np.linspace(0, density_bg_pct.max(), 30)
cf = ax.contourf(cc_bg, rc_bg, density_bg_pct, levels=levels_d, cmap="Blues")
cbar = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Sample Density (% of total)", fontsize=10, fontweight="bold")

# Grid lines at thresholds
for ct in conf_thresholds:
    ax.axvline(ct, color="gray", linewidth=0.8, alpha=0.5)
for rt in rat_thresholds:
    ax.axhline(rt, color="gray", linewidth=0.8, alpha=0.5)

# Annotate each rectangular region
conf_v = vdata["conf"]
rat_v = vdata["rating"]
n_total = len(conf_v)

for i in range(len(conf_thresholds) - 1):
    for j in range(len(rat_thresholds) - 1):
        c_lo, c_hi = conf_thresholds[i], conf_thresholds[i + 1]
        r_lo, r_hi = rat_thresholds[j], rat_thresholds[j + 1]

        in_region = ((conf_v >= c_lo) & (conf_v < c_hi) &
                     (rat_v >= r_lo) & (rat_v < r_hi))
        n_reg = in_region.sum()
        if n_reg == 0:
            continue

        pct_data = n_reg / n_total * 100
        n_acc = accept_mask[in_region].sum()
        n_rej = n_reg - n_acc
        acc_rate = vdata["correct"][in_region].mean() * 100

        cx = (c_lo + c_hi) / 2
        cy = (r_lo + r_hi) / 2

        ax.text(
            cx, cy,
            f"{pct_data:.1f}%\nA:{n_acc} R:{n_rej}\nacc:{acc_rate:.0f}%",
            ha="center", va="center", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85,
                      edgecolor="gray"),
        )

ax.set_xlabel("Model Confidence", fontsize=12, fontweight="bold")
ax.set_ylabel("Reviewer Rating Percentile", fontsize=12, fontweight="bold")
ax.set_title(f"{VIS_KEY}: Sample Density with Region Breakdown\n"
             f"(A=Accept, R=Reject ground truth; acc=model accuracy)",
             fontsize=13, fontweight="bold")
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0.0, 1.0)
ax.tick_params(labelsize=10)
out6 = FIGURES_DIR / "confidence_rating_density_regions_vision.png"
fig.savefig(out6, dpi=DPI, bbox_inches="tight")
print(f"Saved {out6}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 7 (Vision only): Distribution of pct_rating by year (test set)
# ---------------------------------------------------------------------------
years_unique = sorted(set(int(y) for y in vdata["year"]))
n_years = len(years_unique)
ncols = min(4, n_years)
nrows = math.ceil(n_years / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
bins = np.linspace(0, 1, 31)

for idx, yr in enumerate(years_unique):
    ax = axes[idx // ncols][idx % ncols]
    yr_mask = vdata["year"] == yr
    yr_acc = yr_mask & accept_mask
    yr_rej = yr_mask & reject_mask
    n_acc = yr_acc.sum()
    n_rej = yr_rej.sum()

    ax.hist(vdata["rating"][yr_acc], bins=bins, alpha=0.6, color="green",
            label=f"Accept ({n_acc})", edgecolor="white", linewidth=0.5)
    ax.hist(vdata["rating"][yr_rej], bins=bins, alpha=0.6, color="red",
            label=f"Reject ({n_rej})", edgecolor="white", linewidth=0.5)
    ax.set_title(f"{yr} (n={yr_mask.sum()})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=9)
    if idx % ncols == 0:
        ax.set_ylabel("Count", fontsize=10, fontweight="bold")
    if idx // ncols == nrows - 1:
        ax.set_xlabel("pct_rating", fontsize=10, fontweight="bold")

# Hide unused axes
for idx in range(n_years, nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

fig.suptitle(f"{VIS_KEY} Test Set: pct_rating Distribution by Year",
             fontsize=14, fontweight="bold")
plt.tight_layout()
out7 = FIGURES_DIR / "pct_rating_distribution_by_year_vision.png"
fig.savefig(out7, dpi=DPI, bbox_inches="tight")
print(f"Saved {out7}")
plt.close(fig)

print("Done.")
