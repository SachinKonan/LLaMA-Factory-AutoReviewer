#!/usr/bin/env python3
"""
Generate figures from report.md data tables.
Visualizes why ICLR 2026 test accuracy is consistently lower.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data from report.md ──────────────────────────────────────────────

years = [2020, 2021, 2022, 2023, 2025, 2026]
year_labels = ["2020", "2021", "2022", "2023", "2025", "2026"]

# Table 1: Per-year model performance
accuracy =       [71.7, 69.5, 66.5, 67.9, 70.9, 66.5]
accept_recall =  [69.6, 78.0, 77.1, 73.4, 69.3, 69.2]
reject_recall =  [73.9, 61.0, 56.0, 62.3, 72.6, 63.9]
n_samples =      [138,  164,  218,  308,  678,  992]

# Table 2: pct_rating by decision
accept_pct =  [0.8082, 0.8063, 0.7647, 0.7904, 0.7466, 0.7382]
reject_pct =  [0.3337, 0.3405, 0.3417, 0.2897, 0.3117, 0.3487]
pct_gap =     [0.47,   0.47,   0.42,   0.50,   0.43,   0.39]

# Table 3: AUC, Bayes optimal, model accuracy, reviewer variance
auc =              [0.951, 0.950, 0.941, 0.972, 0.939, 0.893]
bayes_optimal =    [93.5,  92.1,  91.3,  93.8,  89.1,  86.5]
model_acc =        [71.7,  69.5,  66.5,  67.9,  70.9,  66.5]
reviewer_variance = [1.94, 0.96,  1.41,  1.35,  1.35,  2.20]

# Table 4: Label cleanliness
clean_accepts = [95.7, 91.5, 87.2, 89.6, 82.6, 78.2]
clean_rejects = [60.9, 64.6, 62.4, 72.1, 67.0, 60.7]

# Colors: highlight 2026
colors = ["#4C72B0"] * 5 + ["#C44E52"]
bar_alpha = 0.85


def fig1_model_performance():
    """Per-year accuracy, accept recall, reject recall (grouped bar chart)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(years))
    w = 0.25

    bars1 = ax.bar(x - w, accuracy, w, label="Accuracy", color="#4C72B0", alpha=bar_alpha)
    bars2 = ax.bar(x, accept_recall, w, label="Accept Recall", color="#55A868", alpha=bar_alpha)
    bars3 = ax.bar(x + w, reject_recall, w, label="Reject Recall", color="#DD8452", alpha=bar_alpha)

    # Highlight 2026 bars
    for bars in [bars1, bars2, bars3]:
        bars[-1].set_edgecolor("#C44E52")
        bars[-1].set_linewidth(2.5)

    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Percentage (%)", fontsize=13)
    ax.set_title("Model Performance by Year (bz16_lr1e-6_vision, ckpt-2648)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(year_labels, fontsize=12)
    ax.set_ylim(50, 85)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Annotate sample sizes
    for i, n in enumerate(n_samples):
        ax.text(i, 52, f"n={n}", ha="center", fontsize=9, color="gray")

    # Reference lines
    ax.axhline(y=69.6, color="#4C72B0", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(5.4, 69.9, "excl. 2026\navg=69.6%", fontsize=8, color="#4C72B0", alpha=0.7, va="bottom")

    plt.tight_layout()
    out = OUT_DIR / "report_model_performance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def fig2_rating_gap():
    """Accept vs Reject pct_rating and gap by year."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(years))

    # Left: Accept and Reject pct_rating
    ax1.plot(x, accept_pct, "o-", color="#55A868", markersize=9, linewidth=2.5, label="Accept pct_rating")
    ax1.plot(x, reject_pct, "s-", color="#C44E52", markersize=9, linewidth=2.5, label="Reject pct_rating")
    ax1.fill_between(x, reject_pct, accept_pct, alpha=0.12, color="gray")
    ax1.set_xlabel("Year", fontsize=13)
    ax1.set_ylabel("pct_rating (percentile)", fontsize=13)
    ax1.set_title("Rating Separability by Decision", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(year_labels, fontsize=12)
    ax1.set_ylim(0.2, 0.9)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Annotate 2026
    ax1.annotate("2026: lowest accept\nhighest reject",
                 xy=(5, accept_pct[5]), xytext=(3.5, 0.85),
                 fontsize=9, color="#C44E52",
                 arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.5))

    # Right: Gap
    bar_colors = ["#4C72B0"] * 5 + ["#C44E52"]
    bars = ax2.bar(x, pct_gap, color=bar_colors, alpha=bar_alpha, width=0.6)
    ax2.set_xlabel("Year", fontsize=13)
    ax2.set_ylabel("pct_rating Gap (Accept - Reject)", fontsize=13)
    ax2.set_title("Rating Gap Between Decisions", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(year_labels, fontsize=12)
    ax2.set_ylim(0.3, 0.55)
    ax2.grid(axis="y", alpha=0.3)

    for i, g in enumerate(pct_gap):
        ax2.text(i, g + 0.005, f"{g:.2f}", ha="center", fontsize=10, fontweight="bold" if i == 5 else "normal")

    plt.tight_layout()
    out = OUT_DIR / "report_rating_gap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def fig3_auc_bayes():
    """AUC, Bayes optimal, model accuracy, and reviewer variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(years))

    # Left: AUC + Bayes optimal + Model accuracy
    ax1.plot(x, auc, "D-", color="#8172B2", markersize=9, linewidth=2.5, label="pct_rating AUC")
    ax1.set_xlabel("Year", fontsize=13)
    ax1.set_ylabel("AUC", fontsize=13, color="#8172B2")
    ax1.tick_params(axis="y", labelcolor="#8172B2")
    ax1.set_xticks(x)
    ax1.set_xticklabels(year_labels, fontsize=12)
    ax1.set_ylim(0.85, 1.0)
    ax1.grid(alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(x, bayes_optimal, "^-", color="#55A868", markersize=8, linewidth=2, label="Bayes Optimal (%)")
    ax1b.plot(x, model_acc, "o-", color="#4C72B0", markersize=8, linewidth=2, label="Model Acc (%)")
    ax1b.set_ylabel("Accuracy (%)", fontsize=13)
    ax1b.set_ylim(60, 100)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower left")
    ax1.set_title("Rating Predictiveness & Model Performance", fontsize=14)

    # Annotate gap for 2026
    ax1b.annotate("", xy=(5, model_acc[5]), xytext=(5, bayes_optimal[5]),
                  arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax1b.text(5.15, (model_acc[5] + bayes_optimal[5]) / 2, "20pp\ngap",
              fontsize=9, color="gray", va="center")

    # Right: Reviewer variance
    bar_colors = ["#4C72B0"] * 5 + ["#C44E52"]
    ax2.bar(x, reviewer_variance, color=bar_colors, alpha=bar_alpha, width=0.6)
    ax2.set_xlabel("Year", fontsize=13)
    ax2.set_ylabel("Reviewer Score Variance", fontsize=13)
    ax2.set_title("Reviewer Disagreement (Score Variance)", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(year_labels, fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    for i, v in enumerate(reviewer_variance):
        ax2.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=10,
                 fontweight="bold" if i == 5 else "normal")

    ax2.axhline(y=np.mean(reviewer_variance[:5]), color="gray", linestyle="--", alpha=0.5)
    ax2.text(0, np.mean(reviewer_variance[:5]) + 0.05, f"2020-2025 avg={np.mean(reviewer_variance[:5]):.2f}",
             fontsize=9, color="gray")

    plt.tight_layout()
    out = OUT_DIR / "report_auc_bayes_variance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def fig4_label_cleanliness():
    """Label cleanliness: % of accepts/rejects agreeing with ratings."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(years))
    w = 0.3

    bars1 = ax.bar(x - w/2, clean_accepts, w, label="% Accepts w/ pct >= 0.6", color="#55A868", alpha=bar_alpha)
    bars2 = ax.bar(x + w/2, clean_rejects, w, label="% Rejects w/ pct <= 0.4", color="#C44E52", alpha=bar_alpha)

    # Highlight 2026
    for bars in [bars1, bars2]:
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(2)

    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Percentage (%)", fontsize=13)
    ax.set_title("Label Cleanliness: Rating-Decision Agreement", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(year_labels, fontsize=12)
    ax.set_ylim(50, 100)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Annotate 2026 accept cleanliness
    ax.annotate(f"78.2%\n(21.8% noisy)",
                xy=(5 - w/2, clean_accepts[5]),
                xytext=(3.8, 98),
                fontsize=9, color="#C44E52", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.5))

    plt.tight_layout()
    out = OUT_DIR / "report_label_cleanliness.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def fig5_summary_dashboard():
    """All-in-one 2x2 dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(years))

    # Panel A: Model performance
    ax = axes[0, 0]
    w = 0.25
    ax.bar(x - w, accuracy, w, label="Accuracy", color="#4C72B0", alpha=bar_alpha)
    ax.bar(x, accept_recall, w, label="Accept Recall", color="#55A868", alpha=bar_alpha)
    ax.bar(x + w, reject_recall, w, label="Reject Recall", color="#DD8452", alpha=bar_alpha)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("A) Model Performance by Year", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(year_labels, fontsize=10)
    ax.set_ylim(50, 85)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for i, n in enumerate(n_samples):
        ax.text(i, 51, f"n={n}", ha="center", fontsize=7, color="gray")

    # Panel B: Rating separability
    ax = axes[0, 1]
    ax.plot(x, accept_pct, "o-", color="#55A868", markersize=8, linewidth=2, label="Accept pct_rating")
    ax.plot(x, reject_pct, "s-", color="#C44E52", markersize=8, linewidth=2, label="Reject pct_rating")
    ax.fill_between(x, reject_pct, accept_pct, alpha=0.1, color="gray")
    ax.set_ylabel("pct_rating", fontsize=11)
    ax.set_title("B) Rating Separability by Decision", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(year_labels, fontsize=10)
    ax.set_ylim(0.2, 0.9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for i, g in enumerate(pct_gap):
        mid = (accept_pct[i] + reject_pct[i]) / 2
        ax.text(i + 0.15, mid, f"{g:.2f}", fontsize=8, va="center",
                color="#C44E52" if i == 5 else "gray")

    # Panel C: AUC + Bayes + Model
    ax = axes[1, 0]
    ax.plot(x, [a * 100 for a in auc], "D-", color="#8172B2", markersize=8, linewidth=2, label="AUC × 100")
    ax.plot(x, bayes_optimal, "^-", color="#55A868", markersize=7, linewidth=2, label="Bayes Optimal (%)")
    ax.plot(x, model_acc, "o-", color="#4C72B0", markersize=7, linewidth=2, label="Model Acc (%)")
    ax.set_ylabel("Score / Accuracy (%)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_title("C) Rating AUC vs Model Performance", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(year_labels, fontsize=10)
    ax.set_ylim(60, 100)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    # Panel D: Label cleanliness
    ax = axes[1, 1]
    w = 0.3
    ax.bar(x - w/2, clean_accepts, w, label="% Accepts w/ pct >= 0.6", color="#55A868", alpha=bar_alpha)
    ax.bar(x + w/2, clean_rejects, w, label="% Rejects w/ pct <= 0.4", color="#C44E52", alpha=bar_alpha)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_title("D) Label Cleanliness", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(year_labels, fontsize=10)
    ax.set_ylim(50, 100)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Why ICLR 2026 Test Accuracy Is Consistently Lower", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "report_summary_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    fig1_model_performance()
    fig2_rating_gap()
    fig3_auc_bayes()
    fig4_label_cleanliness()
    fig5_summary_dashboard()
    print("Done.")
