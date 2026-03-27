"""Horizontal bar chart comparing all methods: baselines through SFT to Bayes ceiling.

Usage:
    uv run python scripts/plot_baseline_comparison.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Data: method name, accuracy, color group
methods = [
    ("Random", 50.0, "baseline"),
    ("Zero-shot Qwen 3.5-122B", 52.1, "baseline"),
    ("TF-IDF + LogReg", 59.6, "baseline"),
    ("LogReg (187 features)", 64.0, "classical"),
    ("HistGBM (187 features)", 64.8, "classical"),
    ("Random Forest (187 features)", 65.0, "classical"),
    ("SFT Text (Trainagreeing)", 66.9, "sft"),
    ("SFT Vision (Trainagreeing)", 70.4, "sft"),
    ("Bayes-optimal ceiling", 80.2, "ceiling"),
]

colors = {
    "baseline": "#9E9E9E",
    "classical": "#FF9800",
    "sft": "#4CAF50",
    "ceiling": "#F44336",
}

fig, ax = plt.subplots(figsize=(10, 6))

names = [m[0] for m in methods]
accs = [m[1] for m in methods]
cols = [colors[m[2]] for m in methods]
alphas = [0.5 if m[2] == "ceiling" else 0.85 for m in methods]

y_pos = range(len(names))
bars = ax.barh(y_pos, accs, color=cols, edgecolor='white', linewidth=0.5)
for bar, a in zip(bars, alphas):
    bar.set_alpha(a)

# Add accuracy labels
for i, (bar, acc) in enumerate(zip(bars, accs)):
    ax.text(acc + 0.5, i, f"{acc:.1f}%", va='center', fontsize=10, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_title('Paper Acceptance Prediction: Method Comparison\n(v7 test set: ICLR 2020+2023+2025, balanced)', fontsize=12)
ax.set_xlim(45, 88)
ax.invert_yaxis()

# Add vertical line at 50% (random)
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.3, linewidth=1)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors["baseline"], alpha=0.85, label='Baselines'),
    Patch(facecolor=colors["classical"], alpha=0.85, label='Classical ML (paper stats)'),
    Patch(facecolor=colors["sft"], alpha=0.85, label='SFT (ours)'),
    Patch(facecolor=colors["ceiling"], alpha=0.5, label='Theoretical ceiling'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "baseline_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()
