#!/usr/bin/env python3
"""
Scaling laws: accuracy / accept-recall / reject-recall vs model size.
Two panels: text (3B, 7B, 14B) and vision (3B, 7B) on the 2025+2026 test set.

All models evaluated at the same epoch-2 checkpoint:
  - Text:   ckpt-1322
  - Vision: ckpt-2648

Output: tmp_latex_dir/figures/scaling_laws.pdf / .png
"""

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style guide (matches existing calibration scripts)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})

labelsize = 20
titlesize = 20
legendsize = 16
ticksize = 16

BLUE = "#6098FF"        # text modality color
ORANGE = "#FECC81"      # vision modality color
ACC_COLOR = "#333333"   # overall accuracy
ACCR_COLOR = "#4DAF4A"  # accept recall (green)
REJR_COLOR = "#E41A1C"  # reject recall (red)

LINEWIDTH = 2.8
MARKERSIZE = 12

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_YEARS = {2025, 2026}


def extract_prediction(text: str) -> str:
    t = text.lower().strip()
    if "\\boxed{accept}" in t or "boxed{accept}" in t:
        return "accept"
    if "\\boxed{reject}" in t or "boxed{reject}" in t:
        return "reject"
    if "accept" in t:
        return "accept"
    if "reject" in t:
        return "reject"
    return "unknown"


def score(jsonl_path: Path, test_data):
    with open(jsonl_path) as f:
        preds = [json.loads(line) for line in f]
    tp = tn = fp = fn = 0
    for i, pred in enumerate(preds):
        if i >= len(test_data):
            break
        meta = test_data[i].get("_metadata", {})
        year = meta.get("year")
        if year not in EVAL_YEARS:
            continue
        p = extract_prediction(pred.get("predict", ""))
        g = extract_prediction(pred.get("label", ""))
        if p == "unknown" or g == "unknown":
            continue
        if p == "accept" and g == "accept":
            tp += 1
        elif p == "reject" and g == "reject":
            tn += 1
        elif p == "accept" and g == "reject":
            fp += 1
        elif p == "reject" and g == "accept":
            fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total * 100 if total else 0.0
    acc_r = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    rej_r = tn / (tn + fp) * 100 if (tn + fp) else 0.0
    return acc, acc_r, rej_r, total


def load_metadata(path: Path):
    with open(path) as f:
        return json.load(f)


# Dataset paths (try labelfix first, fallback to non-labelfix)
TEXT_TDS = [
    ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
    ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_v7_filtered_test/data.json",
]
VIS_TDS = [
    ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json",
    ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_v7_filtered_test/data.json",
]

# Configs: (label, params_in_billions, prediction_file, test_data_candidates)
TEXT_CONFIGS = [
    ("3B",  3.0,  ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/scaling/bz32_lr1e-6_text_3b/finetuned-ckpt-1322.jsonl", TEXT_TDS),
    ("7B",  7.0,  ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/finetuned-ckpt-1322.jsonl",             [TEXT_TDS[1], TEXT_TDS[0]]),
    ("14B", 14.0, ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/scaling/bz32_lr1e-6_text_14b/finetuned-ckpt-1322.jsonl", TEXT_TDS),
]

VIS_CONFIGS = [
    ("3B", 3.0, ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/scaling/bz16_lr1e-6_vision_3b/finetuned-ckpt-2648.jsonl", VIS_TDS),
    ("7B", 7.0, ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl",           VIS_TDS),
]


def collect(configs):
    rows = []
    for label, params, path, tds in configs:
        picked = None
        for td_path in tds:
            if not td_path.exists():
                continue
            td = load_metadata(td_path)
            acc, accr, rejr, n = score(path, td)
            if n > 0:
                picked = (acc, accr, rejr, n)
                break
        if picked is None:
            raise FileNotFoundError(f"No usable test data for {path}")
        rows.append({
            "label": label,
            "params": params,
            "acc": picked[0],
            "accr": picked[1],
            "rejr": picked[2],
            "n": picked[3],
        })
    return rows


def panel(ax, rows, modality, title):
    xs = [r["params"] for r in rows]
    labels = [r["label"] for r in rows]
    acc = [r["acc"] for r in rows]
    accr = [r["accr"] for r in rows]
    rejr = [r["rejr"] for r in rows]

    # Overall accuracy — solid
    ax.plot(xs, acc, "-o", color=ACC_COLOR, linewidth=LINEWIDTH,
            markersize=MARKERSIZE, label="Accuracy", zorder=3)
    # Accept recall — dashed
    ax.plot(xs, accr, "--s", color=ACCR_COLOR, linewidth=LINEWIDTH,
            markersize=MARKERSIZE - 2, label="Accept Recall", alpha=0.9, zorder=2)
    # Reject recall — dashed
    ax.plot(xs, rejr, "--^", color=REJR_COLOR, linewidth=LINEWIDTH,
            markersize=MARKERSIZE - 2, label="Reject Recall", alpha=0.9, zorder=2)

    # Annotate the overall accuracy points
    for x, a in zip(xs, acc):
        ax.annotate(f"{a:.1f}", (x, a), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=ticksize - 2, color=ACC_COLOR,
                    fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=ticksize)
    ax.get_xaxis().set_major_formatter(mpl.ticker.NullFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=ticksize)

    ax.set_xlabel("Model Size", fontsize=labelsize)
    ax.set_ylabel("Score (%)", fontsize=labelsize)
    ax.set_ylim(55, 80)
    ax.tick_params(axis="y", labelsize=ticksize)
    ax.set_title(title, fontsize=titlesize, pad=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=legendsize, loc="lower right", framealpha=0.95)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


def main():
    text_rows = collect(TEXT_CONFIGS)
    vis_rows = collect(VIS_CONFIGS)

    print("Text:")
    for r in text_rows:
        print(f"  {r['label']:<4} acc={r['acc']:.1f}%  accR={r['accr']:.1f}%  rejR={r['rejr']:.1f}%  n={r['n']}")
    print("Vision:")
    for r in vis_rows:
        print(f"  {r['label']:<4} acc={r['acc']:.1f}%  accR={r['accr']:.1f}%  rejR={r['rejr']:.1f}%  n={r['n']}")

    fig, (ax_text, ax_vis) = plt.subplots(1, 2, figsize=(16, 6))
    panel(ax_text, text_rows, "text", "Text Scaling (ckpt-1322)")
    panel(ax_vis, vis_rows, "vision", "Vision Scaling (ckpt-2648)")

    fig.suptitle("Model Scaling on ICLR '25+'26 Test Set",
                 fontsize=titlesize + 2, y=1.01)
    plt.tight_layout()

    out = OUTPUT_DIR / "scaling_laws"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"\nSaved: {out}.pdf / .png")


if __name__ == "__main__":
    main()
