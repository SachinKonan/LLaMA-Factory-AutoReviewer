#!/usr/bin/env python3
"""Integrate the latest completed cross-conference result files.

New inputs covered here:
  - ICLR-trained 14B text, full 8-eval cross-conference sweep.
  - ICLR-trained 3B vision, full 8-eval cross-conference sweep.
  - Arxiv-trained 7B vision LARGE balanced run, all completed checkpoints.

Outputs:
  tmp_latex_dir/figures/neurips26_iclr_scaling_crossconf.{pdf,png}
  tmp_latex_dir/figures/neurips26_large_arxiv_vision_sweep.{pdf,png}
  reports/neurips26_latest_results_update.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_neurips26_paper_figures import (  # noqa: E402
    BLUE,
    DATA,
    FIG_DIR,
    GREEN,
    INK,
    ORANGE,
    PURPLE,
    RED,
    RESULTS,
    best_tau_balanced,
    load_pairs,
    metric_stack,
    save_figure,
    setup_matplotlib,
)


RES = RESULTS / "final_sweep_v7_datasweepv3" / "optim_search_2026"
ARXIV_TRAIN_BASE = RESULTS / "final_sweep_v7_datasweepv3" / "final_data_sweep_v3" / "arxiv_train"
REPORT = ROOT / "reports" / "neurips26_latest_results_update.md"
PAPER_PLAN = ROOT / "reports" / "neurips26_paper_strengthening_plan.md"

CELL_ORDER = [
    ("iclr", "balanced", "ICLR\nbalanced"),
    ("iclr", "natural", "ICLR\nnatural"),
    ("arxiv", "balanced", "Arxiv\nbalanced"),
    ("arxiv", "natural", "Arxiv\nnatural"),
]

META = {
    ("text", "iclr", "balanced", "test"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_test/data.json",
    ("text", "iclr", "balanced", "val"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_validation/data.json",
    ("text", "iclr", "natural", "test"): DATA / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_y25up_test/data.json",
    ("text", "iclr", "natural", "val"): DATA / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_y25up_validation_DERIVED/data.json",
    ("vision", "iclr", "balanced", "test"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_y25up_test/data.json",
    ("vision", "iclr", "balanced", "val"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_y25up_validation/data.json",
    ("vision", "iclr", "natural", "test"): DATA / "iclr_2020_2023_2025_2026_30_70_original_vision_v7_filtered_y25up_test/data.json",
    ("vision", "iclr", "natural", "val"): DATA / "iclr_2020_2023_2025_2026_30_70_original_vision_v7_filtered_y25up_validation_DERIVED/data.json",
    ("text", "arxiv", "balanced", "test"): DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test/data.json",
    ("text", "arxiv", "balanced", "val"): DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation/data.json",
    ("text", "arxiv", "natural", "test"): DATA / "arxiv_natrate_21k_text_wmetadata_filtered24480_y24up_test/data.json",
    ("text", "arxiv", "natural", "val"): DATA / "arxiv_natrate_21k_text_wmetadata_filtered24480_y24up_validation/data.json",
    ("vision", "arxiv", "balanced", "test"): DATA / "arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_test/data.json",
    ("vision", "arxiv", "balanced", "val"): DATA / "arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_validation/data.json",
    ("vision", "arxiv", "natural", "test"): DATA / "arxiv_natrate_21k_vision_wmetadata_filtered24480_y24up_test/data.json",
    ("vision", "arxiv", "natural", "val"): DATA / "arxiv_natrate_21k_vision_wmetadata_filtered24480_y24up_validation/data.json",
}

ICLR_MODELS = [
    {
        "name": "ICLR 3B text",
        "short": "3B text",
        "modality": "text",
        "params": 3,
        "base": RES / "scaling/bz32_lr1e-6_text_3b/8eval",
        "ckpt": 2644,
        "color": BLUE,
    },
    {
        "name": "ICLR 3B vision",
        "short": "3B vision",
        "modality": "vision",
        "params": 3,
        "base": RES / "scaling/bz16_lr1e-6_vision_3b/8eval",
        "ckpt": 5296,
        "color": GREEN,
    },
    {
        "name": "ICLR 14B text",
        "short": "14B text",
        "modality": "text",
        "params": 14,
        "base": RES / "scaling/bz32_lr1e-6_text_14b/8eval",
        "ckpt": 1322,
        "color": RED,
    },
]

LARGE_ARXIV_MODEL = {
    "name": "Arxiv-large 7B vision balanced",
    "modality": "vision",
    "base": ARXIV_TRAIN_BASE / "large/arxiv_balanced_per_venue_vision",
    "ckpts": [4431, 8862, 13293],
}


def subdir_name(dataset: str, prior: str, split: str) -> str:
    prior_tag = "balanced" if prior == "balanced" else "natrate"
    return f"{dataset}_{prior_tag}_{split}"


def pred_path(base: Path, dataset: str, prior: str, split: str, ckpt: int) -> Path:
    sub = base / subdir_name(dataset, prior, split)
    plain = sub / f"finetuned-ckpt-{ckpt}.jsonl"
    if plain.exists():
        return plain
    gpu = sub / f"finetuned-ckpt-{ckpt}-gpu-test.jsonl"
    if gpu.exists():
        return gpu
    return plain


def eval_cell(model: dict, dataset: str, prior: str, ckpt: int | None = None) -> dict:
    modality = model["modality"]
    ckpt = model["ckpt"] if ckpt is None else ckpt
    val_pairs = load_pairs(
        pred_path(model["base"], dataset, prior, "val", ckpt),
        META[(modality, dataset, prior, "val")],
    )
    test_pairs = load_pairs(
        pred_path(model["base"], dataset, prior, "test", ckpt),
        META[(modality, dataset, prior, "test")],
    )
    tau = best_tau_balanced(val_pairs)
    raw = metric_stack(test_pairs, 0.0)
    cal = metric_stack(test_pairs, tau)
    val = metric_stack(val_pairs, tau)
    return {"tau": tau, "val": val, "raw": raw, "cal": cal}


def collect_iclr_models() -> list[dict]:
    rows = []
    for model in ICLR_MODELS:
        for dataset, prior, _ in CELL_ORDER:
            metrics = eval_cell(model, dataset, prior)
            rows.append({
                "model": model["name"],
                "short": model["short"],
                "modality": model["modality"],
                "params": model["params"],
                "ckpt": model["ckpt"],
                "dataset": dataset,
                "prior": prior,
                "tau": metrics["tau"],
                "raw": metrics["raw"],
                "cal": metrics["cal"],
                "val": metrics["val"],
            })
    return rows


def collect_large_arxiv() -> list[dict]:
    rows = []
    model = LARGE_ARXIV_MODEL
    for ckpt in model["ckpts"]:
        for dataset, prior, _ in CELL_ORDER:
            metrics = eval_cell(model, dataset, prior, ckpt=ckpt)
            rows.append({
                "model": model["name"],
                "ckpt": ckpt,
                "dataset": dataset,
                "prior": prior,
                "tau": metrics["tau"],
                "raw": metrics["raw"],
                "cal": metrics["cal"],
                "val": metrics["val"],
            })
    return rows


def make_iclr_crossconf_figure(rows: list[dict]) -> None:
    setup_matplotlib()
    models = ICLR_MODELS
    matrix = np.zeros((len(models), len(CELL_ORDER)))
    raw_matrix = np.zeros_like(matrix)
    auc_matrix = np.zeros_like(matrix)
    for i, model in enumerate(models):
        for j, (dataset, prior, _) in enumerate(CELL_ORDER):
            row = next(r for r in rows if r["model"] == model["name"] and r["dataset"] == dataset and r["prior"] == prior)
            matrix[i, j] = row["cal"].bacc
            raw_matrix[i, j] = row["raw"].bacc
            auc_matrix[i, j] = row["raw"].auc or np.nan

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, mat, title, fmt in [
        (axes[0], matrix, "Calibrated bACC (tau* from matching val)", "{:.1f}"),
        (axes[1], auc_matrix * 100, "AUC x100 (threshold-free)", "{:.1f}"),
    ]:
        im = ax.imshow(mat, cmap="RdYlGn", vmin=55, vmax=75, aspect="auto")
        ax.set_xticks(range(len(CELL_ORDER)))
        ax.set_xticklabels([c[2] for c in CELL_ORDER], fontsize=11)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m["short"] for m in models], fontsize=12)
        ax.set_title(title, fontsize=14)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        fontsize=13, fontweight="bold", color=INK)
        ax.tick_params(length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    fig.suptitle("ICLR-trained cross-conference sweeps: 14B text and 3B vision now have full 8-eval coverage",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "neurips26_iclr_scaling_crossconf")
    plt.close(fig)


def make_large_arxiv_figure(rows: list[dict]) -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharex=True)
    cell_colors = {
        ("arxiv", "balanced"): GREEN,
        ("arxiv", "natural"): ORANGE,
        ("iclr", "balanced"): BLUE,
        ("iclr", "natural"): PURPLE,
    }
    for ax, metric_name in [(axes[0], "cal"), (axes[1], "raw")]:
        for dataset, prior, label in CELL_ORDER:
            xs, ys = [], []
            for ckpt in LARGE_ARXIV_MODEL["ckpts"]:
                row = next(r for r in rows if r["ckpt"] == ckpt and r["dataset"] == dataset and r["prior"] == prior)
                xs.append(ckpt)
                ys.append(row[metric_name].bacc)
            ax.plot(xs, ys, "-o", color=cell_colors[(dataset, prior)], linewidth=2.8,
                    markersize=8, markeredgecolor="black", markeredgewidth=0.8, label=label.replace("\n", " "))
            best_idx = int(np.argmax(ys))
            ax.text(xs[best_idx], ys[best_idx] + 0.8, f"{max(ys):.1f}", ha="center",
                    fontsize=10, color=cell_colors[(dataset, prior)], fontweight="bold")
        ax.set_title("Calibrated bACC" if metric_name == "cal" else "Raw bACC at tau=0", fontsize=14)
        ax.set_xlabel("checkpoint step", fontsize=12)
        ax.set_ylim(50, 80)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("balanced accuracy (%)", fontsize=12)
    axes[1].legend(loc="lower right", fontsize=10, frameon=True)
    fig.suptitle("Arxiv-balanced LARGE 7B vision: all checkpoints across ICLR and arxiv",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "neurips26_large_arxiv_vision_sweep")
    plt.close(fig)


def fmt(v: float | None, nd: int = 1) -> str:
    if v is None:
        return "--"
    return f"{v:.{nd}f}"


def metric_row(row: dict) -> str:
    return (
        f"| {row['model']} | {row.get('ckpt', '')} | {row['dataset']} | {row['prior']} | "
        f"{row['raw'].n} | {row['raw'].bacc:.1f} | {row['cal'].bacc:.1f} | "
        f"{fmt(row['raw'].auc, 3)} | {row['cal'].acc_rec:.0f}/{row['cal'].rej_rec:.0f} | {row['tau']:+.3f} |"
    )


def summarize_best_large(rows: list[dict], dataset: str, prior: str) -> dict:
    candidates = [r for r in rows if r["dataset"] == dataset and r["prior"] == prior]
    return max(candidates, key=lambda r: r["cal"].bacc)


def write_report(iclr_rows: list[dict], large_rows: list[dict]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    best_arxiv_bal = summarize_best_large(large_rows, "arxiv", "balanced")
    best_iclr_bal = summarize_best_large(large_rows, "iclr", "balanced")
    best_arxiv_nat = summarize_best_large(large_rows, "arxiv", "natural")
    best_iclr_nat = summarize_best_large(large_rows, "iclr", "natural")

    lines = []
    lines.append("# Latest Results Update: Cross-Conference Scaling and Large Arxiv Vision\n")
    lines.append("Generated by `scripts/generate_neurips26_latest_results.py`.\n")
    lines.append("## Headline takeaways\n")
    lines.append("- **ICLR-trained 3B vision cross-eval is now complete.** It is competitive with 3B text on ICLR and stronger on arxiv-natural, so the vision story is not only a 7B artifact.")
    lines.append("- **ICLR-trained 14B text cross-eval is now complete.** Scaling text to 14B improves arxiv transfer relative to 3B text, but it still does not erase the central distribution issue: in-domain ICLR and arxiv deployment behave differently.")
    lines.append(f"- **Large arxiv-balanced 7B vision is complete through ckpt {LARGE_ARXIV_MODEL['ckpts'][-1]}.** Best calibrated arxiv-balanced bACC is **{best_arxiv_bal['cal'].bacc:.1f}** at ckpt {best_arxiv_bal['ckpt']}; best ICLR-balanced bACC is **{best_iclr_bal['cal'].bacc:.1f}** at ckpt {best_iclr_bal['ckpt']}.")
    lines.append("- **The large arxiv vision run strengthens the training-distribution story.** It is the right candidate for arxiv deployment, but the ICLR cells still need to be reported as OOD, not as a universal replacement for ICLR-trained PaperLens.\n")

    lines.append("## Figures\n")
    lines.append("![ICLR-trained cross-conference scaling](../tmp_latex_dir/figures/neurips26_iclr_scaling_crossconf.png)")
    lines.append("![Large arxiv vision sweep](../tmp_latex_dir/figures/neurips26_large_arxiv_vision_sweep.png)\n")

    lines.append("## ICLR-trained cross-conference 8-eval sweep\n")
    lines.append("Calibration: `tau*` is learned on the matching validation cell by maximizing balanced accuracy, then applied to the corresponding test cell. AUC is unchanged by thresholding.\n")
    lines.append("| model | ckpt | dataset | prior | n | bACC raw | bACC tau* | AUC | AccR/RejR tau* | tau* |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for row in iclr_rows:
        lines.append(metric_row(row))
    lines.append("")

    lines.append("## Large arxiv-balanced 7B vision checkpoint sweep\n")
    lines.append("| model | ckpt | dataset | prior | n | bACC raw | bACC tau* | AUC | AccR/RejR tau* | tau* |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for row in large_rows:
        lines.append(metric_row(row))
    lines.append("")

    lines.append("## Best large-vision operating points\n")
    lines.append("| eval cell | best ckpt | bACC tau* | bACC raw | AUC | AccR/RejR tau* |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, best in [
        ("arxiv balanced", best_arxiv_bal),
        ("arxiv natural", best_arxiv_nat),
        ("ICLR balanced", best_iclr_bal),
        ("ICLR natural", best_iclr_nat),
    ]:
        lines.append(
            f"| {label} | {best['ckpt']} | {best['cal'].bacc:.1f} | {best['raw'].bacc:.1f} | "
            f"{fmt(best['raw'].auc, 3)} | {best['cal'].acc_rec:.0f}/{best['cal'].rej_rec:.0f} |"
        )
    lines.append("")

    lines.append("## Paper integration\n")
    lines.append("- Add a model-size/cross-conference subsection with the 3B vision and 14B text rows. This lets us separate **modality** from **parameter count**.")
    lines.append("- Update the arxiv-trained distribution subsection: large arxiv-balanced vision is no longer pending, and should be treated as the main arxiv-deployment candidate if its best checkpoint survives common-subset comparisons.")
    lines.append("- Keep reporting ICLR and arxiv cells separately. The new results strengthen, rather than weaken, the claim that training distribution dominates the deployment recommendation.")

    REPORT.write_text("\n".join(lines))


def patch_paper_plan() -> None:
    if not PAPER_PLAN.exists():
        return
    text = PAPER_PLAN.read_text()
    marker = "## Missing or risky analyses to finish next\n"
    insert = (
        "## Latest completed-result update\n\n"
        "The new 8-eval sweeps for **ICLR-trained 14B text** and **ICLR-trained 3B vision**, plus the completed "
        "**large arxiv-balanced 7B vision** checkpoints, are integrated in "
        "`reports/neurips26_latest_results_update.md`. Promote these into the scaling/distribution section before "
        "editing the paper text.\n\n"
    )
    if insert.strip() not in text and marker in text:
        text = text.replace(marker, insert + marker)
    text = text.replace(
        "- Full arxiv-trained large text/vision/natural checkpoint integration once runs finish.",
        "- Large arxiv-balanced vision is now integrated; large arxiv text/natural variants still need integration once complete.",
    )
    PAPER_PLAN.write_text(text)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    iclr_rows = collect_iclr_models()
    large_rows = collect_large_arxiv()
    make_iclr_crossconf_figure(iclr_rows)
    make_large_arxiv_figure(large_rows)
    write_report(iclr_rows, large_rows)
    patch_paper_plan()
    print(f"Wrote {REPORT}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
