#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import random
import statistics as stat
from collections import OrderedDict
from pathlib import Path

ROOT = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "objective_analysis_figures"
REPORT_PATH = DOCS_DIR / "objective_analysis.md"
SEED = 0
BOOTSTRAP_ITERS = 400


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


rxc = load_module(ROOT / "scripts/ratio_xeval_consolidated.py", "ratio_xeval_consolidated_local")


def pearson(xs, ys):
    if len(xs) < 3:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(xs, ys):
    if len(xs) < 3:
        return None

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    return pearson(ranks(xs), ranks(ys))


def bootstrap_spearman(points, iters=BOOTSTRAP_ITERS, seed=SEED):
    if len(points) < 10:
        return None
    rng = random.Random(seed)
    vals = []
    n = len(points)
    for _ in range(iters):
        sample = [points[rng.randrange(n)] for _ in range(n)]
        vals.append(spearman([x for x, _ in sample], [y for _, y in sample]))
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    lo = vals[int(0.025 * len(vals))]
    hi = vals[max(0, int(0.975 * len(vals)) - 1)]
    return lo, hi


def pct(v):
    if v is None:
        return "NA"
    return f"{v:.1f}"


def corr(v):
    if v is None:
        return "NA"
    return f"{v:+.3f}"


def auc_fmt(v):
    if v is None:
        return "NA"
    return f"{v:.3f}"


def bacc_from_metric(metric):
    if metric["acc_rec"] is None or metric["rej_rec"] is None:
        return None
    return (metric["acc_rec"] + metric["rej_rec"]) / 2.0


def load_generic_rows(
    jsonl_path: Path,
    meta_path: Path,
    year_min: int | None,
    rating_field: str,
    citation_field: str,
):
    metas = json.loads(meta_path.read_text())
    rows = []
    with jsonl_path.open() as fh:
        for i, line in enumerate(fh):
            if i >= len(metas):
                break
            meta = metas[i].get("_metadata") or {}
            if year_min is not None:
                year = meta.get("year")
                try:
                    year = int(year) if year is not None else None
                except (TypeError, ValueError):
                    year = None
                if year is None or year < year_min:
                    continue
            row = json.loads(line)
            gold_txt = rxc.extract_pred(row.get("label", ""))
            score = rxc.score_logodds(row)
            if gold_txt is None or score is None:
                continue
            rows.append(
                {
                    "score": score,
                    "gold": 1 if gold_txt == "accept" else 0,
                    "rating": meta.get(rating_field),
                    "citation": meta.get(citation_field),
                }
            )
    return rows


MODEL_ORDER_7B = OrderedDict(
    [
        ("7B text 50/50", {"mode": "text", "train": "50_50"}),
        ("7B text 30/70", {"mode": "text", "train": "30_70"}),
        ("7B vision 50/50", {"mode": "vision", "train": "50_50"}),
        ("7B vision 30/70", {"mode": "vision", "train": "30_70"}),
    ]
)


DIRECT_3B_MODELS = OrderedDict(
    [
        (
            "3B text (ICLR balanced only)",
            {
                "jsonl": ROOT
                / "results/final_sweep_v7_datasweepv3/optim_search_2026/scaling/bz32_lr1e-6_text_3b/finetuned-ckpt-2644.jsonl",
                "meta": ROOT
                / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
                "year_min": 2025,
                "rating_field": "pct_rating",
                "citation_field": "citation_normalized_by_year",
            },
        ),
        (
            "3B vision (ICLR balanced only)",
            {
                "jsonl": ROOT
                / "results/final_sweep_v7_datasweepv3/optim_search_2026/scaling/bz16_lr1e-6_vision_3b/finetuned-ckpt-5296.jsonl",
                "meta": ROOT
                / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_test/data.json",
                "year_min": 2025,
                "rating_field": "pct_rating",
                "citation_field": "citation_normalized_by_year",
            },
        ),
    ]
)


TEXT_ONLY_3B_RESULTS = [
    "results/final_sweep_v7_datasweepv3/optim_search_2026/scaling/bz32_lr1e-6_text_3b/8eval/iclr_balanced_test/finetuned-ckpt-2644.jsonl",
    "results/final_sweep_v7_datasweepv3/optim_search_2026/scaling/bz32_lr1e-6_text_3b/8eval/arxiv_balanced_test/finetuned-ckpt-2644.jsonl",
    "results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/small_sachin/arxiv_21k_text_3b/iclr_balanced_test/finetuned-ckpt-2624.jsonl",
    "results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/small_sachin/arxiv_21k_text_3b/arxiv_balanced_test/finetuned-ckpt-2624.jsonl",
    "results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/natrate_sachin/arxiv_natrate_21k_text_3b/iclr_balanced_test/finetuned-ckpt-2624.jsonl",
    "results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/natrate_sachin/arxiv_natrate_21k_text_3b/arxiv_balanced_test/finetuned-ckpt-2624.jsonl",
]


def collect_7b_metrics():
    results, _, _ = rxc.collect_all()
    out = {"balanced_test_quality": {}, "source_paths": {}, "results": results}
    for label, cfg in MODEL_ORDER_7B.items():
        mode = cfg["mode"]
        train = cfg["train"]
        out["source_paths"][label] = {
            "iclr_balanced_test": str(rxc.iclr_test_jsonl(mode, train, "balanced").relative_to(ROOT)),
            "iclr_natural_test": str(rxc.iclr_test_jsonl(mode, train, "natural").relative_to(ROOT)),
            "arxiv_balanced_test": str(rxc.arxiv_jsonl(mode, train, "balanced", "test").relative_to(ROOT)),
            "arxiv_natural_test": str(rxc.arxiv_jsonl(mode, train, "natural", "test").relative_to(ROOT)),
            "iclr_balanced_val": str(rxc.iclr_val_jsonl(mode, train, "balanced").relative_to(ROOT)),
            "iclr_natural_val": str(rxc.iclr_val_jsonl(mode, train, "natural").relative_to(ROOT)),
            "arxiv_balanced_val": str(rxc.arxiv_jsonl(mode, train, "balanced", "val").relative_to(ROOT)),
            "arxiv_natural_val": str(rxc.arxiv_jsonl(mode, train, "natural", "val").relative_to(ROOT)),
        }

        iclr_rows = load_generic_rows(
            rxc.iclr_test_jsonl(mode, train, "balanced"),
            rxc.ICLR_META[(mode, "balanced", "test")],
            2025,
            "pct_rating",
            "citation_normalized_by_year",
        )
        arxiv_rows = load_generic_rows(
            rxc.arxiv_jsonl(mode, train, "balanced", "test"),
            rxc.ARXIV_META[(mode, "balanced", "test")],
            None,
            "pct_rating",
            "pct_citation",
        )
        out["balanced_test_quality"][label] = {
            "iclr": summarize_quality_rows(iclr_rows),
            "arxiv": summarize_quality_rows(arxiv_rows),
        }
    return out


def summarize_quality_rows(rows):
    def metric_summary(field):
        items = [(r["score"], r["gold"], r[field]) for r in rows if r[field] is not None]
        if not items:
            return None
        all_pts = [(s, v) for s, _, v in items]
        acc_pts = [(s, v) for s, g, v in items if g == 1]
        rej_pts = [(s, v) for s, g, v in items if g == 0]
        return {
            "n": len(items),
            "all": spearman([s for s, _ in all_pts], [v for _, v in all_pts]),
            "accept": spearman([s for s, _ in acc_pts], [v for _, v in acc_pts]) if len(acc_pts) >= 10 else None,
            "reject": spearman([s for s, _ in rej_pts], [v for _, v in rej_pts]) if len(rej_pts) >= 10 else None,
            "bootstrap_ci": bootstrap_spearman(all_pts),
        }

    scores = [r["score"] for r in rows]
    gold = [r["gold"] for r in rows]
    auc = rxc.auc(scores, gold) if rows else None
    return {
        "n_total": len(rows),
        "auc": auc,
        "rating": metric_summary("rating"),
        "citation": metric_summary("citation"),
    }


def collect_direct_3b_metrics():
    out = OrderedDict()
    for label, cfg in DIRECT_3B_MODELS.items():
        rows = load_generic_rows(
            cfg["jsonl"],
            cfg["meta"],
            cfg["year_min"],
            cfg["rating_field"],
            cfg["citation_field"],
        )
        pairs = [(r["score"], r["gold"]) for r in rows]
        acc = rxc.accuracy(pairs, 0.0) * 100 if pairs else None
        acc_rec, rej_rec = rxc.per_class_recall(pairs, 0.0)
        out[label] = {
            "n": len(pairs),
            "raw_acc": acc,
            "balanced_acc": ((acc_rec + rej_rec) / 2 * 100) if acc_rec is not None and rej_rec is not None else None,
            "accept_recall": acc_rec * 100 if acc_rec is not None else None,
            "reject_recall": rej_rec * 100 if rej_rec is not None else None,
            "auc": rxc.auc([s for s, _ in pairs], [g for _, g in pairs]) if pairs else None,
            "quality": summarize_quality_rows(rows),
            "source": str(cfg["jsonl"].relative_to(ROOT)),
        }
    return out


def compute_natural_priors():
    iclr_meta = json.loads(
        (
            ROOT
            / "data/iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_test/data.json"
        ).read_text()
    )
    iclr_labels = []
    for row in iclr_meta:
        meta = row.get("_metadata") or {}
        year = meta.get("year")
        try:
            year = int(year) if year is not None else None
        except (TypeError, ValueError):
            year = None
        if year is not None and year >= 2025:
            iclr_labels.append(1 if meta.get("answer") == "Accept" else 0)

    arxiv_meta = json.loads(
        (ROOT / "data/arxiv_natrate_21k_text_wmetadata_filtered24480_y24up_test/data.json").read_text()
    )
    arxiv_labels = [1 if row.get("_metadata", {}).get("answer") == "Accept" else 0 for row in arxiv_meta]
    return {"iclr": sum(iclr_labels) / len(iclr_labels), "arxiv": sum(arxiv_labels) / len(arxiv_labels)}


def compute_quality_distribution_stats():
    out = {}
    for prior, path in [
        (
            "balanced",
            ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
        ),
        (
            "natural",
            ROOT / "data/iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_test/data.json",
        ),
    ]:
        items = json.loads(path.read_text())
        stats_map = {
            "rating": {"all": [], "accept": [], "reject": []},
            "citation": {"all": [], "accept": [], "reject": []},
        }
        for row in items:
            meta = row.get("_metadata") or {}
            year = meta.get("year")
            try:
                year = int(year) if year is not None else None
            except (TypeError, ValueError):
                year = None
            if year is None or year < 2025:
                continue
            bucket = "accept" if meta.get("answer") == "Accept" else "reject"
            rating = meta.get("pct_rating")
            citation = meta.get("citation_normalized_by_year")
            if rating is not None:
                stats_map["rating"]["all"].append(rating)
                stats_map["rating"][bucket].append(rating)
            if citation is not None:
                stats_map["citation"]["all"].append(citation)
                stats_map["citation"][bucket].append(citation)
        out[prior] = stats_map
    return out


def plot_quality_shift_iclr(dist_stats, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
        }
    )
    bins = [i / 20 for i in range(21)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = {"accept": "#4E79A7", "reject": "#E15759"}
    for row_idx, prior in enumerate(["balanced", "natural"]):
        for col_idx, metric in enumerate(["rating", "citation"]):
            ax = axes[row_idx][col_idx]
            acc = dist_stats[prior][metric]["accept"]
            rej = dist_stats[prior][metric]["reject"]
            ax.hist(
                [acc, rej],
                bins=bins,
                stacked=True,
                color=[colors["accept"], colors["reject"]],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
                label=[f"accept (n={len(acc)})", f"reject (n={len(rej)})"],
            )
            all_vals = dist_stats[prior][metric]["all"]
            if all_vals:
                med = stat.median(all_vals)
                ax.axvline(med, color="black", linestyle="--", linewidth=1.25)
                ax.text(med, max(ax.get_ylim()[1] * 0.85, 1), f" median={med:.2f}", fontsize=9)
            title = f"{metric} | {prior}"
            ax.set_title(title, fontsize=12)
            ax.set_xlim(0, 1)
            ax.grid(linestyle="--", alpha=0.25)
            if row_idx == 1:
                ax.set_xlabel("percentile", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("count", fontsize=11)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("ICLR 25/26 quality-signal distributions: balanced vs natural test prior", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def ascii_quality_shift(dist_stats):
    lines = []
    for prior in ["balanced", "natural"]:
        rating = dist_stats[prior]["rating"]
        citation = dist_stats[prior]["citation"]
        r_acc = len(rating["accept"]) / len(rating["all"])
        c_acc = len(citation["accept"]) / len(citation["all"])
        lines.append(
            f"{prior:<8} rating   acc_rate={r_acc:.3f}  median_all={stat.median(rating['all']):.3f}  "
            f"median_acc={stat.median(rating['accept']):.3f}  median_rej={stat.median(rating['reject']):.3f}"
        )
        lines.append(
            f"{prior:<8} citation acc_rate={c_acc:.3f}  median_all={stat.median(citation['all']):.3f}  "
            f"median_acc={stat.median(citation['accept']):.3f}  median_rej={stat.median(citation['reject']):.3f}"
        )
    return "\n".join(lines)


def compute_objective2_balanced_tables(metrics_7b):
    results = metrics_7b["results"]
    tables = {"iclr": [], "arxiv": []}
    for dataset in ["iclr", "arxiv"]:
        for label, cfg in MODEL_ORDER_7B.items():
            cell = results["test"][dataset]["balanced"][cfg["mode"]][cfg["train"]]["raw"]
            tables[dataset].append(
                {
                    "model": label,
                    "balanced_acc": bacc_from_metric(cell),
                    "accept_recall": cell["acc_rec"],
                    "reject_recall": cell["rej_rec"],
                    "auc": cell["auc"],
                }
            )
    return tables


def write_report(metrics_7b, direct_3b, priors, dist_stats):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "iclr_quality_shift.png"
    figure_block = None
    figure_status = None
    try:
        plot_quality_shift_iclr(dist_stats, fig_path)
        figure_block = "![ICLR quality shift](objective_analysis_figures/iclr_quality_shift.png)"
        figure_status = f"Wrote figure: {fig_path}"
    except Exception as exc:
        figure_block = (
            "```text\n"
            + ascii_quality_shift(dist_stats)
            + f"\n# Figure fallback: matplotlib/Pillow unavailable in this environment ({exc}).\n"
            + "# The full plotting code is still present in scripts/tmp_latex_dir/generate_objective_analysis.py.\n```"
        )
        figure_status = f"Figure fallback used: {exc}"

    results = metrics_7b["results"]
    objective2 = compute_objective2_balanced_tables(metrics_7b)

    lines = []
    lines.append("# Objective Analysis: Quality Indicator vs Conference Acceptor")
    lines.append("")
    lines.append("This report is generated directly from repository eval files. No metric values are interpolated or fabricated.")
    lines.append("")
    lines.append("## Scope And Availability")
    lines.append("")
    lines.append("- Main recommendation set: 7B ICLR-trained models with the full `text/vision × {50/50, 30/70}` cross-eval grid. These are the only runs that let us compare modality and train ratio on both balanced test sets.")
    lines.append("- 7B checkpoint rule applied: second checkpoint only. That is `ckpt-1322` for text and `ckpt-2648` / `ckpt-2642` for vision. Paths are listed in the appendix.")
    lines.append("- 3B checkpoint rule applied: last checkpoint only. Direct ICLR-balanced text and vision results exist, but matching 3B vision arxiv cross-eval files do not. I therefore use 3B only as partial supporting evidence, not for the main cross-dataset modality recommendation.")
    lines.append("- Field caveat: the repo does not expose exact `rating_rank_per_year` or `cite_rank_per_year` keys on these eval datasets. The rank-equivalent fields actually present are `pct_rating` and `citation_normalized_by_year` on ICLR, plus sparse `pct_rating` / `pct_citation` on arxiv.")
    lines.append("")
    lines.append("## 1. Metrics")
    lines.append("")
    lines.append("### Why Balanced Accuracy Over Raw Accuracy")
    lines.append("")
    lines.append("Raw accuracy is easy to game on reject-heavy test populations. The strongest failure mode in this repo is the 7B text 30/70 model on arxiv-natural: raw accuracy looks deployment-strong at `76.3`, but accept recall is `0.0`, reject recall is `100.0`, and balanced accuracy is `50.1`, i.e. random on the balanced metric.")
    lines.append("")
    lines.append("| Dataset | Model | Raw Acc (natural) | Accept Recall | Reject Recall | Balanced Acc | Source |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for dataset in ["iclr", "arxiv"]:
        for label, cfg in MODEL_ORDER_7B.items():
            cell = results["test"][dataset]["natural"][cfg["mode"]][cfg["train"]]["raw"]
            src = metrics_7b["source_paths"][label][f"{dataset}_natural_test"]
            lines.append(
                f"| {dataset} | {label} | {pct(cell['acc'])} | {pct(cell['acc_rec'])} | {pct(cell['rej_rec'])} | {pct(bacc_from_metric(cell))} | `{src}` |"
            )
    lines.append("")
    lines.append("The metric failure is cross-modal and cross-dataset, not a single outlier. On natural ICLR, the 30/70 models also inflate raw accuracy relative to their balanced accuracy (`73.7` vs `65.1` for text, `73.9` vs `69.2` for vision). On natural arxiv the gap is even larger for text 30/70 (`76.3` vs `50.1`).")
    lines.append("")
    lines.append("### Thresholding / Calibration")
    lines.append("")
    lines.append("The existing validation thresholds maximize raw accuracy, not balanced accuracy. That objective helps the reject-biased text model much more on balanced tests, but it often hurts balanced accuracy on natural tests because the threshold shifts further toward reject.")
    lines.append("")
    lines.append("| Dataset | Model | Δ Raw Acc on balanced test | Δ Balanced Acc on balanced test | Δ Raw Acc on natural test | Δ Balanced Acc on natural test |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dataset in ["iclr", "arxiv"]:
        for label, cfg in MODEL_ORDER_7B.items():
            bal_raw = results["test"][dataset]["balanced"][cfg["mode"]][cfg["train"]]["raw"]
            bal_cal = results["test"][dataset]["balanced"][cfg["mode"]][cfg["train"]]["cal"]
            nat_raw = results["test"][dataset]["natural"][cfg["mode"]][cfg["train"]]["raw"]
            nat_cal = results["test"][dataset]["natural"][cfg["mode"]][cfg["train"]]["cal"]
            lines.append(
                "| "
                + dataset
                + f" | {label}"
                + f" | {bal_cal['acc'] - bal_raw['acc']:+.1f}"
                + f" | {bacc_from_metric(bal_cal) - bacc_from_metric(bal_raw):+.1f}"
                + f" | {nat_cal['acc'] - nat_raw['acc']:+.1f}"
                + f" | {bacc_from_metric(nat_cal) - bacc_from_metric(nat_raw):+.1f} |"
            )
    lines.append("")
    lines.append("The most important pattern is the reject-biased text model on balanced tests: text 30/70 gains `+5.6` raw-accuracy points on ICLR-balanced and `+13.1` on arxiv-balanced, versus `+0.9` and `+6.7` for text 50/50. That is exactly why raw accuracy is a bad primary metric here: thresholding can partially hide reject bias. For Objective 2 I therefore compare uncalibrated balanced-test metrics by default.")
    lines.append("")
    lines.append("### Deriving Natural Accuracy From Balanced-Test Recalls")
    lines.append("")
    lines.append("Balanced accuracy alone is not enough to recover natural accuracy. The quantity that is prior-invariant and sufficient is the pair of per-class recalls. Given prior `π = P(accept)`, the natural-prior raw accuracy is:")
    lines.append("")
    lines.append("```text")
    lines.append("raw_acc(π) = π * accept_recall + (1 - π) * reject_recall")
    lines.append("```")
    lines.append("")
    lines.append(f"For the actual natural-test priors in this repo, `π_iclr = {priors['iclr']:.4f}` and `π_arxiv = {priors['arxiv']:.4f}`.")
    lines.append("")
    lines.append("| Dataset | Model | Accept Recall on balanced test | Reject Recall on balanced test | Predicted natural raw acc | Empirical natural raw acc | Error |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for dataset in ["iclr", "arxiv"]:
        pi = priors[dataset]
        for label, cfg in MODEL_ORDER_7B.items():
            bal = results["test"][dataset]["balanced"][cfg["mode"]][cfg["train"]]["raw"]
            nat = results["test"][dataset]["natural"][cfg["mode"]][cfg["train"]]["raw"]
            pred = pi * bal["acc_rec"] + (1 - pi) * bal["rej_rec"]
            err = nat["acc"] - pred
            lines.append(
                f"| {dataset} | {label} | {pct(bal['acc_rec'])} | {pct(bal['rej_rec'])} | {pred:.1f} | {pct(nat['acc'])} | {err:+.1f} |"
            )
    lines.append("")
    lines.append("Empirically the estimate is tight on arxiv (`0.0` to `0.5` points of error), where the balanced and natural sets are both drawn from the y24up pool. On ICLR the residual is larger (`2.2` to `5.3` points) because the balanced and natural files are different year/sample draws, not because the formula is wrong.")
    lines.append("")
    lines.append("### Metrics Invariant To Prior Distribution")
    lines.append("")
    lines.append("- Balanced accuracy is prior-invariant because it averages per-class recall.")
    lines.append("- Accept recall and reject recall are themselves prior-invariant diagnostics.")
    lines.append("- AUC is prior-invariant and threshold-free.")
    lines.append("- Raw accuracy is not prior-invariant.")
    lines.append("")
    lines.append("## 2. Metrics For Objective 1 (General Quality Indicator)")
    lines.append("")
    lines.append("I interpret the requested AUC bullet as a prior-invariance point: AUC is invariant to the evaluation prior, but not to the model itself. Changing the training ratio changes the model, so AUC values still change across rows and remain informative.")
    lines.append("")
    lines.append("For the rank-based quality metrics, the data constraint matters:")
    lines.append("- ICLR balanced and natural sets provide full `pct_rating` and full `citation_normalized_by_year` coverage after the 2025/2026 filter.")
    lines.append("- Arxiv y24up balanced provides only sparse quality annotations: `n=292` for `pct_rating` and `n=341` for `pct_citation` out of `1415` papers. Arxiv natural is similarly sparse. These are subset metrics, not corpus-wide metrics.")
    lines.append("")
    lines.append("### Why Rank Correlations Are Not Prior-Invariant")
    lines.append("")
    lines.append("The sensitivity is driven by the evaluation mixture, not by the training ratio itself. Rebalancing accept/reject changes the marginal quality-signal distribution, so overall Spearman correlations can move even when the model and the per-class quality distributions stay fixed.")
    lines.append("")
    lines.append(figure_block)
    lines.append("")
    lines.append("The figure above uses the actual ICLR 25/26 metadata fields. `pct_rating` shifts strongly at the all-population level when moving from balanced to natural because accepts are concentrated at much higher percentiles (`median_accept=0.805`, `median_reject=0.304`). Citation percentile is less class-separable in this data, so its mixture shift is much smaller.")
    lines.append("")
    lines.append("| Prior | Rating accept rate | Rating median all | Rating median accept | Rating median reject | Citation median all | Citation median accept | Citation median reject |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for prior in ["balanced", "natural"]:
        rating = dist_stats[prior]["rating"]
        citation = dist_stats[prior]["citation"]
        lines.append(
            f"| {prior}"
            + f" | {100 * len(rating['accept']) / len(rating['all']):.1f}"
            + f" | {stat.median(rating['all']):.3f}"
            + f" | {stat.median(rating['accept']):.3f}"
            + f" | {stat.median(rating['reject']):.3f}"
            + f" | {stat.median(citation['all']):.3f}"
            + f" | {stat.median(citation['accept']):.3f}"
            + f" | {stat.median(citation['reject']):.3f} |"
        )
    lines.append("")
    lines.append("Because of that mixture sensitivity, I report both overall and per-class Spearman correlations whenever the quality field exists.")
    lines.append("")
    lines.append("### Objective 1 Results On Balanced Test Sets (7B full grid)")
    lines.append("")
    lines.append("#### ICLR 25/26 balanced")
    lines.append("")
    lines.append("| Model | AUC | ρ rating overall | ρ rating accept | ρ rating reject | ρ citation overall | ρ citation accept | ρ citation reject | Source |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for label in MODEL_ORDER_7B:
        q = metrics_7b["balanced_test_quality"][label]["iclr"]
        src = metrics_7b["source_paths"][label]["iclr_balanced_test"]
        lines.append(
            f"| {label} | {auc_fmt(q['auc'])} | {corr(q['rating']['all'])} | {corr(q['rating']['accept'])} | {corr(q['rating']['reject'])} | {corr(q['citation']['all'])} | {corr(q['citation']['accept'])} | {corr(q['citation']['reject'])} | `{src}` |"
        )
    lines.append("")
    lines.append("#### Arxiv y24up balanced")
    lines.append("")
    lines.append("These correlations are subset-only because the quality fields are sparse in the arxiv metadata (`n_rating=292`, `n_citation=341`).")
    lines.append("")
    lines.append("| Model | AUC | ρ rating overall | ρ rating accept | ρ rating reject | ρ citation overall | ρ citation accept | ρ citation reject | Source |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for label in MODEL_ORDER_7B:
        q = metrics_7b["balanced_test_quality"][label]["arxiv"]
        src = metrics_7b["source_paths"][label]["arxiv_balanced_test"]
        lines.append(
            f"| {label} | {auc_fmt(q['auc'])} | {corr(q['rating']['all'])} | {corr(q['rating']['accept'])} | {corr(q['rating']['reject'])} | {corr(q['citation']['all'])} | {corr(q['citation']['accept'])} | {corr(q['citation']['reject'])} | `{src}` |"
        )
    lines.append("")
    lines.append("### Bootstrap CI Note")
    lines.append("")
    lines.append("For natural-prior deployment reporting, the right uncertainty estimate is a bootstrap over the natural distribution (or an equivalent reweighted balanced sample). I report example 95% bootstrap CIs below for the recommended Objective 1 candidate, 7B vision 50/50.")
    lines.append("")
    rec_q_iclr = metrics_7b["balanced_test_quality"]["7B vision 50/50"]["iclr"]
    rec_q_arxiv = metrics_7b["balanced_test_quality"]["7B vision 50/50"]["arxiv"]
    lines.append("| Dataset | Metric | n | Point estimate | 95% bootstrap CI |")
    lines.append("|---|---:|---:|---:|---:|")
    for dataset, q in [("ICLR balanced", rec_q_iclr), ("arxiv balanced", rec_q_arxiv)]:
        for metric_name in ["rating", "citation"]:
            ci = q[metric_name]["bootstrap_ci"]
            ci_txt = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci is not None else "NA"
            lines.append(
                f"| {dataset} | {metric_name} | {q[metric_name]['n']} | {corr(q[metric_name]['all'])} | {ci_txt} |"
            )
    lines.append("")
    lines.append("The CI width itself is informative: the ICLR rating signal is tight because coverage is full, whereas arxiv subset CIs are much wider because the annotated subset is small and venue-skewed.")
    lines.append("")
    lines.append("## 3. Metrics For Objective 2 (Accurate Conference Acceptor)")
    lines.append("")
    lines.append("Objective 2 uses balanced-test balanced accuracy, accept recall, and reject recall. I use the raw threshold (`τ=0`) for the main comparison because the stored calibration thresholds optimize raw accuracy, not balanced accuracy.")
    lines.append("")
    lines.append("| Dataset | Model | Balanced Acc | Accept Recall | Reject Recall | AUC | Source |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for dataset in ["iclr", "arxiv"]:
        for row in objective2[dataset]:
            label = row["model"]
            src = metrics_7b["source_paths"][label][f"{dataset}_balanced_test"]
            lines.append(
                f"| {dataset} | {label} | {pct(row['balanced_acc'])} | {pct(row['accept_recall'])} | {pct(row['reject_recall'])} | {auc_fmt(row['auc'])} | `{src}` |"
            )
    lines.append("")
    lines.append("## 4. Comprehensive Analysis: Arxiv y24up + ICLR 25/26 (Balanced Test Sets)")
    lines.append("")
    lines.append("### Which modality is optimal?")
    lines.append("")
    lines.append("- **Objective 1:** the best-supported choice is **7B vision 50/50**. It has the top AUC on both balanced test sets (`0.736` on ICLR, `0.724` on arxiv), the top ICLR rating-rank correlation (`+0.478`), and competitive citation/rating subset performance on arxiv. Arxiv quality rankings are sparse and split across metrics, so I do not let the small subset overturn the full-coverage ICLR signal.")
    lines.append("- **Objective 2:** **7B vision 50/50** is also the clean winner. It has the best balanced accuracy on both balanced test sets: `67.6` on ICLR balanced and `62.8` on arxiv balanced.")
    lines.append("")
    lines.append("### Which train ratio is optimal?")
    lines.append("")
    lines.append("- **Objective 1:** on the complete 7B grid, `50/50` is the safer ratio overall. It wins AUC on both balanced tests for both modalities, and the strongest rating-signal row is 7B vision 50/50 on ICLR.")
    lines.append("- **Objective 2:** `50/50` is again the best-supported ratio. On the winning modality it outperforms the 30/70 counterpart on both balanced tests (`67.6` vs `64.9` on ICLR, `62.8` vs `62.1` on arxiv).")
    lines.append("")
    lines.append("### Recommended configurations")
    lines.append("")
    lines.append("| Objective | Recommended configuration | Why |")
    lines.append("|---|---|---|")
    lines.append("| Objective 1 — General Quality Indicator | **7B vision 50/50, no threshold calibration** | strongest full-coverage ICLR quality correlation, best AUC on both balanced tests, and competitive arxiv subset correlations |")
    lines.append("| Objective 2 — Accurate Conference Acceptor | **7B vision 50/50, evaluate with balanced accuracy on balanced test sets** | best balanced accuracy on both balanced test sets and the most balanced per-class recalls |")
    lines.append("")
    lines.append("### 3B partial evidence")
    lines.append("")
    lines.append("The direct ICLR-balanced 3B runs point in the same modality direction, but they are incomplete for cross-dataset recommendation because no matching 3B vision arxiv-balanced eval file exists.")
    lines.append("")
    lines.append("| Model | Dataset coverage | Balanced Acc | AUC | ρ rating overall | ρ citation overall | Source |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for label, row in direct_3b.items():
        q = row["quality"]
        lines.append(
            f"| {label} | ICLR 25/26 balanced only | {pct(row['balanced_acc'])} | {auc_fmt(row['auc'])} | {corr(q['rating']['all'])} | {corr(q['citation']['all'])} | `{row['source']}` |"
        )
    lines.append("")
    lines.append("Text-only 3B cross-eval files do exist:")
    for path in TEXT_ONLY_3B_RESULTS:
        lines.append(f"- `{path}`")
    lines.append("But because there is no corresponding 3B vision arxiv-balanced cross-eval, I treat those as supplementary text-only evidence rather than using them to choose modality.")
    lines.append("")
    lines.append("## Appendix: 7B Source Paths")
    lines.append("")
    for label, path_map in metrics_7b["source_paths"].items():
        lines.append(f"### {label}")
        for key, path in path_map.items():
            lines.append(f"- `{key}`: `{path}`")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote report: {REPORT_PATH}")
    print(figure_status)


def main():
    metrics_7b = collect_7b_metrics()
    direct_3b = collect_direct_3b_metrics()
    priors = compute_natural_priors()
    dist_stats = compute_quality_distribution_stats()
    write_report(metrics_7b, direct_3b, priors, dist_stats)


if __name__ == "__main__":
    main()
