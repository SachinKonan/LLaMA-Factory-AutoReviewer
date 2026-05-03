#!/usr/bin/env python3
"""Generate first-pass NeurIPS/COLM-port paper figures.

This script is intentionally self-contained: it reuses the repository's
prediction jsonl files and dataset metadata, but writes all new artifacts into
this analysis worktree. It focuses on figures that support the revised paper
story:

1. p(accept) is a continuous quality score, while the final Accept/Reject
   decision is a thresholded operating point.
2. Balanced accuracy and per-class recall are the right binary metrics under
   skewed conference priors.
3. The arxiv expansion creates a real cross-venue, temporal evaluation suite.
4. Panel images preserve most vision performance at much lower token cost.
5. Ordering and post-hoc calibration should be shown as compact diagnostics.

Outputs:
  tmp_latex_dir/figures/neurips26_threshold_recall_movement.{pdf,png}
  tmp_latex_dir/figures/neurips26_panel_mosaic.{pdf,png}
  tmp_latex_dir/figures/neurips26_panel_order_ablation.{pdf,png}
  tmp_latex_dir/figures/neurips26_platt_calibration.{pdf,png}
  reports/neurips26_paper_strengthening_plan.md
"""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "tmp_latex_dir" / "figures"
REPORT_PATH = ROOT / "reports" / "neurips26_paper_strengthening_plan.md"

RES = RESULTS / "final_sweep_v7_datasweepv3" / "optim_search_2026"
ICLR_VAL_CALIB_DIR = RESULTS / "iclr_val_calib"
ARXIV_BAL_DIR = RESULTS / "cross_conference_arxiv_y24up"
ARXIV_NATR_DIR = RESULTS / "cross_conference_arxiv_natrate_y24up"
ARXIV_TRAIN_BASE = RESULTS / "final_sweep_v7_datasweepv3" / "final_data_sweep_v3" / "arxiv_train"

DECISION_TOKEN_IDX = 5

BLUE = "#6098FF"
ORANGE = "#FECC81"
GREEN = "#77B25D"
RED = "#FF8988"
PURPLE = "#B28CFF"
GRAY = "#888888"
INK = "#202124"
LIGHT = "#F5F6F8"

MODES = ["text", "vision"]
RATIOS = ["50_50", "30_70"]
PRIORS = ["balanced", "natural"]
RATIO_LABEL = {"50_50": "50/50", "30_70": "30/70"}
MODEL_LABEL = {
    ("text", "50_50"): "Text 50/50",
    ("text", "30_70"): "Text 30/70",
    ("vision", "50_50"): "Vision 50/50",
    ("vision", "30_70"): "Vision 30/70",
}
MODEL_COLOR = {
    ("text", "50_50"): BLUE,
    ("text", "30_70"): RED,
    ("vision", "50_50"): GREEN,
    ("vision", "30_70"): PURPLE,
}

MODEL = {
    ("text", "50_50"): ("bz32_lr1e-6_text", 1322),
    ("text", "30_70"): ("bz32_lr1e-6_text_30_70", 1322),
    ("vision", "50_50"): ("bz16_lr1e-6_vision", 2648),
    ("vision", "30_70"): ("bz16_lr1e-6_vision_30_70", 2642),
}

ICLR_META = {
    ("text", "balanced", "test"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
    ("text", "balanced", "val"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation/data.json",
    ("vision", "balanced", "test"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json",
    ("vision", "balanced", "val"): DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_validation/data.json",
}

ARXIV_META = {
    ("text", "balanced", "test"): DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test/data.json",
    ("text", "balanced", "val"): DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation/data.json",
    ("text", "natural", "test"): DATA / "arxiv_natrate_21k_text_wmetadata_filtered24480_y24up_test/data.json",
    ("text", "natural", "val"): DATA / "arxiv_natrate_21k_text_wmetadata_filtered24480_y24up_validation/data.json",
    ("vision", "balanced", "test"): DATA / "arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_test/data.json",
    ("vision", "balanced", "val"): DATA / "arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_validation/data.json",
    ("vision", "natural", "test"): DATA / "arxiv_natrate_21k_vision_wmetadata_filtered24480_y24up_test/data.json",
    ("vision", "natural", "val"): DATA / "arxiv_natrate_21k_vision_wmetadata_filtered24480_y24up_validation/data.json",
}


@dataclass
class Metric:
    n: int
    acc: float
    bacc: float
    auc: float | None
    acc_rec: float
    rej_rec: float


def setup_matplotlib() -> None:
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })


def extract_pred(text: str | None) -> str | None:
    s = (text or "").lower()
    if "\\boxed{accept}" in s or "boxed{accept}" in s:
        return "accept"
    if "\\boxed{reject}" in s or "boxed{reject}" in s:
        return "reject"
    if "accept" in s:
        return "accept"
    if "reject" in s:
        return "reject"
    return None


def score_logodds(row: dict) -> float | None:
    """Signed p(accept) log-odds. Positive means the model leans Accept."""
    pred = row.get("predict", "")
    chosen = 1 if "Accept" in pred else (0 if "Reject" in pred else None)
    token_logprobs = row.get("token_logprobs") or []
    if chosen is not None and len(token_logprobs) > DECISION_TOKEN_IDX and token_logprobs[DECISION_TOKEN_IDX] is not None:
        p_chosen = math.exp(token_logprobs[DECISION_TOKEN_IDX])
        p_chosen = min(max(p_chosen, 1e-7), 1 - 1e-7)
        logit = math.log(p_chosen / (1 - p_chosen))
        return logit if chosen == 1 else -logit

    la = row.get("logprob_accept")
    lr = row.get("logprob_reject")
    if la and lr and len(la) == len(lr):
        diffs = [abs(a - b) if a is not None and b is not None else -1 for a, b in zip(la, lr)]
        pos = int(np.argmax(diffs))
        if la[pos] is not None and lr[pos] is not None:
            return float(la[pos] - lr[pos])
    if chosen is None:
        return None
    return 6.0 if chosen == 1 else -6.0


def score_logodds_2class(row: dict) -> float | None:
    """Use explicit Accept/Reject logprobs when source reports provide them."""
    la = row.get("logprob_accept")
    lr = row.get("logprob_reject")
    if la and lr and len(la) == len(lr):
        diffs = [abs(a - b) if a is not None and b is not None else -1 for a, b in zip(la, lr)]
        pos = int(np.argmax(diffs))
        if la[pos] is not None and lr[pos] is not None:
            return float(la[pos] - lr[pos])
    return score_logodds(row)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-x))


def auc_score(scores: list[float], labels: list[int]) -> float | None:
    if not scores or len(set(labels)) < 2:
        return None
    n = len(scores)
    npos = sum(labels)
    nneg = n - npos
    indexed = sorted(enumerate(scores), key=lambda p: p[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    rank_sum = sum(r for r, label in zip(ranks, labels) if label == 1)
    return (rank_sum - npos * (npos + 1) / 2) / (npos * nneg)


def best_tau_balanced(pairs: list[tuple[float, int]]) -> float:
    if not pairs:
        return 0.0
    ordered = sorted(pairs)
    npos = sum(g for _, g in ordered)
    nneg = len(ordered) - npos
    if npos == 0 or nneg == 0:
        return 0.0
    tp = npos
    tn = 0
    best_bacc = (tp / npos + tn / nneg) / 2
    best_tau = -math.inf
    for score, group in groupby(ordered, key=lambda p: p[0]):
        for _, gold in group:
            if gold == 1:
                tp -= 1
            else:
                tn += 1
        bacc = (tp / npos + tn / nneg) / 2
        if bacc > best_bacc:
            best_bacc = bacc
            best_tau = score
    return best_tau if best_tau != -math.inf else -1e6


def metric_stack(pairs: list[tuple[float, int]], tau: float = 0.0) -> Metric:
    n = len(pairs)
    n_acc = sum(1 for _, g in pairs if g == 1)
    n_rej = n - n_acc
    correct = sum(1 for score, gold in pairs if (score > tau) == (gold == 1))
    tp = sum(1 for score, gold in pairs if gold == 1 and score > tau)
    tn = sum(1 for score, gold in pairs if gold == 0 and score <= tau)
    acc_rec = 100 * tp / n_acc if n_acc else 0.0
    rej_rec = 100 * tn / n_rej if n_rej else 0.0
    auc = auc_score([s for s, _ in pairs], [g for _, g in pairs])
    return Metric(
        n=n,
        acc=100 * correct / n if n else 0.0,
        bacc=(acc_rec + rej_rec) / 2,
        auc=auc,
        acc_rec=acc_rec,
        rej_rec=rej_rec,
    )


def iclr_test_jsonl(mode: str, train_ratio: str) -> Path:
    short, step = MODEL[(mode, train_ratio)]
    if train_ratio == "50_50":
        return RES / short / f"finetuned-ckpt-{step}.jsonl"
    return RES / "ratio_sweep" / short / "balanced" / f"finetuned-ckpt-{step}.jsonl"


def arxiv_jsonl(mode: str, train_ratio: str, prior: str, split: str) -> Path:
    short, step = MODEL[(mode, train_ratio)]
    base = ARXIV_BAL_DIR if prior == "balanced" else ARXIV_NATR_DIR
    subdir = "arxiv_eval" if split == "test" else "arxiv_val"
    return base / short / subdir / f"finetuned-ckpt-{step}.jsonl"


def load_pairs(
    pred_path: Path,
    meta_path: Path | None = None,
    *,
    year_key: str | None = None,
    year_min: int | None = None,
    prefer_2class: bool = False,
) -> list[tuple[float, int]]:
    metas = None
    if meta_path is not None:
        metas = json.loads(meta_path.read_text())
    pairs: list[tuple[float, int]] = []
    with pred_path.open() as f:
        for i, line in enumerate(f):
            if metas is not None:
                if i >= len(metas):
                    break
                if year_key and year_min is not None:
                    meta = metas[i].get("_metadata") or {}
                    raw_year = meta.get(year_key)
                    try:
                        year = int(raw_year) if raw_year is not None else None
                    except (TypeError, ValueError):
                        year = None
                    if year is None or year < year_min:
                        continue
            row = json.loads(line)
            gold_s = extract_pred(row.get("label", ""))
            score = score_logodds_2class(row) if prefer_2class else score_logodds(row)
            if gold_s is None or score is None:
                continue
            pairs.append((score, 1 if gold_s == "accept" else 0))
    return pairs


def load_arxiv_cell(mode: str, train_ratio: str, prior: str, split: str) -> list[tuple[float, int]]:
    return load_pairs(arxiv_jsonl(mode, train_ratio, prior, split), ARXIV_META[(mode, prior, split)])


def balanced_val_thresholds_for_arxiv() -> dict[tuple[str, str], float]:
    """Learn one global tau per model on the arxiv balanced validation split."""
    thresholds: dict[tuple[str, str], float] = {}
    for mode in MODES:
        for ratio in RATIOS:
            val_pairs = load_arxiv_cell(mode, ratio, "balanced", "val")
            thresholds[(mode, ratio)] = best_tau_balanced(val_pairs)
    return thresholds


def make_threshold_recall_movement() -> list[dict]:
    setup_matplotlib()
    thresholds = balanced_val_thresholds_for_arxiv()
    rows: list[dict] = []

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharex=True, sharey=True)
    label_offsets = {
        ("text", "50_50"): (1.0, 2.4),
        ("text", "30_70"): (1.0, -3.4),
        ("vision", "50_50"): (1.0, -2.1),
        ("vision", "30_70"): (1.0, 1.1),
    }
    for ax, prior in zip(axes, PRIORS):
        ax.set_title(f"Arxiv {prior} test", fontsize=18)
        ax.plot([0, 100], [0, 100], color=GRAY, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.scatter([100], [100], marker="*", s=180, color=INK, zorder=4)
        ax.text(95, 100.7, "ideal", fontsize=11, color=INK)
        for mode in MODES:
            for ratio in RATIOS:
                key = (mode, ratio)
                pairs = load_arxiv_cell(mode, ratio, prior, "test")
                raw = metric_stack(pairs, tau=0.0)
                cal = metric_stack(pairs, tau=thresholds[key])
                color = MODEL_COLOR[key]
                marker = "o" if mode == "vision" else "s"
                alpha = 1.0 if ratio == "30_70" else 0.68
                ax.scatter(raw.acc_rec, raw.rej_rec, marker=marker, s=100, color="white",
                           edgecolor=color, linewidth=2.2, alpha=alpha, zorder=3)
                ax.annotate(
                    "",
                    xy=(cal.acc_rec, cal.rej_rec),
                    xytext=(raw.acc_rec, raw.rej_rec),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.6, alpha=alpha),
                )
                ax.scatter(cal.acc_rec, cal.rej_rec, marker=marker, s=130, color=color,
                           edgecolor="black", linewidth=0.8, alpha=alpha, zorder=4)
                dx, dy = label_offsets[key]
                ax.text(cal.acc_rec + dx, cal.rej_rec + dy, MODEL_LABEL[key],
                        fontsize=10.5, color=color, fontweight="bold")
                rows.append({
                    "prior": prior,
                    "model": MODEL_LABEL[key],
                    "tau_bal_val": thresholds[key],
                    "raw_bacc": raw.bacc,
                    "cal_bacc": cal.bacc,
                    "raw_acc_rec": raw.acc_rec,
                    "raw_rej_rec": raw.rej_rec,
                    "cal_acc_rec": cal.acc_rec,
                    "cal_rej_rec": cal.rej_rec,
                    "auc": raw.auc,
                    "n": raw.n,
                })
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 102)
        ax.set_xlabel("Accept recall (%)", fontsize=15)
        ax.grid(True, linestyle="--", alpha=0.32)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Reject recall (%)", fontsize=15)
    handles = [
        plt.Line2D([0], [0], marker="s", color="white", label="Text", markerfacecolor="white",
                   markeredgecolor=BLUE, markeredgewidth=2, markersize=9),
        plt.Line2D([0], [0], marker="o", color="white", label="Vision", markerfacecolor="white",
                   markeredgecolor=GREEN, markeredgewidth=2, markersize=9),
        plt.Line2D([0], [0], color=INK, lw=2.4, label="raw tau=0 -> balanced-val tau*"),
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=11, frameon=True)
    fig.suptitle("Balanced-validation thresholding moves the binary operating point, not the score ordering",
                 fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "neurips26_threshold_recall_movement")
    plt.close(fig)
    return rows


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def ascii_clean(text: str) -> str:
    return text.encode("ascii", "ignore").decode("ascii")


def answer_of(entry: dict) -> str:
    meta = entry.get("_metadata") or {}
    raw = str(meta.get("answer") or meta.get("training_label") or meta.get("decision") or "").lower()
    if "accept" in raw or "poster" in raw or "oral" in raw or "spotlight" in raw:
        return "accept"
    return "reject"


def image_path_for(entry: dict) -> Path | None:
    images = entry.get("images") or []
    if not images:
        return None
    path = ROOT / images[0]
    return path if path.exists() else None


def find_panel_entry(entries: list[dict], label: str, predicate, used: set[str]) -> dict:
    for entry in entries:
        if answer_of(entry) != label:
            continue
        meta = entry.get("_metadata") or {}
        ident = str(meta.get("submission_id") or meta.get("arxiv_id") or id(entry))
        if ident in used:
            continue
        if predicate(entry) and image_path_for(entry) is not None:
            used.add(ident)
            return entry
    for entry in entries:
        if answer_of(entry) != label:
            continue
        meta = entry.get("_metadata") or {}
        ident = str(meta.get("submission_id") or meta.get("arxiv_id") or id(entry))
        if ident in used:
            continue
        if image_path_for(entry) is not None:
            used.add(ident)
            return entry
    raise RuntimeError(f"No panel entry found for label={label}")


def entry_caption(entry: dict, source: str) -> str:
    meta = entry.get("_metadata") or {}
    if source == "iclr":
        sid = str(meta.get("submission_id", ""))[:7]
        year = meta.get("year", "?")
        pct = meta.get("pct_rating")
        pct_s = f"pct={pct:.2f}" if isinstance(pct, (float, int)) else "pct=na"
        return f"ICLR {year} | {pct_s} | {sid}"
    venue = str(meta.get("pl_venue") or meta.get("venue") or "?").upper()
    year = meta.get("conference_year") or meta.get("pl_year") or "?"
    arxiv_id = str(meta.get("arxiv_id", ""))[:10]
    cats = ascii_clean(str(meta.get("categories") or "")).split()
    cat = cats[0] if cats else "cat=na"
    return f"{venue} {year} | {cat} | {arxiv_id}"


def paste_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill: str, width: int) -> None:
    chars_per_line = max(18, int(width / (font.size * 0.53)))
    wrapped = textwrap.wrap(ascii_clean(text), width=chars_per_line, max_lines=2, placeholder="...")
    y = xy[1]
    for line in wrapped:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = xy[0] + (width - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + 4


def make_panel_mosaic() -> None:
    iclr_entries = json.loads((DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_panel_test/data.json").read_text())
    arxiv_entries = json.loads((DATA / "arxiv_50_50_21k_vision_wmetadata_filtered24480_panel_test/data.json").read_text())

    rows = [
        ("ML 2025", 2025, {"iclr", "icml", "neurips"}),
        ("CV 2026", 2026, {"cvpr", "iccv", "eccv"}),
        ("NLP 2025", 2025, {"acl", "emnlp", "naacl", "colm"}),
        ("General 2026", 2026, {"aaai", "aistats", "corl"}),
    ]
    columns = [("ICLR Accept", "iclr", "accept"), ("ICLR Reject", "iclr", "reject"),
               ("Arxiv Accept", "arxiv", "accept"), ("Arxiv Reject", "arxiv", "reject")]
    used: set[str] = set()
    selected: list[list[tuple[dict, str]]] = []
    for _, iclr_year, arxiv_venues in rows:
        row_cells: list[tuple[dict, str]] = []
        for _, source, label in columns:
            if source == "iclr":
                entry = find_panel_entry(
                    iclr_entries,
                    label,
                    lambda e, year=iclr_year: int((e.get("_metadata") or {}).get("year", 0)) == year,
                    used,
                )
            else:
                entry = find_panel_entry(
                    arxiv_entries,
                    label,
                    lambda e, venues=arxiv_venues: str((e.get("_metadata") or {}).get("pl_venue")
                                                       or (e.get("_metadata") or {}).get("venue")
                                                       or "").lower() in venues,
                    used,
                )
            row_cells.append((entry, source))
        selected.append(row_cells)

    cell_w, cell_h = 690, 470
    caption_h, header_h = 70, 88
    row_label_w, margin, gap = 150, 52, 24
    width = margin * 2 + row_label_w + gap + 4 * cell_w + 3 * gap
    height = margin * 2 + header_h + 4 * (cell_h + caption_h) + 3 * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = _font(38, bold=True)
    head_font = _font(29, bold=True)
    small_font = _font(19)
    row_font = _font(26, bold=True)

    draw.text((margin, 18), "Panelized paper examples: ICLR and arxiv accept/reject splits",
              font=title_font, fill=INK)
    draw.text((margin, 56), "Each thumbnail is the single 5x2 page-panel input used by the panel representation.",
              font=small_font, fill="#5F6368")

    x0 = margin + row_label_w + gap
    y0 = margin + header_h
    for col_idx, (col_name, _, label) in enumerate(columns):
        x = x0 + col_idx * (cell_w + gap)
        color = GREEN if label == "accept" else RED
        draw.rounded_rectangle((x, y0 - 48, x + cell_w, y0 - 10), radius=10, fill=color, outline=color)
        bbox = draw.textbbox((0, 0), col_name, font=head_font)
        draw.text((x + (cell_w - (bbox[2] - bbox[0])) // 2, y0 - 43), col_name, font=head_font, fill="white")

    for row_idx, (row_label, _, _) in enumerate(rows):
        y = y0 + row_idx * (cell_h + caption_h + gap)
        draw.text((margin, y + cell_h // 2 - 14), row_label, font=row_font, fill=INK)
        for col_idx, ((entry, source), (_, _, label)) in enumerate(zip(selected[row_idx], columns)):
            x = x0 + col_idx * (cell_w + gap)
            border = GREEN if label == "accept" else RED
            draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=8,
                                   fill=LIGHT, outline=border, width=5)
            panel_path = image_path_for(entry)
            assert panel_path is not None
            with Image.open(panel_path) as img:
                img = img.convert("RGB")
                img.thumbnail((cell_w - 24, cell_h - 24), Image.LANCZOS)
                paste_x = x + (cell_w - img.width) // 2
                paste_y = y + (cell_h - img.height) // 2
                canvas.paste(img, (paste_x, paste_y))
            caption = entry_caption(entry, source)
            paste_wrapped(draw, (x, y + cell_h + 11), caption, small_font, INK, cell_w)

    out_png = FIG_DIR / "neurips26_panel_mosaic.png"
    out_pdf = FIG_DIR / "neurips26_panel_mosaic.pdf"
    canvas.save(out_png, "PNG", optimize=True)
    canvas.save(out_pdf, "PDF", resolution=200)


def load_year_filtered_metric(pred: Path, meta: Path, year_key: str = "year", year_min: int = 2025) -> Metric:
    return metric_stack(load_pairs(pred, meta, year_key=year_key, year_min=year_min), tau=0.0)


def make_panel_order_ablation() -> list[dict]:
    setup_matplotlib()
    rows: list[dict] = []

    ablations = [
        ("8-page vision", "panel", 9920,
         RES / "bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl",
         DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json"),
        ("Panel-only FT", "panel", 4590,
         RES / "bz16_lr1e-6_panel/finetuned-ckpt-5296.jsonl",
         DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_panel_test/data.json"),
        ("Text forward", "order", None,
         RES / "bz32_lr1e-6_text/finetuned-ckpt-1322.jsonl",
         DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json"),
        ("Text reversed", "order", None,
         RES / "bz32_lr1e-6_text_reversed/finetuned-ckpt-1322.jsonl",
         DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test_reversed/data.json"),
        ("Vision forward", "order", None,
         RES / "bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl",
         DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json"),
        ("Vision reversed", "order", None,
         RES / "bz16_lr1e-6_vision_reversed/finetuned-ckpt-2648.jsonl",
         DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test_reversed/data.json"),
    ]
    for name, family, tokens, pred, meta in ablations:
        metric = load_year_filtered_metric(pred, meta)
        rows.append({
            "name": name,
            "family": family,
            "tokens": tokens,
            "n": metric.n,
            "bacc": metric.bacc,
            "acc": metric.acc,
            "acc_rec": metric.acc_rec,
            "rej_rec": metric.rej_rec,
            "auc": metric.auc,
        })

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    panel_rows = [r for r in rows if r["family"] == "panel"]
    ax = axes[0]
    for row, color in zip(panel_rows, [GREEN, ORANGE]):
        ax.scatter(row["tokens"], row["bacc"], s=260, color=color, edgecolor="black", linewidth=1.3, zorder=3)
        ax.text(row["tokens"] + 170, row["bacc"], f"{row['name']}\n{row['bacc']:.1f} bACC",
                fontsize=12, color=INK, va="center")
    ax.plot([panel_rows[0]["tokens"], panel_rows[1]["tokens"]],
            [panel_rows[0]["bacc"], panel_rows[1]["bacc"]],
            color=GRAY, linewidth=2, linestyle="--", zorder=1)
    ax.set_xlabel("Approx. vision tokens / paper", fontsize=14)
    ax.set_ylabel("ICLR 25/26 balanced accuracy (%)", fontsize=14)
    ax.set_title("Panel representation: token efficiency", fontsize=17)
    ax.set_xlim(3500, 10800)
    ymin = min(r["bacc"] for r in panel_rows) - 3
    ymax = max(r["bacc"] for r in panel_rows) + 3
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    order_rows = [r for r in rows if r["family"] == "order"]
    labels = [r["name"].replace(" ", "\n") for r in order_rows]
    colors = [BLUE, ORANGE, GREEN, PURPLE]
    x = np.arange(len(order_rows))
    bars = ax.bar(x, [r["bacc"] for r in order_rows], color=colors, edgecolor="black", linewidth=1.1)
    for bar, row in zip(bars, order_rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7,
                f"{row['bacc']:.1f}\n{row['acc_rec']:.0f}/{row['rej_rec']:.0f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("ICLR 25/26 balanced accuracy (%)", fontsize=14)
    ax.set_title("Ordering ablation: matched train/test order", fontsize=17)
    ax.set_ylim(45, max(r["bacc"] for r in order_rows) + 8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Representation ablations to keep in the paper", fontsize=20, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "neurips26_panel_order_ablation")
    plt.close(fig)
    return rows


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.optimize import minimize
    except Exception:
        return 1.0, 0.0

    def nll(params: np.ndarray) -> float:
        a, b = params
        probs = np.clip(sigmoid(a * scores + b), 1e-8, 1 - 1e-8)
        return float(-(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean())

    result = minimize(nll, np.array([1.0, 0.0]), method="BFGS")
    if not result.success:
        return 1.0, 0.0
    a, b = result.x
    return float(a), float(b)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    edges = np.linspace(0, 1, n_bins + 1)
    centers, empirical, counts = [], [], []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi < 1.0:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        p_mean = float(probs[mask].mean())
        y_mean = float(labels[mask].mean())
        n = int(mask.sum())
        centers.append(p_mean)
        empirical.append(y_mean)
        counts.append(n)
        ece += abs(p_mean - y_mean) * n / len(probs)
    return np.array(centers), np.array(empirical), np.array(counts), float(ece)


def load_scores_labels(
    pred: Path,
    meta: Path | None = None,
    *,
    year_key: str | None = None,
    year_min: int | None = None,
    prefer_2class: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = load_pairs(pred, meta, year_key=year_key, year_min=year_min, prefer_2class=prefer_2class)
    return np.array([s for s, _ in pairs], dtype=float), np.array([g for _, g in pairs], dtype=float)


def make_platt_calibration() -> list[dict]:
    setup_matplotlib()
    cases = [
        {
            "title": "ICLR-trained vision -> ICLR",
            "val_pred": RES / "bz16_lr1e-6_vision/validation-ckpt-2648.jsonl",
            "val_meta": ICLR_META[("vision", "balanced", "val")],
            "test_pred": RES / "bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl",
            "test_meta": ICLR_META[("vision", "balanced", "test")],
            "year_key": "year",
            "year_min": 2025,
            "prefer_2class": False,
        },
        {
            "title": "ICLR-trained vision -> arxiv",
            "val_pred": ARXIV_BAL_DIR / "bz16_lr1e-6_vision/arxiv_val/finetuned-ckpt-2648.jsonl",
            "val_meta": ARXIV_META[("vision", "balanced", "val")],
            "test_pred": ARXIV_BAL_DIR / "bz16_lr1e-6_vision/arxiv_eval/finetuned-ckpt-2648.jsonl",
            "test_meta": ARXIV_META[("vision", "balanced", "test")],
            "year_key": None,
            "year_min": None,
            "prefer_2class": False,
        },
        {
            "title": "Arxiv-trained vision -> arxiv",
            "val_pred": ARXIV_TRAIN_BASE / "small/arxiv_21k_vision/arxiv_balanced_val/finetuned-ckpt-2618-gpu-test.jsonl",
            "val_meta": None,
            "test_pred": ARXIV_TRAIN_BASE / "small/arxiv_21k_vision/arxiv_balanced_test/finetuned-ckpt-2618-gpu-test.jsonl",
            "test_meta": None,
            "year_key": None,
            "year_min": None,
            "prefer_2class": True,
        },
        {
            "title": "Arxiv-trained text -> arxiv",
            "val_pred": ARXIV_TRAIN_BASE / "small/arxiv_21k_text/arxiv_balanced_val/finetuned-ckpt-2624.jsonl",
            "val_meta": None,
            "test_pred": ARXIV_TRAIN_BASE / "small/arxiv_21k_text/arxiv_balanced_test/finetuned-ckpt-2624.jsonl",
            "test_meta": None,
            "year_key": None,
            "year_min": None,
            "prefer_2class": True,
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11), sharex=True, sharey=True)
    rows: list[dict] = []
    for ax, case in zip(axes.ravel(), cases):
        val_scores, val_labels = load_scores_labels(
            case["val_pred"], case["val_meta"], year_key=case["year_key"], year_min=case["year_min"],
            prefer_2class=case["prefer_2class"]
        )
        test_scores, test_labels = load_scores_labels(
            case["test_pred"], case["test_meta"], year_key=case["year_key"], year_min=case["year_min"],
            prefer_2class=case["prefer_2class"]
        )
        raw_probs = sigmoid(test_scores)
        a, b = fit_platt(val_scores, val_labels)
        cal_probs = sigmoid(a * test_scores + b)

        for probs, label, color, marker in [
            (raw_probs, "raw", GRAY, "o"),
            (cal_probs, "Platt", GREEN, "s"),
        ]:
            x, y, counts, ece = reliability_bins(probs, test_labels)
            size = 35 + np.sqrt(counts) * 12
            ax.scatter(x, y, s=size, color=color, marker=marker, edgecolor="black", linewidth=0.7,
                       alpha=0.84, label=f"{label} ECE={ece:.3f}")
            if label == "Platt":
                rows.append({
                    "case": case["title"],
                    "n_val": len(val_labels),
                    "n_test": len(test_labels),
                    "platt_a": a,
                    "platt_b": b,
                    "raw_ece": reliability_bins(raw_probs, test_labels)[3],
                    "platt_ece": ece,
                    "test_bacc": metric_stack(list(zip(test_scores.tolist(), test_labels.astype(int).tolist()))).bacc,
                    "auc": auc_score(test_scores.tolist(), test_labels.astype(int).tolist()),
                })
        ax.plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=1.2)
        ax.set_title(case["title"], fontsize=15)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper left", fontsize=10, frameon=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Mean predicted p(accept)", fontsize=13)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed accept rate", fontsize=13)
    fig.suptitle("Post-hoc Platt calibration: p(accept) should be an interpretable score",
                 fontsize=19, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "neurips26_platt_calibration")
    plt.close(fig)
    return rows


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=220)


def fmt_auc(v: float | None) -> str:
    return "--" if v is None else f"{v:.3f}"


def write_report(recall_rows: list[dict], ablation_rows: list[dict], calibration_rows: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# NeurIPS 2026 Paper Strengthening Plan and First Figure Pack\n")
    lines.append("Generated by `scripts/generate_neurips26_paper_figures.py`.\n")
    lines.append("## Paper focus\n")
    lines.append("The paper should move from a narrow \"vision beats text on ICLR\" story to a two-objective story:\n")
    lines.append("1. **General paper-quality indicator:** use the model's continuous `p(accept)` score, evaluated by AUC, Spearman correlation to reviewer percentile/rating, and citation correlation where citation has had time to mature.")
    lines.append("2. **Conference accept predictor:** use a binary operating point on that score, evaluated by balanced accuracy and accept/reject recall. Raw accuracy is only meaningful after specifying the deployment prior.\n")
    lines.append("This lets the intro define `p(accept)` early: it is not just a class label, it is the paper-quality score that later gets thresholded for a binary decision.\n")
    lines.append("## First-pass figures generated\n")
    lines.append("| figure | intended paper role |")
    lines.append("|---|---|")
    lines.append("| `tmp_latex_dir/figures/neurips26_threshold_recall_movement.png` | Show why balanced accuracy and recall movement matter: balanced-val thresholding changes the Accept/Reject operating point, while AUC/rank metrics are unchanged. |")
    lines.append("| `tmp_latex_dir/figures/neurips26_panel_mosaic.png` | Dataset/representation teaser: ICLR and arxiv accept/reject examples as 5x2 panels. This can replace the current abstract mock. |")
    lines.append("| `tmp_latex_dir/figures/neurips26_panel_order_ablation.png` | Compact representation ablation: panel efficiency and ordering sensitivity. |")
    lines.append("| `tmp_latex_dir/figures/neurips26_platt_calibration.png` | Post-hoc calibration: the y=x reliability view for `p(accept)` after Platt scaling. |\n")
    lines.append("## Existing temporality figures to promote\n")
    lines.append("The ratio analysis already has the temporal story that should move into the paper rather than staying buried in `reports/ratio_xeval_7b.md`:\n")
    lines.append("| existing figure | paper role |")
    lines.append("|---|---|")
    lines.append("| `tmp_latex_dir/figures/ratio_xeval_year_trajectory.png` | Per-venue/year drift across arxiv y24up; use as the broad temporal diagnostic. |")
    lines.append("| `tmp_latex_dir/figures/ratio_xeval_family_year_trajectory.png` | Family-level trend view; cleaner than per-venue spaghetti when space is tight. |")
    lines.append("| `tmp_latex_dir/figures/ratio_xeval_family_vision_minus_text_2026.png` | Main temporality takeaway: vision is more robust under 2026 balanced-prior shift, especially CV, while natural-prior deployment still favors text today. |\n")
    lines.append("Use one Vero-style takeaway box beside these plots: **balanced-prior temporal shift exposes model robustness; natural-prior accuracy mostly reflects the deployment base rate.**\n")

    lines.append("## Arxiv recall movement from balanced-val thresholding\n")
    lines.append("Thresholds below are learned on **arxiv balanced validation** by maximizing balanced accuracy, then applied to arxiv balanced and natural test. This changes binary recall/accuracy, not AUC or score correlations.\n")
    lines.append("| test prior | model | tau* | n | bACC raw | bACC tau* | AccR raw -> tau* | RejR raw -> tau* | AUC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in recall_rows:
        lines.append(
            f"| {row['prior']} | {row['model']} | {row['tau_bal_val']:+.3f} | {row['n']} | "
            f"{row['raw_bacc']:.1f} | {row['cal_bacc']:.1f} | "
            f"{row['raw_acc_rec']:.0f} -> {row['cal_acc_rec']:.0f} | "
            f"{row['raw_rej_rec']:.0f} -> {row['cal_rej_rec']:.0f} | {fmt_auc(row['auc'])} |"
        )
    lines.append("")
    lines.append("Takeaway to test in prose: the natural-rate text model can look good under raw natural accuracy, but its raw accept recall collapses. Balanced validation exposes and corrects the operating point; it cannot repair AUC/ranking because the underlying score order is unchanged.\n")

    lines.append("## Panel and ordering ablations\n")
    lines.append("| ablation | family | n | approx tokens | bACC | AccR/RejR | AUC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in ablation_rows:
        tok = "--" if row["tokens"] is None else str(row["tokens"])
        lines.append(
            f"| {row['name']} | {row['family']} | {row['n']} | {tok} | "
            f"{row['bacc']:.1f} | {row['acc_rec']:.0f}/{row['rej_rec']:.0f} | {fmt_auc(row['auc'])} |"
        )
    lines.append("")
    lines.append("Panel prose should emphasize tokens-per-point rather than claiming panels are strictly better. Ordering prose should be framed as an invariance/stability ablation; matched reversed training/test is available, while forward-trained-on-reversed evidence is still validation-only in the current result set.\n")

    lines.append("## Post-hoc p(accept) calibration\n")
    lines.append("| case | val n | test n | bACC at tau=0 | AUC | raw ECE | Platt ECE | Platt a | Platt b |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in calibration_rows:
        lines.append(
            f"| {row['case']} | {row['n_val']} | {row['n_test']} | {row['test_bacc']:.1f} | "
            f"{fmt_auc(row['auc'])} | {row['raw_ece']:.3f} | {row['platt_ece']:.3f} | "
            f"{row['platt_a']:.3f} | {row['platt_b']:.3f} |"
        )
    lines.append("")
    lines.append("Calibration prose should be explicit: Platt scaling makes `p(accept)` more interpretable as a probability, but all rank metrics and threshold-free comparisons should still be computed on the same score ordering.\n")

    lines.append("## Recommended paper structure changes\n")
    lines.append("1. **Introduction:** lead with leakage-free paper-quality estimation across ICLR and arxiv, not only multimodal SFT. State the two objectives and the metric split upfront.")
    lines.append("2. **Dataset section:** introduce ICLR-latest-rebuttal curation, then arxiv accepts/rejects with false-positive/false-negative risks. Include the panel mosaic and a venue/year count table.")
    lines.append("3. **Metrics section before results:** define `p(accept)`, bACC, AccR/RejR, AUC, rating/citation correlations, raw-accuracy prior formula, and what threshold calibration can/cannot change.")
    lines.append("4. **Main results:** use a compact table with rows for PaperLens ICLR-trained, PaperLens arxiv-trained, DeepReviewer, Gemini, GPT; columns should separate ICLR and arxiv, bACC/AUC/rating/citation.")
    lines.append("5. **Scaling/distribution results:** train-ratio, arxiv-vs-ICLR training distribution, text/vision/model-size scaling, and temporal family trends.")
    lines.append("6. **Ablations:** ordering and panel representation should be short, visual, and tied to robustness/cost.")
    lines.append("7. **Calibration:** show post-hoc `p(accept)` reliability after the main results, not as the main claim.")
    lines.append("8. **Remove/condense old interpretation section:** keep only the pieces needed for the Claude Code review-improvement story or move them to appendix.\n")

    lines.append("## Missing or risky analyses to finish next\n")
    lines.append("- Common-subset comparison for GPT/Gemini/DeepReviewer/PaperLens before claiming closed-source wins/losses.")
    lines.append("- Base-model zero-shot baselines for the final eval suite.")
    lines.append("- Full arxiv-trained large text/vision/natural checkpoint integration once runs finish.")
    lines.append("- Venue/year count table for ICLR and arxiv, plus temporality plots from `ratio_xeval_family_analysis.py` promoted into the paper.")
    lines.append("- Per-venue/category arxiv validity checks: accept recall is especially important because arxiv reject labels can include papers never submitted to the main conference.")
    lines.append("- Statistical CIs/significance where methodologically clean and important; otherwise report point estimates with sample sizes and avoid overclaiming underpowered comparisons.")
    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    recall_rows = make_threshold_recall_movement()
    make_panel_mosaic()
    ablation_rows = make_panel_order_ablation()
    calibration_rows = make_platt_calibration()
    write_report(recall_rows, ablation_rows, calibration_rows)
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
