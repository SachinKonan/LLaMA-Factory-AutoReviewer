#!/usr/bin/env python3
"""
Plot training and test metrics for the optim_search_2026 experiments.

Sweep over batch size (bz16, bz32, bz64) and learning rate.
All use: 4 epochs, cosine_then_constant (decay_ratio=0.75, min_lr_rate=0.001).

Creates five figures:
1. Train metrics — Text (3×8 grid: rows=bz, cols=metrics)
2. Train metrics — Vision (3×8)
3. Test metrics (4×7 grid: text, vision, text w/o 2026, vision w/o 2026)
4. LR schedule (1×3: one per batch size, text only since schedule is the same)

Usage:
    python scripts/plot_optim_2026.py
"""

import json
import math
import re
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np

# ── Directories ──────────────────────────────────────────────────────────────
BASE = Path(".")
SAVES_BASE = BASE / "saves" / "final_sweep_v7_datasweepv3" / "optim_search_2026"
RESULTS_BASE = BASE / "results" / "final_sweep_v7_datasweepv3" / "optim_search_2026"
OUTPUT_DIR = BASE / "results" / "final_sweep_v7_datasweepv3" / "optim_search_2026" / "_plots"

# ── Test datasets (for year filtering) ───────────────────────────────────────
TEXT_TEST_DATASET = BASE / "data" / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_v7_filtered_test" / "data.json"
VISION_TEST_DATASET = BASE / "data" / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_v7_filtered_filtered24480_test" / "data.json"

# ── Experiments ──────────────────────────────────────────────────────────────
NUM_EPOCHS = 4
BATCH_SIZES = ["bz16", "bz32", "bz64"]

# (exp_dir_name, label)
TEXT_EXPS = {
    "bz16": [("bz16_lr0.5e-6_text", "lr=0.5e-6"), ("bz16_lr1e-6_text", "lr=1e-6")],
    "bz32": [("bz32_lr1e-6_text", "lr=1e-6"), ("bz32_lr2e-6_text", "lr=2e-6")],
    "bz64": [("bz64_lr2e-6_text", "lr=2e-6"), ("bz64_lr4e-6_text", "lr=4e-6")],
}

VISION_EXPS = {
    "bz16": [("bz16_lr1e-6_vision", "lr=1e-6"), ("bz16_lr2e-6_vision", "lr=2e-6")],
    "bz32": [("bz32_lr2e-6_vision", "lr=2e-6"), ("bz32_lr4e-6_vision", "lr=4e-6")],
    "bz64": [("bz64_lr4e-6_vision", "lr=4e-6"), ("bz64_lr5.5e-6_vision", "lr=5.5e-6")],
}

# ── Colors ───────────────────────────────────────────────────────────────────
ALL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
LR_COLORS = ["#1f77b4", "#d62728"]  # blue, red (for batchsize plots)

# Flat experiment lists for overlaid plots
TEXT_EXPS_FLAT = []
for bz in BATCH_SIZES:
    TEXT_EXPS_FLAT.extend(TEXT_EXPS[bz])
VISION_EXPS_FLAT = []
for bz in BATCH_SIZES:
    VISION_EXPS_FLAT.extend(VISION_EXPS[bz])

# ── Plot styling ─────────────────────────────────────────────────────────────
LABELSIZE = 13
TITLESIZE = 14
LEGENDSIZE = 7.5
TICKSIZE = 10


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_trainer_state(save_dir: Path) -> list:
    """Load log_history from trainer_state.json, falling back to trainer_log.jsonl."""
    path = save_dir / "trainer_state.json"
    if path.exists():
        with open(path) as f:
            return json.load(f).get("log_history", [])
    log_path = save_dir / "trainer_log.jsonl"
    if log_path.exists():
        entries = []
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                mapped = {}
                if "loss" in entry:
                    mapped["loss"] = entry["loss"]
                if "lr" in entry:
                    mapped["learning_rate"] = entry["lr"]
                if "epoch" in entry:
                    mapped["epoch"] = entry["epoch"]
                if "current_steps" in entry:
                    mapped["step"] = entry["current_steps"]
                if mapped:
                    entries.append(mapped)
        return entries
    return []


def get_steps_per_epoch(log_history: list) -> float:
    """Compute steps per epoch from trainer_state log_history."""
    for entry in reversed(log_history):
        if "train_loss" in entry:
            return entry["step"] / entry["epoch"]
    for entry in reversed(log_history):
        if "loss" in entry and entry.get("epoch", 0) > 0:
            return entry["step"] / entry["epoch"]
    return 1.0


def load_train_ckpt_metrics(results_dir: Path) -> list:
    """Load train-ckpt-*.json files, return list of (epoch, metrics) sorted by epoch."""
    results = []
    for p in sorted(results_dir.glob("train-ckpt-*.json")):
        with open(p) as f:
            data = json.load(f)
        epoch = data.get("epoch", 0)
        results.append((epoch, data))
    results.sort(key=lambda x: x[0])
    return results


def discover_jsonl_files(results_dir: Path, save_dir: Path) -> list:
    """Discover JSONL files and map them to epochs."""
    log_history = load_trainer_state(save_dir)
    spe = get_steps_per_epoch(log_history)

    results = []
    for p in sorted(results_dir.glob("finetuned*.jsonl")):
        name = p.stem
        if name == "finetuned":
            epoch = float(NUM_EPOCHS)
        else:
            m = re.search(r"ckpt-(\d+)", name)
            if m:
                step = int(m.group(1))
                epoch = round(step / spe)
            else:
                continue
        results.append((epoch, p))

    results.sort(key=lambda x: x[0])
    return results


def extract_prediction(text: str) -> str:
    """Extract Accept/Reject from prediction text."""
    text_lower = text.lower().strip()
    if "\\boxed{accept}" in text_lower or "boxed{accept}" in text_lower:
        return "accept"
    if "\\boxed{reject}" in text_lower or "boxed{reject}" in text_lower:
        return "reject"
    if "\\boxed{yes}" in text_lower or "boxed{yes}" in text_lower:
        return "accept"
    if "\\boxed{no}" in text_lower or "boxed{no}" in text_lower:
        return "reject"
    return "unknown"


def load_test_years(dataset_path: Path) -> list:
    """Load year for each sample in test dataset."""
    with open(dataset_path) as f:
        data = json.load(f)
    return [d.get("_metadata", {}).get("year") for d in data]


def compute_test_metrics_from_jsonl(jsonl_path: Path, year_filter: list = None, exclude_years: set = None):
    """Compute test metrics from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file.
        year_filter: Optional list of years (one per entry). If provided, entries
            whose year is in exclude_years are skipped.
        exclude_years: Set of years to exclude. Only used if year_filter is provided.
    """
    predictions = []
    labels = []
    p_accepts = []
    p_rejects = []
    p_corrects = []
    p_accept_gtaccept = []
    p_reject_gtaccept = []
    p_accept_gtreject = []
    p_reject_gtreject = []

    has_logprobs = False

    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            # Year filtering
            if year_filter is not None and exclude_years is not None:
                if idx < len(year_filter) and year_filter[idx] in exclude_years:
                    continue

            entry = json.loads(line)
            pred_text = entry.get("predict", "")
            label_text = entry.get("label", "")
            token_logprobs = entry.get("token_logprobs")

            pred = extract_prediction(pred_text)
            label = extract_prediction(label_text)

            if pred == "unknown" or label == "unknown":
                continue

            predictions.append(pred)
            labels.append(label)

            if token_logprobs is not None and len(token_logprobs) >= 6:
                has_logprobs = True
                logprob = token_logprobs[5]
                p_pred_class = math.exp(logprob)

                if pred == "accept":
                    p_accept = p_pred_class
                    p_reject = 1.0 - p_pred_class
                else:
                    p_reject = p_pred_class
                    p_accept = 1.0 - p_pred_class

                p_accepts.append(p_accept)
                p_rejects.append(p_reject)

                if label == "accept":
                    p_corrects.append(p_accept)
                    p_accept_gtaccept.append(p_accept)
                    p_reject_gtaccept.append(p_reject)
                else:
                    p_corrects.append(p_reject)
                    p_accept_gtreject.append(p_accept)
                    p_reject_gtreject.append(p_reject)

    if not predictions:
        return None

    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(predictions)
    n_positive = sum(p == "accept" for p in predictions)
    pred_positive_rate = n_positive / len(predictions)

    result = {
        "accuracy": accuracy,
        "pred_positive_rate": pred_positive_rate,
        "n_samples": len(predictions),
    }

    if has_logprobs and p_accepts:
        result["p_accept_mean"] = np.mean(p_accepts)
        result["p_reject_mean"] = np.mean(p_rejects)
        result["p_correct_mean"] = np.mean(p_corrects)
        result["p_accept_gtaccept_mean"] = np.mean(p_accept_gtaccept) if p_accept_gtaccept else None
        result["p_reject_gtaccept_mean"] = np.mean(p_reject_gtaccept) if p_reject_gtaccept else None
        result["p_accept_gtreject_mean"] = np.mean(p_accept_gtreject) if p_accept_gtreject else None
        result["p_reject_gtreject_mean"] = np.mean(p_reject_gtreject) if p_reject_gtreject else None
    else:
        result["p_accept_mean"] = None
        result["p_reject_mean"] = None
        result["p_correct_mean"] = None
        result["p_accept_gtaccept_mean"] = None
        result["p_reject_gtaccept_mean"] = None
        result["p_accept_gtreject_mean"] = None
        result["p_reject_gtreject_mean"] = None

    return result


def compute_accuracy_by_year(jsonl_path: Path, year_list: list) -> dict:
    """Compute accuracy broken down by year.

    Returns: {year: {"accuracy": float, "n": int, "n_correct": int}}
    """
    from collections import defaultdict
    year_correct = defaultdict(int)
    year_total = defaultdict(int)

    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            if idx >= len(year_list):
                break
            year = year_list[idx]
            if year is None:
                continue

            entry = json.loads(line)
            pred = extract_prediction(entry.get("predict", ""))
            label = extract_prediction(entry.get("label", ""))

            if pred == "unknown" or label == "unknown":
                continue

            year_total[year] += 1
            if pred == label:
                year_correct[year] += 1

    result = {}
    for year in sorted(year_total.keys()):
        n = year_total[year]
        result[year] = {
            "accuracy": year_correct[year] / n if n > 0 else 0.0,
            "n": n,
            "n_correct": year_correct[year],
        }
    return result


def slope_label(epochs, values, base_label):
    """Add slope annotation to label for accuracy plots."""
    if len(epochs) >= 2:
        coeffs = np.polyfit(epochs, values, 1)
        return f"{base_label} (m={coeffs[0]:.3f})"
    return base_label


def add_peak_textbox(ax, peak_lines: list):
    """Add a text box showing peak accuracy per method."""
    lines = []
    for label, color, val, ep in peak_lines:
        lines.append(f"{label}: {val:.3f} @ ep{ep:.0f}")
    text = "\n".join(lines)
    props = dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7)
    ax.text(0.98, 0.02, text, transform=ax.transAxes, fontsize=7.5,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=props, family="monospace")


# ── Figure: Train Metrics (simple 2×4) ───────────────────────────────────────

def plot_train_metrics():
    """Create 2×4 grid: rows=text/vision, cols=loss/accuracy/P(correct)/pred_pos.
    All 6 experiments overlaid on same axes per modality.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)

    modalities = [
        (0, TEXT_EXPS_FLAT, "Text"),
        (1, VISION_EXPS_FLAT, "Vision"),
    ]

    for row_idx, exps_flat, row_label in modalities:
        for exp_idx, (exp_name, lr_label) in enumerate(exps_flat):
            save_dir = SAVES_BASE / exp_name
            results_dir = RESULTS_BASE / exp_name
            color = ALL_COLORS[exp_idx % len(ALL_COLORS)]
            bz = re.match(r"(bz\d+)", exp_name).group(1)
            full_label = f"{bz} {lr_label}"

            if not save_dir.exists():
                continue

            # Col 0: Train Loss
            ax = axes[row_idx, 0]
            log_history = load_trainer_state(save_dir)
            loss_epochs = []
            loss_values = []
            for entry in log_history:
                if "loss" in entry and "epoch" in entry and "train_loss" not in entry:
                    loss_epochs.append(entry["epoch"])
                    loss_values.append(entry["loss"])
            if loss_epochs:
                ax.plot(loss_epochs, loss_values, color=color, alpha=0.8,
                        linewidth=0.8, label=full_label)

            # Cols 1-3: Train ckpt metrics
            ckpt_data = load_train_ckpt_metrics(results_dir)
            if ckpt_data:
                ckpt_epochs = [e for e, _ in ckpt_data]
                metric_cols = [
                    (1, "sft_accuracy"),
                    (2, "sft_p_correct_mean"),
                    (3, "sft_pred_positive_rate"),
                ]
                for col, metric_key in metric_cols:
                    ax = axes[row_idx, col]
                    vals = [d.get(metric_key) for _, d in ckpt_data]
                    valid = [(e, v) for e, v in zip(ckpt_epochs, vals) if v is not None]
                    if valid:
                        es, vs = zip(*valid)
                        es, vs = list(es), list(vs)
                        lbl = slope_label(es, vs, full_label) if col == 1 else full_label
                        ax.plot(es, vs, "o-", color=color, markersize=4,
                                linewidth=1.5, label=lbl)

    col_titles = ["Train Loss", "Train Accuracy", "Train P(Correct)", "Train Pred Pos Rate"]
    row_labels = ["Text", "Vision"]

    for col in range(4):
        axes[0, col].set_title(col_titles[col], fontsize=TITLESIZE)
        for row in range(2):
            ax = axes[row, col]
            ax.set_xlabel("Epoch", fontsize=LABELSIZE)
            ax.tick_params(labelsize=TICKSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGENDSIZE, loc="best")
            if col == 0:
                ax.set_ylim(0, 1)
            else:
                ax.set_xticks(list(range(1, NUM_EPOCHS + 1)))
                ax.set_xlim(0.5, NUM_EPOCHS + 0.5)

    for row in range(2):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=LABELSIZE, fontweight="bold")

    fig.suptitle("Optim Search 2026: Train Metrics (4ep, cosine_then_constant)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "optim_2026_train_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure: Test Metrics (simple 4×3, with w/o 2026) ────────────────────────

def plot_test_metrics():
    """Create 4×3 grid: rows=text/vision/text w/o 2026/vision w/o 2026,
    cols=accuracy/P(correct)/pred_pos. All experiments overlaid.
    """
    fig, axes = plt.subplots(4, 3, figsize=(12, 16), squeeze=False)

    text_years = load_test_years(TEXT_TEST_DATASET)
    vision_years = load_test_years(VISION_TEST_DATASET)

    row_configs = [
        (0, TEXT_EXPS_FLAT, "Text (all)", None, None),
        (1, VISION_EXPS_FLAT, "Vision (all)", None, None),
        (2, TEXT_EXPS_FLAT, "Text (w/o 2026)", text_years, {2026}),
        (3, VISION_EXPS_FLAT, "Vision (w/o 2026)", vision_years, {2026}),
    ]

    test_acc_peaks: dict = {i: [] for i in range(4)}

    for row_idx, exps_flat, row_label, year_filter, exclude_years in row_configs:
        for exp_idx, (exp_name, lr_label) in enumerate(exps_flat):
            save_dir = SAVES_BASE / exp_name
            results_dir = RESULTS_BASE / exp_name
            color = ALL_COLORS[exp_idx % len(ALL_COLORS)]
            bz = re.match(r"(bz\d+)", exp_name).group(1)
            full_label = f"{bz} {lr_label}"

            if not results_dir.exists():
                continue

            jsonl_files = discover_jsonl_files(results_dir, save_dir)
            if not jsonl_files:
                continue

            test_epochs = []
            test_acc = []
            test_p_correct = []
            test_pred_pos = []

            for epoch, jsonl_path in jsonl_files:
                metrics = compute_test_metrics_from_jsonl(
                    jsonl_path, year_filter=year_filter, exclude_years=exclude_years
                )
                if metrics is None:
                    continue
                test_epochs.append(epoch)
                test_acc.append(metrics["accuracy"])
                test_p_correct.append(metrics["p_correct_mean"])
                test_pred_pos.append(metrics["pred_positive_rate"])

            metric_series = [
                (0, test_acc, True),
                (1, test_p_correct, False),
                (2, test_pred_pos, False),
            ]

            for col, values, add_slope in metric_series:
                valid = [(e, v) for e, v in zip(test_epochs, values) if v is not None]
                if valid:
                    es, vs = zip(*valid)
                    es, vs = list(es), list(vs)
                    lbl = slope_label(es, vs, full_label) if add_slope else full_label
                    axes[row_idx, col].plot(es, vs, "o-", color=color,
                                            markersize=4, linewidth=1.5, label=lbl)
                    if col == 0:
                        best_idx = int(np.argmax(vs))
                        test_acc_peaks[row_idx].append(
                            (full_label, color, vs[best_idx], es[best_idx])
                        )

    for row_idx in range(4):
        if test_acc_peaks[row_idx]:
            add_peak_textbox(axes[row_idx, 0], test_acc_peaks[row_idx])

    col_titles = ["Test Accuracy", "Test P(Correct)", "Test Pred Pos Rate"]
    row_labels = ["Text\n(all years)", "Vision\n(all years)",
                  "Text\n(w/o 2026)", "Vision\n(w/o 2026)"]

    for col in range(3):
        axes[0, col].set_title(col_titles[col], fontsize=TITLESIZE)
        for row in range(4):
            ax = axes[row, col]
            ax.set_xlabel("Epoch", fontsize=LABELSIZE)
            ax.set_xticks(list(range(1, NUM_EPOCHS + 1)))
            ax.set_xlim(0.5, NUM_EPOCHS + 0.5)
            ax.tick_params(labelsize=TICKSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGENDSIZE, loc="best")

    for row in range(4):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=LABELSIZE, fontweight="bold")

    fig.suptitle("Optim Search 2026: Test Metrics (4ep, cosine_then_constant)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "optim_2026_test_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure: Train Loss by Batch Size (2×3) ──────────────────────────────────

def plot_loss_by_batchsize():
    """Create 2×3 grid: rows=text/vision, cols=bz16/bz32/bz64, lines=LRs."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)

    modalities = [
        (0, TEXT_EXPS, "Text"),
        (1, VISION_EXPS, "Vision"),
    ]

    for row_idx, exps_dict, row_label in modalities:
        for col_idx, bz_key in enumerate(BATCH_SIZES):
            ax = axes[row_idx, col_idx]
            exp_list = exps_dict[bz_key]

            for lr_idx, (exp_name, lr_label) in enumerate(exp_list):
                save_dir = SAVES_BASE / exp_name
                if not save_dir.exists():
                    continue
                log_history = load_trainer_state(save_dir)

                loss_epochs = []
                loss_values = []
                for entry in log_history:
                    if "loss" in entry and "epoch" in entry and "train_loss" not in entry:
                        loss_epochs.append(entry["epoch"])
                        loss_values.append(entry["loss"])

                if loss_epochs:
                    color = LR_COLORS[lr_idx % len(LR_COLORS)]
                    ax.plot(loss_epochs, loss_values, color=color, alpha=0.8,
                            linewidth=0.8, label=lr_label)

            ax.set_xlabel("Epoch", fontsize=LABELSIZE)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=TICKSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGENDSIZE, loc="best")

            if row_idx == 0:
                ax.set_title(f"Batch Size {bz_key[2:]}", fontsize=TITLESIZE)

        axes[row_idx, 0].set_ylabel(f"{row_label}\nTrain Loss", fontsize=LABELSIZE, fontweight="bold")

    fig.suptitle("Optim Search 2026: Train Loss by Batch Size", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "optim_2026_train_loss_by_batchsize.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Summary Table ────────────────────────────────────────────────────────────

def compute_recall_metrics(jsonl_path: Path):
    """Compute accuracy, accept recall, reject recall from a JSONL file."""
    tp = fp = tn = fn = 0
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            pred = extract_prediction(entry.get("predict", ""))
            label = extract_prediction(entry.get("label", ""))
            if pred == "unknown" or label == "unknown":
                continue
            if label == "accept" and pred == "accept":
                tp += 1
            elif label == "accept" and pred == "reject":
                fn += 1
            elif label == "reject" and pred == "reject":
                tn += 1
            elif label == "reject" and pred == "accept":
                fp += 1
    total = tp + fp + tn + fn
    if total == 0:
        return None
    accuracy = (tp + tn) / total
    accept_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    reject_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    return accuracy, accept_recall, reject_recall


def print_summary_table():
    """Print summary table of best checkpoint accuracy for all experiments."""
    print(f"\n{'=' * 80}")
    print(f"  Optim Search 2026: Best Checkpoint Summary")
    print(f"  Format: [acc]/accept_recall/reject_recall/EP{{i}}")
    print(f"{'=' * 80}")

    all_keys = set()
    data = {"text": {}, "vision": {}}

    for modality, exps_dict in [("text", TEXT_EXPS), ("vision", VISION_EXPS)]:
        for bz_key in BATCH_SIZES:
            for exp_name, lr_label in exps_dict[bz_key]:
                save_dir = SAVES_BASE / exp_name
                results_dir = RESULTS_BASE / exp_name

                m = re.match(r"bz(\d+)_lr(.+?)_(text|vision)", exp_name)
                if not m:
                    continue
                bz = int(m.group(1))
                lr_str = m.group(2)

                jsonl_files = discover_jsonl_files(results_dir, save_dir)
                if not jsonl_files:
                    continue

                best_acc = -1
                best_result = None
                for epoch, jsonl_path in jsonl_files:
                    result = compute_recall_metrics(jsonl_path)
                    if result is None:
                        continue
                    acc, ar, rr = result
                    if acc > best_acc:
                        best_acc = acc
                        best_result = (acc, ar, rr, int(epoch))

                if best_result:
                    key = (bz, lr_str)
                    all_keys.add(key)
                    data[modality][key] = best_result

    def lr_sort_key(lr_str):
        return float(lr_str.replace("e-6", "e-6"))

    sorted_keys = sorted(all_keys, key=lambda x: (x[0], lr_sort_key(x[1])))

    col_headers = [f"bz{bz} lr={lr}" for bz, lr in sorted_keys]
    col_width = 24
    header = f"{'Modality':<10}" + "".join(f"{h:>{col_width}}" for h in col_headers)
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for modality in ["text", "vision"]:
        row = f"{modality:<10}"
        for key in sorted_keys:
            if key in data[modality]:
                acc, ar, rr, ep = data[modality][key]
                cell = f"{acc:.1%}/{ar:.1%}/{rr:.1%}/EP{ep}"
                row += f"{cell:>{col_width}}"
            else:
                row += f"{'—':>{col_width}}"
        print(row)

    print(sep)
    print()


# ── Figure: LR Schedule ─────────────────────────────────────────────────────

def plot_lr_schedule():
    """Create 1×3 grid showing LR schedule for each batch size (text experiments)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)

    for col_idx, bz_key in enumerate(BATCH_SIZES):
        ax = axes[0, col_idx]
        exp_list = TEXT_EXPS[bz_key]
        for exp_idx, (exp_name, lr_label) in enumerate(exp_list):
            save_dir = SAVES_BASE / exp_name
            if not save_dir.exists():
                continue
            log_history = load_trainer_state(save_dir)
            color = LR_COLORS[exp_idx]

            lr_epochs = []
            lr_values = []
            for entry in log_history:
                if "learning_rate" in entry and "epoch" in entry:
                    lr_epochs.append(entry["epoch"])
                    lr_values.append(entry["learning_rate"])
            if lr_epochs:
                ax.plot(lr_epochs, lr_values, color=color, alpha=0.8,
                        linewidth=0.8, label=lr_label)

        ax.set_title(f"{bz_key}", fontsize=TITLESIZE)
        ax.set_xlabel("Epoch", fontsize=LABELSIZE)
        ax.set_ylabel("Learning Rate", fontsize=LABELSIZE)
        ax.tick_params(labelsize=TICKSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGENDSIZE, loc="best")

    fig.suptitle("Optim Search 2026: LR Schedule (cosine_then_constant, decay_ratio=0.75)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "optim_2026_lr_schedule.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure: Accuracy-by-Year Heatmaps ────────────────────────────────────────

def plot_accuracy_by_year_heatmaps(modality: str, exps_dict: dict, dataset_path: Path):
    """Create a grid of heatmaps: one per experiment.
    Rows: batch sizes (bz16, bz32, bz64).
    Cols: LR values (2 per batch size).
    Each heatmap: x=epoch, y=year, color=accuracy, white text inside cells.
    """
    years_list = load_test_years(dataset_path)
    all_years = sorted(set(y for y in years_list if y is not None))

    n_rows = len(BATCH_SIZES)
    n_cols = max(len(exps_dict[bz]) for bz in BATCH_SIZES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for row_idx, bz_key in enumerate(BATCH_SIZES):
        exp_list = exps_dict[bz_key]
        for col_idx, (exp_name, lr_label) in enumerate(exp_list):
            ax = axes[row_idx, col_idx]
            save_dir = SAVES_BASE / exp_name
            results_dir = RESULTS_BASE / exp_name

            if not results_dir.exists():
                ax.set_title(f"{bz_key} {lr_label}\n(no data)", fontsize=TITLESIZE)
                ax.set_visible(False)
                continue

            jsonl_files = discover_jsonl_files(results_dir, save_dir)
            if not jsonl_files:
                ax.set_title(f"{bz_key} {lr_label}\n(no data)", fontsize=TITLESIZE)
                ax.set_visible(False)
                continue

            # Build matrix: rows=years, cols=epochs
            epochs = [int(e) for e, _ in jsonl_files]
            matrix = np.full((len(all_years), len(epochs)), np.nan)
            counts = np.full((len(all_years), len(epochs)), 0, dtype=int)

            for col_e, (epoch, jsonl_path) in enumerate(jsonl_files):
                by_year = compute_accuracy_by_year(jsonl_path, years_list)
                for row_y, year in enumerate(all_years):
                    if year in by_year:
                        matrix[row_y, col_e] = by_year[year]["accuracy"]
                        counts[row_y, col_e] = by_year[year]["n"]

            # Plot heatmap
            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=0.85)

            # Add text annotations (white with dark outline for visibility)
            for i in range(len(all_years)):
                for j in range(len(epochs)):
                    val = matrix[i, j]
                    n = counts[i, j]
                    if not np.isnan(val):
                        txt = f"{val:.1%}\n(n={n})"
                        ax.text(j, i, txt, ha="center", va="center",
                                fontsize=13, fontweight="bold", color="white",
                                path_effects=[
                                    patheffects.withStroke(linewidth=3, foreground="black")
                                ])

            ax.set_xticks(range(len(epochs)))
            ax.set_xticklabels([f"Ep {e}" for e in epochs], fontsize=TICKSIZE + 2)
            ax.set_yticks(range(len(all_years)))
            ax.set_yticklabels([str(y) for y in all_years], fontsize=TICKSIZE + 2)
            ax.set_xlabel("Epoch", fontsize=LABELSIZE)
            if col_idx == 0:
                ax.set_ylabel("Year", fontsize=LABELSIZE)
            ax.set_title(f"{bz_key}  {lr_label}", fontsize=TITLESIZE, fontweight="bold")

        # Hide extra columns if this batch size has fewer LRs
        for col_idx in range(len(exp_list), n_cols):
            axes[row_idx, col_idx].set_visible(False)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=0.4, vmax=0.85), cmap="RdYlGn"
    ), cax=cbar_ax)
    cbar.set_label("Accuracy", fontsize=LABELSIZE)
    cbar.ax.tick_params(labelsize=TICKSIZE)

    mod_label = "Text" if modality == "text" else "Vision"
    fig.suptitle(f"Optim Search 2026: Test Accuracy by Year — {mod_label}",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.subplots_adjust(right=0.91)
    fig.tight_layout(rect=[0, 0, 0.91, 0.98])

    out_path = OUTPUT_DIR / f"optim_2026_accuracy_by_year_{modality}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Plotting train metrics...")
    plot_train_metrics()
    print("Plotting test metrics...")
    plot_test_metrics()
    print("Plotting train loss by batch size...")
    plot_loss_by_batchsize()
    print("Plotting LR schedule...")
    plot_lr_schedule()
    print("Plotting accuracy-by-year heatmaps (text)...")
    plot_accuracy_by_year_heatmaps("text", TEXT_EXPS, TEXT_TEST_DATASET)
    print("Plotting accuracy-by-year heatmaps (vision)...")
    plot_accuracy_by_year_heatmaps("vision", VISION_EXPS, VISION_TEST_DATASET)
    print_summary_table()
    print("Done.")
