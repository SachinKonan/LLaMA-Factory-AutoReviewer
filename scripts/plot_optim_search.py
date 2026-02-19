#!/usr/bin/env python3
"""
Plot training and test metrics for optim_search experiments (batch size × learning rate).

Creates two figures:
1. Train metrics (2×6 grid): loss, accuracy, P(Accept), P(Reject), P(Correct), pred positive rate
2. Test metrics (2×5 grid): accuracy, P(Accept), P(Reject), P(Correct), pred positive rate

Usage:
    python scripts/plot_optim_search.py
"""

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Directories ──────────────────────────────────────────────────────────────
BASE = Path(".")
SAVES_BASE = BASE / "saves" / "final_sweep_v7_datasweepv3" / "optim_search_6epochs"
RESULTS_BASE = BASE / "results" / "final_sweep_v7_datasweepv3" / "optim_search_6epochs"
OUTPUT_DIR = RESULTS_BASE

SAVES_3EP = BASE / "saves" / "final_sweep_v7_datasweepv3" / "optim_search"
RESULTS_3EP = BASE / "results" / "final_sweep_v7_datasweepv3" / "optim_search"
OUTPUT_DIR_3EP = RESULTS_3EP

# ── Experiments ──────────────────────────────────────────────────────────────
TEXT_EXPS = [
    "bz16_lr0.5e-6",
    "bz16_lr1e-6",
    "bz32_lr1e-6",
    "bz32_lr2e-6",
    "bz64_lr2e-6",
    "bz64_lr4e-6",
]
VISION_EXPS = [
    "bz16_lr1e-6",
    "bz16_lr2e-6",
    "bz32_lr2e-6",
    "bz32_lr4e-6",
    "bz64_lr4e-6",
    "bz64_lr5.5e-6",
]

# ── Colors ───────────────────────────────────────────────────────────────────
COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]

# ── Plot styling ─────────────────────────────────────────────────────────────
LABELSIZE = 13
TITLESIZE = 14
LEGENDSIZE = 7.5
TICKSIZE = 10


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_trainer_state(save_dir: Path) -> list:
    """Load log_history from trainer_state.json."""
    path = save_dir / "trainer_state.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("log_history", [])


def get_steps_per_epoch(log_history: list) -> float:
    """Compute steps per epoch from trainer_state log_history."""
    # Find the last entry with both epoch and step (training summary)
    for entry in reversed(log_history):
        if "train_loss" in entry:
            # This is the training summary entry
            return entry["step"] / entry["epoch"]
    # Fallback: use log entries
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
    """Discover JSONL files and map them to epochs.

    Returns list of (epoch, jsonl_path) sorted by epoch.
    """
    # Get steps_per_epoch from trainer_state
    log_history = load_trainer_state(save_dir)
    spe = get_steps_per_epoch(log_history)

    results = []
    for p in sorted(results_dir.glob("finetuned*.jsonl")):
        name = p.stem
        if name == "finetuned":
            # Final model = epoch 3
            epoch = 6.0
        else:
            # finetuned-ckpt-{step}
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


def extract_label(text: str) -> str:
    """Extract Accept/Reject from label text."""
    return extract_prediction(text)


def compute_test_metrics_from_jsonl(jsonl_path: Path):
    """Compute test metrics from a JSONL file.

    Returns dict with accuracy, p_accept_mean, p_reject_mean, p_correct_mean,
    pred_positive_rate. Confidence metrics are None if logprobs unavailable.
    """
    predictions = []
    labels = []
    p_accepts = []
    p_rejects = []
    p_corrects = []

    has_logprobs = False

    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            pred_text = entry.get("predict", "")
            label_text = entry.get("label", "")
            token_logprobs = entry.get("token_logprobs")

            pred = extract_prediction(pred_text)
            label = extract_label(label_text)

            if pred == "unknown" or label == "unknown":
                continue

            predictions.append(pred)
            labels.append(label)

            # Logprob-based confidence
            if token_logprobs is not None and len(token_logprobs) >= 6:
                has_logprobs = True
                # Position 5 is the Accept/Reject token
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
                else:
                    p_corrects.append(p_reject)

    if not predictions:
        return None

    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / len(predictions)
    n_positive = sum(p == "accept" for p in predictions)
    pred_positive_rate = n_positive / len(predictions)

    result = {
        "accuracy": accuracy,
        "pred_positive_rate": pred_positive_rate,
    }

    if has_logprobs and p_accepts:
        result["p_accept_mean"] = np.mean(p_accepts)
        result["p_reject_mean"] = np.mean(p_rejects)
        result["p_correct_mean"] = np.mean(p_corrects)
    else:
        result["p_accept_mean"] = None
        result["p_reject_mean"] = None
        result["p_correct_mean"] = None

    return result


def slope_label(epochs, values, base_label):
    """Add slope annotation to label for accuracy plots."""
    if len(epochs) >= 2:
        # Linear regression slope
        coeffs = np.polyfit(epochs, values, 1)
        return f"{base_label} (m={coeffs[0]:.3f})"
    return base_label


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


def print_summary_table(saves_base: Path, results_base: Path, label: str):
    """Print a summary table of best checkpoint accuracy for all experiments."""
    print(f"\n{'=' * 80}")
    print(f"  {label}: Best Checkpoint Summary")
    print(f"  Format: [acc]/accept_recall/reject_recall/EP{{i}}")
    print(f"{'=' * 80}")

    # Collect all (bz, lr) keys and results per modality
    all_keys = set()
    data = {"text": {}, "vision": {}}

    for modality, exps in [("text", TEXT_EXPS), ("vision", VISION_EXPS)]:
        for exp_name in exps:
            dir_name = f"{exp_name}_{modality}"
            save_dir = saves_base / dir_name
            results_dir = results_base / dir_name

            m = re.match(r"bz(\d+)_lr(.+)", exp_name)
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

    # Sort keys by bz, then by lr (parse float for sorting)
    def lr_sort_key(lr_str):
        return float(lr_str.replace("e-6", "e-6"))

    sorted_keys = sorted(all_keys, key=lambda x: (x[0], lr_sort_key(x[1])))

    # Build column headers
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


# ── Figure 1: Train Metrics ─────────────────────────────────────────────────

def plot_train_metrics():
    """Create 2×4 grid of training metrics."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)

    modalities = [
        ("text", TEXT_EXPS, "Text"),
        ("vision", VISION_EXPS, "Vision"),
    ]

    for row_idx, (modality, exps, row_label) in enumerate(modalities):
        for exp_idx, exp_name in enumerate(exps):
            dir_name = f"{exp_name}_{modality}"
            save_dir = SAVES_BASE / dir_name
            results_dir = RESULTS_BASE / dir_name
            color = COLORS[exp_idx]

            # ── Col 0: Train Loss ────────────────────────────────────────
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
                        linewidth=0.8, label=exp_name)

            # ── Cols 1-3: Train ckpt metrics ─────────────────────────────
            ckpt_data = load_train_ckpt_metrics(results_dir)
            if ckpt_data:
                ckpt_epochs = [e for e, _ in ckpt_data]

                metric_cols = [
                    (1, "sft_accuracy", "Train Accuracy"),
                    (2, "sft_p_correct_mean", "Train P(Correct)"),
                    (3, "sft_pred_positive_rate", "Train Pred Positive Rate"),
                ]

                for col, metric_key, _ in metric_cols:
                    ax = axes[row_idx, col]
                    vals = [d.get(metric_key) for _, d in ckpt_data]
                    valid = [(e, v) for e, v in zip(ckpt_epochs, vals)
                             if v is not None]
                    if valid:
                        es, vs = zip(*valid)
                        es, vs = list(es), list(vs)
                        label = exp_name
                        if col == 1:  # accuracy: add slope
                            label = slope_label(es, vs, exp_name)
                        ax.plot(es, vs, "o-", color=color, markersize=4,
                                linewidth=1.5, label=label)

    # ── Formatting ───────────────────────────────────────────────────────
    col_titles = [
        "Train Loss", "Train Accuracy",
        "Train P(Correct)", "Train Pred Positive Rate",
    ]
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
                ax.set_xticks([1, 2, 3, 4, 5, 6])
                ax.set_xlim(0.5, 6.5)

    for row in range(2):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=LABELSIZE, fontweight="bold")

    fig.suptitle("Optim Search: Train Metrics", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "train_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure 2: Test Metrics ──────────────────────────────────────────────────

def plot_test_metrics():
    """Create 2×3 grid of test metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)

    modalities = [
        ("text", TEXT_EXPS, "Text"),
        ("vision", VISION_EXPS, "Vision"),
    ]

    for row_idx, (modality, exps, row_label) in enumerate(modalities):
        for exp_idx, exp_name in enumerate(exps):
            dir_name = f"{exp_name}_{modality}"
            save_dir = SAVES_BASE / dir_name
            results_dir = RESULTS_BASE / dir_name
            color = COLORS[exp_idx]

            # Discover JSONL files and map to epochs
            jsonl_files = discover_jsonl_files(results_dir, save_dir)
            if not jsonl_files:
                continue

            # Compute metrics for each epoch
            test_epochs = []
            test_acc = []
            test_p_correct = []
            test_pred_pos = []

            for epoch, jsonl_path in jsonl_files:
                metrics = compute_test_metrics_from_jsonl(jsonl_path)
                if metrics is None:
                    continue
                test_epochs.append(epoch)
                test_acc.append(metrics["accuracy"])
                test_pred_pos.append(metrics["pred_positive_rate"])
                test_p_correct.append(metrics["p_correct_mean"])

            # ── Col 0: Test Accuracy ─────────────────────────────────────
            if test_acc:
                label = slope_label(test_epochs, test_acc, exp_name)
                axes[row_idx, 0].plot(test_epochs, test_acc, "o-", color=color,
                                      markersize=4, linewidth=1.5, label=label)

            # ── Col 1: Test P(Correct) ───────────────────────────────────
            valid = [(e, v) for e, v in zip(test_epochs, test_p_correct)
                     if v is not None]
            if valid:
                es, vs = zip(*valid)
                axes[row_idx, 1].plot(es, vs, "o-", color=color, markersize=4,
                                      linewidth=1.5, label=exp_name)

            # ── Col 2: Test Pred Positive Rate ───────────────────────────
            if test_pred_pos:
                axes[row_idx, 2].plot(test_epochs, test_pred_pos, "o-",
                                      color=color, markersize=4, linewidth=1.5,
                                      label=exp_name)

    # ── Formatting ───────────────────────────────────────────────────────
    col_titles = [
        "Test Accuracy", "Test P(Correct)", "Test Pred Positive Rate",
    ]
    row_labels = ["Text", "Vision"]

    for col in range(3):
        axes[0, col].set_title(col_titles[col], fontsize=TITLESIZE)
        for row in range(2):
            ax = axes[row, col]
            ax.set_xlabel("Epoch", fontsize=LABELSIZE)
            ax.set_xticks([1, 2, 3, 4, 5, 6])
            ax.set_xlim(0.5, 6.5)
            ax.tick_params(labelsize=TICKSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGENDSIZE, loc="best")

    for row in range(2):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=LABELSIZE, fontweight="bold")

    fig.suptitle("Optim Search: Test Metrics", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "test_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure 3: Train Loss by Batch Size ───────────────────────────────────────

def plot_loss_by_batchsize():
    """Create 2×3 grid: rows = text/vision, cols = batch size, lines = learning rates."""
    LR_COLORS = ["#1f77b4", "#d62728"]  # blue, red

    # Group experiments by batch size
    text_by_bz = {"16": [], "32": [], "64": []}
    for exp in TEXT_EXPS:
        bz = re.match(r"bz(\d+)", exp).group(1)
        text_by_bz[bz].append(exp)

    vision_by_bz = {"16": [], "32": [], "64": []}
    for exp in VISION_EXPS:
        bz = re.match(r"bz(\d+)", exp).group(1)
        vision_by_bz[bz].append(exp)

    batch_sizes = ["16", "32", "64"]
    modalities = [
        ("text", text_by_bz, "Text"),
        ("vision", vision_by_bz, "Vision"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)

    for row_idx, (modality, by_bz, row_label) in enumerate(modalities):
        for col_idx, bz in enumerate(batch_sizes):
            ax = axes[row_idx, col_idx]
            exps = by_bz[bz]

            for lr_idx, exp_name in enumerate(exps):
                dir_name = f"{exp_name}_{modality}"
                save_dir = SAVES_BASE / dir_name
                log_history = load_trainer_state(save_dir)

                loss_epochs = []
                loss_values = []
                for entry in log_history:
                    if "loss" in entry and "epoch" in entry and "train_loss" not in entry:
                        loss_epochs.append(entry["epoch"])
                        loss_values.append(entry["loss"])

                if loss_epochs:
                    # Extract just the LR part for the label
                    lr_str = exp_name.split("_", 1)[1]
                    color = LR_COLORS[lr_idx % len(LR_COLORS)]
                    ax.plot(loss_epochs, loss_values, color=color, alpha=0.8,
                            linewidth=0.8, label=lr_str)

            ax.set_xlabel("Epoch", fontsize=LABELSIZE)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=TICKSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGENDSIZE, loc="best")

            if row_idx == 0:
                ax.set_title(f"Batch Size {bz}", fontsize=TITLESIZE)

        axes[row_idx, 0].set_ylabel(f"{row_label}\nTrain Loss", fontsize=LABELSIZE, fontweight="bold")

    fig.suptitle("Optim Search: Train Loss by Batch Size", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "train_loss_by_batchsize.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure 3b: Train Loss by Batch Size (3-epoch) ────────────────────────────

def plot_loss_by_batchsize_3ep():
    """Create 2×3 grid for 3-epoch optim_search: rows = text/vision, cols = batch size, lines = learning rates."""
    LR_COLORS = ["#1f77b4", "#d62728"]  # blue, red

    text_by_bz = {"16": [], "32": [], "64": []}
    for exp in TEXT_EXPS:
        bz = re.match(r"bz(\d+)", exp).group(1)
        text_by_bz[bz].append(exp)

    vision_by_bz = {"16": [], "32": [], "64": []}
    for exp in VISION_EXPS:
        bz = re.match(r"bz(\d+)", exp).group(1)
        vision_by_bz[bz].append(exp)

    batch_sizes = ["16", "32", "64"]
    modalities = [
        ("text", text_by_bz, "Text"),
        ("vision", vision_by_bz, "Vision"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)

    for row_idx, (modality, by_bz, row_label) in enumerate(modalities):
        for col_idx, bz in enumerate(batch_sizes):
            ax = axes[row_idx, col_idx]
            exps = by_bz[bz]

            for lr_idx, exp_name in enumerate(exps):
                dir_name = f"{exp_name}_{modality}"
                save_dir = SAVES_3EP / dir_name
                log_history = load_trainer_state(save_dir)

                loss_epochs = []
                loss_values = []
                for entry in log_history:
                    if "loss" in entry and "epoch" in entry and "train_loss" not in entry:
                        loss_epochs.append(entry["epoch"])
                        loss_values.append(entry["loss"])

                if loss_epochs:
                    lr_str = exp_name.split("_", 1)[1]
                    color = LR_COLORS[lr_idx % len(LR_COLORS)]
                    ax.plot(loss_epochs, loss_values, color=color, alpha=0.8,
                            linewidth=0.8, label=lr_str)

            ax.set_xlabel("Epoch", fontsize=LABELSIZE)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=TICKSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGENDSIZE, loc="best")

            if row_idx == 0:
                ax.set_title(f"Batch Size {bz}", fontsize=TITLESIZE)

        axes[row_idx, 0].set_ylabel(f"{row_label}\nTrain Loss", fontsize=LABELSIZE, fontweight="bold")

    fig.suptitle("Optim Search (3ep): Train Loss by Batch Size", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR_3EP / "train_loss_by_batchsize.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure 3c: Train Metrics (3-epoch) ───────────────────────────────────────

def plot_train_metrics_3ep():
    """Create 2×4 grid of training metrics for 3-epoch optim_search."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)

    modalities = [
        ("text", TEXT_EXPS, "Text"),
        ("vision", VISION_EXPS, "Vision"),
    ]

    for row_idx, (modality, exps, row_label) in enumerate(modalities):
        for exp_idx, exp_name in enumerate(exps):
            dir_name = f"{exp_name}_{modality}"
            save_dir = SAVES_3EP / dir_name
            results_dir = RESULTS_3EP / dir_name
            color = COLORS[exp_idx]

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
                        linewidth=0.8, label=exp_name)

            ckpt_data = load_train_ckpt_metrics(results_dir)
            if ckpt_data:
                ckpt_epochs = [e for e, _ in ckpt_data]

                metric_cols = [
                    (1, "sft_accuracy", "Train Accuracy"),
                    (2, "sft_p_correct_mean", "Train P(Correct)"),
                    (3, "sft_pred_positive_rate", "Train Pred Positive Rate"),
                ]

                for col, metric_key, _ in metric_cols:
                    ax = axes[row_idx, col]
                    vals = [d.get(metric_key) for _, d in ckpt_data]
                    valid = [(e, v) for e, v in zip(ckpt_epochs, vals)
                             if v is not None]
                    if valid:
                        es, vs = zip(*valid)
                        es, vs = list(es), list(vs)
                        label = exp_name
                        if col == 1:
                            label = slope_label(es, vs, exp_name)
                        ax.plot(es, vs, "o-", color=color, markersize=4,
                                linewidth=1.5, label=label)

    col_titles = [
        "Train Loss", "Train Accuracy",
        "Train P(Correct)", "Train Pred Positive Rate",
    ]
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
                ax.set_xticks([1, 2, 3])
                ax.set_xlim(0.5, 3.5)

    for row in range(2):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=LABELSIZE, fontweight="bold")

    fig.suptitle("Optim Search (3ep): Train Metrics", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR_3EP / "train_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure 3d: Test Metrics (3-epoch) ───────────────────────────────────────

def plot_test_metrics_3ep():
    """Create 2×3 grid of test metrics for 3-epoch optim_search."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)

    modalities = [
        ("text", TEXT_EXPS, "Text"),
        ("vision", VISION_EXPS, "Vision"),
    ]

    for row_idx, (modality, exps, row_label) in enumerate(modalities):
        for exp_idx, exp_name in enumerate(exps):
            dir_name = f"{exp_name}_{modality}"
            save_dir = SAVES_3EP / dir_name
            results_dir = RESULTS_3EP / dir_name
            color = COLORS[exp_idx]

            jsonl_files = discover_jsonl_files(results_dir, save_dir)
            if not jsonl_files:
                continue

            test_epochs = []
            test_acc = []
            test_p_correct = []
            test_pred_pos = []

            for epoch, jsonl_path in jsonl_files:
                metrics = compute_test_metrics_from_jsonl(jsonl_path)
                if metrics is None:
                    continue
                test_epochs.append(epoch)
                test_acc.append(metrics["accuracy"])
                test_pred_pos.append(metrics["pred_positive_rate"])
                test_p_correct.append(metrics["p_correct_mean"])

            if test_acc:
                label = slope_label(test_epochs, test_acc, exp_name)
                axes[row_idx, 0].plot(test_epochs, test_acc, "o-", color=color,
                                      markersize=4, linewidth=1.5, label=label)

            valid = [(e, v) for e, v in zip(test_epochs, test_p_correct)
                     if v is not None]
            if valid:
                es, vs = zip(*valid)
                axes[row_idx, 1].plot(es, vs, "o-", color=color, markersize=4,
                                      linewidth=1.5, label=exp_name)

            if test_pred_pos:
                axes[row_idx, 2].plot(test_epochs, test_pred_pos, "o-",
                                      color=color, markersize=4, linewidth=1.5,
                                      label=exp_name)

    col_titles = [
        "Test Accuracy", "Test P(Correct)", "Test Pred Positive Rate",
    ]
    row_labels = ["Text", "Vision"]

    for col in range(3):
        axes[0, col].set_title(col_titles[col], fontsize=TITLESIZE)
        for row in range(2):
            ax = axes[row, col]
            ax.set_xlabel("Epoch", fontsize=LABELSIZE)
            ax.set_xticks([1, 2, 3])
            ax.set_xlim(0.5, 3.5)
            ax.tick_params(labelsize=TICKSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGENDSIZE, loc="best")

    for row in range(2):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=LABELSIZE, fontweight="bold")

    fig.suptitle("Optim Search (3ep): Test Metrics", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR_3EP / "test_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure 4: Learning Rate Schedule ─────────────────────────────────────────

def plot_lr_schedule():
    """Create 2×3 grid: rows = text/vision, cols = batch size, lines = learning rates.
    Shows both 6-epoch (solid) and 3-epoch (dashed) schedules."""
    SAVES_3EP = BASE / "saves" / "final_sweep_v7_datasweepv3" / "optim_search"
    LR_COLORS = ["#1f77b4", "#d62728"]  # blue, red

    text_by_bz = {"16": [], "32": [], "64": []}
    for exp in TEXT_EXPS:
        bz = re.match(r"bz(\d+)", exp).group(1)
        text_by_bz[bz].append(exp)

    vision_by_bz = {"16": [], "32": [], "64": []}
    for exp in VISION_EXPS:
        bz = re.match(r"bz(\d+)", exp).group(1)
        vision_by_bz[bz].append(exp)

    batch_sizes = ["16", "32", "64"]
    modalities = [
        ("text", text_by_bz, "Text"),
        ("vision", vision_by_bz, "Vision"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)

    for row_idx, (modality, by_bz, row_label) in enumerate(modalities):
        for col_idx, bz in enumerate(batch_sizes):
            ax = axes[row_idx, col_idx]
            exps = by_bz[bz]

            for lr_idx, exp_name in enumerate(exps):
                color = LR_COLORS[lr_idx % len(LR_COLORS)]
                lr_str = exp_name.split("_", 1)[1]

                # 6-epoch (solid)
                dir_name = f"{exp_name}_{modality}"
                save_dir = SAVES_BASE / dir_name
                log_history = load_trainer_state(save_dir)
                lr_epochs = []
                lr_values = []
                for entry in log_history:
                    if "learning_rate" in entry and "epoch" in entry:
                        lr_epochs.append(entry["epoch"])
                        lr_values.append(entry["learning_rate"])
                if lr_epochs:
                    ax.plot(lr_epochs, lr_values, color=color, alpha=0.8,
                            linewidth=0.8, label=f"{lr_str} (6ep)")

                # 3-epoch (dashed)
                save_dir_3ep = SAVES_3EP / dir_name
                log_history_3ep = load_trainer_state(save_dir_3ep)
                lr_epochs_3 = []
                lr_values_3 = []
                for entry in log_history_3ep:
                    if "learning_rate" in entry and "epoch" in entry:
                        lr_epochs_3.append(entry["epoch"])
                        lr_values_3.append(entry["learning_rate"])
                if lr_epochs_3:
                    ax.plot(lr_epochs_3, lr_values_3, color=color, alpha=0.6,
                            linewidth=0.8, linestyle="--", label=f"{lr_str} (3ep)")

            ax.set_xlabel("Epoch", fontsize=LABELSIZE)
            ax.tick_params(labelsize=TICKSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGENDSIZE, loc="best")

            if row_idx == 0:
                ax.set_title(f"Batch Size {bz}", fontsize=TITLESIZE)

        axes[row_idx, 0].set_ylabel(f"{row_label}\nLearning Rate", fontsize=LABELSIZE, fontweight="bold")

    fig.suptitle("Optim Search: LR Schedule by Batch Size", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "lr_schedule.png"
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
    print("Plotting loss by batch size (6ep)...")
    plot_loss_by_batchsize()
    print("Plotting loss by batch size (3ep)...")
    plot_loss_by_batchsize_3ep()
    print("Plotting train metrics (3ep)...")
    plot_train_metrics_3ep()
    print("Plotting test metrics (3ep)...")
    plot_test_metrics_3ep()
    print("Plotting LR schedule...")
    plot_lr_schedule()
    print_summary_table(SAVES_3EP, RESULTS_3EP, "Optim Search (3ep)")
    print_summary_table(SAVES_BASE, RESULTS_BASE, "Optim Search (6ep)")
    print("Done.")
