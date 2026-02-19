#!/usr/bin/env python3
"""
Plot training and test metrics for the LR-drop weight decay sweep experiments.

Two LR schedules (rows):
  Row 0 — wd_sweep_2epoch:        2-epoch cosine decay → 2-epoch linear to 0
  Row 1 — wd_sweep_2epoch_3epochexp: 3-epoch cosine (partial, 2 epochs) → drop → 2-epoch linear to 0

Lines: weight decay values (0.001, 0.002, 0.004).  Text only.

Creates three figures:
1. Train metrics (2×8 grid): loss, accuracy, P(Accept|gt=Accept), P(Reject|gt=Accept),
   P(Accept|gt=Reject), P(Reject|gt=Reject), P(Correct), pred positive rate
2. Test metrics (2×7 grid): accuracy, P(Accept|gt=Accept), P(Reject|gt=Accept),
   P(Accept|gt=Reject), P(Reject|gt=Reject), P(Correct), pred positive rate
3. LR schedule (1×2): both schedules side by side

Usage:
    python scripts/plot_wd_sweep_lrdrop.py
"""

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Directories ──────────────────────────────────────────────────────────────
BASE = Path(".")
SAVES_2EP = BASE / "saves" / "final_sweep_v7_datasweepv3" / "wd_sweep_2epoch"
RESULTS_2EP = BASE / "results" / "final_sweep_v7_datasweepv3" / "wd_sweep_2epoch"
SAVES_3EP = BASE / "saves" / "final_sweep_v7_datasweepv3" / "wd_sweep_2epoch_3epochexp"
RESULTS_3EP = BASE / "results" / "final_sweep_v7_datasweepv3" / "wd_sweep_2epoch_3epochexp"
OUTPUT_DIR = BASE / "results" / "final_sweep_v7_datasweepv3" / "wd_sweep_lrdrop"

# ── Experiments ──────────────────────────────────────────────────────────────
WD_VALUES = ["0.001", "0.002", "0.004"]
NUM_EPOCHS = 4

TEXT_EXPS = [f"bz16_lr1e-6_wd{wd}_text" for wd in WD_VALUES]
WD_LABELS = [f"wd={wd}" for wd in WD_VALUES]

# ── Colors ───────────────────────────────────────────────────────────────────
COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
]

# ── Plot styling ─────────────────────────────────────────────────────────────
LABELSIZE = 13
TITLESIZE = 14
LEGENDSIZE = 9
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


def compute_test_metrics_from_jsonl(jsonl_path: Path):
    """Compute test metrics from a JSONL file."""
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
        for line in f:
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


def slope_label(epochs, values, base_label):
    """Add slope annotation to label for accuracy plots."""
    if len(epochs) >= 2:
        coeffs = np.polyfit(epochs, values, 1)
        return f"{base_label} (m={coeffs[0]:.3f})"
    return base_label


def add_peak_textbox(ax, peak_lines: list[tuple[str, str, float, float]]):
    """Add a text box showing peak accuracy per method."""
    lines = []
    for label, color, val, ep in peak_lines:
        lines.append(f"{label}: {val:.3f} @ ep{ep:.0f}")
    text = "\n".join(lines)
    props = dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7)
    ax.text(0.98, 0.02, text, transform=ax.transAxes, fontsize=7.5,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=props, family="monospace")


# ── Figure 1: Train Metrics ─────────────────────────────────────────────────

def plot_train_metrics():
    """Create 2×8 grid of training metrics.
    Row 0: wd_sweep_2epoch (2ep cosine → linear)
    Row 1: wd_sweep_2epoch_3epochexp (3ep cosine partial → drop → linear)
    """
    fig, axes = plt.subplots(2, 8, figsize=(32, 8), squeeze=False)

    schedules = [
        (SAVES_2EP, RESULTS_2EP, "2ep cosine → linear to 0"),
        (SAVES_3EP, RESULTS_3EP, "3ep cosine (2ep) → drop → linear to 0"),
    ]

    train_acc_peaks: dict[int, list] = {0: [], 1: []}

    for row_idx, (saves_base, results_base, row_label) in enumerate(schedules):
        for exp_idx, exp_name in enumerate(TEXT_EXPS):
            save_dir = saves_base / exp_name
            results_dir = results_base / exp_name
            color = COLORS[exp_idx]
            label = WD_LABELS[exp_idx]

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
                        linewidth=0.8, label=label)

            # ── Cols 1-7: Train ckpt metrics ─────────────────────────────
            ckpt_data = load_train_ckpt_metrics(results_dir)
            if ckpt_data:
                ckpt_epochs = [e for e, _ in ckpt_data]

                metric_cols = [
                    (1, "sft_accuracy", "Train Accuracy"),
                    (2, "sft_p_accept_gtaccept_mean", "Train P(Accept|gt=Accept)"),
                    (3, "sft_p_reject_gtaccept_mean", "Train P(Reject|gt=Accept)"),
                    (4, "sft_p_accept_gtreject_mean", "Train P(Accept|gt=Reject)"),
                    (5, "sft_p_reject_gtreject_mean", "Train P(Reject|gt=Reject)"),
                    (6, "sft_p_correct_mean", "Train P(Correct)"),
                    (7, "sft_pred_positive_rate", "Train Pred Positive Rate"),
                ]

                for col, metric_key, _ in metric_cols:
                    ax = axes[row_idx, col]
                    vals = [d.get(metric_key) for _, d in ckpt_data]
                    valid = [(e, v) for e, v in zip(ckpt_epochs, vals)
                             if v is not None]
                    if valid:
                        es, vs = zip(*valid)
                        es, vs = list(es), list(vs)
                        lbl = label
                        if col == 1:  # accuracy: add slope
                            lbl = slope_label(es, vs, label)
                            best_idx = int(np.argmax(vs))
                            train_acc_peaks[row_idx].append(
                                (label, color, vs[best_idx], es[best_idx])
                            )
                        ax.plot(es, vs, "o-", color=color, markersize=4,
                                linewidth=1.5, label=lbl)

    # ── Add peak textboxes to accuracy columns ───────────────────────────
    for row_idx in range(2):
        if train_acc_peaks[row_idx]:
            add_peak_textbox(axes[row_idx, 1], train_acc_peaks[row_idx])

    # ── Formatting ───────────────────────────────────────────────────────
    col_titles = [
        "Train Loss", "Train Accuracy",
        "P(Accept|gt=Accept)", "P(Reject|gt=Accept)",
        "P(Accept|gt=Reject)", "P(Reject|gt=Reject)",
        "Train P(Correct)", "Train Pred Pos Rate",
    ]
    row_labels = [
        "2ep cosine\n→ linear",
        "3ep cosine (2ep)\n→ drop → linear",
    ]

    for col in range(8):
        axes[0, col].set_title(col_titles[col], fontsize=TITLESIZE)
        for row in range(2):
            ax = axes[row, col]
            ax.set_xlabel("Epoch", fontsize=LABELSIZE)
            ax.tick_params(labelsize=TICKSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGENDSIZE, loc="best")
            if col > 0:
                ax.set_xticks(list(range(1, NUM_EPOCHS + 1)))
                ax.set_xlim(0.5, NUM_EPOCHS + 0.5)

    for row in range(2):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=LABELSIZE, fontweight="bold")

    fig.suptitle("WD Sweep (LR Drop): Train Metrics — Text (bz16, lr=1e-6, 4 epochs)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "wd_sweep_lrdrop_train_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure 2: Test Metrics ──────────────────────────────────────────────────

def plot_test_metrics():
    """Create 2×7 grid of test metrics."""
    fig, axes = plt.subplots(2, 7, figsize=(28, 8), squeeze=False)

    schedules = [
        (SAVES_2EP, RESULTS_2EP, "2ep cosine → linear to 0"),
        (SAVES_3EP, RESULTS_3EP, "3ep cosine (2ep) → drop → linear to 0"),
    ]

    test_acc_peaks: dict[int, list] = {0: [], 1: []}

    for row_idx, (saves_base, results_base, row_label) in enumerate(schedules):
        for exp_idx, exp_name in enumerate(TEXT_EXPS):
            save_dir = saves_base / exp_name
            results_dir = results_base / exp_name
            color = COLORS[exp_idx]
            label = WD_LABELS[exp_idx]

            jsonl_files = discover_jsonl_files(results_dir, save_dir)
            if not jsonl_files:
                continue

            test_epochs = []
            test_acc = []
            test_p_accept_gtaccept = []
            test_p_reject_gtaccept = []
            test_p_accept_gtreject = []
            test_p_reject_gtreject = []
            test_p_correct = []
            test_pred_pos = []

            for epoch, jsonl_path in jsonl_files:
                metrics = compute_test_metrics_from_jsonl(jsonl_path)
                if metrics is None:
                    continue
                test_epochs.append(epoch)
                test_acc.append(metrics["accuracy"])
                test_pred_pos.append(metrics["pred_positive_rate"])
                test_p_accept_gtaccept.append(metrics["p_accept_gtaccept_mean"])
                test_p_reject_gtaccept.append(metrics["p_reject_gtaccept_mean"])
                test_p_accept_gtreject.append(metrics["p_accept_gtreject_mean"])
                test_p_reject_gtreject.append(metrics["p_reject_gtreject_mean"])
                test_p_correct.append(metrics["p_correct_mean"])

            metric_series = [
                (0, test_acc, True),
                (1, test_p_accept_gtaccept, False),
                (2, test_p_reject_gtaccept, False),
                (3, test_p_accept_gtreject, False),
                (4, test_p_reject_gtreject, False),
                (5, test_p_correct, False),
                (6, test_pred_pos, False),
            ]

            for col, values, add_slope in metric_series:
                valid = [(e, v) for e, v in zip(test_epochs, values) if v is not None]
                if valid:
                    es, vs = zip(*valid)
                    es, vs = list(es), list(vs)
                    lbl = slope_label(es, vs, label) if add_slope else label
                    axes[row_idx, col].plot(es, vs, "o-", color=color,
                                            markersize=4, linewidth=1.5, label=lbl)
                    if col == 0:  # accuracy: track peak
                        best_idx = int(np.argmax(vs))
                        test_acc_peaks[row_idx].append(
                            (label, color, vs[best_idx], es[best_idx])
                        )

    # ── Add peak textboxes to accuracy columns ───────────────────────────
    for row_idx in range(2):
        if test_acc_peaks[row_idx]:
            add_peak_textbox(axes[row_idx, 0], test_acc_peaks[row_idx])

    # ── Formatting ───────────────────────────────────────────────────────
    col_titles = [
        "Test Accuracy",
        "P(Accept|gt=Accept)", "P(Reject|gt=Accept)",
        "P(Accept|gt=Reject)", "P(Reject|gt=Reject)",
        "Test P(Correct)", "Test Pred Pos Rate",
    ]
    row_labels = [
        "2ep cosine\n→ linear",
        "3ep cosine (2ep)\n→ drop → linear",
    ]

    for col in range(7):
        axes[0, col].set_title(col_titles[col], fontsize=TITLESIZE)
        for row in range(2):
            ax = axes[row, col]
            ax.set_xlabel("Epoch", fontsize=LABELSIZE)
            ax.set_xticks(list(range(1, NUM_EPOCHS + 1)))
            ax.set_xlim(0.5, NUM_EPOCHS + 0.5)
            ax.tick_params(labelsize=TICKSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGENDSIZE, loc="best")

    for row in range(2):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=LABELSIZE, fontweight="bold")

    fig.suptitle("WD Sweep (LR Drop): Test Metrics — Text (bz16, lr=1e-6, 4 epochs)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "wd_sweep_lrdrop_test_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure 3: LR Schedule ───────────────────────────────────────────────────

def plot_lr_schedule():
    """Create 1×2 grid showing LR schedule for both experiment sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)

    panels = [
        (0, SAVES_2EP, "2ep cosine → linear to 0\n(cosine_then_constant, decay_ratio=0.5)"),
        (1, SAVES_3EP, "3ep cosine (2ep) → drop → linear to 0\n(cosine_drop_linear)"),
    ]

    for col_idx, saves_base, title in panels:
        ax = axes[0, col_idx]
        for exp_idx, exp_name in enumerate(TEXT_EXPS):
            save_dir = saves_base / exp_name
            log_history = load_trainer_state(save_dir)
            color = COLORS[exp_idx]
            label = WD_LABELS[exp_idx]

            lr_epochs = []
            lr_values = []
            for entry in log_history:
                if "learning_rate" in entry and "epoch" in entry:
                    lr_epochs.append(entry["epoch"])
                    lr_values.append(entry["learning_rate"])
            if lr_epochs:
                ax.plot(lr_epochs, lr_values, color=color, alpha=0.8,
                        linewidth=0.8, label=label)

        ax.set_title(title, fontsize=TITLESIZE)
        ax.set_xlabel("Epoch", fontsize=LABELSIZE)
        ax.set_ylabel("Learning Rate", fontsize=LABELSIZE)
        ax.tick_params(labelsize=TICKSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGENDSIZE, loc="best")

    fig.suptitle("WD Sweep (LR Drop): LR Schedule Comparison — Text",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "wd_sweep_lrdrop_lr_schedule.png"
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
    print("Plotting LR schedule...")
    plot_lr_schedule()
    print("Done.")
