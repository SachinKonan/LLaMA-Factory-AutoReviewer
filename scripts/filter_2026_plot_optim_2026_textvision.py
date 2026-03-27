#!/usr/bin/env python3
"""
Same as plot_optim_2026_textvision.py but EXCLUDES 2026 papers from test metrics.

Builds a set of 2026 paper titles from the test datasets, then filters them out
when computing test metrics from JSONL files. Train metrics are unchanged.

Usage:
    uv run python scripts/filter_2026_plot_optim_2026_textvision.py
"""

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Directories ──────────────────────────────────────────────────────────────
BASE = Path(".")
SWEEP_BASE = BASE / "saves" / "final_sweep_v7_datasweepv3"
RESULTS_BASE = BASE / "results" / "final_sweep_v7_datasweepv3"
OUTPUT_DIR = RESULTS_BASE / "optim_search_2026" / "_plots"

# ── Experiments: (saves_subdir, exp_name, label) ─────────────────────────────
NUM_EPOCHS = 4

EXPS = [
    ("optim_search_2026", "bz32_lr1e-6_text",           "bz32 lr1e-6 text"),
    ("optim_search_2026", "bz16_lr1e-6_vision",         "bz16 lr1e-6 vision"),
    ("optim_search_2026", "bz16_lr1e-6_wd0.001_vision", "bz16 lr1e-6 wd0.001 vision"),
    ("optim_search_2026", "bz32_lr1e-6_wd0.001_text",   "bz32 lr1e-6 wd0.001 text"),
    ("trainagreeing",     "bz16_lr1e-6_vision_2026",    "trainagreeing bz16 lr1e-6 vision"),
    ("trainagreeing",     "bz32_lr1e-6_text_2026",      "trainagreeing bz32 lr1e-6 text"),
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
LEGENDSIZE = 9
TICKSIZE = 10


# ── Build set of 2026 paper titles ──────────────────────────────────────────

def _extract_title_from_content(content: str) -> str:
    """Extract the first # TITLE line (skipping # ABSTRACT) from prompt content."""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("# ") and "ABSTRACT" not in line:
            return line[2:].strip().upper()
    return ""


def build_2026_title_set() -> set:
    """Scan test data.json files to collect all 2026 paper titles."""
    titles = set()
    data_dir = BASE / "data"
    # Check all test dataset dirs that include 2026
    for d in data_dir.iterdir():
        if not d.is_dir() or "2026" not in d.name or "test" not in d.name:
            continue
        data_json = d / "data.json"
        if not data_json.exists():
            continue
        try:
            with open(data_json) as f:
                data = json.load(f)
            for entry in data:
                meta = entry.get("_metadata", {})
                if meta.get("year") != 2026:
                    continue
                for conv in entry.get("conversations", []):
                    role = conv.get("role") or conv.get("from", "")
                    if role in ("user", "human"):
                        content = conv.get("content") or conv.get("value", "")
                        title = _extract_title_from_content(content)
                        if title:
                            titles.add(title)
                        break
        except Exception:
            continue
    return titles


def extract_title_from_prompt(prompt: str) -> str:
    """Extract paper title from a JSONL entry's prompt field."""
    return _extract_title_from_content(prompt)


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


def compute_test_metrics_from_jsonl(jsonl_path: Path, exclude_titles: set):
    """Compute test metrics from a JSONL file, excluding papers whose titles are in exclude_titles."""
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
    n_filtered = 0

    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)

            # Filter out 2026 papers
            prompt = entry.get("prompt", "")
            title = extract_title_from_prompt(prompt)
            if title and title in exclude_titles:
                n_filtered += 1
                continue

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
        "n_filtered": n_filtered,
        "n_kept": len(predictions),
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


# ── Figure 1: Train Metrics (unchanged) ─────────────────────────────────────

def plot_train_metrics():
    """Create 1×6 grid of training metrics."""
    fig, axes = plt.subplots(1, 6, figsize=(24, 5), squeeze=False)

    train_acc_peaks = []

    for exp_idx, (subdir, exp_name, exp_label) in enumerate(EXPS):
        save_dir = SWEEP_BASE / subdir / exp_name
        results_dir = RESULTS_BASE / subdir / exp_name
        color = COLORS[exp_idx]

        # ── Col 0: Train Loss ────────────────────────────────────────
        ax = axes[0, 0]
        log_history = load_trainer_state(save_dir)
        loss_epochs = []
        loss_values = []
        for entry in log_history:
            if "loss" in entry and "epoch" in entry and "train_loss" not in entry:
                loss_epochs.append(entry["epoch"])
                loss_values.append(entry["loss"])
        if loss_epochs:
            ax.plot(loss_epochs, loss_values, color=color, alpha=0.8,
                    linewidth=0.8, label=exp_label)

        # ── Cols 1-5: Train ckpt metrics ─────────────────────────────
        ckpt_data = load_train_ckpt_metrics(results_dir)
        if ckpt_data:
            ckpt_epochs = [e for e, _ in ckpt_data]

            metric_cols = [
                (1, "sft_accuracy", "Train Accuracy"),
                (2, "sft_p_accept_gtaccept_mean", "Train P(Accept|gt=Accept)"),
                (3, "sft_p_reject_gtaccept_mean", "Train P(Reject|gt=Accept)"),
                (4, "sft_p_accept_gtreject_mean", "Train P(Accept|gt=Reject)"),
                (5, "sft_p_reject_gtreject_mean", "Train P(Reject|gt=Reject)"),
            ]

            for col, metric_key, _ in metric_cols:
                ax = axes[0, col]
                vals = [d.get(metric_key) for _, d in ckpt_data]
                valid = [(e, v) for e, v in zip(ckpt_epochs, vals)
                         if v is not None]
                if valid:
                    es, vs = zip(*valid)
                    es, vs = list(es), list(vs)
                    lbl = exp_label
                    if col == 1:  # accuracy: add slope
                        lbl = slope_label(es, vs, exp_label)
                        best_idx = int(np.argmax(vs))
                        train_acc_peaks.append(
                            (exp_label, color, vs[best_idx], es[best_idx])
                        )
                    ax.plot(es, vs, "o-", color=color, markersize=4,
                            linewidth=1.5, label=lbl)

    # ── Add peak textbox to accuracy column ──────────────────────────
    if train_acc_peaks:
        add_peak_textbox(axes[0, 1], train_acc_peaks)

    # ── Formatting ───────────────────────────────────────────────────
    col_titles = [
        "Train Loss", "Train Accuracy",
        "P(Accept|gt=Accept)", "P(Reject|gt=Accept)",
        "P(Accept|gt=Reject)", "P(Reject|gt=Reject)",
    ]

    for col in range(6):
        ax = axes[0, col]
        ax.set_title(col_titles[col], fontsize=TITLESIZE)
        ax.set_xlabel("Epoch", fontsize=LABELSIZE)
        ax.tick_params(labelsize=TICKSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGENDSIZE, loc="best")
        if col > 0:
            ax.set_xticks(list(range(1, NUM_EPOCHS + 1)))
            ax.set_xlim(0.5, NUM_EPOCHS + 0.5)

    fig.suptitle("optim_search_2026 (no 2026 in test): Train Metrics — Text vs Vision",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "filter_2026_optim_2026_textvision_train_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure 2: Test Metrics (filtered) ───────────────────────────────────────

def plot_test_metrics(exclude_titles: set):
    """Create 1×5 grid of test metrics, excluding 2026 papers."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), squeeze=False)

    test_acc_peaks = []

    for exp_idx, (subdir, exp_name, exp_label) in enumerate(EXPS):
        save_dir = SWEEP_BASE / subdir / exp_name
        results_dir = RESULTS_BASE / subdir / exp_name
        color = COLORS[exp_idx]

        jsonl_files = discover_jsonl_files(results_dir, save_dir)
        if not jsonl_files:
            continue

        test_epochs = []
        test_acc = []
        test_p_accept_gtaccept = []
        test_p_reject_gtaccept = []
        test_p_accept_gtreject = []
        test_p_reject_gtreject = []

        for epoch, jsonl_path in jsonl_files:
            metrics = compute_test_metrics_from_jsonl(jsonl_path, exclude_titles)
            if metrics is None:
                continue
            test_epochs.append(epoch)
            test_acc.append(metrics["accuracy"])
            test_p_accept_gtaccept.append(metrics["p_accept_gtaccept_mean"])
            test_p_reject_gtaccept.append(metrics["p_reject_gtaccept_mean"])
            test_p_accept_gtreject.append(metrics["p_accept_gtreject_mean"])
            test_p_reject_gtreject.append(metrics["p_reject_gtreject_mean"])

            if epoch == test_epochs[0]:  # print filter stats once per experiment
                print(f"  {exp_label} ep{epoch}: kept {metrics['n_kept']}, "
                      f"filtered {metrics['n_filtered']} (2026)")

        metric_series = [
            (0, test_acc, True),
            (1, test_p_accept_gtaccept, False),
            (2, test_p_reject_gtaccept, False),
            (3, test_p_accept_gtreject, False),
            (4, test_p_reject_gtreject, False),
        ]

        for col, values, add_slope in metric_series:
            valid = [(e, v) for e, v in zip(test_epochs, values) if v is not None]
            if valid:
                es, vs = zip(*valid)
                es, vs = list(es), list(vs)
                lbl = slope_label(es, vs, exp_label) if add_slope else exp_label
                axes[0, col].plot(es, vs, "o-", color=color,
                                  markersize=4, linewidth=1.5, label=lbl)
                if col == 0:  # accuracy: track peak
                    best_idx = int(np.argmax(vs))
                    test_acc_peaks.append(
                        (exp_label, color, vs[best_idx], es[best_idx])
                    )

    # ── Add peak textbox to accuracy column ──────────────────────────
    if test_acc_peaks:
        add_peak_textbox(axes[0, 0], test_acc_peaks)

    # ── Formatting ───────────────────────────────────────────────────
    col_titles = [
        "Test Accuracy",
        "P(Accept|gt=Accept)", "P(Reject|gt=Accept)",
        "P(Accept|gt=Reject)", "P(Reject|gt=Reject)",
    ]

    for col in range(5):
        ax = axes[0, col]
        ax.set_title(col_titles[col], fontsize=TITLESIZE)
        ax.set_xlabel("Epoch", fontsize=LABELSIZE)
        ax.set_xticks(list(range(1, NUM_EPOCHS + 1)))
        ax.set_xlim(0.5, NUM_EPOCHS + 0.5)
        ax.tick_params(labelsize=TICKSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGENDSIZE, loc="best")

    fig.suptitle("optim_search_2026 (no 2026 in test): Test Metrics — Text vs Vision",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "filter_2026_optim_2026_textvision_test_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Figure 3: LR Schedule (unchanged) ───────────────────────────────────────

def plot_lr_schedule():
    """Create 1×1 LR schedule plot for all experiments."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for exp_idx, (subdir, exp_name, exp_label) in enumerate(EXPS):
        save_dir = SWEEP_BASE / subdir / exp_name
        log_history = load_trainer_state(save_dir)
        color = COLORS[exp_idx]

        lr_epochs = []
        lr_values = []
        for entry in log_history:
            if "learning_rate" in entry and "epoch" in entry:
                lr_epochs.append(entry["epoch"])
                lr_values.append(entry["learning_rate"])
        if lr_epochs:
            ax.plot(lr_epochs, lr_values, color=color, alpha=0.8,
                    linewidth=0.8, label=exp_label)

    ax.set_title("LR Schedule", fontsize=TITLESIZE)
    ax.set_xlabel("Epoch", fontsize=LABELSIZE)
    ax.set_ylabel("Learning Rate", fontsize=LABELSIZE)
    ax.tick_params(labelsize=TICKSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGENDSIZE, loc="best")

    fig.suptitle("optim_search_2026: LR Schedule — Text vs Vision",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "filter_2026_optim_2026_textvision_lr_schedule.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building 2026 title set...")
    titles_2026 = build_2026_title_set()
    print(f"Found {len(titles_2026)} unique 2026 paper titles to exclude")

    print("Plotting train metrics...")
    plot_train_metrics()
    print("Plotting test metrics (excluding 2026)...")
    plot_test_metrics(titles_2026)
    print("Plotting LR schedule...")
    plot_lr_schedule()
    print("Done.")
