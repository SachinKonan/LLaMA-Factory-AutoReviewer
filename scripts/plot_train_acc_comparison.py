#!/usr/bin/env python3
"""
Side-by-side comparison of two experiments:
  - Old: LR=2e-6, standard cosine, 6 epochs, vision, trainagreeing
  - New: LR=1e-6, cosine_then_constant (decay_ratio=0.5, min_lr_rate=0.001), 6 epochs, text, wd=0.001
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")


# ── LR schedule simulators (from trainer_utils.py) ─────────────────────

def standard_cosine_lr(base_lr, total_steps, warmup_steps):
    lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            mult = step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        lrs.append(base_lr * mult)
    return lrs


def _cosine_then_constant_lambda(current_step, num_warmup_steps, num_decay_steps, num_total_steps, min_lr_rate):
    if current_step < num_warmup_steps:
        return current_step / max(1, num_warmup_steps)
    decay_progress = current_step - num_warmup_steps
    if decay_progress >= num_decay_steps:
        remaining_total = num_total_steps - num_warmup_steps - num_decay_steps
        if remaining_total <= 0:
            return min_lr_rate
        linear_progress = (decay_progress - num_decay_steps) / remaining_total
        return min_lr_rate * (1.0 - min(linear_progress, 1.0))
    cosine_val = 0.5 * (1.0 + math.cos(math.pi * decay_progress / num_decay_steps))
    return min_lr_rate + (1.0 - min_lr_rate) * cosine_val


def cosine_then_constant_lr(base_lr, total_steps, warmup_steps, decay_ratio, min_lr_rate):
    num_decay_steps = int((total_steps - warmup_steps) * decay_ratio)
    lrs = []
    for step in range(total_steps):
        mult = _cosine_then_constant_lambda(step, warmup_steps, num_decay_steps, total_steps, min_lr_rate)
        lrs.append(base_lr * mult)
    return lrs


# ── Data loading ──────────────────────────────────────────────────────

def load_trainer_log(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_train_ckpt_accuracy(results_dir):
    epochs, accs = [], []
    for f in sorted(Path(results_dir).glob("train-ckpt-*.json")):
        d = json.load(open(f))
        epochs.append(d["epoch"])
        accs.append(d.get("sft_accuracy", d.get("cls_accuracy")))
    pairs = sorted(zip(epochs, accs))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def load_test_accuracy(results_dir, steps_per_epoch):
    epochs, accs = [], []
    for f in sorted(Path(results_dir).glob("finetuned-ckpt-*.jsonl")):
        ckpt = int(f.stem.split("-")[-1])
        epoch = ckpt / steps_per_epoch
        correct = total = 0
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                pred = d.get("predict", "")
                label = d.get("label", "")
                if ("Accept" in pred and "Accept" in label) or \
                   ("Reject" in pred and "Reject" in label):
                    correct += 1
                total += 1
        if total > 0:
            epochs.append(epoch)
            accs.append(correct / total)
    pairs = sorted(zip(epochs, accs))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def main():
    # ── Old experiment ──
    # LR=2e-6, standard cosine, 6 epochs, warmup=5, vision, trainagreeing
    old_total_steps = 4278
    old_warmup = 5
    old_base_lr = 2e-6
    old_epochs_total = 6.0
    old_steps_per_epoch = old_total_steps / old_epochs_total  # 713

    old_sim_lrs = standard_cosine_lr(old_base_lr, old_total_steps, old_warmup)
    old_sim_epochs = [s / old_steps_per_epoch for s in range(old_total_steps)]

    old_log = load_trainer_log(BASE / "saves/final_sweep_v7/balanced_trainagreeing_no2024_vision/trainer_log.jsonl")
    old_actual_epochs = [r["epoch"] for r in old_log if "lr" in r]
    old_actual_lrs = [r["lr"] for r in old_log if "lr" in r]
    old_loss_epochs = [r["epoch"] for r in old_log if "loss" in r]
    old_losses = [r["loss"] for r in old_log if "loss" in r]

    old_acc_epochs, old_acc_vals = load_train_ckpt_accuracy(
        BASE / "results/final_sweep_v7/balanced_trainagreeing_no2024_vision"
    )
    old_test_epochs, old_test_accs = load_test_accuracy(
        BASE / "results/final_sweep_v7/balanced_trainagreeing_no2024_vision",
        old_steps_per_epoch,
    )

    # ── New experiment ──
    # LR=1e-6, cosine_then_constant(decay_ratio=0.5, min_lr_rate=0.001), 6 epochs, warmup=5
    # text, balanced original, wd=0.001
    new_total_steps = 4782
    new_warmup = 5
    new_base_lr = 1e-6
    new_epochs_total = 6.0
    new_steps_per_epoch = new_total_steps / new_epochs_total  # 797

    new_sim_lrs = cosine_then_constant_lr(new_base_lr, new_total_steps, new_warmup,
                                           decay_ratio=0.5, min_lr_rate=0.001)
    new_sim_epochs = [s / new_steps_per_epoch for s in range(new_total_steps)]

    new_log = load_trainer_log(BASE / "saves/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text/trainer_log.jsonl")
    new_actual_epochs = [r["epoch"] for r in new_log if "lr" in r]
    new_actual_lrs = [r["lr"] for r in new_log if "lr" in r]
    new_loss_epochs = [r["epoch"] for r in new_log if "loss" in r]
    new_losses = [r["loss"] for r in new_log if "loss" in r]

    new_acc_epochs, new_acc_vals = load_train_ckpt_accuracy(
        BASE / "results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text"
    )
    new_test_epochs, new_test_accs = load_test_accuracy(
        BASE / "results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text",
        new_steps_per_epoch,
    )

    # ── Plot ──
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    BLUE = "tab:blue"
    RED = "tab:red"
    OLD_LABEL = "Old: 2e-6 cosine, 6ep (vision)"
    NEW_LABEL = "New: 1e-6 cos→const, 6ep (text, wd=.001)"

    # Panel 1: LR Schedule
    ax = axes[0]
    ax.plot(old_sim_epochs, old_sim_lrs, label=OLD_LABEL, color=BLUE, linewidth=2)
    ax.plot(new_sim_epochs, new_sim_lrs, label=NEW_LABEL, color=RED, linewidth=2)
    ax.plot(old_actual_epochs, old_actual_lrs, ".", color=BLUE, alpha=0.3, markersize=2)
    ax.plot(new_actual_epochs, new_actual_lrs, ".", color=RED, alpha=0.3, markersize=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("LR Schedule", fontsize=14)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, 6.2)

    # Panel 2: Train Loss
    ax = axes[1]
    ax.plot(old_loss_epochs, old_losses, label=OLD_LABEL, color=BLUE, alpha=0.8, linewidth=1)
    ax.plot(new_loss_epochs, new_losses, label=NEW_LABEL, color=RED, alpha=0.8, linewidth=1)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Train Loss", fontsize=12)
    ax.set_title("Train Loss", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Train Accuracy (post-hoc)
    ax = axes[2]
    ax.plot(old_acc_epochs, old_acc_vals, "o-", label=OLD_LABEL, color=BLUE, markersize=8, linewidth=2)
    ax.plot(new_acc_epochs, new_acc_vals, "s-", label=NEW_LABEL, color=RED, markersize=8, linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Train Accuracy", fontsize=12)
    ax.set_title("Train Accuracy (post-hoc, 2k sample)", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

    # Panel 4: Test Accuracy
    ax = axes[3]
    ax.plot(old_test_epochs, old_test_accs, "o-", label=OLD_LABEL, color=BLUE, markersize=8, linewidth=2)
    ax.plot(new_test_epochs, new_test_accs, "s-", label=NEW_LABEL, color=RED, markersize=8, linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Test Accuracy", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 0.75)

    plt.tight_layout()
    out = BASE / "figures/train_acc_lr_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
