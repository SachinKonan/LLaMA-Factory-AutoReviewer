#!/usr/bin/env python3
"""
Generate status CSV for optim_search_6epochs experiments.

Checks saves/results dirs and SLURM job states to determine status
of each epoch's training, train-inference, and test-inference.

Usage:
    python scripts/populate_optim6epoch_status.py
"""

import json
import os
import re
import subprocess
from pathlib import Path

BASE = Path("saves/final_sweep_v7_datasweepv3/optim_search_6epochs")
RBASE = Path("results/final_sweep_v7_datasweepv3/optim_search_6epochs")
OUT = Path("instructions/optim_status/status.csv")

# Steps per epoch from the 3-epoch runs (dataset size / batch size)
STEPS_PER_EPOCH = {
    "bz16_text": 797,
    "bz16_vision": 798,
    "bz32_text": 399,
    "bz32_vision": 399,
    "bz64_text": 200,
    "bz64_vision": 200,
}

# All configs, sorted: text first then vision, by bz then lr
CONFIGS = [
    "bz16_lr0.5e-6_text",
    "bz16_lr1e-6_text",
    "bz32_lr1e-6_text",
    "bz32_lr2e-6_text",
    "bz64_lr2e-6_text",
    "bz64_lr4e-6_text",
    "bz16_lr1e-6_vision",
    "bz16_lr2e-6_vision",
    "bz32_lr2e-6_vision",
    "bz32_lr4e-6_vision",
    "bz64_lr4e-6_vision",
    "bz64_lr5.5e-6_vision",
]

DONE = "✅"
RUNNING = "🏃"
QUEUED = "⏳"
FAILED = "❌"
BLANK = ""


def get_spe(name: str) -> int:
    """Get steps-per-epoch for a config name."""
    m = re.match(r"(bz\d+)_lr[\d.e\-]+_(text|vision)", name)
    if m:
        return STEPS_PER_EPOCH[f"{m.group(1)}_{m.group(2)}"]
    raise ValueError(f"Can't parse config name: {name}")


def get_squeue_states() -> dict[str, str]:
    """Query squeue for current user's jobs. Returns {jobid: state}."""
    try:
        r = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "sk7524"),
             "-o", "%i %T", "--noheader"],
            capture_output=True, text=True, timeout=10,
        )
        states = {}
        for line in r.stdout.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2:
                states[parts[0]] = parts[1]
        return states
    except Exception:
        return {}


def sacct_state(job_id: str) -> str:
    """Query sacct for a single job's state."""
    try:
        r = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "-n", "--parsable2"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.strip().split("\n"):
            s = line.strip().rstrip("+")
            if s in ("COMPLETED",):
                return DONE
            if s in ("RUNNING",):
                return RUNNING
            if s in ("PENDING",):
                return QUEUED
            if s in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                      "CANCELLED+", "DEADLINE", "NODE_FAIL"):
                return FAILED
        return BLANK
    except Exception:
        return BLANK


def job_emoji(job_id: str, squeue_states: dict[str, str]) -> str:
    """Determine emoji for a job ID using squeue then sacct."""
    if not job_id or job_id == "N/A":
        return BLANK
    # Check squeue first (faster, covers running/pending)
    for qid, qstate in squeue_states.items():
        if job_id in qid or qid.startswith(job_id):
            if qstate == "RUNNING":
                return RUNNING
            if qstate == "PENDING":
                return QUEUED
    # Fall back to sacct
    return sacct_state(job_id)


def main():
    squeue_states = get_squeue_states()

    header = "config,train-job-id"
    for ep in range(1, 7):
        header += f",ep{ep}-train,ep{ep}-train-infer,ep{ep}-test-infer"

    rows = [header]

    for name in CONFIGS:
        sdir = BASE / name
        rdir = RBASE / name
        spe = get_spe(name)

        # -- Train job ID --
        touch = sdir / ".trainjob.touch"
        job_id = touch.read_text().strip() if touch.exists() else "N/A"

        # -- Training complete? --
        train_done = (sdir / "all_results.json").exists()

        # -- Checkpoint steps present --
        ckpt_steps: set[int] = set()
        if sdir.exists():
            for d in sdir.glob("checkpoint-*"):
                try:
                    ckpt_steps.add(int(d.name.split("-")[1]))
                except (IndexError, ValueError):
                    pass

        # -- Highest training step reached (from trainer_state.json) --
        max_train_step = 0
        ts_path = sdir / "trainer_state.json"
        if ts_path.exists():
            try:
                with open(ts_path) as f:
                    state = json.load(f)
                for entry in reversed(state.get("log_history", [])):
                    if "step" in entry:
                        max_train_step = entry["step"]
                        break
            except Exception:
                pass

        # -- Results files --
        train_eval_steps: set[int] = set()
        test_infer_steps: set[int] = set()
        has_finetuned_jsonl = False
        if rdir.exists():
            for f in os.listdir(rdir):
                if f.startswith("train-ckpt-") and f.endswith(".json"):
                    try:
                        step = int(f.replace("train-ckpt-", "").replace(".json", ""))
                        train_eval_steps.add(step)
                    except ValueError:
                        pass
                elif f == "finetuned.jsonl":
                    has_finetuned_jsonl = True
                elif f.startswith("finetuned-ckpt-") and f.endswith(".jsonl"):
                    try:
                        step = int(f.replace("finetuned-ckpt-", "").replace(".jsonl", ""))
                        test_infer_steps.add(step)
                    except ValueError:
                        pass

        # -- Infer job IDs per checkpoint --
        infer_jobs: dict[int, str] = {}
        if sdir.exists():
            for d in sdir.glob("checkpoint-*"):
                infer_touch = d / ".infer.touch"
                if infer_touch.exists():
                    try:
                        step = int(d.name.split("-")[1])
                        infer_jobs[step] = infer_touch.read_text().strip()
                    except (IndexError, ValueError):
                        pass

        # -- Train job emoji --
        if train_done:
            train_emoji = DONE
        elif job_id == "N/A":
            # No touch file — check if queued in squeue by job name
            train_emoji = QUEUED
        else:
            train_emoji = job_emoji(job_id, squeue_states)
            if train_emoji == BLANK:
                train_emoji = QUEUED

        row = f"{name},[{job_id}] {train_emoji}"

        # -- Per-epoch columns --
        for ep in range(1, 7):
            step = spe * ep

            # ep-train: has training reached this epoch?
            if train_done or step in ckpt_steps or max_train_step >= step:
                ep_train = DONE
            elif job_id != "N/A" and train_emoji in (RUNNING, DONE):
                # Training running but hasn't reached this epoch yet
                ep_train = RUNNING
            elif train_emoji == QUEUED:
                ep_train = QUEUED
            else:
                ep_train = BLANK

            # ep-train-infer: train-ckpt-{step}.json exists?
            infer_jid = infer_jobs.get(step, "")
            jid_prefix = f"[{infer_jid}] " if infer_jid else ""
            if step in train_eval_steps:
                ep_train_infer = f"{jid_prefix}{DONE}"
            elif step in infer_jobs:
                ep_train_infer = f"{jid_prefix}{job_emoji(infer_jobs[step], squeue_states)}"
            elif ep_train == DONE:
                # Checkpoint done but no infer submitted/completed yet
                ep_train_infer = QUEUED
            else:
                ep_train_infer = BLANK

            # ep-test-infer: finetuned-ckpt-{step}.jsonl (or finetuned.jsonl for final)?
            if step in test_infer_steps:
                ep_test_infer = f"{jid_prefix}{DONE}"
            elif ep == 6 and has_finetuned_jsonl:
                ep_test_infer = f"{jid_prefix}{DONE}"
            elif step in infer_jobs:
                ep_test_infer = f"{jid_prefix}{job_emoji(infer_jobs[step], squeue_states)}"
            elif ep_train == DONE:
                ep_test_infer = QUEUED
            else:
                ep_test_infer = BLANK

            row += f",{ep_train},{ep_train_infer},{ep_test_infer}"

        rows.append(row)

    csv_content = "\n".join(rows) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(csv_content)
    print(csv_content)


if __name__ == "__main__":
    main()
