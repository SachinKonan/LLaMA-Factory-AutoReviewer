#!/usr/bin/env python3
"""
Format experiment data into Notion-flavored Markdown tables.
Reads JSON output from gather_notion_data.py.
"""

import json
import re
import sys


def status_icon(done, running=False):
    if done:
        return "\u2705"
    if running:
        return "\u23f3"
    return "\u274c"


def fmt_time(h):
    if h is None:
        return ""
    return f"{h}h"


def fmt_cell_train(ckpt_data):
    """Format Train cell: status/time"""
    done = ckpt_data.get("train_done", False)
    time_h = ckpt_data.get("train_time_h")
    icon = status_icon(done)
    t = fmt_time(time_h)
    parts = [icon]
    if t:
        parts.append(t)
    return "/".join(parts)


def fmt_cell_train_infer(ckpt_data):
    """Format Train-Infer cell: status/time"""
    done = ckpt_data.get("train_infer_done", False)
    time_h = ckpt_data.get("train_infer_time_h")
    icon = status_icon(done)
    t = fmt_time(time_h)
    parts = [icon]
    if t:
        parts.append(t)
    return "/".join(parts)


def fmt_cell_test_infer(ckpt_data):
    """Format Test-Infer cell: job-id/status/time"""
    job_id = ckpt_data.get("test_infer_job_id", "")
    done = ckpt_data.get("test_infer_done", False)
    running = ckpt_data.get("test_infer_running", False)
    time_h = ckpt_data.get("test_infer_time_h")
    icon = status_icon(done, running)
    parts = []
    if job_id:
        parts.append(str(job_id))
    parts.append(icon)
    t = fmt_time(time_h)
    if t:
        parts.append(t)
    return "/".join(parts)


def get_variant_epochs(variant_data):
    """Get checkpoints as epoch list (sorted by step, 1-indexed)."""
    ckpts = variant_data.get("checkpoints", {})
    sorted_steps = sorted(ckpts.keys(), key=int)
    epochs = []
    for i, step in enumerate(sorted_steps, 1):
        epochs.append({"epoch": i, "step": int(step), "data": ckpts[step]})
    return epochs


def build_status_table(group_data):
    """Build Notion markdown status table for an experiment group.
    Uses epoch-based columns instead of step-based to handle different batch sizes.
    """
    variants = group_data["variants"]
    if not variants:
        return "No variants found."

    # Determine max number of epochs across all variants
    max_epochs = max(len(v["checkpoints"]) for v in variants) if variants else 0

    if max_epochs == 0:
        return "No checkpoints found."

    # Build header row
    header_cells = ["<td>Variant</td>", "<td>Train Job</td>"]
    for ep in range(1, max_epochs + 1):
        header_cells.append(f"<td>Ep{ep} Train</td>")
        header_cells.append(f"<td>Ep{ep} TrainInf</td>")
        header_cells.append(f"<td>Ep{ep} TestInf</td>")

    lines = ['<table fit-page-width="true" header-row="true">']
    lines.append("<tr>")
    for cell in header_cells:
        lines.append(cell)
    lines.append("</tr>")

    # Build data rows
    for v in variants:
        epochs = get_variant_epochs(v)
        row_cells = [f"<td>`{v['variant']}`</td>"]
        row_cells.append(f"<td>{v.get('train_job_id', '-')}</td>")

        for ep_idx in range(max_epochs):
            if ep_idx < len(epochs):
                ckpt = epochs[ep_idx]["data"]
                row_cells.append(f"<td>{fmt_cell_train(ckpt)}</td>")
                row_cells.append(f"<td>{fmt_cell_train_infer(ckpt)}</td>")
                row_cells.append(f"<td>{fmt_cell_test_infer(ckpt)}</td>")
            else:
                row_cells.append("<td>-</td>")
                row_cells.append("<td>-</td>")
                row_cells.append("<td>-</td>")

        lines.append("<tr>")
        for cell in row_cells:
            lines.append(cell)
        lines.append("</tr>")

    lines.append("</table>")
    return "\n".join(lines)


def build_per_year_table(group_data):
    """Build Notion markdown per-year results table for an experiment group.
    One row per variant+checkpoint, columns for overall + each year.
    """
    per_year = group_data.get("per_year_results", {})
    if not per_year:
        return "*No test inference results available yet.*"

    # Determine all years across all variants/checkpoints
    all_years = set()
    for variant_name, ckpt_results in per_year.items():
        for step, year_data in ckpt_results.items():
            all_years.update(k for k in year_data.keys() if k != "overall")
    all_years = sorted(all_years)

    if not all_years:
        return "*No per-year data available.*"

    # Build header
    header_cells = ["<td>Variant</td>", "<td>Epoch</td>", "<td>Overall</td>"]
    for year in all_years:
        header_cells.append(f"<td>{year}</td>")

    lines = ['<table fit-page-width="true" header-row="true">']
    lines.append("<tr>")
    for cell in header_cells:
        lines.append(cell)
    lines.append("</tr>")

    # Build data rows - sorted by variant then checkpoint
    # Also find best overall accuracy per variant to bold it
    for variant_name in sorted(per_year.keys()):
        ckpt_results = per_year[variant_name]
        sorted_steps = sorted(ckpt_results.keys(), key=int)

        # Find best epoch for this variant
        best_acc = -1
        best_step = None
        for step in sorted_steps:
            acc = ckpt_results[step].get("overall", {}).get("accuracy")
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_step = step

        for i, step in enumerate(sorted_steps, 1):
            year_data = ckpt_results[step]
            overall_acc = year_data.get("overall", {}).get("accuracy", "-")

            is_best = (step == best_step and overall_acc != "-")
            overall_str = f"**{overall_acc}**" if is_best else str(overall_acc)

            row_cells = [f"<td>`{variant_name}`</td>", f"<td>{i}</td>"]
            row_cells.append(f"<td>{overall_str}</td>")

            for year in all_years:
                acc = year_data.get(str(year), {}).get("accuracy", "-")
                acc_str = f"**{acc}**" if is_best and acc != "-" else str(acc)
                row_cells.append(f"<td>{acc_str}</td>")

            # Color best row
            if is_best:
                lines.append('<tr color="green_bg">')
            else:
                lines.append("<tr>")
            for cell in row_cells:
                lines.append(cell)
            lines.append("</tr>")

    lines.append("</table>")
    return "\n".join(lines)


def main():
    # Read the full output from gather_notion_data.py
    input_text = sys.stdin.read()

    # Parse each experiment group's JSON
    groups = {}
    for match in re.finditer(r"=== (\S+) ===\n(\{.*?\})\n\n", input_text, re.DOTALL):
        group_name = match.group(1)
        try:
            group_data = json.loads(match.group(2))
            groups[group_name] = group_data
        except json.JSONDecodeError as e:
            print(f"Error parsing {group_name}: {e}", file=sys.stderr)

    # Output formatted tables for each group
    for group_name, group_data in groups.items():
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {group_name}")
        print(f"{'='*80}")

        print("\n--- STATUS TABLE ---")
        print(build_status_table(group_data))

        print("\n--- PER-YEAR RESULTS TABLE ---")
        print(build_per_year_table(group_data))
        print()


if __name__ == "__main__":
    main()
