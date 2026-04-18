#!/usr/bin/env python3
"""Render a paper's markdown sections as a 2x5 grid of text panels.

Usage:
    uv run python scripts/tmp_latex_dir/visualize_paper_text.py \\
        --submission_id FhBT596F1X \\
        --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation

Outputs:
    tmp_latex_dir/figures/paper_text_<submission_id>.{png,pdf}

Each panel is one `# HEADER` section (header bold, body raw markdown).
First 10 sections only, truncated to fit.
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "text.usetex": False,
    "text.parse_math": False,
    "font.family": "monospace",
})

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
DATASET_INFO = ROOT / "data" / "dataset_info.json"

ROWS, COLS = 2, 5
MAX_PANELS = ROWS * COLS
FONT_SIZE = 5
CHARS_PER_LINE = 58
MAX_LINES = 46


def resolve_data_json(dataset_name: str) -> Path:
    info = json.loads(DATASET_INFO.read_text())
    if dataset_name not in info:
        raise KeyError(f"Dataset {dataset_name!r} not registered in {DATASET_INFO}")
    file_name = info[dataset_name]["file_name"]
    candidate = ROOT / "data" / file_name
    if candidate.is_dir():
        candidate = candidate / "data.json"
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def find_entry(data: list[dict], submission_id: str) -> dict:
    for e in data:
        sid = (e.get("_metadata") or {}).get("submission_id")
        if sid == submission_id:
            return e
    raise KeyError(f"submission_id {submission_id!r} not found in dataset")


def wrap_body(body: str, width: int) -> str:
    """Wrap paragraph-by-paragraph, preserving blank-line breaks."""
    out = []
    for para in body.split("\n\n"):
        if not para.strip():
            out.append("")
            continue
        # wrap each physical line separately so list bullets etc. stay aligned
        wrapped_lines = []
        for line in para.split("\n"):
            if not line.strip():
                wrapped_lines.append("")
                continue
            wrapped_lines.append(textwrap.fill(line, width=width, break_long_words=False,
                                                break_on_hyphens=False))
        out.append("\n".join(wrapped_lines))
    return "\n\n".join(out)


def render(entry: dict, output_stem: Path) -> None:
    human = [c for c in entry["conversations"] if c["from"] == "human"][0]["value"]
    m = re.search(r"^# ", human, flags=re.MULTILINE)
    if m is None:
        sections = [human]
    else:
        body = human[m.start():]
        sections = re.split(r"\n(?=# )", body)
    sections = sections[:MAX_PANELS]
    n_shown = len(sections)

    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3.0, ROWS * 3.9))
    axes = axes.flatten()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    line_height_ax = 1.0 / MAX_LINES  # axes-fraction per text line

    for idx, ax in enumerate(axes):
        if idx >= n_shown:
            ax.axis("off")
            continue
        sec = sections[idx]
        first_nl = sec.find("\n")
        if first_nl == -1:
            header, body_text = sec, ""
        else:
            header, body_text = sec[:first_nl], sec[first_nl + 1:]

        wrapped_header = textwrap.fill(header, width=CHARS_PER_LINE,
                                       break_long_words=False, break_on_hyphens=False)
        wrapped_body = wrap_body(body_text, CHARS_PER_LINE)

        # Line budget: header + gap + body
        header_lines = wrapped_header.count("\n") + 1
        body_budget = MAX_LINES - header_lines - 1
        body_lines = wrapped_body.split("\n")
        if len(body_lines) > body_budget:
            body_lines = body_lines[:body_budget]
            body_lines[-1] = body_lines[-1].rstrip() + " …"
        wrapped_body = "\n".join(body_lines)

        ax.text(0.02, 0.98, wrapped_header,
                transform=ax.transAxes, fontsize=FONT_SIZE,
                fontweight="bold", family="monospace",
                verticalalignment="top", horizontalalignment="left")
        body_y = 0.98 - (header_lines + 1) * line_height_ax
        ax.text(0.02, body_y, wrapped_body,
                transform=ax.transAxes, fontsize=FONT_SIZE,
                family="monospace",
                verticalalignment="top", horizontalalignment="left")

    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Sections shown: {n_shown}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission_id", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Registered text dataset from data/dataset_info.json")
    parser.add_argument("--save_name", default=None,
                        help="Filename stem (no extension). Default: paper_text_<submission_id>")
    args = parser.parse_args()

    if "vision" in args.dataset:
        raise ValueError(f"--dataset must be a text dataset; got vision dataset {args.dataset!r}")

    data_json = resolve_data_json(args.dataset)
    print(f"Loading {data_json}")
    data = json.loads(data_json.read_text())

    entry = find_entry(data, args.submission_id)
    stem = args.save_name or f"paper_text_{args.submission_id}"
    render(entry, OUTPUT_DIR / stem)


if __name__ == "__main__":
    main()
