#!/usr/bin/env python3
"""Render a paper's markdown sections as a 2x5 grid of "text pages".

Each panel is a PIL-rendered page image (white canvas, monospace text,
bold `# HEADER` on first line, raw markdown body). Uses matplotlib imshow
identically to the vision variant so the grid visually matches.

Usage:
    uv run python scripts/tmp_latex_dir/visualize_paper_text.py \\
        --submission_id FhBT596F1X \\
        --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation

Outputs:
    tmp_latex_dir/figures/paper_text_<submission_id>.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
DATASET_INFO = ROOT / "data" / "dataset_info.json"

ROWS, COLS = 2, 5
MAX_PANELS = ROWS * COLS

# Page canvas sized to match the 3.0 x 3.9 inch panel aspect (0.769)
PAGE_W, PAGE_H = 620, 806
MARGIN = 28
FONT_PATH_REG = "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf"
FONT_SIZE = 12
LINE_GAP = 3  # extra px between lines


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


def wrap_paragraphs(body: str, width: int) -> list[str]:
    """Wrap body text paragraph-by-paragraph; returns list of lines including blanks."""
    out: list[str] = []
    for i, para in enumerate(body.split("\n\n")):
        if i > 0:
            out.append("")  # paragraph separator
        if not para.strip():
            continue
        for line in para.split("\n"):
            if not line.strip():
                out.append("")
                continue
            wrapped = textwrap.fill(line, width=width,
                                    break_long_words=False, break_on_hyphens=False)
            out.extend(wrapped.split("\n"))
    return out


def render_section_page(header: str, body: str) -> Image.Image:
    font_reg = ImageFont.truetype(FONT_PATH_REG, FONT_SIZE)
    font_bold = ImageFont.truetype(FONT_PATH_BOLD, FONT_SIZE)

    # measure char width (monospace so any char works)
    char_w = font_reg.getlength("M")
    usable_w = PAGE_W - 2 * MARGIN
    usable_h = PAGE_H - 2 * MARGIN
    chars_per_line = max(10, int(usable_w // char_w))

    ascent, descent = font_reg.getmetrics()
    line_h = ascent + descent + LINE_GAP
    max_lines = max(1, usable_h // line_h)

    header_lines = textwrap.wrap(header, width=chars_per_line,
                                 break_long_words=False, break_on_hyphens=False) or [header]
    body_lines = wrap_paragraphs(body, chars_per_line)

    # Budget: header + 1 blank + body, all capped at max_lines
    budget_body = max_lines - len(header_lines) - 1
    if budget_body < 1:
        body_lines = []
    elif len(body_lines) > budget_body:
        body_lines = body_lines[:budget_body]
        if body_lines:
            body_lines[-1] = body_lines[-1].rstrip() + " …"

    img = Image.new("RGB", (PAGE_W, PAGE_H), color="white")
    draw = ImageDraw.Draw(img)
    y = MARGIN
    for line in header_lines:
        draw.text((MARGIN, y), line, font=font_bold, fill="black")
        y += line_h
    y += line_h  # blank line between header and body
    for line in body_lines:
        draw.text((MARGIN, y), line, font=font_reg, fill="black")
        y += line_h
    return img


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
        for spine in ax.spines.values():
            spine.set_visible(False)

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
        page_img = render_section_page(header, body_text)
        ax.imshow(page_img)

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
