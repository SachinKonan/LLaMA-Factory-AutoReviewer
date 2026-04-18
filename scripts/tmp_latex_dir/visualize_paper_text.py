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
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
})

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
DATASET_INFO = ROOT / "data" / "dataset_info.json"
FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_PATH_REG = FONT_DIR / "Roboto-Regular.ttf"
FONT_PATH_BOLD = FONT_DIR / "Roboto-Bold.ttf"

ROWS, COLS = 2, 5
MAX_PANELS = ROWS * COLS

# Page canvas sized to match the 3.0 x 3.9 inch panel aspect (0.769)
PAGE_W, PAGE_H = 620, 806
MARGIN = 28
LINE_GAP = 2             # extra px between lines
BLANK_LINE_FRACTION = 0.55  # blank line = this fraction of a normal line
MIN_FONT = 7
MAX_FONT = 22


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


def wrap_to_pixel_width(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    """Greedy word-wrap against a pixel width using the font's own metrics."""
    lines: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            lines.append("")
            continue
        words = line.split(" ")
        current = ""
        for w in words:
            trial = w if not current else current + " " + w
            if font.getlength(trial) <= max_w:
                current = trial
            else:
                if current:
                    lines.append(current)
                # handle pathological long word — hard-break it
                while font.getlength(w) > max_w and len(w) > 1:
                    # find biggest prefix that fits
                    lo, hi = 1, len(w)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if font.getlength(w[:mid]) <= max_w:
                            lo = mid
                        else:
                            hi = mid - 1
                    lines.append(w[:lo])
                    w = w[lo:]
                current = w
        if current:
            lines.append(current)
    return lines


def wrap_body(body: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    """Wrap body preserving paragraph breaks (blank line at '\\n\\n')."""
    out: list[str] = []
    for i, para in enumerate(body.split("\n\n")):
        if i > 0:
            out.append("")
        if para.strip():
            out.extend(wrap_to_pixel_width(para, font, max_w))
    return out


def layout_height(header_lines: list[str], body_lines: list[str],
                  line_h: int, blank_h: int) -> int:
    """Total pixel height needed to render the full (untruncated) layout."""
    h = 0
    h += line_h * len(header_lines)
    h += blank_h  # separator between header and body
    for ln in body_lines:
        h += blank_h if ln == "" else line_h
    return h


def pick_font_size(header: str, body: str, usable_w: float, usable_h: float) -> int:
    """Binary-search the largest font size where the full section fits."""
    def fits(size: int) -> bool:
        font_reg = ImageFont.truetype(str(FONT_PATH_REG), size)
        font_bold = ImageFont.truetype(str(FONT_PATH_BOLD), size)
        ascent, descent = font_reg.getmetrics()
        line_h = ascent + descent + LINE_GAP
        blank_h = int(round(line_h * BLANK_LINE_FRACTION))
        header_lines = wrap_to_pixel_width(header, font_bold, usable_w) or [header]
        body_lines = wrap_body(body, font_reg, usable_w)
        return layout_height(header_lines, body_lines, line_h, blank_h) <= usable_h

    lo, hi = MIN_FONT, MAX_FONT
    best = MIN_FONT
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render_section_page(header: str, body: str) -> Image.Image:
    usable_w = PAGE_W - 2 * MARGIN
    usable_h = PAGE_H - 2 * MARGIN

    size = pick_font_size(header, body, usable_w, usable_h)
    font_reg = ImageFont.truetype(str(FONT_PATH_REG), size)
    font_bold = ImageFont.truetype(str(FONT_PATH_BOLD), size)
    ascent, descent = font_reg.getmetrics()
    line_h = ascent + descent + LINE_GAP
    blank_h = int(round(line_h * BLANK_LINE_FRACTION))

    header_lines = wrap_to_pixel_width(header, font_bold, usable_w) or [header]
    body_lines = wrap_body(body, font_reg, usable_w)

    # Truncate if even MIN_FONT couldn't fit the whole thing
    max_lines_by_height = usable_h
    remaining = max_lines_by_height - len(header_lines) * line_h - blank_h
    kept_body: list[str] = []
    for ln in body_lines:
        h = blank_h if ln == "" else line_h
        if h > remaining:
            break
        kept_body.append(ln)
        remaining -= h
    if len(kept_body) < len(body_lines) and kept_body:
        kept_body[-1] = kept_body[-1].rstrip() + " …"
    body_lines = kept_body

    img = Image.new("RGB", (PAGE_W, PAGE_H), color="white")
    draw = ImageDraw.Draw(img)
    y = MARGIN
    for ln in header_lines:
        draw.text((MARGIN, y), ln, font=font_bold, fill="black")
        y += line_h
    y += blank_h
    for ln in body_lines:
        if ln == "":
            y += blank_h
        else:
            draw.text((MARGIN, y), ln, font=font_reg, fill="black")
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
