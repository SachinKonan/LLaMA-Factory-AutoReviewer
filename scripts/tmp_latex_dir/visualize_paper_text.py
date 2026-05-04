#!/usr/bin/env python3
"""Render a single paper's markdown content as a panel image.

Same canvas + composition as scripts/build_panel_images.py:
- pandoc + xelatex typeset the paper markdown to a letter-size PDF
- pymupdf rasterizes the first 10 pages
- Each page goes through trim_white_margins + letterbox-fit into a
  476x756 cell on a 2380x1512 canvas (5 cols x 2 rows)
- Save as PNG (and PDF for convenience)

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
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_panel_images import (  # noqa: E402  (path hack required)
    MAX_PAGES,
    TARGET_H,
    TARGET_W,
    compose_panel_from_images,
)

OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
DATASET_INFO = ROOT / "data" / "dataset_info.json"
FONT_DIR = Path(__file__).resolve().parent / "fonts"

SID_KEYS = ("submission_id", "arxiv_id")

# Pandoc input format: disable TeX math / raw_tex so $ and \ are escaped, not
# interpreted (the paper markdown uses LaTeX-y notation that won't compile in
# vanilla xelatex without extra packages).
PANDOC_FORMAT = (
    "markdown"
    "-tex_math_dollars"
    "-tex_math_single_backslash"
    "-tex_math_double_backslash"
    "-raw_tex"
    "-raw_html"
    "-latex_macros"
)

LATEX_HEADER = r"""
\usepackage{fontspec}
\setmainfont{Roboto-Regular.ttf}[
  Path = %s/,
  BoldFont = Roboto-Bold.ttf
]
""" % FONT_DIR


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
        meta = e.get("_metadata") or {}
        for k in SID_KEYS:
            if meta.get(k) == submission_id:
                return e
    raise KeyError(f"submission_id {submission_id!r} not found in dataset")


def extract_markdown(entry: dict) -> str:
    """Return the human-prompt content from the first `#` header onward."""
    human = next(c["value"] for c in entry["conversations"] if c["from"] == "human")
    m = re.search(r"^# ", human, flags=re.MULTILINE)
    return human if m is None else human[m.start():]


def md_to_pdf_pages(md: str, dpi: int = 150) -> list[Image.Image]:
    """pandoc -> letter-size PDF, rasterize each page to PIL."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        md_path = td / "doc.md"
        pdf_path = td / "doc.pdf"
        header_path = td / "header.tex"
        md_path.write_text(md)
        header_path.write_text(LATEX_HEADER)

        cmd = [
            "pandoc", "-f", PANDOC_FORMAT, str(md_path), "-o", str(pdf_path),
            "--pdf-engine=xelatex",
            "--include-in-header", str(header_path),
            "-V", "geometry:paperwidth=8.5in",
            "-V", "geometry:paperheight=11in",
            "-V", "geometry:margin=0.6in",
            "-V", "fontsize=10pt",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"pandoc failed (rc={proc.returncode}):\n{proc.stderr[-1500:]}")

        doc = fitz.open(pdf_path)
        try:
            pages: list[Image.Image] = []
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages.append(img)
        finally:
            doc.close()
        return pages


def render(entry: dict, output_stem: Path) -> None:
    md = extract_markdown(entry)
    all_pages = md_to_pdf_pages(md)
    pages = all_pages[:MAX_PAGES]
    panel = compose_panel_from_images(pages)
    assert panel.size == (TARGET_W, TARGET_H), f"unexpected panel size {panel.size}"

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    panel.save(png_path, "PNG", optimize=True, compress_level=6)
    panel.save(pdf_path, "PDF")

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Pages tiled: {len(pages)} of {len(all_pages)} typeset  (canvas {TARGET_W}x{TARGET_H})")


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
