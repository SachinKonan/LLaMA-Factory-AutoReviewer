#!/usr/bin/env python3
"""Render a single paper's panel image (matching the bulk panel dataset).

Same composition logic as scripts/build_panel_images.py:
- Trim white margins from each source page screenshot
- Letterbox-fit into a 476x756 cell (preserves aspect)
- Tile 5 cols x 2 rows on a 2380x1512 canvas
- Save as PNG (and PDF for convenience)

Usage:
    uv run python scripts/tmp_latex_dir/visualize_paper.py \\
        --submission_id FhBT596F1X \\
        --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_validation

Outputs:
    tmp_latex_dir/figures/paper_<submission_id>.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_panel_images import (  # noqa: E402  (path hack required)
    MAX_PAGES,
    TARGET_H,
    TARGET_W,
    compose_panel,
)

OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
DATASET_INFO = ROOT / "data" / "dataset_info.json"

SID_KEYS = ("submission_id", "arxiv_id")


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


def render(entry: dict, submission_id: str, output_stem: Path) -> None:
    images = entry.get("images") or []
    if not images:
        raise ValueError(f"entry {submission_id!r} is not a vision entry (no images)")

    page_paths = [ROOT / p for p in images[:MAX_PAGES]]
    missing = [p for p in page_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing page screenshots: {missing[:3]}")

    panel: Image.Image = compose_panel(page_paths)
    assert panel.size == (TARGET_W, TARGET_H), f"unexpected panel size {panel.size}"

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    panel.save(png_path, "PNG", optimize=True, compress_level=6)
    panel.save(pdf_path, "PDF")

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Pages tiled: {len(page_paths)} of {len(entry['images'])} total  (canvas {TARGET_W}x{TARGET_H})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission_id", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Registered vision dataset name from data/dataset_info.json")
    parser.add_argument("--save_name", default=None,
                        help="Filename stem (no extension). Default: paper_<submission_id>")
    args = parser.parse_args()

    if "vision" not in args.dataset:
        raise ValueError(f"--dataset must be a vision dataset; got {args.dataset!r}")

    data_json = resolve_data_json(args.dataset)
    print(f"Loading {data_json}")
    data = json.loads(data_json.read_text())

    entry = find_entry(data, args.submission_id)
    stem = args.save_name or f"paper_{args.submission_id}"
    render(entry, args.submission_id, OUTPUT_DIR / stem)


if __name__ == "__main__":
    main()
