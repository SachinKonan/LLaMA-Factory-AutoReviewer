#!/usr/bin/env python3
"""Render a paper's page images as a 2x5 grid figure.

Usage:
    uv run python scripts/tmp_latex_dir/visualize_paper.py \\
        --submission_id FhBT596F1X \\
        --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_validation

Outputs:
    tmp_latex_dir/figures/paper_<submission_id>.{png,pdf}

Only the first 10 pages are shown (2 rows × 5 cols, left-to-right).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
DATASET_INFO = ROOT / "data" / "dataset_info.json"

ROWS, COLS = 2, 5
MAX_PAGES = ROWS * COLS


def resolve_data_json(dataset_name: str) -> Path:
    """Look up dataset in dataset_info.json and return its data.json absolute path."""
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


def render(entry: dict, submission_id: str, output_stem: Path) -> None:
    images = entry.get("images") or []
    if not images:
        raise ValueError(f"entry {submission_id!r} has no images")
    if "images" not in entry:
        raise ValueError(f"entry {submission_id!r} is not a vision entry (no images field)")

    images = images[:MAX_PAGES]
    n_shown = len(images)

    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 3.0, ROWS * 3.9))
    axes = axes.flatten()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for idx, ax in enumerate(axes):
        if idx < n_shown:
            img_path = ROOT / images[idx]
            if not img_path.exists():
                ax.text(0.5, 0.5, f"MISSING\n{images[idx]}",
                        ha="center", va="center", fontsize=8, color="red")
            else:
                with Image.open(img_path) as im:
                    ax.imshow(im)
        else:
            ax.axis("off")

    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Pages shown: {n_shown} of {len(entry['images'])} total")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission_id", required=True)
    parser.add_argument(
        "--dataset", required=True,
        help="Registered dataset name (must contain 'vision') from data/dataset_info.json",
    )
    parser.add_argument(
        "--save_name", default=None,
        help="Filename stem (no extension). Default: paper_<submission_id>",
    )
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
