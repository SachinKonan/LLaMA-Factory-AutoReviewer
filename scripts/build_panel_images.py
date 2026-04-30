#!/usr/bin/env python3
"""Build the panel-image vision dataset.

For each paper in the text labelfix v7 filtered dataset (train/val/test),
typeset its markdown (pandoc + xelatex + Roboto), rasterize the first 10
PDF pages, and compose them into a 2240 x 1148 PNG (5 cols x 2 rows of
448 x 574 px panels). Output is written flat to data/images_panel/<sid>.png.

Run on srun with many workers (xelatex is single-threaded but parallelizes
well across processes). Skip-if-exists makes the job fully resumable.

Usage:
    uv run python scripts/build_panel_images.py --workers 32
    # smoke test:
    uv run python scripts/build_panel_images.py --splits validation --limit 5 --workers 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

# Reuse helpers from the single-paper viewer
sys.path.insert(0, str(Path(__file__).resolve().parent / "tmp_latex_dir"))
from visualize_paper_text import md_to_pdf_pages, extract_markdown  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = ROOT / "data" / "images_panel"

TARGET_W, TARGET_H = 2240, 1148  # Qwen2.5-VL smart_resize no-op @ max_pixels=4014080
ROWS, COLS = 2, 5
PANEL_W, PANEL_H = TARGET_W // COLS, TARGET_H // ROWS  # 448 x 574

BASE = "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
SPLIT_DATASETS = {
    "train": f"{BASE}_train",
    "validation": f"{BASE}_validation",
    "test": f"{BASE}_test",
}


def compose_panel(pages: list[Image.Image]) -> Image.Image:
    """Assemble up to ROWS*COLS pages into a TARGET_W x TARGET_H white canvas."""
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), "white")
    for i, page in enumerate(pages[: ROWS * COLS]):
        row, col = divmod(i, COLS)
        thumb = page.resize((PANEL_W, PANEL_H), Image.LANCZOS)
        canvas.paste(thumb, (col * PANEL_W, row * PANEL_H))
    return canvas


def render_one(args: tuple[str, dict, Path]) -> tuple[str, str]:
    sid, entry, dst = args
    if dst.exists():
        return sid, "skipped"
    try:
        md = extract_markdown(entry)
        pages = md_to_pdf_pages(md)
        if not pages:
            return sid, "FAIL: pandoc produced 0 pages"
        img = compose_panel(pages)
        tmp = dst.with_suffix(".png.tmp")
        img.save(tmp, "PNG", optimize=True, compress_level=6)
        os.rename(tmp, dst)
        return sid, "ok"
    except Exception as e:  # noqa: BLE001 — we deliberately swallow per-paper failures
        return sid, f"FAIL: {type(e).__name__}: {e}"


def collect_tasks(splits: list[str], limit: int | None) -> list[tuple[str, dict, Path]]:
    tasks: list[tuple[str, dict, Path]] = []
    for split in splits:
        ds_path = ROOT / "data" / SPLIT_DATASETS[split] / "data.json"
        entries = json.loads(ds_path.read_text())
        if limit is not None:
            entries = entries[:limit]
        for entry in entries:
            sid = entry["_metadata"]["submission_id"]
            tasks.append((sid, entry, PANEL_DIR / f"{sid}.png"))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                    choices=list(SPLIT_DATASETS))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap entries per split (debug)")
    args = ap.parse_args()

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    tasks = collect_tasks(args.splits, args.limit)
    print(f"{len(tasks)} papers across {args.splits}; workers={args.workers}")

    fails: list[tuple[str, str]] = []
    started_at = time.time()
    skipped = ok = 0
    if args.workers <= 1:
        results_iter = (render_one(t) for t in tasks)
    else:
        pool = Pool(args.workers)
        results_iter = pool.imap_unordered(render_one, tasks, chunksize=4)

    for i, (sid, status) in enumerate(results_iter, 1):
        if status == "ok":
            ok += 1
        elif status == "skipped":
            skipped += 1
        else:
            fails.append((sid, status))
        if i % 200 == 0 or i == len(tasks):
            elapsed = time.time() - started_at
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(tasks)}  ok={ok} skipped={skipped} fail={len(fails)}  "
                  f"rate={rate:.2f}/s  eta={eta/60:.1f}min")

    if args.workers > 1:
        pool.close()
        pool.join()

    print(f"\nDONE  total={len(tasks)}  ok={ok}  skipped={skipped}  fail={len(fails)}  "
          f"elapsed={(time.time()-started_at)/60:.1f}min")
    if fails:
        print("First 20 failures:")
        for sid, status in fails[:20]:
            print(f"  {sid}: {status}")


if __name__ == "__main__":
    main()
