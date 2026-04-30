#!/usr/bin/env python3
"""Build the panel-image vision dataset from existing page screenshots.

For each paper in the vision labelfix v7 filtered24480 dataset
(train/val/test), take its existing page PNGs (already rendered to
data/images/<sid>/page_N_noreferences_original.png by an earlier
extraction pass) and compose the first 10 of them into a single
2240 x 1148 PNG (5 cols x 2 rows of 448 x 574 px panels).

Output is flat: data/images_panel/<sid>.png (one file per paper,
~24,888 total). Sized so Qwen2.5-VL smart_resize is a no-op at
max_pixels=4014080 (3,280 vision tokens / paper).

Run on srun with many workers; skip-if-exists makes the run resumable.

Usage:
    uv run python scripts/build_panel_images.py --workers 32
    # smoke test:
    uv run python scripts/build_panel_images.py --splits validation --limit 5 --workers 1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = ROOT / "data" / "images_panel"

TARGET_W, TARGET_H = 2100, 1344  # cell aspect 0.625; ~12% horizontal margin vs ICLR content (0.55)
ROWS, COLS = 2, 5
PANEL_W, PANEL_H = TARGET_W // COLS, TARGET_H // ROWS  # 420 x 672
MAX_PAGES = ROWS * COLS  # 10
CELL_PADDING = 18  # px of white space on every side of each cell -> visible gutter between pages

# Vision dataset whose `images` lists already point at per-page PNGs
VISION_BASE = "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480"
SPLIT_DATASETS = {
    "train": f"{VISION_BASE}_train",
    "validation": f"{VISION_BASE}_validation",
    "test": f"{VISION_BASE}_test",
}


def trim_white_margins(im: Image.Image, tol: int = 8) -> Image.Image:
    """Crop solid-white margins from the page.

    Uses ImageChops.difference vs a pure-white background, thresholds
    near-white pixels (anti-aliased text edges) at `tol`, then takes
    the bounding box of the remaining content.
    """
    rgb = im.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg).convert("L")
    if tol > 0:
        diff = diff.point(lambda p: 255 if p > tol else 0)
    bbox = diff.getbbox()
    return rgb.crop(bbox) if bbox else rgb


def fit_into_cell(im: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    """Resize `im` to fit inside (cell_w, cell_h) preserving aspect, on white."""
    cw, ch = im.size
    scale = min(cell_w / cw, cell_h / ch)
    new_w, new_h = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    thumb = im.resize((new_w, new_h), Image.LANCZOS)
    out = Image.new("RGB", (cell_w, cell_h), "white")
    out.paste(thumb, ((cell_w - new_w) // 2, (cell_h - new_h) // 2))
    return out


def compose_panel(page_paths: list[Path]) -> Image.Image:
    """Trim each page, fit into the inner cell area (with CELL_PADDING gutters), tile."""
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), "white")
    inner_w = PANEL_W - 2 * CELL_PADDING
    inner_h = PANEL_H - 2 * CELL_PADDING
    for i, p in enumerate(page_paths[:MAX_PAGES]):
        with Image.open(p) as src:
            trimmed = trim_white_margins(src)
        cell = fit_into_cell(trimmed, inner_w, inner_h)
        row, col = divmod(i, COLS)
        x = col * PANEL_W + CELL_PADDING
        y = row * PANEL_H + CELL_PADDING
        canvas.paste(cell, (x, y))
    return canvas


def render_one(args: tuple[str, list[str], Path]) -> tuple[str, str]:
    sid, image_rels, dst = args
    if dst.exists():
        return sid, "skipped"
    try:
        page_paths = [ROOT / r for r in image_rels[:MAX_PAGES]]
        missing = [p for p in page_paths if not p.exists()]
        if missing:
            return sid, f"FAIL: missing pages: {[str(m.name) for m in missing[:3]]}"
        img = compose_panel(page_paths)
        tmp = dst.with_suffix(".png.tmp")
        img.save(tmp, "PNG", optimize=True, compress_level=6)
        os.rename(tmp, dst)
        return sid, "ok"
    except Exception as e:  # noqa: BLE001
        return sid, f"FAIL: {type(e).__name__}: {e}"


def collect_tasks(splits: list[str], limit: int | None) -> list[tuple[str, list[str], Path]]:
    tasks: list[tuple[str, list[str], Path]] = []
    seen: set[str] = set()
    for split in splits:
        ds_path = ROOT / "data" / SPLIT_DATASETS[split] / "data.json"
        entries = json.loads(ds_path.read_text())
        if limit is not None:
            entries = entries[:limit]
        for entry in entries:
            sid = entry["_metadata"]["submission_id"]
            if sid in seen:
                continue
            seen.add(sid)
            tasks.append((sid, entry["images"], PANEL_DIR / f"{sid}.png"))
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
    print(f"{len(tasks)} unique papers across {args.splits}; workers={args.workers}")

    fails: list[tuple[str, str]] = []
    started_at = time.time()
    skipped = ok = 0
    pool = None
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
        if i % 500 == 0 or i == len(tasks):
            elapsed = time.time() - started_at
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(tasks)}  ok={ok} skipped={skipped} fail={len(fails)}  "
                  f"rate={rate:.2f}/s  eta={eta/60:.1f}min")

    if pool is not None:
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
