#!/usr/bin/env python3
"""Build the text+panel multimodal dataset.

For each split, join the text-labelfix-v7-filtered dataset with the panel
PNGs by submission_id. For each match, append ' <image>' to the last user
turn and set images=[data/images_panel/<sid>.png]. Skips entries whose
panel PNG isn't present.

Outputs:
    data/iclr_..._text_panel_{train,validation,test}/data.json

Then registers the three new dataset names in data/dataset_info.json,
copying the columns/tags from the source vision (panel) entry so the
sharegpt loader knows how to map the `images` field.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PANEL_DIR = DATA / "images_panel"
DATASET_INFO = DATA / "dataset_info.json"

TEXT_BASE = "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
PANEL_BASE = "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_panel"
OUT_BASE = "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_panel_v7_filtered"
SPLITS = ["train", "validation", "test"]


def panel_relpath(sid: str) -> str:
    return f"data/images_panel/{sid}.png"


def transform_entry(entry: dict) -> dict | None:
    sid = entry["_metadata"]["submission_id"]
    panel_path = PANEL_DIR / f"{sid}.png"
    if not panel_path.exists():
        return None
    out = dict(entry)
    new_convs = []
    last_human_idx = max(
        (i for i, c in enumerate(entry["conversations"]) if c["from"] == "human"),
        default=-1,
    )
    if last_human_idx == -1:
        return None
    for i, c in enumerate(entry["conversations"]):
        if i == last_human_idx:
            new_convs.append({**c, "value": c["value"].rstrip() + "\n\n<image>"})
        else:
            new_convs.append(c)
    out["conversations"] = new_convs
    out["images"] = [panel_relpath(sid)]
    return out


def build_split(split: str) -> tuple[int, int]:
    src = DATA / f"{TEXT_BASE}_{split}" / "data.json"
    dst_dir = DATA / f"{OUT_BASE}_{split}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "data.json"

    out_entries = []
    skipped = 0
    for e in json.loads(src.read_text()):
        t = transform_entry(e)
        if t is None:
            skipped += 1
        else:
            out_entries.append(t)

    dst.write_text(json.dumps(out_entries))
    print(f"  {split}: wrote {len(out_entries)} entries to {dst.relative_to(ROOT)} (skipped: {skipped})")
    return len(out_entries), skipped


def register_in_dataset_info() -> None:
    info = json.loads(DATASET_INFO.read_text())
    template = info[f"{PANEL_BASE}_train"]
    for split in SPLITS:
        name = f"{OUT_BASE}_{split}"
        entry = dict(template)
        entry["file_name"] = f"{OUT_BASE}_{split}/data.json"
        info[name] = entry
        print(f"  registered: {name}")
    DATASET_INFO.write_text(json.dumps(info, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--skip_register", action="store_true")
    args = ap.parse_args()

    print("Building text+panel data.json files:")
    totals = {}
    for split in args.splits:
        n, sk = build_split(split)
        totals[split] = (n, sk)

    if not args.skip_register:
        print("\nRegistering in dataset_info.json:")
        register_in_dataset_info()

    print("\nSummary:")
    for split, (n, sk) in totals.items():
        print(f"  {split}: {n} entries (skipped no-panel: {sk})")


if __name__ == "__main__":
    main()
