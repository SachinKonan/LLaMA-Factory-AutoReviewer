#!/usr/bin/env python3
"""
Build an examples/ directory with one sample per conference/year/label/modality.

Structure:
  examples/{conf}/{year}/{accept|reject}/text/example.md
  examples/{conf}/{year}/{accept|reject}/vision/page_1.png -> symlink to data/images*/...

Usage:
    uv run scripts/build_examples_dir.py
"""

import json
import os
import random
import re
from pathlib import Path

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"

# Prompt header to strip for the text example
PROMPT_HEADER_RE = re.compile(
    r"^I am giving you a paper\..*?acceptance rate\s*\n+",
    re.DOTALL,
)

# Text datasets: (conference_label, dataset_name, years)
TEXT_DATASETS = [
    ("iclr", "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train",
     [2020, 2021, 2022, 2023, 2025, 2026]),
    ("nips", "nips_2021_2025_original_text_v7_train",
     [2021, 2022, 2023, 2024, 2025]),
    ("icml", "icml_colm_2024_2025_clean_binary_train",
     [2024, 2025]),  # will split ICML vs COLM by conference field
]

# Vision datasets (parallel to text)
VISION_DATASETS = [
    ("iclr", "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train"),
    ("nips", "nips_2021_2025_original_vision_v7_train"),
    ("icml", "icml_colm_2024_2025_vision_binary_train"),
]


def load_dataset(name: str) -> list[dict]:
    path = PROJECT_ROOT / "data" / name / "data.json"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return []
    with open(path) as f:
        return json.load(f)


def extract_paper_text(conversations: list[dict]) -> str:
    for msg in conversations:
        if msg["from"] == "human":
            text = msg["value"]
            text = PROMPT_HEADER_RE.sub("", text)
            return text.strip()
    return ""


def get_conf_label(entry: dict, default_conf: str) -> str:
    """Get the conference label, handling ICML vs COLM."""
    conf = entry["_metadata"].get("conference", default_conf)
    return conf.lower()


def main():
    rng = random.Random(SEED)

    # Index vision data by submission_id for image path lookup
    vision_index: dict[str, list[str]] = {}  # submission_id -> image paths
    for conf_label, ds_name in VISION_DATASETS:
        data = load_dataset(ds_name)
        for entry in data:
            sid = entry["_metadata"]["submission_id"]
            if "images" in entry and entry["images"]:
                vision_index[sid] = entry["images"]

    print(f"Vision index: {len(vision_index)} submissions with images")

    # Process text datasets and pick one example per conf/year/label
    for default_conf, ds_name, _ in TEXT_DATASETS:
        print(f"\nProcessing {ds_name}...")
        data = load_dataset(ds_name)
        if not data:
            continue

        # Group by (conference, year, answer)
        groups: dict[tuple[str, int, str], list[dict]] = {}
        for entry in data:
            meta = entry["_metadata"]
            conf = get_conf_label(entry, default_conf)
            year = meta["year"]
            answer = meta["answer"].lower()  # "accept" or "reject"
            key = (conf, year, answer)
            groups.setdefault(key, []).append(entry)

        for (conf, year, answer), entries in sorted(groups.items()):
            # Pick one random example that has vision data with images actually on disk
            def has_images_on_disk(e):
                sid = e["_metadata"]["submission_id"]
                if sid not in vision_index:
                    return False
                return all(
                    (PROJECT_ROOT / img).exists()
                    for img in vision_index[sid]
                )

            candidates = [e for e in entries if has_images_on_disk(e)]
            if not candidates:
                # Fallback: has vision metadata but maybe missing files
                candidates = [e for e in entries if e["_metadata"]["submission_id"] in vision_index]
            if not candidates:
                candidates = entries  # fallback to text-only

            entry = rng.choice(candidates)
            sid = entry["_metadata"]["submission_id"]
            meta = entry["_metadata"]

            # --- Text example ---
            text_dir = EXAMPLES_DIR / conf / str(year) / answer / "text"
            text_dir.mkdir(parents=True, exist_ok=True)

            paper_text = extract_paper_text(entry["conversations"])
            md_content = f"# {conf.upper()} {year} — {answer.capitalize()}\n\n"
            md_content += f"**Submission ID:** `{sid}`\n"
            md_content += f"**Conference:** {meta.get('conference', conf)}\n"
            md_content += f"**Year:** {year}\n"
            md_content += f"**Decision:** {meta.get('decision', answer)}\n"
            if meta.get("ratings"):
                md_content += f"**Ratings:** {meta['ratings']}\n"
            md_content += f"\n---\n\n{paper_text}\n"

            with open(text_dir / "example.md", "w") as f:
                f.write(md_content)

            # --- Vision example (symlinks) ---
            if sid in vision_index:
                vision_dir = EXAMPLES_DIR / conf / str(year) / answer / "vision"
                vision_dir.mkdir(parents=True, exist_ok=True)

                for img_path in vision_index[sid]:
                    # img_path is like "data/images/xxx/page_1.png" or "data/images_extra/xxx/page_1.png"
                    abs_img = PROJECT_ROOT / img_path
                    # Fallback: if data/images/... doesn't exist, try data/images_extra/...
                    if not abs_img.exists() and "data/images/" in img_path:
                        alt_path = img_path.replace("data/images/", "data/images_extra/", 1)
                        alt_abs = PROJECT_ROOT / alt_path
                        if alt_abs.exists():
                            abs_img = alt_abs
                    if abs_img.exists():
                        link_name = vision_dir / abs_img.name
                        if link_name.exists() or link_name.is_symlink():
                            link_name.unlink()
                        link_name.symlink_to(abs_img)

            print(f"  {conf}/{year}/{answer}: sid={sid}, images={len(vision_index.get(sid, []))}")

    print(f"\nDone! Examples written to {EXAMPLES_DIR}")


if __name__ == "__main__":
    main()
