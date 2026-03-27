#!/usr/bin/env python3
"""Filter existing Gemini review datasets to exclude 2026 papers.

Example usage:
    python scripts/build_no2026_reviews_dataset.py \
        --source-datasets \
            iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_geminireviews_x2 \
            iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_geminireviews_x3 \
        --output-suffix _no2026
"""

import argparse
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")
SPLITS = ["train", "validation", "test"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_source_data(dataset_name, split):
    path = DATA_DIR / f"{dataset_name}_{split}" / "data.json"
    with open(path) as f:
        return json.load(f)


def write_dataset(data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _sanitize_surrogates(data)
    with open(output_dir / "data.json", "w") as f:
        json.dump(data, f, indent=None, ensure_ascii=False)


def _sanitize_surrogates(obj):
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: _sanitize_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_surrogates(v) for v in obj]
    return obj


def update_dataset_info(new_entries):
    info_path = DATA_DIR / "dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)
    info.update(new_entries)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")


def make_dataset_info_entry(dataset_name, split, is_vision):
    entry = {
        "file_name": f"{dataset_name}_{split}/data.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    }
    if is_vision:
        entry["columns"]["images"] = "images"
    return entry


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def derive_output_name(source_name):
    """Replace '2020_2023_2025_2026' with '2020_2023_2025' in the dataset name."""
    return source_name.replace("2020_2023_2025_2026", "2020_2023_2025")


def filter_no2026(data):
    """Keep only entries where _metadata.year != 2026."""
    kept = []
    removed = 0
    year_counts_before = Counter()
    year_counts_after = Counter()

    for entry in data:
        year = entry.get("_metadata", {}).get("year")
        year_counts_before[year] += 1
        if year == 2026:
            removed += 1
        else:
            kept.append(entry)
            year_counts_after[year] += 1

    return kept, removed, year_counts_before, year_counts_after


def is_vision_dataset(name):
    """Heuristic: vision datasets contain 'vision' in the name."""
    return "vision" in name.lower()


def process_dataset(source_name):
    """Filter one dataset across all splits, return dataset_info entries."""
    output_name = derive_output_name(source_name)
    vision = is_vision_dataset(source_name)
    info_entries = {}

    print(f"\n{'=' * 80}")
    print(f"Source:  {source_name}")
    print(f"Output:  {output_name}")
    print(f"Vision:  {vision}")
    print(f"{'=' * 80}")

    for split in SPLITS:
        print(f"\n  --- {split} ---")
        data = load_source_data(source_name, split)
        kept, removed, years_before, years_after = filter_no2026(data)

        print(f"  Before: {len(data):>6} entries")
        print(f"  After:  {len(kept):>6} entries  (removed {removed})")
        print(f"  Year distribution (before):")
        for year in sorted(years_before.keys(), key=lambda y: (y is None, y)):
            label = str(year) if year is not None else "None"
            print(f"    {label}: {years_before[year]}")
        print(f"  Year distribution (after):")
        for year in sorted(years_after.keys(), key=lambda y: (y is None, y)):
            label = str(year) if year is not None else "None"
            print(f"    {label}: {years_after[year]}")

        output_dir = DATA_DIR / f"{output_name}_{split}"
        write_dataset(kept, output_dir)
        print(f"  Written to: {output_dir / 'data.json'}")

        key = f"{output_name}_{split}"
        info_entries[key] = make_dataset_info_entry(output_name, split, vision)

    return info_entries


def main():
    parser = argparse.ArgumentParser(
        description="Filter review datasets to exclude 2026 papers."
    )
    parser.add_argument(
        "--source-datasets",
        nargs="+",
        required=True,
        help="Base names of source datasets (without split suffix).",
    )
    parser.add_argument(
        "--output-suffix",
        default="_no2026",
        help="Suffix appended to output name (used for documentation only; "
             "actual name is derived by replacing 2020_2023_2025_2026 with 2020_2023_2025).",
    )
    args = parser.parse_args()

    all_info_entries = {}
    for source in args.source_datasets:
        entries = process_dataset(source)
        all_info_entries.update(entries)

    print(f"\n{'=' * 80}")
    print("Updating dataset_info.json ...")
    update_dataset_info(all_info_entries)
    print(f"Registered {len(all_info_entries)} new entries in dataset_info.json")
    print("Done.")


if __name__ == "__main__":
    main()
