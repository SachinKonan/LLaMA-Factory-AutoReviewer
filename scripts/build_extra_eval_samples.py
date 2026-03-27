"""
Build evaluation sample datasets for NIPS 2021-2025 and ICLR 2017-2019.

NIPS 2021-2025: 500 papers, unbalanced, year-proportional
ICLR 2017-2019: 250 papers, balanced accept/reject per year, year-proportional

Both produce matched text + vision versions (same submission_ids).
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
DATA_DIR = Path("data")


def load_all_splits(prefix: str) -> list[dict]:
    """Load train + validation + test splits and combine."""
    all_data = []
    for split in ["train", "validation", "test"]:
        path = DATA_DIR / f"{prefix}_{split}" / "data.json"
        if path.exists():
            with open(path) as f:
                all_data.extend(json.load(f))
            print(f"  Loaded {path}: {len(json.load(open(path)))} entries")
        else:
            print(f"  MISSING: {path}")
    return all_data


def load_all_splits_efficient(prefix: str) -> list[dict]:
    """Load train + validation + test splits and combine."""
    all_data = []
    for split in ["train", "validation", "test"]:
        path = DATA_DIR / f"{prefix}_{split}" / "data.json"
        if path.exists():
            with open(path) as f:
                entries = json.load(f)
            print(f"  Loaded {path}: {len(entries)} entries")
            all_data.extend(entries)
        else:
            print(f"  MISSING: {path}")
    return all_data


def index_by_submission_id(data: list[dict]) -> dict[str, dict]:
    """Create a dict mapping submission_id -> entry."""
    result = {}
    for entry in data:
        sid = entry["_metadata"]["submission_id"]
        result[sid] = entry
    return result


def get_year(entry: dict) -> int:
    return entry["_metadata"]["year"]


def get_label(entry: dict) -> str:
    return entry["_metadata"]["answer"]


def write_dataset(entries: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data.json"
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"  Wrote {len(entries)} entries to {output_path}")


def build_nips_eval():
    """Build NIPS 2021-2025 eval sample: 500 papers, unbalanced, year-proportional."""
    print("\n=== Building NIPS 2021-2025 Eval (500 papers, unbalanced) ===")

    print("\nLoading text data:")
    text_data = load_all_splits_efficient("nips_2021_2025_original_text_v7")
    print(f"Total text entries: {len(text_data)}")

    print("\nLoading vision data:")
    vision_data = load_all_splits_efficient("nips_2021_2025_original_vision_v7")
    print(f"Total vision entries: {len(vision_data)}")

    text_by_id = index_by_submission_id(text_data)
    vision_by_id = index_by_submission_id(vision_data)

    # Find intersection
    common_ids = set(text_by_id.keys()) & set(vision_by_id.keys())
    print(f"\nCommon submission_ids: {len(common_ids)}")

    # Group by year
    by_year = defaultdict(list)
    for sid in common_ids:
        year = get_year(text_by_id[sid])
        by_year[year].append(sid)

    print("Distribution by year:")
    for year in sorted(by_year.keys()):
        print(f"  {year}: {len(by_year[year])}")

    # Year-proportional sampling (500 total)
    year_targets = {
        2021: 66,
        2022: 76,
        2023: 93,
        2024: 115,
        2025: 150,
    }

    rng = random.Random(SEED)
    sampled_ids = []
    for year in sorted(year_targets.keys()):
        available = by_year[year]
        target = min(year_targets[year], len(available))
        chosen = rng.sample(available, target)
        sampled_ids.extend(chosen)
        print(f"  Sampled {year}: {len(chosen)}/{year_targets[year]} (available: {len(available)})")

    print(f"\nTotal sampled: {len(sampled_ids)}")

    # Build text and vision outputs
    text_entries = [text_by_id[sid] for sid in sampled_ids]
    vision_entries = [vision_by_id[sid] for sid in sampled_ids]

    write_dataset(text_entries, DATA_DIR / "nips_2021_2025_original_text_v7_eval500")
    write_dataset(vision_entries, DATA_DIR / "nips_2021_2025_original_vision_v7_eval500")

    # Print label distribution
    labels = defaultdict(int)
    for sid in sampled_ids:
        labels[get_label(text_by_id[sid])] += 1
    print(f"  Label distribution: {dict(labels)}")


def build_iclr_eval():
    """Build ICLR 2017-2019 eval sample: 250 papers, balanced, year-proportional."""
    print("\n=== Building ICLR 2017-2019 Eval (250 papers, balanced) ===")

    print("\nLoading text data:")
    text_data = load_all_splits_efficient("iclr_2017_2019_original_text_v7")
    print(f"Total text entries: {len(text_data)}")

    print("\nLoading vision data:")
    vision_data = load_all_splits_efficient("iclr_2017_2019_original_vision_v7")
    print(f"Total vision entries: {len(vision_data)}")

    text_by_id = index_by_submission_id(text_data)
    vision_by_id = index_by_submission_id(vision_data)

    # Find intersection
    common_ids = set(text_by_id.keys()) & set(vision_by_id.keys())
    print(f"\nCommon submission_ids: {len(common_ids)}")

    # Group by year and label
    by_year_label = defaultdict(lambda: defaultdict(list))
    for sid in common_ids:
        entry = text_by_id[sid]
        year = get_year(entry)
        label = get_label(entry)
        by_year_label[year][label].append(sid)

    print("Distribution by year and label:")
    for year in sorted(by_year_label.keys()):
        for label in sorted(by_year_label[year].keys()):
            print(f"  {year} {label}: {len(by_year_label[year][label])}")

    # Year-proportional sampling (250 total), balanced accept/reject
    year_targets = {
        2017: 41,
        2018: 77,
        2019: 132,
    }

    rng = random.Random(SEED)
    sampled_ids = []
    for year in sorted(year_targets.keys()):
        target = year_targets[year]
        half = target // 2

        accepts = by_year_label[year].get("Accept", [])
        rejects = by_year_label[year].get("Reject", [])

        # Balance: take equal from each, or as close as possible
        n_accept = min(half, len(accepts))
        n_reject = min(half, len(rejects))

        # If one side has fewer, try to take more from the other to reach target
        remaining = target - n_accept - n_reject
        if remaining > 0:
            extra_accept = min(remaining, len(accepts) - n_accept)
            n_accept += extra_accept
            remaining -= extra_accept
        if remaining > 0:
            extra_reject = min(remaining, len(rejects) - n_reject)
            n_reject += extra_reject

        chosen_accept = rng.sample(accepts, n_accept)
        chosen_reject = rng.sample(rejects, n_reject)
        sampled_ids.extend(chosen_accept)
        sampled_ids.extend(chosen_reject)
        print(f"  Sampled {year}: {n_accept} Accept + {n_reject} Reject = {n_accept + n_reject}/{target}")

    print(f"\nTotal sampled: {len(sampled_ids)}")

    # Build text and vision outputs
    text_entries = [text_by_id[sid] for sid in sampled_ids]
    vision_entries = [vision_by_id[sid] for sid in sampled_ids]

    write_dataset(text_entries, DATA_DIR / "iclr_2017_2019_original_text_v7_eval250_balanced")
    write_dataset(vision_entries, DATA_DIR / "iclr_2017_2019_original_vision_v7_eval250_balanced")

    # Print label distribution
    labels = defaultdict(int)
    for sid in sampled_ids:
        labels[get_label(text_by_id[sid])] += 1
    print(f"  Label distribution: {dict(labels)}")


if __name__ == "__main__":
    build_nips_eval()
    build_iclr_eval()
    print("\nDone!")
