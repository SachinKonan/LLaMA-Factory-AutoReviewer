#!/usr/bin/env python3
"""Create balanced eval datasets from dataMaster for non-ICLR-2020-2026 conference groups."""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATAMASTER_PATH = DATA_DIR / "dataMaster.json"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"

# Config: group_key -> { modality -> { eval_name, total, balanced } }
EVAL_CONFIGS = {
    "iclr_2017_2019": {
        "text": {
            "eval_name": "iclr_2017_2019_original_text_v7_eval200_balanced",
            "total": 200,
            "balanced": True,
        },
        "vision": {
            "eval_name": "iclr_2017_2019_original_vision_v7_eval200_balanced",
            "total": 200,
            "balanced": True,
        },
    },
    "nips_2021_2025": {
        "text": {
            "eval_name": "nips_2021_2025_original_text_v7_noref_eval500",
            "total": 500,
            "balanced": False,
        },
        "vision": {
            "eval_name": "nips_2021_2025_original_vision_v7_eval500",
            "total": 500,
            "balanced": False,
        },
    },
    "icml_colm_2024_2025": {
        "text": {
            "eval_name": "icml_colm_2024_2025_clean_binary_noref_eval350",
            "total": 350,
            "balanced": False,
        },
        "vision": {
            "eval_name": "icml_colm_2024_2025_vision_binary_eval350",
            "total": 350,
            "balanced": False,
        },
    },
}


def load_splits(group_info):
    """Load test split only as the sampling pool."""
    pool = []
    for split_key in ("test",):
        if split_key not in group_info:
            continue
        split_name = group_info[split_key]
        path = DATA_DIR / split_name / "data.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            entries = json.load(f)
        pool.extend(entries)
        print(f"  Loaded {len(entries)} from {split_name}")
    return pool


def proportional_allocation(year_totals, target_total):
    """Allocate target_total across years proportionally, with largest-remainder rounding."""
    grand_total = sum(year_totals.values())
    years = sorted(year_totals.keys())

    # Compute exact allocations
    exact = {y: (year_totals[y] / grand_total) * target_total for y in years}
    floors = {y: int(exact[y]) for y in years}
    remainders = {y: exact[y] - floors[y] for y in years}

    # Distribute remaining slots by largest remainder
    allocated = sum(floors.values())
    remaining = target_total - allocated
    sorted_by_remainder = sorted(years, key=lambda y: remainders[y], reverse=True)
    for y in sorted_by_remainder[:remaining]:
        floors[y] += 1

    return floors


def sample_balanced(pool_by_year, alloc):
    """For balanced sampling: equal accept/reject within each year allocation."""
    sampled = []
    dist = {}
    for year, count in sorted(alloc.items()):
        entries = pool_by_year[year]
        accepts = [e for e in entries if e["_metadata"]["answer"] == "Accept"]
        rejects = [e for e in entries if e["_metadata"]["answer"] == "Reject"]

        n_accept = count // 2
        n_reject = count - n_accept  # handles odd counts

        if len(accepts) < n_accept:
            print(f"  WARNING: year {year} only has {len(accepts)} accepts, need {n_accept}")
            n_accept = len(accepts)
            n_reject = count - n_accept
        if len(rejects) < n_reject:
            print(f"  WARNING: year {year} only has {len(rejects)} rejects, need {n_reject}")
            n_reject = len(rejects)
            n_accept = count - n_reject

        sampled_a = random.sample(accepts, n_accept)
        sampled_r = random.sample(rejects, n_reject)
        sampled.extend(sampled_a)
        sampled.extend(sampled_r)
        dist[str(year)] = {"accepts": n_accept, "rejects": n_reject}
        print(f"  Year {year}: {n_accept}A + {n_reject}R = {n_accept + n_reject}")
    return sampled, dist


def sample_random(pool_by_year, alloc):
    """For unbalanced sampling: random sample within each year."""
    sampled = []
    dist = {}
    for year, count in sorted(alloc.items()):
        entries = pool_by_year[year]
        if len(entries) < count:
            print(f"  WARNING: year {year} only has {len(entries)} entries, need {count}")
            count = len(entries)

        chosen = random.sample(entries, count)
        sampled.extend(chosen)

        n_accept = sum(1 for e in chosen if e["_metadata"]["answer"] == "Accept")
        n_reject = count - n_accept
        dist[str(year)] = {"accepts": n_accept, "rejects": n_reject}
        print(f"  Year {year}: {n_accept}A + {n_reject}R = {count}")
    return sampled, dist


def make_dataset_info_entry(eval_name, is_vision):
    """Create a dataset_info.json entry."""
    columns = {"messages": "conversations"}
    if is_vision:
        columns["images"] = "images"
    return {
        "file_name": f"{eval_name}/data.json",
        "formatting": "sharegpt",
        "columns": columns,
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    }


def main():
    with open(DATAMASTER_PATH) as f:
        datamaster = json.load(f)

    with open(DATASET_INFO_PATH) as f:
        dataset_info = json.load(f)

    for group_key, modalities in EVAL_CONFIGS.items():
        for modality, config in modalities.items():
            eval_name = config["eval_name"]
            total = config["total"]
            balanced = config["balanced"]
            is_vision = modality == "vision"

            print(f"\n{'='*60}")
            print(f"Creating {eval_name} (total={total}, balanced={balanced})")
            print(f"{'='*60}")

            group_info = datamaster[group_key][modality]

            # Load all splits
            pool = load_splits(group_info)
            print(f"  Total pool: {len(pool)}")

            # Deduplicate by submission_id (keep first occurrence)
            seen = set()
            unique_pool = []
            for entry in pool:
                sid = entry["_metadata"]["submission_id"]
                if sid not in seen:
                    seen.add(sid)
                    unique_pool.append(entry)
            pool = unique_pool
            print(f"  Unique pool: {len(pool)}")

            # Group by year
            pool_by_year = defaultdict(list)
            for entry in pool:
                pool_by_year[entry["_metadata"]["year"]].append(entry)

            year_totals = {y: len(entries) for y, entries in pool_by_year.items()}
            print(f"  Year totals: {dict(sorted(year_totals.items()))}")

            # Compute per-year allocation
            alloc = proportional_allocation(year_totals, total)
            print(f"  Allocation: {dict(sorted(alloc.items()))}")

            # Sample
            if balanced:
                # For balanced: ensure even allocation per year (divisible by 2)
                # Adjust: make each year's count even
                for y in alloc:
                    if alloc[y] % 2 != 0:
                        alloc[y] -= 1
                # Redistribute lost samples
                current_total = sum(alloc.values())
                deficit = total - current_total
                sorted_years = sorted(alloc.keys(), key=lambda y: year_totals[y], reverse=True)
                for y in sorted_years[:deficit]:
                    alloc[y] += 2  # add 2 to keep even
                # If we overshot, remove from smallest
                current_total = sum(alloc.values())
                if current_total > total:
                    excess = current_total - total
                    for y in reversed(sorted_years):
                        if excess <= 0:
                            break
                        remove = min(2, excess, alloc[y])
                        if remove % 2 == 0:
                            alloc[y] -= remove
                            excess -= remove
                print(f"  Balanced allocation: {dict(sorted(alloc.items()))} = {sum(alloc.values())}")
                sampled, dist = sample_balanced(pool_by_year, alloc)
            else:
                sampled, dist = sample_random(pool_by_year, alloc)

            # Verify no duplicates
            sids = [e["_metadata"]["submission_id"] for e in sampled]
            assert len(sids) == len(set(sids)), f"Duplicate submission_ids in {eval_name}!"
            assert len(sampled) == total, f"Expected {total}, got {len(sampled)} for {eval_name}"

            total_accepts = sum(d["accepts"] for d in dist.values())
            total_rejects = sum(d["rejects"] for d in dist.values())
            print(f"  Final: {len(sampled)} total, {total_accepts}A/{total_rejects}R")

            # Shuffle final dataset
            random.shuffle(sampled)

            # Save data
            out_dir = DATA_DIR / eval_name
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "data.json", "w") as f:
                json.dump(sampled, f, indent=2, ensure_ascii=False)
            print(f"  Saved to {out_dir / 'data.json'}")

            # Register in dataset_info
            dataset_info[eval_name] = make_dataset_info_entry(eval_name, is_vision)

            # Update dataMaster with eval key and dist
            datamaster[group_key][modality]["eval"] = eval_name
            datamaster[group_key][modality]["_eval_dist"] = dist

    # Save updated dataset_info.json
    with open(DATASET_INFO_PATH, "w") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nUpdated {DATASET_INFO_PATH}")

    # Save updated dataMaster.json
    with open(DATAMASTER_PATH, "w") as f:
        json.dump(datamaster, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Updated {DATAMASTER_PATH}")


if __name__ == "__main__":
    main()
