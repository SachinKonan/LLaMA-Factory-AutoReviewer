#!/usr/bin/env python3
"""Extract paper text from SFT datasets into a pre-training dataset.

Takes sharegpt-format datasets (with conversations), strips the prompt header
from the human message to get raw paper text, deduplicates by submission_id,
and writes a jsonl file suitable for LLaMA-Factory PT stage.

Example:
    python scripts/build_pretrain_dataset.py \
        --datasets iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train \
                   nips_2021_2025_original_text_v7_noref_train \
                   icml_colm_2024_2025_clean_binary_noref_train \
        --output data/pretrain_iclr_nips_icml/data.jsonl \
        --dataset-name pretrain_iclr_nips_icml
"""

import argparse
import json
import re
from pathlib import Path


# Regex to strip the prompt header that appears before the actual paper text.
# Matches: "I am giving you a paper. ... acceptance rate\n+"
PROMPT_HEADER_RE = re.compile(
    r"^I am giving you a paper\..*?acceptance rate\s*\n+",
    re.DOTALL,
)


def extract_paper_text(conversations):
    """Extract paper text from the human message, stripping the prompt header."""
    for msg in conversations:
        if msg.get("from") == "human":
            text = msg["value"]
            text = PROMPT_HEADER_RE.sub("", text)
            text = text.strip()
            return text if text else None
    return None


def load_dataset(dataset_dir):
    """Load a dataset from a directory (data.json or data.jsonl)."""
    json_path = dataset_dir / "data.json"
    jsonl_path = dataset_dir / "data.jsonl"

    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    elif jsonl_path.exists():
        with open(jsonl_path) as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        raise FileNotFoundError(f"No data.json or data.jsonl found in {dataset_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build pre-training dataset from SFT data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset directory names under data/",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output jsonl file path",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Register dataset in data/dataset_info.json with this name",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable deduplication by submission_id",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory for datasets (default: data)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    total_loaded = 0
    total_written = 0
    total_deduped = 0
    total_empty = 0

    with open(output_path, "w") as out_f:
        for dataset_name in args.datasets:
            dataset_dir = data_root / dataset_name
            print(f"Loading {dataset_name}...")
            examples = load_dataset(dataset_dir)
            print(f"  Loaded {len(examples)} examples")
            total_loaded += len(examples)

            ds_written = 0
            ds_deduped = 0
            ds_empty = 0

            for ex in examples:
                # Deduplicate by submission_id
                if not args.no_deduplicate:
                    metadata = ex.get("_metadata", {})
                    sub_id = metadata.get("submission_id")
                    if sub_id and sub_id in seen_ids:
                        ds_deduped += 1
                        continue
                    if sub_id:
                        seen_ids.add(sub_id)

                # Extract paper text
                text = extract_paper_text(ex["conversations"])
                if not text:
                    ds_empty += 1
                    continue

                out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                ds_written += 1

            print(f"  Written: {ds_written}, Deduped: {ds_deduped}, Empty: {ds_empty}")
            total_written += ds_written
            total_deduped += ds_deduped
            total_empty += ds_empty

    print(f"\nSummary:")
    print(f"  Total loaded:  {total_loaded}")
    print(f"  Total written: {total_written}")
    print(f"  Total deduped: {total_deduped}")
    print(f"  Total empty:   {total_empty}")
    print(f"  Output: {output_path}")

    # Register in dataset_info.json
    if args.dataset_name:
        dataset_info_path = data_root / "dataset_info.json"
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)

        # Compute relative path from data root to the output directory
        rel_dir = str(output_path.parent.relative_to(data_root))

        dataset_info[args.dataset_name] = {
            "file_name": rel_dir,
            "columns": {
                "prompt": "text",
            },
        }

        with open(dataset_info_path, "w") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"  Registered '{args.dataset_name}' in {dataset_info_path}")


if __name__ == "__main__":
    main()
