"""Reconstruct LlamaFactory-formatted local datasets from the PaperLens HF release.

The PaperLens HF datasets are content stores: one row per unique paper, with
title + content + metadata + a `references: list[(dataset_key, split)]` field
that names every dataset_info.json entry the paper belongs to. This script
filters by (dataset_key, split) and re-materializes the original sharegpt
data.json files (and per-paper image PNGs for vision) into ``data/<key>/``,
registering each rebuilt key in ``data/dataset_info.json``.

Round-trip is byte-identical for conv[1] (system + human + gpt turns), images
(via the HF Image() feature), and ``_metadata``. See scripts/tests/test_reconstruction.py.

Usage:

    # Reconstruct the canonical 4-cell eval datasets
    python scripts/reconstruction.py --dataset_keys \\
        arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test \\
        arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation \\
        iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_test \\
        iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_validation

    # Or rebuild everything (large -- pulls images via the HF Image() feature)
    python scripts/reconstruction.py --all

    # Read from a locally-built HF dataset dir (skip the HF hub fetch)
    python scripts/reconstruction.py --local_dir /scratch/.../hf_release_local --all
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# These prompts are constant per family. The HF dataset stores `content` with
# whatever whitespace originally separated the prompt from the paper body
# (legacy ICLR text rows have an extra blank line), so reconstruction is just
# `prompt + content` -- no separator guesswork.
SHAREGPT_SYSTEM_PROMPT = "You are an expert academic reviewer tasked with evaluating research papers."

PROMPT_ARXIV = (
    "I am giving you a paper submitted to a top machine-learning venue. "
    "Predict its acceptance outcome.\n"
    " - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n"
    " - Note: typical top-tier ML venues have ~25-30% acceptance rates"
)
PROMPT_ICLR = (
    "I am giving you a paper. I want to predict its acceptance outcome at ICLR.\n"
    " - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n"
    " - Note: ICLR generally has a ~30% acceptance rate"
)

# Default HF repo names. Override via --hf_text_repo / --hf_vision_repo.
DEFAULT_TEXT_REPO = "paperlens/paperlens-text"
DEFAULT_VISION_REPO = "paperlens/paperlens-vision"


def prompt_for_subset(subset: str) -> str:
    if subset == "arxiv":
        return PROMPT_ARXIV
    if subset == "iclr":
        return PROMPT_ICLR
    raise ValueError(f"unknown paper subset: {subset!r}")


def modality_of(key: str) -> str:
    if "_text_" in key or key.startswith("combined_arxiv_iclr_42k_text"):
        return "text"
    if "_vision_" in key or key.startswith("combined_arxiv_iclr_42k_vision"):
        return "vision"
    raise ValueError(f"cannot infer modality from key: {key!r}")


def load_hf_subset(text_or_vision: str, subset: str, local_dir: Path | None,
                   hf_repo: str):
    """Load text/iclr or text/arxiv or vision/iclr or vision/arxiv HF subset."""
    from datasets import load_from_disk, load_dataset
    if local_dir is not None:
        path = local_dir / text_or_vision / subset
        if not path.exists():
            raise FileNotFoundError(f"local_dir/{text_or_vision}/{subset} missing")
        return load_from_disk(str(path))
    # HF hub: subset is a config name
    return load_dataset(hf_repo, name=subset, split="train")


def image_to_relpath(img, paper_id: str, subset: str, page_idx: int,
                     images_root: Path) -> str:
    """Write a PIL image to disk and return its relative path (matching the
    canonical data/images_<subset>/<paper_id>/page_N.png shape).
    """
    rel_dir = images_root / f"images_{subset}" / paper_id
    rel_dir.mkdir(parents=True, exist_ok=True)
    out = rel_dir / f"page_{page_idx + 1}.png"
    if hasattr(img, "save"):
        img.save(out, format="PNG")
    else:
        # raw bytes
        out.write_bytes(img)
    # Return relative to data/
    return str(out.relative_to(images_root.parent))


def gpt_turn_for(label: str) -> str:
    return f"Outcome: \\boxed{{{label}}}"


def rebuild_sharegpt_row(hf_row, key: str, subset: str, modality: str,
                         data_root: Path) -> dict:
    """Reconstruct one sharegpt row from a HF dataset row."""
    prompt = prompt_for_subset(subset)
    content = hf_row["content"]
    human_value = f"{prompt}{content}"
    label = hf_row.get("label") or ""
    metadata = json.loads(hf_row["metadata"])

    conversations = [
        {"from": "system", "value": SHAREGPT_SYSTEM_PROMPT},
        {"from": "human", "value": human_value},
    ]
    if label:
        conversations.append({"from": "gpt", "value": gpt_turn_for(label)})

    row = {"conversations": conversations, "_metadata": metadata}
    if modality == "vision":
        pid = hf_row["paper_id"]
        images_root = data_root
        paths = []
        imgs = hf_row.get("images") or []
        for i, img in enumerate(imgs):
            paths.append(image_to_relpath(img, pid, subset, i, images_root))
        row["images"] = paths
    return row


def reconstruct_key(key: str, dataset_info: dict, hf_text, hf_vision,
                    data_root: Path) -> int:
    """Materialize ``data/<key>/data.json`` (+ images for vision) from the
    HF dataset rows where (key, split) is in row['references'].
    Returns the number of rows written.
    """
    modality = modality_of(key)
    # Each row knows its OWN paper subset by the HF dataset path; we filter
    # both subsets and merge (handles combined_arxiv_iclr_42k_*).
    rows_out = []
    for subset, ds in [("arxiv", hf_text if modality == "text" else hf_vision),
                       ("iclr",  hf_text if modality == "text" else hf_vision)]:
        # Actually each modality has its own (arxiv, iclr) subsets:
        pass
    # Walk both subsets of the matching modality
    if modality == "text":
        subsets = {"arxiv": hf_text["arxiv"], "iclr": hf_text["iclr"]}
    else:
        subsets = {"arxiv": hf_vision["arxiv"], "iclr": hf_vision["iclr"]}

    for subset, ds in subsets.items():
        for hf_row in ds:
            refs = hf_row["references"] or []
            if not any(r[0] == key for r in refs):
                continue
            rows_out.append(rebuild_sharegpt_row(hf_row, key, subset, modality, data_root))

    if not rows_out:
        print(f"  [SKIP] {key}: no matching papers", file=sys.stderr)
        return 0

    # Write data.json
    out_dir = data_root / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.json"
    out_path.write_text(json.dumps(rows_out, indent=2))

    # Register in dataset_info.json (mirror the published columns/tags shape)
    columns = {"messages": "conversations"}
    if modality == "vision":
        columns["images"] = "images"
    dataset_info[key] = {
        "file_name": f"{key}/data.json",
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
    print(f"  [OK]   {key}: {len(rows_out)} rows -> {out_path}")
    return len(rows_out)


def list_all_publishable_keys(local_dir: Path | None) -> list[str]:
    """Read the manifest emitted by build_hf_release_datasets.py."""
    if local_dir is None:
        raise SystemExit(
            "ERROR: --all requires --local_dir <hf_release_local>; "
            "the manifest is bundled with the local HF build."
        )
    manifest = json.loads((local_dir / "manifest.json").read_text())
    return sorted(manifest["keys"].keys())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_keys", nargs="*",
                    help="dataset_info.json keys to reconstruct")
    ap.add_argument("--all", action="store_true",
                    help="Reconstruct every key in the HF release manifest")
    ap.add_argument("--local_dir", type=Path, default=None,
                    help="Path to a locally-built HF release dir (skip hub fetch)")
    ap.add_argument("--hf_text_repo", default=DEFAULT_TEXT_REPO)
    ap.add_argument("--hf_vision_repo", default=DEFAULT_VISION_REPO)
    ap.add_argument("--data_root", type=Path, default=Path("data"),
                    help="Local data/ root to materialize into")
    ap.add_argument("--dataset_info", type=Path, default=None,
                    help="Path to dataset_info.json (default: data_root/dataset_info.json)")
    args = ap.parse_args()

    if not args.dataset_keys and not args.all:
        ap.error("either --dataset_keys ... or --all is required")

    try:
        from datasets import DatasetDict, load_from_disk, load_dataset
    except ImportError as e:
        print(f"ERROR: need `datasets` + Pillow: {e}", file=sys.stderr)
        return 2

    # Load both modality datasets up-front. Subsets (arxiv, iclr) are
    # accessed lazily via [subset].
    print("[reconstruction] loading HF datasets ...")
    if args.local_dir:
        hf_text = {sub: load_from_disk(str(args.local_dir / "text" / sub))
                   for sub in ("arxiv", "iclr")}
        hf_vision = {sub: load_from_disk(str(args.local_dir / "vision" / sub))
                     for sub in ("arxiv", "iclr")}
    else:
        hf_text = {sub: load_dataset(args.hf_text_repo, name=sub, split="train")
                   for sub in ("arxiv", "iclr")}
        hf_vision = {sub: load_dataset(args.hf_vision_repo, name=sub, split="train")
                     for sub in ("arxiv", "iclr")}

    keys = args.dataset_keys or list_all_publishable_keys(args.local_dir)
    print(f"[reconstruction] {len(keys)} key(s) to materialize -> {args.data_root}")

    args.data_root.mkdir(parents=True, exist_ok=True)
    info_path = args.dataset_info or (args.data_root / "dataset_info.json")
    dataset_info = json.loads(info_path.read_text()) if info_path.exists() else {}

    total = 0
    for key in keys:
        try:
            total += reconstruct_key(key, dataset_info, hf_text, hf_vision, args.data_root)
        except Exception as e:
            print(f"  [FAIL] {key}: {e}", file=sys.stderr)

    info_path.write_text(json.dumps(dataset_info, indent=2))
    print(f"[reconstruction] done. {total} total rows. dataset_info.json updated -> {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
