"""Build the 2 paperlens HF release datasets from the main repo's data/.

Reads data/dataset_info.json + per-key data.json files, groups rows by
paper_id, emits 2 HuggingFace Datasets (text + vision), each with arxiv
and iclr subsets:

    hf_release_local/text/arxiv         -> Dataset(paper_id, title, content, metadata, references)
    hf_release_local/text/iclr          -> same
    hf_release_local/vision/arxiv       -> + images: Sequence(Image())
    hf_release_local/vision/iclr        -> same

`references` is a list of (dataset_key, split) tuples — one per (key, split)
pair the paper appears in across the publishable dataset_info entries.

reconstruction.py (in published-branch) consumes these to rebuild any
publishable data/<key>/data.json on demand.

This script is ADDITIVE to the main repo (no deletion or modification of
existing files). Run from the main repo root.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Dataset key inclusion filter — only the entries we want in the public release
INCLUDE_PREFIXES = (
    "arxiv_50_50_21k_",
    "arxiv_50_50_balanced_per_venue_",
    "iclr_2020_2023_2025_2026_85_5_10_balanced_original_",
    "combined_arxiv_iclr_42k_",
)

# If a key contains any of these substrings, exclude it. Two groups:
#   (a) experiment slices we never publish (q4 subsetting, reasoning, review-
#       augmented, panel-eval, residual, balanced_sampler).
#   (b) DERIVED variants that are regenerated locally from the canonical data
#       by a script (mirrors how panels are rebuilt via build_panel_*.py).
#       Storing them as static HF rows is impossible anyway: they share a
#       paper_id with the canonical row but carry transformed content, so the
#       one-row-per-paper store can only hold the canonical version. Their
#       reconstruction is:
#         _reversed  -> scripts/build_reversed_datasets.py     (reverse sections / images)
#         _yearcond  -> scripts/build_year_conditioned_dataset.py (inject year into prompt)
#       Run those AFTER reconstruction.py rebuilds the canonical labelfix sets.
EXCLUDE_SUBSTRINGS = (
    "_q4_seed",
    "_q4b_seed",
    "_reasoning",
    "_panel",
    "_cleaned_",
    "_geminireviews",
    "_qwenreviews",
    "_paperstats",
    "_natrate",
    "_residual",
    "_balanced_sampler",
    "_reversed",        # derived: build_reversed_datasets.py
    "_yearcond",        # derived: build_year_conditioned_dataset.py
)

# Recognized split suffixes (longest-first matters: y24up_test before test).
# Ordering: longer + qualified variants come first so they match before the
# bare _train/_test/_validation suffixes.
SPLIT_SUFFIXES = (
    "_y24up_test",
    "_y24up_validation",
    "_y25up_test",
    "_y25up_validation",
    "_yearcond_test",
    "_yearcond_train",
    "_yearcond_validation",
    "_train_reversed",
    "_test_reversed",
    "_validation_reversed",
    "_train_50pct",
    "_train_75pct",
    "_validation",
    "_train",
    "_test",
)


def is_publishable(key: str) -> bool:
    # Explicit allowlist (see RELEASE_REF below): only the canonical
    # filtered24480 train/val/test/yup keys + the combined-train + the
    # train-only extras (iclr max_rejects, arxiv residual) are published.
    return key in RELEASE_REF


def modality_of(key: str) -> str | None:
    if "_text_" in key or key.startswith("combined_arxiv_iclr_42k_text"):
        return "text"
    if "_vision_" in key or key.startswith("combined_arxiv_iclr_42k_vision"):
        return "vision"
    return None


def split_of(key: str) -> str:
    for s in SPLIT_SUFFIXES:
        if key.endswith(s):
            return s.lstrip("_")
    return "unknown"


# ---------------------------------------------------------------------------
# Release naming. The `references` field uses friendly (name, split) pairs,
# not the internal dataset_info keys. y24up/y25up collapse to *_yup since the
# release name already implies the domain (arxiv -> y24, iclr -> y25).
# ---------------------------------------------------------------------------
_ARXIV_SPLITS = ("train", "validation", "test", "y24up_validation", "y24up_test")
_ICLR_SPLITS = ("train", "validation", "test", "y25up_validation", "y25up_test")
_SPLIT_RENAME = {
    "y24up_validation": "val_yup", "y24up_test": "test_yup",
    "y25up_validation": "val_yup", "y25up_test": "test_yup",
    "train_50pct": "train", "train_75pct": "train",
}


def _release_families(m: str) -> list[tuple[str, list[str]]]:
    """(release_name, [internal_keys]) for modality m in {text, vision}.

    The canonical training variant differs by modality: arxiv uses
    *_wmetadata_filtered24480; iclr text bakes the 24480-token filter into
    '_filtered', while iclr vision adds an explicit '_filtered24480'.
    """
    iclr = "iclr_2020_2023_2025_2026_85_5_10_balanced_original"
    iv = f"{iclr}_{m}_labelfix_v7_filtered" + ("_filtered24480" if m == "vision" else "")
    return [
        ("arxiv-mini",       [f"arxiv_50_50_21k_{m}_wmetadata_filtered24480_{s}" for s in _ARXIV_SPLITS]),
        ("arxiv",            [f"arxiv_50_50_balanced_per_venue_{m}_wmetadata_filtered24480_{s}" for s in _ARXIV_SPLITS]),
        ("iclr",             [f"{iv}_{s}" for s in _ICLR_SPLITS]),
        ("iclr-train_50pct", [f"{iv}_train_50pct"]),
        ("iclr-train_75pct", [f"{iv}_train_75pct"]),
        ("combined",         [f"combined_arxiv_iclr_42k_{m}_filtered24480_train"]),
        ("iclr-max_rejects", [f"iclr_2020_2023_2025_2026_max_rejects_original_{m}_v7_filtered_filtered24480_train"]),
        ("arxiv-residual",   [f"arxiv_residual_{m}_wmetadata_filtered24480_train"]),
    ]


# internal dataset_info key -> (release_name, release_split)
RELEASE_REF: dict[str, tuple[str, str]] = {}
for _m in ("text", "vision"):
    for _name, _keys in _release_families(_m):
        for _k in _keys:
            _sp = split_of(_k)
            RELEASE_REF[_k] = (_name, _SPLIT_RENAME.get(_sp, _sp))


def release_ref(key: str) -> tuple[str, str]:
    """Friendly (release_name, release_split) for a publishable key."""
    return RELEASE_REF[key]


def paper_subset_of(row: dict) -> str | None:
    md = row.get("_metadata") or {}
    if "arxiv_id" in md and md["arxiv_id"]:
        return "arxiv"
    if "submission_id" in md and md["submission_id"]:
        return "iclr"
    return None


def paper_id_of(row: dict, subset: str) -> str | None:
    md = row.get("_metadata") or {}
    return md.get("arxiv_id") if subset == "arxiv" else md.get("submission_id")


def extract_title(row: dict, content: str) -> str:
    md = row.get("_metadata") or {}
    if md.get("title"):
        return str(md["title"]).strip()
    # Fall back: first '# Heading' line in content
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


# Exact known prompt suffixes — slice immediately after the matched suffix
# so whatever whitespace separates the prompt from the body is preserved
# verbatim (legacy ICLR text rows have an extra blank line; arxiv rows
# don't). Reconstruction is then just `prompt + content` — no separator
# guesswork required.
PROMPT_SUFFIXES = (
    "Note: typical top-tier ML venues have ~25-30% acceptance rates",   # arxiv generic
    "Note: ICLR generally has a ~30% acceptance rate",                  # iclr binary
)


def strip_prompt(human_value: str) -> str:
    """Drop the user-prompt prefix and return everything that follows.

    The returned content starts with whatever whitespace the original row
    placed between the prompt and the paper body, so reconstruction is
    simply `prompt + content` (no separator assumptions).
    """
    for suffix in PROMPT_SUFFIXES:
        i = human_value.find(suffix)
        if i >= 0:
            return human_value[i + len(suffix):]
    # Fallback: anchor on the first '# ' heading at start of a line
    idx = human_value.find("\n# ")
    if idx >= 0:
        return human_value[idx:]
    if human_value.startswith("# "):
        return "\n" + human_value  # rebuild needs a leading sep
    return human_value


def label_of(row: dict) -> str | None:
    """Return 'Accept' or 'Reject' or None."""
    md = row.get("_metadata") or {}
    a = md.get("answer") or md.get("training_label") or md.get("decision")
    if a is None:
        return None
    a = str(a).strip().lower()
    if a in ("accept", "accepted"):
        return "Accept"
    if a in ("reject", "rejected"):
        return "Reject"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source_repo", default=".", help="Main repo root (default: cwd)")
    ap.add_argument("--output_dir", default="./hf_release_local", help="Where to save HF datasets")
    ap.add_argument("--keys", nargs="*", help="Restrict to these dataset_info keys (debug)")
    ap.add_argument("--limit_rows", type=int, default=None, help="Per-key row cap (debug)")
    ap.add_argument("--skip_images", action="store_true", help="Skip vision image bytes (debug; produces image-path strings only)")
    ap.add_argument("--max_shard_size", default="50GB", help="Max Parquet/Arrow shard size (e.g. '500MB', '1GB', '50GB'). Larger = fewer shard files. Vision rows are bulky from embedded image bytes, so default is large.")
    ap.add_argument("--num_proc", type=int, default=None, help="Workers for save_to_disk (None=auto)")
    ap.add_argument("--skip_existing", action="store_true", help="Skip (modality/subset) buckets whose output dir already has dataset_info.json")

    # Slurm-array friendly per-shard mode. When these are set the script:
    #   - only processes the single requested (target_modality, target_subset) bucket
    #   - includes only papers whose stable hash(paper_id) % num_shards == shard_idx
    #   - writes ONE parquet file (data-shard-NNNN.parquet) instead of a save_to_disk tree
    # Multiple per-shard runs land alongside each other in the same dir; HF
    # `load_dataset("parquet", data_files="<dir>/*.parquet")` reads them all as
    # one dataset. The sidecar manifest is written only when shard_idx == 0.
    ap.add_argument("--shard_idx", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=None)
    ap.add_argument("--target_modality", choices=["text", "vision"], default=None)
    ap.add_argument("--target_subset", choices=["arxiv", "iclr"], default=None)
    args = ap.parse_args()

    sharded = args.shard_idx is not None and args.num_shards is not None
    if sharded:
        if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
            print(f"ERROR: shard_idx={args.shard_idx} out of range [0,{args.num_shards})", file=sys.stderr); return 2
        if not (args.target_modality and args.target_subset):
            print("ERROR: --shard_idx requires --target_modality and --target_subset", file=sys.stderr); return 2
        print(f"[builder] sharded run: shard {args.shard_idx}/{args.num_shards}  bucket {args.target_modality}/{args.target_subset}")

    # Late import so the script's --help works without the deps
    try:
        from datasets import Dataset, Features, Image, Sequence, Value
    except ImportError as e:
        print(f"ERROR: need `datasets` (and Pillow): {e}", file=sys.stderr)
        return 2

    src = Path(args.source_repo).resolve()
    info_path = src / "data" / "dataset_info.json"
    if not info_path.exists():
        print(f"ERROR: {info_path} not found", file=sys.stderr)
        return 2

    info = json.loads(info_path.read_text())
    all_keys = sorted(k for k in info if is_publishable(k))
    if args.keys:
        all_keys = [k for k in all_keys if k in args.keys]
    print(f"[builder] {len(all_keys)} publishable keys")

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    base_features = {
        "paper_id": Value("string"),
        "title": Value("string"),
        "content": Value("string"),
        "metadata": Value("string"),
        "label": Value("string"),
        "references": Sequence(Sequence(Value("string"))),
    }

    # Process one (paper_subset, modality) bucket at a time so peak RAM
    # stays bounded. Each pass re-reads only the keys whose modality
    # matches, accumulates papers of the matching subset, saves, drops.
    targets = [
        ("arxiv", "text"),
        ("iclr", "text"),
        ("arxiv", "vision"),
        ("iclr", "vision"),
    ]

    for paper_subset, mod in targets:
        # In sharded mode, only process the requested bucket.
        if sharded and (paper_subset != args.target_subset or mod != args.target_modality):
            continue
        relevant_keys = [k for k in all_keys if modality_of(k) == mod]
        if not relevant_keys:
            continue
        out_path = out_root / mod / paper_subset
        # In sharded mode, the per-shard parquet path; in legacy mode, save_to_disk
        # writes a dataset directory we can detect with dataset_info.json.
        if not sharded and args.skip_existing and (out_path / "dataset_info.json").exists():
            print(f"\n[builder] === {mod}/{paper_subset}: already saved at {out_path}, skipping (--skip_existing) ===")
            continue
        if sharded:
            shard_file = out_path / f"data-shard-{args.shard_idx:04d}.parquet"
            if args.skip_existing and shard_file.exists():
                print(f"\n[builder] === {mod}/{paper_subset} shard {args.shard_idx}: {shard_file} already exists, skipping ===")
                continue
        print(f"\n[builder] === {mod}/{paper_subset}: scanning {len(relevant_keys)} {mod} keys ===")

        bucket: dict[str, dict] = {}
        for key in relevant_keys:
            split = split_of(key)
            rel = info[key].get("file_name")
            if not rel:
                print(f"  SKIP (no file_name): {key}")
                continue
            ds_path = src / "data" / rel
            # A dataset dir may hold a single data.json OR be split across
            # data_part*.json (LF concatenates all .json in the dir). Large
            # text train sets (e.g. per_venue) use the multi-part form.
            if ds_path.is_dir():
                json_files = sorted(ds_path.glob("*.json"))
            elif ds_path.exists():
                json_files = [ds_path]
            elif ds_path.with_suffix(".json").exists():
                json_files = [ds_path.with_suffix(".json")]
            else:
                json_files = []
            if not json_files:
                print(f"  SKIP (no json): {key} -> {ds_path}")
                continue

            rows = []
            for jf in json_files:
                with open(jf) as f:
                    rows.extend(json.load(f))
            if args.limit_rows:
                rows = rows[: args.limit_rows]

            n_added = 0
            for row in rows:
                # Filter to papers matching THIS pass's subset
                if paper_subset_of(row) != paper_subset:
                    continue
                pid = paper_id_of(row, paper_subset)
                if not pid:
                    continue
                # Sharded mode: only keep papers in this shard's hash bucket.
                # zlib.crc32 is deterministic across runs / processes (unlike Python's hash()).
                if sharded:
                    import zlib
                    if zlib.crc32(pid.encode()) % args.num_shards != args.shard_idx:
                        continue
                if pid not in bucket:
                    conv = row.get("conversations", [])
                    human_value = next((c["value"] for c in conv if c.get("from") == "human"), "")
                    content = strip_prompt(human_value)
                    bucket[pid] = {
                        "paper_id": pid,
                        "title": extract_title(row, content),
                        "content": content,
                        "metadata": dict(row.get("_metadata") or {}),
                        "label": label_of(row),
                        "references": [],
                        "_images_rel": list(row.get("images") or []) if mod == "vision" else None,
                    }
                    n_added += 1
                ref = list(release_ref(key))
                if ref not in bucket[pid]["references"]:
                    bucket[pid]["references"].append(ref)
            print(f"  + {key}: {len(rows)} rows, +{n_added} new papers (now {len(bucket)} total)")

        if not bucket:
            print(f"[builder] {mod}/{paper_subset}: no papers, skipping")
            continue

        feats = dict(base_features)
        if mod == "vision":
            feats["images"] = Sequence(Image() if not args.skip_images else Value("string"))

        rows_out = []
        missing_imgs = 0
        for pid, p in bucket.items():
            entry = {
                "paper_id": p["paper_id"],
                "title": p["title"],
                "content": p["content"],
                "metadata": json.dumps(p["metadata"], default=str),
                "label": p["label"] or "",
                "references": p["references"],
            }
            if mod == "vision":
                img_rels = p.get("_images_rel") or []
                if args.skip_images:
                    entry["images"] = [str(r) for r in img_rels]
                else:
                    # Embed raw PNG bytes inline so the released parquet is
                    # self-contained (no dependency on the original disk paths).
                    # HF Image() feature: pass {bytes, path} dicts where bytes
                    # is the actual file content; path=None means "no source".
                    img_dicts = []
                    for rel in img_rels:
                        ap_ = src / rel if not rel.startswith("/") else Path(rel)
                        if ap_.exists():
                            img_dicts.append({"bytes": ap_.read_bytes(), "path": None})
                        else:
                            missing_imgs += 1
                    entry["images"] = img_dicts
            rows_out.append(entry)

        # Free the bucket before HF builds its arrow table
        bucket.clear()

        ds = Dataset.from_list(rows_out, features=Features(feats))
        out_path = out_root / mod / paper_subset
        out_path.mkdir(parents=True, exist_ok=True)
        n_refs = sum(len(r["references"]) for r in rows_out)
        if sharded:
            shard_file = out_path / f"data-shard-{args.shard_idx:04d}.parquet"
            ds.to_parquet(str(shard_file))
            print(
                f"[builder] {mod}/{paper_subset} shard {args.shard_idx}: "
                f"{len(rows_out)} papers, {n_refs} refs -> {shard_file}"
                + (f"  [missing imgs: {missing_imgs}]" if missing_imgs else "")
            )
        else:
            save_kwargs = {"max_shard_size": args.max_shard_size}
            if args.num_proc:
                save_kwargs["num_proc"] = args.num_proc
            ds.save_to_disk(str(out_path), **save_kwargs)
            print(
                f"[builder] {mod}/{paper_subset}: {len(rows_out)} papers, "
                f"{n_refs} (key,split) refs -> {out_path}"
                + (f"  [missing imgs: {missing_imgs}]" if missing_imgs else "")
            )
        # Drop references before next pass
        del rows_out
        del ds

    # Companion sidecar: list of all publishable keys + their (subset, modality, split, file_name)
    # so reconstruction.py can validate inputs. Only write in legacy single-job mode
    # or when sharded with shard_idx==0 (one writer, no races).
    if sharded and args.shard_idx != 0:
        return 0
    sidecar = {
        "keys": {
            k: {
                "subset_hint": "arxiv" if k.startswith("arxiv_") or k.startswith("combined_") else "iclr",
                "modality": modality_of(k),
                "split": split_of(k),
                "release_name": release_ref(k)[0],     # friendly name used in `references`
                "release_split": release_ref(k)[1],    # friendly split used in `references`
                "file_name": info[k]["file_name"],
                "columns": info[k].get("columns", {}),
            }
            for k in all_keys
        }
    }
    (out_root / "manifest.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[builder] wrote sidecar -> {out_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
