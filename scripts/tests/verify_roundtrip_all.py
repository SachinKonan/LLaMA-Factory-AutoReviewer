"""Exhaustive round-trip verification: every publishable (dataset_key) must
reconstruct from the local HF parquet release identically to the source
data/<key>/data.json.

For each key we map its internal name to the friendly (release_name,
release_split) via the release manifest -- exactly as reconstruction.py does --
collect the papers whose `references` contain that pair, and rebuild each row
with reconstruction.build_text_row. Checks, per paper:
  - conversations[*].from / .value   (system + human + gpt) byte-identical
  - _metadata key set + values
  - accept_reject_label              (when the key's columns mapping carries it)
  - (vision) image COUNT             (byte-identity of pixels is covered by the
                                      sampled health check; reading every PNG here
                                      would be ~470GB of I/O)

Runs in memory. The reference index + light columns are read with column
projection (no image bytes); image counts come from a cheap list-length pass.

Usage:
  python scripts/tests/verify_roundtrip_all.py \
      --local_dir /scratch/.../hf_release_v2 \
      --source_repo /scratch/.../LLaMA-Factory-AutoReviewer \
      [--modality text|vision|both] [--limit_keys N]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the published-branch reconstruction logic so we test the REAL code path.
# This file lives in scripts/tests/; reconstruction.py is one level up in scripts/.
RECON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RECON))
import reconstruction as R  # noqa: E402


def pid_of(row: dict):
    md = row.get("_metadata") or {}
    return md.get("arxiv_id") or md.get("submission_id")


def load_source(source_repo: Path, key: str, info: dict) -> dict[str, dict]:
    """Load source data.json (single-file OR multi-part data_part*.json),
    indexed by paper_id -- mirroring build_hf_release_datasets.py."""
    rel = info[key].get("file_name")
    if not rel:
        return {}
    p = source_repo / "data" / rel
    if p.is_dir():
        jfs = sorted(p.glob("*.json"))
    elif p.exists():
        jfs = [p]
    elif p.with_suffix(".json").exists():
        jfs = [p.with_suffix(".json")]
    else:
        return {}
    out: dict[str, dict] = {}
    for jf in jfs:
        for r in json.loads(jf.read_text()):
            pid = pid_of(r)
            if pid:
                out[pid] = r
    return out


def compare_paper(key, pid, src, rec, expected_arl) -> list[str]:
    """Return a list of diff messages (empty == identical).

    expected_arl: the accept_reject_label reconstruction would emit (1/0), or
    None if this key does not carry that column.
    """
    diffs = []
    s_conv = src["conversations"]
    r_conv = rec["conversations"]
    if len(s_conv) != len(r_conv):
        diffs.append(f"conv-len src={len(s_conv)} rec={len(r_conv)}")
    else:
        for i, (sc, rc) in enumerate(zip(s_conv, r_conv)):
            if sc.get("from") != rc.get("from"):
                diffs.append(f"conv[{i}].from {sc.get('from')!r}!={rc.get('from')!r}")
            if sc.get("value") != rc.get("value"):
                sv, rv = sc.get("value", ""), rc.get("value", "")
                off = next((j for j in range(min(len(sv), len(rv))) if sv[j] != rv[j]),
                           min(len(sv), len(rv)))
                diffs.append(f"conv[{i}].value diverges@{off} (src_len={len(sv)} rec_len={len(rv)})")
    s_md = src.get("_metadata") or {}
    r_md = rec.get("_metadata") or {}
    if set(s_md) != set(r_md):
        diffs.append(f"metadata keys differ (only_src={set(s_md)-set(r_md)} only_rec={set(r_md)-set(s_md)})")
    else:
        for k in s_md:
            if s_md[k] != r_md[k] and str(s_md[k]) != str(r_md[k]):
                diffs.append(f"metadata[{k!r}] src={s_md[k]!r}!=rec={r_md[k]!r}")
    if expected_arl is not None:
        if src.get("accept_reject_label") != expected_arl:
            diffs.append(f"accept_reject_label src={src.get('accept_reject_label')!r} rec_would_emit={expected_arl!r}")
    if "images" in src or "images" in rec:
        ns = len(src.get("images") or [])
        nr = len(rec.get("images") or [])
        if ns != nr:
            diffs.append(f"image-count src={ns} rec={nr}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True)
    ap.add_argument("--source_repo", default=".")
    ap.add_argument("--modality", choices=["text", "vision", "both"], default="both")
    ap.add_argument("--limit_keys", type=int, default=None)
    ap.add_argument("--max_paper_diffs", type=int, default=3, help="diffs to print per failing key")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    local = Path(args.local_dir).resolve()
    source = Path(args.source_repo).resolve()
    info = json.loads((source / "data" / "dataset_info.json").read_text())
    manifest = json.loads((local / "manifest.json").read_text())["keys"]

    # internal key -> friendly (release_name, release_split), and -> columns map
    key_want = {k: (m["release_name"], m["release_split"]) for k, m in manifest.items()}
    key_cols = {k: m.get("columns", {}) for k, m in manifest.items()}

    mods = ["text", "vision"] if args.modality == "both" else [args.modality]

    overall_ok = True
    for mod in mods:
        keys = sorted(k for k, m in manifest.items() if m["modality"] == mod)
        if args.limit_keys:
            keys = keys[: args.limit_keys]
        print(f"\n{'='*78}\n{mod.upper()}: {len(keys)} publishable keys\n{'='*78}")

        recon_index: dict[str, dict] = {}       # pid -> rebuilt sharegpt row (conv + _metadata)
        pid_label: dict[str, str] = {}           # pid -> hf label ("Accept"/"Reject"/"")
        pair_to_pids: dict[tuple, set] = defaultdict(set)   # (name,split) -> {pid}
        img_count: dict[str, int] = {}           # pid -> n images (vision)

        for subset in ["arxiv", "iclr"]:
            files = sorted(glob.glob(str(local / mod / subset / "*.parquet")))
            if not files:
                continue
            cols = ["paper_id", "content", "metadata", "label", "references"]
            for f in files:
                d = pq.read_table(f, columns=cols).to_pydict()
                for i in range(len(d["paper_id"])):
                    pid = d["paper_id"][i]
                    hf_row = {
                        "paper_id": pid,
                        "content": d["content"][i],
                        "metadata": d["metadata"][i],
                        "label": d["label"][i],
                        "references": d["references"][i],
                    }
                    recon_index[pid] = R.build_text_row(hf_row, subset)
                    pid_label[pid] = d["label"][i] or ""
                    if mod == "vision":
                        # page count == number of <image> placeholders baked into
                        # content (one per page). Avoids reading the ~470GB image
                        # column just to count pages; if it ever disagrees with the
                        # GT image count, the per-key compare flags it.
                        img_count[pid] = d["content"][i].count("<image>")
                    for ref in (d["references"][i] or []):
                        pair_to_pids[(ref[0], ref[1])].add(pid)

        n_pass = n_fail = 0
        for key in keys:
            src = load_source(source, key, info)
            if not src:
                print(f"  [skip] {key}: source data.json missing/empty")
                continue
            want = key_want[key]
            rec_pids = pair_to_pids.get(want, set())
            carries_arl = "accept_reject_label" in key_cols.get(key, {})

            if len(rec_pids) != len(src):
                print(f"  [FAIL] {key}: paper count src={len(src)} rec={len(rec_pids)} (want={want})")
                n_fail += 1
                overall_ok = False
                continue

            key_diffs = []
            for pid, s_row in src.items():
                r_row = recon_index.get(pid)
                if not r_row:
                    key_diffs.append(f"{pid}: missing in parquet")
                    continue
                if mod == "vision":
                    r_row = dict(r_row)
                    r_row["images"] = [None] * img_count.get(pid, 0)
                expected_arl = (1 if pid_label.get(pid) == "Accept" else 0) if carries_arl else None
                d = compare_paper(key, pid, s_row, r_row, expected_arl)
                if d:
                    key_diffs.append(f"{pid}: " + "; ".join(d))
            if key_diffs:
                print(f"  [FAIL] {key}: {len(key_diffs)}/{len(src)} papers differ")
                for msg in key_diffs[: args.max_paper_diffs]:
                    print(f"           {msg}")
                n_fail += 1
                overall_ok = False
            else:
                arl_note = " +arl" if carries_arl else ""
                print(f"  [OK]   {key}: {len(src)} papers identical{arl_note}")
                n_pass += 1
        print(f"\n  {mod}: {n_pass} OK, {n_fail} FAIL")

    print(f"\n{'='*78}\nRESULT: {'ALL PASS' if overall_ok else 'FAILURES DETECTED'}\n{'='*78}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
