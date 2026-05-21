import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "TRAIN_SAMPLE.json"
TEST_PATH = ROOT / "TEST_SAMPLE.json"
DERIVED_DIR = ROOT / "derived"
REVIEW_KEYWORDS = (
    "we propose",
    "we present",
    "we introduce",
    "our method",
    "our framework",
    "our approach",
    "our contributions",
    "results",
    "outperform",
    "state-of-the-art",
    "sota",
    "ablation",
    "limitation",
    "analysis",
    "conclusion",
)
SECTION_KEYWORDS = (
    "abstract",
    "introduction",
    "overview",
    "method",
    "approach",
    "framework",
    "algorithm",
    "theory",
    "analysis",
    "experiment",
    "results",
    "evaluation",
    "ablation",
    "discussion",
    "limitation",
    "conclusion",
)


def load_split(split: str) -> Dict[str, dict]:
    path = TRAIN_PATH if split == "train" else TEST_PATH
    return json.loads(path.read_text())


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_section_header(item: dict) -> bool:
    return item.get("type") == "text" and item.get("text_level") is not None


def extract_sections(paper: dict) -> List[Tuple[str, List[str]]]:
    sections: List[Tuple[str, List[str]]] = []
    current_header = "FRONT MATTER"
    current_items: List[str] = []

    def flush() -> None:
        nonlocal current_items
        if current_items:
            sections.append((current_header, current_items))
            current_items = []

    for item in paper.get("content_list", []):
        item_type = item.get("type")
        if is_section_header(item):
            flush()
            current_header = normalize_text(item.get("text", "UNTITLED SECTION"))
            continue
        if item_type in {"page_number", "page_footnote"}:
            continue
        if item_type == "text":
            text = normalize_text(item.get("text", ""))
            if text:
                current_items.append(text)
        elif item_type == "list":
            for entry in item.get("list_items", []):
                text = normalize_text(entry)
                if text:
                    current_items.append(f"- {text}")
        elif item_type == "image":
            for caption in item.get("image_caption", []):
                text = normalize_text(caption)
                if text:
                    current_items.append(f"[Figure] {text}")
        elif item_type == "table":
            for caption in item.get("table_caption", []):
                text = normalize_text(caption)
                if text:
                    current_items.append(f"[Table] {text}")
        elif item_type == "equation":
            text = normalize_text(item.get("text", ""))
            if text:
                current_items.append(f"[Equation] {text}")

    flush()
    return sections


def pick_relevant_sections(sections: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    picked: List[Tuple[str, List[str]]] = []
    fallback: List[Tuple[str, List[str]]] = []
    for header, items in sections:
        header_l = header.lower()
        compact_items = items[:8]
        if any(keyword in header_l for keyword in SECTION_KEYWORDS):
            picked.append((header, compact_items))
        elif len(fallback) < 3 and items:
            fallback.append((header, compact_items[:4]))
    return picked or fallback or sections[:6]


def collect_keyword_sentences(sections: Iterable[Tuple[str, List[str]]]) -> List[str]:
    sentences: List[str] = []
    for _, items in sections:
        for item in items:
            item_l = item.lower()
            if any(keyword in item_l for keyword in REVIEW_KEYWORDS):
                sentences.append(item)
    unique: List[str] = []
    seen = set()
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            unique.append(sentence)
    return unique[:20]


def render_paper(submission_id: str, paper: dict, split: str) -> str:
    sections = extract_sections(paper)
    relevant_sections = pick_relevant_sections(sections)
    headers = paper.get("headers", [])
    lines = [
        f"# Submission {submission_id}",
        f"Split: {split}",
    ]
    if "label" in paper:
        lines.append(f"Label: {paper['label']}")
    if headers:
        lines.append("Headers: " + " | ".join(headers[:20]))
    lines.append("")
    lines.append("## Key Sections")
    for header, items in relevant_sections:
        lines.append(f"### {header}")
        for item in items:
            lines.append(item)
        lines.append("")
    keyword_sentences = collect_keyword_sentences(sections)
    if keyword_sentences:
        lines.append("## Review Signals")
        for sentence in keyword_sentences:
            lines.append(f"- {sentence}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def cmd_dump_split(args: argparse.Namespace) -> None:
    data = load_split(args.split)
    out_dir = DERIVED_DIR / f"compact_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for submission_id, paper in data.items():
        rendered = render_paper(submission_id, paper, args.split)
        (out_dir / f"{submission_id}.md").write_text(rendered)
    print(f"Wrote {len(data)} files to {out_dir}")


def cmd_sample_train(args: argparse.Namespace) -> None:
    train = load_split("train")
    grouped = defaultdict(list)
    for submission_id, paper in train.items():
        grouped[paper["label"]].append(submission_id)
    rng = random.Random(args.seed)
    out = {}
    for label in ("accept", "reject"):
        candidates = grouped[label][:]
        rng.shuffle(candidates)
        out[label] = candidates[: args.count]
    print(json.dumps(out, indent=2))


def cmd_validate_predictions(args: argparse.Namespace) -> None:
    test = load_split("test")
    predictions = json.loads(Path(args.path).read_text())
    missing = sorted(set(test) - set(predictions))
    extra = sorted(set(predictions) - set(test))
    bad_decisions = sorted(
        sid
        for sid, payload in predictions.items()
        if payload.get("decision") not in {"accept", "reject"}
    )
    bad_why = sorted(
        sid
        for sid, payload in predictions.items()
        if not isinstance(payload.get("why"), str) or not payload["why"].strip()
    )
    print(
        json.dumps(
            {
                "expected": len(test),
                "predicted": len(predictions),
                "missing": missing,
                "extra": extra,
                "bad_decisions": bad_decisions,
                "bad_why": bad_why,
            },
            indent=2,
        )
    )


def cmd_make_batches(args: argparse.Namespace) -> None:
    test = load_split("test")
    batch_dir = DERIVED_DIR / "test_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    submission_ids = sorted(test)
    batch_size = args.batch_size
    batches = [
        submission_ids[i : i + batch_size]
        for i in range(0, len(submission_ids), batch_size)
    ]
    manifest = {}
    for idx, batch in enumerate(batches):
        batch_name = f"batch_{idx:02d}"
        path = batch_dir / f"{batch_name}.txt"
        path.write_text("\n".join(batch) + "\n")
        manifest[batch_name] = batch
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(batches)} batches to {batch_dir}")


def cmd_merge_reviews(args: argparse.Namespace) -> None:
    review_dir = Path(args.review_dir)
    merged = {}
    for path in sorted(review_dir.glob("batch_*.json")):
        payload = json.loads(path.read_text())
        overlap = set(merged) & set(payload)
        if overlap:
            raise ValueError(f"Overlapping review ids in {path}: {sorted(overlap)}")
        merged.update(payload)
    Path(args.output).write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(merged)} predictions to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump = subparsers.add_parser("dump-split")
    dump.add_argument("--split", choices=["train", "test"], required=True)
    dump.set_defaults(func=cmd_dump_split)

    sample = subparsers.add_parser("sample-train")
    sample.add_argument("--count", type=int, default=10)
    sample.add_argument("--seed", type=int, default=0)
    sample.set_defaults(func=cmd_sample_train)

    validate = subparsers.add_parser("validate-predictions")
    validate.add_argument("--path", required=True)
    validate.set_defaults(func=cmd_validate_predictions)

    batches = subparsers.add_parser("make-batches")
    batches.add_argument("--batch-size", type=int, default=10)
    batches.set_defaults(func=cmd_make_batches)

    merge = subparsers.add_parser("merge-reviews")
    merge.add_argument("--review-dir", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=cmd_merge_reviews)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
