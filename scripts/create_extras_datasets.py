#!/usr/bin/env python3
"""
Create dataset variants with appended extras (paper_stats, qwen_reviews, gemini_reviews).

Usage:
    python scripts/create_extras_datasets.py --variant paper_stats
    python scripts/create_extras_datasets.py --variant qwen_reviews
    python scripts/create_extras_datasets.py --variant gemini_reviews
    python scripts/create_extras_datasets.py --variant qwen_reviews_x2
    python scripts/create_extras_datasets.py --variant gemini_reviews_x2
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results/reviews")

SOURCE_DATASETS = {
    "text": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered",
    "vision": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480",
}

VARIANT_SUFFIXES = {
    "paper_stats": "paperstats",
    "paper_stats_full": "paperstats_full",
    "qwen_reviews": "qwenreviews",         # >=1 valid, pick 1st candidate
    "gemini_reviews": "geminireviews",       # >=1 valid, pick 1st candidate
    "qwen_reviews_x2": "qwenreviews_x2",   # >=2 valid, 2 entries per paper
    "gemini_reviews_x2": "geminireviews_x2", # >=2 valid, 2 entries per paper
    "gemini_reviews_x3": "geminireviews_x3", # >=3 valid, 3 entries per paper
}

# x2/x3 variants: create exactly N entries per matched paper
MULTI_VARIANTS = {"qwen_reviews_x2", "gemini_reviews_x2", "gemini_reviews_x3"}

SPLITS = ["train", "test", "validation"]

PAPER_STATS_FIELDS = [
    ("num_figures", "Figures"),
    ("num_figure_images", "Figure images"),
    ("num_pages", "Pages"),
    ("num_text_tokens", "Text tokens"),
    ("num_text_image_tokens", "Text+image tokens"),
    ("num_vision_tokens", "Vision tokens"),
    ("number_of_cited_references", "Cited references"),
    ("number_of_bib_items", "Bibliography items"),
    ("num_equations", "Equations"),
]

QWEN_REVIEW_FIELDS = [
    "paper_quality", "strengths", "weaknesses",
    "most_important_strength", "most_important_weakness", "accept",
]

GEMINI_REVIEW_FIELDS = [
    "paper_quality", "strengths", "weaknesses",
    "most_important_strength", "most_important_weakness",
    "relation_to_other_work", "critical_disputes",
]


def load_source_data(dataset_name: str, split: str) -> list[dict]:
    path = DATA_DIR / f"{dataset_name}_{split}" / "data.json"
    with open(path) as f:
        return json.load(f)


def build_submission_index(data: list[dict]) -> dict[str, int]:
    index = {}
    for i, entry in enumerate(data):
        sid = entry.get("_metadata", {}).get("submission_id")
        if sid:
            index[sid] = i
    return index


def load_paper_stats() -> dict[str, dict]:
    csv_path = DATA_DIR / "massive_metadata_v7.csv"
    stats = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("submission_id")
            if sid:
                stats[sid] = row
    return stats


def load_qwen_reviews() -> dict[str, list[dict]]:
    """Load all qwen review shards, group by submission_id."""
    pattern = str(RESULTS_DIR / "qwen_blind_text_*.jsonl")
    files = sorted(glob.glob(pattern))
    reviews = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            for line in f:
                entry = json.loads(line)
                sid = entry.get("submission_id")
                if not sid:
                    continue
                review = entry.get("review", {})
                if not _validate_qwen_review(review):
                    continue
                entry["_source_file"] = fname
                reviews.setdefault(sid, []).append(entry)
    # Sort by candidate_idx and pick first valid
    for sid in reviews:
        reviews[sid].sort(key=lambda x: x.get("candidate_idx", 999))
    return reviews


def _validate_qwen_review(review: dict) -> bool:
    if not isinstance(review, dict):
        return False
    string_fields = [
        "paper_quality", "strengths", "weaknesses",
        "most_important_strength", "most_important_weakness",
    ]
    for field in string_fields:
        val = review.get(field)
        if not isinstance(val, str) or len(val) < 20:
            return False
    if not isinstance(review.get("accept"), bool):
        return False
    return True


def load_gemini_reviews() -> dict[str, list[dict]]:
    """Load gemini reviews from merged file (preferred) or extracted file."""
    merged_path = RESULTS_DIR / "gemini_blind_text_merged.jsonl"
    if merged_path.exists():
        return _load_gemini_from_file(merged_path)
    extracted_path = RESULTS_DIR / "gemini_blind_text_all_extracted.jsonl"
    if extracted_path.exists():
        return _load_gemini_from_file(extracted_path)

    # Fallback: download from GCS
    print("Pre-extracted gemini file not found, attempting GCS download...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gemini_batch_review", "scripts/gemini_batch_review.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    job_info_files = sorted(glob.glob(str(RESULTS_DIR / "blind_text_retry_shard*_job_info.json")))
    all_entries = []
    for jf in job_info_files:
        with open(jf) as f:
            info = json.load(f)
        output_uri = info.get("output_uri")
        if not output_uri:
            continue
        raw = mod.download_from_gcs(output_uri)
        entries = mod.parse_batch_results(raw)
        all_entries.extend(entries)

    # Save merged file for future use
    with open(extracted_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved merged gemini reviews to {extracted_path} ({len(all_entries)} entries)")

    return _load_gemini_from_file(extracted_path)


def _load_gemini_from_file(path: Path) -> dict[str, list[dict]]:
    reviews = {}
    fname = os.path.basename(path)
    with open(path, errors="surrogateescape") as f:
        for line in f:
            # Strip surrogate characters that can't be encoded as UTF-8
            line = line.encode("utf-8", errors="replace").decode("utf-8")
            entry = json.loads(line)
            sid = entry.get("submission_id")
            if not sid:
                continue
            review = entry.get("review", {})
            if not _validate_gemini_review(review):
                continue
            entry["_source_file"] = fname
            reviews.setdefault(sid, []).append(entry)
    for sid in reviews:
        reviews[sid].sort(key=lambda x: x.get("candidate_idx", 999))
    return reviews


def _validate_gemini_review(review: dict) -> bool:
    if not isinstance(review, dict):
        return False
    for field in GEMINI_REVIEW_FIELDS:
        val = review.get(field)
        if isinstance(val, list):
            joined = " ".join(str(v) for v in val)
            if len(joined) < 20:
                return False
        elif isinstance(val, str):
            if len(val) < 20:
                return False
        else:
            return False
    return True


def load_all_paperstats() -> dict[str, dict]:
    path = DATA_DIR / "all_paperstats_v7.json"
    with open(path) as f:
        return json.load(f)


def format_all_paper_stats(feats: dict) -> str:
    """Format all ~187 features as a compact grouped text block."""
    g = lambda k, default=0: feats.get(k, default)
    yn = lambda v: "yes" if v else "no"

    lines = ["\n\n---\n\nPaper Statistics:"]

    # Page structure
    lines.append(
        f"- Pages: {g('num_pages'):.0f} | Body pages: {g('num_pages_noreferences'):.0f} | "
        f"Original pages: {g('original_total_pages'):.0f} | Appendix pages: {g('appendix_pages'):.0f} | "
        f"Appendix ratio: {g('appendix_ratio'):.2f} | Removed pages: {g('total_removed_pages'):.0f}"
    )

    # Figures
    lines.append(
        f"- Figures: {g('num_figures'):.0f} ({g('figures_per_page'):.2f}/page) | "
        f"Figure images: {g('num_figure_images'):.0f} ({g('subfigs_per_figure'):.2f}/fig) | "
        f"Tables mentioned: {g('num_table_mentions'):.0f} | Figure mentions: {g('num_figure_mentions'):.0f} | "
        f"Fig mention/fig: {g('fig_mention_per_fig'):.2f} | Table mention density: {g('table_mention_density'):.2f}"
    )

    # Math
    lines.append(
        f"- Equations: {g('num_equations'):.0f} ({g('equations_per_page'):.2f}/page) | "
        f"Inline math: {g('num_inline_math'):.0f} | Display math: {g('num_display_math'):.0f} | "
        f"Math density: {g('math_density'):.2f} | "
        f"Theorems: {g('num_theorem_like'):.0f} | Definitions: {g('num_definitions'):.0f} | "
        f"Proofs: {g('num_proofs'):.0f} | Algorithms: {g('num_algorithms'):.0f} | "
        f"Theory score: {g('theory_score'):.0f} | Has proof: {yn(g('has_proof'))}"
    )

    # References & citations
    lines.append(
        f"- References: {g('number_of_cited_references'):.0f} ({g('refs_per_page'):.2f}/page) | "
        f"Bibliography: {g('number_of_bib_items'):.0f} | Refs/author: {g('refs_per_author'):.2f} | "
        f"Eq refs: {g('num_eq_refs'):.0f} | Section refs: {g('num_section_refs'):.0f} | "
        f"Appendix refs: {g('num_appendix_refs'):.0f} | Cross-ref density: {g('cross_ref_density'):.2f}"
    )

    # Citation patterns
    lines.append(
        f"- Citation years: mean={g('cite_year_mean'):.1f} median={g('cite_year_median'):.1f} "
        f"std={g('cite_year_std'):.1f} max={g('cite_year_max'):.0f} | "
        f"Recency gap: {g('cite_year_recency'):.1f} | "
        f"Recent cites: {g('pct_recent_cites'):.0%} | Old cites: {g('pct_old_cites'):.0%} | "
        f"Parens cites: {g('num_cite_parens'):.0f} | Bracket cites: {g('num_cite_brackets'):.0f}"
    )

    # Authors
    lines.append(
        f"- Authors: {g('num_authors'):.0f} | Pages/author: {g('pages_per_author'):.2f} | "
        f"Single author: {yn(g('is_single_author'))} | Large team: {yn(g('is_large_team'))} | "
        f"Avg name len: {g('avg_author_name_len'):.1f} | Avg name parts: {g('avg_name_parts'):.1f}"
    )

    # Title
    lines.append(
        f"- Title: {g('title_words'):.0f} words, {g('title_chars'):.0f} chars | "
        f"Has colon: {yn(g('title_has_colon'))} | Has question: {yn(g('title_has_question'))} | "
        f"Caps ratio: {g('title_caps_ratio'):.2f} | Acronyms: {g('title_num_acronyms'):.0f} | "
        f"Has number: {yn(g('title_has_number'))} | Has hyphen: {yn(g('title_has_hyphen'))} | "
        f"Has parens: {yn(g('title_has_parens'))} | Avg word len: {g('title_avg_word_len'):.1f}"
    )

    # Abstract
    lines.append(
        f"- Abstract: {g('abstract_words'):.0f} words, {g('abstract_chars'):.0f} chars, "
        f"{g('abstract_sentences'):.0f} sentences | Avg sent len: {g('abstract_avg_sent_len'):.1f} | "
        f"Avg word len: {g('abstract_avg_word_len'):.1f} | Vocab richness: {g('abstract_vocab_richness'):.3f} | "
        f"Has numbers: {g('abstract_has_numbers'):.3f}"
    )

    # Full text quality
    lines.append(
        f"- Total words: {g('total_words'):.0f} | Sentences: {g('total_sentences'):.0f} | "
        f"Avg sent len: {g('avg_sentence_length'):.1f} | Sent len std: {g('sentence_len_std'):.1f} | "
        f"Sent len CV: {g('sentence_len_cv'):.2f} | Avg word len: {g('avg_word_length'):.1f} | "
        f"Long words: {g('long_words_ratio'):.3f} | Vocab richness 5k: {g('vocab_richness_5k'):.3f} | "
        f"Hapax ratio: {g('hapax_ratio'):.3f}"
    )

    # Tokens
    lines.append(
        f"- Text tokens: {g('num_text_tokens'):.0f} ({g('tokens_per_page'):.0f}/page) | "
        f"Text+image tokens: {g('num_text_image_tokens'):.0f} | Vision tokens: {g('num_vision_tokens'):.0f} | "
        f"Image token ratio: {g('image_token_ratio'):.3f} | Vision token ratio: {g('vision_token_ratio'):.3f}"
    )

    # Structure
    lines.append(
        f"- Sections: {g('total_sections'):.0f} (h1={g('num_h1'):.0f}, h2={g('num_h2'):.0f}, h3={g('num_h3'):.0f}) | "
        f"Depth: {g('section_depth'):.2f} | Sections/page: {g('sections_per_page'):.2f} | "
        f"Body items: {g('body_items_count'):.0f} | Intro idx: {g('intro_idx'):.0f}"
    )

    # Structure indicators
    lines.append(
        f"- Has related work: {yn(g('has_related_work'))} | Has conclusion: {yn(g('has_conclusion'))} | "
        f"Has limitations: {yn(g('has_limitations'))} | Has ethics: {yn(g('has_ethics'))} | "
        f"Has reproducibility: {yn(g('has_reproducibility'))} | Has future work: {yn(g('has_future_work'))} | "
        f"Has background: {yn(g('has_background'))}"
    )

    # Readability
    lines.append(
        f"- Flesch-Kincaid: {g('flesch_kincaid'):.1f} | Coleman-Liau: {g('coleman_liau'):.1f} | "
        f"Avg syllables: {g('avg_syllables'):.2f} | Word len std: {g('word_len_std'):.2f}"
    )

    # Experimental rigor
    lines.append(
        f"- Ablation: {yn(g('mentions_ablation'))} | Baseline: {yn(g('mentions_baseline'))} ({g('num_baselines'):.0f}x) | "
        f"SOTA: {yn(g('mentions_sota'))} | Benchmark: {yn(g('mentions_benchmark'))} | "
        f"Hyperparams: {yn(g('mentions_hyperparameter'))} | Significance: {yn(g('mentions_significance'))} | "
        f"Code: {yn(g('mentions_code'))} | GitHub: {yn(g('has_github'))} | "
        f"Datasets mentioned: {g('num_datasets_mentioned'):.0f} | Training details: {g('mentions_training_details'):.0f}"
    )

    # Sentiment / rhetoric
    lines.append(
        f"- Hedge: {g('hedge_count'):.0f} (ratio={g('hedge_ratio'):.2f}) | "
        f"Confidence: {g('confidence_count'):.0f} (ratio={g('confidence_ratio'):.2f}) | "
        f"Positive: {g('positive_count'):.0f} | Negative: {g('negative_count'):.0f} | "
        f"Pos/neg ratio: {g('pos_neg_ratio'):.2f} | Rhetoric balance: {g('rhetoric_balance'):.2f}"
    )

    # Novelty language
    lines.append(
        f"- Novel: {g('mentions_novel'):.0f} | First: {g('mentions_first'):.0f} | "
        f"Propose: {g('mentions_propose'):.0f} | Contribution: {g('mentions_contribution'):.0f} | "
        f"Key insight: {g('mentions_key_insight'):.0f} | Self-citation words: {g('self_citation_words'):.0f}"
    )

    # Intro
    lines.append(
        f"- Intro words: {g('intro_words'):.0f} | Intro ratio: {g('intro_ratio'):.3f}"
    )

    # Formatting
    lines.append(
        f"- Bullets: {g('num_bullets'):.0f} | Numbered lists: {g('num_numbered_lists'):.0f} | "
        f"Bold: {g('num_bold'):.0f} | Uses lists: {yn(g('uses_lists'))}"
    )

    # Topics
    topic_keys = [k for k in feats if k.startswith('topic_') and not k.endswith('_density') and not k.startswith('is_dominant_')]
    if topic_keys:
        topic_parts = []
        for tk in sorted(topic_keys):
            count = g(tk)
            density = g(tk + '_density')
            dominant = g('is_dominant_' + tk)
            name = tk.replace('topic_', '')
            if count > 0:
                marker = "*" if dominant else ""
                topic_parts.append(f"{name}={count:.0f}{marker}")
        if topic_parts:
            lines.append(f"- Topics: {', '.join(topic_parts)} (* = dominant)")

    # Ratios
    lines.append(
        f"- Figures/ref: {g('figures_per_ref'):.3f} | Equations/ref: {g('equations_per_ref'):.3f} | "
        f"Authors x refs: {g('authors_x_refs'):.0f} | Length x refs: {g('length_x_refs'):.2f} | "
        f"Figures x tables: {g('figures_x_tables'):.0f} | Theory x experiment: {g('theory_x_experiment'):.0f}"
    )

    # Year
    lines.append(f"- Year: {g('year'):.0f}")

    return "\n".join(lines)


def format_paper_stats(stats_row: dict) -> str:
    lines = ["\n\n---\n\nPaper Statistics:"]
    for csv_field, label in PAPER_STATS_FIELDS:
        val = stats_row.get(csv_field, "N/A")
        lines.append(f"- {label}: {val}")
    return "\n".join(lines)


def format_qwen_review(review: dict) -> str:
    return (
        f"\n\n---\n\nAI Review:\n"
        f"Paper Quality: {review['paper_quality']}\n\n"
        f"Strengths: {review['strengths']}\n\n"
        f"Weaknesses: {review['weaknesses']}\n\n"
        f"Most Important Strength: {review['most_important_strength']}\n\n"
        f"Most Important Weakness: {review['most_important_weakness']}"
    )


def format_gemini_review(review: dict) -> str:
    def format_list_field(val):
        if isinstance(val, list):
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(val))
        return str(val)

    strengths = format_list_field(review.get("strengths", ""))
    weaknesses = format_list_field(review.get("weaknesses", ""))

    return (
        f"\n\n---\n\nAI Review:\n"
        f"Paper Quality: {review['paper_quality']}\n\n"
        f"Strengths:\n{strengths}\n\n"
        f"Weaknesses:\n{weaknesses}\n\n"
        f"Most Important Strength: {review['most_important_strength']}\n\n"
        f"Most Important Weakness: {review['most_important_weakness']}\n\n"
        f"Relation to Other Work: {review['relation_to_other_work']}\n\n"
        f"Critical Disputes: {review['critical_disputes']}"
    )


def append_extras_to_data(
    data: list[dict],
    extras: dict,
    variant: str,
    formatter,
    split: str,
) -> tuple[list[dict], int, int]:
    """Append formatted extras to human turn. Returns (new_data, matched, unmatched)."""
    new_data = []
    matched = 0
    unmatched = 0

    for entry in data:
        new_entry = json.loads(json.dumps(entry))  # deep copy
        sid = new_entry.get("_metadata", {}).get("submission_id")

        if sid and sid in extras:
            extra_info = extras[sid]
            if variant in ("paper_stats", "paper_stats_full"):
                text = formatter(extra_info)
                source_file = "all_paperstats_v7.json" if variant == "paper_stats_full" else "massive_metadata_v7.csv"
                candidate_idx = None
            else:
                # Pick first valid candidate
                candidate = extra_info[0]
                review = candidate["review"]
                text = formatter(review)
                source_file = candidate.get("_source_file", "unknown")
                candidate_idx = candidate.get("candidate_idx", 0)

            # Append to the last human turn
            for conv in reversed(new_entry["conversations"]):
                if conv.get("from") == "human":
                    conv["value"] += text
                    break

            # Add provenance to metadata
            new_entry["_metadata"]["_review_source_file"] = source_file
            new_entry["_metadata"]["_review_candidate_idx"] = candidate_idx
            matched += 1
        else:
            unmatched += 1

        new_data.append(new_entry)

    return new_data, matched, unmatched


def append_extras_multi(
    data: list[dict],
    extras: dict,
    variant: str,
    formatter,
    split: str,
) -> tuple[list[dict], int, int]:
    """Create exactly 2 entries per matched paper. Unmatched papers are included once (no review)."""
    new_data = []
    matched = 0
    unmatched = 0

    for entry in data:
        sid = entry.get("_metadata", {}).get("submission_id")

        if sid and sid in extras:
            candidates = extras[sid]
            n_copies = 3 if "_x3" in variant else 2
            for candidate in candidates[:n_copies]:
                new_entry = json.loads(json.dumps(entry))  # deep copy
                review = candidate["review"]
                text = formatter(review)
                source_file = candidate.get("_source_file", "unknown")
                candidate_idx = candidate.get("candidate_idx", 0)

                for conv in reversed(new_entry["conversations"]):
                    if conv.get("from") == "human":
                        conv["value"] += text
                        break

                new_entry["_metadata"]["_review_source_file"] = source_file
                new_entry["_metadata"]["_review_candidate_idx"] = candidate_idx
                new_data.append(new_entry)
            matched += 1
        else:
            # Include unmatched entries once with no review
            new_data.append(json.loads(json.dumps(entry)))
            unmatched += 1

    return new_data, matched, unmatched


def _sanitize_surrogates(obj):
    """Recursively replace surrogate characters in strings."""
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: _sanitize_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_surrogates(v) for v in obj]
    return obj


def write_dataset(data: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _sanitize_surrogates(data)
    with open(output_dir / "data.json", "w") as f:
        json.dump(data, f, indent=None, ensure_ascii=False)
    print(f"  Wrote {len(data)} entries to {output_dir / 'data.json'}")


def update_dataset_info(new_entries: dict[str, dict]):
    info_path = DATA_DIR / "dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)
    info.update(new_entries)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Updated {info_path} with {len(new_entries)} new entries")


def make_dataset_info_entry(dataset_name: str, split: str, is_vision: bool) -> dict:
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


def main():
    parser = argparse.ArgumentParser(description="Create extras-appended datasets")
    parser.add_argument(
        "--variant",
        required=True,
        choices=list(VARIANT_SUFFIXES.keys()),
    )
    args = parser.parse_args()

    variant = args.variant
    suffix = VARIANT_SUFFIXES[variant]

    # Load extras
    print(f"Loading extras for variant: {variant}")
    if variant == "paper_stats":
        extras = load_paper_stats()
        formatter = format_paper_stats
    elif variant == "paper_stats_full":
        extras = load_all_paperstats()
        formatter = format_all_paper_stats
    elif variant in ("qwen_reviews", "qwen_reviews_x2"):
        extras = load_qwen_reviews()
        formatter = format_qwen_review
    elif variant in ("gemini_reviews", "gemini_reviews_x2", "gemini_reviews_x3"):
        extras = load_gemini_reviews()
        formatter = format_gemini_review
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # For single-candidate review variants, filter to >=1 valid candidate
    if variant in ("gemini_reviews", "qwen_reviews"):
        before = len(extras)
        extras = {sid: cands for sid, cands in extras.items() if len(cands) >= 1}
        print(f"Filtered to >=1 valid candidates: {before} -> {len(extras)}")
    elif variant in MULTI_VARIANTS:
        min_cands = 3 if "_x3" in variant else 2
        before = len(extras)
        extras = {sid: cands for sid, cands in extras.items() if len(cands) >= min_cands}
        print(f"Filtered to >=2 valid candidates: {before} -> {len(extras)}")

    print(f"Loaded extras for {len(extras)} papers")

    new_info_entries = {}

    for modality, source_dataset in SOURCE_DATASETS.items():
        is_vision = modality == "vision"
        output_dataset = f"{source_dataset}_{suffix}"
        print(f"\n{'='*60}")
        print(f"Processing: {modality} -> {output_dataset}")

        for split in SPLITS:
            print(f"\n  Split: {split}")
            data = load_source_data(source_dataset, split)
            print(f"  Loaded {len(data)} entries")

            if variant in MULTI_VARIANTS:
                new_data, matched, unmatched = append_extras_multi(
                    data, extras, variant, formatter, split
                )
            else:
                new_data, matched, unmatched = append_extras_to_data(
                    data, extras, variant, formatter, split
                )
            print(f"  Matched: {matched}, Unmatched: {unmatched}")

            output_dir = DATA_DIR / f"{output_dataset}_{split}"
            write_dataset(new_data, output_dir)

            # Register in dataset_info
            entry_name = f"{output_dataset}_{split}"
            new_info_entries[entry_name] = make_dataset_info_entry(
                output_dataset, split, is_vision
            )

    update_dataset_info(new_info_entries)
    print(f"\nDone! Created {len(new_info_entries)} dataset entries for variant '{variant}'.")


if __name__ == "__main__":
    main()
