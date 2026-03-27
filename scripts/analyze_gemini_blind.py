#!/usr/bin/env python3
"""
Analyze Gemini blind text review diversity and content.

No accept/reject field available, so this focuses on:
- Coverage stats (how many papers, candidates, validity)
- Field length / content analysis
- Multi-generation diversity (similarity across candidates)
- Concrete examples
"""

import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from itertools import combinations

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REVIEW_FILE = "results/reviews/gemini_blind_text_all_extracted.jsonl"
GT_DIRS = [
    "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train",
    "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test",
    "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation",
]
METADATA_ARROW_DIR = "data/massive_metadata_v7_5"
METADATA_ARROW_SHARDS = 4  # use the 4-shard version (has original_reviews)
OUTPUT = "results/gemini_analysis.md"

# Gemini has strengths/weaknesses as lists; other fields as strings
LIST_FIELDS = ["strengths", "weaknesses"]
STRING_FIELDS = [
    "critical_disputes", "most_important_strength",
    "most_important_weakness", "relation_to_other_work", "paper_quality",
]
ALL_FIELDS = LIST_FIELDS + STRING_FIELDS
NUM_CANDIDATES = 3
SAMPLE_SIZE_DIVERSITY = 500
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def field_to_text(val):
    """Convert a field value (str or list) to a single string."""
    if isinstance(val, list):
        return " ".join(str(x) for x in val)
    return str(val) if val else ""


def is_valid_review(entry: dict) -> bool:
    """Check if a review entry has substantive content in all fields."""
    review = entry.get("review")
    if not isinstance(review, dict):
        return False
    for field in ALL_FIELDS:
        text = field_to_text(review.get(field))
        if len(text) < 20:
            return False
    return True


def jaccard_words(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    def ngrams(s):
        s = s.lower()
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    sa, sb = ngrams(a), ngrams(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def fmt_pct(num, denom):
    if denom == 0:
        return "N/A"
    return f"{num/denom*100:.1f}%"


def avg(lst):
    return sum(lst) / len(lst) if lst else 0


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_reviews():
    reviews = defaultdict(list)
    total = 0
    valid = 0
    with open(REVIEW_FILE) as f:
        for line in f:
            total += 1
            entry = json.loads(line)
            if is_valid_review(entry):
                valid += 1
            reviews[entry["submission_id"]].append(entry)
    print(f"Total entries: {total}, valid: {valid}")
    return reviews, total, valid


def load_ground_truth():
    gt = {}
    for d in GT_DIRS:
        p = Path(d) / "data.json"
        if not p.exists():
            print(f"  WARNING: {p} not found")
            continue
        with open(p) as f:
            data = json.load(f)
        for item in data:
            meta = item.get("_metadata", {})
            sid = meta.get("submission_id")
            if sid:
                gt[sid] = meta
    print(f"Ground truth papers: {len(gt)}")
    return gt


def load_human_reviews(sids: set) -> dict:
    """Load original human reviews from arrow shards for given submission IDs."""
    import pyarrow as pa

    human = {}
    for i in range(METADATA_ARROW_SHARDS):
        path = Path(METADATA_ARROW_DIR) / f"data-{i:05d}-of-{METADATA_ARROW_SHARDS:05d}.arrow"
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue
        reader = pa.ipc.open_stream(str(path))
        table = reader.read_all()
        sid_col = table.column("submission_id").to_pylist()
        for j, sid in enumerate(sid_col):
            if sid in sids:
                raw_reviews = table.column("original_reviews")[j].as_py()
                raw_meta = table.column("original_metareview")[j].as_py()
                title = table.column("title")[j].as_py()
                reviews = json.loads(raw_reviews) if raw_reviews else []
                metareview = json.loads(raw_meta) if raw_meta else {}
                human[sid] = {
                    "title": title,
                    "reviews": reviews if isinstance(reviews, list) else [],
                    "metareview": metareview if isinstance(metareview, dict) else {},
                }
    print(f"Loaded human reviews for {len(human)}/{len(sids)} requested papers")
    return human


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Gemini Blind Text Review Analysis")
    print("=" * 60)

    reviews_by_paper, total_entries, valid_entries = load_reviews()
    gt = load_ground_truth()

    # Candidate counts
    cand_counts = Counter(len(v) for v in reviews_by_paper.values())
    papers_3 = {sid: entries for sid, entries in reviews_by_paper.items() if len(entries) >= NUM_CANDIDATES}
    # Deduplicate by candidate_idx
    for sid in papers_3:
        seen = set()
        unique = []
        for e in sorted(papers_3[sid], key=lambda x: x["candidate_idx"]):
            if e["candidate_idx"] not in seen:
                seen.add(e["candidate_idx"])
                unique.append(e)
        papers_3[sid] = unique[:NUM_CANDIDATES]

    print(f"Unique papers: {len(reviews_by_paper)}")
    print(f"Papers with 3 candidates: {len(papers_3)}")

    # Merge with GT where available
    merged = []
    for sid, candidates in papers_3.items():
        meta = gt.get(sid, {})
        merged.append({
            "sid": sid,
            "candidates": candidates,
            "year": candidates[0].get("year") or meta.get("year"),
            "gt_answer": meta.get("answer"),
            "pct_rating": meta.get("pct_rating"),
        })

    report = []
    report.append("# Gemini Blind Text Review Analysis\n")

    # ===== Coverage =====
    report.append("## Coverage\n")
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Total review entries | {total_entries} |")
    report.append(f"| Valid entries (all fields ≥20 chars) | {valid_entries} ({fmt_pct(valid_entries, total_entries)}) |")
    report.append(f"| Unique papers | {len(reviews_by_paper)} |")
    report.append(f"| Papers with 3/3 candidates | {len(papers_3)} |")
    report.append(f"| Papers with ground truth | {sum(1 for m in merged if m['gt_answer'])} |")
    report.append("")

    report.append("### Candidates per Paper\n")
    report.append("| # Candidates | Papers |")
    report.append("|-------------|--------|")
    for k in sorted(cand_counts.keys()):
        report.append(f"| {k} | {cand_counts[k]} |")
    report.append("")

    # By year
    by_year = defaultdict(list)
    for m in merged:
        if m["year"]:
            by_year[m["year"]].append(m)
    report.append("### Papers by Year\n")
    report.append("| Year | Papers (3 cands) | With GT |")
    report.append("|------|-----------------|---------|")
    for yr in sorted(by_year.keys()):
        subset = by_year[yr]
        with_gt = sum(1 for m in subset if m["gt_answer"])
        report.append(f"| {yr} | {len(subset)} | {with_gt} |")
    report.append("")

    # ===== Field Length Stats =====
    report.append("## Field Length Statistics\n")
    report.append("### Character Counts (all entries)\n")
    field_lens = defaultdict(list)
    for m in merged:
        for c in m["candidates"]:
            rev = c["review"]
            for field in ALL_FIELDS:
                text = field_to_text(rev.get(field))
                field_lens[field].append(len(text))

    report.append("| Field | Type | Mean | Median | Min | Max |")
    report.append("|-------|------|------|--------|-----|-----|")
    for field in ALL_FIELDS:
        vals = sorted(field_lens[field])
        if not vals:
            continue
        ftype = "list" if field in LIST_FIELDS else "str"
        mean = sum(vals) / len(vals)
        median = vals[len(vals)//2]
        report.append(f"| {field} | {ftype} | {mean:.0f} | {median} | {vals[0]} | {vals[-1]} |")
    report.append("")

    # Strengths vs weaknesses length comparison
    s_lens = field_lens["strengths"]
    w_lens = field_lens["weaknesses"]
    report.append(f"**Strengths vs Weaknesses**: Mean strengths={avg(s_lens):.0f} chars, mean weaknesses={avg(w_lens):.0f} chars  ")
    if avg(w_lens) > avg(s_lens):
        report.append(f"Weaknesses are {avg(w_lens)-avg(s_lens):.0f} chars longer on average (critical bias)  ")
    else:
        report.append(f"Strengths are {avg(s_lens)-avg(w_lens):.0f} chars longer on average  ")
    report.append("")

    # List item counts for strengths/weaknesses
    report.append("### Bullet Point Counts (strengths/weaknesses are lists)\n")
    for field in LIST_FIELDS:
        counts = []
        for m in merged:
            for c in m["candidates"]:
                val = c["review"].get(field, [])
                if isinstance(val, list):
                    counts.append(len(val))
        if counts:
            report.append(f"**{field}**: mean={avg(counts):.1f} items, median={sorted(counts)[len(counts)//2]}, min={min(counts)}, max={max(counts)}  ")
    report.append("")

    # ===== Multi-Generation Diversity =====
    report.append("## Multi-Generation Diversity Analysis\n")

    # Content similarity on sample
    sample = random.sample(merged, min(SAMPLE_SIZE_DIVERSITY, len(merged)))

    similarity_results = {}
    for field in ALL_FIELDS:
        jaccards = []
        ngrams_list = []
        for m in sample:
            cands = m["candidates"]
            for i, j in combinations(range(NUM_CANDIDATES), 2):
                t_i = field_to_text(cands[i]["review"].get(field))
                t_j = field_to_text(cands[j]["review"].get(field))
                jaccards.append(jaccard_words(t_i, t_j))
                ngrams_list.append(ngram_overlap(t_i, t_j))
        similarity_results[field] = {
            "jaccard": avg(jaccards),
            "ngram": avg(ngrams_list),
        }

    report.append(f"### Content Similarity (sample of {len(sample)} papers, pairwise between 3 candidates)\n")
    report.append("| Field | Word Jaccard | 3-gram Overlap |")
    report.append("|-------|-------------|----------------|")
    for field in ALL_FIELDS:
        r = similarity_results[field]
        report.append(f"| {field} | {r['jaccard']:.4f} | {r['ngram']:.4f} |")
    report.append("")

    # Overall similarity (average across key fields)
    key_fields = ["strengths", "weaknesses", "critical_disputes"]
    avg_jaccard = avg([similarity_results[f]["jaccard"] for f in key_fields])
    avg_ngram = avg([similarity_results[f]["ngram"] for f in key_fields])
    report.append(f"**Average similarity (strengths+weaknesses+critical_disputes)**: Jaccard={avg_jaccard:.4f}, 3-gram={avg_ngram:.4f}  \n")

    # ===== Concrete Examples (with human reviews side-by-side) =====
    report.append("## Concrete Examples (Gemini vs Human Reviews)\n")

    # Compute per-paper similarity for strengths
    paper_sims = []
    for m in merged:
        cands = m["candidates"]
        sims = []
        for i, j in combinations(range(NUM_CANDIDATES), 2):
            t_i = field_to_text(cands[i]["review"].get("strengths"))
            t_j = field_to_text(cands[j]["review"].get("strengths"))
            sims.append(jaccard_words(t_i, t_j))
        paper_sims.append((avg(sims), m))
    paper_sims.sort(key=lambda x: x[0])

    # Pick examples
    example_papers = []
    # 1. High similarity
    example_papers.append((f"Example 1: Highest Candidate Similarity (Jaccard={paper_sims[-1][0]:.3f})", paper_sims[-1][1]))
    # 2. Low similarity
    example_papers.append((f"Example 2: Lowest Candidate Similarity (Jaccard={paper_sims[0][0]:.3f})", paper_sims[0][1]))
    # 3. Median
    mid = len(paper_sims) // 2
    example_papers.append((f"Example 3: Median Similarity (Jaccard={paper_sims[mid][0]:.3f})", paper_sims[mid][1]))
    # 4. GT Accept
    gt_accept_papers = [m for m in merged if m["gt_answer"] == "Accept"]
    if gt_accept_papers:
        example_papers.append(("Example 4: Ground Truth Accept Paper", random.choice(gt_accept_papers)))
    # 5. GT Reject
    gt_reject_papers = [m for m in merged if m["gt_answer"] == "Reject"]
    if gt_reject_papers:
        example_papers.append(("Example 5: Ground Truth Reject Paper", random.choice(gt_reject_papers)))

    # Load human reviews for all example papers
    example_sids = {m["sid"] for _, m in example_papers}
    human_reviews = load_human_reviews(example_sids)

    for label, m in example_papers:
        lines = []
        lines.append(f"### {label}\n")
        sid = m["sid"]
        hr = human_reviews.get(sid, {})
        title = hr.get("title", "Unknown")
        lines.append(f"**Paper**: `{sid}` — *{title}*  ")
        lines.append(f"**Year**: {m['year']}  ")
        if m["gt_answer"]:
            lines.append(f"**GT Decision**: {m['gt_answer']}  ")
            lines.append(f"**pct_rating**: {m['pct_rating']}  ")
        # Human review ratings
        h_revs = hr.get("reviews", [])
        if h_revs:
            ratings = [r.get("rating") for r in h_revs if r.get("rating") is not None]
            lines.append(f"**Human reviewer ratings**: {ratings} (mean={avg(ratings):.1f})  ")
        h_meta = hr.get("metareview", {})
        if h_meta:
            meta_text = h_meta.get("metareview", "")
            if meta_text:
                display = meta_text[:400]
                if len(meta_text) > 400:
                    display += "..."
                lines.append(f"**Metareview**: {display}  ")
        lines.append("")

        # --- Human reviews ---
        lines.append("#### Human Reviews\n")
        for ri, rev in enumerate(h_revs):
            rating = rev.get("rating", "?")
            confidence = rev.get("confidence", "?")
            lines.append(f"<details><summary>Human Reviewer {ri+1} (rating={rating}, confidence={confidence})</summary>\n")
            for field in ["summary", "strengths", "weaknesses", "questions"]:
                val = rev.get(field, "")
                if val:
                    display = str(val)[:800]
                    if len(str(val)) > 800:
                        display += "..."
                    lines.append(f"**{field}**:\n{display}\n")
            lines.append("</details>\n")

        # --- Gemini reviews ---
        lines.append("#### Gemini Reviews (3 candidates)\n")
        for idx, c in enumerate(m["candidates"]):
            rev = c["review"]
            lines.append(f"<details><summary>Gemini Candidate {idx}</summary>\n")
            for field in ALL_FIELDS:
                val = rev.get(field, "")
                text = field_to_text(val)
                display = text[:800]
                if len(text) > 800:
                    display += "..."
                lines.append(f"**{field}**:\n{display}\n")
            lines.append("</details>\n")

        lines.append("---\n")
        report.append("\n".join(lines))

    # ===== Key Findings =====
    report.append("## Key Findings\n")
    report.append(f"1. **Coverage**: {len(papers_3)} papers with 3/3 candidates out of {len(reviews_by_paper)} unique papers ({fmt_pct(len(papers_3), len(reviews_by_paper))})")
    report.append(f"2. **Field structure**: strengths/weaknesses are lists (avg {avg([len(c['review'].get('strengths',[])) for m in merged for c in m['candidates'] if isinstance(c['review'].get('strengths'), list)]):.1f} bullet points each)")
    report.append(f"3. **Content length**: Gemini reviews are substantially longer than Qwen — strengths avg {avg(s_lens):.0f} chars, weaknesses avg {avg(w_lens):.0f} chars")
    report.append(f"4. **Diversity**: Average pairwise Jaccard across key fields = {avg_jaccard:.3f} (lower = more diverse)")
    report.append(f"5. **No accept/reject field**: Gemini reviews use a different schema without a boolean accept field")
    report.append("")

    # Write
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write("\n".join(report))
    print(f"\nReport written to {OUTPUT}")
    print(f"Papers in report: {len(merged)}")


if __name__ == "__main__":
    main()
