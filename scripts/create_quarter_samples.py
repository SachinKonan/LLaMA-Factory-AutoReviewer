#!/usr/bin/env python3
"""Generate balanced quarter-scale and full-scale data for arxiv-search Codex experiments.

Quarter: 50 test (25 accept, 25 reject), 100 train (50 accept, 50 reject), seed=42.
Full: copy of existing unbiased data.

Also generates gated model-prior variants where model predictions are stripped from
TEST_SAMPLE.json and placed in a separate model_priors.json file, accessible only
via the get_model_prior MCP tool.

Sources:
  - coding_agents_2026_codex_unbiased/{TEST,TRAIN}_SAMPLE.json
  - coding_agents_2026_codex_with_best_model_and_model_prior_unbiased/TEST_SAMPLE.json
  - data/agent_ground_truth_2026_unbiased.json
"""

import json
import math
import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEED = 42

# Source dirs
SRC_UNBIASED = REPO / "coding_agents_2026_codex_unbiased"
SRC_MODEL = REPO / "coding_agents_2026_codex_with_best_model_and_model_prior_unbiased"
GT_PATH = REPO / "data" / "agent_ground_truth_2026_unbiased.json"

# Vision data.json for image path lookup
VISION_DATA_JSON = REPO / "data" / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test" / "data.json"

# Target dirs
TARGETS = {
    "full_no_model": REPO / "coding_agents_2026_codex_arxiv_unbiased",
    "full_model": REPO / "coding_agents_2026_codex_arxiv_with_model_unbiased",
    "quarter_no_model": REPO / "coding_agents_2026_codex_arxiv_unbiased_quarter",
    "quarter_model": REPO / "coding_agents_2026_codex_arxiv_with_model_unbiased_quarter",
}

# Pred-fixed: no-model variants with model preds stripped from test
PRED_FIXED_NO_MODEL_TARGETS = {
    "full_no_model_fixed": REPO / "coding_agents_2026_codex_arxiv_unbiased_pred_fixed",
    "quarter_no_model_fixed": REPO / "coding_agents_2026_codex_arxiv_unbiased_pred_fixed_quarter",
}

# Pred-fixed: model variants with model predictions in both train and test
PRED_FIXED_MODEL_TARGETS = {
    "full_model_fixed": REPO / "coding_agents_2026_codex_arxiv_with_model_unbiased_pred_fixed",
    "quarter_model_fixed": REPO / "coding_agents_2026_codex_arxiv_with_model_unbiased_pred_fixed_quarter",
}

# Gated model-prior target dirs (model predictions stripped from test, placed in model_priors.json)
GATED_TARGETS = {
    "full_gated": REPO / "coding_agents_2026_codex_arxiv_with_model_unbiased_gated",
    "quarter_gated": REPO / "coding_agents_2026_codex_arxiv_with_model_unbiased_gated_quarter",
}

# Gated v2.3 targets (with img_pages, delegated pipeline)
GATED_V23_TARGETS = {
    "full_gated_v2.3": REPO / "coding_agents_2026_codex_arxiv_with_model_unbiased_gated_v2.3",
    "quarter_gated_v2.3": REPO / "coding_agents_2026_codex_arxiv_with_model_unbiased_gated_v2.3_quarter",
}

# Gated v2.4 targets (two-role: reviewer + reconciler, spawn_agents_on_csv)
GATED_V24_TARGETS = {
    "full_gated_v2.4": REPO / "coding_agents_2026_codex_arxiv_with_model_unbiased_gated_v2.4",
    "quarter_gated_v2.4": REPO / "coding_agents_2026_codex_arxiv_with_model_unbiased_gated_v2.4_quarter",
}

# Gated v2.5 targets (meta-calibration on train before test)
GATED_V25_TARGETS = {
    "full_gated_v2.5": REPO / "coding_agents_2026_codex_arxiv_with_model_unbiased_gated_v2.5",
    "quarter_gated_v2.5": REPO / "coding_agents_2026_codex_arxiv_with_model_unbiased_gated_v2.5_quarter",
}

# Aligned structured review targets (human reviews in training data)
ALIGNED_TARGETS = {
    "quarter_aligned_no_model_v2": REPO / "coding_agents_2026_codex_arxiv_aligned_no_model_v2_quarter",
    "quarter_aligned_with_model_v2": REPO / "coding_agents_2026_codex_arxiv_aligned_with_model_v2_quarter",
}

# Aligned v3 targets: full-scale (200 test, 400 train), identical instructions except data
ALIGNED_V3_TARGETS = {
    "full_aligned_no_model_v3": REPO / "coding_agents_2026_codex_arxiv_aligned_no_model_v3",
    "full_aligned_with_model_v3": REPO / "coding_agents_2026_codex_arxiv_aligned_with_model_v3",
}

# Aligned v3 targets for 2025 papers
ALIGNED_V3_2025_TARGETS = {
    "full_aligned_no_model_v3_2025": REPO / "coding_agents_2025_codex_arxiv_aligned_no_model_v3",
    "full_aligned_with_model_v3_2025": REPO / "coding_agents_2025_codex_arxiv_aligned_with_model_v3",
}

# 2025 data sources (text content from dataset test split, predictions from SFT model)
TEXT_TEST_DATA_JSON = REPO / "data" / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test" / "data.json"
VISION_PREDICTIONS_PATH = REPO / "results" / "final_sweep_v7_datasweepv3" / "optim_search_2026" / "bz16_lr1e-6_vision" / "finetuned-ckpt-2648.jsonl"

# Metadata path for human reviews
METADATA_PATH = REPO / "data" / "massive_metadata_v7_5"


def load_img_pages_lookup(max_pages: int = 10) -> dict[str, list[str]]:
    """Build submission_id -> list of image paths from the vision data.json.

    Returns paths relative to the agent workspace via the paper_images symlink,
    e.g. 'paper_images/{sid}/page_1_noreferences_original.png'.
    """
    with VISION_DATA_JSON.open() as f:
        rows = json.load(f)
    lookup: dict[str, list[str]] = {}
    for row in rows:
        sid = row["_metadata"]["submission_id"]
        images = row.get("images", [])
        # Convert from 'data/images/{sid}/...' to 'paper_images/{sid}/...'
        rel_paths = []
        for img_path in images[:max_pages]:
            # img_path looks like 'data/images/{sid}/page_N_noreferences_original.png'
            parts = Path(img_path).parts  # ('data', 'images', sid, filename)
            if len(parts) >= 4:
                rel_paths.append(f"paper_images/{parts[2]}/{parts[3]}")
            else:
                rel_paths.append(f"paper_images/{Path(img_path).name}")
        lookup[sid] = rel_paths
    return lookup


def add_img_pages(data: dict, img_lookup: dict[str, list[str]]) -> dict:
    """Add img_pages field to each paper entry using the vision data lookup."""
    result = {}
    for sid, paper in data.items():
        entry = dict(paper)
        entry["img_pages"] = img_lookup.get(sid, [])
        result[sid] = entry
    return result


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"  Wrote {path} ({len(data)} entries)")


def get_label(sid: str, paper: dict, gt: dict):
    """Get label from ground truth or inline label field."""
    return gt.get(sid) or paper.get("label")


def balanced_sample(data: dict, gt: dict, n_per_class: int, seed: int) -> dict:
    """Sample n_per_class accepts and n_per_class rejects from data using ground truth or inline labels."""
    accepts = [sid for sid in data if get_label(sid, data[sid], gt) == "accept"]
    rejects = [sid for sid in data if get_label(sid, data[sid], gt) == "reject"]

    rng = random.Random(seed)
    rng.shuffle(accepts)
    rng.shuffle(rejects)

    sampled_ids = accepts[:n_per_class] + rejects[:n_per_class]
    return {sid: data[sid] for sid in sampled_ids}


def add_model_predictions(test_data: dict, model_test: dict) -> dict:
    """Add model_prediction and model_confidence from model source to test data."""
    result = {}
    for sid, paper in test_data.items():
        entry = dict(paper)
        if sid in model_test:
            entry["model_prediction"] = model_test[sid].get("model_prediction")
            entry["model_confidence"] = model_test[sid].get("model_confidence")
        result[sid] = entry
    return result


def strip_model_predictions(test_data: dict) -> dict:
    """Remove model_prediction and model_confidence from test data."""
    result = {}
    for sid, paper in test_data.items():
        entry = {k: v for k, v in paper.items() if k not in ("model_prediction", "model_confidence")}
        result[sid] = entry
    return result


def extract_model_priors(test_data: dict) -> dict:
    """Extract model_prediction and model_confidence into a separate lookup dict."""
    priors = {}
    for sid, paper in test_data.items():
        if "model_prediction" in paper:
            priors[sid] = {
                "model_prediction": paper["model_prediction"],
                "model_confidence": paper["model_confidence"],
            }
    return priors


def add_human_reviews(data: dict, meta_index: dict) -> dict:
    """Add human_reviews field to each paper entry from massive_metadata_v7_5.

    Keeps only ICLR review fields: rating, confidence, soundness, presentation,
    contribution, summary, strengths, weaknesses, questions.
    """
    REVIEW_FIELDS = [
        "rating", "confidence", "soundness", "presentation", "contribution",
        "summary", "strengths", "weaknesses", "questions",
    ]
    result = {}
    found = 0
    for sid, paper in data.items():
        entry = dict(paper)
        reviews = []
        meta = meta_index.get(sid)
        if meta and meta.get("original_reviews"):
            raw_reviews = meta["original_reviews"]
            if isinstance(raw_reviews, str):
                try:
                    raw_reviews = json.loads(raw_reviews)
                except (json.JSONDecodeError, TypeError):
                    raw_reviews = []
            if isinstance(raw_reviews, list):
                for rev in raw_reviews:
                    if not isinstance(rev, dict):
                        continue
                    filtered = {}
                    for field in REVIEW_FIELDS:
                        if field in rev:
                            filtered[field] = rev[field]
                    if filtered:
                        reviews.append(filtered)
        entry["human_reviews"] = reviews
        if reviews:
            found += 1
        result[sid] = entry
    print(f"    Human reviews: {found}/{len(data)} papers have reviews")
    return result


def add_model_predictions_to_train(train_data: dict, model_test: dict) -> dict:
    """Add model_prediction and model_confidence to training data for calibration."""
    result = {}
    for sid, paper in train_data.items():
        entry = dict(paper)
        if sid in model_test:
            entry["model_prediction"] = model_test[sid].get("model_prediction")
            entry["model_confidence"] = model_test[sid].get("model_confidence")
        result[sid] = entry
    return result


def _sanitize_element(elem: dict) -> dict | None:
    """Keep only safe fields from a content_list element."""
    KEEP_FIELDS = {
        "type", "text", "text_level", "text_format",
        "page_idx", "bbox",
        "image_caption", "image_footnote",
        "table_body", "table_caption", "table_footnote",
        "sub_type", "list_items",
    }
    cleaned = {k: v for k, v in elem.items() if k in KEEP_FIELDS}
    return cleaned if cleaned else None


def _extract_headers(content_list: list[dict]) -> list[str]:
    """Extract section headers from content_list."""
    return [
        elem["text"]
        for elem in content_list
        if elem.get("text_level") is not None and elem.get("text")
    ]


def build_2025_paper_pool(meta_index: dict):
    """Build 2025 paper pool from dataset test split with model predictions.

    Uses massive_metadata_v7_5 for paper content (content_list_json) and
    vision predictions JSONL for model predictions.

    Returns dict: sid -> {content_list, headers, label, model_prediction, model_confidence}
    """
    # Load vision test data to identify 2025 papers and get labels
    with VISION_DATA_JSON.open() as f:
        vision_rows = json.load(f)

    # Load predictions (1:1 aligned with vision rows)
    predictions = []
    with VISION_PREDICTIONS_PATH.open() as f:
        for line in f:
            predictions.append(json.loads(line))

    # Decision token index (token 5 = Accept/Reject in "Outcome: \boxed{Accept}")
    DECISION_TOKEN_IDX = 5

    # Build prediction + label lookup for 2025 papers
    paper_info = {}  # sid -> {label, model_prediction, model_confidence}
    for vrow, pred in zip(vision_rows, predictions):
        meta = vrow.get("_metadata", {})
        if meta.get("year") != 2025:
            continue
        sid = meta["submission_id"]
        label = meta.get("answer", "").lower()
        predict_text = pred.get("predict", "")
        if "Accept" in predict_text:
            decision = "accept"
        elif "Reject" in predict_text:
            decision = "reject"
        else:
            continue
        logprobs = pred.get("token_logprobs", [])
        conf_idx = min(DECISION_TOKEN_IDX, len(logprobs) - 1) if logprobs else 0
        confidence = round(math.exp(logprobs[conf_idx]), 4) if logprobs else 0.5
        paper_info[sid] = {
            "label": label,
            "model_prediction": decision,
            "model_confidence": confidence,
        }

    # Build paper pool using content from massive_metadata_v7_5
    pool = {}
    for sid, info in paper_info.items():
        meta_row = meta_index.get(sid)
        if not meta_row:
            continue
        content_json = meta_row.get("content_list_json", "[]")
        content_list_raw = json.loads(content_json) if isinstance(content_json, str) else content_json
        content_list = [c for elem in content_list_raw if (c := _sanitize_element(elem)) is not None]
        if not content_list:
            continue
        headers = _extract_headers(content_list)

        pool[sid] = {
            "content_list": content_list,
            "headers": headers,
            "label": info["label"],
            "model_prediction": info["model_prediction"],
            "model_confidence": info["model_confidence"],
        }

    return pool


def main():
    print("Loading source data ...")
    gt = load_json(GT_PATH)
    test_unbiased = load_json(SRC_UNBIASED / "TEST_SAMPLE.json")
    train_unbiased = load_json(SRC_UNBIASED / "TRAIN_SAMPLE.json")
    test_model = load_json(SRC_MODEL / "TEST_SAMPLE.json")

    # Source for training model predictions (same 400 papers, with model_prediction/model_confidence)
    train_model_src = REPO / "coding_agents_2026_codex_with_best_model_and_model_prior_2" / "TRAIN_SAMPLE.json"
    train_model = load_json(train_model_src)

    print(f"Ground truth: {len(gt)} entries")
    print(f"Test unbiased: {len(test_unbiased)} papers")
    print(f"Train unbiased: {len(train_unbiased)} papers")
    print(f"Test model: {len(test_model)} papers")
    print(f"Train model (for gated calibration): {len(train_model)} papers")

    # Quarter samples
    print("\nCreating quarter samples (seed={}) ...".format(SEED))
    quarter_test = balanced_sample(test_unbiased, gt, n_per_class=25, seed=SEED)
    quarter_train = balanced_sample(train_unbiased, gt, n_per_class=50, seed=SEED + 1)
    quarter_test_model = add_model_predictions(quarter_test, test_model)

    # Verify balance
    for name, data in [
        ("quarter_test", quarter_test),
        ("quarter_train", quarter_train),
    ]:
        labels = [get_label(sid, data[sid], gt) for sid in data]
        accept_count = labels.count("accept")
        reject_count = labels.count("reject")
        print(f"  {name}: {len(data)} total ({accept_count} accept, {reject_count} reject)")

    # Full model test (add model predictions to full unbiased test)
    full_test_model = add_model_predictions(test_unbiased, test_model)

    # Write all experiment dirs
    print("\nWriting experiment directories ...")

    for key, target_dir in TARGETS.items():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {key} -> {target_dir.name} ---")

        if "quarter" in key:
            test_data = quarter_test_model if "model" in key else quarter_test
            train_data = quarter_train
        else:
            test_data = full_test_model if "model" in key else test_unbiased
            train_data = train_unbiased

        save_json(test_data, target_dir / "TEST_SAMPLE.json")
        save_json(train_data, target_dir / "TRAIN_SAMPLE.json")

    # Write pred-fixed no-model dirs (model preds stripped from test, no preds in train)
    print("\nWriting pred-fixed no-model directories ...")

    for key, target_dir in PRED_FIXED_NO_MODEL_TARGETS.items():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {key} -> {target_dir.name} ---")

        if "quarter" in key:
            test_data = strip_model_predictions(quarter_test_model)
            train_data = quarter_train
        else:
            test_data = strip_model_predictions(full_test_model)
            train_data = train_unbiased

        save_json(test_data, target_dir / "TEST_SAMPLE.json")
        save_json(train_data, target_dir / "TRAIN_SAMPLE.json")

    # Write pred-fixed model dirs (model predictions in both train and test)
    print("\nWriting pred-fixed model directories ...")

    for key, target_dir in PRED_FIXED_MODEL_TARGETS.items():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {key} -> {target_dir.name} ---")

        if "quarter" in key:
            test_data = quarter_test_model
            train_data = quarter_train
        else:
            test_data = full_test_model
            train_data = train_unbiased

        save_json(test_data, target_dir / "TEST_SAMPLE.json")
        train_with_model = add_model_predictions_to_train(train_data, train_model)
        save_json(train_with_model, target_dir / "TRAIN_SAMPLE.json")

    # Write gated model-prior experiment dirs
    print("\nWriting gated model-prior directories ...")

    for key, target_dir in GATED_TARGETS.items():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {key} -> {target_dir.name} ---")

        if "quarter" in key:
            test_with_model = quarter_test_model
            train_data = quarter_train
        else:
            test_with_model = full_test_model
            train_data = train_unbiased

        # TEST_SAMPLE.json: stripped of model predictions
        test_stripped = strip_model_predictions(test_with_model)
        save_json(test_stripped, target_dir / "TEST_SAMPLE.json")

        # model_priors.json: stored in data/ dir, NOT in agent workspace
        # Agent must use MCP tool to access these — no filesystem shortcut
        priors = extract_model_priors(test_with_model)
        priors_filename = "model_priors_gated_quarter.json" if "quarter" in key else "model_priors_gated_full.json"
        save_json(priors, REPO / "data" / priors_filename)

        # TRAIN_SAMPLE.json: includes model predictions for calibration
        train_with_model = add_model_predictions_to_train(train_data, train_model)
        save_json(train_with_model, target_dir / "TRAIN_SAMPLE.json")

    # Write gated v2.3 experiment dirs (with img_pages)
    print("\nWriting gated v2.3 directories (with img_pages) ...")

    img_lookup = load_img_pages_lookup(max_pages=10)
    print(f"  Image lookup: {len(img_lookup)} papers with image paths")

    for key, target_dir in GATED_V23_TARGETS.items():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {key} -> {target_dir.name} ---")

        if "quarter" in key:
            test_with_model = quarter_test_model
            train_data = quarter_train
        else:
            test_with_model = full_test_model
            train_data = train_unbiased

        # TEST_SAMPLE.json: stripped of model predictions, with img_pages
        test_stripped = strip_model_predictions(test_with_model)
        test_with_images = add_img_pages(test_stripped, img_lookup)
        save_json(test_with_images, target_dir / "TEST_SAMPLE.json")

        # model_priors.json: stored in data/ dir
        priors = extract_model_priors(test_with_model)
        priors_filename = "model_priors_gated_v2.3_quarter.json" if "quarter" in key else "model_priors_gated_v2.3_full.json"
        save_json(priors, REPO / "data" / priors_filename)

        # TRAIN_SAMPLE.json: includes model predictions + img_pages
        train_with_model = add_model_predictions_to_train(train_data, train_model)
        train_with_images = add_img_pages(train_with_model, img_lookup)
        save_json(train_with_images, target_dir / "TRAIN_SAMPLE.json")

    # Write gated v2.4 experiment dirs (same data as v2.3, two-role pipeline)
    print("\nWriting gated v2.4 directories (with img_pages) ...")

    if not img_lookup:
        img_lookup = load_img_pages_lookup(max_pages=10)
        print(f"  Image lookup: {len(img_lookup)} papers with image paths")

    for key, target_dir in GATED_V24_TARGETS.items():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {key} -> {target_dir.name} ---")

        if "quarter" in key:
            test_with_model = quarter_test_model
            train_data = quarter_train
        else:
            test_with_model = full_test_model
            train_data = train_unbiased

        # TEST_SAMPLE.json: stripped of model predictions, with img_pages
        test_stripped = strip_model_predictions(test_with_model)
        test_with_images = add_img_pages(test_stripped, img_lookup)
        save_json(test_with_images, target_dir / "TEST_SAMPLE.json")

        # model_priors.json: stored in data/ dir
        priors = extract_model_priors(test_with_model)
        priors_filename = "model_priors_gated_v2.4_quarter.json" if "quarter" in key else "model_priors_gated_v2.4_full.json"
        save_json(priors, REPO / "data" / priors_filename)

        # TRAIN_SAMPLE.json: includes model predictions + img_pages
        train_with_model = add_model_predictions_to_train(train_data, train_model)
        train_with_images = add_img_pages(train_with_model, img_lookup)
        save_json(train_with_images, target_dir / "TRAIN_SAMPLE.json")

    # Write gated v2.5 experiment dirs (same data as v2.4, meta-calibration)
    print("\nWriting gated v2.5 directories (with img_pages) ...")

    for key, target_dir in GATED_V25_TARGETS.items():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {key} -> {target_dir.name} ---")

        if "quarter" in key:
            test_with_model = quarter_test_model
            train_data = quarter_train
        else:
            test_with_model = full_test_model
            train_data = train_unbiased

        # TEST_SAMPLE.json: stripped of model predictions, with img_pages
        test_stripped = strip_model_predictions(test_with_model)
        test_with_images = add_img_pages(test_stripped, img_lookup)
        save_json(test_with_images, target_dir / "TEST_SAMPLE.json")

        # model_priors.json: includes BOTH test and train IDs (for meta-calibration)
        test_priors = extract_model_priors(test_with_model)
        train_with_model = add_model_predictions_to_train(train_data, train_model)
        train_priors = extract_model_priors(train_with_model)
        all_priors = {**train_priors, **test_priors}
        priors_filename = "model_priors_gated_v2.5_quarter.json" if "quarter" in key else "model_priors_gated_v2.5_full.json"
        save_json(all_priors, REPO / "data" / priors_filename)

        # TRAIN_SAMPLE.json: includes model predictions + img_pages
        train_with_images = add_img_pages(train_with_model, img_lookup)
        save_json(train_with_images, target_dir / "TRAIN_SAMPLE.json")

    # Write aligned structured review experiment dirs
    print("\nWriting aligned structured review directories ...")

    # Load metadata for human reviews
    meta_index = {}
    if METADATA_PATH.exists():
        from datasets import load_from_disk

        meta_ds = load_from_disk(str(METADATA_PATH))
        for row in meta_ds:
            sid = row.get("submission_id")
            if sid:
                meta_index[sid] = row
        print(f"  Metadata index: {len(meta_index)} papers")
    else:
        print(f"  WARNING: metadata not found at {METADATA_PATH}, human_reviews will be empty")

    if not img_lookup:
        img_lookup = load_img_pages_lookup(max_pages=10)
        print(f"  Image lookup: {len(img_lookup)} papers with image paths")

    for key, target_dir in ALIGNED_TARGETS.items():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {key} -> {target_dir.name} ---")

        is_with_model = "with_model" in key

        # TEST_SAMPLE.json
        if is_with_model:
            # Keep model predictions in test
            test_data = quarter_test_model
        else:
            # Strip model predictions from test
            test_data = strip_model_predictions(quarter_test_model)
        test_data = add_img_pages(test_data, img_lookup)
        save_json(test_data, target_dir / "TEST_SAMPLE.json")

        # TRAIN_SAMPLE.json: human reviews + model predictions + img_pages
        train_data = dict(quarter_train)
        train_data = add_model_predictions_to_train(train_data, train_model)
        train_data = add_human_reviews(train_data, meta_index)
        train_data = add_img_pages(train_data, img_lookup)
        save_json(train_data, target_dir / "TRAIN_SAMPLE.json")

    # Write aligned v3 full-scale experiment dirs (200 test, 400 train)
    print("\nWriting aligned v3 full-scale directories ...")

    # Reuse metadata index already loaded above
    if not meta_index and METADATA_PATH.exists():
        from datasets import load_from_disk

        meta_ds = load_from_disk(str(METADATA_PATH))
        for row in meta_ds:
            sid = row.get("submission_id")
            if sid:
                meta_index[sid] = row
        print(f"  Metadata index: {len(meta_index)} papers")

    if not img_lookup:
        img_lookup = load_img_pages_lookup(max_pages=10)
        print(f"  Image lookup: {len(img_lookup)} papers with image paths")

    # Full-scale samples: 200 test (100 accept, 100 reject), 400 train (200 accept, 200 reject)
    full_aligned_test = balanced_sample(test_unbiased, gt, n_per_class=100, seed=SEED)
    full_aligned_train = balanced_sample(train_unbiased, gt, n_per_class=200, seed=SEED + 1)
    full_aligned_test_model = add_model_predictions(full_aligned_test, test_model)

    for name, data in [
        ("full_aligned_test", full_aligned_test),
        ("full_aligned_train", full_aligned_train),
    ]:
        labels = [get_label(sid, data[sid], gt) for sid in data]
        accept_count = labels.count("accept")
        reject_count = labels.count("reject")
        print(f"  {name}: {len(data)} total ({accept_count} accept, {reject_count} reject)")

    for key, target_dir in ALIGNED_V3_TARGETS.items():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {key} -> {target_dir.name} ---")

        is_with_model = "with_model" in key

        # TEST_SAMPLE.json
        if is_with_model:
            test_data = full_aligned_test_model
        else:
            test_data = strip_model_predictions(full_aligned_test_model)
        test_data = add_img_pages(test_data, img_lookup)
        save_json(test_data, target_dir / "TEST_SAMPLE.json")

        # TRAIN_SAMPLE.json: human reviews + model predictions + img_pages
        train_data = dict(full_aligned_train)
        train_data = add_model_predictions_to_train(train_data, train_model)
        train_data = add_human_reviews(train_data, meta_index)
        train_data = add_img_pages(train_data, img_lookup)
        save_json(train_data, target_dir / "TRAIN_SAMPLE.json")

    # Write aligned v3 2025 experiment dirs (200 test, 400 train from 2025 papers)
    print("\nWriting aligned v3 2025 directories ...")

    # Ensure metadata index is loaded (needed for content_list)
    if not meta_index and METADATA_PATH.exists():
        from datasets import load_from_disk
        meta_ds = load_from_disk(str(METADATA_PATH))
        for row in meta_ds:
            sid = row.get("submission_id")
            if sid:
                meta_index[sid] = row
        print(f"  Metadata index: {len(meta_index)} papers")

    paper_pool_2025 = build_2025_paper_pool(meta_index)
    accepts_2025 = {sid: p for sid, p in paper_pool_2025.items() if p["label"] == "accept"}
    rejects_2025 = {sid: p for sid, p in paper_pool_2025.items() if p["label"] == "reject"}
    print(f"  2025 pool: {len(paper_pool_2025)} papers ({len(accepts_2025)} accept, {len(rejects_2025)} reject)")

    # Sample test: 100 accept + 100 reject
    rng_2025 = random.Random(SEED + 2025)
    accept_sids = list(accepts_2025.keys())
    reject_sids = list(rejects_2025.keys())
    rng_2025.shuffle(accept_sids)
    rng_2025.shuffle(reject_sids)

    test_sids_2025 = accept_sids[:100] + reject_sids[:100]
    remaining_accept = accept_sids[100:]
    remaining_reject = reject_sids[100:]

    # Sample train: 200 accept + 200 reject from remaining
    train_sids_2025 = remaining_accept[:200] + remaining_reject[:200]

    test_2025 = {sid: paper_pool_2025[sid] for sid in test_sids_2025}
    train_2025 = {sid: paper_pool_2025[sid] for sid in train_sids_2025}

    for name, data in [("test_2025", test_2025), ("train_2025", train_2025)]:
        labels = [d["label"] for d in data.values()]
        print(f"  {name}: {len(data)} total ({labels.count('accept')} accept, {labels.count('reject')} reject)")

    # Add human reviews + img_pages to both
    if not meta_index and METADATA_PATH.exists():
        from datasets import load_from_disk
        meta_ds = load_from_disk(str(METADATA_PATH))
        for row in meta_ds:
            sid = row.get("submission_id")
            if sid:
                meta_index[sid] = row
        print(f"  Metadata index: {len(meta_index)} papers")

    if not img_lookup:
        img_lookup = load_img_pages_lookup(max_pages=10)
        print(f"  Image lookup: {len(img_lookup)} papers with image paths")

    for key, target_dir in ALIGNED_V3_2025_TARGETS.items():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {key} -> {target_dir.name} ---")

        is_with_model = "with_model" in key

        # TEST_SAMPLE.json
        if is_with_model:
            test_data = {sid: dict(p) for sid, p in test_2025.items()}
        else:
            test_data = {sid: {k: v for k, v in p.items()
                               if k not in ("model_prediction", "model_confidence")}
                         for sid, p in test_2025.items()}
        # Strip label from test data
        for sid in test_data:
            test_data[sid].pop("label", None)
        test_data = add_img_pages(test_data, img_lookup)
        save_json(test_data, target_dir / "TEST_SAMPLE.json")

        # TRAIN_SAMPLE.json: keep labels + model predictions + human reviews + img_pages
        train_data = {sid: dict(p) for sid, p in train_2025.items()}
        train_data = add_human_reviews(train_data, meta_index)
        train_data = add_img_pages(train_data, img_lookup)
        save_json(train_data, target_dir / "TRAIN_SAMPLE.json")

    # Write ground truth for 2025
    gt_2025 = {sid: p["label"] for sid, p in test_2025.items()}
    gt_path = REPO / "data" / "agent_ground_truth_2025_aligned_v3.json"
    save_json(gt_2025, gt_path)

    print("\nDone. Run AGENTS.override.md and main.py creation separately.")


if __name__ == "__main__":
    main()
