#!/usr/bin/env python3
"""
Q1: Cluster ground truth reviews into archetypes.

Loads individual reviewer reviews from the HuggingFace dataset (Arrow format)
which contains `original_reviews` as a JSON string of review dicts per paper.
Filters to test-set papers using submission IDs from the LLaMA Factory test split.

Embeds reviews with sentence-transformers, clusters with K-Means, and produces
cluster labels + centroids for downstream use.

Usage:
    python 2_8_26/q1_reviewer_archetypes/cluster_reviews.py
    python 2_8_26/q1_reviewer_archetypes/cluster_reviews.py --n_clusters 5
"""

import argparse
import json
import os
import pickle
import re
import sys

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# HuggingFace dataset (Arrow format) with original_reviews column
DEFAULT_HF_DATASET = (
    "/n/fs/vision-mix/sk7524/NipsIclrData/AutoReviewer/data/"
    "hf_dataset_new8_noref_cropped_2017_2026_with_decisions"
)

# LLaMA Factory test split (used to filter to test-set papers)
DEFAULT_TEST_SPLIT = (
    "/n/fs/vision-mix/sk7524/LLaMA-Factory/data/"
    "iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test/data.json"
)


def get_test_submission_ids(test_split_path):
    """Load submission IDs from the LLaMA Factory test split."""
    with open(test_split_path, "r") as f:
        data = json.load(f)

    ids = set()
    for sample in data:
        sid = sample.get("_metadata", {}).get("submission_id", "")
        if sid:
            ids.add(sid)

    print(f"Loaded {len(ids)} test submission IDs from {test_split_path}")
    return ids


def load_ground_truth_reviews(hf_dataset_path, test_ids=None):
    """Load ground truth reviews from the HF dataset.

    Each row has `original_reviews` (JSON string of list of review dicts)
    with keys like: summary, strengths, weaknesses, rating, confidence, etc.

    Args:
        hf_dataset_path: Path to HF dataset on disk (Arrow format).
        test_ids: Optional set of submission_ids to filter to.

    Returns:
        List of review dicts with text, fields, and metadata.
    """
    try:
        from datasets import load_from_disk
    except ImportError:
        print("Error: datasets library not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)

    print(f"Loading HF dataset from {hf_dataset_path}...")
    ds = load_from_disk(hf_dataset_path)
    print(f"  Total papers in dataset: {len(ds)}")

    all_reviews = []
    skipped_no_reviews = 0

    for row in ds:
        submission_id = row["submission_id"]

        # Filter to test set if specified
        if test_ids is not None and submission_id not in test_ids:
            continue

        # Parse original_reviews JSON
        reviews_json = row.get("original_reviews", "")
        if not reviews_json or reviews_json == "null":
            skipped_no_reviews += 1
            continue

        try:
            reviews = json.loads(reviews_json)
        except (json.JSONDecodeError, TypeError):
            skipped_no_reviews += 1
            continue

        if not isinstance(reviews, list):
            skipped_no_reviews += 1
            continue

        # Get decision info
        tech_json = row.get("technical_indicators", "")
        try:
            tech = json.loads(tech_json) if tech_json else {}
        except (json.JSONDecodeError, TypeError):
            tech = {}
        decision = tech.get("binary_decision", "")

        for idx, review in enumerate(reviews):
            if not isinstance(review, dict):
                continue

            # Build text representation from review fields
            text_parts = []
            for field in ["summary", "strengths", "weaknesses", "questions"]:
                val = review.get(field, "")
                if val:
                    text_parts.append(f"{field}: {val}")

            text = "\n".join(text_parts)
            if len(text) < 50:
                continue

            # Extract rating (handle both int and string formats like "6: marginally above")
            rating_raw = review.get("rating", "")
            rating_match = re.search(r'(\d+)', str(rating_raw))
            rating = int(rating_match.group(1)) if rating_match else None

            all_reviews.append({
                "text": text,
                "review_idx": idx,
                "submission_id": submission_id,
                "year": row.get("year", 0),
                "decision": decision,
                "rating": rating,
                "confidence": review.get("confidence", ""),
                "source": "ground_truth",
            })

    if skipped_no_reviews:
        print(f"  Skipped {skipped_no_reviews} papers with no reviews")

    return all_reviews


def embed_reviews(reviews, model_name="Qwen/Qwen3-Embedding-8B", batch_size=8, use_cpu=False):
    """Embed review texts using sentence-transformers or transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    print(f"Loading embedding model: {model_name}...")
    device = "cpu" if use_cpu else "cuda"
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    texts = [r["text"][:8192] for r in reviews]  # Qwen3-Embedding supports longer context

    print(f"Embedding {len(texts)} reviews with {model_name} on {device}...")
    print(f"  Batch size: {batch_size}")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                             normalize_embeddings=True, device=device)

    return np.array(embeddings)


def cluster_reviews(embeddings, n_clusters=5):
    """Cluster embeddings using K-Means."""
    from sklearn.cluster import KMeans

    print(f"Clustering {len(embeddings)} reviews into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Print cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} reviews ({count / len(labels):.1%})")

    return labels, kmeans


def label_clusters(reviews, labels, n_clusters):
    """Generate descriptive labels for clusters using keyword analysis and rating stats."""
    from collections import Counter

    cluster_labels = {}
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i in range(len(reviews)) if labels[i] == cluster_id]
        cluster_texts = [reviews[i]["text"] for i in cluster_indices]

        # Keyword frequency
        all_words = " ".join(cluster_texts).lower().split()
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "that",
            "this", "these", "those", "it", "its", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "and", "or", "not",
            "but", "if", "as", "no", "paper", "work", "authors", "also",
            "more", "some", "however", "proposed", "model", "method",
            "results", "based", "using", "used", "provide", "provided",
            "summary:", "strengths:", "weaknesses:", "questions:",
        }
        word_counts = Counter(w for w in all_words if w not in stopwords and len(w) > 3)
        top_words = [w for w, _ in word_counts.most_common(8)]

        # Rating statistics
        ratings = [r["rating"] for r in (reviews[i] for i in cluster_indices) if r["rating"] is not None]

        # Strengths/weaknesses text length ratio
        str_lengths = []
        weak_lengths = []
        for i in cluster_indices:
            text = reviews[i]["text"]
            str_match = re.search(r'strengths:\s*(.*?)(?=weaknesses:|questions:|$)', text, re.DOTALL)
            weak_match = re.search(r'weaknesses:\s*(.*?)(?=questions:|$)', text, re.DOTALL)
            if str_match:
                str_lengths.append(len(str_match.group(1).strip()))
            if weak_match:
                weak_lengths.append(len(weak_match.group(1).strip()))

        info = {
            "top_keywords": top_words,
            "size": len(cluster_texts),
            "avg_length": float(np.mean([len(t) for t in cluster_texts])),
        }

        if ratings:
            info["avg_rating"] = float(np.mean(ratings))
            info["std_rating"] = float(np.std(ratings))
        if str_lengths and weak_lengths:
            info["avg_strengths_len"] = float(np.mean(str_lengths))
            info["avg_weaknesses_len"] = float(np.mean(weak_lengths))
            info["weakness_to_strength_ratio"] = float(np.mean(weak_lengths)) / max(float(np.mean(str_lengths)), 1)

        cluster_labels[cluster_id] = info

    return cluster_labels


def main():
    parser = argparse.ArgumentParser(description="Q1: Cluster ground truth reviews")
    parser.add_argument("--hf_dataset", type=str, default=DEFAULT_HF_DATASET,
                        help="Path to HF dataset (Arrow) with original_reviews column")
    parser.add_argument("--test_split", type=str, default=DEFAULT_TEST_SPLIT,
                        help="Path to LLaMA Factory test split JSON (for filtering)")
    parser.add_argument("--no_filter", action="store_true",
                        help="Use all papers (don't filter to test set)")
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU for embeddings")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embeddings")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load test submission IDs for filtering
    test_ids = None
    if not args.no_filter:
        test_ids = get_test_submission_ids(args.test_split)

    # Load ground truth reviews from HF dataset
    gt_reviews = load_ground_truth_reviews(args.hf_dataset, test_ids)
    print(f"Parsed {len(gt_reviews)} individual reviews")

    if not gt_reviews:
        print("No reviews found. Check dataset path.")
        return

    # Print sample
    print(f"\n  Sample review (first 200 chars):")
    print(f"    {gt_reviews[0]['text'][:200]}...")
    print(f"    rating={gt_reviews[0]['rating']}, decision={gt_reviews[0]['decision']}")

    # Embed reviews
    embeddings = embed_reviews(gt_reviews, args.model_name, batch_size=args.batch_size, use_cpu=args.cpu)

    # Cluster
    labels, kmeans = cluster_reviews(embeddings, args.n_clusters)

    # Label clusters
    cluster_info = label_clusters(gt_reviews, labels, args.n_clusters)
    print("\nCluster descriptions:")
    for cid, info in sorted(cluster_info.items()):
        rating_str = f", avg_rating={info['avg_rating']:.1f}" if "avg_rating" in info else ""
        ratio_str = f", weak/str={info['weakness_to_strength_ratio']:.1f}" if "weakness_to_strength_ratio" in info else ""
        print(f"  Cluster {cid}: {info['top_keywords'][:5]} "
              f"(n={info['size']}, avg_len={info['avg_length']:.0f}{rating_str}{ratio_str})")

    # Save results (exclude full text to keep file size manageable)
    for i, review in enumerate(gt_reviews):
        review["cluster"] = int(labels[i])

    reviews_for_save = []
    for r in gt_reviews:
        save_r = {k: v for k, v in r.items() if k != "text"}
        reviews_for_save.append(save_r)

    with open(os.path.join(RESULTS_DIR, "gt_reviews_clustered.json"), "w") as f:
        json.dump(reviews_for_save, f, indent=2, ensure_ascii=False)

    with open(os.path.join(RESULTS_DIR, "cluster_info.json"), "w") as f:
        json.dump(cluster_info, f, indent=2)

    np.save(os.path.join(RESULTS_DIR, "gt_embeddings.npy"), embeddings)

    with open(os.path.join(RESULTS_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)

    print(f"\nSaved clustering results to {RESULTS_DIR}/")
    print(f"  gt_reviews_clustered.json  ({len(gt_reviews)} reviews)")
    print(f"  cluster_info.json          ({args.n_clusters} clusters)")
    print(f"  gt_embeddings.npy          ({embeddings.shape})")
    print(f"  kmeans_model.pkl")
    print("\nNext: run analyze.py to visualize the clusters.")


if __name__ == "__main__":
    main()
