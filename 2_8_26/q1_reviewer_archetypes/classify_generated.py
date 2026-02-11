#!/usr/bin/env python3
"""
Q1: Classify generated reviews into discovered archetypes.

Takes the K-Means model from cluster_reviews.py and maps generated reviews
to the same archetype clusters. This enables distribution shift analysis.

Usage:
    python 2_8_26/q1_reviewer_archetypes/classify_generated.py \
        --predictions inference_scaling/results/clean/new/predictions.jsonl
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_generated_reviews(predictions_path):
    """Load generated reviews from predictions.jsonl."""
    reviews = []
    with open(predictions_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            preds = data.get("predict", [])
            if isinstance(preds, str):
                preds = [preds]

            for i, pred in enumerate(preds):
                if pred and len(pred) > 50:
                    reviews.append({
                        "text": pred,
                        "label": data.get("label", ""),
                        "generation_idx": i,
                        "source": "generated",
                    })

    return reviews


def embed_reviews(reviews, model_name="all-MiniLM-L6-v2", batch_size=64):
    """Embed review texts."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [r["text"][:2000] for r in reviews]

    print(f"Embedding {len(texts)} generated reviews...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings)


def main():
    parser = argparse.ArgumentParser(description="Q1: Classify generated reviews")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to generated predictions.jsonl")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    # Load K-Means model
    kmeans_path = os.path.join(RESULTS_DIR, "kmeans_model.pkl")
    if not os.path.exists(kmeans_path):
        print(f"K-Means model not found: {kmeans_path}")
        print("Run cluster_reviews.py first.")
        return

    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)
    print(f"Loaded K-Means model ({kmeans.n_clusters} clusters)")

    # Load generated reviews
    print(f"Loading generated reviews from {args.predictions}...")
    gen_reviews = load_generated_reviews(args.predictions)
    print(f"  Found {len(gen_reviews)} generated reviews")

    if not gen_reviews:
        print("No reviews found in predictions.")
        return

    # Embed
    gen_embeddings = embed_reviews(gen_reviews, args.model_name)

    # Classify
    gen_labels = kmeans.predict(gen_embeddings)

    # Stats
    unique, counts = np.unique(gen_labels, return_counts=True)
    print("\nGenerated review cluster distribution:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} reviews ({count / len(gen_labels):.1%})")

    # Save
    for i, review in enumerate(gen_reviews):
        review["cluster"] = int(gen_labels[i])

    output_path = os.path.join(RESULTS_DIR, "gen_reviews_classified.json")
    with open(output_path, "w") as f:
        json.dump(gen_reviews, f, indent=2, ensure_ascii=False)

    np.save(os.path.join(RESULTS_DIR, "gen_embeddings.npy"), gen_embeddings)

    print(f"\nSaved classified reviews to {output_path}")
    print("Next: run analyze.py for distribution shift analysis.")


if __name__ == "__main__":
    main()
