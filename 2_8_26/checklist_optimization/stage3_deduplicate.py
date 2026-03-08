#!/usr/bin/env python3
"""
Stage 3: Semantic Deduplication via Clustering.

Clusters questions by semantic similarity and selects the best representative
from each cluster using local Qwen3-30B-A3B via vLLM.

Usage:
    # Full deduplication (needs 2x L40 GPUs for LLM selection)
    python 2_8_26/checklist_optimization/stage3_deduplicate.py \
        --input 2_8_26/checklist_optimization/data/filtered_questions.jsonl \
        --output 2_8_26/checklist_optimization/data/deduplicated_questions.jsonl \
        --similarity_threshold 0.7 \
        --embeddings_output 2_8_26/checklist_optimization/data/question_embeddings.npy

    # Debug mode (skip LLM selection)
    python 2_8_26/checklist_optimization/stage3_deduplicate.py \
        --input 2_8_26/checklist_optimization/data/filtered_questions.jsonl \
        --output 2_8_26/checklist_optimization/data/deduplicated_questions.jsonl \
        --similarity_threshold 0.7 --skip_llm_selection --debug
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_vllm_model,
    load_jsonl,
    query_vllm,
    save_json,
    save_jsonl,
)


# ============================================================================
# Embedding
# ============================================================================

def embed_questions(questions: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Embed question texts using sentence-transformers."""
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [q.get("text", "") for q in questions]
    print(f"Embedding {len(texts)} questions...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    return np.array(embeddings)


# ============================================================================
# Clustering
# ============================================================================

def cluster_questions(embeddings: np.ndarray, similarity_threshold: float = 0.7) -> tuple[np.ndarray, dict]:
    """Cluster questions by semantic similarity."""
    print(f"\nClustering with similarity threshold: {similarity_threshold}")

    n = len(embeddings)
    print(f"  Computing {n}x{n} distance matrix...")
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = cosine(embeddings[i], embeddings[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    distance_threshold = 1.0 - similarity_threshold

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance_matrix)

    n_clusters = len(set(labels))
    cluster_sizes = {}
    for label in labels:
        cluster_sizes[int(label)] = cluster_sizes.get(int(label), 0) + 1

    print(f"  Found {n_clusters} clusters")
    print(f"  Cluster sizes: min={min(cluster_sizes.values())}, max={max(cluster_sizes.values())}, "
          f"mean={np.mean(list(cluster_sizes.values())):.1f}")

    cluster_info = {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "similarity_threshold": similarity_threshold,
        "distance_threshold": distance_threshold,
    }

    return labels, cluster_info


# ============================================================================
# Representative Selection
# ============================================================================

def select_representative_llm(cluster_questions: list[dict], llm=None) -> int:
    """Select the best representative question from a cluster using local LLM."""
    numbered_questions = []
    for i, q in enumerate(cluster_questions, 1):
        numbered_questions.append(f"{i}. {q.get('text', '')}")

    prompt = f"""Here are {len(cluster_questions)} similar binary evaluation questions for reviewing academic papers:

{chr(10).join(numbered_questions)}

Select the question number (1-{len(cluster_questions)}) that is:
1. Most clear and unambiguous
2. Most actionable (easier to answer from review text)
3. Most likely to correlate with accept/reject decisions

Output only the number."""

    response = query_vllm(prompt, llm, temperature=0.3, max_tokens=10)

    try:
        selected_num = int(response.strip().split()[0])
        if 1 <= selected_num <= len(cluster_questions):
            return selected_num - 1
    except (ValueError, IndexError):
        pass

    return 0


def select_representatives(
    questions: list[dict],
    labels: np.ndarray,
    n_clusters: int,
    skip_llm: bool = False,
    llm=None,
) -> list[dict]:
    """Select representative questions from each cluster."""
    representatives = []

    for cluster_id in range(n_clusters):
        cluster_indices = [i for i in range(len(questions)) if labels[i] == cluster_id]
        cluster_qs = [questions[i] for i in cluster_indices]

        print(f"\nCluster {cluster_id} ({len(cluster_qs)} questions):")

        if skip_llm or len(cluster_qs) == 1:
            selected_idx = 0
            if skip_llm and len(cluster_qs) > 1:
                print(f"  Taking first question (skip_llm=True)")
        else:
            print(f"  Using LLM to select best representative...")
            selected_idx = select_representative_llm(cluster_qs, llm)
            print(f"  Selected question {selected_idx + 1}/{len(cluster_qs)}")

        selected_question = cluster_qs[selected_idx].copy()
        selected_question["cluster_id"] = cluster_id
        selected_question["cluster_size"] = len(cluster_qs)

        representatives.append(selected_question)
        print(f"  Representative: {selected_question.get('text', '')}")

    return representatives


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Deduplicate questions via clustering")
    parser.add_argument("--input", type=str, required=True, help="Path to filtered_questions.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output path for deduplicated_questions.jsonl")
    parser.add_argument("--similarity_threshold", type=float, default=0.7)
    parser.add_argument("--embeddings_output", type=str, default=None, help="Save embeddings (.npy)")
    parser.add_argument("--skip_llm_selection", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        print("DEBUG MODE")

    # Load questions
    print(f"\nLoading filtered questions from: {args.input}")
    questions = load_jsonl(args.input)
    print(f"  Loaded {len(questions)} questions")

    # Embed questions
    embeddings = embed_questions(questions)
    print(f"  Embedding shape: {embeddings.shape}")

    if args.embeddings_output:
        print(f"\nSaving embeddings to: {args.embeddings_output}")
        np.save(args.embeddings_output, embeddings)

    # Cluster questions
    labels, cluster_info = cluster_questions(embeddings, args.similarity_threshold)

    # Select representatives
    print(f"\nSelecting representatives (skip_llm={args.skip_llm_selection})...")
    llm = None
    if not args.skip_llm_selection:
        print("Loading vLLM model for representative selection...")
        llm = get_vllm_model(tensor_parallel_size=args.tensor_parallel_size)

    representatives = select_representatives(
        questions, labels, cluster_info["n_clusters"],
        skip_llm=args.skip_llm_selection, llm=llm,
    )

    # Save
    print(f"\nDeduplication complete: {len(questions)} → {len(representatives)} questions")
    save_jsonl(representatives, args.output)
    print(f"Saved to: {args.output}")

    clusters_path = args.output.replace("deduplicated_questions.jsonl", "").rstrip("/")
    clusters_path = clusters_path.replace("/data", "/results") + "/clusters.json"
    save_json(cluster_info, clusters_path)
    print(f"Saved cluster info to: {clusters_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
