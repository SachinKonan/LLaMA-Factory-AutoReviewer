#!/usr/bin/env python3
"""
Generate SFT and DPO Training Datasets from Checklist-Evaluated Reviews.

SFT: Top 20% of reviews by checkmark score
DPO: Contrastive pairs (high vs low quality reviews) from accepted papers

Usage:
    python 2_8_26/training_checklist/scripts/generate_training_data.py \
        --review_evaluations 2_8_26/training_checklist/data/review_evaluations.jsonl \
        --output_dir 2_8_26/training_checklist/data
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# System and user prompts
SYSTEM_PROMPT = "You are an expert academic reviewer tasked with evaluating research papers."

USER_PROMPT_TEMPLATE = """I am giving you a paper. I want to predict its acceptance outcome at ICLR.
 - Your answer will either be: \\boxed{{Accept}} or \\boxed{{Reject}}
 - Note: ICLR generally has a ~30% acceptance rate

{paper_content}"""


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def save_json(data: List[Dict], file_path: str):
    """Save data to JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} samples to {file_path}")


def load_paper_data(submission_id: str, year: int) -> str:
    """
    Load paper content from existing datasets.

    For now, returns a placeholder. In full implementation, this would:
    1. Load from /n/fs/vision-mix/sk7524/LLaMA-Factory/data/{dataset}/data.json
    2. Find the entry matching submission_id
    3. Extract the paper content from conversations[1]["value"]
    """
    # This will be populated from the actual dataset in real usage
    return f"[Paper content for {submission_id} would be loaded here]"


def create_sft_sample(review_eval: Dict, paper_content: str) -> Dict:
    """
    Create SFT training sample from review evaluation.

    Format:
        conversations: [
            {from: "system", value: "..."},
            {from: "human", value: "...paper..."},
            {from: "gpt", value: "Outcome: \\boxed{Accept/Reject}"}
        ]
        _metadata: {...}
    """
    decision = review_eval["decision"]
    if not decision:
        decision = "Accept"  # Default if missing

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": USER_PROMPT_TEMPLATE.format(paper_content=paper_content)},
            {"from": "gpt", "value": f"Outcome: \\boxed{{{decision}}}"},
        ],
        "_metadata": {
            "submission_id": review_eval["submission_id"],
            "review_idx": review_eval["review_idx"],
            "answer": decision,
            "year": review_eval["year"],
            "rating": review_eval["rating"],
            "checkmark_score": review_eval["checkmark_score"],
            "checkmark_count": review_eval["checkmark_count"],
            "confidence": review_eval.get("confidence", ""),
            "soundness": review_eval.get("soundness", ""),
            "presentation": review_eval.get("presentation", ""),
            "contribution": review_eval.get("contribution", ""),
        }
    }


def create_dpo_sample(chosen_review: Dict, rejected_review: Dict, paper_content: str) -> Dict:
    """
    Create DPO training sample from two contrastive reviews.

    Format:
        conversations: [
            {from: "system", value: "..."},
            {from: "human", value: "...paper..."}
        ]
        chosen: {from: "gpt", value: "Outcome: \\boxed{Accept}"}
        rejected: {from: "gpt", value: "Outcome: \\boxed{Accept}"}  # Same decision, different quality
        _metadata: {...}
    """
    decision = chosen_review["decision"]

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": USER_PROMPT_TEMPLATE.format(paper_content=paper_content)},
        ],
        "chosen": {"from": "gpt", "value": f"Outcome: \\boxed{{{decision}}}"},
        "rejected": {"from": "gpt", "value": f"Outcome: \\boxed{{{decision}}}"},
        "_metadata": {
            "submission_id": chosen_review["submission_id"],
            "decision": decision,
            "year": chosen_review["year"],
            "chosen_review_idx": chosen_review["review_idx"],
            "rejected_review_idx": rejected_review["review_idx"],
            "chosen_rating": chosen_review["rating"],
            "rejected_rating": rejected_review["rating"],
            "rating_diff": chosen_review["rating"] - rejected_review["rating"],
            "chosen_checkmark_count": chosen_review["checkmark_count"],
            "rejected_checkmark_count": rejected_review["checkmark_count"],
            "checkmark_diff": chosen_review["checkmark_count"] - rejected_review["checkmark_count"],
        }
    }


def filter_top_percentile(reviews: List[Dict], percentile: float = 0.8) -> List[Dict]:
    """
    Filter reviews to top (1-percentile)% by checkmark score.

    Args:
        reviews: List of review evaluations
        percentile: Percentile threshold (0.8 = top 20%)

    Returns:
        Filtered list of top reviews
    """
    scores = [r["checkmark_score"] for r in reviews]
    threshold = sorted(scores)[int(len(scores) * percentile)]

    filtered = [r for r in reviews if r["checkmark_score"] >= threshold]
    print(f"  Threshold (80th percentile): {threshold:.3f}")
    print(f"  Filtered: {len(filtered)}/{len(reviews)} reviews ({len(filtered)/len(reviews)*100:.1f}%)")

    return filtered


def generate_dpo_pairs(reviews: List[Dict], max_pairs_per_paper: int = 2) -> List[Tuple[Dict, Dict]]:
    """
    Generate DPO preference pairs from reviews.

    Criteria:
    - Only accepted papers
    - Chosen: rating > 5 AND high checkmark count
    - Rejected: rating <= 5 AND low checkmark count
    - Minimum contrast: rating_diff >= 2, checkmark_diff >= 2

    Args:
        reviews: List of review evaluations
        max_pairs_per_paper: Maximum number of pairs per paper

    Returns:
        List of (chosen_review, rejected_review) tuples
    """
    # Group reviews by paper
    papers = defaultdict(list)
    for review in reviews:
        papers[review["submission_id"]].append(review)

    pairs = []
    stats = {"total_papers": len(papers), "accepted_papers": 0, "papers_with_pairs": 0, "total_pairs": 0}

    for submission_id, paper_reviews in papers.items():
        # Check if paper is accepted
        decisions = [r["decision"] for r in paper_reviews if r["decision"]]
        if not decisions or decisions[0] != "Accept":
            continue

        stats["accepted_papers"] += 1

        # Separate high and low quality reviews
        high_quality = [
            r for r in paper_reviews
            if r["rating"] is not None and r["rating"] > 5
        ]
        low_quality = [
            r for r in paper_reviews
            if r["rating"] is not None and r["rating"] <= 5
        ]

        if not high_quality or not low_quality:
            continue

        # Create pairs with sufficient contrast
        paper_pairs = []
        for high_r in high_quality:
            for low_r in low_quality:
                rating_diff = high_r["rating"] - low_r["rating"]
                checkmark_diff = high_r["checkmark_count"] - low_r["checkmark_count"]

                # Require minimum contrast
                if rating_diff >= 2 and checkmark_diff >= 2:
                    paper_pairs.append((high_r, low_r, rating_diff + checkmark_diff))

        # Sort by total contrast and take top N
        paper_pairs.sort(key=lambda x: x[2], reverse=True)
        for high_r, low_r, _ in paper_pairs[:max_pairs_per_paper]:
            pairs.append((high_r, low_r))

        if paper_pairs:
            stats["papers_with_pairs"] += 1
            stats["total_pairs"] += min(len(paper_pairs), max_pairs_per_paper)

    print(f"\nDPO Pair Generation Statistics:")
    print(f"  Total papers: {stats['total_papers']}")
    print(f"  Accepted papers: {stats['accepted_papers']}")
    print(f"  Papers with valid pairs: {stats['papers_with_pairs']}")
    print(f"  Total pairs generated: {stats['total_pairs']}")

    return pairs


def transform_to_modality(sample: Dict, modality: str) -> Dict:
    """
    Transform sample to specific modality.

    Modalities:
    - clean: Text-only
    - clean_images: Text + images
    - vision: Vision-focused

    Note: For now, all modalities use the same format.
    Image handling would be added here in full implementation.
    """
    # For now, all modalities are the same (text-only)
    # In full implementation:
    # - clean_images: Add "images" field with image paths
    # - vision: Modify prompt to focus on figures

    return sample


def main():
    parser = argparse.ArgumentParser(description="Generate training datasets from checklist evaluations")
    parser.add_argument(
        "--review_evaluations",
        type=str,
        required=True,
        help="Path to review_evaluations.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--top_percentile",
        type=float,
        default=0.8,
        help="Percentile threshold for SFT (0.8 = top 20%)",
    )
    parser.add_argument(
        "--max_dpo_pairs",
        type=int,
        default=2,
        help="Maximum DPO pairs per paper",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Training split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 80)
    print("GENERATE TRAINING DATASETS")
    print("=" * 80)

    # Load review evaluations
    print(f"\nLoading review evaluations from {args.review_evaluations}...")
    reviews = load_jsonl(args.review_evaluations)
    print(f"  Loaded {len(reviews)} review evaluations")

    # Generate SFT dataset (top 20%)
    print(f"\n{'='*80}")
    print("GENERATING SFT DATASET (Top {:.0f}%)".format((1 - args.top_percentile) * 100))
    print("=" * 80)

    sft_reviews = filter_top_percentile(reviews, percentile=args.top_percentile)

    # Create SFT samples
    # Note: In full implementation, we would load actual paper content here
    sft_samples = []
    for review in sft_reviews:
        paper_content = load_paper_data(review["submission_id"], review["year"])
        sample = create_sft_sample(review, paper_content)
        sft_samples.append(sample)

    # Train/val split
    random.shuffle(sft_samples)
    split_idx = int(len(sft_samples) * args.train_split)
    sft_train = sft_samples[:split_idx]
    sft_val = sft_samples[split_idx:]

    print(f"  SFT train: {len(sft_train)} samples")
    print(f"  SFT val: {len(sft_val)} samples")

    # Generate DPO dataset
    print(f"\n{'='*80}")
    print(f"GENERATING DPO DATASET (Max {args.max_dpo_pairs} pairs/paper)")
    print("=" * 80)

    dpo_pairs = generate_dpo_pairs(reviews, max_pairs_per_paper=args.max_dpo_pairs)

    # Create DPO samples
    dpo_samples = []
    for chosen, rejected in dpo_pairs:
        paper_content = load_paper_data(chosen["submission_id"], chosen["year"])
        sample = create_dpo_sample(chosen, rejected, paper_content)
        dpo_samples.append(sample)

    # Train/val split
    random.shuffle(dpo_samples)
    split_idx = int(len(dpo_samples) * args.train_split)
    dpo_train = dpo_samples[:split_idx]
    dpo_val = dpo_samples[split_idx:]

    print(f"  DPO train: {len(dpo_train)} samples")
    print(f"  DPO val: {len(dpo_val)} samples")

    # Save datasets for all modalities
    print(f"\n{'='*80}")
    print("SAVING DATASETS (All Modalities)")
    print("=" * 80)

    modalities = ["clean", "clean_images", "vision"]
    for modality in modalities:
        print(f"\nModality: {modality}")

        # Transform and save SFT
        sft_train_modal = [transform_to_modality(s, modality) for s in sft_train]
        sft_val_modal = [transform_to_modality(s, modality) for s in sft_val]

        save_json(sft_train_modal, f"{args.output_dir}/sft_{modality}_train/data.json")
        save_json(sft_val_modal, f"{args.output_dir}/sft_{modality}_val/data.json")

        # Transform and save DPO
        dpo_train_modal = [transform_to_modality(s, modality) for s in dpo_train]
        dpo_val_modal = [transform_to_modality(s, modality) for s in dpo_val]

        save_json(dpo_train_modal, f"{args.output_dir}/dpo_{modality}_train/data.json")
        save_json(dpo_val_modal, f"{args.output_dir}/dpo_{modality}_val/data.json")

    # Save statistics
    stats = {
        "total_reviews": len(reviews),
        "sft": {
            "threshold_percentile": args.top_percentile,
            "total": len(sft_samples),
            "train": len(sft_train),
            "val": len(sft_val),
        },
        "dpo": {
            "max_pairs_per_paper": args.max_dpo_pairs,
            "total": len(dpo_samples),
            "train": len(dpo_train),
            "val": len(dpo_val),
        },
        "modalities": modalities,
    }

    stats_path = f"{args.output_dir}/dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to {stats_path}")

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Validate datasets: python 2_8_26/training_checklist/scripts/validate_datasets.py")
    print(f"2. Register datasets in data/dataset_info.json")
    print(f"3. Submit training jobs: sbatch 2_8_26/training_checklist/sbatch/train_pipeline.sbatch")


if __name__ == "__main__":
    main()
