#!/bin/bash
#SBATCH --job-name=q1_cluster
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=2_8_26/logs/q1_reviewer_archetypes/cluster_%j.out
#SBATCH --error=2_8_26/logs/q1_reviewer_archetypes/cluster_%j.err
#SBATCH --comment="Q1: Cluster reviews with Qwen3-Embedding-8B"

# ============================================================================
# Q1: Reviewer Archetype Clustering
# ============================================================================
# Clusters ground truth reviews using Qwen3-Embedding-8B semantic embeddings.
# Generates 5 archetypes with K-Means clustering.
#
# Usage:
#   sbatch 2_8_26/q1_reviewer_archetypes/run_clustering.sbatch
# ============================================================================

set -e

PROJECT_DIR="/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer"
cd "${PROJECT_DIR}"

mkdir -p 2_8_26/logs/q1_reviewer_archetypes

echo "=============================================="
echo "Q1: Clustering Ground Truth Reviews"
echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=============================================="

# Run clustering with smaller batch size to avoid OOM
python 2_8_26/q1_reviewer_archetypes/cluster_reviews.py \
    --model_name Qwen/Qwen3-Embedding-8B \
    --batch_size 4 \
    --n_clusters 5

echo "=============================================="
echo "Done! Clustering results saved."
echo "=============================================="
