#!/bin/bash
#SBATCH --job-name=sync_gcs
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --output=logs/sync_from_gcs/%j.out
#SBATCH --error=logs/sync_from_gcs/%j.err

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer

mkdir -p logs/sync_from_gcs

echo "Starting GCS sync at $(date)"
/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/sync_from_gcs.sh
echo "Completed GCS sync at $(date)"
