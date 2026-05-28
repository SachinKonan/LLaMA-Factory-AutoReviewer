#!/bin/bash
# Build the 4 (modality x paper_subset) HF release buckets as parquet shards
# in parallel via a Slurm array. Each task writes one parquet:
#   hf_release_local/<modality>/<subset>/data-shard-NNNN.parquet
#
# Task layout:
#   0          -> text/arxiv  shard 0/1   (text fits in one task, no sharding)
#   1          -> text/iclr   shard 0/1
#   2..65      -> vision/arxiv shard 0..63 / 64
#   66..129    -> vision/iclr  shard 0..63 / 64
#
# Total 130 tasks. With %16, at most 16 concurrent; finishes in ~30-60 min wall.
# After completion, hf_release_local/<modality>/<subset>/*.parquet is a valid
# HF dataset (load via `datasets.load_dataset("parquet", data_files="dir/*.parquet")`).
#
# Resubmit-safe: --skip_existing in the script means already-written shards
# get skipped.

#SBATCH --job-name=hf_release_array
#SBATCH --array=0-129%16
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=cpu
#SBATCH --output=logs/build_hf_release/array_%A_%a.out
#SBATCH --error=logs/build_hf_release/array_%A_%a.err

set -e

mkdir -p logs/build_hf_release

REPO=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/hf_release_local}   # override via --export=ALL,OUTPUT_DIR=...
VENV=$REPO/.venv
TID=$SLURM_ARRAY_TASK_ID

if   [[ $TID -eq 0 ]];                                  then MOD=text;   SUBSET=arxiv; SHARD=0;            NSHARDS=1
elif [[ $TID -eq 1 ]];                                  then MOD=text;   SUBSET=iclr;  SHARD=0;            NSHARDS=1
elif [[ $TID -ge 2   && $TID -le 65  ]];                then MOD=vision; SUBSET=arxiv; SHARD=$((TID-2));   NSHARDS=64
elif [[ $TID -ge 66  && $TID -le 129 ]];                then MOD=vision; SUBSET=iclr;  SHARD=$((TID-66));  NSHARDS=64
else
  echo "ERROR: SLURM_ARRAY_TASK_ID=$TID out of range" >&2
  exit 2
fi

echo "[array-task $TID] host=$(hostname) mod=$MOD subset=$SUBSET shard=$SHARD/$NSHARDS"

cd "$REPO"
exec "$VENV/bin/python" scripts/build_hf_release_datasets.py \
    --output_dir   "$OUTPUT_DIR" \
    --target_modality "$MOD" \
    --target_subset   "$SUBSET" \
    --shard_idx       "$SHARD" \
    --num_shards      "$NSHARDS" \
    --skip_existing
