#!/bin/bash
set -euo pipefail
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
uv run python scripts/create_extras_datasets.py --variant gemini_reviews
