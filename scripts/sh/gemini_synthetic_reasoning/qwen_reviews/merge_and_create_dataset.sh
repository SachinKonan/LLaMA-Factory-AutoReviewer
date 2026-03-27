#!/bin/bash
set -euo pipefail
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer
python scripts/create_extras_datasets.py --variant qwen_reviews
