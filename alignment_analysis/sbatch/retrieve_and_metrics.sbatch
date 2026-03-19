#!/bin/bash
#SBATCH --job-name=retrieve_metrics
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/retrieve_metrics_%j.out
#SBATCH --error=logs/retrieve_metrics_%j.err

cd /scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer
source .venv_vllm_inf/bin/activate

echo "=== Re-retrieving specific strict (unconditioned was empty) ==="
python alignment_analysis/grade_alignment.py --retrieve --ai_prompt_variant specific

echo ""
echo "=== Computing metrics: specific strict ==="
python alignment_analysis/grade_alignment.py --metrics_only --ai_prompt_variant specific

echo ""
echo "=== Computing metrics: specific lenient ==="
python alignment_analysis/grade_alignment.py --metrics_only --ai_prompt_variant specific --lenient

echo ""
echo "=== Full comparison (reads existing CSVs, no Gemini needed) ==="
python alignment_analysis/compute_metrics_comparison.py
