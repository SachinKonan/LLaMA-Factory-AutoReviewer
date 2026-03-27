#!/bin/bash
# Submit inference jobs for balanced NeurIPS/ICML eval set
# Usage: bash sbatch/inference/run_balanced_nips_icml_eval.sh

set -e
cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer

RESULTS_BASE="results/balanced_nips_icml_eval"

echo "=============================================="
echo "Submitting balanced NeurIPS/ICML eval jobs"
echo "=============================================="

# --- Model 1: SFT Text best (bz32 ep2) ---
echo ""
echo "=== Model 1: SFT Text best (bz32 ep2) ==="
sbatch --export=ALL,\
CHECKPOINT_DIR=saves/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr2e-6_text/checkpoint-1322,\
DATASET=balanced_nips_icml_eval,\
TEMPLATE=qwen,\
CKPT_STEP=1322,\
RESULTS_DIR=${RESULTS_BASE}/sft_text_bz32_ep2,\
SAVE_LOGPROBS=1 \
  sbatch/inference/text_inference.sbatch

# --- Model 2: PT Instruct lr4e-6 ep4 ---
echo ""
echo "=== Model 2: PT Instruct lr4e-6 ep4 ==="
sbatch --export=ALL,\
CHECKPOINT_DIR=saves/pretrain_then_sft/instruct_lr4e-6/sft/checkpoint-2644,\
DATASET=balanced_nips_icml_eval,\
TEMPLATE=qwen,\
CKPT_STEP=2644,\
RESULTS_DIR=${RESULTS_BASE}/pt_instruct_lr4e-6_ep4,\
SAVE_LOGPROBS=1 \
  sbatch/inference/text_inference.sbatch

# --- Model 3: PT Base lr4e-6 ep1 ---
echo ""
echo "=== Model 3: PT Base lr4e-6 ep1 ==="
sbatch --export=ALL,\
CHECKPOINT_DIR=saves/pretrain_then_sft/base_lr4e-6/sft/checkpoint-661,\
DATASET=balanced_nips_icml_eval,\
TEMPLATE=qwen,\
CKPT_STEP=661,\
RESULTS_DIR=${RESULTS_BASE}/pt_base_lr4e-6_ep1,\
SAVE_LOGPROBS=1 \
  sbatch/inference/text_inference.sbatch

# --- Model 4: SFT Vision best (bz16 ep2) ---
echo ""
echo "=== Model 4: SFT Vision best (bz16 ep2) ==="
sbatch --export=ALL,\
CHECKPOINT_DIR=saves/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/checkpoint-2648,\
DATASET=balanced_nips_icml_eval_vision,\
TEMPLATE=qwen2_vl,\
CKPT_STEP=2648,\
RESULTS_DIR=${RESULTS_BASE}/sft_vision_bz16_ep2,\
SAVE_LOGPROBS=1 \
  sbatch/inference/vision_inference.sbatch

echo ""
echo "=============================================="
echo "All 4 jobs submitted. Check with: squeue -u \$USER"
echo "=============================================="
