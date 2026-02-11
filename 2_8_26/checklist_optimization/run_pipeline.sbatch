#!/bin/bash
#SBATCH --job-name=checklist_opt
#SBATCH --output=2_8_26/logs/checklist_optimization/%x_%A_%a.out
#SBATCH --error=2_8_26/logs/checklist_optimization/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=neu316
#SBATCH --array=0-4

# Checklist Optimization Pipeline using Qwen/Qwen2.5-7B-Instruct
# GPU: 1x L40 (48GB) - model ~14GB at fp16
# Job array: 0=stage1, 1=stage2, 2=stage3, 3=stage4, 4=stage5+analyze

source ~/.bashrc
cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer

# Activate environment
source .vllm/bin/activate


BASE_DIR="2_8_26/checklist_optimization"

# Ensure output dirs exist
mkdir -p ${BASE_DIR}/data ${BASE_DIR}/results ${BASE_DIR}/metrics 2_8_26/logs/checklist_optimization

case $SLURM_ARRAY_TASK_ID in
    0)
        echo "Stage 1: Generating candidate questions from ICLR review criteria..."
        echo "  Model: Qwen/Qwen2.5-7B-Instruct (1x L40 GPU)"
        python ${BASE_DIR}/stage1_generate_candidates.py \
            --output ${BASE_DIR}/data/candidate_questions.jsonl \
            --n_questions 100 \
            --sample_size 50 \
            --tensor_parallel_size 1
        ;;
    1)
        echo "Stage 2: Filtering questions by enforceability..."
        python ${BASE_DIR}/stage2_filter_questions.py \
            --input_questions ${BASE_DIR}/data/candidate_questions.jsonl \
            --output ${BASE_DIR}/data/filtered_questions.jsonl \
            --consistency_threshold 0.80 \
            --n_test_reviews 5 \
            --n_repeats 3 \
            --tensor_parallel_size 1
        ;;
    2)
        echo "Stage 3: Deduplicating questions via semantic clustering..."
        python ${BASE_DIR}/stage3_deduplicate.py \
            --input ${BASE_DIR}/data/filtered_questions.jsonl \
            --output ${BASE_DIR}/data/deduplicated_questions.jsonl \
            --similarity_threshold 0.7 \
            --embeddings_output ${BASE_DIR}/data/question_embeddings.npy \
            --tensor_parallel_size 1
        ;;
    3)
        echo "Stage 4: Beam search optimization (most expensive stage)..."
        echo "  Answering all questions on all 4682 real ICLR reviews"
        python ${BASE_DIR}/stage4_beam_search.py \
            --input_questions ${BASE_DIR}/data/deduplicated_questions.jsonl \
            --output ${BASE_DIR}/data/optimal_checklist.json \
            --beam_width 10 \
            --max_questions 20 \
            --tensor_parallel_size 1 \
            --batch_size 64
        ;;
    4)
        echo "Stage 5: Evaluation + Analysis..."
        python ${BASE_DIR}/stage5_evaluate.py \
            --checklist ${BASE_DIR}/data/optimal_checklist.json \
            --evaluations ${BASE_DIR}/data/paper_evaluations.jsonl \
            --output_metrics ${BASE_DIR}/metrics/checklist_metrics.json \
            --output_predictions ${BASE_DIR}/results/final_predictions.jsonl

        echo "Generating plots..."
        python ${BASE_DIR}/analyze.py \
            --checklist ${BASE_DIR}/data/optimal_checklist.json \
            --metrics ${BASE_DIR}/metrics/checklist_metrics.json \
            --baseline_results 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl \
            --output_dir ${BASE_DIR}/metrics
        ;;
esac

echo "Stage ${SLURM_ARRAY_TASK_ID} complete at $(date)"
