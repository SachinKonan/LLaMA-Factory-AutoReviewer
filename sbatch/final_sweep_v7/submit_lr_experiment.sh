#!/bin/bash
# Submit learning rate experiments for both text and vision models
# This will launch 6 total jobs (3 LR configs × 2 model types)

cd /scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer

echo "=========================================="
echo "Submitting Learning Rate Experiments"
echo "=========================================="
echo ""

# Submit text experiments (array 0-2)
echo "Submitting TEXT experiments..."
TEXT_JOB=$(sbatch sbatch/final_sweep_v7/lr_experiment_text_trainagreeing.sbatch | awk '{print $4}')
echo "  Job ID: ${TEXT_JOB}"
echo "  Configurations:"
echo "    - Array 0: LR=2e-5 (uniform)"
echo "    - Array 1: LR=2e-6 (uniform)"
echo "    - Array 2: LR=2e-5 head, 2e-6 backbone (separate)"
echo ""

# Submit vision experiments (array 0-2)
echo "Submitting VISION experiments..."
VISION_JOB=$(sbatch sbatch/final_sweep_v7/lr_experiment_vision_trainagreeing.sbatch | awk '{print $4}')
echo "  Job ID: ${VISION_JOB}"
echo "  Configurations:"
echo "    - Array 0: LR=2e-5 (uniform)"
echo "    - Array 1: LR=2e-6 (uniform)"
echo "    - Array 2: LR=2e-5 head, 2e-6 backbone (separate)"
echo ""

echo "=========================================="
echo "Submission Complete"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f logs/lr_experiment_v7/${TEXT_JOB}_*.out"
echo "  tail -f logs/lr_experiment_v7/${VISION_JOB}_*.out"
echo ""
echo "Results will be saved to:"
echo "  results/lr_experiment_v7/text_trainagreeing_lr_*/"
echo "  results/lr_experiment_v7/vision_trainagreeing_lr_*/"
echo ""
