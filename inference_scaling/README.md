# Inference Scaling Experiments

This folder contains scripts and configurations for running inference scaling experiments on paper review prediction.

## Overview

The experiments test different prompt strategies and ensembling methods:

### Prompt Variants
1. **Original**: Baseline prompt with simple accept/reject boxed output
2. **New**: Detailed reviewer prompt with JSON structured output
3. **New + Fewshot**: New prompt with 1 accept + 1 reject example (~60k tokens)

### Modalities
- **clean**: Text-only (paper markdown) → Qwen2.5-7B-Instruct
- **clean_images**: Text + embedded images → Qwen2.5-VL-7B-Instruct
- **vision**: Vision-only (paper as images) → Qwen2.5-VL-7B-Instruct

### Ensembling Strategies
For the `new_fewshot` variant with `n_generations=5`:
- **Single**: Use first generation only (Variant 3)
- **Majority**: Majority vote across 5 generations (Variant 4)
- **Meta-review**: LLM aggregates 5 reviews into final decision (Variant 5)

## Directory Structure

```
inference_scaling/
├── README.md               # This file
├── run_pipeline.sh         # Master pipeline script
├── data/                   # Generated datasets
│   ├── dataset_info.json   # LlamaFactory dataset configuration
│   └── {dataset}_*/        # Modified dataset folders
├── scripts/
│   ├── generate_datasets.py      # Create prompt-modified datasets
│   ├── generate_dataset_info.py  # Generate dataset_info.json
│   ├── vllm_infer_ensemble.py    # vLLM inference with n_generations
│   ├── extract_results.py        # Extract decisions from predictions
│   ├── run_metareview.py         # Meta-review aggregation
│   └── compute_metrics.py        # Metrics calculation and plotting
├── sbatch/
│   ├── run_inference.sbatch      # Main inference job array (9 jobs)
│   └── run_metareview.sbatch     # Meta-review job array (3 jobs)
├── results/                # Inference outputs
│   └── {modality}/{variant}/
│       ├── predictions.jsonl
│       ├── results_single.jsonl
│       ├── results_majority.jsonl
│       └── results_metareview.jsonl
└── metrics/                # Computed metrics and plots
    ├── summary.csv
    ├── all_metrics.json
    └── *.png
```

## Usage

### Quick Start

Run the entire pipeline:

```bash
cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer
./inference_scaling/run_pipeline.sh all
```

### Step-by-Step

1. **Generate datasets** (modifies prompts):
   ```bash
   ./inference_scaling/run_pipeline.sh generate
   ```

2. **Submit inference jobs** (SLURM job array):
   ```bash
   ./inference_scaling/run_pipeline.sh inference
   ```

3. **Wait for inference to complete**, then submit meta-review jobs:
   ```bash
   ./inference_scaling/run_pipeline.sh metareview
   ```

4. **Extract results** from predictions:
   ```bash
   ./inference_scaling/run_pipeline.sh extract
   ```

5. **Compute metrics** and generate plots:
   ```bash
   ./inference_scaling/run_pipeline.sh metrics
   ```

### Configuration

Models are automatically selected based on modality:
- **clean** (text): `Qwen/Qwen2.5-7B-Instruct` with `qwen` template
- **clean_images/vision**: `Qwen/Qwen2.5-VL-7B-Instruct` with `qwen2_vl` template

Environment variables to override defaults:
- `MODEL`: Override text model (for clean modality)
- `MODEL_VL`: Override vision-language model (for clean_images, vision)
- `TEMPLATE`: Override text template
- `TEMPLATE_VL`: Override VL template

Example:
```bash
MODEL_VL="Qwen/Qwen2.5-VL-72B-Instruct" sbatch inference_scaling/sbatch/run_inference.sbatch
```

## Metrics

The following metrics are computed:
- **Overall Accuracy**: Correct predictions / Total
- **Accept Recall** (Sensitivity): TP / (TP + FN)
- **Reject Recall** (Specificity): TN / (TN + FP)
- **In-Distribution**: Performance on 2020-2024 papers
- **Out-of-Distribution**: Performance on 2025 papers
- **By Year**: Breakdown by publication year

## Output Format

### Predictions (predictions.jsonl)
```json
{
  "prompt": "...",
  "predict": ["review1", "review2", ...] or "single_review",
  "label": "Outcome: \\boxed{Accept}",
  "n_generations": 5
}
```

### Results (results_*.jsonl)
```json
{
  "prediction": "Accept",
  "ground_truth": "Accept",
  "correct": true,
  "vote_breakdown": {"Accept": 3, "Reject": 2}
}
```

## Job Array Configuration

Main inference (9 jobs):
| Task ID | Modality | Prompt Variant | n_generations | Model |
|---------|----------|----------------|---------------|-------|
| 0 | clean | original | 1 | Qwen2.5-7B-Instruct |
| 1 | clean | new | 1 | Qwen2.5-7B-Instruct |
| 2 | clean | new_fewshot | 5 | Qwen2.5-7B-Instruct |
| 3 | clean_images | original | 1 | Qwen2.5-VL-7B-Instruct |
| 4 | clean_images | new | 1 | Qwen2.5-VL-7B-Instruct |
| 5 | clean_images | new_fewshot | 5 | Qwen2.5-VL-7B-Instruct |
| 6 | vision | original | 1 | Qwen2.5-VL-7B-Instruct |
| 7 | vision | new | 1 | Qwen2.5-VL-7B-Instruct |
| 8 | vision | new_fewshot | 5 | Qwen2.5-VL-7B-Instruct |

Meta-review (3 jobs):
| Task ID | Modality | Model |
|---------|----------|-------|
| 0 | clean | Qwen2.5-7B-Instruct |
| 1 | clean_images | Qwen2.5-VL-7B-Instruct |
| 2 | vision | Qwen2.5-VL-7B-Instruct |

## Notes

- Few-shot examples use a fixed seed (42) for reproducibility
- After generating initial results, you should identify good few-shot examples (1 true positive, 1 false negative) and regenerate
- The meta-reviewer uses the same model as the individual reviewers
- Calibration threshold defaults to 6 (overall score >= 6 = Accept)

Current error:
- model is failing and claiming max length is set at 40960 or 86016, doesn't match the inputs I am passing in.


TODO: 
Replace the format_fewshot_string function in generate_datasets.py with real examples later. 

Have logs saved to a /logs/inference_scaling/experiment_i folder.
Make ./inference_scaling/run_pipeline.sh extract save outputs to a file within /logs/inference_scaling/experiment_i folder.


check jobs

python3 -c "                                         
import os; os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/u/jl0796/.config/gcloud/application_default_credentials.json'
from google import genai
client = genai.Client(vertexai=True, project='hip-gecko-485003-c4', location='us-central1')
for job in client.batches.list():
    print(f'{job.state}: {job.name}')
"


python inference_scaling/scripts/gemini_inference.py submit     --data_dir inference_scaling/data     --output_dir inference_scaling/results/gemini     --project hip-gecko-485003-c4     --gcs_staging gs://jl0796-autoreviewer-staging/inference_scaling     --gcs_base gs://jl0796-autoreviewer-staging     --modality clean,vision