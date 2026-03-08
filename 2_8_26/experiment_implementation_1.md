# Checklist Optimization for Review Evaluation

**Experiment ID**: C2
**Date**: 2026-02-11
**Location**: `2_8_26/checklist_optimization/`
**Inspired by**: [EMNLP 2025 Industry Track #104](https://aclanthology.org/2025.emnlp-industry.104.pdf)

---

## Overview

This experiment develops a binary checklist system for evaluating LLM-generated reviews of academic papers. The goal is to identify the minimal set of yes/no questions that maximally correlates with ground truth accept/reject decisions.

### Key Hypothesis

A small, optimized set of binary checklist questions applied to generated reviews can predict paper outcomes as well as (or better than) the raw LLM decision, while providing interpretable evaluation criteria.

### Success Criteria

- **Primary**: Checklist-based decisions achieve accuracy ≥ raw LLM decisions from B2 standard/clean (baseline: ~70%)
- **Secondary**: Final checklist has ≤15 questions (interpretable, not overwhelming)
- **Tertiary**: Point-biserial correlation between individual questions and outcomes > 0.15

---

## Directory Structure

```
2_8_26/checklist_optimization/
├── data/
│   ├── candidate_questions.jsonl        # Stage 1: Generated candidates
│   ├── filtered_questions.jsonl         # Stage 2: After quality control
│   ├── deduplicated_questions.jsonl     # Stage 3: After semantic clustering
│   ├── question_embeddings.npy          # Stage 3: Sentence embeddings
│   ├── checklist_evaluations.jsonl      # Stage 4: Binary answers per review
│   └── optimal_checklist.json           # Final optimized checklist
├── results/
│   ├── enforceability_scores.json       # Stage 2: Consistency metrics
│   ├── clusters.json                    # Stage 3: Clustering output
│   ├── beam_search_trace.jsonl          # Stage 4: Optimization trajectory
│   └── final_predictions.jsonl          # Stage 5: Checklist-based decisions
├── metrics/
│   ├── correlation_analysis.json        # Per-question correlation scores
│   ├── checklist_metrics.json           # Final accuracy/recall/precision
│   ├── optimization_curve.png           # Beam search progress
│   ├── question_importance.png          # Top-K question rankings
│   ├── correlation_heatmap.png          # Question × outcome correlation
│   └── comparison_plot.png              # Checklist vs LLM vs majority
├── stage1_generate_candidates.py
├── stage2_filter_questions.py
├── stage3_deduplicate.py
├── stage4_beam_search.py
├── stage5_evaluate.py
├── analyze.py
├── run_pipeline.sbatch                  # Master SLURM script (all stages)
└── README.md                            # Experiment documentation
```

---

## Data Requirements

### Input Data

| Path | Count | Format | Description |
|------|-------|--------|-------------|
| `/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl` | 2024 | JSONL | LLM-generated reviews (JSON format with summary, strengths, weaknesses, decision) |
| `/n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl` | 2024 | JSONL | Ground truth decisions |

### Review Format (from predictions.jsonl)

Each line contains:
```json
{
  "prompt": "system\n...\nuser\n...",
  "predict": "{\"summary\": \"...\", \"strengths\": \"...\", \"weaknesses\": \"...\", \"score\": 7, \"decision\": \"Accept\"}",
  "label": "Outcome: \\boxed{Accept}",
  "n_generations": 1
}
```

The `predict` field is a JSON string with structured review content.

---

## Stage 1: Checklist Question Generation

### Objective
Generate a diverse pool of ~200-300 binary questions that could be answered from reading a review.

### Method
Use Gemini 2.5 Flash API to generate candidate questions from:
1. **Review content analysis**: Sample 100 reviews (50 Accept, 50 Reject) to extract evaluation patterns
2. **Meta-prompting**: Ask LLM to generate yes/no questions that distinguish strong vs weak reviews
3. **Diversity prompting**: Request questions across categories (novelty, soundness, clarity, impact, feasibility)

### Implementation: `stage1_generate_candidates.py`

**System Prompt:**
```python
f"""You are an expert in academic peer review. Generate binary (yes/no) questions that can be answered by reading a single paper review.

Generate {n_questions} questions across these categories:
- Novelty & Originality (20%)
- Technical Soundness (25%)
- Clarity & Presentation (15%)
- Experimental Validation (20%)
- Impact & Significance (20%)

Requirements:
- Each question must be answerable as yes/no from the review text alone
- Questions should correlate with accept/reject decisions
- Avoid questions requiring knowledge of the paper content
- Focus on what the reviewer wrote, not the paper itself

Output format:
{{
  "questions": [
    {{"text": "Does the review mention significant novelty?", "category": "novelty"}},
    ...
  ]
}}
"""
```

**User Prompt:**
```python
f"""Here are {n_examples} sample reviews to inform your question generation:

ACCEPT REVIEWS:
{accept_samples}

REJECT REVIEWS:
{reject_samples}

Generate {n_questions} binary evaluation questions."""
```

**Output:** `data/candidate_questions.jsonl`
```json
{"question_id": "q001", "text": "Does the review mention significant novelty?", "category": "novelty", "source": "generated"}
{"question_id": "q002", "text": "Does the review identify major technical flaws?", "category": "soundness", "source": "generated"}
...
```

**Run Command:**
```bash
# Full generation (200 questions)
python 2_8_26/checklist_optimization/stage1_generate_candidates.py \
    --input 2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl \
    --output data/candidate_questions.jsonl \
    --n_questions 200 \
    --sample_size 100

# Debug mode (20 questions, 10 samples)
python 2_8_26/checklist_optimization/stage1_generate_candidates.py \
    --input 2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl \
    --output data/candidate_questions.jsonl \
    --n_questions 20 \
    --sample_size 10 \
    --debug
```

---

## Stage 2: Quality Control and Enforceability Testing

### Objective
Filter out vague, overly-specific, or inconsistently answerable questions.

### Method
1. **Answerability test**: For each question, apply to 20 random reviews and check if LLM can answer yes/no (reject if >10% non-responses)
2. **Enforceability test**: For each question, ask LLM the same question 3 times on 5 reviews (reject if consistency < 80%)
3. **Manual filters**: Remove questions with "always yes" or "always no" answers (>90% uniform)

### Implementation: `stage2_filter_questions.py`

**Enforceability Prompt:**
```python
f"""Review: {review_text}

Question: {question_text}

Answer only "Yes" or "No" based strictly on the review content above."""
```

**Filtering Logic:**
```python
def compute_enforceability(question, reviews, n_repeats=3):
    """Test if LLM gives consistent answers to the same question."""
    consistency_scores = []

    for review in reviews:
        answers = []
        for _ in range(n_repeats):
            answer = query_gemini(question, review)
            answers.append(answer)

        # Consistency = fraction of majority vote
        majority_count = max(answers.count("Yes"), answers.count("No"))
        consistency_scores.append(majority_count / n_repeats)

    return np.mean(consistency_scores)

# Keep questions with enforceability >= 0.80
```

**Output:** `data/filtered_questions.jsonl` (~100-150 questions remain)

**Run Command:**
```bash
# Full filtering (5 reviews, 3 repeats per question)
python 2_8_26/checklist_optimization/stage2_filter_questions.py \
    --input_questions data/candidate_questions.jsonl \
    --input_reviews 2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl \
    --output data/filtered_questions.jsonl \
    --consistency_threshold 0.80 \
    --n_test_reviews 5 \
    --n_repeats 3

# Debug mode (2 reviews, 2 repeats, lower threshold)
python 2_8_26/checklist_optimization/stage2_filter_questions.py \
    --input_questions data/candidate_questions.jsonl \
    --input_reviews 2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl \
    --output data/filtered_questions.jsonl \
    --consistency_threshold 0.60 \
    --n_test_reviews 2 \
    --n_repeats 2 \
    --debug
```

---

## Stage 3: Semantic Deduplication

### Objective
Remove redundant questions by clustering semantically similar ones and selecting the best representative.

### Method
1. **Embed questions**: Use `sentence-transformers/all-MiniLM-L6-v2` (same as Q1 experiment)
2. **Cluster**: Agglomerative hierarchical clustering with cosine distance, cut at threshold (e.g., 0.7 similarity)
3. **Select representatives**: For each cluster, use LLM to choose the clearest/most actionable question

### Implementation: `stage3_deduplicate.py`

**Embedding Model:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
question_texts = [q["text"] for q in questions]
embeddings = model.encode(question_texts, show_progress_bar=True)
```

**Clustering:**
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine

# Distance matrix
n = len(embeddings)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        distance_matrix[i, j] = cosine(embeddings[i], embeddings[j])
        distance_matrix[j, i] = distance_matrix[i, j]

# Cluster
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.3,  # 0.7 similarity threshold
    metric='precomputed',
    linkage='average'
)
labels = clustering.fit_predict(distance_matrix)
```

**Representative Selection Prompt:**
```python
f"""Here are {len(cluster_questions)} similar binary evaluation questions:

{numbered_questions}

Select the question number (1-{len(cluster_questions)}) that is:
1. Most clear and unambiguous
2. Most actionable (easier to answer from review text)
3. Most likely to correlate with accept/reject decisions

Output only the number."""
```

**Output:** `data/deduplicated_questions.jsonl` (~50-80 questions)

**Run Command:**
```bash
# Full deduplication
python 2_8_26/checklist_optimization/stage3_deduplicate.py \
    --input data/filtered_questions.jsonl \
    --output data/deduplicated_questions.jsonl \
    --similarity_threshold 0.7 \
    --embeddings_output data/question_embeddings.npy

# Debug mode (no LLM selection, just take first in cluster)
python 2_8_26/checklist_optimization/stage3_deduplicate.py \
    --input data/filtered_questions.jsonl \
    --output data/deduplicated_questions.jsonl \
    --similarity_threshold 0.7 \
    --skip_llm_selection \
    --debug
```

---

## Stage 4: Beam Search Optimization

### Objective
Find the optimal subset of K questions that maximizes correlation with ground truth while maintaining diversity.

### Method
Beam search with composite objective:
- **Accuracy**: Correlation between checklist score and ground truth
- **Coverage**: Questions span multiple semantic clusters
- **Parsimony**: Prefer shorter checklists (penalty term)

### Implementation: `stage4_beam_search.py`

**Step 1: Answer All Questions on All Reviews**

Apply all deduplicated questions to all 2024 reviews using Gemini API.

**Prompt:**
```python
f"""Review: {review_text}

Answer the following questions about this review with "Yes" or "No":

1. {question_1_text}
2. {question_2_text}
...

Respond in JSON format:
{{
  "answers": [
    {{"question_id": "q001", "answer": "Yes"}},
    {{"question_id": "q002", "answer": "No"}},
    ...
  ]
}}
"""
```

**Output:** `data/checklist_evaluations.jsonl`
```json
{"review_idx": 0, "answers": {"q001": "Yes", "q002": "No", ...}, "ground_truth": "Accept"}
```

**Step 2: Beam Search**

```python
def compute_score(question_subset, evaluations, ground_truth):
    """Composite score for a subset of questions."""
    # Checklist score = fraction of yes answers
    checklist_scores = []
    for eval in evaluations:
        yes_count = sum(1 for qid in question_subset if eval["answers"][qid] == "Yes")
        checklist_scores.append(yes_count / len(question_subset))

    # Binary predictions: threshold at median
    threshold = np.median(checklist_scores)
    predictions = ["Accept" if score >= threshold else "Reject" for score in checklist_scores]

    # Accuracy
    accuracy = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt) / len(ground_truth)

    # Diversity penalty: prefer questions from different clusters
    cluster_ids = [question_to_cluster[qid] for qid in question_subset]
    diversity = len(set(cluster_ids)) / len(cluster_ids)

    # Length penalty
    length_penalty = 1.0 - (len(question_subset) / 30.0)  # Prefer <= 15 questions

    return 0.6 * accuracy + 0.3 * diversity + 0.1 * length_penalty

# Beam search
beam_width = 10
max_questions = 20

# Initialize: single best question
initial_scores = []
for qid in all_questions:
    score = compute_score([qid], evaluations, ground_truth)
    initial_scores.append((score, [qid]))

beam = sorted(initial_scores, reverse=True)[:beam_width]

# Expand beam
for step in range(1, max_questions):
    candidates = []
    for score, subset in beam:
        for qid in all_questions:
            if qid not in subset:
                new_subset = subset + [qid]
                new_score = compute_score(new_subset, evaluations, ground_truth)
                candidates.append((new_score, new_subset))

    # Keep top beam_width
    beam = sorted(candidates, reverse=True)[:beam_width]

    # Early stopping if no improvement
    if beam[0][0] < best_score_prev:
        break

# Best checklist
best_score, best_subset = beam[0]
```

**Output:** `data/optimal_checklist.json`
```json
{
  "questions": [
    {"id": "q015", "text": "Does the review mention significant novelty?", "weight": 0.23},
    {"id": "q042", "text": "Does the review identify major technical flaws?", "weight": -0.31},
    ...
  ],
  "threshold": 0.45,
  "beam_search_steps": 12,
  "final_accuracy": 0.73,
  "diversity_score": 0.85
}
```

**Run Command:**
```bash
# Full beam search
python 2_8_26/checklist_optimization/stage4_beam_search.py \
    --input_questions data/deduplicated_questions.jsonl \
    --input_reviews 2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl \
    --input_results 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl \
    --output data/optimal_checklist.json \
    --beam_width 10 \
    --max_questions 20

# Debug mode (beam_width=3, max 5 questions, only 50 reviews)
python 2_8_26/checklist_optimization/stage4_beam_search.py \
    --input_questions data/deduplicated_questions.jsonl \
    --input_reviews 2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl \
    --input_results 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl \
    --output data/optimal_checklist.json \
    --beam_width 3 \
    --max_questions 5 \
    --debug \
    --n_reviews 50
```

---

## Stage 5: Evaluation and Analysis

### Objective
Comprehensive evaluation of the optimized checklist against baselines.

### Method
1. **Apply checklist to all reviews**: Generate predictions using optimal checklist
2. **Compute metrics**: Accuracy, precision, recall, F1, by-year breakdowns
3. **Compare baselines**: Raw LLM decision, majority voting (if available), random checklist
4. **Correlation analysis**: Point-biserial correlation per question, identify most predictive questions

### Implementation: `stage5_evaluate.py`

**Metrics:**
```python
def evaluate_checklist(checklist, evaluations, ground_truth):
    """Full evaluation of checklist."""
    # Generate predictions
    predictions = apply_checklist(checklist, evaluations)

    # Overall metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, pos_label="Accept")
    recall = recall_score(ground_truth, predictions, pos_label="Accept")
    f1 = f1_score(ground_truth, predictions, pos_label="Accept")

    # By-year breakdown (requires year metadata)
    by_year = {}
    for year in [2020, 2021, 2022, 2023, 2024, 2025]:
        year_mask = [meta["year"] == year for meta in metadata]
        if sum(year_mask) > 0:
            by_year[year] = {
                "accuracy": accuracy_score(ground_truth[year_mask], predictions[year_mask]),
                "count": sum(year_mask)
            }

    # Per-question correlation (point-biserial)
    question_correlations = {}
    for qid in checklist["questions"]:
        answers = [eval["answers"][qid["id"]] for eval in evaluations]
        binary_answers = [1 if a == "Yes" else 0 for a in answers]
        binary_gt = [1 if gt == "Accept" else 0 for gt in ground_truth]

        from scipy.stats import pointbiserialr
        corr, pval = pointbiserialr(binary_answers, binary_gt)
        question_correlations[qid["id"]] = {"correlation": corr, "p_value": pval}

    return {
        "overall": {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1},
        "by_year": by_year,
        "question_correlations": question_correlations
    }
```

**Output:** `metrics/checklist_metrics.json`

**Run Command:**
```bash
# Full evaluation
python 2_8_26/checklist_optimization/stage5_evaluate.py \
    --checklist data/optimal_checklist.json \
    --evaluations data/checklist_evaluations.jsonl \
    --input_results 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl \
    --output_metrics metrics/checklist_metrics.json \
    --output_predictions results/final_predictions.jsonl

# Debug mode
python 2_8_26/checklist_optimization/stage5_evaluate.py \
    --checklist data/optimal_checklist.json \
    --evaluations data/checklist_evaluations.jsonl \
    --input_results 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl \
    --output_metrics metrics/checklist_metrics.json \
    --output_predictions results/final_predictions.jsonl \
    --debug
```

---

## Visualization and Analysis

### Implementation: `analyze.py`

**Plot 1: Optimization Curve** (`metrics/optimization_curve.png`)
- X-axis: Number of questions in checklist
- Y-axis: Composite score (accuracy + diversity + parsimony)
- Line plot showing beam search progress
- Marker at optimal point

**Plot 2: Question Importance Rankings** (`metrics/question_importance.png`)
- Horizontal bar chart of top 20 questions
- X-axis: Point-biserial correlation with ground truth
- Color-coded by category (novelty, soundness, clarity, etc.)
- Error bars showing p-value confidence

**Plot 3: Correlation Heatmap** (`metrics/correlation_heatmap.png`)
- Rows: Final checklist questions
- Columns: Accept vs Reject correlation
- Cell values: Correlation coefficient
- Annotations showing yes/no percentage by outcome

**Plot 4: Comparison Plot** (`metrics/comparison_plot.png`)
- Grouped bar chart comparing:
  - Raw LLM decision (B2 standard baseline)
  - Checklist-based decision
  - Random checklist (control)
  - Majority voting (if B1 PDR data available)
- Metrics: Accuracy, Precision, Recall
- By-modality breakdown (clean only for this experiment)

**Run Command:**
```bash
# Generate all plots
python 2_8_26/checklist_optimization/analyze.py

# Specify custom inputs
python 2_8_26/checklist_optimization/analyze.py \
    --checklist data/optimal_checklist.json \
    --metrics metrics/checklist_metrics.json \
    --baseline_results 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl
```

---

## SLURM Execution

### Master Pipeline: `run_pipeline.sbatch`

```bash
#!/bin/bash
#SBATCH --job-name=checklist_opt
#SBATCH --output=2_8_26/logs/checklist_optimization/%x_%A_%a.out
#SBATCH --error=2_8_26/logs/checklist_optimization/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4

# Job array: 0=stage1, 1=stage2, 2=stage3, 3=stage4, 4=stage5

source ~/.bashrc
cd /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer

# Activate environment (has sentence-transformers, datasets, google-genai)
source .venv_vllm_inf/bin/activate

BASE_DIR="2_8_26/checklist_optimization"

case $SLURM_ARRAY_TASK_ID in
    0)
        echo "Stage 1: Generating candidate questions..."
        python ${BASE_DIR}/stage1_generate_candidates.py \
            --input 2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl \
            --output ${BASE_DIR}/data/candidate_questions.jsonl \
            --n_questions 200 \
            --sample_size 100
        ;;
    1)
        echo "Stage 2: Filtering questions..."
        python ${BASE_DIR}/stage2_filter_questions.py \
            --input_questions ${BASE_DIR}/data/candidate_questions.jsonl \
            --input_reviews 2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl \
            --output ${BASE_DIR}/data/filtered_questions.jsonl \
            --consistency_threshold 0.80 \
            --n_test_reviews 5 \
            --n_repeats 3
        ;;
    2)
        echo "Stage 3: Deduplicating questions..."
        python ${BASE_DIR}/stage3_deduplicate.py \
            --input ${BASE_DIR}/data/filtered_questions.jsonl \
            --output ${BASE_DIR}/data/deduplicated_questions.jsonl \
            --similarity_threshold 0.7 \
            --embeddings_output ${BASE_DIR}/data/question_embeddings.npy
        ;;
    3)
        echo "Stage 4: Beam search optimization..."
        python ${BASE_DIR}/stage4_beam_search.py \
            --input_questions ${BASE_DIR}/data/deduplicated_questions.jsonl \
            --input_reviews 2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl \
            --input_results 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl \
            --output ${BASE_DIR}/data/optimal_checklist.json \
            --beam_width 10 \
            --max_questions 20
        ;;
    4)
        echo "Stage 5: Evaluation..."
        python ${BASE_DIR}/stage5_evaluate.py \
            --checklist ${BASE_DIR}/data/optimal_checklist.json \
            --evaluations ${BASE_DIR}/data/checklist_evaluations.jsonl \
            --input_results 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl \
            --output_metrics ${BASE_DIR}/metrics/checklist_metrics.json \
            --output_predictions ${BASE_DIR}/results/final_predictions.jsonl

        echo "Generating plots..."
        python ${BASE_DIR}/analyze.py
        ;;
esac

echo "Stage ${SLURM_ARRAY_TASK_ID} complete"
```

**Run Commands:**
```bash
# Submit all stages sequentially (with dependencies)
STAGE1=$(sbatch --parsable 2_8_26/checklist_optimization/run_pipeline.sbatch --array=0)
STAGE2=$(sbatch --parsable --dependency=afterok:$STAGE1 2_8_26/checklist_optimization/run_pipeline.sbatch --array=1)
STAGE3=$(sbatch --parsable --dependency=afterok:$STAGE2 2_8_26/checklist_optimization/run_pipeline.sbatch --array=2)
STAGE4=$(sbatch --parsable --dependency=afterok:$STAGE3 2_8_26/checklist_optimization/run_pipeline.sbatch --array=3)
STAGE5=$(sbatch --parsable --dependency=afterok:$STAGE4 2_8_26/checklist_optimization/run_pipeline.sbatch --array=4)

# Submit all in parallel (for debugging after initial run)
sbatch 2_8_26/checklist_optimization/run_pipeline.sbatch --array=0-4

# Run individual stages
sbatch 2_8_26/checklist_optimization/run_pipeline.sbatch --array=3  # Just beam search
```

---

## Reused Infrastructure

| Component | Location | Usage |
|-----------|----------|-------|
| Gemini API client | `inference_scaling/scripts/gemini_inference.py` | Reference for batch submission patterns |
| Sentence-transformers | `2_8_26/q1_reviewer_archetypes/cluster_reviews.py` | Embedding model setup |
| Dataset loading | `shared/analysis_utils.py` | `load_results()`, `load_predictions()` |
| Metrics computation | `shared/analysis_utils.py` | Accuracy, precision, recall |

---

## Dependencies

### Python Packages
```bash
pip install google-genai sentence-transformers scikit-learn scipy numpy matplotlib seaborn
```

### Environment Variables
```bash
export GOOGLE_API_KEY="..."  # For Gemini API access
export GOOGLE_CLOUD_PROJECT="hip-gecko-485003-c4"
```

---

## Expected Outputs

### Intermediate Files

| File | Size | Description |
|------|------|-------------|
| `data/candidate_questions.jsonl` | ~200 KB | 200-300 generated questions |
| `data/filtered_questions.jsonl` | ~100 KB | 100-150 filtered questions |
| `data/deduplicated_questions.jsonl` | ~50 KB | 50-80 unique questions |
| `data/checklist_evaluations.jsonl` | ~50 MB | All reviews × all questions = 2024 × 60 answers |
| `data/optimal_checklist.json` | ~10 KB | Final 10-15 questions with weights |

### Final Metrics

| Metric | Expected Value | Notes |
|--------|----------------|-------|
| Checklist accuracy | 0.70-0.75 | Should match or exceed B2 standard baseline (~0.70) |
| Final checklist size | 10-15 questions | Interpretable, actionable |
| Top question correlation | 0.20-0.35 | Point-biserial with ground truth |
| Beam search steps | 10-15 | Converges quickly with good initialization |

---

## Testing Protocol

### Stage 1 Test
```bash
# Generate 20 questions from 10 samples
python 2_8_26/checklist_optimization/stage1_generate_candidates.py \
    --input 2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl \
    --output data/candidate_questions_debug.jsonl \
    --n_questions 20 \
    --sample_size 10 \
    --debug

# Verify output
wc -l 2_8_26/checklist_optimization/data/candidate_questions_debug.jsonl  # Should be ~20
head -2 2_8_26/checklist_optimization/data/candidate_questions_debug.jsonl  # Inspect format
```

### Stage 2 Test
```bash
# Filter with relaxed thresholds
python 2_8_26/checklist_optimization/stage2_filter_questions.py \
    --input_questions data/candidate_questions_debug.jsonl \
    --input_reviews 2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl \
    --output data/filtered_questions_debug.jsonl \
    --consistency_threshold 0.60 \
    --n_test_reviews 2 \
    --n_repeats 2 \
    --debug

# Verify output
wc -l 2_8_26/checklist_optimization/data/filtered_questions_debug.jsonl  # Should be ~15
```

### Stage 3 Test
```bash
# Deduplicate without LLM selection
python 2_8_26/checklist_optimization/stage3_deduplicate.py \
    --input data/filtered_questions_debug.jsonl \
    --output data/deduplicated_questions_debug.jsonl \
    --similarity_threshold 0.7 \
    --skip_llm_selection \
    --debug

# Verify output
wc -l 2_8_26/checklist_optimization/data/deduplicated_questions_debug.jsonl  # Should be ~10
```

### Stage 4 Test
```bash
# Beam search on 50 reviews, max 5 questions
python 2_8_26/checklist_optimization/stage4_beam_search.py \
    --input_questions data/deduplicated_questions_debug.jsonl \
    --input_reviews 2_8_26/b2_role_prompts/results/clean/standard/predictions.jsonl \
    --input_results 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl \
    --output data/optimal_checklist_debug.json \
    --beam_width 3 \
    --max_questions 5 \
    --n_reviews 50 \
    --debug

# Verify output
cat 2_8_26/checklist_optimization/data/optimal_checklist_debug.json  # Should have 3-5 questions
```

### Stage 5 Test
```bash
# Evaluate debug checklist
python 2_8_26/checklist_optimization/stage5_evaluate.py \
    --checklist data/optimal_checklist_debug.json \
    --evaluations data/checklist_evaluations.jsonl \
    --input_results 2_8_26/b2_role_prompts/results/clean/standard/results_single.jsonl \
    --output_metrics metrics/checklist_metrics_debug.json \
    --output_predictions results/final_predictions_debug.jsonl \
    --debug

# Verify output
cat 2_8_26/checklist_optimization/metrics/checklist_metrics_debug.json  # Should have accuracy field
```

---

## Verification Checklist

After full pipeline run:

- [ ] `data/candidate_questions.jsonl` has 200-300 lines
- [ ] `data/filtered_questions.jsonl` has 100-150 lines
- [ ] `data/deduplicated_questions.jsonl` has 50-80 lines
- [ ] `data/checklist_evaluations.jsonl` has 2024 lines (one per review)
- [ ] `data/optimal_checklist.json` has 10-15 questions
- [ ] `metrics/checklist_metrics.json` shows accuracy ≥ 0.70
- [ ] `metrics/optimization_curve.png` shows convergence
- [ ] `metrics/question_importance.png` shows top questions with correlation > 0.15
- [ ] `metrics/comparison_plot.png` shows checklist competitive with LLM baseline

---

## Expected Runtime

**Model**: `Qwen/Qwen3-30B-A3B-Thinking-2507` on 2x L40 (48GB each) via vLLM

| Stage | Runtime (Full) | Runtime (Debug) | Notes |
|-------|----------------|-----------------|-------|
| Stage 1 | ~15 min | ~5 min | 38 ICLR seed + LLM expansion to 100 questions |
| Stage 2 | ~1 hour | ~10 min | 100 questions × 5 reviews × 3 repeats = 1500 prompts (batched via vLLM) |
| Stage 3 | ~30 min | ~5 min | Embedding (fast) + LLM cluster representative selection |
| Stage 4 | ~6 hours | ~30 min | ~50 questions × 4682 reviews = ~4682 batch prompts (vLLM) + beam search |
| Stage 5 | ~10 min | ~2 min | Pure computation, no inference |
| **Total** | **~8 hours** | **~1 hour** | All inference is local (no API costs) |

---

## Cost Estimation

**No API costs** — all inference runs locally on SLURM cluster GPUs.

| Resource | Allocation | Notes |
|----------|------------|-------|
| GPUs | 2x L40 (48GB each) | Tensor parallelism for 30B param MoE model |
| Memory | 64GB RAM | For dataset loading + embeddings |
| Storage | ~100MB outputs | Intermediate files + plots |

---

## Troubleshooting

### Issue: Stage 2 filters out too many questions
**Solution:** Lower `--consistency_threshold` to 0.70 or reduce `--n_repeats` to 2

### Issue: Stage 3 produces too few clusters
**Solution:** Increase `--similarity_threshold` to 0.8 (stricter clustering)

### Issue: Stage 4 beam search doesn't converge
**Solution:** Increase `--beam_width` to 15 or adjust composite score weights in code

### Issue: Model doesn't fit on 2x L40s
**Solution:** Increase `--tensor_parallel_size` to 4, or reduce `--max_model_len` in `utils.py`

### Issue: Checklist accuracy below baseline
**Solution:** Try weighted threshold optimization instead of median threshold in beam search

---

## Future Extensions

1. **Multi-modality expansion**: Test on `clean_images` and `vision` modalities
2. **Weighted checklists**: Learn question weights instead of binary yes/no aggregation
3. **Adaptive checklists**: Different checklists for different paper domains (NLP vs CV vs RL)
4. **Human validation**: Collect human answers to checklist questions for calibration
5. **Causal analysis**: Use causal inference to identify which review aspects drive decisions

---

## References

- EMNLP 2025 Industry Track #104: Checklist-based evaluation framework
- Q1 experiment: Reviewer archetypes clustering (sentence-transformers, HF dataset loading patterns)
- B2 experiment: Role prompts and LLM-generated prediction baseline
- vLLM inference patterns: `inference_scaling/scripts/vllm_infer_ensemble.py`

---

## Implementation Contract

### File Deliverables

1. **`stage1_generate_candidates.py`**
   - Input: predictions.jsonl
   - Output: candidate_questions.jsonl
   - Functions: `load_reviews()`, `sample_reviews()`, `generate_questions_gemini()`, `save_questions()`
   - CLI: `--input`, `--output`, `--n_questions`, `--sample_size`, `--debug`

2. **`stage2_filter_questions.py`**
   - Input: candidate_questions.jsonl, predictions.jsonl
   - Output: filtered_questions.jsonl, enforceability_scores.json
   - Functions: `test_answerability()`, `test_enforceability()`, `filter_questions()`, `save_filtered()`
   - CLI: `--input_questions`, `--input_reviews`, `--output`, `--consistency_threshold`, `--n_test_reviews`, `--n_repeats`, `--debug`

3. **`stage3_deduplicate.py`**
   - Input: filtered_questions.jsonl
   - Output: deduplicated_questions.jsonl, question_embeddings.npy, clusters.json
   - Functions: `embed_questions()`, `cluster_questions()`, `select_representatives_llm()`, `save_deduplicated()`
   - CLI: `--input`, `--output`, `--similarity_threshold`, `--embeddings_output`, `--skip_llm_selection`, `--debug`

4. **`stage4_beam_search.py`**
   - Input: deduplicated_questions.jsonl, predictions.jsonl, results_single.jsonl
   - Output: optimal_checklist.json, checklist_evaluations.jsonl, beam_search_trace.jsonl
   - Functions: `answer_questions_gemini()`, `compute_composite_score()`, `beam_search()`, `save_optimal_checklist()`
   - CLI: `--input_questions`, `--input_reviews`, `--input_results`, `--output`, `--beam_width`, `--max_questions`, `--n_reviews`, `--debug`

5. **`stage5_evaluate.py`**
   - Input: optimal_checklist.json, checklist_evaluations.jsonl, results_single.jsonl
   - Output: checklist_metrics.json, final_predictions.jsonl
   - Functions: `apply_checklist()`, `compute_metrics()`, `per_question_correlation()`, `save_metrics()`
   - CLI: `--checklist`, `--evaluations`, `--input_results`, `--output_metrics`, `--output_predictions`, `--debug`

6. **`analyze.py`**
   - Input: optimal_checklist.json, checklist_metrics.json, baseline results
   - Output: 4 plots (optimization_curve.png, question_importance.png, correlation_heatmap.png, comparison_plot.png)
   - Functions: `plot_optimization_curve()`, `plot_question_importance()`, `plot_correlation_heatmap()`, `plot_comparison()`
   - CLI: `--checklist`, `--metrics`, `--baseline_results`

7. **`run_pipeline.sbatch`**
   - SLURM array job (0-4) for sequential execution
   - Dependencies: Stage N+1 depends on Stage N
   - Logging: `2_8_26/logs/checklist_optimization/`

### Shared Utilities

Create `2_8_26/checklist_optimization/utils.py`:
```python
def query_gemini_batch(prompts, model="gemini-2.5-flash", max_retries=3):
    """Batch query Gemini API with exponential backoff."""
    pass

def load_predictions(path):
    """Load predictions.jsonl and parse JSON reviews."""
    pass

def load_results(path):
    """Load results_single.jsonl with ground truth."""
    pass

def parse_json_review(predict_str):
    """Parse JSON review from predict field."""
    pass

def compute_point_biserial(binary_answers, binary_outcomes):
    """Compute point-biserial correlation."""
    pass
```

### Data Flow Diagram

```
predictions.jsonl (2024 reviews)
    ↓
[Stage 1: Generate] → candidate_questions.jsonl (200-300 questions)
    ↓
[Stage 2: Filter] → filtered_questions.jsonl (100-150 questions)
    ↓
[Stage 3: Deduplicate] → deduplicated_questions.jsonl (50-80 questions)
    ↓
[Stage 4: Beam Search] → optimal_checklist.json (10-15 questions)
    ↓                        ↓
    ↓                   checklist_evaluations.jsonl (2024 × 60 answers)
    ↓                        ↓
[Stage 5: Evaluate] → final_predictions.jsonl + checklist_metrics.json
    ↓
[Analyze] → 4 plots in metrics/
```

---

## Notes

- All scripts must support `--debug` flag for rapid iteration
- Gemini API calls should use batch submission when possible (see `gemini_inference.py`)
- Intermediate files should be saved to enable re-running from any stage
- All plots should follow `shared/analysis_utils.py` style conventions
- Use `sentence-transformers/all-MiniLM-L6-v2` for consistency with Q1
- Point-biserial correlation is the standard metric for binary × continuous correlation
