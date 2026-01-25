# Correlative Analysis Script

Analyze correlation between model predictions and structural features (tokens, pages, images) to reveal if models are overfitting to document characteristics.

## Single-Run Analysis

Analyze how predictions correlate with structural features like token count, page count, or image count.

### Dataset Types and Features

| Dataset Type | Features Plotted |
|--------------|------------------|
| `clean` | tokens |
| `clean_images` | tokens, images |
| `vision` | pages |

### Examples

```bash
# Analyze clean text model
uv run python scripts/analyze_correlative.py \
    --results_dir results/data_sweep/balanced_deepreview \
    --dataset_type clean \
    --output results/correlative/data_sweep_balanced_deepreview

# Analyze vision model
uv run python scripts/analyze_correlative.py \
    --results_dir results/hyperparam_vision_sweep/bs16_proj_frozen \
    --dataset_type vision \
    --output results/correlative/hyperparam_vision_sweep_bs16_frozen

# Analyze clean+images model
uv run python scripts/analyze_correlative.py \
    --results_dir results/hyperparam_clean_images_sweep/some_variant \
    --dataset_type clean_images \
    --output results/correlative/clean_images_analysis

# Analyze Gemini results (uses different answer parsing)
uv run python scripts/analyze_correlative.py \
    --results_dir results/gemini/clean \
    --dataset_type clean \
    --is_gemini \
    --output results/correlative/gemini_clean
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--results_dir` | Path to results directory containing `finetuned.jsonl` or `full.jsonl` |
| `--dataset_type` | One of: `clean`, `clean_images`, `vision` |
| `--output` | Output directory for plots (default: `results/correlative/{results_dir_name}`) |
| `--split` | Dataset split: `train` or `test` (default: `test`) |
| `--is_gemini` | Use Gemini-style answer parsing |
| `--variants` | Comma-separated variant names for batch processing |

### Outputs

1. **correlation.png** - Heatmap showing Pearson correlation between predictions, labels, and structural features
2. **logistic_regression.png** - Density distribution plots with mean markers

## Cross-Run Comparison

Compare predictions between two different model runs to analyze agreement and combined predictive power.

### Example

```bash
uv run python scripts/analyze_correlative.py \
    --compare \
    --run1_results results/data_sweep/balanced_deepreview \
    --run1_dataset iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3 \
    --run1_name "qwen3_clean" \
    --run2_results results/hyperparam_vision_sweep/bs16_proj_frozen \
    --run2_dataset iclr_2020_2025_80_20_split5_balanced_deepreview_vision_binary_no_reviews_titleabs_corrected_v3 \
    --run2_name "bs16_frozen_vision" \
    --output results/correlative/cross_run_clean_vs_vision
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--compare` | Enable cross-run comparison mode |
| `--run1_results` | Path to first run's results directory |
| `--run1_dataset` | Full dataset name for run 1 (without `_test` suffix) |
| `--run1_name` | Short display name for run 1 |
| `--run2_results` | Path to second run's results directory |
| `--run2_dataset` | Full dataset name for run 2 (without `_test` suffix) |
| `--run2_name` | Short display name for run 2 |
| `--output` | Output directory for plots |

### Outputs

1. **cross_correlation.png** - Correlation matrix between run1 predictions, run2 predictions, and ground truth
2. **agreement_heatmap.png** - 2x2 confusion matrix showing prediction agreement with GT breakdown
3. **3d_hyperplane.png** - 3D scatter plot with fitted linear regression hyperplane
4. **3d_hyperplane.html** - Interactive Plotly version (rotate with mouse)

## Dataset Name Reference

```python
DATASETS = {
    "clean": "iclr_2020_2025_80_20_split5_balanced_deepreview_clean_binary_no_reviews_v3",
    "clean_images": "iclr_2020_2025_80_20_split5_balanced_deepreview_clean+images_binary_no_reviews_titleabs_corrected_v3",
    "vision": "iclr_2020_2025_80_20_split5_balanced_deepreview_vision_binary_no_reviews_titleabs_corrected_v3",
}
```

## Interpreting Results

### Correlation Matrix
- High `pred-feature` correlation with low `label-feature` correlation = model overfitting to structural characteristics
- Similar `pred-feature` and `label-feature` correlations = model learning legitimate patterns

### Logistic Regression Accuracy
- `LogReg acc` shows how well a single feature predicts Accept/Reject
- 50% = random, 58% = weak signal, 65%+ = feature is predictive
- If model accuracy >> LogReg accuracy, model uses content not just structure

### Cross-Run Agreement
- High agreement = models learning similar patterns
- Low agreement = models making independent decisions
- 3D hyperplane R² shows combined predictive power
