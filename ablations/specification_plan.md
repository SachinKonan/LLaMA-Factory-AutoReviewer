# Specification Plan for Ablations

## Goal
Implement the ablation studies described in `ablations/plan.md`. This involves creating new datasets with modified labels and content, and running inference and training on these datasets.

## User Review Required
- **Dataset Storage**: Identifying the correct location to store the generated datasets (likely `data/` or a new `ablations/data/` directory) and registering them in `dataset_info.json`.
- **Script Location**: Confirmation of where to place the new generation and training scripts.

## Proposed Changes

### 1. Dataset Generation
Create a new script `ablations/generate_ablation_datasets.py` to generate the required datasets.

#### Dataset 1: Label Flipping (Counterfactual)
- **Source**: `iclr_2020_2025_85_5_10_split6_original_{clean, clean+images, vision}_binary_noreviews_v6_{train, validation, test}`
- **Logic**:
    - Filter for papers < 2024.
    - Filter for papers with label "Accept".
    - Change labels to "Reject".
    - *Clarification needed*: "turn the labels... into rejects". Does this mean *only* modify those entries in the existing dataset? Or create a subset of *only* modified entries? "Each new dataset should be a subset of the base datasets below." implies we might keep the rest or filter. "subset" usually means "a portion of".
        - *Interpretation*: Select <2024 Accepted papers. Flip label to Reject. Save this as a new dataset.
- **Modalities**: `clean` (text), `clean+images`, `vision`.

#### Dataset 2: Content Filtering (Section Ablation)
- **Source**: `iclr_2020_2025_85_5_10_split6_original_{clean, clean+images}_binary_noreviews_v6_{train, validation, test}` (Text & Text+Images only).
- **Logic**:
    - Always keep **Title** and **Abstract**.
    - Parse the markdown content to identify sections: Introduction, Related Work, Methodology, Results, Discussion, Conclusion.
    - Header detection: Search for `# 1 INTRODUCTION`, `# Related Work`, etc. and their variations (case insensitive, numbering variations).
    - **Ablations**:
        1. **Introduction**: Keep ONLY Intro (plus Title/Abs) OR Keep EVERYTHING BUT Intro.
        2. **Related Work**: Keep ONLY Related Work OR Keep EVERYTHING BUT Related Work.
        3. **Methodology**: Keep ONLY Methodology OR Keep EVERYTHING BUT Methodology.
        4. **Results**: Keep ONLY Results OR Keep EVERYTHING BUT Results.
        5. **Discussion**: Keep ONLY Discussion OR Keep EVERYTHING BUT Discussion.
        6. **Intro + Conclusion**: Keep ONLY Intro+Concl OR Keep EVERYTHING BUT Intro+Concl.
    - **Filtering**: If a header is not found, filter that paper OUT.
    - **Statistics**: Plot header presence statistics.
- **Outputs**: 12 new dataset variations per modality (2 ops * 6 sections).

### 2. Dataset Registration
- Update `data/dataset_info.json` to include the new dataset definitions.

### 3. Inference & Training Scripts
- **Inference**:
    - Create `ablations/run_inference.sh` (or python script) to run inference on the Dataset 2 (Section Ablations).
    - Based on `inference_scaling/scripts/vllm_infer_ensemble.py` or similar?
- **Training**:
    - Create `ablations/run_training.sh` (or python script).
    - Adapt from `configs/qwen2_5_3b_full_sft_ds3.yaml`.
    - Create new config files in `ablations/configs/` or `configs/ablations/`.

## Verification Plan
### Automated Tests
- Verify dataset generation:
    - Check file counts.
    - Randomly sample entries to verify content filtering (e.g., "Introduction" missing in "No Intro" dataset).
    - Check JSONL validity.
- Dry run:
    - Run training on a small subset (sanity check).
    - Run inference on a small subset.

### Manual Verification
- Inspect generated `data.json` files for correctness.
- Check `dataset_info.json` entries.
