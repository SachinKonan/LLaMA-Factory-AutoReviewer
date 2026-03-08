Implementation Resource Map & Modification Plan
This document identifies the existing scripts that will be adapted for the new experimental pipeline, details the specific modifications required, and confirms the data sources.

1. Data Sources (Confirmed)
We will use the v7 split7 datasets located in /n/fs/vision-mix/sk7524/LLaMA-Factory/data/.

Modality	Dataset Path	Model
Text-Only	iclr_2020_2025_85_5_10_split7_balanced_clean_binary_noreviews_v7_test	Qwen/Qwen2.5-7B-Instruct (and Base)
Text + Images	iclr_2024_2025_85_5_10_split7_balanced_clean_images_binary_noreviews_v7_test	Qwen/Qwen2.5-VL-7B-Instruct
Vision	iclr_2020_2025_85_5_10_split7_balanced_vision_binary_noreviews_v7_test	Qwen/Qwen2.5-VL-7B-Instruct
2. Infrastructure & Scripts
A. Title-Only Contamination Check
Goal: Estimate correlation between "Title Only" prediction confidence and acceptance/citation count.

New Script Name	Base / Similar File	Modifications Required
generate_title_dataset.py	inference_scaling/scripts/generate_datasets.py	1. Stripping Content: Create a function to strip everything except the paper title from the input.
2. Prompt Adjustment: Update prompts to ask for a decision based only on the title.
gemini_title_infer.py	inference_scaling/scripts/gemini_inference.py	1. Prompt Config: Ensure the system prompt forces prediction from title only (or just use the dataset's prompt).
2. Output Parsing: Keep logic same, but ensure we capture confidence scores or logprobs if available (Gemini usually returns text confidence; might need logprobs for better calibration check if possible, otherwise text score 1-5).
B. Role-Playing Datasets (Bias Mitigation)
Goal: Generate reviews from "Critical" and "Enthusiastic" personas.

Target Script	Existing File	Modifications Required
generate_datasets.py	inference_scaling/scripts/generate_datasets.py	1. New Flag: Add --system_prompt_type argument (choices: standard, critical, enthusiastic).
2. Prompt Templates:
- Critical: "You are a critical reviewer... If you are unsure, reject."
- Enthusiastic: "You are an optimistic reviewer... Look for novelty... If unsure, accept."
3. Output Naming: Append _critical or _enthusiastic to output directory names.
C. Base Model Inference (Bias baseline)
Goal: Run Qwen-Base (non-instruct) to see raw next-token preference for "Accept" vs "Reject".

Target Script	Existing File	Modifications Required
vllm_infer_base.py (or adapt existing)	inference_scaling/scripts/vllm_infer_ensemble.py	1. Template Handling: Base models don't use Chat templates. Need to format input as raw text completion (Result: "Review: ... Decision:").
2. Parsing: The output won't be JSON. We need to parse the generated text for "Accept" or "Reject" keywords, or restrict generation to just the decision token if doing a simple probe.
3. Logprobs: Enable logprobs for "Accept" vs "Reject" tokens to get a continuous score.
D. Strategy D (Meta-Review Pipeline)
Goal: Aggregating Critical and Enthusiastic reviews.

New Script Name	Base / Similar File	Modifications Required
run_strategy_d.py	inference_scaling/scripts/run_metareview.py	1. Input Handling: Needs to load two result files (one from Critical run, one from Enthusiastic run).
2. Prompt Construction: Construct a meta-review prompt that explicitly sections the input: "Critical Review says: [X]", "Enthusiastic Review says: [Y]".
3. Logic: "Synthesize these perspectives into a final decision."
3. Immediate Implementation Steps
Create generate_title_dataset.py:
Copy inference_scaling/scripts/generate_datasets.py.
Modify transform_to_new to extract only the title.
Update generate_datasets.py:
Add the SYSTEM_PROMPTS dictionary for roles.
Add argument parsing for prompt_type.
Create gemini_title_infer.py:
Copy inference_scaling/scripts/gemini_inference.py.
Verify it works with the title-only dataset structure.
4. Verification
Step 1: Run generate_title_dataset.py on a small subset (limit=10). Verify output JSON has only titles.
Step 2: Run generate_datasets.py with --system_prompt_type critical. Verify system prompt in output JSON.
