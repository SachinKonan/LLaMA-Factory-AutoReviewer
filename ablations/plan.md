Use scripts in a new folder called ablations to create the following new datasets: 
1) turn the labels for all <2024 papers that were accepted into rejects. Use all 3 modalities (text, text+images, vision)
2) filters for parts of the paper and creates ablations for each set: Always keep title and abstract, then add the following 1) only/everything but introduction, 2) only/everything but related works (or similar background information header), 3) only/everything except for methodology, 4) only/everything except for results, 5) only/everythign except for the discussion, 6) only/everythign except for the introduction+discussion. Use 2 modalities text, text+images.
- For this task, plot the statistics of header names and in the case that a header is not found in the paper, err on the side of caution and filter that paper out. 

Each new dataset should be a subset of the base datasets below.

The base datasets should be:
iclr_2020_2025_85_5_10_split6_original_{clean/clean+images/vision}_binary_noreviews_v6_{train/validation/test}

After that, set up inference scripts to directly do inference on the datasets created in 2), and also training scripts adapted from /n/fs/vision-mix/jl0796/LLaMA-Factory-AutoReviewer/configs/qwen2_5_3b_full_sft_ds3.yaml to finetunes the model on the datasets.