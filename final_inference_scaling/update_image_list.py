import json
import os

vision_test_path = "/scratch/gpfs/ZHUANGL/jl0796/shared/data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json"
output_path = "/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/inference_scaling/test_images_list.txt"

with open(vision_test_path) as f:
    data = json.load(f)

image_list = set()
for item in data:
    images = item.get("images", [])
    if isinstance(images, list):
        for img in images:
            image_list.add(img)
    elif isinstance(images, str):
        image_list.add(images)

with open(output_path, "w") as f:
    for img in sorted(list(image_list)):
        f.write(f"{img}\n")

print(f"Extracted {len(image_list)} unique images from test set to {output_path}")
