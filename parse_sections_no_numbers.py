import os
import json
import glob
from collections import defaultdict
import re

base_dir = "/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/outputs/best_2025_2026_vision_v-1"
json_files = glob.glob(os.path.join(base_dir, "*", "summary_image", "section_attn.json"))

paper_sections = defaultdict(list)
unique_sections_per_paper = {}

for jf in json_files:
    paper_id = os.path.basename(os.path.dirname(os.path.dirname(jf)))
    try:
        with open(jf, "r") as f:
            data = json.load(f)
            secs_in_paper = set()
            for step_data in data:
                weights = step_data.get("weights", {})
                for sec_name in weights.keys():
                    secs_in_paper.add(sec_name.strip())
            unique_sections_per_paper[paper_id] = secs_in_paper
    except Exception as e:
        print(f"Error reading {jf}: {e}")

filtered_counts = defaultdict(int)

for paper_id, secs in unique_sections_per_paper.items():
    for sec in secs:
        # Match sub-level "N.M Title"
        sub_match = re.match(r'^\d+\.\d+(?:\.\d+)*\s+', sec) or re.match(r'^\d+\.\d+(?:\.\d+)*$', sec)
        if sub_match:
            continue
            
        # Strip leading numbers/dots and spaces
        cleaned = re.sub(r'^[\d\s\.]+', '', sec).strip().upper()
        if cleaned:
            filtered_counts[cleaned] += 1
        elif sec.upper() == "Preamble" or sec.upper() == "None":
            filtered_counts[sec.upper()] += 1

print(f"Total Top-Level Occurrences (1 per paper max) without leading numbers:")
for sec, count in sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:50]:
    print(f"  {count:5d} : {sec}")

with open("/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/outputs/best_2025_2026_vision_v-1/filtered_section_counts_no_numbers.json", "w") as f:
    json.dump(sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True), f, indent=2)
