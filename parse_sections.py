import os
import json
import glob
from collections import defaultdict
import re

base_dir = "/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/outputs/best_2025_2026_vision_v-1"
json_files = glob.glob(os.path.join(base_dir, "*", "summary_image", "section_attn.json"))

print(f"Found {len(json_files)} section_attn.json files.")

paper_sections = defaultdict(list)
unique_sections_per_paper = {}

for jf in json_files:
    paper_id = os.path.basename(os.path.dirname(os.path.dirname(jf)))
    try:
        with open(jf, "r") as f:
            data = json.load(f)
            # Find all unique section names in this paper
            secs_in_paper = set()
            for step_data in data:
                weights = step_data.get("weights", {})
                for sec_name in weights.keys():
                    secs_in_paper.add(sec_name.strip())
            unique_sections_per_paper[paper_id] = secs_in_paper
    except Exception as e:
        print(f"Error reading {jf}: {e}")

# Filter N.i sub-level sections and group top-level sections
filtered_counts = defaultdict(int)

for paper_id, secs in unique_sections_per_paper.items():
    for sec in secs:
        # Match sub-level "N.M Title" or "N.M.K Title"
        # We also need to skip things like "A.1" or "IV.A" if possible, but let's just stick to digit.digit first
        sub_match = re.match(r'^\d+\.\d+(?:\.\d+)*\s+', sec) or re.match(r'^\d+\.\d+(?:\.\d+)*$', sec)
        if sub_match:
            continue
            
        # Top-level match "N Title" -> Let's keep it
        top_match = re.match(r'^(\d+)(?:\.|\s+)(.+)$', sec)
        if top_match:
             # Normalize it a bit: e.g. "1. INTRODUCTION" -> "1 INTRODUCTION"
             num = top_match.group(1)
             title = top_match.group(2).strip().upper()
             norm_sec = f"{num} {title}"
             filtered_counts[norm_sec] += 1
        elif re.match(r'^[A-Z]+\s+(.+)$', sec, re.IGNORECASE) or len(sec) < 30: 
             # Things without numbers like "Conclusion", "Preamble", "Methodology"
             filtered_counts[sec.upper()] += 1

print(f"\nTotal Top-Level Occurrences (1 per paper max):")
for sec, count in sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:50]:
    print(f"  {count:5d} : {sec}")

with open("/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/outputs/best_2025_2026_vision_v-1/filtered_section_counts.json", "w") as f:
    json.dump(sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True), f, indent=2)

print("\nOutput saved to filtered_section_counts.json")
