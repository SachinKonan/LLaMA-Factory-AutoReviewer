import json
from collections import defaultdict

with open("/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/outputs/best_2025_2026_vision_v-1/filtered_section_counts_no_numbers.json", "r") as f:
    section_counts = json.load(f)

def get_category(sec):
    if 'INTRO' in sec:
        return 'Introduction'
    elif 'RELATED' in sec or 'BACKGROUND' in sec or 'PRELIMINAR' in sec or 'LITERATURE' in sec:
        return 'Background_RelatedWork'
    elif 'METHOD' in sec or 'APPROACH' in sec or 'MODEL' in sec or 'ARCHITECTURE' in sec or 'PROPOSED' in sec:
        return 'Methodology'
    elif 'EXPERIMENT' in sec or 'EVALUATION' in sec or 'RESULT' in sec or 'ANALYSIS' in sec or 'SETUP' in sec or 'STUDY' in sec:
        return 'Experiments_Results'
    elif 'DISCUSS' in sec:
        return 'Discussion'
    elif 'CONCLUS' in sec or 'SUMMARY' in sec or 'FUTURE' in sec or 'REMARK' in sec:
        return 'Conclusion'
    elif 'APPENDIX' in sec or 'SUPPLEMENTARY' in sec or 'PROOF' in sec or 'ALGORITHM' in sec:
        return 'Appendix_Proofs'
    else:
        return 'Other'

categorized_sections = defaultdict(list)
category_counts = defaultdict(int)

for sec_name, count in section_counts:
    cat = get_category(sec_name)
    categorized_sections[cat].append((sec_name, count))
    category_counts[cat] += count

print("=== Categories and their Sections ===")
for cat, items in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"\n{cat} (Total: {items})")
    for sec_name, count in categorized_sections[cat]:
        print(f"  {count:3d} | {sec_name}")

with open("/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/outputs/best_2025_2026_vision_v-1/categorized_sections_no_numbers.json", "w") as f:
    json.dump({
        "summary_counts": dict(category_counts),
        "detailed_mapping": dict(categorized_sections)
    }, f, indent=2)
