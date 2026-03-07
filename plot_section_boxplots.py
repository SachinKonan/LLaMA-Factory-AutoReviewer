import os
import json
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configuration for the two balanced output directories
CONFIGS = [
    {
        'name': 'vision_balanced',
        'base_dir': '/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/outputs/best_2025_2026_vision_balanced',
        'glob_pattern': '*/summary_image/section_attn.json'
    },
    {
        'name': 'text_balanced',
        'base_dir': '/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/outputs/best_2025_2026_text_balanced',
        'glob_pattern': '*/summary/section_attn.json'
    }
]

def get_category(sec_name):
    sec = sec_name.upper()
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

def process_and_plot(config):
    base_dir = config['base_dir']
    json_files = glob.glob(os.path.join(base_dir, config['glob_pattern']))
    print(f"Processing {config['name']} with {len(json_files)} files...")

    # We will collect all (category, unnormalized_weight) pairs
    data_records = []

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
                # data: [{"step": idx, "token": text, "weights": {"Section Name": weight, ...}}, ...]
                for step_data in data:
                    weights = step_data.get("weights", {})
                    for sec_name, weight in weights.items():
                        # Same logic to strip numbers and resolve category
                        sub_match = re.match(r'^\d+\.\d+(?:\.\d+)*\s+', sec_name) or re.match(r'^\d+\.\d+(?:\.\d+)*$', sec_name)
                        if sub_match:
                            continue # Skip subsections
                            
                        cleaned = re.sub(r'^[\d\s\.]+', '', sec_name).strip().upper()
                        if cleaned or (sec_name.upper() in ["PREAMBLE", "NONE"]):
                            cat = get_category(cleaned if cleaned else sec_name)
                            # Handle weight being a list or a single float
                            actual_weight = weight[0] if isinstance(weight, list) else weight
                            data_records.append({
                                "Category": cat,
                                "Weight": actual_weight
                            })
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    if not data_records:
        print(f"No data records found for {config['name']}")
        return

    df = pd.DataFrame(data_records)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Category', y='Weight', order=[
        'Introduction', 'Background_RelatedWork', 'Methodology', 
        'Experiments_Results', 'Discussion', 'Conclusion', 'Appendix_Proofs', 'Other'
    ])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Distribution of Attention Weights across Section Categories ({config["name"]})')
    plt.ylabel('Attention Weight')
    plt.tight_layout()

    out_path = os.path.join(base_dir, "section_weights_boxplot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved boxplot to {out_path}")

def main():
    for config in CONFIGS:
        process_and_plot(config)

if __name__ == "__main__":
    main()
