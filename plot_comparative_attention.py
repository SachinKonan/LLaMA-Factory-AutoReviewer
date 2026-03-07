import json
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy import stats

# Configuration
# BALANCED_EVAL_JSON = '/scratch/gpfs/ZHUANGL/jl0796/shared/data/iclr_balanced_50_test.json'
METADATA_FILES = [
    '/scratch/gpfs/ZHUANGL/jl0796/shared/data/iclr_balanced_50_vision_test.json',
    '/scratch/gpfs/ZHUANGL/jl0796/shared/data/iclr_balanced_50_text_test.json'
]

CONFIGS = [
    {
        'name': 'vision',
        'out_dir': '/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/outputs/best_2025_2026_vision_balanced'
    },
    {
        'name': 'text',
        'out_dir': '/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/outputs/best_2025_2026_text_balanced'
    }
]

def get_topic(title):
    title = title.upper()
    if any(k in title for k in ['META-LEARNING', 'META LEARNING', 'META-TRANSFER', 'LEARNING TO LEARN']): return 'Meta-Learning'
    if 'GRAPH' in title: return 'Graphs'
    if any(k in title for k in ['RECURRENT', 'SEQUENTIAL', 'TRANSFORMER', 'RNN', 'LSTM']): return 'NLP/Sequential'
    if any(k in title for k in ['IMAGE', 'VISION', 'CONVOLUTION', 'CNN']): return 'Computer Vision'
    if any(k in title for k in ['ADVERSARIAL', 'ROBUSTNESS', 'PERTURBATION']): return 'Robustness'
    return 'Core AI'

def get_category(sec_name):
    sec = sec_name.upper()
    if 'INTRO' in sec: return 'Introduction'
    if any(k in sec for k in ['RELATED', 'BACKGROUND', 'PRELIMINAR', 'LITERATURE', 'PROBLEM SETUP', 'NOTATION', 'FORMULATION']): return 'Background'
    if any(k in sec for k in ['METHOD', 'APPROACH', 'MODEL', 'ARCHITECTURE', 'ALGORITHM', 'FRAMEWORK', 'PROPOSED']): return 'Methodology'
    if any(k in sec for k in ['EXPERIMENT', 'EVALUATION', 'RESULT', 'ANALYSIS', 'ABLATION', 'PERFORMANCE']): return 'Experiments'
    if any(k in sec for k in ['DISCUSS', 'LIMITATION', 'IMPLICATION']): return 'Discussion'
    if any(k in sec for k in ['CONCLUS', 'SUMMARY']): return 'Conclusion'
    if any(k in sec for k in ['APPENDIX', 'SUPPLEMENTARY', 'PROOF', 'THEOR', 'LEMMA']): return 'Appendix'
    return 'Other'

def load_metadata_mapping():
    mapping = {}
    for meta_file in METADATA_FILES:
        if not os.path.exists(meta_file):
            print(f"Dataset not found: {meta_file}")
            continue
        
        with open(meta_file, 'r') as f:
            eval_data = json.load(f)
        
        for eval_item in eval_data:
            meta = eval_item['_metadata']
            sub_id = meta['submission_id']
            year = meta['year']
            gt = meta['answer']
            decision_raw = meta.get('decision', 'reject').capitalize()
            ratings = meta.get('ratings', [])
            avg_rating = np.mean(ratings) if ratings else 0
            citation = meta.get('citation', 0)
            
            prompt = eval_item['conversations'][1]['value']
            m = re.search(r'# (.+?)\n', prompt)
            title = m.group(1) if m else 'Unknown'
            
            score_tier = "Low (1-3)"
            if avg_rating >= 7: score_tier = "High (7+)"
            elif avg_rating >= 5: score_tier = "Mid (5-6)"
            
            cite_tier = "Low (<5)"
            if citation >= 20: cite_tier = "High (20+)"
            elif citation >= 5: cite_tier = "Med (5-19)"
            
            mapping[sub_id] = {
                'year': str(year),
                'gt': gt,
                'topic': get_topic(title),
                'decision_type': decision_raw,
                'score_tier': score_tier,
                'cite_tier': cite_tier
            }
    return mapping

def main():
    metadata = load_metadata_mapping()
    
    for config in CONFIGS:
        name = config['name']
        out_dir = config['out_dir']
        
        if not os.path.exists(out_dir):
            print(f"Skipping missing directory: {out_dir}")
            continue
            
        data_records = []
        files = glob.glob(os.path.join(out_dir, '*/summary_image/section_attn.json'))
        if not files:
            files = glob.glob(os.path.join(out_dir, '*/summary/section_attn.json'))
    
        for af in files:
            sub_id = af.split('/')[-3]
            if sub_id not in metadata: continue
            meta = metadata[sub_id]
            
            with open(af, 'r') as f:
                attn_data = json.load(f)
                target_step = None
                pred_label = 'None'
                for step in reversed(attn_data):
                    token = step.get('token', '').lower()
                    if 'accept' in token:
                        target_step = step
                        pred_label = 'Accept'
                        break
                    elif 'reject' in token:
                        target_step = step
                        pred_label = 'Reject'
                        break
                
                if not target_step: continue
                
                weights = target_step.get('weights', {})
                for sec_name, weight in weights.items():
                    if re.match(r'^\d+\.\d+(\.\d+)*', sec_name): continue
                    cleaned = re.sub(r'^[\d\s\.]+', '', sec_name).strip().upper()
                    if not cleaned and sec_name.upper() not in ['PREAMBLE', 'NONE']: continue
                    
                    cat = get_category(cleaned if cleaned else sec_name)
                    data_records.append({
                        'Year': meta['year'],
                        'GroundTruth': meta['gt'],
                        'Prediction': pred_label,
                        'Topic': meta['topic'],
                        'DecisionType': meta['decision_type'],
                        'ScoreTier': meta['score_tier'],
                        'CiteTier': meta['cite_tier'],
                        'Section': cat,
                        'AttentionWeight': weight[0] if isinstance(weight, list) else weight
                    })
        
        if not data_records:
            print(f"No records found for {out_dir}")
            continue

        df = pd.DataFrame(data_records)
        df_pred = df[df['Prediction'].isin(['Accept', 'Reject'])].copy()
        df_gt = df[df['GroundTruth'].isin(['Accept', 'Reject'])].copy()

        # Premium plotting setup
        plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16})
        sns.set_theme(style='whitegrid', palette='muted')
        
        fig, axes = plt.subplots(4, 2, figsize=(26, 34))
        axes = axes.flatten()
        
        order = ['Introduction', 'Background', 'Methodology', 'Experiments', 'Discussion', 'Conclusion', 'Appendix', 'Other']
        
        comparisons = [
            ('GroundTruth', 'by Ground Truth (Accept vs Reject)', df_gt, ['Accept', 'Reject']),
            ('Prediction', 'by Model Prediction (Accept vs Reject)', df_pred, ['Accept', 'Reject']),
            ('Year', 'by Paper Year', df, sorted(df['Year'].unique())),
            ('Topic', 'by Research Topic', df, sorted(df['Topic'].unique())),
            ('DecisionType', 'by Granular Decision', df, ['Oral', 'Spotlight', 'Poster', 'Reject']),
            ('ScoreTier', 'by Reviewer Score Tier', df, ['High (7+)', 'Mid (5-6)', 'Low (1-3)']),
            ('CiteTier', 'by Citation Impact', df, ['High (20+)', 'Med (5-19)', 'Low (<5)']),
        ]

        for i, (col, title_suffix, plot_df, hue_order) in enumerate(comparisons):
            if plot_df.empty:
                axes[i].text(0.5, 0.5, f"No data for {col}", ha='center', va='center')
                continue

            # Calculate stats for the title and p-values
            unique_papers = plot_df.drop_duplicates(subset=['Year','GroundTruth','Prediction','Topic','DecisionType','ScoreTier','CiteTier'])
            counts = unique_papers[col].value_counts()
            n_total = len(unique_papers)
            
            p_values = {}
            if col in ['GroundTruth', 'Prediction']:
                for sec in order:
                    sec_data = plot_df[plot_df['Section'] == sec]
                    a = sec_data[sec_data[col] == 'Accept']['AttentionWeight']
                    r = sec_data[sec_data[col] == 'Reject']['AttentionWeight']
                    if len(a) > 2 and len(r) > 2:
                        try:
                            _, p = stats.mannwhitneyu(a, r, alternative='two-sided')
                            p_values[sec] = p
                        except: pass
            
            # Subplot Title
            axes[i].set_title(f"Attention {title_suffix}\n(Total N={n_total})", fontweight='bold', pad=20)
            
            # Plot
            sns.boxplot(data=plot_df, x='Section', y='AttentionWeight', hue=col, 
                        hue_order=hue_order, order=order, ax=axes[i], 
                        showfliers=False, palette='viridis' if len(hue_order) > 2 else 'Set2')
            
            # Update X-labels with p-values
            x_labels_final = []
            for sec in order:
                lbl = sec
                if sec in p_values:
                    p = p_values[sec]
                    if p < 0.001: lbl += "\n(p<.001)"
                    elif p < 0.05: lbl += f"\n(p={p:.3f})"
                    else: lbl += "\n(ns)"
                x_labels_final.append(lbl)
            
            axes[i].set_xticks(range(len(order)))
            axes[i].set_xticklabels(x_labels_final, rotation=35, ha='right')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Attention Weight')
            
            # Legend with counts
            handles, labels = axes[i].get_legend_handles_labels()
            new_labels = []
            for l in labels:
                count = counts.get(l, 0)
                new_labels.append(f"{l} (n={count})")
            axes[i].legend(handles, new_labels, title=col, frameon=True, fontsize=12)

        # Remove extra subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.subplots_adjust(hspace=0.6, wspace=0.3)
        out_path = os.path.join(out_dir, f'{name}_attn_comparative_matrix_stat.png')
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f'Saved refined statistical matrix to {out_path}')

if __name__ == "__main__":
    main()
