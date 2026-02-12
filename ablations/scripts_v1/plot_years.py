
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import os
import argparse

def parse_yearly_metrics(row, metric_idx):
    """
    Parse the "Acc/AccRec/RejRec(n=...)" string.
    metric_idx: 0=Accuracy, 1=AcceptRecall, 2=RejectRecall
    """
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    values = []
    
    for year in years:
        col = f"metrics_{year}"
        if col not in row or pd.isna(row[col]):
            values.append(None)
            continue
            
        # Format: "41.2/93.3/0.0(n=34)"
        try:
            parts = str(row[col]).split('(')[0].split('/')
            val = float(parts[metric_idx])
            values.append(val)
        except (ValueError, IndexError):
            values.append(None)
            
    return values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="ablations/results_v1")
    parser.add_argument("--gemini_dir", type=str, default="ablations/results_v1/gemini_2.5_flash")
    args = parser.parse_args()

    # Load Standard (Qwen) Results
    qwen_path = Path(args.results_dir) / "analysis_summary_intersection.csv" # Use intersection for fairness
    if not qwen_path.exists():
         qwen_path = Path(args.results_dir) / "analysis_summary.csv"
    
    # Load Gemini Results
    gemini_path = Path(args.gemini_dir) / "analysis_summary_intersection.csv"
    if not gemini_path.exists():
         gemini_path = Path(args.gemini_dir) / "analysis_summary.csv"

    dfs = []
    if qwen_path.exists():
        df_q = pd.read_csv(qwen_path)
        df_q['Model'] = 'Qwen2.5-VL' # Or infer from name, but this is the primary baseline
        dfs.append(df_q)
    
    if gemini_path.exists():
        df_g = pd.read_csv(gemini_path)
        df_g['Model'] = 'Gemini 2.5 Flash'
        dfs.append(df_g)

    if not dfs:
        print("No data found.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Filter/Clean dataset names
    def clean_name(name):
        return name.replace("iclr_2020_2025_85_5_10_split7_", "").replace("_binary_noreviews_v7_test", "").replace("ablation_v1_", "")
    
    df['ShortName'] = df['dataset_name'].apply(clean_name)

    # Prepare long-form DataFrame for plotting
    plot_data = []
    
    for idx, row in df.iterrows():
        years = [2020, 2021, 2022, 2023, 2024, 2025]
        accuracies = parse_yearly_metrics(row, 0)
        rej_recalls = parse_yearly_metrics(row, 2)
        
        for i, year in enumerate(years):
            if accuracies[i] is not None:
                plot_data.append({
                    'Year': year,
                    'Dataset': row['ShortName'],
                    'Model': row['Model'],
                    'Metric': 'Accuracy',
                    'Value': accuracies[i]
                })
            if rej_recalls[i] is not None:
                plot_data.append({
                    'Year': year,
                    'Dataset': row['ShortName'],
                    'Model': row['Model'],
                    'Metric': 'Reject Recall',
                    'Value': rej_recalls[i]
                })

    long_df = pd.DataFrame(plot_data)

    # Determine "Methodology" groups for faceting
    # e.g., Base, Fewshot Full, Fewshot Parts, Critical Modifiers
    def categorize(name):
        if "base" in name: return "Base Baselines"
        if "critical" in name: return "Critical Modifiers"
        if "fewshot_fullpaper" in name: return "Fewshot Full Paper"
        if "fewshot_" in name and "fullpaper" not in name: return "Fewshot Paper Parts"
        return "Other"

    long_df['Category'] = long_df['Dataset'].apply(categorize)

    # Plot Settings
    sns.set_theme(style="whitegrid")
    categories = long_df['Category'].unique()
    metrics = ['Accuracy', 'Reject Recall']

    for category in categories:
        cat_df = long_df[long_df['Category'] == category]
        if cat_df.empty: continue

        # We want to compare Models for the same Dataset across Years
        # But there are many datasets in a category. 
        # Let's facet by Dataset.
        
        datasets = cat_df['Dataset'].unique()
        # Filter out less interesting ones if too many? No, show all.
        
        # Create a FacetGrid
        # Rows: Metric, Cols: Dataset (might be too wide)
        # Maybe separate plots per Metric
        
        for metric in metrics:
            metric_df = cat_df[cat_df['Metric'] == metric]
            
            # If "Fewshot Paper Parts", there are MANY datasets. 
            # Group by Part? (Intro, Method, etc)
            
            g = sns.FacetGrid(metric_df, col="Dataset", col_wrap=3, height=4, aspect=1.5, sharey=True)
            g.map_dataframe(sns.lineplot, x="Year", y="Value", hue="Model", style="Model", markers=True, dashes=False)
            g.add_legend()
            g.set_axis_labels("Year", f"{metric} (%)")
            g.fig.suptitle(f"{metric} over Years - {category}", y=1.02)
            
            output_name = f"plot_years_{category.replace(' ', '_').lower()}_{metric.replace(' ', '_').lower()}.png"
            g.savefig(os.path.join(args.results_dir, output_name), bbox_inches='tight', dpi=150)
            print(f"Saved {output_name}")

if __name__ == "__main__":
    main()
