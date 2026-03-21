import os
import json
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Configuration for the two balanced output directories
CONFIGS = [
    {
        'name': 'vision_balanced',
        'base_dir': '/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/outputs/best_2025_2026_vision_balanced',
        'glob_pattern': '*/summary_image/section_attn.json',
        'dataset_path': '/scratch/gpfs/ZHUANGL/jl0796/shared/data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test_500/data.json',
    },
    {
        'name': 'text_balanced',
        'base_dir': '/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/outputs/best_2025_2026_text_balanced',
        'glob_pattern': '*/summary/section_attn.json',
        'dataset_path': '/scratch/gpfs/ZHUANGL/jl0796/shared/data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json',
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
    # elif 'APPENDIX' in sec or 'SUPPLEMENTARY' in sec or 'PROOF' in sec or 'ALGORITHM' in sec:
    #     return 'Appendix_Proofs'
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
                                "paper_id": os.path.basename(os.path.dirname(os.path.dirname(jf))),
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
        'Experiments_Results', 'Discussion', 'Conclusion', 'Other'
    ])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Distribution of Attention Weights across Section Categories ({config["name"]})')
    plt.ylabel('Attention Weight')
    plt.tight_layout()

    out_path = os.path.join(base_dir, "section_weights_boxplot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved boxplot to {out_path}")

    # Build paper × category pivot (mean attention weight per category per paper)
    pivot = (
        df.groupby(["paper_id", "Category"])["Weight"]
        .mean()
        .unstack("Category")
        .reset_index()
    )
    csv_path = os.path.join(base_dir, "section_weights_by_paper.csv")
    pivot.to_csv(csv_path, index=False)
    print(f"Saved paper×category CSV to {csv_path}")

    # Generate NORMALIZED boxplot (excluding 'Other')
    section_cols = [c for c in pivot.columns if c not in ("paper_id", "Other")]
    pivot_norm = pivot.copy()
    row_sums = pivot_norm[section_cols].sum(axis=1)
    pivot_norm[section_cols] = pivot_norm[section_cols].div(row_sums, axis=0)
    
    # Melt back to long format and DROP NAs to exclude "null" sections (0s for missing sections)
    df_norm = pivot_norm.melt(id_vars="paper_id", value_vars=section_cols, 
                              var_name="Category", value_name="Normalized_Weight").dropna()
    
    plt.figure(figsize=(12, 6))
    order = ['Introduction', 'Background_RelatedWork', 'Methodology', 
             'Experiments_Results', 'Discussion', 'Conclusion']
    sns.boxplot(data=df_norm, x='Category', y='Normalized_Weight', order=[o for o in order if o in section_cols])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Normalized Attention Weights across Section Categories ({config["name"]}) - Excluding Other')
    plt.ylabel('Normalized Attention Weight (Sum=1 per paper)')
    plt.tight_layout()

    out_norm_path = os.path.join(base_dir, "section_weights_boxplot_normalized.png")
    plt.savefig(out_norm_path, dpi=150)
    plt.close()
    print(f"Saved normalized boxplot to {out_norm_path}")
def cluster_and_plot_umap(config, k_range=None, dist_threshold=None, random_state=42):
    """
    Loads the paper×category CSV, then:
      1. Normalizes each row to sum=1 (excluding 'Other')
      2. Fills remaining NaN → 0 (missing section = 0 st.devs after z-scoring)
      3. Z-scores each column (StandardScaler)
      4. Silhouette sweep over k_range to select best k
      5. K-Means with best k
      6. UMAP 2D projection + scatter plot
    """
    from sklearn.metrics import silhouette_score

    base_dir = config['base_dir']
    csv_path = os.path.join(base_dir, "section_weights_by_paper.csv")
    if not os.path.exists(csv_path):
        print(f"  CSV not found for {config['name']}, skipping clustering.")
        return

    df = pd.read_csv(csv_path)
    paper_ids = df["paper_id"].values

    # Section categories to use (exclude paper_id and Other)
    section_cols = [c for c in df.columns if c not in ("paper_id", "Other")]
    X = df[section_cols].copy()

    # 1. Row-normalize to sum=1 (on the non-Other sections)
    row_sums = X.sum(axis=1)
    X = X.div(row_sums, axis=0)

    # 2. Fill NaN → 0 (papers missing a section get 0 after z-scoring = mean)
    X = X.fillna(0.0)

    # 3. Z-score each column
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Dendrogram (Ward Hierarchical Clustering)
    print(f"  Computing Ward linkage and plotting dendrogram...")
    Z = linkage(X_scaled, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=paper_ids, leaf_rotation=90, leaf_font_size=2, no_labels=True)
    plt.title(f"Hierarchical Clustering Dendrogram (Ward) - {config['name']}")
    plt.xlabel("Paper Index")
    plt.ylabel("Distance (Ward)")
    plt.tight_layout()
    dend_path = os.path.join(base_dir, "refined_section_dendrogram.png")
    plt.savefig(dend_path, dpi=150)
    plt.close()
    print(f"  Saved dendrogram to {dend_path}")

    # 5. Determine clusters (Distance threshold OR Silhouette sweep)
    if dist_threshold is not None:
        if dist_threshold == "half_max":
            max_dist = np.max(Z[:, 2])
            dist_threshold = 0.5 * max_dist
            print(f"  Calculated half_max distance: {dist_threshold:.4f}")

        print(f"  Cutting dendrogram at distance={dist_threshold}...")
        # fcluster returns 1-indexed labels
        labels = fcluster(Z, t=dist_threshold, criterion='distance')
        labels = labels - 1
        best_k = len(np.unique(labels))
        print(f"  → Distance cut results in k={best_k}")
        
        # We'll still save a 'dummy' silhouette plot or skip it? 
        # Let's skip the sweep plots if using distance, or just mark the k.
    else:
        if k_range is None:
            k_range = range(2, 11)
        
        scores = {}
        all_labels = {}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            lbl = km.fit_predict(X_scaled)
            sc = silhouette_score(X_scaled, lbl)
            scores[k] = sc
            all_labels[k] = lbl
            print(f"  k={k}  silhouette={sc:.4f}")

        best_k = max(scores, key=scores.get)
        print(f"  → Best k={best_k} (silhouette={scores[best_k]:.4f})")

        # Plot silhouette scores
        fig, ax = plt.subplots(figsize=(7, 4))
        ks = list(scores.keys())
        sc_vals = [scores[k] for k in ks]
        ax.plot(ks, sc_vals, marker='o', color='steelblue')
        ax.axvline(best_k, color='red', linestyle='--', label=f"Best k={best_k}")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Silhouette score")
        ax.set_title(f"K-Means silhouette sweep ({config['name']})")
        ax.legend()
        plt.tight_layout()
        sil_path = os.path.join(base_dir, "refined_section_silhouette_sweep.png")
        plt.savefig(sil_path, dpi=150)
        plt.close()
        print(f"  Saved silhouette plot to {sil_path}")
        
        labels = all_labels[best_k]

    # 6. UMAP → 2D
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    embedding = reducer.fit_transform(X_scaled)

    # Plot UMAP scatter
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = plt.cm.tab10.colors
    for k in range(best_k):
        mask = labels == k
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            color=palette[k % len(palette)],
            label=f"Cluster {k} (n={mask.sum()})",
            s=18, alpha=0.75, linewidths=0
        )
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"Paper clusters by section attention ({config['name']}, k={best_k})")
    plt.tight_layout()
    out_path = os.path.join(base_dir, f"refined_section_umap_k{best_k}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved UMAP plot to {out_path}")

    # Save cluster labels + UMAP coords
    cluster_df = pd.DataFrame({"paper_id": paper_ids, "cluster": labels})
    cluster_df["umap_1"] = embedding[:, 0]
    cluster_df["umap_2"] = embedding[:, 1]
    cluster_csv = os.path.join(base_dir, f"refined_section_clusters_k{best_k}.csv")
    cluster_df.to_csv(cluster_csv, index=False)
    print(f"  Saved cluster assignments to {cluster_csv}")
    
    return cluster_csv




def plot_cluster_median_profiles(config, cluster_csv=None):
    """
    Plot the median attention profile (across section categories) for each cluster
    as overlaid line plots.
    """
    base_dir = config['base_dir']
    csv_path = os.path.join(base_dir, "section_weights_by_paper.csv")
    if not os.path.exists(csv_path):
        print(f"  CSV not found for {config['name']}, skipping median profiles.")
        return

    # Find cluster CSV (pick highest k) if not provided
    if cluster_csv is None:
        cluster_csvs = sorted(
            glob.glob(os.path.join(base_dir, "refined_section_clusters_k*.csv")),
            key=lambda p: int(re.search(r'k(\d+)', p).group(1))
        )
        if not cluster_csvs:
            print(f"  No cluster CSV found for {config['name']}, skipping.")
            return
        cluster_csv = cluster_csvs[-1]
    
    best_k = int(re.search(r'k(\d+)', cluster_csv).group(1))

    weights_df = pd.read_csv(csv_path)
    clusters_df = pd.read_csv(cluster_csv)

    merged = weights_df.merge(clusters_df[["paper_id", "cluster"]], on="paper_id", how="inner")

    section_cols = [c for c in weights_df.columns if c not in ("paper_id", "Other")]
    X = merged[section_cols].copy()

    # Row-normalize to sum=1
    row_sums = X.sum(axis=1)
    X = X.div(row_sums, axis=0).fillna(0.0)
    X["cluster"] = merged["cluster"].values

    # Compute median profile per cluster
    medians = X.groupby("cluster")[section_cols].median()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = plt.cm.tab10.colors
    for k in sorted(medians.index):
        ax.plot(
            section_cols, medians.loc[k],
            marker='o', linewidth=2, markersize=6,
            color=palette[k % len(palette)],
            label=f"Cluster {k} (n={int((merged['cluster'] == k).sum())})"
        )

    ax.set_xlabel("Section Category")
    ax.set_ylabel("Median Normalized Attention Weight")
    ax.set_title(f"Median Attention Profile per Cluster ({config['name']}, k={best_k})")
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    out_path = os.path.join(base_dir, "refined_cluster_median_profiles.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved median profile plot to {out_path}")


def plot_cluster_profile_boxplots(config, cluster_csv=None):
    """
    For each section category, plot side-by-side boxplots of normalized attention
    weight grouped by cluster. Adds Kruskal-Wallis p-values per section to show
    whether clusters differ significantly.
    """
    from scipy.stats import kruskal

    base_dir = config['base_dir']
    csv_path = os.path.join(base_dir, "section_weights_by_paper.csv")
    if not os.path.exists(csv_path):
        print(f"  CSV not found for {config['name']}, skipping profile boxplots.")
        return

    if cluster_csv is None:
        cluster_csvs = sorted(
            glob.glob(os.path.join(base_dir, "refined_section_clusters_k*.csv")),
            key=lambda p: int(re.search(r'k(\d+)', p).group(1))
        )
        if not cluster_csvs:
            print(f"  No cluster CSV found for {config['name']}, skipping.")
            return
        cluster_csv = cluster_csvs[-1]
    best_k = int(re.search(r'k(\d+)', cluster_csv).group(1))

    weights_df = pd.read_csv(csv_path)
    clusters_df = pd.read_csv(cluster_csv)

    merged = weights_df.merge(clusters_df[["paper_id", "cluster"]], on="paper_id", how="inner")

    section_cols = [c for c in weights_df.columns if c not in ("paper_id", "Other")]
    X = merged[section_cols].copy()

    # Row-normalize to sum=1
    row_sums = X.sum(axis=1)
    X = X.div(row_sums, axis=0).fillna(0.0)
    X["cluster"] = merged["cluster"].values

    # Melt to long format for seaborn
    long_df = X.melt(id_vars="cluster", value_vars=section_cols,
                     var_name="Section", value_name="Weight")
    long_df["cluster"] = long_df["cluster"].apply(lambda c: f"C{c}")

    cluster_order = sorted(long_df["cluster"].unique())

    # Kruskal-Wallis test per section
    pvals = {}
    for sec in section_cols:
        groups = [X.loc[X["cluster"] == k, sec].dropna().values
                  for k in sorted(X["cluster"].unique())]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            _, p = kruskal(*groups)
            pvals[sec] = p
        else:
            pvals[sec] = float('nan')

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=long_df, x="Section", y="Weight", hue="cluster",
                hue_order=cluster_order, order=section_cols, ax=ax)

    # Annotate p-values above each section group
    y_max = long_df["Weight"].quantile(0.98) * 1.05
    for i, sec in enumerate(section_cols):
        p = pvals[sec]
        if np.isnan(p):
            label = "n/a"
        elif p < 0.001:
            label = "p<.001***"
        elif p < 0.01:
            label = f"p={p:.3f}**"
        elif p < 0.05:
            label = f"p={p:.3f}*"
        else:
            label = f"p={p:.2f}"
        ax.text(i, y_max, label, ha='center', va='bottom', fontsize=7, fontstyle='italic')

    ax.set_xlabel("Section Category")
    ax.set_ylabel("Normalized Attention Weight")
    ax.set_title(f"Attention by Section & Cluster ({config['name']}, k={best_k}) — Kruskal-Wallis")
    ax.legend(title="Cluster", fontsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    out_path = os.path.join(base_dir, "refined_cluster_profile_boxplots.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved cluster profile boxplots to {out_path}")


def analyze_cluster_distributions(config, cluster_csv=None, random_state=42):
    """
    Merges cluster assignments with paper metadata from the dataset,
    then plots per-cluster distributions of:
      1. accept/reject decision
      2. year
      3. pct_rating (violin)
      4. citation_normalized_by_year (violin)
    """
    from ast import literal_eval

    base_dir = config['base_dir']
    dataset_path = config.get('dataset_path')
    if not dataset_path:
        print(f"  No dataset_path for {config['name']}, skipping distribution analysis.")
        return

    # Find cluster CSV — sort by numeric k to pick the highest if not provided
    if cluster_csv is None:
        cluster_csvs = sorted(
            glob.glob(os.path.join(base_dir, "refined_section_clusters_k*.csv")),
            key=lambda p: int(re.search(r'k(\d+)', p).group(1))
        )
        if not cluster_csvs:
            print(f"  No cluster CSV found for {config['name']}, skipping.")
            return
        cluster_csv = cluster_csvs[-1]
    
    print(f"  Using cluster file: {os.path.basename(cluster_csv)}")

    clusters = pd.read_csv(cluster_csv)

    # Load dataset and extract metadata
    dataset = pd.read_json(dataset_path)
    dataset["_metadata"] = dataset["_metadata"].apply(
        lambda x: literal_eval(x) if isinstance(x, str) else x
    )
    dataset["submission_id"]              = dataset["_metadata"].apply(lambda m: m.get("submission_id"))
    dataset["decision"]                   = dataset["_metadata"].apply(lambda m: m.get("answer"))
    dataset["year"]                       = dataset["_metadata"].apply(lambda m: m.get("year"))
    dataset["pct_rating"]                 = dataset["_metadata"].apply(lambda m: m.get("pct_rating"))
    dataset["citation_normalized_by_year"]= dataset["_metadata"].apply(lambda m: m.get("citation_normalized_by_year"))

    # Merge on paper_id == submission_id
    merged = clusters.merge(
        dataset[["submission_id", "decision", "year", "pct_rating", "citation_normalized_by_year"]],
        left_on="paper_id", right_on="submission_id", how="left"
    )
    n_missing = merged["decision"].isna().sum()
    if n_missing:
        print(f"  Warning: {n_missing}/{len(merged)} papers not found in dataset.")

    n_clusters = merged["cluster"].nunique()
    cluster_order = sorted(merged["cluster"].unique())
    palette = [plt.cm.tab10.colors[k % 10] for k in cluster_order]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Cluster distributions — {config['name']}", fontsize=13)

    # 1. Decision (Accept/Reject) — grouped bar
    ax = axes[0, 0]
    decision_counts = (
        merged.groupby(["cluster", "decision"]).size()
        .unstack(fill_value=0)
        .reindex(cluster_order)
    )
    decision_pct = decision_counts.div(decision_counts.sum(axis=1), axis=0) * 100
    decision_pct.plot(kind="bar", ax=ax, color=["#e05252", "#5295e0"], edgecolor="white", width=0.6)
    ax.set_title("Decision (% Accept/Reject)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("% of papers")
    ax.set_xticklabels([f"C{k}" for k in cluster_order], rotation=0)
    ax.legend(title="Decision", fontsize=8)

    # 2. Year — grouped bar
    ax = axes[0, 1]
    year_counts = (
        merged.groupby(["cluster", "year"]).size()
        .unstack(fill_value=0)
        .reindex(cluster_order)
    )
    year_pct = year_counts.div(year_counts.sum(axis=1), axis=0) * 100
    year_pct.plot(kind="bar", ax=ax, edgecolor="white", width=0.6)
    ax.set_title("Year distribution (%)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("% of papers")
    ax.set_xticklabels([f"C{k}" for k in cluster_order], rotation=0)
    ax.legend(title="Year", fontsize=8, ncol=2)

    # 3. pct_rating — violin
    ax = axes[1, 0]
    data_by_cluster = [merged.loc[merged["cluster"] == k, "pct_rating"].dropna().values for k in cluster_order]
    parts = ax.violinplot(data_by_cluster, positions=cluster_order, showmedians=True, widths=0.6)
    for pc, color in zip(parts["bodies"], palette):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_title("Reviewer score percentile (pct_rating)")
    ax.set_xlabel("Cluster")
    ax.set_xticks(cluster_order)
    ax.set_xticklabels([f"C{k}" for k in cluster_order])
    ax.set_ylabel("pct_rating")

    # 4. citation_normalized_by_year — violin
    ax = axes[1, 1]
    data_by_cluster = [merged.loc[merged["cluster"] == k, "citation_normalized_by_year"].dropna().values for k in cluster_order]
    parts = ax.violinplot(data_by_cluster, positions=cluster_order, showmedians=True, widths=0.6)
    for pc, color in zip(parts["bodies"], palette):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_title("Citations normalized by year")
    ax.set_xlabel("Cluster")
    ax.set_xticks(cluster_order)
    ax.set_xticklabels([f"C{k}" for k in cluster_order])
    ax.set_ylabel("citation_normalized_by_year")

    plt.tight_layout()
    out_path = os.path.join(base_dir, "refined_cluster_distributions.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved distribution plot to {out_path}")

    # Save merged table for further inspection
    merged_out = os.path.join(base_dir, "refined_cluster_metadata.csv")
    merged.to_csv(merged_out, index=False)
    print(f"  Saved merged metadata to {merged_out}")


def main():
    for config in CONFIGS:
        process_and_plot(config)
        print(f"Running clustering+UMAP for {config['name']} (dist_threshold='half_max')...")
        cluster_csv = cluster_and_plot_umap(config, dist_threshold='half_max')
        
        print(f"Plotting median attention profiles for {config['name']}...")
        plot_cluster_median_profiles(config, cluster_csv=cluster_csv)
        
        print(f"Plotting cluster profile boxplots for {config['name']}...")
        plot_cluster_profile_boxplots(config, cluster_csv=cluster_csv)
        
        print(f"Analyzing cluster distributions for {config['name']}...")
        analyze_cluster_distributions(config, cluster_csv=cluster_csv)

if __name__ == "__main__":
    main()
