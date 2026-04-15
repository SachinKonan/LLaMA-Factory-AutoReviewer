import os
import json
import glob
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Lab plot style
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "helvetica",
})
LABELSIZE = 20
TITLESIZE = 20
LEGENDSIZE = 20
TICKSIZE = 16

# Lab color palette
RED = "#FF8988"
ORANGE = "#FECC81"
BLUE = "#6098FF"
GREEN = "#77B25D"
PURPLE = "#B28CFF"
COLOR_PALETTE = [RED, ORANGE, BLUE, GREEN, PURPLE]

# Canonical section order (follows paper structure)
SECTION_ORDER = [
    'Introduction',
    'Background_RelatedWork',
    'Methodology',
    'Experiments_Results',
    'Discussion_Conclusion',
]

# Display labels: replace underscores with slashes
SECTION_DISPLAY = {s: s.replace('_', '/') for s in SECTION_ORDER}
SECTION_DISPLAY['Other'] = 'Other'


def _display_labels(section_cols):
    """Return display-ready labels for a list of section column names."""
    return [SECTION_DISPLAY.get(s, s.replace('_', '/')) for s in section_cols]


def _ordered_sections(available, include_other=False):
    """Return section_cols in canonical paper order, optionally appending Other."""
    ordered = [s for s in SECTION_ORDER if s in available]
    if include_other and 'Other' in available:
        ordered.append('Other')
    return ordered


def _pval_str(p):
    """Format p-value for display (used in x-axis tick labels)."""
    if np.isnan(p):
        return ""
    elif p < 0.001:
        return r"$p<.001$***"
    elif p < 0.01:
        return f"$p={p:.3f}$**"
    elif p < 0.05:
        return f"$p={p:.3f}$*"
    else:
        return f"$p={p:.2f}$"


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
    if 'INTRO' in sec or 'PREAMBLE' in sec:
        return 'Introduction'
    elif any(kw in sec for kw in ['RELATED', 'BACKGROUND', 'PRELIMINAR', 'LITERATURE']):
        return 'Background_RelatedWork'
    elif any(kw in sec for kw in ['METHOD', 'APPROACH', 'MODEL', 'ARCHITECTURE', 'PROPOSED', 'TASK DEFINITION', 'PROBLEM FORMULATION']):
        return 'Methodology'
    elif any(kw in sec for kw in ['EXPERIMENT', 'EVALUATION', 'RESULT', 'ANALYSIS', 'SETUP', 'STUDY', 'ABLATION']):
        return 'Experiments_Results'
    elif any(kw in sec for kw in ['DISCUSS', 'CONCLUS', 'SUMMARY', 'FUTURE', 'REMARK', 'LIMITATION']):
        return 'Discussion_Conclusion'
    elif any(kw in sec for kw in ['APPENDIX', 'SUPPLEMENTARY', 'PROOF', 'ALGORITHM', 'ETHICS', 'REPRODUCIBILITY']):
        return 'Appendix_Ethics_Other'
    else:
        return 'Other'

def resolve_paper_sections(paper_sections):
    # paper_sections is a list of raw section names in order
    mapping = {}
    categories = []
    for s in paper_sections:
        cleaned = re.sub(r'^[\d\s\.]+', '', s).strip().upper()
        cat = get_category(cleaned if cleaned else s)
        mapping[s] = cat
        categories.append(cat)
    
    # Identify anchor indices
    intro_idx = categories.index('Introduction') if 'Introduction' in categories else -1
    bg_idx = categories.index('Background_RelatedWork') if 'Background_RelatedWork' in categories else -1
    meth_idx = categories.index('Methodology') if 'Methodology' in categories else -1
    exp_idx = categories.index('Experiments_Results') if 'Experiments_Results' in categories else -1
    disc_idx = categories.index('Discussion_Conclusion') if 'Discussion_Conclusion' in categories else -1
    
    # Fill gaps using positional rules
    # 1. Background vs Methodology between Intro and Experiments
    if intro_idx != -1 and exp_idx != -1 and exp_idx > intro_idx:
        other_indices = [i for i in range(intro_idx + 1, exp_idx) if categories[i] == 'Other']
        if other_indices:
            if meth_idx != -1:
                for i in other_indices:
                    if i < meth_idx:
                        mapping[paper_sections[i]] = 'Background_RelatedWork'
                    else:
                        mapping[paper_sections[i]] = 'Methodology'
            elif bg_idx != -1:
                for i in other_indices:
                    if i > bg_idx:
                        mapping[paper_sections[i]] = 'Methodology'
                    else:
                        mapping[paper_sections[i]] = 'Background_RelatedWork'
            else:
                if len(other_indices) == 1:
                    mapping[paper_sections[other_indices[0]]] = 'Methodology'
                else:
                    mapping[paper_sections[other_indices[0]]] = 'Background_RelatedWork'
                    for i in other_indices[1:]:
                        mapping[paper_sections[i]] = 'Methodology'

    # 2. Discussion_Conclusion gaps
    if exp_idx != -1 and disc_idx == -1:
        appendix_idx = categories.index('Appendix_Ethics_Other') if 'Appendix_Ethics_Other' in categories else len(categories)
        for i in range(exp_idx + 1, appendix_idx):
            if categories[i] == 'Other':
                mapping[paper_sections[i]] = 'Discussion_Conclusion'
    
    # 3. Preamble/None before Introduction
    if intro_idx != -1:
        for i in range(intro_idx):
            if categories[i] == 'Other':
                mapping[paper_sections[i]] = 'Introduction'

    return mapping

def process_and_plot(config):
    base_dir = config['base_dir']
    json_files = glob.glob(os.path.join(base_dir, config['glob_pattern']))
    print(f"Processing {config['name']} with {len(json_files)} files...")

    # Group by paper
    paper_to_files = {}
    for jf in json_files:
        paper_id = os.path.basename(os.path.dirname(os.path.dirname(jf)))
        paper_to_files.setdefault(paper_id, []).append(jf)

    data_records = []

    for paper_id, files in paper_to_files.items():
        try:
            # First pass: identify all top-level sections for this paper
            paper_sections = []
            file_data = []
            for jf in files:
                with open(jf, "r") as f:
                    data = json.load(f)
                    file_data.append(data)
                    for step_data in data:
                        weights = step_data.get("weights", {})
                        for sec_name in weights.keys():
                            sub_match = re.match(r'^\d+\.\d+(?:\.\d+)*\s+', sec_name) or re.match(r'^\d+\.\d+(?:\.\d+)*$', sec_name)
                            if sub_match: continue
                            if sec_name not in paper_sections:
                                paper_sections.append(sec_name)
            
            # Resolve mapping for this paper
            resolved_mapping = resolve_paper_sections(paper_sections)
            
            # Second pass: record weights
            for data in file_data:
                for step_data in data:
                    weights = step_data.get("weights", {})
                    for sec_name, weight in weights.items():
                        if sec_name in resolved_mapping:
                            cat = resolved_mapping[sec_name]
                            actual_weight = weight[0] if isinstance(weight, list) else weight
                            data_records.append({
                                "paper_id": paper_id,
                                "Category": cat,
                                "Weight": actual_weight
                            })
        except Exception as e:
            print(f"Error processing paper {paper_id}: {e}")

    if not data_records:
        print(f"No data records found for {config['name']}")
        return

    df = pd.DataFrame(data_records)

    # Build paper × category pivot First
    pivot = (
        df.groupby(["paper_id", "Category"])["Weight"]
        .mean()
        .unstack("Category")
        .reset_index()
    )

    section_cols = [c for c in pivot.columns if c not in ("paper_id", "Other")]

    # 1. Filter: only use papers that have either 5 or 6 sections
    num_sections = pivot[section_cols].notna().sum(axis=1)
    pivot = pivot[num_sections.isin([5, 6])].copy()

    # 2. Filter out "Appendix_Ethics_Other" and use core sections
    core_cols = [c for c in section_cols if c != 'Appendix_Ethics_Other']
    
    # 3. Row-normalize to sum=1 on the remaining core sections
    row_sums = pivot[core_cols].sum(axis=1)
    pivot[core_cols] = pivot[core_cols].div(row_sums, axis=0)

    # Actually drop the Appendix column if it exists
    if 'Appendix_Ethics_Other' in pivot.columns:
        pivot = pivot.drop(columns=['Appendix_Ethics_Other'])

    # Build filtered long format dataframe for plotting raw
    df_filtered = pivot.melt(id_vars=["paper_id"], value_vars=[c for c in pivot.columns if c != "paper_id"], 
                             var_name="Category", value_name="Weight").dropna()

    plt.figure(figsize=(12, 6))
    box_order = _ordered_sections(df_filtered['Category'].unique(), include_other=True)
    sns.boxplot(data=df_filtered, x='Category', y='Weight', order=box_order,
                palette=COLOR_PALETTE[:len(box_order)])
    plt.xticks(range(len(box_order)), _display_labels(box_order), rotation=45, ha='right', fontsize=TICKSIZE)
    plt.yticks(fontsize=TICKSIZE)
    plt.title('Distribution of Attention Weights', fontsize=TITLESIZE)
    plt.ylabel('Attention Weight', fontsize=LABELSIZE)
    plt.xlabel('Section Category', fontsize=LABELSIZE)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(base_dir, "section_weights_boxplot.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot to {out_path}")

    csv_path = os.path.join(base_dir, "section_weights_by_paper.csv")
    pivot.to_csv(csv_path, index=False)
    print(f"Saved paper×category CSV to {csv_path}")

    # Generate NORMALIZED boxplot (excluding 'Other')
    section_cols_norm = [c for c in pivot.columns if c not in ("paper_id", "Other")]
    pivot_norm = pivot.copy()
    row_sums_norm = pivot_norm[section_cols_norm].sum(axis=1)
    pivot_norm[section_cols_norm] = pivot_norm[section_cols_norm].div(row_sums_norm, axis=0)
    
    # Melt back to long format and DROP NAs to exclude "null" sections (0s for missing sections)
    df_norm = pivot_norm.melt(id_vars="paper_id", value_vars=section_cols_norm, 
                              var_name="Category", value_name="Normalized_Weight").dropna()
    
    plt.figure(figsize=(12, 6))
    norm_order = _ordered_sections(section_cols_norm)
    sns.boxplot(data=df_norm, x='Category', y='Normalized_Weight', order=norm_order,
                palette=COLOR_PALETTE[:len(norm_order)])
    plt.xticks(range(len(norm_order)), _display_labels(norm_order), rotation=45, ha='right', fontsize=TICKSIZE)
    plt.yticks(fontsize=TICKSIZE)
    plt.title('Normalized Attention Weights', fontsize=TITLESIZE)
    plt.ylabel('Normalized Attention Weight', fontsize=LABELSIZE)
    plt.xlabel('Section Category', fontsize=LABELSIZE)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_norm_path = os.path.join(base_dir, "section_weights_boxplot_normalized.png")
    plt.savefig(out_norm_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved normalized boxplot to {out_norm_path}")
    return df_filtered.to_dict(orient="records")


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
    plt.title("Hierarchical Clustering Dendrogram (Ward)", fontsize=TITLESIZE)
    plt.xlabel("Paper Index", fontsize=LABELSIZE)
    plt.ylabel("Distance (Ward)", fontsize=LABELSIZE)
    plt.xticks(fontsize=TICKSIZE)
    plt.yticks(fontsize=TICKSIZE)
    plt.tight_layout()
    dend_path = os.path.join(base_dir, "refined_section_dendrogram.png")
    plt.savefig(dend_path, dpi=200, bbox_inches='tight')
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
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ks = list(scores.keys())
        sc_vals = [scores[k] for k in ks]
        ax.plot(ks, sc_vals, marker='o', linewidth=3.0, markersize=10, alpha=0.9, color=BLUE)
        ax.axvline(best_k, color=RED, linestyle='--', linewidth=2.0, label=f"Best k={best_k}")
        ax.set_xlabel(r"Number of clusters ($k$)", fontsize=LABELSIZE)
        ax.set_ylabel("Silhouette score", fontsize=LABELSIZE)
        ax.set_title("K-Means silhouette sweep", fontsize=TITLESIZE)
        ax.legend(fontsize=LEGENDSIZE)
        ax.tick_params(axis='both', labelsize=TICKSIZE)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        sil_path = os.path.join(base_dir, "refined_section_silhouette_sweep.png")
        plt.savefig(sil_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved silhouette plot to {sil_path}")
        
        labels = all_labels[best_k]

    # 6. UMAP → 2D
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    embedding = reducer.fit_transform(X_scaled)

    # Plot UMAP scatter
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in range(best_k):
        mask = labels == k
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            color=COLOR_PALETTE[k % len(COLOR_PALETTE)],
            label=f"Cluster {k} (n={mask.sum()})",
            s=40, alpha=0.8, linewidths=0
        )
    ax.legend(fontsize=LEGENDSIZE, markerscale=2, loc='upper left')
    ax.set_xlabel("UMAP-1", fontsize=LABELSIZE)
    ax.set_ylabel("UMAP-2", fontsize=LABELSIZE)
    ax.set_title(f"Paper clusters by section attention (k={best_k})", fontsize=TITLESIZE)
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    plt.tight_layout()
    out_path = os.path.join(base_dir, f"refined_section_umap_k{best_k}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
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

    all_section_cols = [c for c in weights_df.columns if c not in ("paper_id", "Other")]
    section_cols = _ordered_sections(all_section_cols)
    X = merged[section_cols].copy()

    # Row-normalize to sum=1
    row_sums = X.sum(axis=1)
    X = X.div(row_sums, axis=0).fillna(0.0)
    X["cluster"] = merged["cluster"].values

    # Compute median profile per cluster
    medians = X.groupby("cluster")[section_cols].median()

    # Plot
    markers = ["v", "^", "o", "s", "D"]
    display = _display_labels(section_cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in sorted(medians.index):
        ax.plot(
            display, medians.loc[k],
            marker=markers[k % len(markers)], linewidth=3.0, markersize=10, alpha=0.9,
            color=COLOR_PALETTE[k % len(COLOR_PALETTE)],
            label=f"Cluster {k} (n={int((merged['cluster'] == k).sum())})"
        )

    ax.set_xlabel("Section Category", fontsize=LABELSIZE)
    ax.set_ylabel("Median Normalized Attention Weight", fontsize=LABELSIZE)
    ax.set_title(f"Median Attention Profile per Cluster (k={best_k})", fontsize=TITLESIZE)
    ax.legend(fontsize=LEGENDSIZE, loc='upper left')
    ax.tick_params(axis='both', labelsize=TICKSIZE)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(base_dir, "refined_cluster_median_profiles.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
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

    all_section_cols = [c for c in weights_df.columns if c not in ("paper_id", "Other")]
    section_cols = _ordered_sections(all_section_cols)
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

    cluster_palette = {c: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, c in enumerate(cluster_order)}
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=long_df, x="Section", y="Weight", hue="cluster",
                hue_order=cluster_order, order=section_cols, palette=cluster_palette, ax=ax)

    # Build x-axis labels with p-values underneath
    display = _display_labels(section_cols)
    tick_labels = []
    for i, sec in enumerate(section_cols):
        pstr = _pval_str(pvals[sec])
        tick_labels.append(f"{display[i]}\n{pstr}" if pstr else display[i])
    ax.set_xticks(range(len(section_cols)))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=TICKSIZE)

    ax.set_xlabel("Section Category", fontsize=LABELSIZE)
    ax.set_ylabel("Normalized Attention Weight", fontsize=LABELSIZE)
    ax.set_title(f"Attention by Section \\& Cluster (k={best_k})", fontsize=TITLESIZE)
    ax.legend(title="Cluster", fontsize=LEGENDSIZE - 4, title_fontsize=LEGENDSIZE - 4,
              loc='upper left')
    ax.tick_params(axis='y', labelsize=TICKSIZE)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()

    out_path = os.path.join(base_dir, "refined_cluster_profile_boxplots.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
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
    palette = [COLOR_PALETTE[k % len(COLOR_PALETTE)] for k in cluster_order]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cluster distributions", fontsize=TITLESIZE)

    # 1. Decision (Accept/Reject) — grouped bar
    ax = axes[0, 0]
    decision_counts = (
        merged.groupby(["cluster", "decision"]).size()
        .unstack(fill_value=0)
        .reindex(cluster_order)
    )
    decision_pct = decision_counts.div(decision_counts.sum(axis=1), axis=0) * 100
    decision_pct.plot(kind="bar", ax=ax, color=[RED, BLUE], edgecolor="white", width=0.6)
    ax.set_title(r"Decision (\% Accept/Reject)", fontsize=TITLESIZE - 4)
    ax.set_xlabel("Cluster", fontsize=LABELSIZE - 4)
    ax.set_ylabel(r"\% of papers", fontsize=LABELSIZE - 4)
    ax.set_xticklabels([f"C{k}" for k in cluster_order], rotation=0, fontsize=TICKSIZE)
    ax.tick_params(axis='y', labelsize=TICKSIZE)
    ax.legend(title="Decision", fontsize=LEGENDSIZE - 6, title_fontsize=LEGENDSIZE - 6)

    # 2. Year — grouped bar
    ax = axes[0, 1]
    year_counts = (
        merged.groupby(["cluster", "year"]).size()
        .unstack(fill_value=0)
        .reindex(cluster_order)
    )
    year_pct = year_counts.div(year_counts.sum(axis=1), axis=0) * 100
    year_pct.plot(kind="bar", ax=ax, color=COLOR_PALETTE[:len(year_pct.columns)], edgecolor="white", width=0.6)
    ax.set_title(r"Year distribution (\%)", fontsize=TITLESIZE - 4)
    ax.set_xlabel("Cluster", fontsize=LABELSIZE - 4)
    ax.set_ylabel(r"\% of papers", fontsize=LABELSIZE - 4)
    ax.set_xticklabels([f"C{k}" for k in cluster_order], rotation=0, fontsize=TICKSIZE)
    ax.tick_params(axis='y', labelsize=TICKSIZE)
    ax.legend(title="Year", fontsize=LEGENDSIZE - 6, title_fontsize=LEGENDSIZE - 6, ncol=2)

    # 3. pct_rating — violin
    ax = axes[1, 0]
    data_by_cluster = [merged.loc[merged["cluster"] == k, "pct_rating"].dropna().values for k in cluster_order]
    parts = ax.violinplot(data_by_cluster, positions=cluster_order, showmedians=True, widths=0.6)
    for pc, color in zip(parts["bodies"], palette):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_title("Reviewer score percentile", fontsize=TITLESIZE - 4)
    ax.set_xlabel("Cluster", fontsize=LABELSIZE - 4)
    ax.set_xticks(cluster_order)
    ax.set_xticklabels([f"C{k}" for k in cluster_order], fontsize=TICKSIZE)
    ax.set_ylabel("pct\_rating", fontsize=LABELSIZE - 4)
    ax.tick_params(axis='y', labelsize=TICKSIZE)

    # 4. citation_normalized_by_year — violin
    ax = axes[1, 1]
    data_by_cluster = [merged.loc[merged["cluster"] == k, "citation_normalized_by_year"].dropna().values for k in cluster_order]
    parts = ax.violinplot(data_by_cluster, positions=cluster_order, showmedians=True, widths=0.6)
    for pc, color in zip(parts["bodies"], palette):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_title("Citations normalized by year", fontsize=TITLESIZE - 4)
    ax.set_xlabel("Cluster", fontsize=LABELSIZE - 4)
    ax.set_xticks(cluster_order)
    ax.set_xticklabels([f"C{k}" for k in cluster_order], fontsize=TICKSIZE)
    ax.set_ylabel("citation\_norm\_by\_year", fontsize=LABELSIZE - 4)
    ax.tick_params(axis='y', labelsize=TICKSIZE)

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.tight_layout()
    out_path = os.path.join(base_dir, "refined_cluster_distributions.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved distribution plot to {out_path}")

    # Save merged table for further inspection
    merged_out = os.path.join(base_dir, "refined_cluster_metadata.csv")
    merged.to_csv(merged_out, index=False)
    print(f"  Saved merged metadata to {merged_out}")


def partition_by_data_and_plot_profiles(config):
    """
    Partitions papers by data attributes (decision, year, pct_rating, citations)
    rather than cluster, and plots the median attention profiles (line plots)
    as well as profile boxplots for each partition category.
    """
    from ast import literal_eval
    from scipy.stats import kruskal

    base_dir = config['base_dir']
    dataset_path = config.get('dataset_path')
    csv_path = os.path.join(base_dir, "section_weights_by_paper.csv")
    
    if not dataset_path or not os.path.exists(csv_path):
        print(f"  Missing dataset or CSV for {config['name']}, skipping partitioning.")
        return

    weights_df = pd.read_csv(csv_path)

    # Load dataset and extract metadata
    dataset = pd.read_json(dataset_path)
    dataset["_metadata"] = dataset["_metadata"].apply(
        lambda x: literal_eval(x) if isinstance(x, str) else x
    )
    dataset["submission_id"]              = dataset["_metadata"].apply(lambda m: m.get("submission_id"))
    dataset["decision"]                   = dataset["_metadata"].apply(lambda m: m.get("answer"))
    dataset["year"]                       = dataset["_metadata"].apply(lambda m: int(m.get("year")) if m.get("year") else None)
    dataset["pct_rating"]                 = dataset["_metadata"].apply(lambda m: m.get("pct_rating"))
    dataset["citation_normalized_by_year"]= dataset["_metadata"].apply(lambda m: m.get("citation_normalized_by_year"))

    # Merge on paper_id == submission_id
    merged = weights_df.merge(
        dataset[["submission_id", "decision", "year", "pct_rating", "citation_normalized_by_year"]],
        left_on="paper_id", right_on="submission_id", how="inner"
    )

    all_section_cols = [c for c in weights_df.columns if c not in ("paper_id", "Other")]

    # 1. Filter: only use papers that have either 5 or 6 sections
    num_sections = merged[all_section_cols].notna().sum(axis=1)
    merged = merged[num_sections.isin([5, 6])].copy()

    # 2. Filter out "Appendix_Ethics_Other" and use core sections in paper order
    core_cols = [c for c in all_section_cols if c != 'Appendix_Ethics_Other']
    section_cols = _ordered_sections(core_cols)
    
    # 3. Row-normalize to sum=1 on the remaining sections
    X_weights = merged[section_cols].copy()
    row_sums = X_weights.sum(axis=1)
    X_weights = X_weights.div(row_sums, axis=0).fillna(0.0)
    
    # Attach partition variables back to normalized weights
    X_weights["decision"] = merged["decision"].astype(str)
    X_weights.loc[X_weights["decision"] == 'nan', "decision"] = np.nan
    
    # Year without decimal point
    X_weights["year"] = merged["year"].dropna().astype(int).astype(str)
    
    # Bin continuous variables into Tertiles
    X_weights["pct_rating_bin"] = pd.qcut(merged["pct_rating"].dropna(), q=3, labels=["Low", "Medium", "High"]).astype(str)
    
    try:
        X_weights["citation_bin"] = pd.qcut(merged["citation_normalized_by_year"].dropna(), q=3, labels=["Low", "Medium", "High"], duplicates='drop').astype(str)
    except ValueError:
        # fallback if too many duplicate edges (e.g., zeros)
        X_weights["citation_bin"] = pd.cut(merged["citation_normalized_by_year"].dropna(), bins=3, labels=["Low", "Medium", "High"]).astype(str)
    
    partitions = {
        "decision": ("Decision", "decision"),
        "year": ("Year", "year"),
        "pct_rating_bin": ("Pct Rating (Tertiles)", "pct_rating_bin"),
        "citation_bin": ("Normalized Citations (Tertiles)", "citation_bin")
    }

    markers = ["v", "^", "o", "s", "D"]

    display = _display_labels(section_cols)

    fig_med, axes_med = plt.subplots(2, 2, figsize=(16, 12))
    fig_med.suptitle("Median Attention Profiles by Data Partitions", fontsize=TITLESIZE)
    axes_med = axes_med.flatten()

    fig_box, axes_box = plt.subplots(2, 2, figsize=(24, 14))
    fig_box.suptitle("Attention Section Boxplots by Data Partitions", fontsize=TITLESIZE)
    axes_box = axes_box.flatten()

    for idx, (part_col, (part_name, safe_part_name)) in enumerate(partitions.items()):
        valid_mask = X_weights[part_col].notna() & (X_weights[part_col] != 'nan')
        df_part = X_weights[valid_mask].copy()

        if df_part.empty:
            continue

        # 1. Plot Median Profiles (Line plot)
        medians = df_part.groupby(part_col)[section_cols].median()

        ax_med = axes_med[idx]

        # Determine order
        if part_col in ["pct_rating_bin", "citation_bin"]:
            groups_order = [g for g in ["Low", "Medium", "High"] if g in medians.index]
        else:
            groups_order = sorted(medians.index)

        for i, grp in enumerate(groups_order):
            n_grp = int((df_part[part_col] == grp).sum())
            ax_med.plot(
                display, medians.loc[grp],
                marker=markers[i % len(markers)], linewidth=3.0, markersize=10, alpha=0.9,
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                label=f"{grp} (n={n_grp})"
            )

        ax_med.set_xlabel("Section Category", fontsize=LABELSIZE - 4)
        ax_med.set_ylabel("Median Norm. Attn Weight", fontsize=LABELSIZE - 4)
        ax_med.set_title(f"by {part_name}", fontsize=TITLESIZE - 4)
        ax_med.legend(title=part_name, fontsize=LEGENDSIZE - 6, title_fontsize=LEGENDSIZE - 6,
                      loc='upper left')
        ax_med.tick_params(axis='x', rotation=45, labelsize=TICKSIZE - 2)
        ax_med.tick_params(axis='y', labelsize=TICKSIZE - 2)
        ax_med.grid(True, linestyle='--', alpha=0.5)

        # 2. Plot Section Boxplots + Kruskal-Wallis p-values
        long_df = df_part.melt(id_vars=part_col, value_vars=section_cols,
                               var_name="Section", value_name="Weight")

        pvals = {}
        for sec in section_cols:
            groups = [df_part.loc[df_part[part_col] == k, sec].dropna().values for k in groups_order]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) >= 2:
                try:
                    _, p = kruskal(*groups)
                    pvals[sec] = p
                except ValueError:
                    pvals[sec] = float('nan')
            else:
                pvals[sec] = float('nan')

        ax_box = axes_box[idx]
        grp_palette = {g: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, g in enumerate(groups_order)}
        sns.boxplot(data=long_df, x="Section", y="Weight", hue=part_col,
                    hue_order=groups_order, order=section_cols, palette=grp_palette, ax=ax_box)

        # Build x-axis labels with p-values underneath
        tick_labels_box = []
        for i, sec in enumerate(section_cols):
            pstr = _pval_str(pvals.get(sec, float('nan')))
            tick_labels_box.append(f"{display[i]}\n{pstr}" if pstr else display[i])
        ax_box.set_xticks(range(len(section_cols)))
        ax_box.set_xticklabels(tick_labels_box, rotation=45, ha='right', fontsize=TICKSIZE - 2)

        ax_box.set_xlabel("Section Category", fontsize=LABELSIZE - 4)
        ax_box.set_ylabel("Normalized Attention Weight", fontsize=LABELSIZE - 4)
        ax_box.set_title(f"by {part_name}", fontsize=TITLESIZE - 4)
        ax_box.legend(title=part_name, fontsize=LEGENDSIZE - 6, title_fontsize=LEGENDSIZE - 6,
                      loc='upper left')
        ax_box.tick_params(axis='y', labelsize=TICKSIZE - 2)
        ax_box.grid(True, linestyle='--', alpha=0.5, axis='y')

    # Save consolidated median profiles
    fig_med.subplots_adjust(hspace=0.45, wspace=0.3)
    fig_med.tight_layout(rect=[0, 0, 1, 0.96])
    out_path_med_comb = os.path.join(base_dir, f"partitioned_median_profiles_combined.png")
    fig_med.savefig(out_path_med_comb, dpi=200, bbox_inches='tight')
    plt.close(fig_med)
    print(f"  Saved consolidated median profiles to {out_path_med_comb}")

    # Save consolidated boxplots
    fig_box.subplots_adjust(hspace=0.45, wspace=0.3)
    fig_box.tight_layout(rect=[0, 0, 1, 0.96])
    out_path_box_comb = os.path.join(base_dir, f"partitioned_profile_boxplots_combined.png")
    fig_box.savefig(out_path_box_comb, dpi=200, bbox_inches='tight')
    plt.close(fig_box)
    print(f"  Saved consolidated profile boxplots to {out_path_box_comb}")


def main():
    cluster = False
    partition = True
    for config in CONFIGS:
        data_records = process_and_plot(config)
        ### structure:
        # [{
        #     "paper_id": paper_id,
        #     "Category": cat,
        #     "Weight": actual_weight
        # }, ...]
        
        if cluster:
            print(f"Running clustering+UMAP for {config['name']} (dist_threshold='half_max')...")
            cluster_csv = cluster_and_plot_umap(config, dist_threshold='half_max')
            
            print(f"Plotting median attention profiles for {config['name']}...")
            plot_cluster_median_profiles(config, cluster_csv=cluster_csv)
            
            print(f"Plotting cluster profile boxplots for {config['name']}...")
            plot_cluster_profile_boxplots(config, cluster_csv=cluster_csv)
            
            print(f"Analyzing cluster distributions for {config['name']}...")
            analyze_cluster_distributions(config, cluster_csv=cluster_csv)
        if partition:
            print(f"Partitioning data and plotting profiles for {config['name']}...")
            partition_by_data_and_plot_profiles(config)
            
            


if __name__ == "__main__":
    main()
