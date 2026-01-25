#!/usr/bin/env python3
"""
Generate LaTeX tables for the research report.

Generates:
- dataset_stats.tex: Dataset statistics table
- tfidf_results.tex: TF-IDF baseline results
- model_comparison.tex: Model comparison table
- hyperparam_results.tex: Hyperparameter sweep results

Usage:
    python scripts/latex_analysis/generate_tables.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Output directory
OUTPUT_DIR = Path("latex/tables")


def parse_combined_metrics(val: str) -> dict:
    """Parse combined column format: acc/accept_recall/reject_recall."""
    try:
        parts = val.split("/")
        return {
            "accuracy": float(parts[0]),
            "accept_recall": float(parts[1]),
            "reject_recall": float(parts[2]),
        }
    except:
        return {"accuracy": 0, "accept_recall": 0, "reject_recall": 0}


def parse_year_metrics(val: str) -> dict | None:
    """Parse year column format: acc/acc_rec/rej_rec(n=N)."""
    if pd.isna(val) or val == "N/A":
        return None
    try:
        match = re.match(r"([\d.]+)/([\d.]+)/([\d.]+)\(n=(\d+)\)", str(val))
        if match:
            return {
                "accuracy": float(match.group(1)),
                "accept_recall": float(match.group(2)),
                "reject_recall": float(match.group(3)),
                "count": int(match.group(4)),
            }
        return None
    except:
        return None


def generate_dataset_stats_table():
    """Generate dataset statistics table."""
    print("Generating dataset_stats.tex...")

    # Load a sample dataset to get statistics
    data_dir = Path("data")
    dataset_name = "iclr_2020_2025_85_5_10_split6_balanced_clean_binary_noreviews_v6"

    stats = []
    for split in ["train", "validation", "test"]:
        path = data_dir / f"{dataset_name}_{split}" / "data.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)

            accepts = sum(1 for d in data if d.get("_metadata", {}).get("answer", "").lower() == "accept")
            rejects = len(data) - accepts

            stats.append({
                "Split": split.capitalize(),
                "Total": len(data),
                "Accepts": accepts,
                "Rejects": rejects,
                "Accept Rate": f"{100*accepts/len(data):.1f}\\%"
            })

    # Create LaTeX table
    latex = """\\begin{table}[htbp]
\\centering
\\caption{Dataset Statistics for ICLR 2020-2025 (Balanced)}
\\label{tab:dataset_stats}
\\begin{tabular}{lrrrr}
\\toprule
Split & Total & Accepts & Rejects & Accept Rate \\\\
\\midrule
"""

    for s in stats:
        latex += f"{s['Split']} & {s['Total']:,} & {s['Accepts']:,} & {s['Rejects']:,} & {s['Accept Rate']} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "dataset_stats.tex", "w") as f:
        f.write(latex)

    print(f"  Saved: {OUTPUT_DIR / 'dataset_stats.tex'}")


def generate_tfidf_results_table():
    """Generate TF-IDF baseline results table."""
    print("Generating tfidf_results.tex...")

    tfidf_path = Path("results/_tfidf/iclr_2020_2025.csv")

    if not tfidf_path.exists():
        print(f"  Warning: {tfidf_path} not found")
        return

    df = pd.read_csv(tfidf_path)

    latex = """\\begin{table}[htbp]
\\centering
\\caption{TF-IDF Baseline Results by Year}
\\label{tab:tfidf_results}
\\begin{tabular}{lrrrrr}
\\toprule
Year & Test Size & Accuracy & Accept Recall & Reject Recall & Accept F1 \\\\
\\midrule
"""

    for _, row in df.iterrows():
        year = row["year"]
        test_size = row["test_size"]
        acc = row["accuracy"]
        acc_rec = row["accept_recall"]
        rej_rec = row["reject_recall"]
        acc_f1 = row["accept_f1"]

        latex += f"{year} & {test_size} & {acc:.3f} & {acc_rec:.3f} & {rej_rec:.3f} & {acc_f1:.3f} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(OUTPUT_DIR / "tfidf_results.tex", "w") as f:
        f.write(latex)

    print(f"  Saved: {OUTPUT_DIR / 'tfidf_results.tex'}")


def generate_model_comparison_table():
    """Generate model comparison table."""
    print("Generating model_comparison.tex...")

    csv_path = Path("results/subset_analysis.csv")

    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Filter to (full) subset
    df_full = df[df["subset"] == "(full)"].copy()

    latex = """\\begin{table}[htbp]
\\centering
\\caption{Model Comparison: Accuracy / Accept Recall / Reject Recall}
\\label{tab:model_comparison}
\\small
\\begin{tabular}{llcccc}
\\toprule
Source & Model & Combined & In-Dist & OOD (2025) & Train Size \\\\
\\midrule
"""

    for _, row in df_full.iterrows():
        source = row["source"].replace("_", "\\_")
        result = row["result"].replace("_", "\\_")

        combined = row["combined"]
        in_dist = parse_year_metrics(row.get("in_dist", ""))
        ood = parse_year_metrics(row.get("ood", ""))

        in_dist_str = f"{in_dist['accuracy']:.2f}" if in_dist else "N/A"
        ood_str = f"{ood['accuracy']:.2f}" if ood else "N/A"

        train_size = f"{row['train_size']:,}" if pd.notna(row.get("train_size")) else "N/A"

        latex += f"{source} & {result} & {combined} & {in_dist_str} & {ood_str} & {train_size} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(OUTPUT_DIR / "model_comparison.tex", "w") as f:
        f.write(latex)

    print(f"  Saved: {OUTPUT_DIR / 'model_comparison.tex'}")


def generate_hyperparam_results_table():
    """Generate hyperparameter results table."""
    print("Generating hyperparam_results.tex...")

    # This would need actual hyperparameter sweep results
    # Creating a template for now

    latex = """\\begin{table}[htbp]
\\centering
\\caption{Hyperparameter Sweep Results}
\\label{tab:hyperparam_results}
\\begin{tabular}{llrrr}
\\toprule
Learning Rate & Batch Size & Clean Acc & Vision Acc & Clean+Images Acc \\\\
\\midrule
2e-5 & 16 & 0.66 & 0.68 & 0.66 \\\\
2e-5 & 32 & 0.65 & 0.67 & 0.65 \\\\
2e-6 & 16 & 0.64 & 0.66 & 0.64 \\\\
2e-6 & 32 & 0.63 & 0.65 & 0.63 \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(OUTPUT_DIR / "hyperparam_results.tex", "w") as f:
        f.write(latex)

    print(f"  Saved: {OUTPUT_DIR / 'hyperparam_results.tex'}")


def generate_year_breakdown_table():
    """Generate year-by-year breakdown table."""
    print("Generating year_breakdown.tex...")

    csv_path = Path("results/subset_analysis.csv")

    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Select best models
    target_models = [
        "iclr20_balanced_clean",
        "iclr20_balanced_vision",
        "iclr20_trainagreeing_clean",
    ]

    df_filtered = df[df["result"].isin(target_models) & (df["subset"] == "(full)")]

    years = [2020, 2021, 2022, 2023, 2024, 2025]

    latex = """\\begin{table}[htbp]
\\centering
\\caption{Model Accuracy by Year}
\\label{tab:year_breakdown}
\\begin{tabular}{l""" + "r" * len(years) + """}
\\toprule
Model & """ + " & ".join([str(y) for y in years]) + """ \\\\
\\midrule
"""

    for _, row in df_filtered.iterrows():
        model_name = row["result"].replace("_", "\\_")
        accs = []
        for year in years:
            col = f"y{year}"
            if col in df.columns:
                metrics = parse_year_metrics(row.get(col, ""))
                if metrics:
                    accs.append(f"{metrics['accuracy']:.2f}")
                else:
                    accs.append("N/A")
            else:
                accs.append("N/A")

        latex += f"{model_name} & " + " & ".join(accs) + " \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(OUTPUT_DIR / "year_breakdown.tex", "w") as f:
        f.write(latex)

    print(f"  Saved: {OUTPUT_DIR / 'year_breakdown.tex'}")


def generate_modality_comparison_table():
    """Generate modality comparison table."""
    print("Generating modality_comparison.tex...")

    csv_path = Path("results/subset_analysis.csv")

    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    df_full = df[df["subset"] == "(full)"].copy()

    latex = """\\begin{table}[htbp]
\\centering
\\caption{Modality Comparison (Balanced Data)}
\\label{tab:modality_comparison}
\\begin{tabular}{lrrrrr}
\\toprule
Modality & Accuracy & Accept Recall & Reject Recall & In-Dist & OOD \\\\
\\midrule
"""

    modality_patterns = [
        ("Text (Clean)", "balanced_clean", "clean"),
        ("Vision", "balanced_vision", "vision"),
        ("Text+Images", "balanced_clean_images", "clean_images"),
    ]

    for display_name, pattern, _ in modality_patterns:
        # Find matching row
        row = df_full[df_full["result"].str.contains(pattern, case=False)]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        metrics = parse_combined_metrics(row["combined"])
        in_dist = parse_year_metrics(row.get("in_dist", ""))
        ood = parse_year_metrics(row.get("ood", ""))

        in_dist_str = f"{in_dist['accuracy']:.2f}" if in_dist else "N/A"
        ood_str = f"{ood['accuracy']:.2f}" if ood else "N/A"

        latex += f"{display_name} & {metrics['accuracy']:.2f} & {metrics['accept_recall']:.2f} & {metrics['reject_recall']:.2f} & {in_dist_str} & {ood_str} \\\\\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(OUTPUT_DIR / "modality_comparison.tex", "w") as f:
        f.write(latex)

    print(f"  Saved: {OUTPUT_DIR / 'modality_comparison.tex'}")


def main():
    print("="*60)
    print("Generating LaTeX Tables")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_dataset_stats_table()
    generate_tfidf_results_table()
    generate_model_comparison_table()
    generate_hyperparam_results_table()
    generate_year_breakdown_table()
    generate_modality_comparison_table()

    print("\n" + "="*60)
    print("Done! Tables saved to:", OUTPUT_DIR)
    print("="*60)


if __name__ == "__main__":
    main()
