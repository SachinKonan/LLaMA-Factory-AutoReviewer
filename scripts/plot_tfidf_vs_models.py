#!/usr/bin/env python3
"""
Compare TF-IDF baseline results with trained model results by year.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re


# Mapping from TF-IDF file to model variants in subset_analysis.csv
# Format: (result_name, source) to match against subset_analysis.csv
TFIDF_TO_MODELS = {
    "iclr_2020_2025_results.csv": {
        "name": "iclr20_balanced",
        "models": [
            # (result_name, source)
            ("iclr20_balanced_clean", "data_sweep_v2"),
            ("iclr20_balanced_vision", "data_sweep_v2"),
            ("balanced_clean_images", "data_sweep_clean_images"),
        ],
        "years": [2020, 2021, 2022, 2023, 2024, 2025],
    },
    "iclr_2020_2025_trainagreeing.csv": {
        "name": "iclr20_trainagreeing",
        "models": [
            ("iclr20_trainagreeing_clean", "data_sweep_v2"),
            ("iclr20_trainagreeing_vision", "data_sweep_v2"),
            ("trainagreeing_clean_images", "data_sweep_clean_images"),
        ],
        "years": [2020, 2021, 2022, 2023, 2024, 2025],
    },
    "iclr_2017_2025.csv": {
        "name": "iclr17_balanced",
        "models": [
            ("iclr17_balanced_clean", "data_sweep_v2_long"),
            ("iclr17_balanced_vision", "data_sweep_v2_long"),
        ],
        "years": [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    },
    "iclr_nips_2020_2025.csv": {
        "name": "iclr_nips_balanced",
        "models": [
            ("iclr_nips_balanced_clean", "data_sweep_v2"),
            ("iclr_nips_balanced_vision", "data_sweep_v2_long"),
        ],
        "years": [2020, 2021, 2022, 2023, 2024, 2025],
    },
    "iclr_nips_2020_2025_nips_accepts.csv": {
        "name": "iclr_nips_accepts",
        "models": [
            ("iclr_nips_accepts_clean", "data_sweep_v2_long"),
        ],
        "years": [2020, 2021, 2022, 2023, 2024, 2025],
    },
}


def parse_metrics(metric_str: str) -> dict:
    """Parse 'acc/accept_recall/reject_recall(n=N)' format."""
    if metric_str == "N/A" or pd.isna(metric_str):
        return None

    # Extract metrics and sample size
    match = re.match(r"([\d.]+)/([\d.]+)/([\d.]+)(?:\(n=(\d+)\))?", metric_str)
    if not match:
        return None

    return {
        "accuracy": float(match.group(1)),
        "accept_recall": float(match.group(2)),
        "reject_recall": float(match.group(3)),
        "n": int(match.group(4)) if match.group(4) else None,
    }


def load_model_results(csv_path: str) -> pd.DataFrame:
    """Load and parse the subset_analysis.csv file."""
    df = pd.read_csv(csv_path)
    return df


def load_tfidf_results(tfidf_dir: str) -> dict:
    """Load all TF-IDF result files."""
    tfidf_dir = Path(tfidf_dir)
    results = {}

    for filename, config in TFIDF_TO_MODELS.items():
        filepath = tfidf_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            # Filter out 'Total' row and convert year to int
            df = df[df["year"] != "Total"].copy()
            df["year"] = df["year"].astype(int)
            results[config["name"]] = {
                "df": df,
                "models": config["models"],
                "years": config["years"],
            }

    return results


def plot_comparison(tfidf_data: dict, model_df: pd.DataFrame, output_dir: str):
    """Create comparison plots for each data variant."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant_name, tfidf_info in tfidf_data.items():
        tfidf_df = tfidf_info["df"]
        model_specs = tfidf_info["models"]  # List of (result_name, source) tuples
        years = tfidf_info["years"]

        # Get model results for this variant
        model_results = {}
        for model_spec in model_specs:
            if isinstance(model_spec, tuple):
                model_name, source = model_spec
            else:
                # Backward compatibility
                model_name = model_spec
                source = None

            # Find rows matching this model and source
            if source:
                rows = model_df[
                    (model_df["result"] == model_name) &
                    (model_df["source"] == source) &
                    (model_df["subset"] == "(full)")
                ]
            else:
                rows = model_df[
                    (model_df["result"] == model_name) &
                    (model_df["subset"] == "(full)")
                ]

            if len(rows) == 0:
                print(f"  No rows found for {model_name} (source={source})")
                continue

            row = rows.iloc[0]
            model_results[model_name] = {}

            for year in years:
                col = f"y{year}"
                if col in row and row[col] != "N/A":
                    parsed = parse_metrics(row[col])
                    if parsed:
                        model_results[model_name][year] = parsed

        if not model_results:
            print(f"No model results found for {variant_name}")
            continue

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{variant_name}: TF-IDF vs Trained Models", fontsize=14, fontweight="bold")

        metrics = [
            ("accuracy", "Accuracy"),
            ("accept_recall", "Accept Recall"),
            ("reject_recall", "Reject Recall"),
        ]

        colors = {
            "tfidf": "#888888",
            "clean": "#2ecc71",
            "vision": "#e74c3c",
            "clean_images": "#3498db",
        }

        for ax, (metric_key, metric_label) in zip(axes, metrics):
            # Plot TF-IDF baseline
            tfidf_years = tfidf_df["year"].tolist()
            tfidf_values = tfidf_df[metric_key].tolist()
            ax.plot(tfidf_years, tfidf_values, "o--", color=colors["tfidf"],
                   label="TF-IDF", linewidth=2, markersize=8)

            # Plot each model variant
            for model_name, year_data in model_results.items():
                if "clean_images" in model_name:
                    color = colors["clean_images"]
                    label = "clean_images"
                elif "vision" in model_name:
                    color = colors["vision"]
                    label = "vision"
                else:
                    color = colors["clean"]
                    label = "clean"

                model_years = sorted(year_data.keys())
                model_values = [year_data[y][metric_key] for y in model_years]
                ax.plot(model_years, model_values, "o-", color=color,
                       label=label, linewidth=2, markersize=8)

            ax.set_xlabel("Year", fontsize=11)
            ax.set_ylabel(metric_label, fontsize=11)
            ax.set_title(metric_label, fontsize=12)
            ax.set_ylim(0.3, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            ax.set_xticks(years)

        plt.tight_layout()

        # Save plot
        output_path = output_dir / f"{variant_name}_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")


def plot_correlation_scatter(tfidf_data: dict, model_df: pd.DataFrame, output_dir: str):
    """Create scatter plot showing correlation between TF-IDF and model accuracy."""
    output_dir = Path(output_dir)

    all_points = []

    for variant_name, tfidf_info in tfidf_data.items():
        tfidf_df = tfidf_info["df"]
        model_specs = tfidf_info["models"]
        years = tfidf_info["years"]

        for model_spec in model_specs:
            if isinstance(model_spec, tuple):
                model_name, source = model_spec
            else:
                model_name = model_spec
                source = None

            if source:
                rows = model_df[
                    (model_df["result"] == model_name) &
                    (model_df["source"] == source) &
                    (model_df["subset"] == "(full)")
                ]
            else:
                rows = model_df[
                    (model_df["result"] == model_name) &
                    (model_df["subset"] == "(full)")
                ]

            if len(rows) == 0:
                continue

            row = rows.iloc[0]

            for year in years:
                col = f"y{year}"
                if col not in row or row[col] == "N/A":
                    continue

                parsed = parse_metrics(row[col])
                if not parsed:
                    continue

                tfidf_row = tfidf_df[tfidf_df["year"] == year]
                if len(tfidf_row) == 0:
                    continue

                tfidf_acc = tfidf_row["accuracy"].values[0]

                modality = "vision" if "vision" in model_name else "clean"
                if "clean_images" in model_name:
                    modality = "clean_images"

                all_points.append({
                    "tfidf_acc": tfidf_acc,
                    "model_acc": parsed["accuracy"],
                    "year": year,
                    "variant": variant_name,
                    "modality": modality,
                })

    if not all_points:
        print("No data points for correlation plot")
        return

    df = pd.DataFrame(all_points)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {"clean": "#2ecc71", "vision": "#e74c3c", "clean_images": "#3498db"}

    for modality, color in colors.items():
        subset = df[df["modality"] == modality]
        if len(subset) > 0:
            ax.scatter(subset["tfidf_acc"], subset["model_acc"],
                      c=color, label=modality, alpha=0.7, s=80, edgecolors="white")

    # Add diagonal line (y=x)
    ax.plot([0.4, 0.8], [0.4, 0.8], "k--", alpha=0.5, label="y=x")

    # Compute correlation
    corr = df["tfidf_acc"].corr(df["model_acc"])

    ax.set_xlabel("TF-IDF Accuracy", fontsize=12)
    ax.set_ylabel("Model Accuracy", fontsize=12)
    ax.set_title(f"TF-IDF vs Model Accuracy by Year\n(Pearson r = {corr:.3f})", fontsize=13)
    ax.set_xlim(0.4, 0.8)
    ax.set_ylim(0.4, 0.85)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    output_path = output_dir / "tfidf_model_correlation.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_improvement_over_tfidf(tfidf_data: dict, model_df: pd.DataFrame, output_dir: str):
    """Plot the improvement of models over TF-IDF baseline by year."""
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    plot_idx = 0

    for variant_name, tfidf_info in tfidf_data.items():
        if variant_name == "iclr_nips_accepts":  # Skip this one
            continue
        if plot_idx >= 4:
            break

        tfidf_df = tfidf_info["df"]
        model_specs = tfidf_info["models"]
        years = tfidf_info["years"]

        ax = axes[plot_idx]

        colors = {"clean": "#2ecc71", "vision": "#e74c3c", "clean_images": "#3498db"}
        offsets = {"clean": -0.25, "clean_images": 0, "vision": 0.25}

        for model_spec in model_specs:
            if isinstance(model_spec, tuple):
                model_name, source = model_spec
            else:
                model_name = model_spec
                source = None

            if source:
                rows = model_df[
                    (model_df["result"] == model_name) &
                    (model_df["source"] == source) &
                    (model_df["subset"] == "(full)")
                ]
            else:
                rows = model_df[
                    (model_df["result"] == model_name) &
                    (model_df["subset"] == "(full)")
                ]

            if len(rows) == 0:
                continue

            row = rows.iloc[0]

            improvements = []
            valid_years = []

            for year in years:
                col = f"y{year}"
                if col not in row or row[col] == "N/A":
                    continue

                parsed = parse_metrics(row[col])
                if not parsed:
                    continue

                tfidf_row = tfidf_df[tfidf_df["year"] == year]
                if len(tfidf_row) == 0:
                    continue

                tfidf_acc = tfidf_row["accuracy"].values[0]
                improvement = (parsed["accuracy"] - tfidf_acc) * 100  # percentage points

                improvements.append(improvement)
                valid_years.append(year)

            if not improvements:
                continue

            modality = "vision" if "vision" in model_name else "clean"
            if "clean_images" in model_name:
                modality = "clean_images"

            offset = offsets.get(modality, 0)
            ax.bar([y + offset for y in valid_years],
                  improvements, width=0.23, color=colors[modality],
                  label=modality, alpha=0.8)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Improvement over TF-IDF (pp)", fontsize=11)
        ax.set_title(variant_name, fontsize=12, fontweight="bold")
        ax.set_xticks(years)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="best")
        ax.set_ylim(-10, 25)

        plot_idx += 1

    plt.suptitle("Model Improvement over TF-IDF Baseline", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "improvement_over_tfidf.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare TF-IDF vs trained models")
    parser.add_argument("--tfidf-dir", type=str,
                       default="results/_tfidf",
                       help="Directory containing TF-IDF results")
    parser.add_argument("--model-csv", type=str,
                       default="results/subset_analysis.csv",
                       help="Path to subset_analysis.csv")
    parser.add_argument("--output-dir", type=str,
                       default="results/_tfidf/plots",
                       help="Output directory for plots")

    args = parser.parse_args()

    print("Loading TF-IDF results...")
    tfidf_data = load_tfidf_results(args.tfidf_dir)
    print(f"Loaded {len(tfidf_data)} TF-IDF result files")

    print("Loading model results...")
    model_df = load_model_results(args.model_csv)
    print(f"Loaded {len(model_df)} model result rows")

    print("\nGenerating comparison plots...")
    plot_comparison(tfidf_data, model_df, args.output_dir)

    print("\nGenerating correlation scatter plot...")
    plot_correlation_scatter(tfidf_data, model_df, args.output_dir)

    print("\nGenerating improvement plot...")
    plot_improvement_over_tfidf(tfidf_data, model_df, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
