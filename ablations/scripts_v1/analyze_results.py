#!/usr/bin/env python3
"""
Analyze ablation inference results and generate CSV summaries.

Metrics computed:
- Dataset name, size
- Overall accuracy
- Accept recall, Reject recall
- Accuracy by year (2020-2025)

Outputs:
- analysis_summary.csv: Full results for each dataset
- analysis_summary_intersection.csv: Results computed on intersection of paper IDs

Gemini Processing:
python3 ablations/scripts_v1/analyze_results.py \
    --results_dir ablations/results_v1/gemini_2.5_flash \
    --gemini

python3 ablations/scripts_v1/analyze_results.py \
    --results_dir ablations/results_v1/

"""

import json
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional


def extract_decision(text: str) -> str | None:
    """Extract Accept/Reject decision from model output or label."""
    if not text:
        return None

    text = text.strip()

    # Try boxed format first
    boxed_match = re.search(r'\\boxed\{(Accept|Reject)\}', text, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).capitalize()

    # Try JSON format
    json_match = re.search(r'"decision"\s*:\s*"(Accept|Reject)"', text, re.IGNORECASE)
    if json_match:
        return json_match.group(1).capitalize()

    # Simple keyword match (last resort)
    text_lower = text.lower()
    if 'accept' in text_lower and 'reject' not in text_lower:
        return 'Accept'
    if 'reject' in text_lower and 'accept' not in text_lower:
        return 'Reject'

    # Check if the entire text is just Accept or Reject
    if text_lower.strip() in ['accept', 'reject']:
        return text.strip().capitalize()

    return None


def load_metadata_mapping(base_dataset_path: Path) -> list[dict]:
    """Load metadata from base dataset."""
    data_file = base_dataset_path / "data.json"
    if not data_file.exists():
        return []

    with open(data_file) as f:
        data = json.load(f)

    return [item.get("_metadata", {}) for item in data]


def load_gemini_metadata(subdir_path: Path) -> Tuple[list[dict], str]:
    """Find and load the latest gemini_metadata file in a subdirectory.

    Returns:
        Tuple of (metadata_list, timestamp)
    """
    metadata_files = list(subdir_path.glob("gemini_metadata_*.json"))
    if not metadata_files:
        return [], ""

    # Sort by timestamp in filename (newest first)
    metadata_files.sort(key=lambda x: x.name, reverse=True)
    latest_meta = metadata_files[0]

    # Extract timestamp from filename
    match = re.search(r"gemini_metadata_(.*)\.json", latest_meta.name)
    timestamp = match.group(1) if match else ""

    with open(latest_meta) as f:
        return json.load(f), timestamp


def load_gemini_predictions(subdir_path: Path) -> List[Tuple[str, dict, dict]]:
    """Load Gemini predictions from output.json or predictions_*.jsonl and pair with metadata.

    Supports two formats:
    1. Old format: output.json with concatenated JSON objects containing "response" field
    2. New format: predictions_*.jsonl with standard {prompt, predict, label} format
    """
    # Try new format first (predictions_*.jsonl)
    pred_files = list(subdir_path.glob("predictions_*.jsonl"))
    # Filter out raw files
    pred_files = [f for f in pred_files if "_raw" not in f.name]

    if pred_files:
        # Sort by timestamp (newest first) and use latest
        pred_files.sort(key=lambda x: x.name, reverse=True)
        pred_file = pred_files[0]

        # Load metadata for paper IDs and labels
        metadata_list, _ = load_gemini_metadata(subdir_path)

        # Load predictions from JSONL
        predictions = []
        with open(pred_file) as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))

        pair_results = []
        for i, pred in enumerate(predictions):
            if i < len(metadata_list):
                meta = metadata_list[i]
                paper_id = meta.get("submission_id", f"idx_{i}")
            else:
                paper_id = f"idx_{i}"
                meta = {}

            # The new format already has predict and label fields
            pred_dict = {
                "predict": pred.get("predict", ""),
                "label": pred.get("label", "") or meta.get("label", ""),
            }
            pair_results.append((paper_id, pred_dict, meta))

        return pair_results

    # Fall back to old format (output.json)
    output_file = subdir_path / "output.json"
    if not output_file.exists():
        return []

    metadata_list, _ = load_gemini_metadata(subdir_path)
    if not metadata_list:
        return []

    # Parse concatenated JSON objects from output.json
    results = []
    with open(output_file) as f:
        content = f.read()

    decoder = json.JSONDecoder()
    pos = 0
    raw_results = []
    while pos < len(content):
        # Skip leading whitespace
        match = re.match(r'\s*', content[pos:])
        if match:
            pos += match.end()

        if pos >= len(content):
            break

        try:
            obj, next_pos = decoder.raw_decode(content, pos)
            raw_results.append(obj)
            pos = next_pos
        except json.JSONDecodeError:
            # Try to find next bit that looks like {
            next_start = content.find('{', pos)
            if next_start == -1:
                break
            pos = next_start

    # Gemini output.json structure usually has "response" key
    # or it might be directly the result if retrieved differently.
    # Based on research, it's a list of objects with "response" field.
    pair_results = []
    for i, raw in enumerate(raw_results):
        if i < len(metadata_list):
            meta = metadata_list[i]
            paper_id = meta.get("submission_id", f"idx_{i}")

            # Extract prediction from response field
            response = raw.get("response", {})
            # parse_prediction logic equivalent from infer_gemini.py
            predictions = []
            try:
                candidates = response.get("candidates", [])
                for candidate in candidates:
                    content_obj = candidate.get("content", {})
                    parts = content_obj.get("parts", [])
                    text = ""
                    for part in parts:
                        if "text" in part:
                            text += part.get("text", "")
                    predictions.append(text)
            except (KeyError, IndexError, TypeError):
                pass

            pred_dict = {
                "predict": predictions[0] if predictions else "",
                "label": meta.get("label", ""),
                "all_predictions": predictions if len(predictions) > 1 else None
            }
            pair_results.append((paper_id, pred_dict, meta))

    return pair_results


def load_predictions_with_ids(pred_file: Path, metadata_list: list[dict]) -> List[Tuple[str, dict, dict]]:
    """Load predictions and pair with metadata, returning (paper_id, prediction, metadata) tuples."""
    if not pred_file.exists():
        return []

    predictions = []
    with open(pred_file) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    results = []
    for i, pred in enumerate(predictions):
        if i < len(metadata_list):
            meta = metadata_list[i]
            paper_id = meta.get("submission_id", f"idx_{i}")
        else:
            paper_id = f"idx_{i}"
            meta = {}

        results.append((paper_id, pred, meta))

    return results


def compute_metrics_for_subset(predictions: List[Tuple[str, dict, dict]],
                               paper_ids: Optional[Set[str]] = None) -> dict:
    """Compute metrics for a subset of predictions filtered by paper IDs.

    Args:
        predictions: List of (paper_id, prediction_dict, metadata_dict)
        paper_ids: Optional set of paper IDs to include. If None, use all.

    Returns:
        Dictionary of computed metrics
    """
    results = {
        "dataset_size": 0,
        "true_acceptance_rate": 0.0,
        "predicted_acceptance_rate": 0.0,
        "accuracy": 0.0,
        "accept_recall": 0.0,
        "reject_recall": 0.0,
    }

    years = [2020, 2021, 2022, 2023, 2024, 2025]
    for year in years:
        results[f"metrics_{year}"] = ""

    # Filter by paper IDs if provided
    if paper_ids is not None:
        predictions = [(pid, pred, meta) for pid, pred, meta in predictions if pid in paper_ids]

    results["dataset_size"] = len(predictions)

    if not predictions:
        return results

    # Count stats
    total_correct = 0
    total = 0

    true_accepts = 0
    true_rejects = 0
    correct_accepts = 0
    correct_rejects = 0
    predicted_accepts = 0

    # Per-year stats
    year_correct = defaultdict(int)
    year_total = defaultdict(int)
    year_true_accepts = defaultdict(int)
    year_true_rejects = defaultdict(int)
    year_correct_accepts = defaultdict(int)
    year_correct_rejects = defaultdict(int)

    for paper_id, pred, meta in predictions:
        # Get ground truth from label
        label = pred.get("label", "")
        ground_truth = extract_decision(label)

        # Get prediction - handle both string and list
        pred_text = pred.get("predict", "")
        if isinstance(pred_text, list):
            pred_text = pred_text[0] if pred_text else ""
        predicted = extract_decision(pred_text)

        if ground_truth is None:
            continue

        total += 1
        year = meta.get("year")

        # Track predicted accepts
        if predicted == "Accept":
            predicted_accepts += 1

        # Track recalls
        if ground_truth == "Accept":
            true_accepts += 1
            if year:
                year_true_accepts[year] += 1
            if predicted == "Accept":
                correct_accepts += 1
                if year:
                    year_correct_accepts[year] += 1
        elif ground_truth == "Reject":
            true_rejects += 1
            if year:
                year_true_rejects[year] += 1
            if predicted == "Reject":
                correct_rejects += 1
                if year:
                    year_correct_rejects[year] += 1

        # Track accuracy
        if predicted == ground_truth:
            total_correct += 1
            if year:
                year_correct[year] += 1

        if year:
            year_total[year] += 1

    # Compute metrics
    if total > 0:
        results["true_acceptance_rate"] = round(true_accepts / total * 100, 2)
        results["predicted_acceptance_rate"] = round(predicted_accepts / total * 100, 2)
        results["accuracy"] = round(total_correct / total * 100, 2)
        
        # Raw decimal values for gsheets
        results["predicted_acceptance_rate_dec"] = round(predicted_accepts / total, 3)
        results["accuracy_dec"] = round(total_correct / total, 3)

    if true_accepts > 0:
        results["accept_recall"] = round(correct_accepts / true_accepts * 100, 2)
        results["accept_recall_dec"] = round(correct_accepts / true_accepts, 3)

    if true_rejects > 0:
        results["reject_recall"] = round(correct_rejects / true_rejects * 100, 2)
        results["reject_recall_dec"] = round(correct_rejects / true_rejects, 3)

    # Per-year metrics
    for year in years:
        if year_total[year] > 0:
            y_acc_val = year_correct[year] / year_total[year]
            y_acc = round(y_acc_val * 100, 1)
            y_acc_recall = round(year_correct_accepts[year] / year_true_accepts[year] * 100, 1) if year_true_accepts[year] > 0 else 0.0
            y_rej_recall = round(year_correct_rejects[year] / year_true_rejects[year] * 100, 1) if year_true_rejects[year] > 0 else 0.0
            results[f"metrics_{year}"] = f"{y_acc}/{y_acc_recall}/{y_rej_recall}(n={year_total[year]})"
            results[f"acc_dec_{year}"] = round(y_acc_val, 3)
        else:
            results[f"acc_dec_{year}"] = ""

    return results


def analyze_predictions(pred_file: Path, metadata_list: list[dict]) -> dict:
    """Analyze a single predictions file (legacy interface)."""
    predictions = load_predictions_with_ids(pred_file, metadata_list)
    results = compute_metrics_for_subset(predictions)
    results["dataset_name"] = pred_file.stem.replace("_predictions", "")
    return results


def get_base_dataset_for_ablation(ablation_name: str) -> str:
    """Determine the base dataset path for an ablation dataset."""
    # Map ablation datasets to their base datasets
    if "vision" in ablation_name:
        base = "iclr_2020_2025_85_5_10_split7_ablation_v1_base_vision_test"
    elif "clean_images" in ablation_name:
        base = "iclr_2020_2025_85_5_10_split7_ablation_v1_base_clean_images_test"
    else:
        base = "iclr_2020_2025_85_5_10_split7_ablation_v1_base_clean_test"
    return base


def find_paper_id_intersection(all_predictions: Dict[str, List[Tuple[str, dict, dict]]]) -> Set[str]:
    """Find the intersection of paper IDs across all prediction files."""
    if not all_predictions:
        return set()

    # Get paper IDs from each file
    id_sets = []
    for dataset_name, predictions in all_predictions.items():
        ids = {pid for pid, _, _ in predictions}
        id_sets.append(ids)
        print(f"  {dataset_name}: {len(ids)} papers")

    # Compute intersection
    intersection = id_sets[0]
    for id_set in id_sets[1:]:
        intersection = intersection.intersection(id_set)

    return intersection


def write_csv(results: List[dict], output_file: Path, fieldnames: List[str]):
    """Write results to CSV file."""
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def write_gsheets_summary(results: List[dict], output_file: Path):
    """Write results to CSV formatted for Google Sheets."""
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    headers = [
        "dataset name", "size", "acceptance rate", "results",
    ] + [str(year) for year in years]

    rows = []
    for r in results:
        # Results format: accuracy/accept_recall/reject_recall (all in decimal)
        acc = r.get("accuracy_dec", 0.0)
        a_rec = r.get("accept_recall_dec", 0.0)
        r_rec = r.get("reject_recall_dec", 0.0)
        results_str = f"{acc}/{a_rec}/{r_rec}"

        row = {
            "dataset name": r["dataset_name"],
            "size": r["dataset_size"],
            "acceptance rate": r.get("predicted_acceptance_rate_dec", 0.0),
            "results": results_str,
        }
        for year in years:
            row[str(year)] = r.get(f"acc_dec_{year}", "")
        
        rows.append(row)

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def print_results_table(results: List[dict], title: str = "Results"):
    """Print formatted results table."""
    print("\n" + "=" * 140)
    print(f"{title}")
    print("=" * 140)
    print(f"{'Dataset':<70} {'Size':>6} {'T_AR':>6} {'P_AR':>6} {'Acc':>6} {'Acc+':>6} {'Rej+':>6}")
    print("-" * 140)
    for r in results:
        name = r["dataset_name"]
        # Shorten the name for display
        short_name = name.replace("iclr_2020_2025_85_5_10_split7_", "").replace("_binary_noreviews_v7_test", "").replace("ablation_v1_", "")
        print(f"{short_name:<70} {r['dataset_size']:>6} {r['true_acceptance_rate']:>5.1f}% {r['predicted_acceptance_rate']:>5.1f}% {r['accuracy']:>5.1f}% {r['accept_recall']:>5.1f}% {r['reject_recall']:>5.1f}%")
    print("=" * 140)


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation results.")
    parser.add_argument("--results_dir", type=str, help="Path to results directory")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--gemini", action="store_true", help="Analyze Gemini results in subdirectories")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent.parent
    
    # Default paths
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = project_dir / "ablations" / "results"
        # Also check for results_v1 directory
        results_v1_dir = project_dir / "ablations" / "results_v1"
        if results_v1_dir.exists():
            results_dir = results_v1_dir

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_dir / "data"
    
    base_data_dir = Path("/n/fs/vision-mix/sk7524/LLaMA-Factory/data")

    output_file = results_dir / "analysis_summary.csv"
    intersection_output_file = results_dir / "analysis_summary_intersection.csv"

    all_results = []
    metadata_cache = {}
    all_predictions = {}

    # Master Year Map: Check local first, then remote
    print("\nLoading Master Year Map...")
    master_year_map = {}
    master_base_path = data_dir / "iclr_2020_2025_85_5_10_split7_ablation_v1_base_clean_test"
    if not master_base_path.exists():
        master_base_path = base_data_dir / "iclr_2020_2025_85_5_10_split7_ablation_v1_base_clean_test"
    
    if master_base_path.exists():
        try:
            base_meta = load_metadata_mapping(master_base_path)
            for m in base_meta:
                 if m.get("submission_id"):
                     master_year_map[m["submission_id"]] = m.get("year")
            print(f"Loaded master year map with {len(master_year_map)} entries from {master_base_path}")
        except Exception as e:
            print(f"Warning: Failed to load master year map: {e}")
    else:
        print(f"Warning: Master base dataset not found at {master_base_path}")

    if args.gemini:
        # Gemini results are in subdirectories
        subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
        subdirs.sort()

        for subdir in subdirs:
            print(f"Analyzing Gemini results in: {subdir.name}")
            predictions = load_gemini_predictions(subdir)
            if not predictions:
                print(f"  No predictions found or metadata missing in {subdir.name}, skipping.")
                continue
            
            # INJECT YEARS from master map
            for p in predictions:
                # p is (paper_id, pred_dict, metadata)
                if p[2] and p[0] in master_year_map:
                     p[2]["year"] = master_year_map[p[0]]

            # Store for intersection analysis
            all_predictions[subdir.name] = predictions

            # Compute metrics
            results = compute_metrics_for_subset(predictions)
            results["dataset_name"] = subdir.name
            all_results.append(results)

            print(f"  Size: {results['dataset_size']}, True AR: {results['true_acceptance_rate']}%, "
                  f"Pred AR: {results['predicted_acceptance_rate']}%, Accuracy: {results['accuracy']}%")
    else:
        # Standard processing
        pred_files = sorted(results_dir.glob("*_predictions.jsonl"))
        if not pred_files:
            print("No prediction files found!")
            return

        print(f"Found {len(pred_files)} prediction files")

        for pred_file in pred_files:
            print(f"Analyzing: {pred_file.name}")

            # Determine base dataset and load metadata
            ablation_name = pred_file.stem.replace("_predictions", "")
            base_dataset = get_base_dataset_for_ablation(ablation_name)

            if base_dataset not in metadata_cache:
                # Try local data dir first, then base data dir
                base_path = data_dir / base_dataset
                if not base_path.exists():
                    base_path = base_data_dir / base_dataset
                metadata_cache[base_dataset] = load_metadata_mapping(base_path)

            metadata_list = metadata_cache[base_dataset]

            # Load predictions with IDs
            predictions = load_predictions_with_ids(pred_file, metadata_list)
            
            # INJECT YEARS from master map
            for p in predictions:
                # p is (paper_id, pred_dict, metadata)
                if p[2] and p[0] in master_year_map:
                     p[2]["year"] = master_year_map[p[0]]
            
            all_predictions[ablation_name] = predictions

            # Compute full metrics
            results = compute_metrics_for_subset(predictions)
            results["dataset_name"] = ablation_name
            all_results.append(results)

            # Print summary
            print(f"  Size: {results['dataset_size']}, True AR: {results['true_acceptance_rate']}%, "
                  f"Pred AR: {results['predicted_acceptance_rate']}%, Accuracy: {results['accuracy']}%, "
                  f"Accept Recall: {results['accept_recall']}%, Reject Recall: {results['reject_recall']}%")

    # Define CSV fieldnames
    fieldnames = [
        "dataset_name", "dataset_size", "true_acceptance_rate", "predicted_acceptance_rate",
        "accuracy", "accept_recall", "reject_recall",
        "metrics_2020", "metrics_2021", "metrics_2022",
        "metrics_2023", "metrics_2024", "metrics_2025"
    ]

    # Write full results CSV
    if all_results:
        write_csv(all_results, output_file, fieldnames)
        print(f"\nResults saved to: {output_file}")
        
        # Write GSheets summary
        gsheets_output_file = results_dir / "analysis_summary_gsheets.csv"
        write_gsheets_summary(all_results, gsheets_output_file)
        print(f"GSheets summary saved to: {gsheets_output_file}")
        
        print_results_table(all_results, "Full Results")

    # =========================================================================
    # Intersection Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("Computing Intersection Analysis")
    print("=" * 70)

    # Find paper ID intersection
    print("\nPaper counts per dataset:")
    intersection_ids = find_paper_id_intersection(all_predictions)
    print(f"\nIntersection size: {len(intersection_ids)} papers")

    if len(intersection_ids) > 0:
        # Compute metrics on intersection
        intersection_results = []

        for dataset_name, predictions in all_predictions.items():
            results = compute_metrics_for_subset(predictions, intersection_ids)
            results["dataset_name"] = dataset_name
            intersection_results.append(results)

        # Write intersection CSV
        write_csv(intersection_results, intersection_output_file, fieldnames)
        print(f"\nIntersection results saved to: {intersection_output_file}")
        print_results_table(intersection_results, f"Intersection Results (n={len(intersection_ids)} papers)")
    else:
        print("Warning: No common papers found across all datasets!")


if __name__ == "__main__":
    main()
