"""Add accept_recall and reject_recall columns to OVERALL.csv from metrics CSVs."""
import csv, re, os
from pathlib import Path
from collections import defaultdict

OVERALL_CSV = Path("results/summarized_investigation/modality_v7/OVERALL.csv")
METRICS_DIR = Path("results/final_sweep_v7_datasweepv3")
SAVES_PREFIX = "saves/final_sweep_v7_datasweepv3/"


def load_metrics_csv(path: Path) -> dict:
    """Load metrics CSV and index by (variant, checkpoint, year)."""
    if not path.exists():
        return {}
    index = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                ckpt = int(row["checkpoint"])
            except (ValueError, TypeError):
                continue  # skip "finetuned" or other non-numeric checkpoints
            key = (row["variant"], ckpt, int(row["year"]))
            index[key] = {
                "accuracy": float(row["accuracy"]),
                "accept_recall": float(row["accept_recall"]),
                "reject_recall": float(row["reject_recall"]),
                "n": int(row["n"]),
            }
    return index


def parse_checkpoint_path(ckpt_path: str):
    """Parse checkpoint path to (subdir, variant, checkpoint_num).

    Examples:
        saves/final.../optim_search_2026/bz16_lr1e-6_vision/checkpoint-2648
          → ('optim_search_2026', 'bz16_lr1e-6_vision', 2648)
        saves/final.../optim_search_2026_with_extras/paper_stats/bz16_lr1e-6_vision/checkpoint-2648
          → ('optim_search_2026_with_extras/paper_stats', 'bz16_lr1e-6_vision', 2648)
    """
    if not ckpt_path or not ckpt_path.startswith(SAVES_PREFIX):
        return None, None, None

    # Clean up annotations like " (final merged)"
    rel = re.sub(r'\s*\(.*?\)\s*', '', ckpt_path[len(SAVES_PREFIX):])
    parts = rel.rstrip("/").split("/")

    # Find checkpoint part
    ckpt_idx = None
    for i, p in enumerate(parts):
        if p.startswith("checkpoint-"):
            ckpt_idx = i
            break

    if ckpt_idx is None:
        # Maybe path ends with "(final merged)" or similar
        # Try the last directory as variant
        ckpt_match = re.search(r'checkpoint-(\d+)', ckpt_path)
        if ckpt_match:
            ckpt_num = int(ckpt_match.group(1))
            # Everything before the variant
            variant = parts[-1] if not parts[-1].startswith("checkpoint") else parts[-2]
            subdir = "/".join(parts[:-2]) if len(parts) > 2 else parts[0]
            return subdir, variant, ckpt_num
        return None, None, None

    ckpt_num = int(parts[ckpt_idx].split("-")[1])
    variant = parts[ckpt_idx - 1]
    subdir = "/".join(parts[:ckpt_idx - 1])
    return subdir, variant, ckpt_num


def compute_aggregate(metrics_index, variant, ckpt_num, target_years):
    """Compute weighted aggregate metrics for a set of years."""
    total_correct = 0
    total_accept_correct = 0
    total_reject_correct = 0
    total_n = 0
    total_accept = 0
    total_reject = 0

    for year in target_years:
        key = (variant, ckpt_num, year)
        if key not in metrics_index:
            continue
        m = metrics_index[key]
        n = m["n"]
        # accept_recall and reject_recall are percentages
        # We need to figure out how many accepts and rejects there are
        # accuracy = (accept_correct + reject_correct) / n
        # accept_recall = accept_correct / n_accept * 100
        # reject_recall = reject_correct / n_reject * 100
        # We don't know n_accept/n_reject directly, but:
        # n_accept + n_reject = n
        # acc_recall * n_accept / 100 + rej_recall * n_reject / 100 = accuracy * n / 100
        # So we can derive: we need to find n_accept. Actually let's just use
        # weighted average of per-year recalls.
        total_n += n
        total_correct += m["accuracy"] * n / 100

    if total_n == 0:
        return None, None, None

    # For accept/reject recall, we need to sum up per-year accept/reject counts
    # and correct counts. But we don't have the raw counts, only the recall %.
    # However, we CAN compute aggregate recall correctly if we know per-year
    # accept/reject counts. Since we don't, we'll use the weighted approach.
    # Actually, the metrics CSV doesn't tell us n_accept per year.
    # Let's just compute per-year-aggregated accuracy and recalls using
    # weighted averages by n.
    # This is an approximation; for exact, we'd need per-paper predictions.

    # Better approach: load per-year accuracy/recall and compute weighted average
    acc_vals, rej_vals = [], []
    acc_ns, rej_ns = [], []
    for year in target_years:
        key = (variant, ckpt_num, year)
        if key not in metrics_index:
            continue
        m = metrics_index[key]
        n = m["n"]
        # Approximate: assume balanced (n/2 accept, n/2 reject)
        # In our dataset, per-year it's approximately balanced
        acc_vals.append(m["accept_recall"] * n)
        rej_vals.append(m["reject_recall"] * n)
        acc_ns.append(n)
        rej_ns.append(n)

    total_n_acc = sum(acc_ns)
    total_n_rej = sum(rej_ns)

    agg_acc = round(total_correct / total_n * 100, 1) if total_n > 0 else None
    agg_accr = round(sum(acc_vals) / total_n_acc, 1) if total_n_acc > 0 else None
    agg_rejr = round(sum(rej_vals) / total_n_rej, 1) if total_n_rej > 0 else None

    return agg_acc, agg_accr, agg_rejr


def main():
    # Load all metrics CSVs
    all_metrics = {}
    for csv_file in sorted(METRICS_DIR.rglob("*_metrics.csv")):
        rel = csv_file.relative_to(METRICS_DIR)
        # Convert path to subdir name: strip _metrics.csv suffix
        subdir = str(rel).replace("_metrics.csv", "")
        idx = load_metrics_csv(csv_file)
        if idx:
            all_metrics[subdir] = idx
            print(f"Loaded {len(idx)} entries from {rel}")

    # Load OVERALL.csv
    with open(OVERALL_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Add new columns
    new_cols = [
        "Best 2025 AccR (%)", "Best 2025 RejR (%)",
        "Best 2026 AccR (%)", "Best 2026 RejR (%)",
        "Best 2025+2026 AccR (%)", "Best 2025+2026 RejR (%)",
    ]
    new_fieldnames = list(fieldnames)
    for col in new_cols:
        if col not in new_fieldnames:
            new_fieldnames.append(col)

    updated = 0
    for row in rows:
        ckpt_path = row.get("Best Checkpoint Path", "").strip()
        if not ckpt_path:
            continue

        subdir, variant, ckpt_num = parse_checkpoint_path(ckpt_path)
        if subdir is None:
            print(f"  SKIP (no checkpoint): {row.get('Category')}/{row.get('Model')}")
            continue

        # Find matching metrics index
        metrics_index = all_metrics.get(subdir)
        if metrics_index is None:
            print(f"  SKIP (no metrics CSV for '{subdir}'): {row.get('Category')}/{row.get('Model')}")
            continue

        # Look up per-year data
        found_years = set()
        for (v, c, y) in metrics_index:
            if v == variant and c == ckpt_num:
                found_years.add(y)

        if not found_years:
            print(f"  SKIP (variant/ckpt not found): {variant} ckpt-{ckpt_num} in {subdir}")
            continue

        # 2025 metrics
        for year, prefix in [(2025, "Best 2025"), (2026, "Best 2026")]:
            key = (variant, ckpt_num, year)
            if key in metrics_index:
                m = metrics_index[key]
                row[f"{prefix} AccR (%)"] = round(m["accept_recall"], 1)
                row[f"{prefix} RejR (%)"] = round(m["reject_recall"], 1)

        # 2025+2026 combined (weighted by n)
        years_2526 = [y for y in [2025, 2026] if (variant, ckpt_num, y) in metrics_index]
        if years_2526:
            total_acc_correct = 0
            total_rej_correct = 0
            total_accepts = 0
            total_rejects = 0
            for year in years_2526:
                m = metrics_index[(variant, ckpt_num, year)]
                n = m["n"]
                # Per-year: approximately balanced dataset (n/2 each)
                # But to be more precise, use accept_recall and reject_recall
                # accept_recall = TP / (TP + FN) * 100, reject_recall = TN / (TN + FP) * 100
                # With n_accept = actual accepts, n_reject = actual rejects:
                # TP = accept_recall/100 * n_accept, TN = reject_recall/100 * n_reject
                # n_accept + n_reject = n
                # accuracy = (TP + TN) / n * 100
                # So: accuracy * n / 100 = accept_recall/100 * n_accept + reject_recall/100 * n_reject
                # And: n_accept + n_reject = n
                # Therefore: n_accept = (accuracy * n / 100 - reject_recall/100 * n) / (accept_recall/100 - reject_recall/100)
                # But this can be numerically unstable. Let's use a simpler approach.
                acc_r = m["accept_recall"] / 100
                rej_r = m["reject_recall"] / 100
                acc_pct = m["accuracy"] / 100
                # n_accept * acc_r + n_reject * rej_r = acc_pct * n
                # n_accept + n_reject = n
                # n_accept * acc_r + (n - n_accept) * rej_r = acc_pct * n
                # n_accept * (acc_r - rej_r) = (acc_pct - rej_r) * n
                if abs(acc_r - rej_r) > 1e-10:
                    n_accept = (acc_pct - rej_r) / (acc_r - rej_r) * n
                    n_reject = n - n_accept
                else:
                    n_accept = n / 2
                    n_reject = n / 2
                n_accept = max(0, min(n, n_accept))
                n_reject = n - n_accept

                total_acc_correct += acc_r * n_accept
                total_rej_correct += rej_r * n_reject
                total_accepts += n_accept
                total_rejects += n_reject

            if total_accepts > 0:
                row["Best 2025+2026 AccR (%)"] = round(total_acc_correct / total_accepts * 100, 1)
            if total_rejects > 0:
                row["Best 2025+2026 RejR (%)"] = round(total_rej_correct / total_rejects * 100, 1)

        updated += 1
        print(f"  OK: {row.get('Category')}/{row.get('Model')}/{row.get('Type')} "
              f"→ 2025 AccR={row.get('Best 2025 AccR (%)', 'N/A')}, "
              f"2025 RejR={row.get('Best 2025 RejR (%)', 'N/A')}, "
              f"2026 AccR={row.get('Best 2026 AccR (%)', 'N/A')}, "
              f"2026 RejR={row.get('Best 2026 RejR (%)', 'N/A')}")

    # Write updated CSV
    with open(OVERALL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nUpdated {updated} rows in {OVERALL_CSV}")
    print(f"New columns: {new_cols}")


if __name__ == "__main__":
    main()
