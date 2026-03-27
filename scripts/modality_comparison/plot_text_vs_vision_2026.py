#!/usr/bin/env python3
"""
Text vs Vision Granular Analysis (2026) — Unified N-Model Script

Adapted from plot_text_vs_vision_v7.py for the optim_search_2026 experiments.
Best models: bz16_lr1e-6_vision (ckpt-2648) and bz32_lr1e-6_text (ckpt-1322).

RL Text is a placeholder (queued) — set rl_text predictions path when ready.

Output: results/summarized_investigation/text_vs_vision_2026/
  - modality_analysis/  (13 plots — Parts 1-6 + RL confidence)
"""

import json
import re
import warnings
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# Configuration — Model Registry
# ============================================================

BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "summarized_investigation" / "text_vs_vision_2026"

MODELS = OrderedDict([
    ("sft_text", {
        "label": "SFT Text (bz32)",
        "color": "#1f77b4",          # Blue
        "loader": "sft",
        "test_data": "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
        "predictions": "results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/finetuned-ckpt-1322.jsonl",
        "train_data": "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train/data.json",
        "ookf_years": [2025, 2026],
        "wd_sweep": {
            "base": "results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text",
            "train_steps": [661, 1322, 1983, 2644],
            "test_steps": [661, 1322, 1983, 2644],
            "best_epoch_idx": 1,     # epoch 2
        },
    }),
    ("sft_vision", {
        "label": "SFT Vision (bz16)",
        "color": "#ff7f0e",          # Orange
        "loader": "sft",
        "test_data": "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json",
        "predictions": "results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl",
        "train_data": "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train/data.json",
        "ookf_years": [2025, 2026],
        "wd_sweep": {
            "base": "results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision",
            "train_steps": [1324, 2648, 3972, 5296],
            "test_steps": [1324, 2648, 3972, 5296],
            "best_epoch_idx": 1,     # epoch 2
        },
    }),
    # RL Text placeholder — update path when job finishes
    # ("rl_text", {
    #     "label": "RL Text",
    #     "color": "#9467bd",          # Purple
    #     "loader": "rl_majority_vote",
    #     "predictions": "TBD",
    #     "n_votes": 10,
    #     "ookf_years": [],
    # }),
])

# --- Backward-compatible aliases (used by ensemble analysis) ---
TEXT_BEST = {k: MODELS["sft_text"][k] for k in ("test_data", "predictions", "train_data")}
VISION_BEST = {k: MODELS["sft_vision"][k] for k in ("test_data", "predictions", "train_data")}
TEXT_COLOR = MODELS["sft_text"]["color"]
VISION_COLOR = MODELS["sft_vision"]["color"]

METADATA_PATH = "data/massive_metadata_v7.csv"

# Year classification
IKF_YEARS = [2020, 2023]         # In Knowledge Frontier (training years)
OOKF_YEARS = [2025, 2026]       # Out Of Knowledge Frontier

# Plot colours
ACCEPT_COLOR = "#2ecc71"
REJECT_COLOR = "#e74c3c"
CORRECT_COLOR = "#2ecc71"
INCORRECT_COLOR = "#e74c3c"

# Factor definitions
BASE_FACTORS = [
    ('num_authors', 'Authors'), ('num_figures', 'Figures'),
    ('num_pages', 'Pages'), ('num_text_tokens', 'Text Tokens'),
    ('num_text_image_tokens', 'Text+Img Tokens'), ('num_vision_tokens', 'Vision Tokens'),
    ('number_of_cited_references', 'Citations'), ('number_of_bib_items', 'Bib Items'),
    ('num_equations', 'Equations'),
]
METADATA_FACTORS = [
    ('removed_before_intro_count', 'Pre-Intro Removed'),
    ('removed_after_refs_pages', 'Appendix Pages'),
    ('removed_reproducibility_count', 'Has Reproducibility'),
    ('removed_acknowledgments_count', 'Has Acknowledgments'),
    ('removed_aside_text_count', 'Aside Text Removed'),
]
RATING_FACTORS = [
    ('pct_rating', 'Pct Rating'),
    ('pct_citation', 'Pct Citation'),
]
ALL_FACTORS = BASE_FACTORS + METADATA_FACTORS + RATING_FACTORS


# ============================================================
# Utility Functions
# ============================================================

def extract_boxed_answer(text):
    """Parse \\boxed{Accept} or \\boxed{Reject} from model output."""
    if text is None:
        return None
    match = re.search(r'\\boxed\{(\w+)\}', text)
    if match:
        return match.group(1)
    if 'Accept' in text:
        return 'Accept'
    if 'Reject' in text:
        return 'Reject'
    return None


def normalize_label(label):
    """Normalize to 'Accept' or 'Reject'."""
    if label is None:
        return None
    label = label.strip()
    if label.lower() in ('accept', 'accepted', 'yes', 'y'):
        return 'Accept'
    if label.lower() in ('reject', 'rejected', 'no', 'n'):
        return 'Reject'
    return label


def find_decision_token_idx(all_logprobs):
    """Find the token index with highest variance (the Accept/Reject decision token)."""
    variances = np.var(all_logprobs, axis=0)
    return int(np.argmax(variances))


def extract_rl_prediction(predict_list):
    """Extract majority-vote prediction from RL's N predictions."""
    votes = []
    for pred_text in predict_list:
        match = re.search(r'\\boxed\{(\w+)\}', pred_text)
        if match:
            v = normalize_label(match.group(1))
            if v:
                votes.append(v)
                continue
        last_match = list(re.finditer(r'(accept|reject)', pred_text.lower()))
        if last_match:
            v = normalize_label(last_match[-1].group(1).capitalize())
            if v:
                votes.append(v)

    if not votes:
        return None, 0.0

    vote_counts = Counter(votes)
    pred = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[pred] / len(votes)
    return pred, confidence


def extract_title_from_sft(conversations):
    """Extract paper title from SFT conversation format."""
    for conv in conversations:
        if conv['from'] in ('human', 'user'):
            content = conv['value']
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('# ') and len(line) > 5:
                    title = line[2:].strip()
                    if title.upper() in ('ABSTRACT',) or re.match(r'^\d+\s+', title):
                        continue
                    return title.upper()
    return ''


def extract_title_from_rl(prompt):
    """Extract paper title from RL prompt format."""
    idx = prompt.find('<|im_start|>user')
    if idx < 0:
        return ''
    after = prompt[idx:]
    lines = after.split('\n')
    for line in lines[1:15]:
        line = line.strip()
        if line.startswith('# ') and len(line) > 5:
            title = line[2:].strip()
            if title.upper() in ('ABSTRACT',) or re.match(r'^\d+\s+', title):
                continue
            return title.upper()
    return ''


# ============================================================
# Data Loading
# ============================================================

def load_sft_predictions(model_cfg):
    """Load SFT test data and predictions, return DataFrame with standard columns."""
    test_path = BASE_DIR / model_cfg["test_data"]
    pred_path = Path(model_cfg["predictions"])
    if not pred_path.is_absolute():
        pred_path = BASE_DIR / pred_path

    with open(test_path, 'r') as f:
        test_data = json.load(f)

    preds = []
    with open(pred_path, 'r') as f:
        for line in f:
            preds.append(json.loads(line))

    rows = []
    for i, (item, pred) in enumerate(zip(test_data, preds)):
        meta = item.get('_metadata', {})
        title = extract_title_from_sft(item.get('conversations', []))
        pred_label = normalize_label(extract_boxed_answer(pred.get('predict', '')))
        gt_label = normalize_label(meta.get('answer'))

        rows.append({
            'index': i,
            'title': title,
            'submission_id': meta.get('submission_id'),
            'year': meta.get('year'),
            'ground_truth': gt_label,
            'prediction': pred_label,
            'pct_rating': meta.get('pct_rating'),
            'pct_citation': meta.get('citation_normalized_by_year'),
            'token_logprobs': pred.get('token_logprobs', []),
        })

    df = pd.DataFrame(rows)
    df['correct'] = df['prediction'] == df['ground_truth']

    # Compute per-sample confidence from decision token
    if len(df) > 0 and df.iloc[0]['token_logprobs'] and len(df.iloc[0]['token_logprobs']) > 0:
        all_logprobs = np.array(df['token_logprobs'].tolist())
        dec_idx = find_decision_token_idx(all_logprobs)
        df['confidence'] = np.exp(all_logprobs[:, dec_idx])
        print(f"  Decision token index: {dec_idx} "
              f"(variance: {np.var(all_logprobs, axis=0)[dec_idx]:.6f})")
    else:
        df['confidence'] = np.nan

    return df


def load_rl_predictions(model_cfg):
    """Load RL results with majority-vote predictions."""
    pred_path = Path(model_cfg["predictions"])
    if not pred_path.is_absolute():
        pred_path = BASE_DIR / pred_path

    rows = []
    with open(pred_path) as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            title = extract_title_from_rl(obj['prompt'])
            gt_label = normalize_label(extract_boxed_answer(obj['label']))
            pred_label, confidence = extract_rl_prediction(obj['predict'])

            rows.append({
                'index': i,
                'title': title,
                'ground_truth': gt_label,
                'prediction': pred_label,
                'rl_confidence': confidence,
            })

    df = pd.DataFrame(rows)
    df['correct'] = df['prediction'] == df['ground_truth']
    return df


def load_train_ckpt_json(path):
    """Load train checkpoint evaluation JSON."""
    with open(BASE_DIR / path, 'r') as f:
        return json.load(f)


def load_finetuned_ckpt_jsonl(path):
    """Load finetuned checkpoint JSONL, return list of dicts."""
    records = []
    with open(BASE_DIR / path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_massive_metadata(submission_ids=None):
    """Load massive_metadata_v7.csv with relevant columns."""
    usecols = [
        'submission_id', 'num_authors', 'num_figures', 'num_pages',
        'num_text_tokens', 'num_text_image_tokens', 'num_vision_tokens',
        'number_of_cited_references', 'number_of_bib_items', 'num_equations',
        'metadata_of_changes',
    ]

    print("  Loading massive_metadata_v7.csv...")
    df = pd.read_csv(BASE_DIR / METADATA_PATH, usecols=usecols)

    if submission_ids is not None:
        df = df[df['submission_id'].isin(submission_ids)].copy()

    def parse_meta_changes(val):
        if pd.isna(val):
            return {}
        try:
            return json.loads(val)
        except Exception:
            return {}

    changes = df['metadata_of_changes'].apply(parse_meta_changes)
    for col in ['removed_before_intro_count', 'removed_after_refs_pages',
                'removed_reproducibility_count', 'removed_acknowledgments_count',
                'removed_aside_text_count']:
        df[col] = changes.apply(lambda x, c=col: x.get(c, 0))

    df = df.drop(columns=['metadata_of_changes'])
    return df


def load_all_data():
    """Load all models' data and create merged DataFrames."""
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    model_dfs = {}
    for key, cfg in MODELS.items():
        print(f"\nLoading {cfg['label']} predictions...")
        if cfg['loader'] == 'sft':
            model_dfs[key] = load_sft_predictions(cfg)
        elif cfg['loader'] == 'rl_majority_vote':
            model_dfs[key] = load_rl_predictions(cfg)
        print(f"  {cfg['label']}: {len(model_dfs[key])} samples, "
              f"accuracy: {model_dfs[key]['correct'].mean():.4f}")

    # For RL models: add year/pct_rating/submission_id via title match
    sft_keys = [k for k, c in MODELS.items() if c['loader'] == 'sft']
    if sft_keys:
        ref_df = model_dfs[sft_keys[0]]
        for key, cfg in MODELS.items():
            if cfg['loader'] == 'rl_majority_vote':
                rl_df = model_dfs[key]
                ref_meta = ref_df[['title', 'submission_id', 'year', 'pct_rating',
                                   'pct_citation']].drop_duplicates('title')
                rl_with_meta = pd.merge(rl_df, ref_meta, on='title', how='inner')
                model_dfs[key] = rl_with_meta
                print(f"  {cfg['label']} after title-join: {len(rl_with_meta)} samples")

    # Build merged_df: N-way inner join on title
    keys = list(MODELS.keys())
    first_sft_key = sft_keys[0] if sft_keys else keys[0]
    merged = model_dfs[first_sft_key][
        ['title', 'submission_id', 'year', 'ground_truth', 'pct_rating', 'pct_citation']
    ].copy()

    for key in keys:
        df = model_dfs[key]
        cols = ['title', 'prediction', 'correct']
        renames = {'prediction': f'{key}_pred', 'correct': f'{key}_correct'}
        if 'confidence' in df.columns:
            cols.append('confidence')
            renames['confidence'] = f'{key}_confidence'
        if 'rl_confidence' in df.columns:
            cols.append('rl_confidence')
        extra = df[cols].copy().rename(columns=renames)
        merged = pd.merge(merged, extra, on='title', how='inner')

    print(f"\n  N-way merge: {len(merged)} papers")

    # Load massive metadata
    all_sids = set()
    for df in model_dfs.values():
        if 'submission_id' in df.columns:
            all_sids |= set(df['submission_id'].dropna())
    meta_df = load_massive_metadata(submission_ids=all_sids)
    print(f"  Metadata matched: {len(meta_df)}")

    # Training data sizes
    train_sizes = {}
    for key, cfg in MODELS.items():
        if 'train_data' in cfg:
            with open(BASE_DIR / cfg['train_data'], 'r') as f:
                train_sizes[key] = len(json.load(f))

    return {
        'model_dfs': model_dfs,
        'merged_df': merged,
        'meta_df': meta_df,
        'train_sizes': train_sizes,
    }


# ============================================================
# Shared Metrics
# ============================================================

def compute_metrics(df):
    """Compute accuracy metrics for a prediction DataFrame."""
    metrics = {}
    metrics['n_samples'] = len(df)
    metrics['overall_accuracy'] = df['correct'].mean()

    ikf_mask = df['year'].isin(IKF_YEARS) if 'year' in df.columns else pd.Series(False, index=df.index)
    metrics['ikf_accuracy'] = df[ikf_mask]['correct'].mean() if ikf_mask.any() else np.nan

    ookf_mask = df['year'].isin(OOKF_YEARS) if 'year' in df.columns else pd.Series(False, index=df.index)
    metrics['ookf_accuracy'] = df[ookf_mask]['correct'].mean() if ookf_mask.any() else np.nan

    accept_mask = df['ground_truth'] == 'Accept'
    reject_mask = df['ground_truth'] == 'Reject'
    metrics['accept_recall'] = df[accept_mask]['correct'].mean() if accept_mask.any() else np.nan
    metrics['reject_recall'] = df[reject_mask]['correct'].mean() if reject_mask.any() else np.nan

    pred_accepts = (df['prediction'] == 'Accept').sum()
    metrics['pred_accept_rate'] = pred_accepts / len(df) if len(df) > 0 else np.nan

    metrics['accuracy_by_year'] = {}
    metrics['accept_recall_by_year'] = {}
    metrics['reject_recall_by_year'] = {}
    metrics['pred_accept_rate_by_year'] = {}

    if 'year' in df.columns:
        for year in sorted(df['year'].dropna().unique()):
            ydf = df[df['year'] == year]
            metrics['accuracy_by_year'][int(year)] = ydf['correct'].mean()

            ya = ydf[ydf['ground_truth'] == 'Accept']
            if len(ya) > 0:
                metrics['accept_recall_by_year'][int(year)] = ya['correct'].mean()

            yr = ydf[ydf['ground_truth'] == 'Reject']
            if len(yr) > 0:
                metrics['reject_recall_by_year'][int(year)] = yr['correct'].mean()

            pred_accepts_y = (ydf['prediction'] == 'Accept').sum()
            metrics['pred_accept_rate_by_year'][int(year)] = (
                pred_accepts_y / len(ydf) if len(ydf) > 0 else np.nan)

    return metrics


# ============================================================
# Part 1: Human Rating Interval Analysis
# ============================================================

def plot_rating_intervals(model_dfs, output_dir):
    """1×N panels: accuracy by rating interval for each model."""
    print("\nPart 1: Rating Interval Analysis")

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]']

    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (key, cfg) in zip(axes, MODELS.items()):
        df = model_dfs[key].copy()
        df['rating_bin'] = pd.cut(df['pct_rating'], bins=bins, right=False,
                                  labels=bin_labels, include_lowest=True)

        for bin_idx, bl in enumerate(bin_labels):
            bin_data = df[df['rating_bin'] == bl]
            n = len(bin_data)
            if n == 0:
                continue

            acc = bin_data['correct'].mean()
            err = 1 - acc

            ax.bar(bin_idx, acc, color=CORRECT_COLOR, edgecolor='black', alpha=0.8)
            ax.bar(bin_idx, err, bottom=acc, color=INCORRECT_COLOR, edgecolor='black', alpha=0.4)
            ax.text(bin_idx, acc + err + 0.02, f'{acc:.1%}\nn={n}',
                    ha='center', va='bottom', fontsize=8)

        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, fontsize=8, rotation=15)
        ax.set_ylabel('Proportion', fontsize=11)
        ax.set_ylim(0, 1.25)
        ax.set_title(f'{cfg["label"]}: Accuracy by Rating Interval',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Rating Interval Analysis (2026 Experiments)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = output_dir / 'rating_interval_analysis.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Part 2: Confidence Over Time
# ============================================================

def plot_confidence_over_time(output_dir):
    """4-subplot figure: train confidence, test accuracy, test confidence, calibration.

    Only includes models that have a ``wd_sweep`` config entry.
    """
    sweep_models = [(k, cfg) for k, cfg in MODELS.items() if 'wd_sweep' in cfg]
    if not sweep_models:
        print("\nPart 2: Confidence Over Time — skipped (no models with wd_sweep)")
        return

    print("\nPart 2: Confidence Over Time")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    n_epochs = max(len(cfg['wd_sweep']['train_steps']) for _, cfg in sweep_models)
    epochs = list(range(1, n_epochs + 1))

    # --- Subplot 1: Train set mean confidence (p_correct) ---
    ax = axes[0, 0]
    for key, cfg in sweep_models:
        sweep = cfg['wd_sweep']
        p_correct_means = []
        for step in sweep['train_steps']:
            path = f"{sweep['base']}/train-ckpt-{step}.json"
            data = load_train_ckpt_json(path)
            val = data.get('eval_sft_p_correct_mean', data.get('sft_p_correct_mean'))
            p_correct_means.append(val)

        ep = list(range(1, len(p_correct_means) + 1))
        ax.plot(ep, p_correct_means, '-o', color=cfg['color'], linewidth=2,
                markersize=6, label=cfg['label'])
        best_idx = int(np.argmax(p_correct_means))
        ax.annotate(f'{p_correct_means[best_idx]:.3f}',
                    (ep[best_idx], p_correct_means[best_idx]),
                    textcoords="offset points", xytext=(5, 8), fontsize=9)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean P(correct)', fontsize=12)
    ax.set_title('Train Set: Mean Confidence', fontsize=13, fontweight='bold')
    ax.set_xticks(epochs)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    # --- Subplot 2: Test set accuracy over epochs ---
    ax = axes[0, 1]
    cached_records = {}
    for key, cfg in sweep_models:
        sweep = cfg['wd_sweep']
        accuracies = []
        for ep_idx, step in enumerate(sweep['test_steps']):
            path = f"{sweep['base']}/finetuned-ckpt-{step}.jsonl"
            records = load_finetuned_ckpt_jsonl(path)
            cached_records[(key, ep_idx)] = records
            correct = sum(
                1 for r in records
                if normalize_label(extract_boxed_answer(r.get('predict', ''))) ==
                   normalize_label(extract_boxed_answer(r.get('label', '')))
            )
            accuracies.append(correct / len(records))

        ep = list(range(1, len(accuracies) + 1))
        ax.plot(ep, accuracies, '-o', color=cfg['color'], linewidth=2,
                markersize=6, label=cfg['label'])
        best_idx = int(np.argmax(accuracies))
        ax.annotate(f'{accuracies[best_idx]:.3f}',
                    (ep[best_idx], accuracies[best_idx]),
                    textcoords="offset points", xytext=(5, 8), fontsize=9)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Test Set: Accuracy Over Epochs', fontsize=13, fontweight='bold')
    ax.set_xticks(epochs)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    # --- Subplot 3: Test set confidence over epochs ---
    ax = axes[1, 0]
    calibration_data = {}
    for key, cfg in sweep_models:
        sweep = cfg['wd_sweep']
        best_ep_idx = sweep.get('best_epoch_idx', 0)
        mean_confidences = []
        for ep_idx in range(len(sweep['test_steps'])):
            records = cached_records[(key, ep_idx)]
            all_logprobs = np.array([r['token_logprobs'] for r in records])
            dec_idx = find_decision_token_idx(all_logprobs)
            confidences = np.exp(all_logprobs[:, dec_idx])
            mean_confidences.append(float(np.mean(confidences)))

            if ep_idx == best_ep_idx:
                correct_arr = np.array([
                    1 if normalize_label(extract_boxed_answer(r.get('predict', ''))) ==
                         normalize_label(extract_boxed_answer(r.get('label', ''))) else 0
                    for r in records
                ])
                calibration_data[key] = (confidences, correct_arr)

        ep = list(range(1, len(mean_confidences) + 1))
        ax.plot(ep, mean_confidences, '-o', color=cfg['color'], linewidth=2,
                markersize=6, label=cfg['label'])

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Confidence', fontsize=12)
    ax.set_title('Test Set: Mean Confidence Over Epochs', fontsize=13, fontweight='bold')
    ax.set_xticks(epochs)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # --- Subplot 4: Confidence calibration at best epoch ---
    ax = axes[1, 1]
    # Offset annotations slightly per model to avoid overlap
    y_offsets = [8, -14]
    for m_idx, (key, cfg) in enumerate(sweep_models):
        if key not in calibration_data:
            continue
        confidences, correct_arr = calibration_data[key]
        n_bins = 10
        bin_edges = np.linspace(confidences.min(), confidences.max(), n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            else:
                mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
            if mask.sum() >= 5:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_accuracies.append(correct_arr[mask].mean())
                bin_counts.append(int(mask.sum()))

        ax.plot(bin_centers, bin_accuracies, '-o', color=cfg['color'], linewidth=2,
                markersize=6, label=cfg['label'])

        # Annotate each point with sample count
        y_off = y_offsets[m_idx % len(y_offsets)]
        for bc, ba, bn in zip(bin_centers, bin_accuracies, bin_counts):
            ax.annotate(f'n={bn}', (bc, ba), textcoords="offset points",
                        xytext=(0, y_off), ha='center', fontsize=7,
                        color=cfg['color'], alpha=0.85)

    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Perfect calibration')
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Confidence Calibration (Best Epoch)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_aspect('equal')

    plt.suptitle('Confidence Over Training (2026 Experiments)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = output_dir / 'confidence_over_time.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Part 3: Modality Investigation (metrics table, recall bars, accuracy by year)
# ============================================================

def plot_modality_metrics(model_dfs, train_sizes, output_dir):
    """N-model metrics: table, recall bars, accuracy-by-year line plots."""
    print("\nPart 3: Modality Investigation")

    all_metrics = {}
    for key, cfg in MODELS.items():
        all_metrics[key] = compute_metrics(model_dfs[key])
        if key in train_sizes:
            all_metrics[key]['n_train'] = train_sizes[key]

    # --- Metrics Table ---
    rows_spec = [
        ('Training Size', 'n_train', False),
        ('Testing Size', 'n_samples', False),
        ('Overall Accuracy', 'overall_accuracy', True),
        ('IKF Accuracy (2020+2023)', 'ikf_accuracy', True),
        ('OOKF Accuracy (2025+2026)', 'ookf_accuracy', True),
        ('Accept Recall', 'accept_recall', True),
        ('Reject Recall', 'reject_recall', True),
        ('Pred Accept Rate', 'pred_accept_rate', True),
    ]

    table_data = {}
    for key, cfg in MODELS.items():
        col = []
        for label, mkey, is_pct in rows_spec:
            val = all_metrics[key].get(mkey, np.nan)
            if is_pct:
                if isinstance(val, float) and np.isnan(val):
                    cell = "N/A"
                else:
                    cell = f"{val:.1%}"
                    if mkey == 'ookf_accuracy' and not cfg.get('ookf_years'):
                        cell += "*"
            else:
                cell = f"{int(val):,}" if not (isinstance(val, float) and np.isnan(val)) else "N/A"
            col.append(cell)
        table_data[cfg['label']] = col

    table_df = pd.DataFrame(table_data, index=[r[0] for r in rows_spec])

    csv_path = output_dir / 'metrics_table.csv'
    table_df.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    fig, ax = plt.subplots(figsize=(4 + 3 * len(MODELS), 5))
    ax.axis('tight')
    ax.axis('off')
    col_colors = [cfg['color'] + '40' for cfg in MODELS.values()]
    table = ax.table(
        cellText=table_df.values,
        rowLabels=table_df.index,
        colLabels=table_df.columns,
        cellLoc='center', loc='center',
        colColours=col_colors,
        rowColours=['#f0f0f0'] * len(table_df),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.5)

    # Bold the best value in each numeric row
    keys_list = list(MODELS.keys())
    for row_idx, (_, mkey, is_pct) in enumerate(rows_spec):
        if is_pct and mkey not in ('pred_accept_rate',):
            vals = [all_metrics[k].get(mkey, 0) for k in keys_list]
            vals = [0 if (isinstance(v, float) and np.isnan(v)) else v for v in vals]
            best_col = int(np.argmax(vals))
            cell = table[row_idx + 1, best_col]
            cell.set_text_props(fontweight='bold')

    has_ookf_caveat = any(not cfg.get('ookf_years') for cfg in MODELS.values())
    title = 'Metrics Comparison (2026 Experiments)'
    if has_ookf_caveat:
        title += '\n(* 2025+2026 is not out-of-distribution for this model)'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    png_path = output_dir / 'metrics_table.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {png_path}")

    # --- Recall Bars ---
    n = len(MODELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(2)  # Accept Recall, Reject Recall
    width = 0.8 / n

    for i, (key, cfg) in enumerate(MODELS.items()):
        m = all_metrics[key]
        vals = [m['accept_recall'], m['reject_recall']]
        offset = -0.4 + i * width + width / 2
        bars = ax.bar(x + offset, vals, width * 0.9, label=cfg['label'],
                      color=cfg['color'], edgecolor='black', alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.1%}',
                    ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(['Accept Recall', 'Reject Recall'], fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Accept/Reject Recall (2026 Experiments)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = output_dir / 'recall_bars.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

    # --- Accuracy by Year (4 subplots) ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    all_years = sorted(set().union(
        *(m['accuracy_by_year'].keys() for m in all_metrics.values())))

    metric_configs = [
        ('accuracy_by_year', 'Accuracy', 'Accuracy by Year'),
        ('accept_recall_by_year', 'Accept Recall', 'Accept Recall by Year'),
        ('reject_recall_by_year', 'Reject Recall', 'Reject Recall by Year'),
        ('pred_accept_rate_by_year', 'Pred. Accept Rate', 'Pred. Accept Rate by Year'),
    ]

    for ax, (mkey, ylabel, title) in zip(axes, metric_configs):
        for key, cfg in MODELS.items():
            m = all_metrics[key]
            years = []
            values = []
            for y in all_years:
                val = m.get(mkey, {}).get(y)
                if val is not None:
                    years.append(y)
                    values.append(val)
            if not years:
                continue

            ax.plot(years, values, '-', color=cfg['color'], linewidth=2, label=cfg['label'])

            # IKF markers (filled x)
            ikf_y = [y for y in years if y in IKF_YEARS]
            ikf_v = [values[years.index(y)] for y in ikf_y]
            if ikf_y:
                ax.scatter(ikf_y, ikf_v, marker='x', s=80, color=cfg['color'], zorder=5)

            # OOKF markers (hollow circle)
            if cfg.get('ookf_years'):
                ookf_y = [y for y in years if y in cfg['ookf_years']]
                ookf_v = [values[years.index(y)] for y in ookf_y]
                if ookf_y:
                    ax.scatter(ookf_y, ookf_v, marker='o', s=80, color=cfg['color'],
                               zorder=5, facecolors='white', edgecolors=cfg['color'],
                               linewidths=2)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(all_years)
        ax.set_ylim(0.2, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(MODELS),
               fontsize=11, bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

    path = output_dir / 'accuracy_by_year.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Part 4: Prediction Agreement (2-model version)
# ============================================================

def plot_agreement(merged_df, output_dir):
    """2-model agreement analysis: Venn-style + breakdown."""
    print("\nPart 4: Prediction Agreement")

    keys = list(MODELS.keys())
    n = len(merged_df)

    correct = {k: merged_df[f'{k}_correct'] for k in keys}

    if len(keys) == 2:
        # 2-model: 4 regions
        both_right = correct[keys[0]] & correct[keys[1]]
        both_wrong = ~correct[keys[0]] & ~correct[keys[1]]
        only_0 = correct[keys[0]] & ~correct[keys[1]]
        only_1 = ~correct[keys[0]] & correct[keys[1]]

        counts = {
            'Both Right': int(both_right.sum()),
            'Both Wrong': int(both_wrong.sum()),
            f'{MODELS[keys[0]]["label"]} Only': int(only_0.sum()),
            f'{MODELS[keys[1]]["label"]} Only': int(only_1.sum()),
        }

        print(f"  Total papers: {n}")
        for region, cnt in counts.items():
            print(f"    {region}: {cnt} ({cnt / n:.1%})")

        # --- Venn with embedded rating-distribution mini-heatmaps ---
        from matplotlib.colors import to_rgba, LinearSegmentedColormap

        lbl0 = MODELS[keys[0]]['label']
        lbl1 = MODELS[keys[1]]['label']
        m_colors = [MODELS[keys[0]]['color'], MODELS[keys[1]]['color']]

        # Attach pct_rating + year to each mask
        merged_tmp = merged_df.copy()
        region_info = [
            ('Both Right',   both_right,  CORRECT_COLOR,   'Greens'),
            (f'{lbl0} Only', only_0,      m_colors[0],     'Blues'),
            (f'{lbl1} Only', only_1,      m_colors[1],     'Oranges'),
            ('Both Wrong',   both_wrong,  INCORRECT_COLOR, 'Reds'),
        ]

        years = sorted(merged_tmp['year'].dropna().unique())
        years = [int(y) for y in years]
        rating_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        rating_labels = ['0-.2', '.2-.4', '.4-.6', '.6-.8', '.8-1']

        fig, ax = plt.subplots(figsize=(28, 20))

        # Draw Venn circles
        centers = [(-0.20, 0), (0.20, 0)]
        radius = 0.45
        for i, (cx, cy) in enumerate(centers):
            circle = plt.Circle((cx, cy), radius, fill=True,
                                 facecolor=m_colors[i], alpha=0.08,
                                 edgecolor=m_colors[i], linewidth=3)
            ax.add_patch(circle)

        # Circle labels at top
        ax.text(-0.48, 0.50, lbl0.upper(), ha='center', va='center',
                fontsize=18, fontweight='bold', color=m_colors[0])
        ax.text(0.48, 0.50, lbl1.upper(), ha='center', va='center',
                fontsize=18, fontweight='bold', color=m_colors[1])

        # Inset positions: [x, y, width, height] in axes fraction
        # We'll place them at the Venn region centers
        # Convert data coords to axes fraction for inset placement
        def data_to_axes(ax, x, y):
            """Convert data coordinates to axes fraction."""
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            return ((x - xlim[0]) / (xlim[1] - xlim[0]),
                    (y - ylim[0]) / (ylim[1] - ylim[0]))

        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-0.72, 0.62)
        ax.set_aspect('equal')
        ax.axis('off')

        # Inset center positions (data coords) and sizes (axes fraction)
        inset_w, inset_h = 0.22, 0.42
        inset_positions = {
            'Both Right':   (0.0, -0.05),    # center overlap
            f'{lbl0} Only': (-0.42, -0.05),  # left
            f'{lbl1} Only': (0.42, -0.05),   # right
            'Both Wrong':   (0.0, -0.62),    # below
        }

        for reg_name, mask, base_color, cmap_name in region_info:
            subset = merged_tmp[mask].copy()
            cnt_total = len(subset)

            cx, cy = inset_positions[reg_name]
            ax_frac_x, ax_frac_y = data_to_axes(ax, cx, cy)
            # Center the inset on the position
            inset_ax = ax.inset_axes([ax_frac_x - inset_w / 2,
                                       ax_frac_y - inset_h / 2,
                                       inset_w, inset_h])

            # Build the heatmap: rows=years, cols=rating bins
            # Each cell = count of papers in (year, rating_bin) for this region
            heatmap = np.zeros((len(years), len(rating_labels)))
            for yi, year in enumerate(years):
                year_sub = subset[subset['year'] == year]
                if len(year_sub) == 0:
                    continue
                binned = pd.cut(year_sub['pct_rating'], bins=rating_bins,
                                right=False, labels=rating_labels,
                                include_lowest=True)
                for ri, rl in enumerate(rating_labels):
                    heatmap[yi, ri] = int((binned == rl).sum())

            # Normalize each row by total papers in that year (across ALL regions)
            row_norm = np.zeros_like(heatmap)
            for yi, year in enumerate(years):
                year_total = int((merged_tmp['year'] == year).sum())
                if year_total > 0:
                    row_norm[yi, :] = heatmap[yi, :] / year_total

            cmap = plt.get_cmap(cmap_name)
            # Use row_norm for color, show counts as text
            vmax = max(row_norm.max(), 0.01)
            im = inset_ax.imshow(row_norm, aspect='auto', cmap=cmap,
                                  vmin=0, vmax=vmax, interpolation='nearest')

            # Annotate cells with counts
            for yi in range(len(years)):
                for ri in range(len(rating_labels)):
                    val = int(heatmap[yi, ri])
                    if val > 0:
                        text_color = 'white' if row_norm[yi, ri] > vmax * 0.55 else 'black'
                        inset_ax.text(ri, yi, str(val), ha='center', va='center',
                                       fontsize=7, fontweight='bold', color=text_color)

            inset_ax.set_xticks(range(len(rating_labels)))
            inset_ax.set_xticklabels(rating_labels, fontsize=7)
            inset_ax.set_yticks(range(len(years)))
            inset_ax.set_yticklabels([str(y) for y in years], fontsize=7)
            inset_ax.set_xlabel('pct_rating', fontsize=7)

            pct_of_total = cnt_total / n
            inset_ax.set_title(f'{reg_name}\n{cnt_total} ({pct_of_total:.1%})',
                                fontsize=10, fontweight='bold', color=base_color, pad=3)

            # Style the inset border
            for spine in inset_ax.spines.values():
                spine.set_edgecolor(base_color)
                spine.set_linewidth(2)

        ax.set_title(f'Prediction Agreement with Rating Distribution — {n} Papers\n'
                     f'Cell values = paper count; color intensity = fraction of year total',
                     fontsize=18, fontweight='bold', pad=20)

        plt.tight_layout()
        path = output_dir / 'prediction_agreement_venn.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {path}")

        # --- Agreement Breakdown ---
        categories = list(counts.keys())
        masks = [both_right, both_wrong, only_0, only_1]
        model_colors = [MODELS[keys[0]]['color'], MODELS[keys[1]]['color']]
        cat_colors = [CORRECT_COLOR, INCORRECT_COLOR, model_colors[0], model_colors[1]]

    else:
        # 3+ model: use full Venn from v7
        categories, masks, cat_colors = _compute_3way_regions(merged_df, correct, keys, n)

    # Breakdown bar charts
    merged = merged_df.copy()
    merged['category'] = 'Unknown'
    for cat, mask in zip(categories, masks):
        merged.loc[mask, 'category'] = cat

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Subplot 1: By ground truth
    ax = axes[0]
    x = np.arange(len(categories))
    width = 0.35
    for i, gt in enumerate(['Accept', 'Reject']):
        gt_data = merged[merged['ground_truth'] == gt]
        cnt = [int((gt_data['category'] == cat).sum()) for cat in categories]
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, cnt, width, label=gt,
                      color=ACCEPT_COLOR if gt == 'Accept' else REJECT_COLOR,
                      edgecolor='black', alpha=0.8)
        for bar, c_val in zip(bars, cnt):
            if c_val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                        str(c_val), ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Agreement by Ground Truth', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Subplot 2: By year (frequency — proportion within each year)
    ax = axes[1]
    years = sorted(merged['year'].dropna().unique())
    x = np.arange(len(categories))
    width_y = 0.8 / len(years)

    for i, year in enumerate(years):
        year_data = merged[merged['year'] == year]
        year_n = len(year_data)
        freq = [(year_data['category'] == cat).sum() / year_n if year_n > 0 else 0
                for cat in categories]
        offset = -0.4 + i * width_y + width_y / 2
        is_ookf = int(year) in OOKF_YEARS
        hatch = '//' if is_ookf else ''
        ax.bar(x + offset, freq, width_y * 0.9,
               label=f'{int(year)} (n={year_n})' + (' OOKF' if is_ookf else ''),
               edgecolor='black', alpha=0.8, hatch=hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Frequency (within year)', fontsize=12)
    ax.set_title('Agreement by Year (normalized)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Prediction Agreement Breakdown (2026 Experiments)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = output_dir / 'agreement_breakdown.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Part 5: Disagreement Analysis
# ============================================================

def plot_disagreement_analysis(merged_df, meta_df, output_dir):
    """Analyze what drives disagreements among models."""
    print("\nPart 5: Disagreement Analysis")

    try:
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        HAS_SKLEARN = True
    except ImportError:
        print("  Warning: sklearn not available, skipping Decision Tree analysis")
        HAS_SKLEARN = False

    keys = list(MODELS.keys())
    correct_cols = [f'{k}_correct' for k in keys]

    # Disagreement: not all models have the same correctness
    all_same = merged_df[correct_cols].apply(lambda row: row.nunique() == 1, axis=1)
    disagree = merged_df[~all_same].copy()

    # Identify single-edge cases (exactly one model correct)
    disagree['n_correct'] = sum(disagree[c].astype(int) for c in correct_cols)

    if len(keys) == 2:
        # For 2 models, single-edge = exactly one correct
        single_edge = disagree[disagree['n_correct'] == 1].copy()
        for k in keys:
            mask = single_edge[f'{k}_correct']
            single_edge.loc[mask, 'edge_model'] = MODELS[k]['label']
    else:
        single_edge = disagree[disagree['n_correct'] == 1].copy()
        for k in keys:
            mask = single_edge[f'{k}_correct']
            single_edge.loc[mask, 'edge_model'] = MODELS[k]['label']

    edge_labels = [MODELS[k]['label'] for k in keys]
    edge_colors = [MODELS[k]['color'] for k in keys]

    print(f"  Disagreement cases: {len(disagree)}")
    print(f"  Single-edge cases: {len(single_edge)}")
    for k in keys:
        cnt = int(single_edge[f'{k}_correct'].sum()) if len(single_edge) > 0 else 0
        print(f"    {MODELS[k]['label']} only correct: {cnt}")

    # Join with metadata
    disagree_meta = pd.merge(single_edge, meta_df, on='submission_id', how='inner')

    # --- Descriptive Statistics (boxplots) ---
    feature_cols = ['pct_rating', 'num_pages', 'num_figures',
                    'num_text_tokens', 'num_equations', 'num_authors']
    feature_labels_list = ['Pct Rating', 'Pages', 'Figures',
                           'Text Tokens', 'Equations', 'Authors']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (col, flabel) in enumerate(zip(feature_cols, feature_labels_list)):
        ax = axes[idx // 3, idx % 3]

        if col not in disagree_meta.columns:
            ax.text(0.5, 0.5, f'{flabel}\nNot available', transform=ax.transAxes,
                    ha='center', va='center')
            continue

        data = []
        box_labels = []
        for k, label, color in zip(keys, edge_labels, edge_colors):
            vals = disagree_meta[disagree_meta[f'{k}_correct']][col].dropna()
            if len(vals) > 0:
                data.append(vals.values)
                box_labels.append(f'{label}\nOnly')

        if len(data) >= 2:
            bp = ax.boxplot(data, tick_labels=box_labels, patch_artist=True, widths=0.6)
            for i_box, color in enumerate(edge_colors[:len(data)]):
                bp['boxes'][i_box].set_facecolor(color)
                bp['boxes'][i_box].set_alpha(0.5)

            try:
                _, p_val = stats.mannwhitneyu(data[0], data[1], alternative='two-sided')
                sig = '*' if p_val < 0.05 else ''
                ax.set_title(f'{flabel} (p={p_val:.3f}{sig})', fontsize=12, fontweight='bold')
            except Exception:
                ax.set_title(flabel, fontsize=12, fontweight='bold')

            for i_box, d in enumerate(data):
                ax.text(i_box + 1, np.median(d), f'med={np.median(d):.2f}',
                        fontsize=8, va='center')
        else:
            ax.text(0.5, 0.5, f'{flabel}\nInsufficient data', transform=ax.transAxes,
                    ha='center', va='center')

        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Feature Comparison: Single-Edge Disagreements (2026 Experiments)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = output_dir / 'disagreement_analysis.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

    # --- Decision Tree Analysis ---
    if HAS_SKLEARN and len(disagree_meta) > 50:
        disagree_meta = disagree_meta.copy()

        target_map = {MODELS[k]['label']: i for i, k in enumerate(keys)}
        disagree_meta['target'] = disagree_meta['edge_model'].map(target_map)
        disagree_meta = disagree_meta.dropna(subset=['target'])
        disagree_meta['target'] = disagree_meta['target'].astype(int)

        feat_cols = [c for c in ['num_pages', 'num_figures', 'num_authors',
                                  'num_text_tokens', 'num_equations',
                                  'number_of_cited_references', 'number_of_bib_items',
                                  'pct_rating', 'num_vision_tokens']
                     if c in disagree_meta.columns]

        X = disagree_meta[feat_cols].fillna(0)
        y = disagree_meta['target']

        if len(X) > 30 and y.nunique() > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y,
            )

            dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
            dt.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, dt.predict(X_train))
            test_acc = accuracy_score(y_test, dt.predict(X_test))
            print(f"  Decision Tree: train acc={train_acc:.3f}, test acc={test_acc:.3f}")

            class_names = [f'{MODELS[k]["label"]} Edge' for k in keys]

            fig, ax = plt.subplots(figsize=(24, 12))
            plot_tree(dt, feature_names=feat_cols, class_names=class_names,
                      filled=True, rounded=True, fontsize=9, ax=ax)
            ax.set_title(f'Decision Tree: What Predicts Which Model Is Correct?\n'
                         f'(Train acc: {train_acc:.1%}, Test acc: {test_acc:.1%})',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            path = output_dir / 'disagreement_decision_tree.png'
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Saved: {path}")

            # Feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            importances = dt.feature_importances_
            sorted_idx = np.argsort(importances)
            ax.barh(range(len(sorted_idx)), importances[sorted_idx],
                    color=TEXT_COLOR, edgecolor='black')
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([feat_cols[i] for i in sorted_idx], fontsize=11)
            ax.set_xlabel('Feature Importance', fontsize=12)
            ax.set_title('Decision Tree: Feature Importance\n(Which Model Gets Single-Edge Cases Right)',
                         fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            path = output_dir / 'disagreement_feature_importance.png'
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Saved: {path}")


# ============================================================
# Part 6: Factor Analysis
# ============================================================

def compute_factor_correlations(pred_df, meta_df):
    """Compute correlations between factors and model accuracy/predictions."""
    merged = pd.merge(pred_df, meta_df, on='submission_id', how='inner')
    merged['pred_binary'] = (merged['prediction'] == 'Accept').astype(int)
    merged['correct_int'] = merged['correct'].astype(int)
    merged['gt_binary'] = (merged['ground_truth'] == 'Accept').astype(int)

    results = {}
    for factor_col, factor_label in ALL_FACTORS:
        if factor_col not in merged.columns:
            continue

        valid = merged[[factor_col, 'correct_int', 'ground_truth', 'pred_binary', 'gt_binary']].dropna()
        if len(valid) < 10:
            continue

        accepts = valid[valid['ground_truth'] == 'Accept']
        rejects = valid[valid['ground_truth'] == 'Reject']

        try:
            if len(accepts) > 5:
                r_acc, p_acc = stats.pointbiserialr(accepts['correct_int'], accepts[factor_col])
            else:
                r_acc, p_acc = np.nan, np.nan

            if len(rejects) > 5:
                r_rej, p_rej = stats.pointbiserialr(rejects['correct_int'], rejects[factor_col])
            else:
                r_rej, p_rej = np.nan, np.nan

            r_pred, p_pred = stats.pearsonr(valid[factor_col], valid['pred_binary'])
            r_gt, p_gt = stats.pearsonr(valid[factor_col], valid['gt_binary'])

            results[factor_label] = {
                'accept_corr': r_acc, 'accept_p': p_acc,
                'reject_corr': r_rej, 'reject_p': p_rej,
                'pred_r': r_pred, 'pred_r2': r_pred ** 2, 'pred_p': p_pred,
                'gt_r': r_gt, 'gt_r2': r_gt ** 2, 'gt_p': p_gt,
            }
        except Exception as e:
            print(f"  Warning: correlation error for {factor_col}: {e}")

    return results


def plot_factor_analysis(model_dfs, meta_df, output_dir):
    """Factor analysis: correlation with accuracy and R² with predictions."""
    print("\nPart 6: Factor Analysis")

    all_correlations = {}
    for key, cfg in MODELS.items():
        print(f"  Computing correlations for {cfg['label']}...")
        all_correlations[cfg['label']] = compute_factor_correlations(model_dfs[key], meta_df)

    base_labels = [l for _, l in BASE_FACTORS]
    meta_labels = [l for _, l in METADATA_FACTORS]
    rating_labels = [l for _, l in RATING_FACTORS]
    factor_groups = [
        (base_labels, 'Paper Features'),
        (meta_labels, 'Structural Changes'),
        (rating_labels, 'Rating & Citation'),
    ]

    variants = list(all_correlations.keys())
    n_variants = len(variants)

    # --- Factor Analysis (correlation with correctness) ---
    fig, axes = plt.subplots(n_variants, 3, figsize=(18, 4 * n_variants))
    if n_variants == 1:
        axes = axes[np.newaxis, :]

    for row_idx, variant in enumerate(variants):
        corrs = all_correlations[variant]
        for col_idx, (flabels, col_title) in enumerate(factor_groups):
            ax = axes[row_idx, col_idx]
            factors = [f for f in flabels if f in corrs]
            if not factors:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue

            x = np.arange(len(factors))
            width = 0.35

            ac = [corrs[f]['accept_corr'] for f in factors]
            rc = [corrs[f]['reject_corr'] for f in factors]

            ax.barh(x - width / 2, ac, width, label='Accept GT',
                    color=ACCEPT_COLOR, edgecolor='black')
            ax.barh(x + width / 2, rc, width, label='Reject GT',
                    color=REJECT_COLOR, edgecolor='black')

            for i, f in enumerate(factors):
                if not np.isnan(corrs[f]['accept_p']) and corrs[f]['accept_p'] < 0.05:
                    xp = ac[i] + 0.01 if ac[i] >= 0 else ac[i] - 0.03
                    ax.text(xp, i - width / 2, '*', fontsize=14, va='center', fontweight='bold')
                if not np.isnan(corrs[f]['reject_p']) and corrs[f]['reject_p'] < 0.05:
                    xp = rc[i] + 0.01 if rc[i] >= 0 else rc[i] - 0.03
                    ax.text(xp, i + width / 2, '*', fontsize=14, va='center', fontweight='bold')

            ax.set_yticks(x)
            ax.set_yticklabels(factors, fontsize=11)
            ax.set_xlabel('Correlation with Accuracy', fontsize=11)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_xlim(-0.25, 0.25)
            ax.grid(axis='x', alpha=0.3)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=13, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(variant, fontsize=13, fontweight='bold')
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='lower right', fontsize=10)

    plt.suptitle('Factor Correlation with Model Accuracy (* = p < 0.05)\n(2026 Experiments)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = output_dir / 'factor_analysis.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

    # --- Factor Prediction R² ---
    fig, axes = plt.subplots(n_variants, 3, figsize=(18, 4 * n_variants))
    if n_variants == 1:
        axes = axes[np.newaxis, :]

    for row_idx, variant in enumerate(variants):
        corrs = all_correlations[variant]
        for col_idx, (flabels, col_title) in enumerate(factor_groups):
            ax = axes[row_idx, col_idx]
            factors = [f for f in flabels if f in corrs]
            if not factors:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue

            x = np.arange(len(factors))
            r2_pct = [corrs[f]['pred_r2'] * 100 for f in factors]
            colors_r2 = [ACCEPT_COLOR if corrs[f]['pred_r'] >= 0 else REJECT_COLOR
                         for f in factors]

            ax.barh(x, r2_pct, color=colors_r2, edgecolor='black')

            for i, f in enumerate(factors):
                if corrs[f]['pred_p'] < 0.05:
                    ax.text(r2_pct[i] + 0.2, i, '*', fontsize=12,
                            va='center', fontweight='bold')

            ax.set_yticks(x)
            ax.set_yticklabels(factors, fontsize=11)
            ax.set_xlabel('R² % (Variance Explained)', fontsize=11)
            ax.set_xlim(0, 15)
            ax.grid(axis='x', alpha=0.3)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f'{v:.1f}%'))

            if row_idx == 0:
                ax.set_title(col_title, fontsize=13, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(variant, fontsize=13, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor=ACCEPT_COLOR, edgecolor='black', label='Higher -> Accept'),
        mpatches.Patch(facecolor=REJECT_COLOR, edgecolor='black', label='Higher -> Reject'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11,
               bbox_to_anchor=(0.98, 0.98))
    plt.suptitle('R² Between Factors and Model Prediction (* = p < 0.05)\n(2026 Experiments)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = output_dir / 'factor_prediction_r2.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# RL Confidence Distribution (placeholder — runs when RL data available)
# ============================================================

def plot_rl_confidence_distribution(model_dfs, merged_df, output_dir):
    """RL vote-confidence histogram + accuracy-by-threshold.
    Only runs for models that have an ``rl_confidence`` column."""
    rl_keys = [k for k in MODELS if 'rl_confidence' in model_dfs[k].columns]
    if not rl_keys:
        print("\n  RL Confidence Distribution — skipped (no RL models)")
        return
    # (same as v7 — will be populated when RL job finishes)


# ============================================================
# Main
# ============================================================

def main():
    """Orchestrate all analysis parts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modality_dir = OUTPUT_DIR / "modality_analysis"
    modality_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    data = load_all_data()

    model_dfs = data['model_dfs']
    merged_df = data['merged_df']
    meta_df = data['meta_df']

    # Part 1: Rating Interval Analysis
    plot_rating_intervals(model_dfs, modality_dir)

    # Part 2: Confidence Over Time
    plot_confidence_over_time(modality_dir)

    # Part 3: Modality Investigation
    plot_modality_metrics(model_dfs, data['train_sizes'], modality_dir)

    # Part 4: Agreement
    plot_agreement(merged_df, modality_dir)

    # Part 5: Disagreement Analysis
    plot_disagreement_analysis(merged_df, meta_df, modality_dir)

    # Part 6: Factor Analysis
    plot_factor_analysis(model_dfs, meta_df, modality_dir)

    # RL Confidence Distribution (no-op until RL available)
    plot_rl_confidence_distribution(model_dfs, merged_df, modality_dir)

    # Summary
    print("\n" + "=" * 60)
    print("All visualizations generated!")
    print("=" * 60)
    print(f"\nOutput structure:")
    import os
    for root, dirs, files in os.walk(OUTPUT_DIR):
        level = len(Path(root).relative_to(OUTPUT_DIR).parts)
        indent = "  " * level
        print(f"{indent}{Path(root).name}/")
        for f in sorted(files):
            print(f"{indent}  {f}")


if __name__ == "__main__":
    main()
