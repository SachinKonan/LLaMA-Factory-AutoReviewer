#!/usr/bin/env python3
"""
Text vs Vision Granular Analysis (v7) — Unified N-Model Script

Config-driven comparison across an extensible model registry.
Current models: SFT Text, SFT Vision, RL Text (Qwen3-4B).

To add a new model: add an entry to MODELS and implement a loader if needed.

Output: results/summarized_investigation/text_vs_vision_v7/
  - modality_analysis/  (13 plots — Parts 1-6 + RL confidence)
  - ensemble_*/         (ensemble analysis — Part 7, text+vision only)
"""

import json
import re
import warnings
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# Configuration — Model Registry
# ============================================================

BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "results" / "summarized_investigation" / "text_vs_vision_v7"

# Each model is a dict with standard fields.  To add a new model, just add an
# entry here and (if the data format is new) implement a loader function.
MODELS = OrderedDict([
    ("sft_text", {
        "label": "SFT Text",
        "color": "#1f77b4",          # Blue
        "loader": "sft",
        "test_data": "data/iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered_test/data.json",
        "predictions": "results/final_sweep_v7_datasweepv3/wd_sweep_2epoch_3epochexp/bz16_lr1e-6_wd0.001_text/finetuned-ckpt-1594.jsonl",
        "train_data": "data/iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered_train/data.json",
        "ookf_years": [2025],
        "wd_sweep": {
            "base": "results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_text",
            "train_steps": [797, 1594, 2391, 3188, 3985, 4782],
            "test_steps": [797, 1594, 2391, 3188, 3985, 4782],
            "best_epoch_idx": 1,     # epoch 2
        },
    }),
    ("sft_vision", {
        "label": "SFT Vision",
        "color": "#ff7f0e",          # Orange
        "loader": "sft",
        "test_data": "data/iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered_test/data.json",
        "predictions": "results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-798.jsonl",
        "train_data": "data/iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered_train/data.json",
        "ookf_years": [2025],
        "wd_sweep": {
            "base": "results/final_sweep_v7_datasweepv3/wd_sweep/bz16_lr1e-6_wd0.001_vision",
            "train_steps": [798, 1596, 2394, 3192, 3990, 4788],
            "test_steps": [798, 1596, 2394, 3192, 3990, 4788],
            "best_epoch_idx": 0,     # epoch 1
        },
    }),
    ("rl_text", {
        "label": "RL Text",
        "color": "#9467bd",          # Purple
        "loader": "rl_majority_vote",
        "predictions": "/scratch/gpfs/ZHUANGL/sk7524/SkyRLSearchEnvs/skyrl-train/results/passat/review_roles_qwen3_4b_20stepRL.jsonl",
        "n_votes": 10,
        "ookf_years": [],            # Qwen3-4B trained on data including 2025
    }),
])

# --- Backward-compatible aliases (used by ensemble analysis, Part 7) ---
TEXT_BEST = {k: MODELS["sft_text"][k] for k in ("test_data", "predictions", "train_data")}
VISION_BEST = {k: MODELS["sft_vision"][k] for k in ("test_data", "predictions", "train_data")}
TEXT_COLOR = MODELS["sft_text"]["color"]
VISION_COLOR = MODELS["sft_vision"]["color"]
WD_SWEEP = {
    "text": MODELS["sft_text"]["wd_sweep"],
    "vision": MODELS["sft_vision"]["wd_sweep"],
}

METADATA_PATH = "data/massive_metadata_v7.csv"

# Validation set predictions (for ensemble meta-learner training)
TEXT_VAL = {
    "val_data": "data/iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered_validation/data.json",
    "predictions": "results/final_sweep_v7_datasweepv3/textvisionevalruns/text/val-ckpt-1594.jsonl",
}
VISION_VAL = {
    "val_data": "data/iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered_validation/data.json",
    "predictions": "results/final_sweep_v7_datasweepv3/textvisionevalruns/vision/val-ckpt-798.jsonl",
}

# Train subset predictions (2k samples from train-ckpt eval)
TEXT_TRAIN_CKPT = {
    "train_data": "data/iclr_2020_2023_2025_85_5_10_balanced_original_text_v7_filtered_train/data.json",
    "train_ckpt": "results/final_sweep_v7_datasweepv3/wd_sweep_2epoch_3epochexp/bz16_lr1e-6_wd0.001_text/train-ckpt-1594.json",
}
VISION_TRAIN_CKPT = {
    "train_data": "data/iclr_2020_2023_2025_85_5_10_balanced_original_vision_v7_filtered_train/data.json",
    "train_ckpt": "results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.001_vision/train-ckpt-798.json",
}

# Year classification
IKF_YEARS = [2020, 2023]   # In Knowledge Frontier (training years for SFT)
OOKF_YEARS = [2025]        # Out Of Knowledge Frontier (for SFT; NOT OOD for RL)

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

# Ensemble colours (Part 7)
MAJORITY_COLOR = "#1abc9c"
COMBO_COLORS = {
    'tfidf+preds': '#e67e22',
    'tfidf+conf': '#9b59b6',
    'meta+tfidf+preds': '#2c3e50',
    'meta+tfidf+conf': '#c0392b',
}


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
    first_sft_key = next(k for k, c in MODELS.items() if c['loader'] == 'sft')
    ref_df = model_dfs[first_sft_key]

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

    plt.suptitle('Rating Interval Analysis', fontsize=15, fontweight='bold', y=1.02)
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
    epochs = list(range(1, 7))

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

        ax.plot(epochs, p_correct_means, '-o', color=cfg['color'], linewidth=2,
                markersize=6, label=cfg['label'])
        best_idx = int(np.argmax(p_correct_means))
        ax.annotate(f'{p_correct_means[best_idx]:.3f}',
                    (epochs[best_idx], p_correct_means[best_idx]),
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

        ax.plot(epochs, accuracies, '-o', color=cfg['color'], linewidth=2,
                markersize=6, label=cfg['label'])
        best_idx = int(np.argmax(accuracies))
        ax.annotate(f'{accuracies[best_idx]:.3f}',
                    (epochs[best_idx], accuracies[best_idx]),
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

        ax.plot(epochs, mean_confidences, '-o', color=cfg['color'], linewidth=2,
                markersize=6, label=cfg['label'])

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Confidence', fontsize=12)
    ax.set_title('Test Set: Mean Confidence Over Epochs', fontsize=13, fontweight='bold')
    ax.set_xticks(epochs)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # --- Subplot 4: Confidence calibration at best epoch ---
    ax = axes[1, 1]
    for key, cfg in sweep_models:
        if key not in calibration_data:
            continue
        confidences, correct_arr = calibration_data[key]
        n_bins = 10
        bin_edges = np.linspace(confidences.min(), confidences.max(), n_bins + 1)
        bin_centers = []
        bin_accuracies = []

        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            else:
                mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
            if mask.sum() >= 5:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_accuracies.append(correct_arr[mask].mean())

        ax.plot(bin_centers, bin_accuracies, '-o', color=cfg['color'], linewidth=2,
                markersize=6, label=cfg['label'])

    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Perfect calibration')
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Confidence Calibration (Best Epoch)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_aspect('equal')

    plt.suptitle('Confidence Over Training', fontsize=16, fontweight='bold', y=1.01)
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
        ('OOKF Accuracy (2025)', 'ookf_accuracy', True),
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
                    # OOKF caveat: mark models where 2025 is not OOD
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

    # Footnote for OOKF caveat
    has_ookf_caveat = any(not cfg.get('ookf_years') for cfg in MODELS.values())
    title = 'Metrics Comparison'
    if has_ookf_caveat:
        title += '\n(* 2025 is not out-of-distribution for this model)'
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
    ax.set_title('Accept/Reject Recall', fontsize=14, fontweight='bold')
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

            # OOKF markers (hollow circle) — only for models where 2025 is OOD
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
# Part 4: Prediction Agreement Venn Diagram + Breakdown
# ============================================================

def plot_agreement_venn(merged_df, output_dir):
    """3-circle Venn diagram + agreement breakdown bars."""
    print("\nPart 4: Prediction Agreement Venn Diagram")

    keys = list(MODELS.keys())
    n = len(merged_df)

    # Compute correctness masks
    correct = {k: merged_df[f'{k}_correct'] for k in keys}

    # --- 3-circle Venn Diagram ---
    # 8 regions
    all_right = correct[keys[0]] & correct[keys[1]] & correct[keys[2]]
    all_wrong = ~correct[keys[0]] & ~correct[keys[1]] & ~correct[keys[2]]
    pair_01 = correct[keys[0]] & correct[keys[1]] & ~correct[keys[2]]
    pair_02 = correct[keys[0]] & ~correct[keys[1]] & correct[keys[2]]
    pair_12 = ~correct[keys[0]] & correct[keys[1]] & correct[keys[2]]
    only_0 = correct[keys[0]] & ~correct[keys[1]] & ~correct[keys[2]]
    only_1 = ~correct[keys[0]] & correct[keys[1]] & ~correct[keys[2]]
    only_2 = ~correct[keys[0]] & ~correct[keys[1]] & correct[keys[2]]

    counts = {
        'all_right': int(all_right.sum()),
        'all_wrong': int(all_wrong.sum()),
        f'pair_{keys[0]}_{keys[1]}': int(pair_01.sum()),
        f'pair_{keys[0]}_{keys[2]}': int(pair_02.sum()),
        f'pair_{keys[1]}_{keys[2]}': int(pair_12.sum()),
        f'only_{keys[0]}': int(only_0.sum()),
        f'only_{keys[1]}': int(only_1.sum()),
        f'only_{keys[2]}': int(only_2.sum()),
    }

    labels = [MODELS[k]['label'] for k in keys]
    colors = [MODELS[k]['color'] for k in keys]

    print(f"  Total papers: {n}")
    for region, cnt in counts.items():
        print(f"    {region}: {cnt} ({cnt / n:.1%})")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Circle positions (triangle arrangement)
    centers = [(0, 0.28), (-0.25, -0.14), (0.25, -0.14)]
    radius = 0.42

    for i, (cx, cy) in enumerate(centers):
        circle = plt.Circle((cx, cy), radius, fill=True,
                             facecolor=colors[i], alpha=0.10,
                             edgecolor=colors[i], linewidth=3)
        ax.add_patch(circle)

    # Circle labels
    label_offsets = [(0, 0.75), (-0.55, -0.45), (0.55, -0.45)]
    for i, (lx, ly) in enumerate(label_offsets):
        ax.text(lx, ly, labels[i].upper(), ha='center', va='center',
                fontsize=14, fontweight='bold', color=colors[i])

    # Region labels — approximate positions
    c = counts
    def _fmt(cnt):
        return f"{cnt}\n({cnt / n:.1%})"

    # All 3 correct (center)
    ax.text(0, 0.02, f"All Right\n{_fmt(c['all_right'])}",
            ha='center', va='center', fontsize=11, fontweight='bold', color=CORRECT_COLOR)

    # Pair overlaps
    ax.text(-0.16, 0.18, f"{labels[0]}+\n{labels[1]}\n{_fmt(c[f'pair_{keys[0]}_{keys[1]}'])}",
            ha='center', va='center', fontsize=8, color='#333333')
    ax.text(0.16, 0.18, f"{labels[0]}+\n{labels[2]}\n{_fmt(c[f'pair_{keys[0]}_{keys[2]}'])}",
            ha='center', va='center', fontsize=8, color='#333333')
    ax.text(0, -0.22, f"{labels[1]}+\n{labels[2]}\n{_fmt(c[f'pair_{keys[1]}_{keys[2]}'])}",
            ha='center', va='center', fontsize=8, color='#333333')

    # Single-only regions
    ax.text(0, 0.48, f"{labels[0]} Only\n{_fmt(c[f'only_{keys[0]}'])}",
            ha='center', va='center', fontsize=10, fontweight='bold', color=colors[0])
    ax.text(-0.42, -0.28, f"{labels[1]} Only\n{_fmt(c[f'only_{keys[1]}'])}",
            ha='center', va='center', fontsize=10, fontweight='bold', color=colors[1])
    ax.text(0.42, -0.28, f"{labels[2]} Only\n{_fmt(c[f'only_{keys[2]}'])}",
            ha='center', va='center', fontsize=10, fontweight='bold', color=colors[2])

    # All wrong (outside)
    ax.text(0.65, 0.6, f"All Wrong\n{_fmt(c['all_wrong'])}",
            ha='center', va='center', fontsize=11, fontweight='bold', color=INCORRECT_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fce4e4', edgecolor=INCORRECT_COLOR))

    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.65, 0.85)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Prediction Agreement on {n} Shared Papers', fontsize=16, fontweight='bold')

    plt.tight_layout()
    path = output_dir / 'prediction_agreement_venn.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

    # --- Agreement Breakdown (bar charts by GT and by year) ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    merged = merged_df.copy()

    categories = [
        'All Right', 'All Wrong',
        f'{labels[0]}+{labels[1]}', f'{labels[0]}+{labels[2]}',
        f'{labels[1]}+{labels[2]}',
        f'{labels[0]} Only', f'{labels[1]} Only', f'{labels[2]} Only',
    ]
    masks = [all_right, all_wrong, pair_01, pair_02, pair_12, only_0, only_1, only_2]
    cat_colors = [
        CORRECT_COLOR, INCORRECT_COLOR,
        '#3498db', '#8e44ad', '#e67e22',
        colors[0], colors[1], colors[2],
    ]

    merged['category'] = 'Unknown'
    for cat, mask in zip(categories, masks):
        merged.loc[mask, 'category'] = cat

    # Subplot 1: By ground truth
    ax = axes[0]
    x = np.arange(len(categories))
    gt_labels_list = ['Accept', 'Reject']
    width = 0.35

    for i, gt in enumerate(gt_labels_list):
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

    # Subplot 2: By year
    ax = axes[1]
    years = sorted(merged['year'].dropna().unique())
    x = np.arange(len(categories))
    width_y = 0.8 / len(years)

    for i, year in enumerate(years):
        year_data = merged[merged['year'] == year]
        cnt = [int((year_data['category'] == cat).sum()) for cat in categories]
        offset = -0.4 + i * width_y + width_y / 2
        is_ookf = int(year) in OOKF_YEARS
        hatch = '//' if is_ookf else ''
        ax.bar(x + offset, cnt, width_y * 0.9,
               label=f'{int(year)}' + (' (OOKF)' if is_ookf else ''),
               edgecolor='black', alpha=0.8, hatch=hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Agreement by Year', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Prediction Agreement Breakdown', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = output_dir / 'agreement_breakdown.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Part 5: Disagreement Analysis (3-way)
# ============================================================

def plot_disagreement_analysis(merged_df, meta_df, output_dir):
    """Analyze what drives disagreements among N models."""
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
    single_edge = disagree[disagree['n_correct'] == 1].copy()

    # Assign edge label
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

            # Mann-Whitney U between first two groups
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

    plt.suptitle('Feature Comparison: Single-Edge Disagreements',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = output_dir / 'disagreement_analysis.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

    # --- Decision Tree Analysis (multi-class: which model is correct) ---
    if HAS_SKLEARN and len(disagree_meta) > 50:
        disagree_meta = disagree_meta.copy()

        # Target: which model is exclusively correct (0, 1, 2)
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

            # Plot tree
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


def plot_factor_analysis_v7(model_dfs, meta_df, output_dir):
    """Factor analysis: correlation with accuracy and R² with predictions.

    N rows (one per model) x 3 columns (factor groups).
    factor_ratio.png is intentionally omitted.
    """
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

    plt.suptitle('Factor Correlation with Model Accuracy (* = p < 0.05)',
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
    plt.suptitle('R² Between Factors and Model Prediction (* = p < 0.05)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = output_dir / 'factor_prediction_r2.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# RL Confidence Distribution
# ============================================================

def plot_rl_confidence_distribution(model_dfs, merged_df, output_dir):
    """RL vote-confidence histogram + accuracy-by-threshold + high-conf Venn + rating dist.

    Only runs for models that have an ``rl_confidence`` column.
    """
    rl_keys = [k for k in MODELS if 'rl_confidence' in model_dfs[k].columns]
    if not rl_keys:
        return

    CONF_THRESHOLD = 0.8  # threshold for the deep-dive panels

    for key in rl_keys:
        cfg = MODELS[key]
        rl_df = model_dfs[key]
        print(f"\n  Generating confidence distribution for {cfg['label']}...")

        # Filter to papers in the N-way merge, bring along other models' correctness
        keys_all = list(MODELS.keys())
        merge_cols = ['title', 'pct_rating', 'ground_truth'] + \
                     [f'{k}_correct' for k in keys_all]
        rl_merged = pd.merge(
            rl_df[['title', 'rl_confidence', 'correct', 'prediction', 'ground_truth']],
            merged_df[merge_cols].drop_duplicates('title'),
            on='title', how='inner', suffixes=('', '_merge'),
        )
        # Use rl_df's ground_truth (drop the duplicate from merged)
        if 'ground_truth_merge' in rl_merged.columns:
            rl_merged = rl_merged.drop(columns=['ground_truth_merge'])

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # --- Top-left: Confidence distribution ---
        ax = axes[0, 0]
        correct_conf = rl_merged[rl_merged['correct']]['rl_confidence']
        wrong_conf = rl_merged[~rl_merged['correct']]['rl_confidence']

        ax.hist(correct_conf, bins=10, alpha=0.7, color=CORRECT_COLOR,
                label=f'Correct (n={len(correct_conf)})', edgecolor='black')
        ax.hist(wrong_conf, bins=10, alpha=0.7, color=INCORRECT_COLOR,
                label=f'Wrong (n={len(wrong_conf)})', edgecolor='black')
        ax.set_xlabel('Vote Confidence (fraction agreeing)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{cfg["label"]} Confidence Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # --- Top-right: Accuracy at different confidence thresholds ---
        ax = axes[0, 1]
        thresholds = np.arange(0.5, 1.05, 0.1)
        accs = []
        ns = []
        for thresh in thresholds:
            subset = rl_merged[rl_merged['rl_confidence'] >= thresh]
            if len(subset) > 0:
                accs.append(subset['correct'].mean())
                ns.append(len(subset))
            else:
                accs.append(np.nan)
                ns.append(0)

        ax.plot(thresholds, accs, '-o', color=cfg['color'], linewidth=2, markersize=8)
        for t, a, n_val in zip(thresholds, accs, ns):
            if not np.isnan(a):
                ax.annotate(f'{a:.1%}\nn={n_val}', (t, a), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=8)

        ax.axvline(x=CONF_THRESHOLD, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Min Vote Confidence Threshold', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{cfg["label"]} Accuracy vs Threshold\n(filtered subset only)',
                     fontsize=13, fontweight='bold')
        ax.set_ylim(0.4, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.3)

        # --- Subset: high-confidence correct RL papers ---
        hi_conf = rl_merged[rl_merged['rl_confidence'] >= CONF_THRESHOLD]
        hi_correct = hi_conf[hi_conf['correct']].copy()
        n_hi = len(hi_conf)
        n_hi_correct = len(hi_correct)
        print(f"    Conf >= {CONF_THRESHOLD}: {n_hi} papers, "
              f"{n_hi_correct} correct ({n_hi_correct/n_hi:.1%})")

        # --- Bottom-left: Venn of model agreement on high-conf correct RL papers ---
        ax = axes[1, 0]
        other_keys = [k for k in keys_all if k != key]

        # Correctness of each model on these papers
        rl_right = n_hi_correct
        m0_right = int(hi_correct[f'{other_keys[0]}_correct'].sum())
        m1_right = int(hi_correct[f'{other_keys[1]}_correct'].sum())
        m0_label = MODELS[other_keys[0]]['label']
        m1_label = MODELS[other_keys[1]]['label']
        m0_color = MODELS[other_keys[0]]['color']
        m1_color = MODELS[other_keys[1]]['color']

        # 3-way breakdown on the hi_correct set (RL is always correct by definition)
        all3 = int((hi_correct[f'{other_keys[0]}_correct'] &
                     hi_correct[f'{other_keys[1]}_correct']).sum())
        rl_and_m0 = int((hi_correct[f'{other_keys[0]}_correct'] &
                          ~hi_correct[f'{other_keys[1]}_correct']).sum())
        rl_and_m1 = int((~hi_correct[f'{other_keys[0]}_correct'] &
                           hi_correct[f'{other_keys[1]}_correct']).sum())
        rl_only = int((~hi_correct[f'{other_keys[0]}_correct'] &
                        ~hi_correct[f'{other_keys[1]}_correct']).sum())

        categories = [f'All 3\nCorrect', f'{cfg["label"]}+\n{m0_label}',
                       f'{cfg["label"]}+\n{m1_label}', f'{cfg["label"]}\nOnly']
        vals = [all3, rl_and_m0, rl_and_m1, rl_only]
        bar_colors = [CORRECT_COLOR, m0_color, m1_color, cfg['color']]

        bars = ax.bar(range(4), vals, color=bar_colors, edgecolor='black', alpha=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    f'{v}\n({v/n_hi_correct:.1%})', ha='center', va='bottom', fontsize=10)

        ax.set_xticks(range(4))
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Who Else Gets These Right?\n'
                     f'({n_hi_correct} papers where {cfg["label"]} conf >= {CONF_THRESHOLD} & correct)',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # --- Bottom-right: Rating distribution of high-conf correct RL papers ---
        ax = axes[1, 1]
        hi_correct_ratings = hi_correct['pct_rating'].dropna()
        all_ratings = rl_merged['pct_rating'].dropna()

        bins_r = np.linspace(0, 1, 11)
        ax.hist(all_ratings, bins=bins_r, alpha=0.3, color='gray', edgecolor='black',
                label=f'All papers (n={len(all_ratings)})', density=True)
        ax.hist(hi_correct_ratings, bins=bins_r, alpha=0.7, color=cfg['color'],
                edgecolor='black',
                label=f'Conf >= {CONF_THRESHOLD} & correct (n={len(hi_correct_ratings)})',
                density=True)

        ax.set_xlabel('Pct Rating (reviewer score percentile)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Rating Distribution: High-Confidence Correct Papers',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Add accept/reject split annotation
        n_accept = int((hi_correct['ground_truth'] == 'Accept').sum())
        n_reject = int((hi_correct['ground_truth'] == 'Reject').sum())
        ax.text(0.98, 0.95, f'Accept: {n_accept}  Reject: {n_reject}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.suptitle(f'{cfg["label"]} Confidence Analysis', fontsize=16,
                     fontweight='bold', y=1.01)
        plt.tight_layout()
        path = output_dir / 'rl_confidence_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {path}")


# ============================================================
# Part 7: Ensemble Analysis (text + vision only)
# ============================================================

def _compute_year_metrics(df, pred_col, gt_col='ground_truth'):
    """Compute per-year accuracy, accept recall, reject recall, pred accept rate."""
    result = {'accuracy': {}, 'accept_recall': {}, 'reject_recall': {}, 'pred_accept_rate': {}}
    for year in sorted(df['year'].dropna().unique()):
        ydf = df[df['year'] == year]
        correct = ydf[pred_col] == ydf[gt_col]
        result['accuracy'][int(year)] = correct.mean()

        ya = ydf[ydf[gt_col] == 'Accept']
        if len(ya) > 0:
            result['accept_recall'][int(year)] = (ya[pred_col] == 'Accept').mean()

        yr = ydf[ydf[gt_col] == 'Reject']
        if len(yr) > 0:
            result['reject_recall'][int(year)] = (yr[pred_col] == 'Reject').mean()

        result['pred_accept_rate'][int(year)] = (ydf[pred_col] == 'Accept').mean()

    return result


def _load_paper_texts(data_json_path, submission_ids=None):
    """Extract paper text from data.json, keyed by submission_id."""
    with open(data_json_path, 'r') as f:
        data = json.load(f)

    texts = {}
    for item in data:
        sid = item.get('_metadata', {}).get('submission_id')
        if sid is None:
            continue
        if submission_ids is not None and sid not in submission_ids:
            continue
        convs = item.get('conversations', [])
        if len(convs) >= 2:
            texts[sid] = convs[1].get('value', '')
        else:
            texts[sid] = ''
    return texts


def _load_val_predictions(val_config, modality_name):
    """Load validation set ground truth + predictions, return DataFrame."""
    val_data_path = BASE_DIR / val_config["val_data"]
    pred_path = BASE_DIR / val_config["predictions"]

    with open(val_data_path, 'r') as f:
        val_data = json.load(f)

    rows = []
    for i, item in enumerate(val_data):
        meta = item.get('_metadata', {})
        rows.append({
            'index': i,
            'submission_id': meta.get('submission_id'),
            'year': meta.get('year'),
            'ground_truth': normalize_label(meta.get('answer')),
            'pct_rating': meta.get('pct_rating'),
        })
    meta_df = pd.DataFrame(rows)

    preds = []
    with open(pred_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            pred = normalize_label(extract_boxed_answer(data.get('predict', '')))
            logprobs = data.get('token_logprobs', [])
            preds.append({
                'index': i,
                'prediction': pred,
                'token_logprobs': logprobs,
            })
    pred_df = pd.DataFrame(preds)

    df = pd.merge(meta_df, pred_df, on='index')

    if len(df) > 0 and len(df.iloc[0]['token_logprobs']) > 0:
        all_logprobs = np.array(df['token_logprobs'].tolist())
        dec_idx = find_decision_token_idx(all_logprobs)
        df['confidence'] = np.exp(all_logprobs[:, dec_idx])
    else:
        df['confidence'] = np.nan

    df = df.drop(columns=['token_logprobs'])
    print(f"  {modality_name} val: {len(df)} samples, "
          f"accuracy: {(df['prediction'] == df['ground_truth']).mean():.4f}")
    return df


def _load_train_ckpt_predictions(ckpt_config, modality_name):
    """Load predictions from train-ckpt JSON (eval_sft_predictions on 2k train samples)."""
    data_path = BASE_DIR / ckpt_config["train_data"]
    ckpt_path = BASE_DIR / ckpt_config["train_ckpt"]

    with open(data_path, 'r') as f:
        train_data = json.load(f)

    with open(ckpt_path, 'r') as f:
        ckpt = json.load(f)

    preds_list = ckpt['eval_sft_predictions']
    rows = []
    for idx, p_accept, p_reject, gt_binary in preds_list:
        idx = int(idx)
        meta = train_data[idx].get('_metadata', {})
        predicted = 'Accept' if p_accept > p_reject else 'Reject'
        rows.append({
            'submission_id': meta.get('submission_id'),
            'year': meta.get('year'),
            'ground_truth': normalize_label(meta.get('answer')),
            'prediction': predicted,
            'confidence': max(p_accept, p_reject),
            'pct_rating': meta.get('pct_rating'),
        })

    df = pd.DataFrame(rows)
    acc = (df['prediction'] == df['ground_truth']).mean()
    print(f"  {modality_name} train-ckpt: {len(df)} samples, accuracy: {acc:.4f}")
    return df


def _save_ensemble_predictions(df, method_name, pred_col, output_dir):
    """Save per-sample predictions as JSONL."""
    records = []
    for _, row in df.iterrows():
        records.append({
            'submission_id': row['submission_id'],
            'predict': row[pred_col],
            'label': row['ground_truth'],
            'text_predict': row['text_pred'],
            'vision_predict': row['vision_pred'],
            'ensemble_predict': row[pred_col],
            'ensemble_confidence': float(row.get(f'{pred_col}_prob', 0.0)),
        })

    safe_name = method_name.lower().replace(' ', '-').replace(':', '').replace('_', '-')
    path = output_dir / f'{safe_name}.jsonl'
    with open(path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    print(f"  Saved predictions: {path} ({len(records)} rows)")
    return path


def _prepare_ensemble_features(df):
    """Add binary/confidence features to a merged text+vision DataFrame."""
    df['text_pred_bin'] = (df['text_pred'] == 'Accept').astype(int)
    df['vision_pred_bin'] = (df['vision_pred'] == 'Accept').astype(int)
    df['gt_binary'] = (df['ground_truth'] == 'Accept').astype(int)
    df['text_p_accept'] = np.where(
        df['text_pred'] == 'Accept', df['text_confidence'], 1 - df['text_confidence'])
    df['vision_p_accept'] = np.where(
        df['vision_pred'] == 'Accept', df['vision_confidence'], 1 - df['vision_confidence'])
    df['text_correct'] = (df['text_pred'] == df['ground_truth']).astype(int)
    df['vision_correct'] = (df['vision_pred'] == df['ground_truth']).astype(int)
    return df


def _run_ensemble_variant(variant_tag, train_df, test_df,
                          tfidf_train, tfidf_test, tfidf_feature_names,
                          model_configs, output_dir, from_sklearn):
    """Train modality-selector ensemble: predict which model to trust."""
    from scipy.sparse import csr_matrix, hstack as sparse_hstack

    DecisionTreeClassifier = from_sklearn['DecisionTreeClassifier']
    RandomForestClassifier = from_sklearn['RandomForestClassifier']
    plot_tree = from_sklearn['plot_tree']
    accuracy_score = from_sklearn['accuracy_score']

    pred_feat_cols = ['text_pred_bin', 'vision_pred_bin']
    conf_feat_cols = ['text_p_accept', 'vision_p_accept']
    meta_feat_cols = [c for c in [
        'num_pages', 'num_figures', 'num_authors', 'num_text_tokens',
        'num_equations', 'number_of_cited_references', 'number_of_bib_items',
        'num_vision_tokens', 'removed_before_intro_count', 'removed_after_refs_pages',
        'removed_reproducibility_count', 'removed_acknowledgments_count',
        'removed_aside_text_count',
    ] if c in train_df.columns]

    variant_dir = output_dir / f"ensemble_{variant_tag}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = variant_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    trees_dir = variant_dir / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)

    feature_combos = {
        'tfidf+preds': (pred_feat_cols, []),
        'tfidf+conf': (conf_feat_cols, []),
        'meta+tfidf+preds': (pred_feat_cols, meta_feat_cols),
        'meta+tfidf+conf': (conf_feat_cols, meta_feat_cols),
    }

    test_df = test_df.copy()
    test_df['text_only_pred'] = test_df['text_pred']
    test_df['vision_only_pred'] = test_df['vision_pred']
    test_df['majority_pred'] = np.where(
        test_df['text_pred'] == test_df['vision_pred'],
        test_df['text_pred'],
        test_df['text_pred'],
    )

    y_train = train_df['text_correct'].values

    test_either_correct = (test_df['text_correct'] | test_df['vision_correct']).astype(int)
    oracle_acc = test_either_correct.mean()
    print(f"    Oracle ceiling (either modality correct): {oracle_acc:.3f}")

    trained = {}
    for combo_name, (model_cols, extra_meta_cols) in feature_combos.items():
        all_dense_cols = model_cols + extra_meta_cols
        feature_names = list(tfidf_feature_names) + all_dense_cols

        dense_train = csr_matrix(train_df[all_dense_cols].fillna(0).values)
        dense_test = csr_matrix(test_df[all_dense_cols].fillna(0).values)
        X_train = sparse_hstack([tfidf_train, dense_train]).tocsr()
        X_test = sparse_hstack([tfidf_test, dense_test]).tocsr()

        model_types = {
            'DT': DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=42),
            'RF': RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42),
        }

        for mt_name, model in model_types.items():
            mname = f'{mt_name}-{combo_name}'
            model.fit(X_train, y_train)

            train_selector_acc = accuracy_score(y_train, model.predict(X_train))
            test_selector_preds = model.predict(X_test)
            test_selector_probs = model.predict_proba(X_test)[:, 1]
            test_selector_acc = accuracy_score(
                test_df['text_correct'].values, test_selector_preds)

            pred_col = f'ens_{mname}_pred'
            test_df[pred_col] = np.where(
                test_selector_preds == 1,
                test_df['text_pred'],
                test_df['vision_pred'],
            )
            test_df[f'{pred_col}_prob'] = test_selector_probs

            final_acc = (test_df[pred_col] == test_df['ground_truth']).mean()
            pct_text = test_selector_preds.mean()

            trained[mname] = {
                'model': model, 'feature_names': feature_names,
                'pred_col': pred_col,
                'selector_train_acc': train_selector_acc,
                'selector_test_acc': test_selector_acc,
                'final_acc': final_acc,
                'pct_text_selected': pct_text,
                'combo_name': combo_name, 'model_type': mt_name,
            }
            print(f"    {mname}: selector acc={test_selector_acc:.3f}, "
                  f"final acc={final_acc:.3f}, text%={pct_text:.1%}")

    methods = {
        'Text Only': {'pred_col': 'text_only_pred', 'color': TEXT_COLOR,
                      'marker': 'x', 'linestyle': '-'},
        'Vision Only': {'pred_col': 'vision_only_pred', 'color': VISION_COLOR,
                        'marker': '^', 'linestyle': '-'},
        'Majority Vote': {'pred_col': 'majority_pred', 'color': MAJORITY_COLOR,
                          'marker': 'v', 'linestyle': '-'},
    }
    for mname, mcfg in trained.items():
        methods[mname] = {
            'pred_col': mcfg['pred_col'],
            'color': model_configs[mname]['color'],
            'marker': model_configs[mname]['marker'],
            'linestyle': model_configs[mname]['linestyle'],
        }

    all_method_metrics = {}
    for mname, mcfg in methods.items():
        pc = mcfg['pred_col']
        correct = test_df[pc] == test_df['ground_truth']
        acc = correct.mean()

        ikf = test_df[test_df['year'].isin(IKF_YEARS)]
        ookf = test_df[test_df['year'].isin(OOKF_YEARS)]
        ikf_acc = (ikf[pc] == ikf['ground_truth']).mean() if len(ikf) > 0 else np.nan
        ookf_acc = (ookf[pc] == ookf['ground_truth']).mean() if len(ookf) > 0 else np.nan

        gt_acc = test_df[test_df['ground_truth'] == 'Accept']
        gt_rej = test_df[test_df['ground_truth'] == 'Reject']
        acc_recall = (gt_acc[pc] == 'Accept').mean() if len(gt_acc) > 0 else np.nan
        rej_recall = (gt_rej[pc] == 'Reject').mean() if len(gt_rej) > 0 else np.nan

        all_method_metrics[mname] = {
            'accuracy': acc, 'ikf_accuracy': ikf_acc, 'ookf_accuracy': ookf_acc,
            'accept_recall': acc_recall, 'reject_recall': rej_recall,
        }

    for mname, mcfg in trained.items():
        _save_ensemble_predictions(test_df, f'{variant_tag}-{mname}',
                                   mcfg['pred_col'], predictions_dir)

    rows_spec = [
        ('Train Size', None, False),
        ('Test Size', None, False),
        ('Overall Accuracy', 'accuracy', True),
        ('IKF Accuracy', 'ikf_accuracy', True),
        ('OOKF Accuracy', 'ookf_accuracy', True),
        ('Accept Recall', 'accept_recall', True),
        ('Reject Recall', 'reject_recall', True),
    ]
    variant_label = 'val' if variant_tag == 'val' else 'val + 2k train'

    def _build_table_data(method_subset):
        tdata = {}
        for mn in method_subset:
            col = []
            for label, key, is_pct in rows_spec:
                if key is None:
                    col.append(f"{len(train_df):,}" if 'Train' in label else f"{len(test_df):,}")
                else:
                    val = all_method_metrics[mn].get(key, np.nan)
                    col.append(f"{val:.1%}" if not (isinstance(val, float) and np.isnan(val)) else "N/A")
            tdata[mn] = col
        return pd.DataFrame(tdata, index=[r[0] for r in rows_spec])

    def _render_table(table_df_out, method_subset, title_suffix, save_stem):
        fig, ax = plt.subplots(figsize=(max(14, 2 * len(method_subset)), 5))
        ax.axis('tight')
        ax.axis('off')
        col_colors = [method_subset[m]['color'] for m in method_subset]
        col_colors_light = []
        for c in col_colors:
            rv, gv, bv = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
            col_colors_light.append(
                f'#{min(255, rv+100):02x}{min(255, gv+100):02x}{min(255, bv+100):02x}')
        tbl = ax.table(
            cellText=table_df_out.values, rowLabels=table_df_out.index,
            colLabels=table_df_out.columns, cellLoc='center', loc='center',
            colColours=col_colors_light, rowColours=['#f0f0f0'] * len(table_df_out),
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.2, 1.5)
        for ri in range(len(rows_spec)):
            if rows_spec[ri][1] is not None:
                vals = [all_method_metrics[mn].get(rows_spec[ri][1], 0)
                        if not np.isnan(all_method_metrics[mn].get(rows_spec[ri][1], 0)) else 0
                        for mn in method_subset]
                best_col = int(np.argmax(vals))
                cell = tbl[ri + 1, best_col]
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#d5f5e3')
        plt.title(f'Modality-Selector Ensemble [{variant_tag}] {title_suffix}\n'
                  f'(Train={len(train_df)} [{variant_label}], Test={len(test_df)}, '
                  f'Oracle={oracle_acc:.1%})',
                  fontsize=13, fontweight='bold', pad=20)
        plt.tight_layout()
        png_path = variant_dir / f'{save_stem}.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {png_path}")

    baselines = {k: methods[k] for k in ['Text Only', 'Vision Only', 'Majority Vote']}
    for mt in ['DT', 'RF']:
        mt_methods = dict(baselines)
        for k, v in methods.items():
            if k.startswith(mt + '-'):
                mt_methods[k] = v
        tdf = _build_table_data(mt_methods)
        csv_path = variant_dir / f'metrics_table_{mt.lower()}.csv'
        tdf.to_csv(csv_path)
        print(f"    Saved: {csv_path}")
        _render_table(tdf, mt_methods, f'{mt} Models', f'metrics_table_{mt.lower()}')

    all_tdf = _build_table_data(methods)
    csv_path = variant_dir / 'metrics_table.csv'
    all_tdf.to_csv(csv_path)
    print(f"    Saved: {csv_path}")

    # Accuracy bar chart
    fig, ax = plt.subplots(figsize=(18, 6))
    method_names = list(methods.keys())
    x = np.arange(len(method_names))
    accs = [all_method_metrics[m]['accuracy'] for m in method_names]
    colors_list = [methods[m]['color'] for m in method_names]
    bars = ax.bar(x, accs, color=colors_list, edgecolor='black', alpha=0.85)
    for bar, acc_val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{acc_val:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.axhline(y=oracle_acc, color='gray', linestyle=':', alpha=0.6, linewidth=1.5,
               label=f'Oracle ({oracle_acc:.1%})')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=7, rotation=35, ha='right')
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Modality-Selector Ensemble [{variant_tag}]\n'
                 f'(Train={len(train_df)} [{variant_label}], Test={len(test_df)})',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.5, max(max(accs), oracle_acc) + 0.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=accs[0], color=TEXT_COLOR, linestyle='--', alpha=0.4, linewidth=1)
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    path = variant_dir / 'accuracy_bars.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path}")

    # Accuracy by year (1x4)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    all_years = sorted(test_df['year'].dropna().unique().astype(int))

    metric_cfgs = [
        ('accuracy', 'Accuracy', 'Accuracy by Year'),
        ('accept_recall', 'Accept Recall', 'Accept Recall by Year'),
        ('reject_recall', 'Reject Recall', 'Reject Recall by Year'),
        ('pred_accept_rate', 'Pred. Accept Rate', 'Pred. Accept Rate by Year'),
    ]

    for ax, (mkey, ylabel, title) in zip(axes, metric_cfgs):
        for mname, mcfg in methods.items():
            year_metrics = _compute_year_metrics(test_df, mcfg['pred_col'])
            years = sorted(year_metrics[mkey].keys())
            vals = [year_metrics[mkey][y] for y in years]
            if not years:
                continue
            ax.plot(years, vals, color=mcfg['color'], linewidth=2,
                    marker=mcfg['marker'], markersize=7,
                    linestyle=mcfg['linestyle'], label=mname)
            for y, v in zip(years, vals):
                if y in OOKF_YEARS:
                    ax.scatter([y], [v], marker=mcfg['marker'], s=100,
                               facecolors='white', edgecolors=mcfg['color'],
                               linewidths=2, zorder=6)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(all_years)
        ax.set_ylim(0.2, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(methods), 6),
               fontsize=6, bbox_to_anchor=(0.5, 1.12))
    plt.tight_layout()
    fig.subplots_adjust(top=0.80)
    path = variant_dir / 'accuracy_by_year.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path}")

    # Decision tree visualizations
    combo_names = list(feature_combos.keys())
    for mname, mcfg in trained.items():
        if mcfg['model_type'] != 'DT':
            continue
        fig, ax = plt.subplots(figsize=(28, 14))
        plot_tree(mcfg['model'], feature_names=mcfg['feature_names'],
                  class_names=['Trust Vision', 'Trust Text'],
                  filled=True, rounded=True, fontsize=6, ax=ax,
                  max_depth=3)
        ax.set_title(f'{mname} [{variant_tag}] — Modality Selector\n'
                     f'(Selector acc: {mcfg["selector_test_acc"]:.1%}, '
                     f'Final acc: {mcfg["final_acc"]:.1%}, '
                     f'Text selected: {mcfg["pct_text_selected"]:.0%})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        safe_name = mname.lower().replace('+', '_plus_')
        path = trees_dir / f'{safe_name}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {path}")

    # Feature importance (2x4)
    fig, axes = plt.subplots(2, 4, figsize=(32, 14))
    model_type_order = ['DT', 'RF']

    for row_idx, mt in enumerate(model_type_order):
        for col_idx, combo in enumerate(combo_names):
            ax = axes[row_idx, col_idx]
            mname = f'{mt}-{combo}'
            if mname not in trained:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
                continue

            mcfg = trained[mname]
            importances = mcfg['model'].feature_importances_
            feat_names = mcfg['feature_names']

            top_k = min(20, len(importances))
            top_idx = np.argsort(importances)[-top_k:]

            color = model_configs[mname]['color']
            ax.barh(range(len(top_idx)), importances[top_idx],
                    color=color, edgecolor='black', alpha=0.8)
            ax.set_yticks(range(len(top_idx)))
            ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=7)
            ax.set_xlabel('Importance', fontsize=9)
            ax.set_title(f'{mname}\n(Sel: {mcfg["selector_test_acc"]:.1%}, '
                         f'Final: {mcfg["final_acc"]:.1%})',
                         fontsize=9, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

    plt.suptitle(f'Feature Importance for Modality Selection (Top 20) [{variant_tag}]',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = variant_dir / 'feature_importance.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path}")

    return all_method_metrics


def plot_ensemble_analysis(ensemble_df, meta_df, output_dir):
    """Part 7: Comprehensive ensemble with TF-IDF + Metadata + Model Outputs.

    Uses only SFT Text + SFT Vision (the two models with epoch-wise confidence).
    """
    print("\nPart 7: Comprehensive Ensemble Analysis")

    try:
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("  Warning: sklearn not available, skipping ensemble analysis")
        return

    from_sklearn = {
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'plot_tree': plot_tree,
        'accuracy_score': accuracy_score,
    }

    text_val_pred_path = BASE_DIR / TEXT_VAL["predictions"]
    vision_val_pred_path = BASE_DIR / VISION_VAL["predictions"]
    if not text_val_pred_path.exists() or not vision_val_pred_path.exists():
        print(f"  Warning: Validation predictions not found, skipping ensemble analysis")
        print(f"    Text val:   {text_val_pred_path} (exists={text_val_pred_path.exists()})")
        print(f"    Vision val: {vision_val_pred_path} (exists={vision_val_pred_path.exists()})")
        print(f"    Run: sbatch sbatch/final_sweep_v7/datasweep_v3/tree_text_and_vision/val_inference.sbatch")
        return

    print("\n  Loading validation predictions...")
    text_val_df = _load_val_predictions(TEXT_VAL, "Text")
    vision_val_df = _load_val_predictions(VISION_VAL, "Vision")

    val_df = pd.merge(
        text_val_df[['submission_id', 'year', 'ground_truth', 'prediction', 'confidence', 'pct_rating']],
        vision_val_df[['submission_id', 'prediction', 'confidence']],
        on='submission_id', suffixes=('_text', '_vision'), how='inner',
    )
    val_df = val_df.rename(columns={
        'prediction_text': 'text_pred', 'prediction_vision': 'vision_pred',
        'confidence_text': 'text_confidence', 'confidence_vision': 'vision_confidence',
    })
    print(f"  Val set: {len(val_df)} papers (text+vision inner join on submission_id)")

    text_train_ckpt_path = BASE_DIR / TEXT_TRAIN_CKPT["train_ckpt"]
    vision_train_ckpt_path = BASE_DIR / VISION_TRAIN_CKPT["train_ckpt"]
    have_train_ckpt = text_train_ckpt_path.exists() and vision_train_ckpt_path.exists()

    train2k_df = None
    if have_train_ckpt:
        print("\n  Loading train-ckpt predictions (2k train samples)...")
        text_train_df = _load_train_ckpt_predictions(TEXT_TRAIN_CKPT, "Text")
        vision_train_df = _load_train_ckpt_predictions(VISION_TRAIN_CKPT, "Vision")

        train2k_df = pd.merge(
            text_train_df[['submission_id', 'year', 'ground_truth', 'prediction', 'confidence', 'pct_rating']],
            vision_train_df[['submission_id', 'prediction', 'confidence']],
            on='submission_id', suffixes=('_text', '_vision'), how='inner',
        )
        train2k_df = train2k_df.rename(columns={
            'prediction_text': 'text_pred', 'prediction_vision': 'vision_pred',
            'confidence_text': 'text_confidence', 'confidence_vision': 'vision_confidence',
        })
        print(f"  Train2k set: {len(train2k_df)} papers (text+vision inner join on submission_id)")
    else:
        print("\n  Train-ckpt files not found, skipping val+train2k variant")
        print(f"    Text:   {text_train_ckpt_path}")
        print(f"    Vision: {vision_train_ckpt_path}")

    test_df = ensemble_df.copy()
    print(f"  Test set: {len(test_df)} papers (cached test predictions)")

    all_sids = set(val_df['submission_id']) | set(test_df['submission_id'])
    if train2k_df is not None:
        all_sids |= set(train2k_df['submission_id'])
    meta_subset = load_massive_metadata(submission_ids=all_sids)

    val_df = pd.merge(val_df, meta_subset, on='submission_id', how='inner').copy()
    test_df = pd.merge(test_df, meta_subset, on='submission_id', how='inner').copy()
    if train2k_df is not None:
        train2k_df = pd.merge(train2k_df, meta_subset, on='submission_id', how='inner').copy()
    print(f"  After metadata join — val: {len(val_df)}, test: {len(test_df)}"
          + (f", train2k: {len(train2k_df)}" if train2k_df is not None else ""))

    _prepare_ensemble_features(val_df)
    _prepare_ensemble_features(test_df)
    if train2k_df is not None:
        _prepare_ensemble_features(train2k_df)

    print("\n  Loading paper texts for TF-IDF...")
    all_text_sids = set(val_df['submission_id']) | set(test_df['submission_id'])
    if train2k_df is not None:
        all_text_sids |= set(train2k_df['submission_id'])

    val_texts = _load_paper_texts(BASE_DIR / TEXT_VAL["val_data"], submission_ids=all_text_sids)
    test_texts = _load_paper_texts(BASE_DIR / TEXT_BEST["test_data"], submission_ids=all_text_sids)
    train_texts = _load_paper_texts(BASE_DIR / TEXT_TRAIN_CKPT["train_data"], submission_ids=all_text_sids)

    paper_texts = {}
    paper_texts.update(train_texts)
    paper_texts.update(val_texts)
    paper_texts.update(test_texts)
    print(f"  Loaded texts for {len(paper_texts)} papers")

    model_configs = {}
    for combo_name, color in COMBO_COLORS.items():
        model_configs[f'DT-{combo_name}'] = {'color': color, 'marker': 'o', 'linestyle': '-'}
        model_configs[f'RF-{combo_name}'] = {'color': color, 'marker': 's', 'linestyle': '--'}

    # Variant A: val only
    print(f"\n  --- Variant A: val only (n={len(val_df)}) ---")

    val_train_texts_ordered = [paper_texts.get(sid, '') for sid in val_df['submission_id']]
    test_texts_ordered = [paper_texts.get(sid, '') for sid in test_df['submission_id']]

    vectorizer_val = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_val_train = vectorizer_val.fit_transform(val_train_texts_ordered)
    tfidf_val_test = vectorizer_val.transform(test_texts_ordered)
    tfidf_feature_names_val = vectorizer_val.get_feature_names_out().tolist()
    print(f"    TF-IDF: {tfidf_val_train.shape[1]} features")

    _run_ensemble_variant('val', val_df, test_df,
                          tfidf_val_train, tfidf_val_test, tfidf_feature_names_val,
                          model_configs, output_dir, from_sklearn)

    # Variant B: val + train2k
    if train2k_df is not None:
        test_sids_set = set(test_df['submission_id'])
        train2k_clean = train2k_df[~train2k_df['submission_id'].isin(test_sids_set)].copy()
        val_plus_train2k = pd.concat([val_df, train2k_clean], ignore_index=True)
        val_plus_train2k = val_plus_train2k.drop_duplicates(subset='submission_id', keep='first')
        print(f"\n  --- Variant B: val+train2k (n={len(val_plus_train2k)}, "
              f"after dedup & removing {len(train2k_df) - len(train2k_clean)} test overlaps) ---")

        vt_train_texts_ordered = [paper_texts.get(sid, '') for sid in val_plus_train2k['submission_id']]
        test_texts_ordered_vt = [paper_texts.get(sid, '') for sid in test_df['submission_id']]

        vectorizer_vt = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf_vt_train = vectorizer_vt.fit_transform(vt_train_texts_ordered)
        tfidf_vt_test = vectorizer_vt.transform(test_texts_ordered_vt)
        tfidf_feature_names_vt = vectorizer_vt.get_feature_names_out().tolist()
        print(f"    TF-IDF: {tfidf_vt_train.shape[1]} features")

        _run_ensemble_variant('val+train2k', val_plus_train2k, test_df,
                              tfidf_vt_train, tfidf_vt_test, tfidf_feature_names_vt,
                              model_configs, output_dir, from_sklearn)


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

    # Part 2: Confidence Over Time (only models with wd_sweep)
    plot_confidence_over_time(modality_dir)

    # Part 3: Modality Investigation
    plot_modality_metrics(model_dfs, data['train_sizes'], modality_dir)

    # Part 4: Agreement Venn Diagram + Breakdown
    plot_agreement_venn(merged_df, modality_dir)

    # Part 5: Disagreement Analysis
    plot_disagreement_analysis(merged_df, meta_df, modality_dir)

    # Part 6: Factor Analysis (no factor_ratio)
    plot_factor_analysis_v7(model_dfs, meta_df, modality_dir)

    # RL Confidence Distribution
    plot_rl_confidence_distribution(model_dfs, merged_df, modality_dir)

    # Part 7: Comprehensive Ensemble (text+vision only, manages its own subdirs)
    # Build the 2-model merged df expected by ensemble code
    sft_text_df = model_dfs['sft_text']
    sft_vision_df = model_dfs['sft_vision']
    ensemble_df = pd.merge(
        sft_text_df[['submission_id', 'year', 'ground_truth', 'prediction',
                      'correct', 'confidence', 'pct_rating']],
        sft_vision_df[['submission_id', 'prediction', 'correct', 'confidence']],
        on='submission_id', suffixes=('_text', '_vision'), how='inner',
    )
    ensemble_df = ensemble_df.rename(columns={
        'prediction_text': 'text_pred', 'correct_text': 'text_correct',
        'prediction_vision': 'vision_pred', 'correct_vision': 'vision_correct',
        'confidence_text': 'text_confidence', 'confidence_vision': 'vision_confidence',
    })
    plot_ensemble_analysis(ensemble_df, meta_df, OUTPUT_DIR)

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
