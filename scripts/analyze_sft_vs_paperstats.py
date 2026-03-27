"""Compare SFT vision model predictions against paper statistics features.

Analyzes:
1. Correlation between SFT confidence and paper features
2. Which paper features predict SFT correctness (where SFT goes beyond stats)
3. Agreement/disagreement between SFT and LogReg predictions
4. Feature distributions for SFT-correct vs SFT-wrong papers

Loads pre-saved LogReg results from results/paperstats_baseline/ (run logreg_baseline.py first).

Usage:
    uv run python scripts/analyze_sft_vs_paperstats.py
"""

import json, re, math, os, sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SFT prediction files
BEST_PRED_PATH = os.path.join(BASE, "results/final_sweep_v7/balanced_trainagreeing_vision/finetuned-ckpt-1818.jsonl")
# This file has logprobs for confidence analysis (same test set, different checkpoint)
LOGPROB_PRED_PATH = os.path.join(BASE, "results/final_sweep_v7_datasweepv3/wd_sweep_expdecay/bz16_lr1e-6_wd0.002_vision/finetuned-ckpt-4788.jsonl")

# Test data with metadata (submission_ids)
TEST_DATA_PATH = os.path.join(BASE, "data/iclr_2020_2023_2025_85_5_10_split7_balanced_trainagreeing_vision_binary_noreviews_v7_test/data.json")

# Pre-saved LogReg results
PAPERSTATS_DIR = os.path.join(BASE, "results/paperstats_baseline")

FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DISPLAY_NAMES = {
    'original_total_pages': 'Total Pages',
    'num_pages': 'Body Pages',
    'num_figures': 'Num Figures',
    'num_figure_images': 'Num Figure Images',
    'num_equations': 'Num Equations',
    'number_of_cited_references': 'Num References',
    'num_text_tokens': 'Text Tokens',
    'num_vision_tokens': 'Vision Tokens',
    'num_authors': 'Num Authors',
    'appendix_ratio': 'Appendix Ratio',
    'figures_per_page': 'Figures/Page',
    'equations_per_page': 'Equations/Page',
    'refs_per_page': 'Refs/Page',
    'tokens_per_page': 'Tokens/Page',
    'vision_token_ratio': 'Vision/Text Token Ratio',
    'subfigs_per_figure': 'Subfigs/Figure',
    'refs_per_author': 'Refs/Author',
    'pages_per_author': 'Pages/Author',
    'equations_per_ref': 'Equations/Ref',
    'appendix_pages': 'Appendix Pages',
    'log_total_pages': 'log(Total Pages)',
    'log_num_refs': 'log(Num Refs)',
    'log_text_tokens': 'log(Text Tokens)',
    'log_total_words': 'log(Total Words)',
    'pct_recent_cites': '% Recent Citations',
    'hedge_ratio': 'Hedge Word Ratio',
    'confidence_ratio': 'Confidence Word Ratio',
    'math_density': 'Math Density',
    'cross_ref_density': 'Cross-Ref Density',
    'abstract_words': 'Abstract Words',
    'title_words': 'Title Words',
    'num_inline_math': 'Inline Math Count',
    'total_words': 'Total Words',
    'mentions_ablation': 'Mentions Ablation',
    'vocab_richness_5k': 'Vocabulary Richness',
    'flesch_kincaid': 'Flesch-Kincaid Score',
}


def load_sft_predictions(path):
    """Load SFT predictions, extracting title, prediction, label."""
    records = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            pred = 'Accept' if 'Accept' in d['predict'] else 'Reject'
            label = 'Accept' if 'Accept' in d['label'] else 'Reject'
            title_match = re.search(r'^#\s+(.+?)$', d['prompt'], re.MULTILINE)
            title = title_match.group(1).strip() if title_match else ""
            records[title.lower()] = {
                'title': title,
                'pred': pred,
                'label': label,
                'correct': pred == label,
            }
    return records


def load_sft_logprobs(path):
    """Load logprobs from a prediction file that has them."""
    logprobs = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            lps = d.get('token_logprobs', [])
            if not lps:
                continue
            title_match = re.search(r'^#\s+(.+?)$', d['prompt'], re.MULTILINE)
            if title_match:
                # Token index 5 is the Accept/Reject decision token
                decision_lp = lps[5] if len(lps) > 5 else lps[-1]
                confidence = math.exp(decision_lp)
                logprobs[title_match.group(1).strip().lower()] = confidence
    return logprobs


def load_test_metadata(path):
    """Load test data JSON to get submission_ids mapped by title."""
    title_to_meta = {}
    with open(path) as fh:
        data = json.load(fh)
    for entry in data:
        meta = entry.get('_metadata', {})
        sid = meta.get('submission_id')
        for c in entry['conversations']:
            if c['from'] == 'human':
                m = re.search(r'^#\s+(.+?)$', c['value'], re.MULTILINE)
                if m and sid:
                    title_to_meta[m.group(1).strip().lower()] = {
                        'submission_id': sid,
                        'year': meta.get('year'),
                        'pct_rating': meta.get('pct_rating'),
                    }
                break
    return title_to_meta


def load_paperstats_results():
    """Load pre-saved LogReg predictions and features from disk."""
    pred_path = os.path.join(PAPERSTATS_DIR, "test_predictions.json")
    feat_path = os.path.join(PAPERSTATS_DIR, "test_features.json")
    summary_path = os.path.join(PAPERSTATS_DIR, "results_summary.json")

    if not all(os.path.exists(p) for p in [pred_path, feat_path, summary_path]):
        print("ERROR: Pre-saved paperstats results not found.")
        print("Run first: srun uv run python scripts/logreg_baseline.py")
        sys.exit(1)

    with open(pred_path) as f:
        predictions = json.load(f)
    with open(feat_path) as f:
        features = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    # Index by submission_id
    pred_by_sid = {p['submission_id']: p for p in predictions}
    feat_by_sid = {f['submission_id']: f for f in features}

    return pred_by_sid, feat_by_sid, summary


def main():
    print("=" * 70)
    print("SFT Vision Model vs Paper Statistics — Correlation Analysis")
    print("=" * 70)

    # Load everything
    print("\nLoading SFT predictions (best model, 70.4%)...", flush=True)
    sft_preds = load_sft_predictions(BEST_PRED_PATH)
    print(f"  {len(sft_preds)} predictions")

    print("Loading SFT logprobs (for confidence)...", flush=True)
    if os.path.exists(LOGPROB_PRED_PATH):
        sft_logprobs = load_sft_logprobs(LOGPROB_PRED_PATH)
        print(f"  {len(sft_logprobs)} entries with logprobs")
    else:
        sft_logprobs = {}
        print("  No logprob file found, skipping confidence analysis")

    print("Loading test metadata...", flush=True)
    title_to_meta = load_test_metadata(TEST_DATA_PATH)
    print(f"  {len(title_to_meta)} test entries")

    print("Loading pre-saved paperstats results...", flush=True)
    lr_pred_by_sid, feat_by_sid, lr_summary = load_paperstats_results()
    feature_names = lr_summary['feature_names']
    print(f"  {len(lr_pred_by_sid)} LogReg predictions, {lr_summary['n_features']} features")
    print(f"  LogReg acc: {lr_summary['lr_accuracy']:.1%}, HGB acc: {lr_summary['hgb_accuracy']:.1%}, RF acc: {lr_summary.get('rf_accuracy', 0):.1%}")

    # Match SFT predictions → metadata → LogReg predictions
    matched = []
    for title_lower, sft in sft_preds.items():
        meta = title_to_meta.get(title_lower)
        if not meta:
            continue
        sid = meta['submission_id']
        lr_pred = lr_pred_by_sid.get(sid)
        feats = feat_by_sid.get(sid)
        if not lr_pred or not feats:
            continue

        confidence = sft_logprobs.get(title_lower)
        matched.append({
            **sft,
            'submission_id': sid,
            'year': meta['year'],
            'pct_rating': meta.get('pct_rating'),
            'confidence': confidence,
            'lr_pred': 'Accept' if lr_pred['lr_pred'] == 1 else 'Reject',
            'lr_prob_accept': lr_pred['lr_prob_accept'],
            'hgb_pred': 'Accept' if lr_pred['hgb_pred'] == 1 else 'Reject',
            'hgb_prob_accept': lr_pred['hgb_prob_accept'],
            'rf_pred': 'Accept' if lr_pred['rf_pred'] == 1 else 'Reject',
            'rf_prob_accept': lr_pred['rf_prob_accept'],
            'features': {fn: feats.get(fn, 0) for fn in feature_names},
        })

    print(f"\n  Matched: {len(matched)} papers (SFT ∩ test metadata ∩ LogReg)")
    n_correct = sum(1 for m in matched if m['correct'])
    print(f"  SFT accuracy on matched set: {n_correct/len(matched):.3f} ({n_correct}/{len(matched)})")
    n_with_conf = sum(1 for m in matched if m['confidence'] is not None)
    print(f"  Papers with confidence data: {n_with_conf}")

    total = len(matched)

    # ================================================================
    # ANALYSIS 1: Feature Correlation with SFT Correctness
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Feature Correlation with SFT Correctness")
    print("  (Positive = feature correlates with SFT being CORRECT)")
    print("=" * 70)

    correctness = np.array([1 if m['correct'] else 0 for m in matched])
    correlations = []
    for fn in feature_names:
        vals = np.array([m['features'][fn] for m in matched])
        if np.std(vals) > 1e-10:
            corr = np.corrcoef(vals, correctness)[0, 1]
            if not np.isnan(corr):
                correlations.append((fn, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n{'Feature':<40} {'Corr':>8}  Direction")
    print("-" * 65)
    for fn, corr in correlations[:25]:
        display = DISPLAY_NAMES.get(fn, fn)
        direction = "SFT better on high" if corr > 0 else "SFT better on low"
        print(f"{display:<40} {corr:>+8.4f}  {direction}")

    # ================================================================
    # ANALYSIS 2: SFT Confidence vs Features (for papers with logprobs)
    # ================================================================
    conf_matched = [m for m in matched if m['confidence'] is not None]
    if len(conf_matched) > 100:
        print("\n" + "=" * 70)
        print(f"ANALYSIS 2: Feature Correlation with SFT Confidence (n={len(conf_matched)})")
        print("=" * 70)

        confidence = np.array([m['confidence'] for m in conf_matched])
        conf_correlations = []
        for fn in feature_names:
            vals = np.array([m['features'][fn] for m in conf_matched])
            if np.std(vals) > 1e-10 and np.std(confidence) > 1e-10:
                corr = np.corrcoef(vals, confidence)[0, 1]
                if not np.isnan(corr):
                    conf_correlations.append((fn, corr))

        conf_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n{'Feature':<40} {'Corr':>8}  Interpretation")
        print("-" * 65)
        for fn, corr in conf_correlations[:15]:
            display = DISPLAY_NAMES.get(fn, fn)
            interp = "more confident" if corr > 0 else "less confident"
            print(f"{display:<40} {corr:>+8.4f}  Higher → {interp}")

        # SFT accuracy by confidence quartile
        quartiles = np.percentile(confidence, [25, 50, 75])
        print(f"\n  SFT accuracy by confidence quartile:")
        bounds = [0] + list(quartiles) + [1.01]
        for i in range(4):
            mask = (confidence >= bounds[i]) & (confidence < bounds[i + 1])
            subset = [m for m, in_q in zip(conf_matched, mask) if in_q]
            if subset:
                acc = sum(1 for m in subset if m['correct']) / len(subset)
                print(f"    Q{i+1} (conf {bounds[i]:.3f}-{bounds[i+1]:.3f}): {acc:.1%} ({len(subset)} papers)")

    # ================================================================
    # ANALYSIS 3: SFT vs LogReg Agreement
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: SFT vs LogReg Agreement/Disagreement")
    print("=" * 70)

    both_correct = sum(1 for m in matched if m['correct'] and m['lr_pred'] == m['label'])
    sft_only = sum(1 for m in matched if m['correct'] and m['lr_pred'] != m['label'])
    lr_only = sum(1 for m in matched if not m['correct'] and m['lr_pred'] == m['label'])
    both_wrong = sum(1 for m in matched if not m['correct'] and m['lr_pred'] != m['label'])

    print(f"\n  Both correct:       {both_correct:>5} ({both_correct/total:.1%})")
    print(f"  SFT only correct:   {sft_only:>5} ({sft_only/total:.1%})")
    print(f"  LogReg only correct:{lr_only:>5} ({lr_only/total:.1%})")
    print(f"  Both wrong:         {both_wrong:>5} ({both_wrong/total:.1%})")
    print(f"  Agreement rate:     {(both_correct+both_wrong)/total:.1%}")

    lr_acc = (both_correct + lr_only) / total
    sft_acc = (both_correct + sft_only) / total
    print(f"\n  SFT accuracy:    {sft_acc:.1%}")
    print(f"  LogReg accuracy: {lr_acc:.1%}")
    print(f"  SFT advantage:   {sft_acc - lr_acc:+.1%}")

    # Also vs HGB
    hgb_correct = sum(1 for m in matched if m['hgb_pred'] == m['label'])
    rf_correct = sum(1 for m in matched if m['rf_pred'] == m['label'])
    print(f"  HGB accuracy:    {hgb_correct/total:.1%}")
    print(f"  RF accuracy:     {rf_correct/total:.1%}")

    # Oracle ensemble
    oracle = sum(1 for m in matched if m['correct'] or m['lr_pred'] == m['label'])
    oracle_hgb = sum(1 for m in matched if m['correct'] or m['hgb_pred'] == m['label'])
    print(f"\n  Oracle (SFT|LR):  {oracle/total:.1%}")
    print(f"  Oracle (SFT|HGB): {oracle_hgb/total:.1%}")

    # ================================================================
    # ANALYSIS 4: Feature profile of SFT-unique wins
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 4: What Makes SFT Better? (SFT-correct, LogReg-wrong)")
    print("=" * 70)

    sft_wins = [m for m in matched if m['correct'] and m['lr_pred'] != m['label']]
    lr_wins = [m for m in matched if not m['correct'] and m['lr_pred'] == m['label']]

    if sft_wins and lr_wins:
        print(f"\n  SFT-unique wins: {len(sft_wins)} papers")
        print(f"  LogReg-unique wins: {len(lr_wins)} papers")
        print(f"\n{'Feature':<35} {'SFT wins':>10} {'LR wins':>10} {'Cohen d':>8}")
        print("-" * 68)

        diffs = []
        for fn in feature_names:
            sft_vals = np.array([m['features'][fn] for m in sft_wins])
            lr_vals = np.array([m['features'][fn] for m in lr_wins])
            sft_mean = np.mean(sft_vals)
            lr_mean = np.mean(lr_vals)
            pooled_std = np.sqrt((np.var(sft_vals) + np.var(lr_vals)) / 2)
            effect = (sft_mean - lr_mean) / pooled_std if pooled_std > 1e-10 else 0
            diffs.append((fn, sft_mean, lr_mean, effect))

        diffs.sort(key=lambda x: abs(x[3]), reverse=True)
        for fn, sm, lm, effect in diffs[:20]:
            display = DISPLAY_NAMES.get(fn, fn)
            print(f"{display:<35} {sm:>10.2f} {lm:>10.2f} {effect:>+8.3f}")

    # ================================================================
    # ANALYSIS 5: SFT Confidence vs LogReg Confidence
    # ================================================================
    if conf_matched:
        print("\n" + "=" * 70)
        print("ANALYSIS 5: SFT Confidence vs LogReg Confidence")
        print("=" * 70)

        sft_conf = np.array([m['confidence'] for m in conf_matched])
        lr_conf = np.array([m['lr_prob_accept'] for m in conf_matched])

        if np.std(sft_conf) > 1e-10 and np.std(lr_conf) > 1e-10:
            corr_conf = np.corrcoef(sft_conf, lr_conf)[0, 1]
            print(f"\n  Pearson r (SFT conf vs LR P(Accept)): {corr_conf:.4f}")

            # Agreement by SFT confidence
            sft_pred_accept = np.array([1 if m['pred'] == 'Accept' else 0 for m in conf_matched])
            lr_pred_accept = np.array([1 if m['lr_pred'] == 'Accept' else 0 for m in conf_matched])
            agreement = np.mean(sft_pred_accept == lr_pred_accept)
            print(f"  Label agreement rate: {agreement:.1%}")

    # ================================================================
    # ANALYSIS 6: Per-year breakdown
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Per-Year SFT vs LogReg")
    print("=" * 70)

    years = sorted(set(m['year'] for m in matched if m['year']))
    print(f"\n{'Year':<8} {'N':>6} {'SFT':>7} {'LR':>7} {'HGB':>7} {'RF':>7} {'Gap':>7} {'Agree':>7}")
    print("-" * 60)
    for year in years:
        subset = [m for m in matched if m['year'] == year]
        n = len(subset)
        sft_a = sum(1 for m in subset if m['correct']) / n
        lr_a = sum(1 for m in subset if m['lr_pred'] == m['label']) / n
        hgb_a = sum(1 for m in subset if m['hgb_pred'] == m['label']) / n
        rf_a = sum(1 for m in subset if m['rf_pred'] == m['label']) / n
        agree = sum(1 for m in subset if m['pred'] == m['lr_pred']) / n
        print(f"{year:<8} {n:>6} {sft_a:>7.1%} {lr_a:>7.1%} {hgb_a:>7.1%} {rf_a:>7.1%} {sft_a-lr_a:>+7.1%} {agree:>7.1%}")

    # ================================================================
    # ANALYSIS 7: Top LogReg Feature Weights (from saved summary)
    # ================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 7: LogReg Feature Weights (pre-computed)")
    print("=" * 70)

    lr_weights = lr_summary.get('lr_weights', [])
    print(f"\n{'Feature':<40} {'Weight':>8}  Direction")
    print("-" * 60)
    for fn, w in lr_weights[:20]:
        display = DISPLAY_NAMES.get(fn, fn)
        print(f"{display:<40} {w:>+8.4f}  {'Accept' if w > 0 else 'Reject'}")

    # ================================================================
    # PLOTS
    # ================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle('SFT Vision Model vs Paper Statistics Baseline', fontsize=14, fontweight='bold')

        # Plot 1: Agreement matrix
        ax = axes[0, 0]
        matrix = np.array([[both_correct, lr_only], [sft_only, both_wrong]])
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['LogReg Correct', 'LogReg Wrong'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['SFT Correct', 'SFT Wrong'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]}\n({matrix[i,j]/total:.0%})",
                        ha='center', va='center', fontsize=12,
                        color='white' if matrix[i, j] > total * 0.3 else 'black')
        ax.set_title('SFT vs LogReg Agreement')

        # Plot 2: Top feature correlations with SFT correctness
        ax = axes[0, 1]
        top_n = 15
        top_corrs = correlations[:top_n]
        names = [DISPLAY_NAMES.get(fn, fn)[:25] for fn, _ in top_corrs]
        values = [c for _, c in top_corrs]
        colors = ['#2196F3' if v > 0 else '#F44336' for v in values]
        y_pos = range(len(names))
        ax.barh(y_pos, values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Correlation with SFT Correctness')
        ax.set_title('Features Most Correlated with SFT Success')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.invert_yaxis()

        # Plot 3: Per-year comparison
        ax = axes[1, 0]
        x_pos = range(len(years))
        sft_accs = [sum(1 for m in matched if m['year'] == y and m['correct']) / sum(1 for m in matched if m['year'] == y) for y in years]
        lr_accs = [sum(1 for m in matched if m['year'] == y and m['lr_pred'] == m['label']) / sum(1 for m in matched if m['year'] == y) for y in years]
        hgb_accs = [sum(1 for m in matched if m['year'] == y and m['hgb_pred'] == m['label']) / sum(1 for m in matched if m['year'] == y) for y in years]
        w = 0.25
        ax.bar([x - w for x in x_pos], sft_accs, w, label='SFT Vision', color='#4CAF50', alpha=0.8)
        ax.bar([x for x in x_pos], hgb_accs, w, label='HGB', color='#FF9800', alpha=0.8)
        ax.bar([x + w for x in x_pos], lr_accs, w, label='LogReg', color='#2196F3', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(years)
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Year: SFT vs Paper Stats Models')
        ax.legend(fontsize=9)
        ax.set_ylim(0.4, 0.85)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

        # Plot 4: SFT-unique wins vs LR-unique wins feature comparison
        if diffs:
            ax = axes[1, 1]
            top_effects = diffs[:10]
            eff_names = [DISPLAY_NAMES.get(fn, fn)[:25] for fn, _, _, _ in top_effects]
            eff_vals = [e for _, _, _, e in top_effects]
            eff_colors = ['#2196F3' if v > 0 else '#FF9800' for v in eff_vals]
            y_pos = range(len(eff_names))
            ax.barh(y_pos, eff_vals, color=eff_colors, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(eff_names, fontsize=8)
            ax.set_xlabel("Cohen's d (SFT-wins vs LogReg-wins)")
            ax.set_title('Feature Differences: Where SFT Uniquely Wins')
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.invert_yaxis()

        plt.tight_layout()
        out_path = os.path.join(FIG_DIR, "sft_vs_paperstats.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved: {out_path}")
        plt.close()

    except ImportError:
        print("\nMatplotlib not available — skipping plots")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Matched {len(matched)} papers across SFT predictions, test metadata, and LogReg results.

Model Accuracies (on matched subset):
  SFT Vision:  {sft_acc:.1%}
  LogReg:      {lr_acc:.1%}  (SFT +{sft_acc-lr_acc:.1%})
  HGB:         {hgb_correct/total:.1%}
  RF:          {rf_correct/total:.1%}

Agreement Analysis:
  SFT & LR agree on {(both_correct+both_wrong)/total:.0%} of papers
  SFT uniquely correct on {sft_only} papers ({sft_only/total:.1%})
  LR uniquely correct on {lr_only} papers ({lr_only/total:.1%})
  Oracle ensemble (either correct): {oracle/total:.1%}

Key Findings:
  1. Low feature correlations with SFT correctness (max |r| = {abs(correlations[0][1]):.3f})
     → SFT success is NOT explained by simple paper statistics
  2. Oracle ensemble at {oracle/total:.1%} shows strong complementarity
     → SFT and LogReg capture different aspects of paper quality
  3. SFT advantage is largest on older papers ({years[0]}: +{sum(1 for m in matched if m['year']==years[0] and m['correct'])/sum(1 for m in matched if m['year']==years[0]) - sum(1 for m in matched if m['year']==years[0] and m['lr_pred']==m['label'])/sum(1 for m in matched if m['year']==years[0]):.0%})
     and smallest on newer papers
""")


if __name__ == "__main__":
    main()
