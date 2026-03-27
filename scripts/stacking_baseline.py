"""Stacking meta-model: paper stats (187 features) + SFT predictions (decision + confidence)."""
import json, csv, re, math, string, sys, os
from collections import Counter
import numpy as np

# Reuse extract_all_features and helpers from logreg_baseline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logreg_baseline import extract_all_features, safe_float, load_split

DATA_DIR = "data"
CSV_PATH = f"{DATA_DIR}/massive_metadata_v7.csv"

# Train: text SFT predictions aligned 1:1 with text train data.json
TRAIN_DATA_PATH = f"{DATA_DIR}/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train/data.json"
TRAIN_PRED_PATH = "results/data_cleaning/text_train_predictions.jsonl"

# Test: text SFT predictions aligned 1:1 with text test data.json
TEXT_TEST_DATA_PATH = f"{DATA_DIR}/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json"
TEXT_TEST_PRED_PATH = "results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_text/finetuned-ckpt-2644.jsonl"


def load_predictions(jsonl_path):
    """Load SFT predictions: extract decision (1=Accept, 0=Reject) and confidence."""
    preds = []
    with open(jsonl_path) as fh:
        for line in fh:
            obj = json.loads(line)
            decision = 1 if "Accept" in obj["predict"] else 0
            # token_logprobs[5] is the Accept/Reject decision token
            logprob = obj["token_logprobs"][5]
            confidence = math.exp(logprob)
            preds.append({"decision": decision, "confidence": confidence})
    return preds


def load_data_json(path):
    """Load data.json and return list of (submission_id, label, metadata, text)."""
    with open(path) as fh:
        data = json.load(fh)
    records = []
    for entry in data:
        meta = entry.get("_metadata", {})
        sid = meta.get("submission_id")
        label, text = None, ""
        for conv in entry["conversations"]:
            if conv["from"] == "gpt":
                label = 1 if "Accept" in conv["value"] else 0
            if conv["from"] == "human":
                text = conv["value"]
        records.append((sid, label, meta, text))
    return records


def build_stacking_matrix(records, stats, sft_preds):
    """Build feature matrix: 187 paper stats + 2 SFT features."""
    all_feats, labels = [], []
    for i, (sid, label, meta, text) in enumerate(records):
        if i % 2000 == 0:
            print(f"  extracting features: {i}/{len(records)}", flush=True)
        if sid not in stats:
            # Skip papers without CSV metadata
            continue
        f = extract_all_features(text, meta, stats[sid])
        # Add SFT model features
        f["sft_decision"] = float(sft_preds[i]["decision"])
        f["sft_confidence"] = float(sft_preds[i]["confidence"])
        all_feats.append(f)
        labels.append(label)
    feature_names = sorted(all_feats[0].keys())
    X = np.array([[f.get(fn, 0) for fn in feature_names] for f in all_feats])
    return X, np.array(labels), feature_names


def per_year_metrics(records, y_true, y_pred, label="Model"):
    """Print per-year accuracy, acceptance rate accuracy, rejection rate accuracy."""
    years = {}
    for i, (sid, lbl, meta, text) in enumerate(records):
        yr = meta.get("year", "?")
        if yr not in years:
            years[yr] = {"correct": 0, "total": 0, "acc_correct": 0, "acc_total": 0,
                         "rej_correct": 0, "rej_total": 0}
        years[yr]["total"] += 1
        if y_pred[i] == y_true[i]:
            years[yr]["correct"] += 1
        if y_true[i] == 1:
            years[yr]["acc_total"] += 1
            if y_pred[i] == 1:
                years[yr]["acc_correct"] += 1
        else:
            years[yr]["rej_total"] += 1
            if y_pred[i] == 0:
                years[yr]["rej_correct"] += 1

    print(f"\n--- {label} Per-Year Metrics ---")
    print(f"{'Year':<8} {'N':>5} {'Acc':>7} {'AccR':>7} {'RejR':>7}")
    print("-" * 40)
    for yr in sorted(years.keys()):
        d = years[yr]
        acc = d["correct"] / max(d["total"], 1)
        accr = d["acc_correct"] / max(d["acc_total"], 1)
        rejr = d["rej_correct"] / max(d["rej_total"], 1)
        print(f"{yr:<8} {d['total']:>5} {acc:>7.1%} {accr:>7.1%} {rejr:>7.1%}")

    total_acc = sum(d["correct"] for d in years.values()) / max(sum(d["total"] for d in years.values()), 1)
    print(f"{'TOTAL':<8} {sum(d['total'] for d in years.values()):>5} {total_acc:>7.1%}")
    return total_acc


def main():
    print("Loading metadata CSV...", flush=True)
    stats = {}
    with open(CSV_PATH) as fh:
        for row in csv.DictReader(fh):
            sid = row.get("submission_id")
            if sid:
                stats[sid] = row

    # Load train data + predictions
    print("Loading train data and SFT predictions...", flush=True)
    train_records = load_data_json(TRAIN_DATA_PATH)
    train_preds = load_predictions(TRAIN_PRED_PATH)
    assert len(train_records) == len(train_preds), \
        f"Train mismatch: {len(train_records)} records vs {len(train_preds)} predictions"
    print(f"  Train: {len(train_records)} records, {len(train_preds)} predictions")

    # Load text test data + predictions (same model family as train)
    print("Loading text test data and SFT predictions...", flush=True)
    test_records_raw = load_data_json(TEXT_TEST_DATA_PATH)
    test_preds_raw = load_predictions(TEXT_TEST_PRED_PATH)
    assert len(test_records_raw) == len(test_preds_raw), \
        f"Test mismatch: {len(test_records_raw)} records vs {len(test_preds_raw)} predictions"
    print(f"  Text test: {len(test_records_raw)} records, {len(test_preds_raw)} predictions")

    # Filter train to only papers with CSV metadata
    train_with_stats = [(i, r) for i, r in enumerate(train_records) if r[0] in stats]
    print(f"  Train with CSV metadata: {len(train_with_stats)}/{len(train_records)}")

    # Filter test to only papers with CSV metadata
    test_with_stats = [(i, r) for i, r in enumerate(test_records_raw) if r[0] in stats]
    print(f"  Test with CSV metadata: {len(test_with_stats)}/{len(test_records_raw)}")

    # Build feature matrices
    print("\nBuilding train feature matrix...", flush=True)
    filtered_train_records = [r for i, r in train_with_stats]
    filtered_train_preds = [train_preds[i] for i, r in train_with_stats]
    X_train, y_train, feature_names = build_stacking_matrix(
        filtered_train_records, stats, filtered_train_preds)

    print("Building test feature matrix...", flush=True)
    final_test_records = [r for i, r in test_with_stats]
    final_test_preds = [test_preds_raw[i] for i, r in test_with_stats]
    X_test, y_test, _ = build_stacking_matrix(
        final_test_records, stats, final_test_preds)

    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    print(f"\nFeatures: {len(feature_names)} (should be 189)")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Verify SFT features are present
    sft_idx = [i for i, fn in enumerate(feature_names) if fn.startswith("sft_")]
    print(f"SFT feature indices: {[(feature_names[i], i) for i in sft_idx]}")

    # Build sample weights: upweight 2025/2026 training examples
    year_idx = feature_names.index("year")
    train_years = X_train[:, year_idx]
    WEIGHT_MULTIPLIERS = {2025: 3.0, 2026: 3.0}
    sample_weights = np.array([
        WEIGHT_MULTIPLIERS.get(int(y), 1.0) for y in train_years
    ])
    n_upweighted = np.sum(sample_weights > 1.0)
    print(f"Sample weights: {n_upweighted} papers upweighted (2025/2026 x{WEIGHT_MULTIPLIERS[2025]}), "
          f"effective train size = {sample_weights.sum():.0f}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.ensemble import HistGradientBoostingClassifier

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Logistic Regression ---
    print("\n=== LOGISTIC REGRESSION (Paper Stats + SFT) ===", flush=True)
    best_lr_acc, best_lr, best_lr_C = 0, None, None
    for C in [0.01, 0.1, 1.0, 5.0]:
        m = LogisticRegression(max_iter=5000, C=C)
        m.fit(X_train_s, y_train, sample_weight=sample_weights)
        preds = m.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        tacc = accuracy_score(y_train, m.predict(X_train_s))
        print(f"  C={C:<5} train={tacc:.1%}  test={acc:.1%}", flush=True)
        if acc > best_lr_acc:
            best_lr_acc, best_lr, best_lr_C = acc, m, C
    print(f"Best LogReg: {best_lr_acc:.1%} (C={best_lr_C})", flush=True)

    # Per-year for best LR
    lr_preds = best_lr.predict(X_test_s)
    per_year_metrics(final_test_records, y_test, lr_preds, "LogReg (Paper Stats + SFT)")

    # --- HistGradientBoosting ---
    print("\n=== HIST GRADIENT BOOSTING (Paper Stats + SFT) ===", flush=True)
    best_gb_acc, best_gb, best_gb_params = 0, None, None
    for n_est in [100, 300, 500, 800]:
        for depth in [3, 5, 7, 10]:
            for lr in [0.03, 0.05, 0.1]:
                m = HistGradientBoostingClassifier(
                    max_iter=n_est, max_depth=depth,
                    learning_rate=lr, min_samples_leaf=20,
                    random_state=42, early_stopping=True,
                    validation_fraction=0.1, n_iter_no_change=20,
                )
                m.fit(X_train, y_train, sample_weight=sample_weights)
                preds = m.predict(X_test)
                acc = accuracy_score(y_test, preds)
                tacc = accuracy_score(y_train, m.predict(X_train))
                if acc > best_gb_acc:
                    best_gb_acc, best_gb = acc, m
                    best_gb_params = (n_est, depth, lr)
                    print(f"  NEW BEST: n={n_est} d={depth} lr={lr} train={tacc:.1%} test={acc:.1%}", flush=True)
    print(f"\nBest HGB: {best_gb_acc:.1%} params={best_gb_params}", flush=True)

    # Per-year for best HGB
    gb_preds = best_gb.predict(X_test)
    per_year_metrics(final_test_records, y_test, gb_preds, "HGB (Paper Stats + SFT)")

    # --- LR weights for SFT features ---
    print(f"\n--- Top 30 LogReg Weights ---", flush=True)
    weights = sorted(zip(feature_names, best_lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    print(f"{'Feature':<40} {'Weight':>10}")
    print("-" * 52)
    for fn, w in weights[:30]:
        marker = " *** SFT ***" if fn.startswith("sft_") else ""
        print(f"{fn:<40} {w:>10.4f}  {'→ Accept' if w > 0 else '→ Reject'}{marker}")

    # Show SFT feature weights specifically
    print(f"\n--- SFT Feature Weights ---")
    for fn, w in weights:
        if fn.startswith("sft_"):
            print(f"  {fn}: {w:.4f}")

    # --- Feature importances from best HGB ---
    print(f"\n--- Top 30 Feature Importances (HGB, permutation) ---", flush=True)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(best_gb, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    idx = np.argsort(perm.importances_mean)[::-1]
    print(f"{'Feature':<40} {'Importance':>10}")
    print("-" * 52)
    for i in idx[:30]:
        marker = " *** SFT ***" if feature_names[i].startswith("sft_") else ""
        print(f"{feature_names[i]:<40} {perm.importances_mean[i]:>10.4f}{marker}")

    # --- Classification reports ---
    print(f"\n--- Best HGB Classification Report ---", flush=True)
    print(classification_report(y_test, gb_preds, target_names=["Reject", "Accept"]))

    # --- COMPARISON TABLE ---
    print(f"\n{'='*60}")
    print(f"{'MODEL':<50} {'TEST ACC':>8}")
    print(f"{'='*60}")
    print(f"{'Random baseline':<50} {'50.0%':>8}")
    print(f"{'Paper Stats Only - LogReg (187 feat)':<50} {'64.0%':>8}")
    print(f"{'Paper Stats Only - HGB (187 feat)':<50} {'64.8%':>8}")
    print(f"{'SFT Vision Only (best checkpoint)':<50} {'66.5%':>8}")
    nf = len(feature_names)
    print(f"{'Paper Stats + SFT - LogReg (' + str(nf) + ' feat)':<50} {best_lr_acc:>7.1%}")
    print(f"{'Paper Stats + SFT - HGB (' + str(nf) + ' feat)':<50} {best_gb_acc:>7.1%}")
    print(f"{'='*60}", flush=True)

    # --- SAVE RESULTS ---
    import pickle
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "stacking_baseline")
    os.makedirs(out_dir, exist_ok=True)

    pickle.dump(best_lr, open(os.path.join(out_dir, "logreg_model.pkl"), "wb"))
    pickle.dump(best_gb, open(os.path.join(out_dir, "hgb_model.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(out_dir, "scaler.pkl"), "wb"))

    # Save per-paper predictions
    lr_probs = best_lr.predict_proba(X_test_s)
    gb_probs = best_gb.predict_proba(X_test)
    predictions = []
    for i, (sid, label, meta, text) in enumerate(final_test_records):
        predictions.append({
            "submission_id": sid,
            "label": int(label),
            "year": meta.get("year"),
            "sft_decision": final_test_preds[i]["decision"],
            "sft_confidence": final_test_preds[i]["confidence"],
            "lr_pred": int(lr_preds[i]),
            "lr_prob_accept": float(lr_probs[i][1]),
            "hgb_pred": int(gb_preds[i]),
            "hgb_prob_accept": float(gb_probs[i][1]),
        })

    with open(os.path.join(out_dir, "test_predictions.json"), "w") as fh:
        json.dump(predictions, fh, indent=2)

    results_summary = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "lr_accuracy": best_lr_acc,
        "lr_C": best_lr_C,
        "hgb_accuracy": best_gb_acc,
        "hgb_params": best_gb_params,
        "lr_weights": sorted(zip(feature_names, best_lr.coef_[0].tolist()),
                             key=lambda x: abs(x[1]), reverse=True),
        "hgb_feature_importances": sorted(
            zip(feature_names, perm.importances_mean.tolist()),
            key=lambda x: abs(x[1]), reverse=True
        ),
    }
    with open(os.path.join(out_dir, "results_summary.json"), "w") as fh:
        json.dump(results_summary, fh, indent=2)

    print(f"\nResults saved to {out_dir}/")
    print(f"  - test_predictions.json ({len(predictions)} papers)")
    print(f"  - results_summary.json")


if __name__ == "__main__":
    main()
