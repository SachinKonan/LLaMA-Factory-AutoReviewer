"""TF-IDF + Logistic Regression baseline for paper acceptance prediction."""
import json, os
import numpy as np


DATA_DIR = "data"
TRAIN_PATH = f"{DATA_DIR}/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train/data.json"
TEST_PATH = f"{DATA_DIR}/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json"


def load_split(path):
    with open(path) as fh:
        data = json.load(fh)
    records = []
    for entry in data:
        meta = entry.get('_metadata', {})
        sid = meta.get('submission_id')
        label, text = None, ""
        for conv in entry['conversations']:
            if conv['from'] == 'gpt':
                label = 1 if 'Accept' in conv['value'] else 0
            if conv['from'] == 'human':
                text = conv['value']
        if sid and label is not None:
            records.append({
                "submission_id": sid,
                "label": label,
                "year": meta.get("year"),
                "text": text,
            })
    return records


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print("Loading data...", flush=True)
    train_records = load_split(TRAIN_PATH)
    test_records = load_split(TEST_PATH)
    print(f"Train: {len(train_records)}, Test: {len(test_records)}", flush=True)

    train_texts = [r["text"] for r in train_records]
    test_texts = [r["text"] for r in test_records]
    y_train = np.array([r["label"] for r in train_records])
    y_test = np.array([r["label"] for r in test_records])

    print("Fitting TF-IDF...", flush=True)
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print("Training LogisticRegression...", flush=True)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    # Per-year metrics
    years = sorted(set(r["year"] for r in test_records if r["year"] is not None))
    print(f"\n{'Year':<8} {'Acc':>8} {'AccR':>8} {'RejR':>8} {'N':>6}")
    print("-" * 42)
    for year in years:
        mask = np.array([r["year"] == year for r in test_records])
        y_t = y_test[mask]
        p_t = preds[mask]
        acc = accuracy_score(y_t, p_t) * 100
        acc_recall = (p_t[y_t == 1].sum() / max(y_t.sum(), 1)) * 100
        rej_recall = ((1 - p_t[y_t == 0]).sum() / max((1 - y_t).sum(), 1)) * 100
        print(f"{year:<8} {acc:>7.1f}% {acc_recall:>7.1f}% {rej_recall:>7.1f}% {mask.sum():>6}")

    # Overall
    acc = accuracy_score(y_test, preds) * 100
    acc_recall = (preds[y_test == 1].sum() / max(y_test.sum(), 1)) * 100
    rej_recall = ((1 - preds[y_test == 0]).sum() / max((1 - y_test).sum(), 1)) * 100
    print(f"{'Overall':<8} {acc:>7.1f}% {acc_recall:>7.1f}% {rej_recall:>7.1f}% {len(y_test):>6}")

    # 2025 only
    mask_2025 = np.array([r["year"] == 2025 for r in test_records])
    if mask_2025.any():
        y_25 = y_test[mask_2025]
        p_25 = preds[mask_2025]
        print(f"\n2025: Acc={accuracy_score(y_25, p_25)*100:.1f}%, "
              f"AccR={(p_25[y_25==1].sum()/max(y_25.sum(),1))*100:.1f}%, "
              f"RejR={((1-p_25[y_25==0]).sum()/max((1-y_25).sum(),1))*100:.1f}%")

    # 2026 only
    mask_2026 = np.array([r["year"] == 2026 for r in test_records])
    if mask_2026.any():
        y_26 = y_test[mask_2026]
        p_26 = preds[mask_2026]
        print(f"2026: Acc={accuracy_score(y_26, p_26)*100:.1f}%, "
              f"AccR={(p_26[y_26==1].sum()/max(y_26.sum(),1))*100:.1f}%, "
              f"RejR={((1-p_26[y_26==0]).sum()/max((1-y_26).sum(),1))*100:.1f}%")

    # 2025+2026
    mask_2526 = mask_2025 | mask_2026
    if mask_2526.any():
        y_2526 = y_test[mask_2526]
        p_2526 = preds[mask_2526]
        print(f"2025+2026: Acc={accuracy_score(y_2526, p_2526)*100:.1f}%, "
              f"AccR={(p_2526[y_2526==1].sum()/max(y_2526.sum(),1))*100:.1f}%, "
              f"RejR={((1-p_2526[y_2526==0]).sum()/max((1-y_2526).sum(),1))*100:.1f}%")

    # Save predictions
    out_dir = os.path.join("results", "tfidf_baseline")
    os.makedirs(out_dir, exist_ok=True)
    predictions = []
    for i, rec in enumerate(test_records):
        predictions.append({
            "submission_id": rec["submission_id"],
            "label": rec["label"],
            "year": rec["year"],
            "pred": int(preds[i]),
            "prob_accept": float(probs[i][1]),
        })
    with open(os.path.join(out_dir, "test_predictions.json"), "w") as fh:
        json.dump(predictions, fh, indent=2)
    print(f"\nPredictions saved to {out_dir}/test_predictions.json ({len(predictions)} papers)")


if __name__ == "__main__":
    main()
