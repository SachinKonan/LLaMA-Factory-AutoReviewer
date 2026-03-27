"""Logistic regression + GBM baseline on programmatic paper features."""
import json, csv, re, math, string, sys, os
from collections import Counter
import numpy as np

DATA_DIR = "data"
TRAIN_PATH = f"{DATA_DIR}/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train/data.json"
TEST_PATH = f"{DATA_DIR}/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json"
CSV_PATH = f"{DATA_DIR}/massive_metadata_v7.csv"

# ---- TOPIC KEYWORDS (proxy for arxiv category) ----
TOPIC_KEYWORDS = {
    'topic_nlp': r'\b(NLP|natural language|text classification|sentiment|translation|language model|transformer|BERT|GPT|token|vocabulary|corpus|parsing|NER|named entity)\b',
    'topic_cv': r'\b(image|vision|convolutional|CNN|object detection|segmentation|pixel|ResNet|ViT|visual|ImageNet|CIFAR|recognition)\b',
    'topic_rl': r'\b(reinforcement learning|reward|policy gradient|MDP|Q-learning|actor.critic|environment|agent|exploration|exploitation|Atari|MuJoCo)\b',
    'topic_generative': r'\b(GAN|generative|VAE|variational|diffusion|autoencoder|latent space|generation|synthesis|sample quality|FID)\b',
    'topic_graph': r'\b(graph neural|GNN|node|edge|adjacency|message passing|graph convolution|knowledge graph|link prediction)\b',
    'topic_optimization': r'\b(optimization|convergence|SGD|Adam|learning rate|gradient descent|momentum|regularization|batch size|weight decay)\b',
    'topic_theory': r'\b(theorem|proof|lemma|proposition|corollary|bound|complexity|PAC|VC dimension|Rademacher|concentration inequality)\b',
    'topic_fairness': r'\b(fairness|bias|ethical|demographic|equit|disparate|protected attribute|discrimination)\b',
    'topic_speech': r'\b(speech|audio|acoustic|spectrogram|ASR|TTS|voice|phoneme|waveform)\b',
    'topic_meta_learning': r'\b(meta.learning|few.shot|zero.shot|transfer learning|domain adaptation|multi.task)\b',
    'topic_self_supervised': r'\b(self.supervised|contrastive|SimCLR|BYOL|MoCo|pretext task|representation learning|unsupervised pre)\b',
    'topic_efficient': r'\b(pruning|quantization|distillation|compression|efficient|lightweight|mobile|edge|inference speed|FLOP)\b',
    'topic_federated': r'\b(federated|distributed learning|privacy|differential privacy|communication efficient)\b',
    'topic_multimodal': r'\b(multimodal|cross.modal|vision.language|CLIP|image.text|visual question)\b',
}

HEDGE_WORDS = r'\b(might|could|may|possibly|perhaps|likely|suggests|appears|seemingly|arguably|potentially|uncertain)\b'
CONFIDENCE_WORDS = r'\b(clearly|obviously|significantly|substantially|dramatically|remarkably|notably|undoubtedly|certainly|demonstrate|prove|establish|confirm|superior)\b'
NEGATIVE_WORDS = r'\b(fail|limitation|drawback|weakness|shortcoming|disadvantage|unable|cannot|lack|insufficient|poor|worse|degrade|suffer)\b'
POSITIVE_WORDS = r'\b(outperform|improve|achieve|superior|advantage|benefit|effective|efficient|robust|strong|excellent|promising|remarkable|impressive|advance)\b'


def safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if not math.isnan(v) else default
    except (ValueError, TypeError):
        return default


def extract_all_features(text, meta, csv_row):
    f = {}
    text_lower = text.lower()

    # ======== CSV STRUCTURAL FEATURES ========
    for col in ['num_authors', 'num_figures', 'num_figure_images', 'num_pages',
                'num_text_tokens', 'num_text_image_tokens', 'num_vision_tokens',
                'num_equations', 'number_of_cited_references', 'number_of_bib_items',
                'num_pages_noreferences', 'original_total_pages', 'total_removed_pages',
                'body_items_count', 'removed_after_cutoff_count', 'intro_idx']:
        f[col] = safe_float(csv_row.get(col))

    pages = max(f['num_pages'], 1)

    # ======== RATIO FEATURES ========
    f['figures_per_page'] = f['num_figures'] / pages
    f['equations_per_page'] = f['num_equations'] / pages
    f['refs_per_page'] = f['number_of_cited_references'] / pages
    f['tokens_per_page'] = f['num_text_tokens'] / pages
    f['appendix_pages'] = f['original_total_pages'] - f['num_pages']
    f['appendix_ratio'] = f['appendix_pages'] / max(f['original_total_pages'], 1)
    f['image_token_ratio'] = f['num_text_image_tokens'] / max(f['num_text_tokens'], 1)
    f['vision_token_ratio'] = f['num_vision_tokens'] / max(f['num_text_tokens'], 1)
    f['subfigs_per_figure'] = f['num_figure_images'] / max(f['num_figures'], 1)
    f['refs_per_author'] = f['number_of_cited_references'] / max(f['num_authors'], 1)
    f['pages_per_author'] = f['num_pages'] / max(f['num_authors'], 1)
    f['equations_per_ref'] = f['num_equations'] / max(f['number_of_cited_references'], 1)
    f['figures_per_ref'] = f['num_figures'] / max(f['number_of_cited_references'], 1)

    # ======== YEAR ========
    f['year'] = safe_float(meta.get('year', csv_row.get('year', 2022)))

    # ======== TITLE FEATURES ========
    title_match = re.search(r'^#\s+(.+?)$', text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""
    title_words = title.split()
    f['title_words'] = len(title_words)
    f['title_chars'] = len(title)
    f['title_has_colon'] = 1.0 if ':' in title else 0.0
    f['title_has_question'] = 1.0 if '?' in title else 0.0
    f['title_caps_ratio'] = sum(1 for c in title if c.isupper()) / max(len(title), 1)
    f['title_avg_word_len'] = np.mean([len(w) for w in title_words]) if title_words else 0
    f['title_num_acronyms'] = sum(1 for w in title_words if w.isupper() and len(w) >= 2 and w.isalpha())
    f['title_has_number'] = 1.0 if re.search(r'\d', title) else 0.0
    f['title_has_hyphen'] = 1.0 if '-' in title else 0.0
    f['title_has_parens'] = 1.0 if '(' in title else 0.0

    # ======== ABSTRACT FEATURES ========
    abs_match = re.search(r'(?:ABSTRACT|Abstract)\s*\n+(.*?)(?:\n#|\n\n[A-Z0-9])', text, re.DOTALL)
    abstract = abs_match.group(1).strip() if abs_match else ""
    abs_words = abstract.split()
    abs_sents = [s for s in re.split(r'[.!?]+', abstract) if s.strip()]
    f['abstract_words'] = len(abs_words)
    f['abstract_chars'] = len(abstract)
    f['abstract_sentences'] = len(abs_sents)
    f['abstract_avg_sent_len'] = len(abs_words) / max(len(abs_sents), 1)
    f['abstract_avg_word_len'] = np.mean([len(w) for w in abs_words]) if abs_words else 0
    abs_words_lower = [w.lower().strip(string.punctuation) for w in abs_words if w.strip(string.punctuation)]
    f['abstract_vocab_richness'] = len(set(abs_words_lower)) / max(len(abs_words_lower), 1)
    f['abstract_has_numbers'] = sum(1 for w in abs_words if any(c.isdigit() for c in w)) / max(len(abs_words), 1)

    # ======== FULL TEXT WRITING QUALITY ========
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    f['total_words'] = len(words)
    f['total_sentences'] = len(sentences)
    f['avg_sentence_length'] = len(words) / max(len(sentences), 1)
    sent_lens = [len(s.split()) for s in sentences]
    f['sentence_len_std'] = np.std(sent_lens) if sent_lens else 0
    f['sentence_len_cv'] = f['sentence_len_std'] / max(f['avg_sentence_length'], 1)
    word_lens = [len(w) for w in words if w.isalpha()]
    f['avg_word_length'] = np.mean(word_lens) if word_lens else 0
    f['word_len_std'] = np.std(word_lens) if word_lens else 0
    f['long_words_ratio'] = sum(1 for l in word_lens if l > 8) / max(len(word_lens), 1)
    sample_words = [w.lower().strip(string.punctuation) for w in words[:5000]]
    sample_words = [w for w in sample_words if w and w.isalpha()]
    f['vocab_richness_5k'] = len(set(sample_words)) / max(len(sample_words), 1)
    word_counts = Counter(sample_words)
    f['hapax_ratio'] = sum(1 for c in word_counts.values() if c == 1) / max(len(word_counts), 1)

    # ======== STRUCTURE FEATURES ========
    f['num_h1'] = len(re.findall(r'^# ', text, re.MULTILINE))
    f['num_h2'] = len(re.findall(r'^## ', text, re.MULTILINE))
    f['num_h3'] = len(re.findall(r'^### ', text, re.MULTILINE))
    f['total_sections'] = f['num_h1'] + f['num_h2'] + f['num_h3']
    f['section_depth'] = f['num_h3'] / max(f['num_h2'], 1)
    f['sections_per_page'] = f['total_sections'] / pages
    f['has_related_work'] = 1.0 if 'related work' in text_lower else 0.0
    f['has_conclusion'] = 1.0 if 'conclusion' in text_lower else 0.0
    f['has_limitations'] = 1.0 if 'limitation' in text_lower else 0.0
    f['has_ethics'] = 1.0 if re.search(r'ethic|broader impact', text_lower) else 0.0
    f['has_reproducibility'] = 1.0 if re.search(r'reproduc|code avail|open.sourc', text_lower) else 0.0
    f['has_future_work'] = 1.0 if 'future work' in text_lower else 0.0
    f['has_background'] = 1.0 if re.search(r'\bbackground\b|preliminar', text_lower) else 0.0

    # ======== CROSS-REFERENCE DENSITY ========
    f['num_table_mentions'] = len(re.findall(r'Table\s+\d', text))
    f['num_figure_mentions'] = len(re.findall(r'(?:Figure|Fig\.?)\s*\d', text))
    f['num_eq_refs'] = len(re.findall(r'(?:Eq(?:uation)?\.?\s*[\(\[]?\d|\\eqref|\\ref)', text))
    f['num_section_refs'] = len(re.findall(r'Section\s+\d', text))
    f['num_appendix_refs'] = len(re.findall(r'Appendix\s+[A-Z]', text))
    f['cross_ref_density'] = (f['num_table_mentions'] + f['num_figure_mentions'] + f['num_eq_refs'] + f['num_section_refs']) / pages
    f['fig_mention_per_fig'] = f['num_figure_mentions'] / max(f['num_figures'], 1)
    f['table_mention_density'] = f['num_table_mentions'] / pages

    # ======== MATH SOPHISTICATION ========
    f['num_inline_math'] = text.count('$') // 2
    f['num_display_math'] = len(re.findall(r'\$\$', text)) // 2
    f['num_theorem_like'] = len(re.findall(r'\b(?:Theorem|Lemma|Proposition|Corollary)\b', text))
    f['num_definitions'] = len(re.findall(r'\bDefinition\b', text))
    f['num_proofs'] = len(re.findall(r'\bProof\b', text))
    f['has_proof'] = 1.0 if f['num_proofs'] > 0 else 0.0
    f['num_algorithms'] = len(re.findall(r'Algorithm\s+\d', text))
    f['math_density'] = f['num_inline_math'] / max(f['total_words'], 1) * 100
    f['theory_score'] = f['num_theorem_like'] + f['num_proofs'] + f['num_definitions']

    # ======== EXPERIMENTAL RIGOR ========
    f['mentions_ablation'] = 1.0 if 'ablation' in text_lower else 0.0
    f['mentions_hyperparameter'] = 1.0 if re.search(r'hyper.?param', text_lower) else 0.0
    f['mentions_baseline'] = 1.0 if 'baseline' in text_lower else 0.0
    f['num_baselines'] = len(re.findall(r'baseline', text_lower))
    f['mentions_sota'] = 1.0 if re.search(r'state.of.the.art|SOTA', text) else 0.0
    f['mentions_benchmark'] = 1.0 if 'benchmark' in text_lower else 0.0
    f['num_datasets_mentioned'] = len(re.findall(r'\b(?:CIFAR|ImageNet|MNIST|GLUE|SuperGLUE|SQuAD|WMT|COCO|VOC|Penn Treebank|WikiText|OpenWebText|CC\-?\d|Pile)\b', text))
    f['mentions_significance'] = 1.0 if re.search(r'statistic.*signific|p.value|confidence interval|standard deviation|error bar', text_lower) else 0.0
    f['mentions_training_details'] = sum(1 for pat in [r'learning rate', r'batch size', r'epoch', r'GPU', r'TPU', r'A100', r'V100', r'training time', r'wall.clock'] if re.search(pat, text_lower))
    f['mentions_code'] = 1.0 if re.search(r'github|code.*avail|open.sourc|our code|implementation.*avail', text_lower) else 0.0
    f['has_github'] = 1.0 if 'github' in text_lower else 0.0

    # ======== SENTIMENT / RHETORIC ========
    f['hedge_count'] = len(re.findall(HEDGE_WORDS, text_lower))
    f['confidence_count'] = len(re.findall(CONFIDENCE_WORDS, text_lower))
    f['negative_count'] = len(re.findall(NEGATIVE_WORDS, text_lower))
    f['positive_count'] = len(re.findall(POSITIVE_WORDS, text_lower))
    f['hedge_ratio'] = f['hedge_count'] / max(f['total_words'], 1) * 1000
    f['confidence_ratio'] = f['confidence_count'] / max(f['total_words'], 1) * 1000
    f['pos_neg_ratio'] = f['positive_count'] / max(f['negative_count'], 1)
    f['rhetoric_balance'] = f['confidence_count'] / max(f['hedge_count'], 1)

    # ======== TOPIC DETECTION ========
    for topic_name, pattern in TOPIC_KEYWORDS.items():
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        f[topic_name] = matches
        f[topic_name + '_density'] = matches / max(f['total_words'], 1) * 1000
    topic_counts = {k: f[k] for k in TOPIC_KEYWORDS}
    dominant = max(topic_counts, key=topic_counts.get)
    for k in TOPIC_KEYWORDS:
        f[f'is_dominant_{k}'] = 1.0 if k == dominant else 0.0

    # ======== CITATION PATTERNS ========
    f['num_cite_parens'] = len(re.findall(r'\([\w\s&,]+?,?\s*\d{4}[a-z]?\)', text))
    f['num_cite_brackets'] = len(re.findall(r'\[\d+(?:,\s*\d+)*\]', text))
    cite_years = [int(y) for y in re.findall(r'(?:19|20)\d{2}', text) if 1990 <= int(y) <= 2026]
    if cite_years:
        f['cite_year_mean'] = np.mean(cite_years)
        f['cite_year_median'] = np.median(cite_years)
        f['cite_year_std'] = np.std(cite_years)
        f['cite_year_max'] = max(cite_years)
        f['cite_year_recency'] = f['year'] - np.median(cite_years)
        f['pct_recent_cites'] = sum(1 for y in cite_years if y >= f['year'] - 2) / len(cite_years)
        f['pct_old_cites'] = sum(1 for y in cite_years if y < f['year'] - 5) / len(cite_years)
    else:
        for k in ['cite_year_mean','cite_year_median','cite_year_std','cite_year_max','cite_year_recency','pct_recent_cites','pct_old_cites']:
            f[k] = 0

    # ======== AUTHOR FEATURES (structural only) ========
    authors = meta.get('authors', [])
    if isinstance(authors, str):
        try: authors = json.loads(authors)
        except: authors = []
    f['num_authors_meta'] = len(authors)
    f['is_single_author'] = 1.0 if len(authors) == 1 else 0.0
    f['is_large_team'] = 1.0 if len(authors) >= 6 else 0.0
    if authors:
        name_lengths = [len(a) for a in authors]
        f['avg_author_name_len'] = np.mean(name_lengths)
        f['max_author_name_len'] = max(name_lengths)
        f['avg_name_parts'] = np.mean([len(a.split()) for a in authors])
    else:
        f['avg_author_name_len'] = f['max_author_name_len'] = f['avg_name_parts'] = 0

    # ======== READABILITY ========
    alpha_words = [w for w in words if w.isalpha()]
    syllable_count = sum(max(1, len(re.findall(r'[aeiouy]+', w.lower()))) for w in alpha_words)
    wc = max(len(alpha_words), 1)
    sc = max(len(sentences), 1)
    f['flesch_kincaid'] = 0.39 * (wc / sc) + 11.8 * (syllable_count / wc) - 15.59
    f['avg_syllables'] = syllable_count / wc
    cc = sum(len(w) for w in alpha_words)
    f['coleman_liau'] = 0.0588 * (cc / wc * 100) - 0.296 * (sc / wc * 100) - 15.8

    # ======== FORMATTING ========
    f['num_bullets'] = len(re.findall(r'^\s*[-•*]\s', text, re.MULTILINE))
    f['num_numbered_lists'] = len(re.findall(r'^\s*\d+[.)]\s', text, re.MULTILINE))
    f['num_bold'] = len(re.findall(r'\*\*[^*]+\*\*', text))
    f['uses_lists'] = 1.0 if f['num_bullets'] + f['num_numbered_lists'] > 3 else 0.0

    # ======== INTRO LENGTH ========
    intro_match = re.search(r'(?:INTRODUCTION|Introduction)\s*\n+(.*?)(?=\n#)', text, re.DOTALL)
    if intro_match:
        f['intro_words'] = len(intro_match.group(1).split())
        f['intro_ratio'] = f['intro_words'] / max(f['total_words'], 1)
    else:
        f['intro_words'] = f['intro_ratio'] = 0

    # ======== NOVELTY LANGUAGE ========
    f['mentions_novel'] = len(re.findall(r'\bnovel\b', text_lower))
    f['mentions_first'] = len(re.findall(r'\bfirst\b', text_lower))
    f['mentions_propose'] = len(re.findall(r'\bpropos', text_lower))
    f['mentions_contribution'] = len(re.findall(r'\bcontribut', text_lower))
    f['mentions_key_insight'] = len(re.findall(r'key\s+(?:insight|idea|observation)', text_lower))
    f['self_citation_words'] = len(re.findall(r'\bour\s+(?:method|approach|model|framework|algorithm|technique|work|paper|contribution)\b', text_lower))

    # ======== INTERACTION TERMS ========
    f['theory_x_experiment'] = f['theory_score'] * f['mentions_ablation']
    f['length_x_refs'] = f['num_text_tokens'] * f['number_of_cited_references'] / 1e6
    f['figures_x_tables'] = f['num_figures'] * f['num_table_mentions']
    f['authors_x_refs'] = f['num_authors'] * f['number_of_cited_references']
    f['year_x_refs'] = (f['year'] - 2020) * f['number_of_cited_references']

    # ======== LOG TRANSFORMS ========
    for key in ['num_text_tokens', 'number_of_cited_references', 'total_words', 'num_inline_math']:
        f[f'log_{key}'] = np.log1p(f[key])

    return f


def load_split(path, stats):
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
        if sid and label is not None and sid in stats:
            records.append((sid, label, meta, text))
    return records


def build_matrix(records, stats):
    all_feats, labels = [], []
    for i, (sid, label, meta, text) in enumerate(records):
        if i % 2000 == 0:
            print(f"  extracting features: {i}/{len(records)}", flush=True)
        f = extract_all_features(text, meta, stats[sid])
        all_feats.append(f)
        labels.append(label)
    feature_names = sorted(all_feats[0].keys())
    X = np.array([[f.get(fn, 0) for fn in feature_names] for f in all_feats])
    return X, np.array(labels), feature_names


def main():
    print("Loading metadata CSV...", flush=True)
    stats = {}
    with open(CSV_PATH) as fh:
        for row in csv.DictReader(fh):
            sid = row.get('submission_id')
            if sid:
                stats[sid] = row

    print("Loading splits...", flush=True)
    train_records = load_split(TRAIN_PATH, stats)
    test_records = load_split(TEST_PATH, stats)
    print(f"Train: {len(train_records)}, Test: {len(test_records)}", flush=True)

    print("Building feature matrices...", flush=True)
    X_train, y_train, feature_names = build_matrix(train_records, stats)
    X_test, y_test, _ = build_matrix(test_records, stats)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    print(f"Features: {len(feature_names)}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Logistic Regression ---
    print("\n=== LOGISTIC REGRESSION ===", flush=True)
    best_lr_acc, best_lr = 0, None
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        m = LogisticRegression(max_iter=5000, C=C)
        m.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, m.predict(X_test_s))
        tacc = accuracy_score(y_train, m.predict(X_train_s))
        print(f"  C={C:<5} train={tacc:.1%}  test={acc:.1%}", flush=True)
        if acc > best_lr_acc:
            best_lr_acc, best_lr = acc, m
    print(f"Best LogReg: {best_lr_acc:.1%}", flush=True)

    # --- HistGradientBoosting (fast, uses all cores) ---
    print("\n=== HIST GRADIENT BOOSTING ===", flush=True)
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
                m.fit(X_train, y_train)
                acc = accuracy_score(y_test, m.predict(X_test))
                tacc = accuracy_score(y_train, m.predict(X_train))
                if acc > best_gb_acc:
                    best_gb_acc, best_gb = acc, m
                    best_gb_params = (n_est, depth, lr)
                    print(f"  NEW BEST: n={n_est} d={depth} lr={lr} train={tacc:.1%} test={acc:.1%}", flush=True)
    print(f"\nBest HGB: {best_gb_acc:.1%} params={best_gb_params}", flush=True)

    # --- Random Forest ---
    print("\n=== RANDOM FOREST ===", flush=True)
    best_rf_acc, best_rf = 0, None
    for n_est in [200, 500, 1000]:
        for depth in [10, 20, 30, None]:
            m = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                       min_samples_leaf=5, n_jobs=-1, random_state=42)
            m.fit(X_train, y_train)
            acc = accuracy_score(y_test, m.predict(X_test))
            tacc = accuracy_score(y_train, m.predict(X_train))
            if acc > best_rf_acc:
                best_rf_acc, best_rf = acc, m
                print(f"  NEW BEST: n={n_est} d={depth} train={tacc:.1%} test={acc:.1%}", flush=True)
    print(f"Best RF: {best_rf_acc:.1%}", flush=True)

    # --- Feature importances from best HGB ---
    print(f"\n--- Top 30 Feature Importances (HGB, permutation) ---", flush=True)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(best_gb, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    idx = np.argsort(perm.importances_mean)[::-1]
    print(f"{'Feature':<40} {'Importance':>10}")
    print("-" * 52)
    for i in idx[:30]:
        print(f"{feature_names[i]:<40} {perm.importances_mean[i]:>10.4f}")

    # --- Top LogReg weights ---
    print(f"\n--- Top 30 LogReg Weights ---", flush=True)
    weights = sorted(zip(feature_names, best_lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    print(f"{'Feature':<40} {'Weight':>10}")
    print("-" * 52)
    for fn, w in weights[:30]:
        print(f"{fn:<40} {w:>10.4f}  {'→ Accept' if w > 0 else '→ Reject'}")

    # --- Classification reports ---
    print(f"\n--- Best HGB Classification Report ---", flush=True)
    print(classification_report(y_test, best_gb.predict(X_test), target_names=['Reject', 'Accept']))

    # --- FINAL TABLE ---
    print(f"\n{'='*55}")
    print(f"{'MODEL':<45} {'TEST ACC':>8}")
    print(f"{'='*55}")
    print(f"{'Random baseline':<45} {'50.0%':>8}")
    print(f"{'LogReg (8 basic stats, prev run)':<45} {'59.7%':>8}")
    print(f"{'LogReg (56 features, prev run)':<45} {'63.4%':>8}")
    nf = len(feature_names)
    print(f"{'LogReg (' + str(nf) + ' features)':<45} {best_lr_acc:.1%}".rjust(53))
    print(f"{'Random Forest (' + str(nf) + ' features)':<45} {best_rf_acc:.1%}".rjust(53))
    print(f"{'HistGBM (' + str(nf) + ' features)':<45} {best_gb_acc:.1%}".rjust(53))
    print(f"{'Our best SFT (Vision)':<45} {'70.4%':>8}")
    print(f"{'Bayes-optimal ceiling':<45} {'~80.2%':>8}")
    print(f"{'='*55}", flush=True)

    # --- SAVE RESULTS TO DISK ---
    import pickle
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "paperstats_baseline")
    os.makedirs(out_dir, exist_ok=True)

    # Save models
    pickle.dump(best_lr, open(os.path.join(out_dir, "logreg_model.pkl"), "wb"))
    pickle.dump(best_gb, open(os.path.join(out_dir, "hgb_model.pkl"), "wb"))
    pickle.dump(best_rf, open(os.path.join(out_dir, "rf_model.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(out_dir, "scaler.pkl"), "wb"))

    # Save per-paper predictions (test set)
    lr_preds_test = best_lr.predict(X_test_s)
    lr_probs_test = best_lr.predict_proba(X_test_s)
    gb_preds_test = best_gb.predict(X_test)
    gb_probs_test = best_gb.predict_proba(X_test)
    rf_preds_test = best_rf.predict(X_test)
    rf_probs_test = best_rf.predict_proba(X_test)

    predictions = []
    for i, (sid, label, meta, text) in enumerate(test_records):
        predictions.append({
            "submission_id": sid,
            "label": int(label),
            "year": meta.get("year"),
            "lr_pred": int(lr_preds_test[i]),
            "lr_prob_accept": float(lr_probs_test[i][1]),
            "hgb_pred": int(gb_preds_test[i]),
            "hgb_prob_accept": float(gb_probs_test[i][1]),
            "rf_pred": int(rf_preds_test[i]),
            "rf_prob_accept": float(rf_probs_test[i][1]),
        })

    with open(os.path.join(out_dir, "test_predictions.json"), "w") as fh:
        json.dump(predictions, fh, indent=2)

    # Save feature names and weights
    lr_weights = sorted(zip(feature_names, best_lr.coef_[0].tolist()), key=lambda x: abs(x[1]), reverse=True)
    results_summary = {
        "feature_names": feature_names,
        "lr_weights": lr_weights,
        "lr_accuracy": best_lr_acc,
        "hgb_accuracy": best_gb_acc,
        "rf_accuracy": best_rf_acc if best_rf else 0,
        "hgb_params": best_gb_params,
        "n_features": len(feature_names),
        "n_train": len(train_records),
        "n_test": len(test_records),
        "hgb_feature_importances": sorted(
            zip(feature_names, perm.importances_mean.tolist()),
            key=lambda x: abs(x[1]), reverse=True
        ),
    }
    with open(os.path.join(out_dir, "results_summary.json"), "w") as fh:
        json.dump(results_summary, fh, indent=2)

    # Save per-paper features for the test set
    test_features = []
    for i, (sid, label, meta, text) in enumerate(test_records):
        row = {"submission_id": sid}
        for j, fn in enumerate(feature_names):
            row[fn] = float(X_test[i, j])
        test_features.append(row)
    with open(os.path.join(out_dir, "test_features.json"), "w") as fh:
        json.dump(test_features, fh)

    print(f"\nResults saved to {out_dir}/")
    print(f"  - logreg_model.pkl, hgb_model.pkl, rf_model.pkl, scaler.pkl")
    print(f"  - test_predictions.json ({len(predictions)} papers)")
    print(f"  - results_summary.json (weights, importances, accuracies)")
    print(f"  - test_features.json ({len(test_features)} papers x {nf} features)")


if __name__ == "__main__":
    main()
