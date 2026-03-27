## 3. Data Pipeline and Dataset Construction

### 3.1 Source Data

We collect papers from ICLR 2020, 2021, 2022, 2023, 2025, and 2026 via the OpenReview API. **We exclude ICLR 2024** because the 2024 review data was not fully publicly available with consistent decision labels at the time of dataset construction. For each paper, we obtain:
- The submitted PDF
- The final accept/reject decision
- Reviewer ratings and confidence scores (used for analysis, not as model input)
- Paper metadata (title, abstract, categories)

### 3.2 Text Extraction: MinerU Pipeline

We use MinerU, a high-fidelity PDF-to-text conversion tool, to extract clean text from each paper PDF. The pipeline performs:
1. **Layout analysis**: Identifies text blocks, figures, tables, equations, and captions
2. **OCR fallback**: For scanned or image-heavy PDFs
3. **Reference removal**: We strip the references/bibliography section to prevent the model from exploiting citation patterns as a prestige signal
4. **Author de-identification**: Author names, affiliations, and acknowledgment sections are removed

For the **vision modality**, we render each PDF page as an image and provide the first $k$ pages (typically 8--10) as visual input to the VLM. This preserves layout, figures, equations, and typographic quality that text extraction may lose.

### 3.3 Input Modalities

We evaluate two input representations:

| Modality | Model | Description |
|----------|-------|-------------|
| **Text** | Qwen2.5-7B | Extracted clean text from MinerU (no figures) |
| **Vision** | Qwen2.5-VL-7B | Rendered PDF pages as images (full visual input) |

### 3.4 Dataset Composition

<!-- TODO: Insert dataset composition bar charts from scripts/plot_dataset_composition.py -->
<!-- ![Dataset composition](../../figures/dataset_composition.png) -->

**Dataset sizes (v7 splits, Text and Vision modalities):**

| Dataset Variant | Modality | Train | Val | Test | Best Acc |
|----------------|----------|-------|-----|------|----------|
| 2020-2025 Balanced | Text | 12,745 | ~1,000 | 2,024 | 66.2% |
| 2020-2025 Balanced | Vision | 12,765 | ~1,000 | 2,026 | 69.8% |
| 2017-2025 Balanced | Text | 14,577 | ~1,100 | 2,234 | 65.7% |
| 2017-2025 Balanced | Vision | 14,594 | ~1,100 | 2,236 | 69.5% |
| 2024-2025 Balanced | Text | 3,889 | ~500 | 1,189 | 63.8% |
| 2024-2025 Balanced | Vision | 3,893 | ~500 | 1,190 | 70.9% |

*All accuracies are measured on the v7 test set.*

**Trainagreeing dataset** (filtered training set, identical val/test):

| Modality | Train | Val | Test | Best Acc |
|----------|-------|-----|------|----------|
| Text | 8,296 | ~1,000 | 2,024 | 66.9% |
| Vision | 8,315 | ~1,000 | 2,026 | 70.4% |

The **trainagreeing** dataset is constructed by filtering the balanced training set to retain only examples where early-checkpoint predictions agree with the ground truth label. This produces a smaller but cleaner training set (~35% fewer examples).

![Year distribution in our dataset](../../figures/latex/data/year_distribution.png)

**Figure 3.** Distribution of papers by year across train/val/test splits.

**Key findings.** The dataset is heavily weighted toward recent years (2025 and 2026 contribute the most papers), reflecting the rapid growth in ICLR submissions. The 85/5/10 train/val/test split ensures sufficient test set size for reliable evaluation.

### 3.5 Label Quality

Not all labels are equally reliable. We measure label "cleanliness" using pct_rating as a proxy: accepted papers with pct_rating >= 0.6 and rejected papers with pct_rating <= 0.4 are considered "clean" labels (the decision aligns with the aggregate reviewer sentiment).

| Year | % Accepts w/ pct >= 0.6 | % Rejects w/ pct <= 0.4 |
|------|------------------------|------------------------|
| 2020 | 95.7% | 60.9% |
| 2021 | 91.5% | 64.6% |
| 2022 | 87.2% | 62.4% |
| 2023 | 89.6% | 72.1% |
| 2025 | 82.6% | 67.0% |
| 2026 | 78.2% | 60.7% |

For 2026, 21.8% of accepted papers have pct_rating < 0.6, meaning over one-fifth of accepted papers were accepted *despite* receiving below-threshold aggregate ratings. This high label noise directly impacts model training and evaluation.

![Label cleanliness across years](../../figures/report_label_cleanliness.png)

**Figure 4.** Label cleanliness by year: percentage of accepted papers with above-threshold ratings and rejected papers with below-threshold ratings.

**Key findings.** Label quality deteriorates for more recent years, with 2026 showing the lowest accept cleanliness (78.2%). This trend is consistent with increasing submission volume placing strain on the review process, and contributes to the lower model accuracy on 2026 data.

### 3.6 Comparison to Prior Datasets

| Dataset | Papers | Venues | Years | Content | Labels | Anonymized |
|---------|--------|--------|-------|---------|--------|------------|
| PeerRead | ~14K | ACL, NIPS, ICLR | 2013-2017 | Abstract only | Accept/Reject | No |
| MOPRD | ~6K | ICLR | 2017-2020 | Full text | Multi-class | No |
| **Ours** | **~25K** | **ICLR** | **2020-2026** | **Full text + vision** | **Binary** | **Yes** |

Our dataset is approximately 2x larger than PeerRead and 4x larger than MOPRD, covers more recent years (including 2025 and 2026), provides both text and vision modalities, and enforces author anonymization.
