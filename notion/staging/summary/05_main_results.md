## 5. Supervised Fine-Tuning Results

> **Test set specification.** Unless otherwise noted, all results in this section are evaluated on the v7 test set, which contains ICLR 2020+2023+2025 papers. ICLR 2025 is out-of-distribution for the base model (Qwen2.5-7B was trained on data through early 2025).

### 5.1 Setup

We fine-tune two model families:
- **Qwen2.5-7B** for text input (extracted and cleaned paper text)
- **Qwen2.5-VL-7B** for vision input (rendered PDF pages, first 8--10 pages)

Training configuration:
- **Epochs**: 4
- **Learning rate schedule**: cosine decay with warm-up, then constant
- **Batch sizes**: 16 or 32 (swept)
- **Learning rates**: 1e-6, 2e-6, 5e-6 (swept)
- **LoRA**: rank 64, alpha 128
- **Precision**: bf16

The optimal configuration is **batch size 16, learning rate 1e-6** for vision models and **batch size 32, learning rate 1e-6** for text models.

### 5.2 Modality Comparison

We compare text and vision across all dataset configurations:

| Config | Modality | Accuracy | Accept Recall | Reject Recall | Pred Accept Rate |
|--------|----------|----------|---------------|---------------|------------------|
| Trainagreeing | Vision | **70.4%** | 80.3% | 60.5% | 59.9% |
| 2024-2025 Balanced | Vision | **70.9%** | 63.2% | 78.7% | 42.3% |
| 2020-2025 Balanced | Vision | 69.8% | 68.9% | 70.8% | 49.1% |
| Trainagreeing | Text | 66.9% | 78.5% | 55.4% | 61.6% |
| 2020-2025 Balanced | Text | 66.2% | 65.2% | 67.2% | 49.0% |

*Table 1. Best configurations for Text and Vision modalities on the v7 test set.*

Vision-based input yields a consistent accuracy advantage of 3--4 pp over text across all configurations. However, the two modalities exhibit notably different recall profiles:

- **Vision** achieves higher accept recall (80.3% for trainagreeing) but lower reject recall (60.5%), predicting accept more liberally (59.9% predicted accept rate vs 50% base rate).
- **Text** achieves more balanced recall (65.2% accept, 67.2% reject for balanced), with a predicted accept rate closer to the base rate.

This asymmetry suggests that visual features (layout quality, figure density, typographic professionalism) correlate more strongly with acceptance than with rejection, while text content provides more symmetric discriminative signal.

### 5.3 Per-Year Accuracy

The v7 test set spans three years: 2020, 2023, and 2025. ICLR 2025 is particularly important as it represents a genuine OOD evaluation for the base model.

![Accuracy by year](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/accuracy_by_year.png)

**Figure 8.** Per-year accuracy for text and vision models on the v7 test set.

**Key findings.** Vision models maintain a consistent advantage across all test years. The model achieves 70.9% on 2025 papers (the 2024-2025 balanced vision configuration), demonstrating strong OOD generalization. Performance on 2020 papers (in-distribution for training) is comparable to 2025, suggesting the model learns year-invariant quality signals rather than memorizing year-specific patterns.

### 5.4 Recall Analysis

The complementary recall profiles of text and vision models suggest they capture different aspects of paper quality:

![Recall bars](../../results/summarized_investigation/text_vs_vision_v7/modality_analysis/recall_bars.png)

**Figure 9.** Accept and reject recall by modality.

**Key findings.** Vision's high accept recall (80.3%) indicates it reliably identifies papers that will be accepted---useful for positive screening. Text's more balanced recall (65.2% / 67.2%) makes it better suited for symmetric prediction tasks. The 20-point accept-reject recall gap in vision (80.3% vs 60.5%) means the trainagreeing vision model over-predicts acceptance; the balanced configuration (68.9% / 70.8%) is more symmetric but achieves slightly lower overall accuracy.
