# Predicting Paper Acceptance at Machine Learning Conferences via Fine-Tuned Large Language Models

## Table of Contents

This document is split into modular section files for easier editing and review.

| Section | File | Description |
|---------|------|-------------|
| Abstract | [00_abstract.md](00_abstract.md) | Paper abstract |
| 1. Introduction | [01_introduction.md](01_introduction.md) | Problem, prior work, contributions |
| 2. Theory | [02_theory.md](02_theory.md) | DPI, Bayes-optimal ceiling, rating separability |
| 3. Data | [03_data.md](03_data.md) | Pipeline, modalities, dataset variants, label quality |
| 4. Baselines | [04_baselines.md](04_baselines.md) | TF-IDF, LogReg on paper stats, zero-shot LLMs |
| 5. Main Results | [05_main_results.md](05_main_results.md) | SFT setup, modality comparison, per-year accuracy |
| 6. Ablations | [06_ablations.md](06_ablations.md) | Year range, trainagreeing, weight decay, train size, 2026 generalization |
| 7. Analysis | [07_analysis.md](07_analysis.md) | Confidence/calibration, coverage, agreement, categories, mediation |
| 8. Discussion | [08_discussion.md](08_discussion.md) | Limitations, applications, ethics, related work, conclusion |
| 9. Appendix | [09_appendix.md](09_appendix.md) | Full tables, zero-shot details, weight decay, paper stats |
| 10. Agent Benchmark | [10_agent_benchmark.md](10_agent_benchmark.md) | Codex agent as reviewer: strategy, rubric, 63% accuracy, comparison to SFT |

## Key Numbers

| Metric | Value |
|--------|-------|
| Best accuracy (v7 test, ICLR 2025 OOD) | **70.4%** (Vision, Trainagreeing) |
| Bayes-optimal ceiling | ~80.2% |
| % of ceiling captured | ~88% |
| Best classical ML baseline | 65.0% (Random Forest, 187 features) |
| Zero-shot LLM | 52.1% (Qwen 3.5-122B) |
| Selective prediction (50% coverage) | ~77% |
| 2026 generalization | 68.4% (Vision) |
| Dataset size | ~25,000 papers |

## Modalities

**Two modalities only** (Text+Images removed throughout):
- **Text**: Qwen2.5-7B on extracted clean text
- **Vision**: Qwen2.5-VL-7B on rendered PDF pages

## What Was Removed (vs previous draft)

- All Text+Images modality rows
- All RL results (SFT only)
- Fano's inequality section
- pct_rating violin / Mann-Whitney plot
- Cross-year reviewer variance comparisons (per-year only)
- "Vision > Text" ranking language (replaced with nuanced comparison)
