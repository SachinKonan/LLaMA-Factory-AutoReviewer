## 8. Discussion

### 8.1 Fundamental vs Modeling Limitations

Our analysis distinguishes two sources of prediction error:

1. **Fundamental limitations.** The data processing inequality guarantees that bypassing the review chain loses information. Our Bayesian analysis of the NeurIPS 2021 consistency experiment establishes a Bayes-optimal ceiling of ~80% on balanced sets. Our best model (70.4%) captures ~88% of this ceiling, with a 9.8 pp gap remaining.

2. **Modeling limitations.** The 9.8 pp gap between our best model and the Bayes-optimal ceiling reflects both irreducible information loss from bypassing the review chain and addressable modeling limitations:
   - Limited context window (current models see at most 8--10 pages of the PDF)
   - No multi-paper comparison (reviewers implicitly benchmark papers against each other)
   - No awareness of submission pool composition (acceptance rates vary by year and area)
   - Single-paper inference (no access to concurrent submission quality distribution)

We cannot currently decompose the 9.8 pp gap into its fundamental and modeling components. However, the convergence of text and vision accuracy at low coverage (~84% at 25% coverage, exceeding the 80.2% ceiling on the retained subset) suggests that for the most confident predictions, both modalities capture nearly all available signal.

### 8.2 SFT vs Paper Statistics: Complementarity

Our correlation analysis (Section 7) reveals that SFT predictions and paper-statistics-based predictions are only moderately correlated, with an agreement rate of ~67%. The oracle ensemble ceiling (accuracy if either model is correct) reaches ~81%, substantially above either individual model, confirming that SFT captures semantic signals orthogonal to surface statistics.

The feature correlations with SFT correctness are uniformly weak (max |r| < 0.08), meaning no single paper statistic reliably predicts when the SFT model will succeed or fail. This supports the interpretation that the SFT model's advantage comes from holistic content understanding rather than any individual structural feature.

### 8.3 Practical Applications via Selective Prediction

Our confidence analysis reveals that the model can achieve much higher accuracy by abstaining on uncertain predictions:

- **Desk reject screening**: At 78.7% reject recall (2024-2025 balanced vision), the model identifies the majority of papers that will be rejected, potentially reducing reviewer workload.
- **Confidence-weighted triage**: Area chairs could prioritize review of papers where the model is uncertain, allocating reviewer effort where it matters most.
- **Pre-submission quality feedback**: Model confidence alongside predictions gives authors calibrated quality estimates before submission.

### 8.4 Ethical Considerations

Automated acceptance prediction raises several concerns:
- **Bias amplification**: If the model learns biases in historical decisions (topic bias, methodology bias), it perpetuates them.
- **Gaming**: If prediction signals are known, authors could optimize for predicted acceptance rather than genuine contribution.
- **Fairness**: Content-only prediction may disadvantage papers from non-traditional research paradigms.
- **Transparency**: Any deployment should be transparent about limitations and the role of human judgment.

We emphasize that our system is designed as a research tool for understanding the peer review process, not as a replacement for human reviewers.

### 8.5 Related Work

**Peer review prediction.** PeerRead (Kang et al., 2018) achieved ~65% accuracy on ICLR 2017 using hand-crafted features. MOPRD (Gao et al., 2019) focused on multi-outcome prediction. DeepReviewer (2023) and PaperDecision (2024) use LLMs for review generation but achieve near-random prediction accuracy (~52--55%), consistent with our zero-shot findings.

**LLMs for scientific text.** Large language models have been applied to scientific text for summarization (SciBERT, SPECTER), claim verification (SciFact), and review generation (ReviewerGPT). Our work differs by focusing on binary classification with systematic modality comparison.

**Vision-language models.** Recent VLMs (Qwen-VL, InternVL, GPT-4V) show strong document understanding performance. We apply VLMs to a novel domain---scientific paper quality assessment---where layout and figures carry complementary information to text.

### 8.6 Conclusion

We present a comprehensive study of paper acceptance prediction at ICLR using fine-tuned LLMs, yielding five key findings:

1. **Fine-tuning dramatically improves over zero-shot prediction.** Our best model (70.4%) outperforms zero-shot Qwen 3.5-122B (52.1%) by 18.3 pp, demonstrating that venue-specific calibration is essential.

2. **Vision-based input provides a consistent advantage.** Vision models outperform text by 3--4 pp across configurations, with complementary recall profiles (higher accept recall, lower reject recall) suggesting the two modalities capture different quality signals.

3. **Content-only prediction captures ~88% of the Bayes-optimal ceiling.** The 9.8 pp gap between 70.4% and the theoretical 80.2% ceiling reflects both information loss from bypassing the review chain and modeling limitations.

4. **2026 presents a measurably harder target.** Compressed rating gaps (0.39 vs 0.43) and noisier labels (21.8% noisy accepts) explain the 2--4 pp accuracy drop on 2026 data.

5. **Selective prediction enables practical deployment.** At 50% coverage, accuracy reaches ~77%; at 25% coverage, ~84%.

**Limitations.** Our study is limited to ICLR and may not generalize to other venues. The binary framing ignores review score structure. Models are constrained by context window (8--10 pages for vision, ~32K tokens for text).

**Future work.** Promising directions include: (1) longer-context models processing full papers with appendices, (2) multi-venue training, (3) ensemble methods formally combining text and vision predictions, and (4) incorporating auxiliary metadata signals.
