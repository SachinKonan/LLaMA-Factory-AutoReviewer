# 7B Ratio Cross-Eval — Consolidated Report

**Scope:** 4 model configs (text/vision × 50/50/30/70 train) × ICLR + arxiv × balanced/natural test priors × test+val splits.

- ICLR test+val are filtered to year ∈ {2025, 2026}.
- Arxiv test+val are y24up (conference_year ≥ 2024).
- For ICLR "natural" prior: test ratio = 30/70 (matches ICLR's ~30% accept rate).
- For arxiv "natural" prior: per-conference natural acceptance rates (natrate; openaccept.org 5-year averages).
- Score: log-odds = lp_accept[k] - lp_reject[k] at the decision step.
- Calibration: τ* learned on val.
  - ICLR: single τ* per (modality, train, test_prior) cell.
  - Arxiv: per-(modality, train, eval_set, venue) τ* (sparse venues, val n<10, fall back to model-global τ*).

## Headline

- **AUC**: vision 50/50 is the most *consistent* recipe (≈0.72-0.74 across all 4 test cells). On the matched-prior ICLR-natural (test=30/70) cell, *30/70-trained* models hit a much higher AUC (~0.80) for both modalities — suggests test-population drift between the balanced and 30/70 ICLR test sets, not just a prior difference.
- **ACC** (depends on test prior): on natural-prior tests (ICLR 30/70, arxiv natrate), the matched-prior train recipe wins raw. On balanced tests, calibration converges all 4 (modality × ratio) cells to a tighter 64-68% band.
- **Reject-recall** is the diagnostic for bias: 30/70-trained models on balanced tests have extreme reject-recall (~80-100%) at τ=0; calibration trades reject-recall for accept-recall to recover ACC — the biggest single calibration lift is **text 30/70 on arxiv balanced: +13.1pp**.

## Figures

![ACC raw vs calibrated](../tmp_latex_dir/figures/ratio_xeval_acc_grid.png)
![AUC](../tmp_latex_dir/figures/ratio_xeval_auc_grid.png)
![Recall scatter (raw → calibrated)](../tmp_latex_dir/figures/ratio_xeval_recall_scatter.png)

## Metrics — TEST split

### ICLR balanced (test=50/50, year=25/26)

| model | n | ACC raw | ACC cal | AUC | AccR raw / RejR raw | AccR cal / RejR cal |
|---|---:|---:|---:|---:|---:|---:|
| text 50/50 | 1667 | 65.2 | 66.1 | 0.721 | 56/74 | 79/54 |
| text 30/70 | 1667 | 61.5 | 67.1 | 0.720 | 40/83 | 81/53 |
| vision 50/50 | 1670 | 67.6 | 67.5 | 0.736 | 65/70 | 80/55 |
| vision 30/70 | 1670 | 64.9 | 65.5 | 0.723 | 55/75 | 63/69 |

### ICLR natural  (test=30/70, year=25/26)

| model | n | ACC raw | ACC cal | AUC | AccR raw / RejR raw | AccR cal / RejR cal |
|---|---:|---:|---:|---:|---:|---:|
| text 50/50 | 1586 | 65.6 | 68.7 | 0.697 | 60/68 | 31/86 |
| text 30/70 | 1594 | 73.7 | 73.5 | 0.805 | 42/88 | 32/93 |
| vision 50/50 | 1594 | 66.2 | 70.1 | 0.725 | 66/66 | 33/87 |
| vision 30/70 | 1594 | 73.9 | 73.2 | 0.804 | 57/82 | 27/94 |

### Arxiv balanced (50/50 y24up)

| model | n | ACC raw | ACC cal | AUC | AccR raw / RejR raw | AccR cal / RejR cal |
|---|---:|---:|---:|---:|---:|---:|
| text 50/50 | 1414 | 59.5 | 66.2 | 0.720 | 24/93 | 72/60 |
| text 30/70 | 1415 | 51.8 | 64.9 | 0.697 | 0/100 | 78/53 |
| vision 50/50 | 1414 | 63.4 | 65.9 | 0.724 | 45/80 | 67/65 |
| vision 30/70 | 1414 | 62.6 | 63.7 | 0.704 | 48/76 | 74/54 |

### Arxiv natural  (natrate y24up)

| model | n | ACC raw | ACC cal | AUC | AccR raw / RejR raw | AccR cal / RejR cal |
|---|---:|---:|---:|---:|---:|---:|
| text 50/50 | 1430 | 76.7 | 76.1 | 0.725 | 25/93 | 24/92 |
| text 30/70 | 1431 | 76.3 | 76.2 | 0.700 | 0/100 | 22/93 |
| vision 50/50 | 1468 | 72.0 | 73.1 | 0.736 | 46/81 | 19/91 |
| vision 30/70 | 1468 | 70.0 | 73.9 | 0.714 | 49/77 | 18/92 |

## Metrics — VAL split

### ICLR balanced (test=50/50, year=25/26)

| model | n | ACC raw | ACC cal | AUC | AccR raw / RejR raw | AccR cal / RejR cal |
|---|---:|---:|---:|---:|---:|---:|
| text 50/50 | 844 | 64.9 | 66.2 | 0.717 | 56/74 | 80/53 |
| text 30/70 | 844 | 62.6 | 65.8 | 0.712 | 42/83 | 78/54 |
| vision 50/50 | 845 | 65.1 | 65.9 | 0.721 | 63/67 | 78/54 |
| vision 30/70 | 845 | 66.6 | 67.8 | 0.720 | 56/78 | 65/71 |

### ICLR natural  (test=30/70, year=25/26)

| model | n | ACC raw | ACC cal | AUC | AccR raw / RejR raw | AccR cal / RejR cal |
|---|---:|---:|---:|---:|---:|---:|
| text 50/50 | 600 | 67.8 | 73.3 | 0.715 | 53/74 | 31/91 |
| text 30/70 | 600 | 70.5 | 72.2 | 0.702 | 40/83 | 30/90 |
| vision 50/50 | 606 | 68.0 | 73.9 | 0.749 | 69/67 | 37/90 |
| vision 30/70 | 606 | 71.0 | 73.3 | 0.738 | 56/78 | 28/93 |

### Arxiv balanced (50/50 y24up)

| model | n | ACC raw | ACC cal | AUC | AccR raw / RejR raw | AccR cal / RejR cal |
|---|---:|---:|---:|---:|---:|---:|
| text 50/50 | 722 | 57.1 | 67.7 | 0.682 | 19/92 | 72/64 |
| text 30/70 | 722 | 52.1 | 65.5 | 0.656 | 0/100 | 76/56 |
| vision 50/50 | 719 | 63.7 | 68.7 | 0.708 | 43/82 | 70/68 |
| vision 30/70 | 719 | 62.3 | 68.3 | 0.691 | 48/75 | 77/61 |

### Arxiv natural  (natrate y24up)

| model | n | ACC raw | ACC cal | AUC | AccR raw / RejR raw | AccR cal / RejR cal |
|---|---:|---:|---:|---:|---:|---:|
| text 50/50 | 732 | 77.0 | 80.2 | 0.702 | 22/93 | 27/96 |
| text 30/70 | 732 | 77.5 | 79.4 | 0.670 | 0/100 | 22/96 |
| vision 50/50 | 729 | 73.7 | 79.7 | 0.712 | 48/81 | 26/96 |
| vision 30/70 | 729 | 69.0 | 79.7 | 0.684 | 50/75 | 25/96 |

## Thresholds used (τ*)

### ICLR (single τ* per cell, learned on val year≥2025)

| modality | train | test prior | τ* |
|---|---|---|---:|
| text | 50/50 | balanced | -0.374 |
| text | 50/50 | natural | +0.499 |
| text | 30/70 | balanced | -0.873 |
| text | 30/70 | natural | +0.249 |
| vision | 50/50 | balanced | -0.499 |
| vision | 50/50 | natural | +0.874 |
| vision | 30/70 | balanced | -0.124 |
| vision | 30/70 | natural | +0.749 |

### Arxiv (per-(modality, train, eval_set, venue) τ*; sparse venues use the model-global τ* marked `g`)

| modality | train | eval | venue | τ* | source |
|---|---|---|---|---:|:---|
| text | 50/50 | balanced | _GLOBAL_ | -1.374 | (used as fallback for sparse venues) |
| text | 50/50 | balanced | aaai | -2.249 | venue |
| text | 50/50 | balanced | acl_family | -1.249 | venue |
| text | 50/50 | balanced | aistats | -1.374 | global |
| text | 50/50 | balanced | colm | -1.374 | venue |
| text | 50/50 | balanced | corl | -1.374 | global |
| text | 50/50 | balanced | cvpr | -0.874 | venue |
| text | 50/50 | balanced | eccv | -1.249 | venue |
| text | 50/50 | balanced | iccv | -1.624 | venue |
| text | 50/50 | balanced | iclr | -0.374 | venue |
| text | 50/50 | balanced | icml | -0.624 | venue |
| text | 50/50 | balanced | neurips | -1.374 | venue |

| text | 50/50 | natural | _GLOBAL_ | +0.499 | (used as fallback for sparse venues) |
| text | 50/50 | natural | aaai | -0.001 | venue |
| text | 50/50 | natural | acl_family | -0.374 | venue |
| text | 50/50 | natural | aistats | +0.499 | global |
| text | 50/50 | natural | colm | -1.374 | venue |
| text | 50/50 | natural | corl | +0.499 | global |
| text | 50/50 | natural | cvpr | +0.499 | venue |
| text | 50/50 | natural | eccv | -0.999 | venue |
| text | 50/50 | natural | iccv | +0.499 | venue |
| text | 50/50 | natural | iclr | -0.001 | venue |
| text | 50/50 | natural | icml | -0.624 | venue |
| text | 50/50 | natural | neurips | +0.499 | venue |

| text | 30/70 | balanced | _GLOBAL_ | -4.248 | (used as fallback for sparse venues) |
| text | 30/70 | balanced | aaai | -5.744 | venue |
| text | 30/70 | balanced | acl_family | -5.246 | venue |
| text | 30/70 | balanced | aistats | -4.248 | global |
| text | 30/70 | balanced | colm | -5.744 | venue |
| text | 30/70 | balanced | corl | -4.248 | global |
| text | 30/70 | balanced | cvpr | -4.123 | venue |
| text | 30/70 | balanced | eccv | -6.119 | venue |
| text | 30/70 | balanced | iccv | -3.748 | venue |
| text | 30/70 | balanced | iclr | -1.999 | venue |
| text | 30/70 | balanced | icml | -3.622 | venue |
| text | 30/70 | balanced | neurips | -3.747 | venue |

| text | 30/70 | natural | _GLOBAL_ | -1.124 | (used as fallback for sparse venues) |
| text | 30/70 | natural | aaai | -1.624 | venue |
| text | 30/70 | natural | acl_family | -2.124 | venue |
| text | 30/70 | natural | aistats | -1.124 | global |
| text | 30/70 | natural | colm | -3.873 | venue |
| text | 30/70 | natural | corl | -1.124 | global |
| text | 30/70 | natural | cvpr | -1.249 | venue |
| text | 30/70 | natural | eccv | -0.374 | venue |
| text | 30/70 | natural | iccv | -0.999 | venue |
| text | 30/70 | natural | iclr | -1.999 | venue |
| text | 30/70 | natural | icml | -2.373 | venue |
| text | 30/70 | natural | neurips | -1.249 | venue |

| vision | 50/50 | balanced | _GLOBAL_ | -1.124 | (used as fallback for sparse venues) |
| vision | 50/50 | balanced | aaai | -1.499 | venue |
| vision | 50/50 | balanced | acl_family | -1.124 | venue |
| vision | 50/50 | balanced | aistats | -1.124 | global |
| vision | 50/50 | balanced | colm | -1.999 | venue |
| vision | 50/50 | balanced | corl | -1.124 | global |
| vision | 50/50 | balanced | cvpr | -0.625 | venue |
| vision | 50/50 | balanced | eccv | -0.375 | venue |
| vision | 50/50 | balanced | iccv | -0.250 | venue |
| vision | 50/50 | balanced | iclr | -0.124 | venue |
| vision | 50/50 | balanced | icml | -1.500 | venue |
| vision | 50/50 | balanced | neurips | +0.124 | venue |

| vision | 50/50 | natural | _GLOBAL_ | +1.124 | (used as fallback for sparse venues) |
| vision | 50/50 | natural | aaai | -0.000 | venue |
| vision | 50/50 | natural | acl_family | +0.374 | venue |
| vision | 50/50 | natural | aistats | +1.124 | global |
| vision | 50/50 | natural | colm | -1.624 | venue |
| vision | 50/50 | natural | corl | +1.124 | global |
| vision | 50/50 | natural | cvpr | +1.124 | venue |
| vision | 50/50 | natural | eccv | +0.874 | venue |
| vision | 50/50 | natural | iccv | -0.250 | venue |
| vision | 50/50 | natural | iclr | +0.874 | venue |
| vision | 50/50 | natural | icml | +0.249 | venue |
| vision | 50/50 | natural | neurips | +1.873 | venue |

| vision | 30/70 | balanced | _GLOBAL_ | -0.249 | (used as fallback for sparse venues) |
| vision | 30/70 | balanced | aaai | -0.999 | venue |
| vision | 30/70 | balanced | acl_family | -0.749 | venue |
| vision | 30/70 | balanced | aistats | -0.249 | global |
| vision | 30/70 | balanced | colm | -1.374 | venue |
| vision | 30/70 | balanced | corl | -0.249 | global |
| vision | 30/70 | balanced | cvpr | -0.125 | venue |
| vision | 30/70 | balanced | eccv | -0.375 | venue |
| vision | 30/70 | balanced | iccv | +0.250 | venue |
| vision | 30/70 | balanced | iclr | +0.249 | venue |
| vision | 30/70 | balanced | icml | -0.874 | venue |
| vision | 30/70 | balanced | neurips | -0.749 | venue |

| vision | 30/70 | natural | _GLOBAL_ | +1.249 | (used as fallback for sparse venues) |
| vision | 30/70 | natural | aaai | +0.125 | venue |
| vision | 30/70 | natural | acl_family | +0.874 | venue |
| vision | 30/70 | natural | aistats | +1.249 | global |
| vision | 30/70 | natural | colm | +0.125 | venue |
| vision | 30/70 | natural | corl | +1.249 | global |
| vision | 30/70 | natural | cvpr | +1.124 | venue |
| vision | 30/70 | natural | eccv | +0.499 | venue |
| vision | 30/70 | natural | iccv | +0.250 | venue |
| vision | 30/70 | natural | iclr | +0.749 | venue |
| vision | 30/70 | natural | icml | +0.125 | venue |
| vision | 30/70 | natural | neurips | +0.999 | venue |


## Notes / caveats

- `accept-recall` = TP / total_positives (sensitivity); `reject-recall` = TN / total_negatives (specificity).
- `ACC raw` uses τ=0 (the natural decision boundary on the log-odds score).
- For ICLR, calibrated metrics use the single τ* from val matching the test prior.
- For arxiv, calibrated metrics use the per-venue τ*, with the model-global τ* as fallback for venues with val n<10. The pooled accept/reject recalls are weighted by venue test counts.
- All numbers are 7B (Qwen2.5-7B for text, Qwen2.5-VL-7B for vision).