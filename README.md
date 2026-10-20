# Polypharmacy readmission ranking (UCI Diabetes 130-US hospitals, id 296)

Question: can a class-balanced, CV-tuned LightGBM with polypharmacy-count and
co-prescription features rank older patients on five or more concurrent drugs by
30-day readmission risk better than a standard neural-network scorer (MLP)?

## Running

    pip install -r requirements.txt
    python analysis.py

Writes `results.json` and `figures/pr_auc_comparison.png`.

## Method

- Cohort: age >= 60 and `num_medications` >= 5 (concurrent drugs).
- Stratified 70/30 split, seed 42. Test set is used once, for the final summaries.
- Baseline: standardised MLPClassifier on one-hot medication levels, seeds 42/43/44.
- Improved: class-balanced LightGBM, hyper-parameters chosen by 3-fold CV on the
  training set only, with drug count and top-10 co-prescription pair features,
  isotonic calibration, seeds 42/43/44.
- Metric: PR-AUC on the held-out test split (ROC-AUC, Brier and F1 also reported).

## Caveats

- Rows are encounters, not patients; the split is not patient-grouped.
- The cohort filter uses `num_medications`. Counting only the 23 drug columns gave
  just 38 eligible older encounters, which is too few to evaluate.
- Only medication features are used, so absolute performance is limited.

## Results (latest run)

On the UCI Diabetes 130-US hospitals cohort (age 60+, five or more drugs), a standardised MLP baseline reached PR-AUC 0.130 versus 0.132 for my class-balanced, CV-tuned LightGBM with drug-count features, a small gain.

| Metric | Value |
|---|---|
| baseline_score | 0.1296 |
| improved_score | 0.1324 |
| improvement | 0.00283 |
| baseline_score_std | 0.001487 |
| improved_score_std | 0.001087 |
| baseline_roc_auc | 0.5336 |
| improved_roc_auc | 0.548 |
| baseline_brier | 0.1034 |
| improved_brier | 0.103 |
| brier_improvement | 0.0004145 |
| baseline_f1 | 0.2071 |
| improved_f1 | 0.2075 |
| best_cv_pr_auc | 0.1265 |
| n_pairs | 10 |
| n_test_positives | 2348 |
| runtime_seconds | 18.27 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Rows are encounters; patient_nbr was not de-duplicated, so the split is not patient-grouped and may be optimistic. Concurrent drug count uses the num_medications column: counting only the 23 drug columns gave an active-drug count of >=5 for just 38 older encounters, too few to evaluate, so the cohort filter was changed to num_medications >= 5 (the 23 drug columns still feed the one-hot and co-prescription features). Only medication features are used (no diagnosis or lab features), so absolute performance is limited. The baseline MLP was not tuned. F1 uses the fixed training positive rate as threshold. Pair features are chosen by co-prescription frequency on training data. Best LightGBM params: {'reg_lambda': 1.0, 'num_leaves': 31, 'n_estimators': 200, 'min_child_samples': 20, 'learning_rat

## Reference

This project is motivated by the following paper:

> A Larouche, A Durand, R Khoury, C Sirois (2023). *Neural bandits for data mining: searching for dangerous polypharmacy*. International Workshop on. https://link.springer.com/chapter/10.1007/978-3-031-36938-4_5

Paper author: Prof. Audrey Durand, Université Laval (audrey.durand@ift.ulaval.ca)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
