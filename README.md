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

## Reference

This project is motivated by the following paper:

> A Larouche, A Durand, R Khoury, C Sirois (2023). *Neural bandits for data mining: searching for dangerous polypharmacy*. International Workshop on. https://link.springer.com/chapter/10.1007/978-3-031-36938-4_5

Paper author: Prof. Audrey Durand, Université Laval (audrey.durand@ift.ulaval.ca)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
