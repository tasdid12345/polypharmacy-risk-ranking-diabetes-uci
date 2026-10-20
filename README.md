# Polypharmacy readmission ranking: LightGBM vs neural scorer

Motivation: The paper "Neural bandits for data mining: searching for dangerous polypharmacy" (2023) frames the search for dangerous multi-drug combinations in hospital data as a learning problem. This experiment asks whether a simpler, class-balanced gradient-boosted model with explicit polypharmacy features can rank readmission risk better than a standard neural-network scorer on a public proxy dataset.

Question: Can a class-balanced, CV-tuned LightGBM with explicit polypharmacy-count features rank older patients on five or more concurrent drugs by 30-day readmission risk better than a standard neural-network scorer?

Dataset: UCI Diabetes 130-US hospitals for years 1999-2008 (UCI id 296), loaded with the ucimlrepo package. The cohort is restricted to older patients on five or more active medication columns. The target is readmission within thirty days. This is not the paper's own dataset.

Baseline: a standardised scikit-learn MLPClassifier on one-hot medication features, standing in for the paper's neural scorer.

Improved model: a class-balanced LightGBM tuned by cross-validation on the training split. It adds the drug count and selected pairwise co-prescription indicators, and uses isotonic calibration.

Evaluation: a single stratified 70/30 split with seed 42. The main metric is PR-AUC on the test split; ROC-AUC, Brier score and F1 are secondary.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to results.json (charts go to figures/).

## Reference

This project is motivated by the following paper:

> A Larouche, A Durand, R Khoury, C Sirois (2023). *Neural bandits for data mining: searching for dangerous polypharmacy*. International Workshop on. https://link.springer.com/chapter/10.1007/978-3-031-36938-4_5

Paper author: Prof. Audrey Durand, Université Laval (audrey.durand@ift.ulaval.ca)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
