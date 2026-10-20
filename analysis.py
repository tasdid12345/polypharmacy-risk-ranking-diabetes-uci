"""Polypharmacy readmission ranking on UCI Diabetes 130-US hospitals (id 296).

Baseline: standardised MLP on one-hot medication features (stand-in for a neural scorer).
Improved: class-balanced, CV-tuned LightGBM with drug-count and pairwise co-prescription
indicators, plus isotonic calibration. The test split is used once, at the end."""
import json
import os
import socket
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, f1_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]                 # both models are run with these seeds
N_PAIRS = 10                         # co-prescription pairs kept (ranked on training data only)
MIN_AGE, MIN_DRUGS = 60, 5           # cohort: older patients on >= 5 concurrent drugs
LEVELS = ["Steady", "Up", "Down"]    # 'No' is the reference level
MEDS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
        "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
URL = "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"

socket.setdefaulttimeout(60)         # 60 s network timeout for all downloads (ucimlrepo uses requests)


def download():
    """Fetch dataset 296 with retries and exponential backoff (covers HTTP 429/5xx failures)."""
    for attempt in range(5):
        try:
            return fetch_ucirepo(id=296)
        except Exception as exc:
            wait = 2 ** attempt
            print(f"Download attempt {attempt + 1} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise SystemExit("ERROR: could not download UCI dataset 296")


def evaluate(p, y, thr):
    """Main metric PR-AUC plus ROC-AUC, Brier score and F1 at a fixed threshold."""
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "f1": float(f1_score(y, p >= thr))}


def summarise(res, key):
    v = np.array([r[key] for r in res])
    return float(v.mean()), float(v.std())


def main():
    t0 = time.time()
    print("Loading UCI dataset 296 via ucimlrepo ...")
    ds = download()
    feats = ds.data.features
    tgt = ds.data.targets
    tgt = tgt.iloc[:, 0] if isinstance(tgt, pd.DataFrame) else tgt

    # Age lower bound from brackets like "[60-70)"; medication levels: No/Steady/Up/Down
    age = feats["age"].astype(str).str.extract(r"\[(\d+)")[0].astype(float).values
    M = feats[MEDS].astype(str).apply(lambda s: s.str.strip())
    A_all = (M != "No").values.astype(int)           # active-drug indicators (23 drug columns)
    # Concurrent drug count: the dataset's num_medications column (the 23 drug columns alone
    # record only ~1-3 active drugs per encounter, which yields a cohort far too small to evaluate).
    n_drugs_all = pd.to_numeric(feats["num_medications"], errors="coerce").fillna(0).values.astype(int)
    keep = (age >= MIN_AGE) & (n_drugs_all >= MIN_DRUGS)
    M, A, n_drugs = M[keep].reset_index(drop=True), A_all[keep], n_drugs_all[keep]
    # Target: readmitted within 30 days ('<30')
    y = (tgt.astype(str).str.strip().values[keep] == "<30").astype(int)
    print(f"Cohort (age >= {MIN_AGE}, >= {MIN_DRUGS} concurrent drugs): {len(y)} rows, "
          f"positive rate {y.mean():.3f}")
    if len(y) < 1000 or y.sum() < 50:
        raise SystemExit("ERROR: cohort too small for meaningful metrics")

    # Single stratified 70/30 split with fixed seed
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=SEED)
    y_tr, y_te = y[tr], y[te]
    thr = float(y_tr.mean())            # F1 threshold = training positive rate (not tuned on test)
    print(f"Split: train={len(tr)}, test={len(te)}, test positives={int(y_te.sum())}")

    # Baseline features: one-hot medication levels
    OH = np.column_stack([(M[m] == lv).values for m in MEDS for lv in LEVELS]).astype(float)
    # Improved features: same one-hot + drug count + top co-prescribed pairs (chosen on TRAIN only)
    iu, ju = np.triu_indices(len(MEDS), 1)
    co = A[tr].T @ A[tr]
    top = np.argsort(-co[iu, ju])[:N_PAIRS]
    pairs = list(zip(iu[top], ju[top]))
    PAIR = np.column_stack([A[:, i] * A[:, j] for i, j in pairs])
    XI = np.column_stack([OH, n_drugs, PAIR]).astype(float)

    # ---- Baseline: standardised MLP, three seeds ----
    print("Baseline: standardised MLPClassifier on one-hot medication features")
    base_res = []
    for s in SEEDS:
        mlp = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300,
            early_stopping=True, random_state=s))
        mlp.fit(OH[tr], y_tr)
        base_res.append(evaluate(mlp.predict_proba(OH[te])[:, 1], y_te, thr))
        print(f"  seed {s}: test PR-AUC {base_res[-1]['pr_auc']:.4f}")

    # ---- Improved: class-balanced LightGBM, CV-tuned on train only, isotonic calibration ----
    print("Improved: CV-tuning class-balanced LightGBM on training data only ...")
    search = RandomizedSearchCV(
        LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=2, verbose=-1),
        param_distributions={"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.1],
                             "n_estimators": [200, 400], "min_child_samples": [20, 50],
                             "reg_lambda": [0.0, 1.0]},
        n_iter=12, scoring="average_precision",
        cv=StratifiedKFold(3, shuffle=True, random_state=SEED), random_state=SEED, n_jobs=1)
    search.fit(XI[tr], y_tr)
    print(f"  best CV PR-AUC {search.best_score_:.4f} with {search.best_params_}")

    imp_res = []
    for s in SEEDS:
        est = clone(search.best_estimator_).set_params(random_state=s)
        cal = CalibratedClassifierCV(est, method="isotonic",
                                     cv=StratifiedKFold(3, shuffle=True, random_state=s))
        cal.fit(XI[tr], y_tr)
        imp_res.append(evaluate(cal.predict_proba(XI[te])[:, 1], y_te, thr))
        print(f"  seed {s}: test PR-AUC {imp_res[-1]['pr_auc']:.4f}")

    # ---- Summaries (test set used only here) ----
    b_mean, b_std = summarise(base_res, "pr_auc")
    i_mean, i_std = summarise(imp_res, "pr_auc")
    b_roc, _ = summarise(base_res, "roc_auc")
    i_roc, _ = summarise(imp_res, "roc_auc")
    b_brier, _ = summarise(base_res, "brier")
    i_brier, _ = summarise(imp_res, "brier")
    b_f1, _ = summarise(base_res, "f1")
    i_f1, _ = summarise(imp_res, "f1")
    print(f"PR-AUC baseline {b_mean:.4f} +/- {b_std:.4f} vs improved {i_mean:.4f} +/- {i_std:.4f}")

    # Bar chart of main metric
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Baseline MLP", "Improved LightGBM"], [b_mean, i_mean], yerr=[b_std, i_std],
           capsize=6, color=["gray", "steelblue"])
    ax.set_ylabel("Test PR-AUC (mean +/- std over seeds)")
    ax.set_title("30-day readmission, older patients on 5+ drugs")
    fig.tight_layout()
    fig.savefig("figures/pr_auc_comparison.png", dpi=120)
    plt.close(fig)

    results = {
        "question": "Can a class-balanced, CV-tuned LightGBM with explicit polypharmacy-count features "
                    "rank older patients on five or more concurrent drugs by 30-day readmission risk "
                    "better than a standard neural-network scorer?",
        "method": "Stratified 70/30 split (seed 42) of UCI 296 cohort (age >= 60, >= 5 concurrent drugs "
                  "from num_medications). Baseline: standardised MLP on one-hot medication features. "
                  "Improved: class-balanced LightGBM tuned by 3-fold CV on training data, with drug count "
                  "and top pairwise co-prescription indicators, isotonic calibration. Seeds 42, 43, 44 "
                  "for both models.",
        "dataset": {
            "name": "Diabetes 130-US hospitals for years 1999-2008 (UCI id 296), filtered cohort",
            "source_url": URL,
            "n_rows": int(len(y)),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "PR-AUC (average precision) on held-out test split",
        "higher_is_better": True,
        "baseline": "Standardised scikit-learn MLPClassifier (64-32 hidden units, early stopping) on "
                    "one-hot medication features; untuned standard configuration.",
        "improved": "Class-balanced LightGBM with CV-tuned hyper-parameters (RandomizedSearchCV, "
                    "3-fold, PR-AUC), added drug-count and top co-prescription pair features, "
                    "and isotonic calibration.",
        "metrics": {
            "baseline_score": b_mean,
            "improved_score": i_mean,
            "improvement": i_mean - b_mean,
            "baseline_score_std": b_std,
            "improved_score_std": i_std,
            "baseline_roc_auc": b_roc,
            "improved_roc_auc": i_roc,
            "baseline_brier": b_brier,
            "improved_brier": i_brier,
            "brier_improvement": b_brier - i_brier,
            "baseline_f1": b_f1,
            "improved_f1": i_f1,
            "best_cv_pr_auc": float(search.best_score_),
            "n_pairs": N_PAIRS,
            "n_test_positives": int(y_te.sum()),
            "runtime_seconds": time.time() - t0,
        },
        "notes": "Rows are encounters; patient_nbr was not de-duplicated, so the split is not "
                 "patient-grouped and may be optimistic. Concurrent drug count uses the num_medications "
                 "column: counting only the 23 drug columns gave an active-drug count of >=5 for just 38 "
                 "older encounters, too few to evaluate, so the cohort filter was changed to "
                 "num_medications >= 5 (the 23 drug columns still feed the one-hot and co-prescription "
                 "features). Only medication features are used (no diagnosis or lab features), so absolute "
                 "performance is limited. The baseline MLP was not tuned. F1 uses the fixed training "
                 "positive rate as threshold. Pair features are chosen by co-prescription frequency on "
                 f"training data. Best LightGBM params: {search.best_params_}. The paper's neural-bandit "
                 "model is not reproduced; the MLP is a stand-in scorer. ucimlrepo exposes no timeout, so "
                 "a socket-level 60 s timeout is used.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
