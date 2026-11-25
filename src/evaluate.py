"""Honest evaluation: repeated CV on the training set, one look at the test set,
and paired statistics to ask "is the difference real or just noise?"."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, f1_score, make_scorer, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

METRICS = ["accuracy", "roc_auc", "f1", "recall", "precision"]

# zero_division=0 stops warnings for the baseline, which may never predict "disease"
SCORING = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
}


def make_splits(X, y, n_splits: int, n_repeats: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create the CV folds ONCE, so every model is scored on exactly the same folds.
    Same folds = the per-fold scores can be compared pair by pair."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(cv.split(X, y))


def run_cv(models: dict, X, y, splits, n_jobs: int = -1) -> pd.DataFrame:
    """Score every model on every fold. Returns one row per (model, fold)."""
    rows = []
    for name, model in models.items():
        print(f"  cross-validating: {name}")
        res = cross_validate(model, X, y, cv=splits, scoring=SCORING, n_jobs=n_jobs, error_score="raise")
        for fold in range(len(splits)):
            row = {"model": name, "fold": fold}
            row.update({m: res[f"test_{m}"][fold] for m in METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_cv(cv_folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric per model (order of models kept)."""
    order = list(dict.fromkeys(cv_folds["model"]))
    g = cv_folds.groupby("model")[METRICS].agg(["mean", "std"])
    return g.loc[order]


def best_single_model(cv_folds: pd.DataFrame, single_names: list[str], primary: str) -> str:
    """Pick the best single model using CV on the TRAINING data only (never the test set)."""
    means = cv_folds[cv_folds["model"].isin(single_names)].groupby("model")[primary].mean()
    return means.idxmax()


def corrected_paired_ttest(diffs: np.ndarray, n_train: float, n_test: float) -> tuple[float, float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test.

    CV folds overlap, so their scores are not independent and an ordinary paired
    t-test gives p-values that are too small. The correction inflates the variance
    by (1/k + n_test/n_train). Returns (mean difference, t statistic, p-value).
    """
    diffs = np.asarray(diffs, dtype=float)
    k = len(diffs)
    mean = diffs.mean()
    var = diffs.var(ddof=1)
    if var == 0:
        return mean, float("nan"), float("nan")
    t = mean / np.sqrt((1.0 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return mean, t, p


def compare_in_cv(cv_folds: pd.DataFrame, splits, model_a: str, model_b: str, metric: str) -> dict:
    """Paired comparison of model_a vs model_b on identical folds."""
    a = cv_folds[cv_folds["model"] == model_a].sort_values("fold")[metric].to_numpy()
    b = cv_folds[cv_folds["model"] == model_b].sort_values("fold")[metric].to_numpy()
    diffs = a - b
    n_train = np.mean([len(tr) for tr, _ in splits])
    n_test = np.mean([len(te) for _, te in splits])
    mean, t, p = corrected_paired_ttest(diffs, n_train, n_test)
    return {
        "model_a": model_a, "model_b": model_b, "metric": metric,
        "mean_diff": mean, "std_diff": diffs.std(ddof=1),
        "a_wins": int((diffs > 0).sum()), "ties": int((diffs == 0).sum()),
        "b_wins": int((diffs < 0).sum()), "n_folds": len(diffs),
        "t": t, "p_value": p,
    }


def metric_from_proba(metric: str, y_true, proba, threshold: float = 0.5) -> float:
    """Compute one metric from predicted probabilities of the positive class."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    if metric == "roc_auc":
        return roc_auc_score(y_true, proba)
    if metric == "accuracy":
        return accuracy_score(y_true, pred)
    if metric == "f1":
        return f1_score(y_true, pred, zero_division=0)
    if metric == "recall":
        return recall_score(y_true, pred, zero_division=0)
    if metric == "precision":
        return precision_score(y_true, pred, zero_division=0)
    raise ValueError(f"Unknown metric: {metric}")


def evaluate_on_test(models: dict, X_train, y_train, X_test, y_test, threshold: float = 0.5):
    """Fit every model on the training set, score ONCE on the test set.
    Returns (results table, dict of test probabilities, dict of fitted models)."""
    rows, probas, fitted = [], {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        probas[name], fitted[name] = proba, model
        rows.append({"model": name, **{m: metric_from_proba(m, y_test, proba, threshold) for m in METRICS}})
    return pd.DataFrame(rows).set_index("model"), probas, fitted


def paired_bootstrap_diff(y_true, proba_a, proba_b, metric: str, n_resamples: int, seed: int,
                          threshold: float = 0.5) -> dict:
    """Resample the test patients with replacement; each time compute metric(A) - metric(B).
    The spread of these differences shows how uncertain the test-set comparison is."""
    y_true, proba_a, proba_b = map(np.asarray, (y_true, proba_a, proba_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:   # AUC needs both classes
            continue
        diffs.append(metric_from_proba(metric, y_true[idx], proba_a[idx], threshold)
                     - metric_from_proba(metric, y_true[idx], proba_b[idx], threshold))
    diffs = np.array(diffs)
    return {
        "observed_diff": metric_from_proba(metric, y_true, proba_a, threshold)
                         - metric_from_proba(metric, y_true, proba_b, threshold),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "share_positive": float((diffs > 0).mean()),
        "n_resamples": len(diffs),
    }
