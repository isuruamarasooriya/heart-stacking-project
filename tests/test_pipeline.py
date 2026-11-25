"""Small, fast tests for the parts that are easy to get silently wrong.
Run with:  pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src import error_analysis as ea
from src import evaluate as ev
from src import models as mdl
from src.config import load_config
from src.data import COLUMNS, make_xy, validate


def fake_heart_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A tiny table with the same columns as the real data (including a missing value)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n), "sex": rng.integers(0, 2, n), "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(95, 200, n), "chol": rng.integers(125, 400, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(80, 200, n), "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 5, n).round(1), "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n), "num": rng.integers(0, 5, n),
    })[COLUMNS]
    df.loc[0, "ca"] = np.nan  # the real data has a few missing values
    return df


def test_target_is_binary_and_matches_num_gt_0():
    df = fake_heart_frame()
    X, y = make_xy(df)
    assert set(y.unique()) <= {0, 1}
    assert ((df["num"] > 0).astype(int).to_numpy() == y.to_numpy()).all()
    assert "num" not in X.columns  # the target must not leak into the features


def test_validate_rejects_bad_target():
    df = fake_heart_frame()
    df.loc[1, "num"] = 9
    with pytest.raises(ValueError):
        validate(df)


def test_stack_has_three_base_models_and_a_meta_learner():
    cfg = load_config()
    stack = mdl.build_stacking(cfg)
    assert [name for name, _ in stack.estimators] == ["lr", "rf", "svm"]
    assert stack.final_estimator.__class__.__name__ == "LogisticRegression"


def test_every_model_handles_missing_values_and_returns_probabilities():
    cfg = load_config()
    cfg["models"]["random_forest"]["n_estimators"] = 20  # keep the test fast
    X, y = make_xy(fake_heart_frame())
    for name, model in mdl.build_all_models(cfg).items():
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2), name
        assert np.isfinite(proba).all(), name


def test_cv_splits_are_identical_for_every_call():
    X, y = make_xy(fake_heart_frame())
    a = ev.make_splits(X, y, 5, 2, seed=1)
    b = ev.make_splits(X, y, 5, 2, seed=1)
    assert all((ta == tb).all() and (va == vb).all() for (ta, va), (tb, vb) in zip(a, b))


def test_corrected_ttest_is_more_cautious_than_plain_ttest():
    from scipy import stats
    diffs = np.array([0.02, 0.03, 0.01, 0.025, 0.015, 0.02, 0.03, 0.01, 0.02, 0.015])
    _, _, p_corrected = ev.corrected_paired_ttest(diffs, n_train=200, n_test=50)
    p_plain = stats.ttest_1samp(diffs, 0).pvalue
    assert p_corrected > p_plain


def test_bootstrap_ci_contains_observed_difference_and_zero_for_identical_models():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 80)
    p = rng.uniform(size=80)
    res = ev.paired_bootstrap_diff(y, p, p, "roc_auc", n_resamples=200, seed=0)
    assert res["observed_diff"] == 0 and res["ci_low"] == 0 and res["ci_high"] == 0


def test_error_table_labels_error_types_correctly():
    X = fake_heart_frame(4).drop(columns="num")
    y = [1, 0, 1, 0]
    proba = [0.2, 0.9, 0.8, 0.1]  # miss, false alarm, correct, correct
    t = ea.build_prediction_table(X, y, proba)
    assert list(t["correct"]) == [False, False, True, True]
    assert t.loc[t.index[0], "error_type"].startswith("missed disease")
    assert t.loc[t.index[1], "error_type"].startswith("false alarm")
    assert len(ea.worst_errors(t, 10)) == 2
