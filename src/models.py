"""All models used in the experiment.

Every model is a scikit-learn Pipeline, so imputing and scaling are learned on
the training part of each split only. This prevents data leakage.
"""
from __future__ import annotations

from collections import OrderedDict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]  # codes, not quantities -> one-hot
BINARY = ["sex", "fbs", "exang"]                  # already 0/1

# Display names used in tables and plots
BASELINE = "Majority-class baseline"
LR, RF, SVM = "Logistic Regression", "Random Forest", "SVM (RBF)"
VOTING = "Soft voting (3 models)"
STACK = "Stacking (3 models)"
SINGLE_MODELS = [LR, RF, SVM]


def build_preprocessor() -> ColumnTransformer:
    """Impute missing values, scale numbers, one-hot encode categories."""
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
             CATEGORICAL),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
        ]
    )


def _pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def build_base_models(cfg: dict) -> "OrderedDict[str, Pipeline]":
    """The three diverse base learners: linear, tree-ensemble, kernel."""
    seed = cfg["seed"]
    m = cfg["models"]
    return OrderedDict(
        [
            (LR, _pipeline(LogisticRegression(C=m["logistic_regression"]["C"],
                                              max_iter=m["logistic_regression"]["max_iter"],
                                              random_state=seed))),
            (RF, _pipeline(RandomForestClassifier(n_estimators=m["random_forest"]["n_estimators"],
                                                  min_samples_leaf=m["random_forest"]["min_samples_leaf"],
                                                  random_state=seed, n_jobs=1))),
            # SVC(probability=True) is deprecated in scikit-learn 1.9 (removed in 1.11).
            # The recommended replacement is Platt scaling (sigmoid) fitted with
            # internal CV: ensemble=False -> one final SVC + one sigmoid calibrator.
            (SVM, _pipeline(CalibratedClassifierCV(
                SVC(C=m["svm"]["C"], kernel=m["svm"]["kernel"], random_state=seed),
                method="sigmoid", ensemble=False, cv=5))),
        ]
    )


def _estimator_list(base_models) -> list[tuple[str, Pipeline]]:
    """sklearn ensembles want short names: lr, rf, svm."""
    short = {LR: "lr", RF: "rf", SVM: "svm"}
    return [(short[name], pipe) for name, pipe in base_models.items()]


def build_voting(cfg: dict) -> VotingClassifier:
    """Simple ensemble: average the predicted probabilities of the 3 models."""
    return VotingClassifier(_estimator_list(build_base_models(cfg)), voting="soft")


def build_stacking(cfg: dict) -> StackingClassifier:
    """Stacking: a logistic-regression meta-learner learns how to combine the 3 models.

    The meta-learner is trained on OUT-OF-FOLD probabilities (inner CV), so it
    never sees predictions a base model made on its own training rows.
    """
    seed = cfg["seed"]
    inner_cv = StratifiedKFold(n_splits=cfg["stacking"]["inner_cv_folds"], shuffle=True, random_state=seed)
    return StackingClassifier(
        estimators=_estimator_list(build_base_models(cfg)),
        final_estimator=LogisticRegression(C=cfg["stacking"]["final_estimator_C"], random_state=seed),
        stack_method="predict_proba",
        cv=inner_cv,
        n_jobs=1,
    )


def build_all_models(cfg: dict) -> "OrderedDict[str, object]":
    """Everything that gets compared, in the order it appears in tables."""
    models = OrderedDict()
    models[BASELINE] = _pipeline(DummyClassifier(strategy="most_frequent"))
    models.update(build_base_models(cfg))
    models[VOTING] = build_voting(cfg)
    models[STACK] = build_stacking(cfg)
    return models
