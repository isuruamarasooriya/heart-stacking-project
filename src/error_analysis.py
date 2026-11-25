"""Error analysis: look at the patients the model gets wrong.

A test set of ~61 patients may contain fewer than 10 mistakes, so we ALSO collect
out-of-fold (OOF) predictions on the training set. Every training patient is
predicted by a model that never saw them, which gives ~40 honest mistakes to study.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import CP_LABELS


def oof_probabilities(model, X, y, n_splits: int, seed: int) -> np.ndarray:
    """Probability of disease for each patient, predicted by a model trained without them."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba", n_jobs=-1)
    return proba[:, 1]


def build_prediction_table(X: pd.DataFrame, y, proba, threshold: float = 0.5) -> pd.DataFrame:
    """One row per patient: features + truth + prediction + what kind of error (if any)."""
    df = X.copy()
    df["y_true"] = np.asarray(y)
    df["proba_disease"] = np.round(np.asarray(proba), 3)
    df["y_pred"] = (df["proba_disease"] >= threshold).astype(int)
    df["correct"] = df["y_true"] == df["y_pred"]
    df["error_type"] = np.select(
        [df["correct"],
         (df["y_true"] == 1) & (df["y_pred"] == 0)],
        ["", "missed disease (false negative)"],
        default="false alarm (false positive)",
    )
    # How sure was the model about its WRONG answer? (0.5 = unsure, 1.0 = certain)
    df["confidence_in_wrong_answer"] = np.where(
        df["correct"], np.nan,
        np.where(df["y_pred"] == 1, df["proba_disease"], 1 - df["proba_disease"]),
    ).round(3)
    df["cp_label"] = df["cp"].map(CP_LABELS)
    return df


def worst_errors(table: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n most confidently wrong predictions (the most instructive ones)."""
    wrong = table[~table["correct"]]
    return wrong.sort_values("confidence_in_wrong_answer", ascending=False).head(n)


def error_rate_by_group(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Is the model worse for some kinds of patients?"""
    t = table.copy()
    t["age_band"] = pd.cut(t["age"], [0, 45, 55, 65, 120], labels=["<=45", "46-55", "56-65", ">65"])
    t["sex_label"] = t["sex"].map({0: "female", 1: "male"})
    out = {}
    for col in ["sex_label", "age_band", "cp_label"]:
        g = t.groupby(col, observed=True)["correct"].agg(n="size", errors=lambda s: int((~s).sum()))
        g["error_rate"] = (g["errors"] / g["n"]).round(3)
        out[col] = g.reset_index()
    return out


def confusion_counts(table: pd.DataFrame) -> dict[str, int]:
    return {
        "true_negative": int(((table.y_true == 0) & (table.y_pred == 0)).sum()),
        "false_positive": int(((table.y_true == 0) & (table.y_pred == 1)).sum()),
        "false_negative": int(((table.y_true == 1) & (table.y_pred == 0)).sum()),
        "true_positive": int(((table.y_true == 1) & (table.y_pred == 1)).sum()),
    }


def shared_errors(table_a: pd.DataFrame, table_b: pd.DataFrame) -> dict[str, int]:
    """Do two models fail on the same patients? (Same row order is required.)
    Little overlap = the models make different mistakes = stacking has room to help."""
    wrong_a = ~table_a["correct"].to_numpy()
    wrong_b = ~table_b["correct"].to_numpy()
    return {
        "wrong_in_both": int((wrong_a & wrong_b).sum()),
        "only_a_wrong": int((wrong_a & ~wrong_b).sum()),
        "only_b_wrong": int((~wrong_a & wrong_b).sum()),
    }


def interpretation_markdown(oof: pd.DataFrame, groups: dict, conf: dict, overlap: dict,
                            best_name: str, stack_name: str, verdict: str,
                            cv_p: float, boot: dict, n_worst: int = 10, min_group: int = 20) -> str:
    """Write answers to the five interpretation questions, using ONLY numbers from this run."""
    n, wrong = len(oof), oof[~oof["correct"]]
    n_wrong = len(wrong)
    fn, fp = conf["false_negative"], conf["false_positive"]
    rate = n_wrong / n

    # 1. type of mistake
    if abs(fn - fp) <= max(2, 0.15 * n_wrong):
        worse = "split roughly evenly between missed disease and false alarms"
    else:
        worse = "mostly missed disease" if fn > fp else "mostly false alarms"
    a1 = (f"1. **Type of mistake.** The stacking model made {n_wrong} out-of-fold mistakes on {n} training patients "
          f"({rate:.1%}): {fn} missed-disease cases (false negatives) and {fp} false alarms (false positives), "
          f"so the mistakes are {worse}. A missed case is the more serious error for a doctor, because a sick "
          f"patient would be sent home untreated, whereas a false alarm usually only leads to extra tests.")

    # 2. confidently wrong patients
    w = worst_errors(oof, n_worst)
    top_cp = w["cp_label"].value_counts()
    top_type = w["error_type"].value_counts()
    a2 = (f"2. **Most confidently wrong patients.** Among the {len(w)} most confident mistakes "
          f"(model sure at {w['confidence_in_wrong_answer'].min():.2f}-{w['confidence_in_wrong_answer'].max():.2f}), "
          f"{top_type.iloc[0]} are '{top_type.index[0]}'. The commonest chest-pain type is "
          f"'{top_cp.index[0]}' ({top_cp.iloc[0]} of {len(w)}); {int((w['sex'] == 1).sum())} are male, and their median age is "
          f"{w['age'].median():.0f} (all patients: {oof['age'].median():.0f}). "
          + ("The model was strongly convinced of its wrong answer for these patients, so they are worth "
             "inspecting by hand." if w["confidence_in_wrong_answer"].median() >= 0.7 else
             "The model was only moderately sure about them, so these are borderline cases rather than clear failures."))

    # 3. groups
    cand = []
    for key, label in [("sex_label", "sex"), ("age_band", "age band"), ("cp_label", "chest-pain type")]:
        for _, r in groups[key].iterrows():
            if r["n"] >= min_group:
                cand.append((r["error_rate"], label, str(r[key]), int(r["n"]), int(r["errors"])))
    if cand:
        er, label, name, gn, ge = max(cand)
        excess = ge - gn * rate
        z = excess / max((gn * rate * (1 - rate)) ** 0.5, 1e-9)   # rough binomial z-score
        caution = (f"that is about {excess:.0f} more mistakes than expected (z = {z:.1f}), which is "
                   + ("not clearly beyond chance, especially because several groups were compared"
                      if z < 2 else
                      "larger than chance alone would usually give, though several groups were compared "
                      "and the group is small"))
        a3 = (f"3. **Groups.** Looking only at groups with at least {min_group} patients, the highest error rate is for "
              f"{label} '{name}': {er:.1%} ({ge} of {gn}) versus {rate:.1%} overall; {caution}. "
              f"Smaller groups are not ranked because their rates are too unstable to trust.")
    else:
        a3 = "3. **Groups.** No group reached the minimum size, so group error rates are not reliable."

    # 4. shared errors
    both, only_s, only_b = overlap["wrong_in_both"], overlap["only_a_wrong"], overlap["only_b_wrong"]
    best_wrong = both + only_b
    share = both / best_wrong if best_wrong else float("nan")
    room = ("the models mostly fail on the same patients, which leaves stacking little room to help"
            if share >= 0.7 else
            "the models partly fail on different patients, which leaves some room for stacking to help")
    a4 = (f"4. **Shared failures.** {both} patients are wrong for both stacking and {best_name}; {only_s} are wrong only for "
          f"stacking and {only_b} only for {best_name}. About {share:.0%} of {best_name}'s mistakes are also made by "
          f"stacking, so {room}.")

    # 5. verdict
    reliable = (cv_p < 0.05) and not (boot["ci_low"] <= 0 <= boot["ci_high"])
    if reliable:
        tail = "The gain is statistically reliable, so the extra cost of stacking may be justified."
    else:
        tail = (f"The difference is too small to be told apart from noise, so the simpler {best_name} is the better "
                f"practical choice: it is cheaper to train and easier to explain.")
    a5 = f"5. **Did stacking beat the best single model?** {verdict} {tail}"
    return "\n\n".join([a1, a2, a3, a4, a5])
