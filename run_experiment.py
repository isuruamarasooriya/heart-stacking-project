"""Project 5 - Stacking and ensembling on UCI Heart Disease (Cleveland).

QUESTION: Does a stacking ensemble of Logistic Regression, Random Forest and an
SVM predict heart disease better than the best of those three models alone?

Run:  python run_experiment.py
All settings are in config.yaml. Everything is written to results/.
"""
from __future__ import annotations

import platform
import random
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src import error_analysis as ea
from src import evaluate as ev
from src import models as mdl
from src import plots
from src.config import PROJECT_ROOT, load_config, resolve
from src.data import load_dataset
from src.report import (INTERP_END, INTERP_START, cv_table_md, df_to_md, update_readme_results,
                        verdict_text)

QUESTION = ("Does a stacking ensemble of Logistic Regression, Random Forest and an SVM predict heart disease "
            "better than the best of those three models used alone?")


def main(config_path=None) -> None:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    primary = cfg["metrics"]["primary"]
    thr = cfg["metrics"]["threshold"]

    out = resolve(cfg["output"]["dir"])
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1. data
    print("1/7 Loading data ...")
    X, y = load_dataset(cfg, resolve(cfg["data"]["raw_path"]))
    data_summary = pd.DataFrame({
        "patients": [len(X)],
        "no disease": [int((y == 0).sum())],
        "disease": [int((y == 1).sum())],
        "share with disease": [round(float(y.mean()), 3)],
        "values missing": [int(X.isna().sum().sum())],
    })
    X.isna().sum().loc[lambda s: s > 0].to_csv(out / "missing_values.csv", header=["n_missing"])

    # ------------------------------------------------- 2. split (test set locked away)
    print("2/7 Splitting: the test set stays untouched until step 5 ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], stratify=y, random_state=seed)

    # ------------------------------------------------------- 3. cross-validation
    print("3/7 Cross-validating all models on the training set ...")
    all_models = mdl.build_all_models(cfg)
    splits = ev.make_splits(X_train, y_train, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)
    cv_folds = ev.run_cv(all_models, X_train, y_train, splits)
    cv_summary = ev.summarise_cv(cv_folds)
    cv_folds.to_csv(out / "cv_fold_scores.csv", index=False)
    cv_summary.to_csv(out / "cv_summary.csv")

    best_single = ev.best_single_model(cv_folds, mdl.SINGLE_MODELS, primary)  # chosen WITHOUT the test set
    print(f"    best single model by CV {primary}: {best_single}")
    cmp_rows = [ev.compare_in_cv(cv_folds, splits, mdl.STACK, best_single, m) for m in ev.METRICS]
    cmp_rows.append(ev.compare_in_cv(cv_folds, splits, mdl.STACK, mdl.VOTING, primary))
    comparison = pd.DataFrame(cmp_rows)
    comparison.to_csv(out / "cv_paired_comparison.csv", index=False)
    primary_cmp = cmp_rows[ev.METRICS.index(primary)]

    plots.plot_cv_boxplot(cv_folds, primary, list(all_models.keys()), out / "figures" / "cv_boxplot.png")

    # ------------------------------------------------------ 4. fit + 5. test set
    print("4/7 Fitting every model on the full training set ...")
    print("5/7 Scoring ONCE on the held-out test set ...")
    test_results, probas, fitted = ev.evaluate_on_test(all_models, X_train, y_train, X_test, y_test, thr)
    test_results.to_csv(out / "test_results.csv")

    boot = ev.paired_bootstrap_diff(y_test, probas[mdl.STACK], probas[best_single], primary,
                                    cfg["bootstrap"]["n_resamples"], seed, thr)
    pd.DataFrame([boot]).to_csv(out / "test_bootstrap_stack_vs_best.csv", index=False)

    plots.plot_roc(y_test, probas, mdl.SINGLE_MODELS + [mdl.STACK], out / "figures" / "roc_test.png")
    plots.plot_confusion(y_test, probas[mdl.STACK], f"{mdl.STACK} - test set", out / "figures" / "confusion_stack_test.png", thr)
    stack_fitted = fitted[mdl.STACK]
    meta_w = stack_fitted.final_estimator_.coef_[0]
    meta_table = pd.DataFrame({"base model": mdl.SINGLE_MODELS, "meta-learner coefficient": meta_w})
    meta_table.to_csv(out / "stacking_meta_weights.csv", index=False)
    plots.plot_meta_weights(mdl.SINGLE_MODELS, meta_w, out / "figures" / "meta_weights.png")

    joblib.dump(stack_fitted, resolve(cfg["output"]["model_path"]))

    # --------------------------------------------------------- 6. error analysis
    print("6/7 Error analysis ...")
    n_show = cfg["error_analysis"]["n_errors_to_show"]
    oof_stack = ea.build_prediction_table(X_train, y_train, ea.oof_probabilities(
        all_models[mdl.STACK], X_train, y_train, cfg["cv"]["n_splits"], seed), thr)
    oof_best = ea.build_prediction_table(X_train, y_train, ea.oof_probabilities(
        all_models[best_single], X_train, y_train, cfg["cv"]["n_splits"], seed), thr)
    test_stack = ea.build_prediction_table(X_test, y_test, probas[mdl.STACK], thr)

    oof_stack[~oof_stack["correct"]].to_csv(out / "errors_stack_out_of_fold.csv")
    test_stack[~test_stack["correct"]].to_csv(out / "errors_stack_test.csv")
    n_wrong_oof, n_wrong_test = int((~oof_stack["correct"]).sum()), int((~test_stack["correct"]).sum())

    show_cols = ["age", "sex", "cp_label", "trestbps", "chol", "thalach", "exang", "oldpeak", "ca", "thal",
                 "y_true", "proba_disease", "error_type", "confidence_in_wrong_answer"]
    worst = ea.worst_errors(oof_stack, n_show)[show_cols].reset_index(names="patient_row")
    groups = ea.error_rate_by_group(oof_stack)
    conf_oof = ea.confusion_counts(oof_stack)
    overlap = ea.shared_errors(oof_stack, oof_best)
    verdict = verdict_text(primary_cmp, boot, primary, best_single, mdl.STACK)
    interpretation = ea.interpretation_markdown(oof_stack, groups, conf_oof, overlap, best_single, mdl.STACK,
                                                verdict, primary_cmp["p_value"], boot)

    err_md = [
        "# Error analysis - Stacking model\n",
        f"Out-of-fold predictions on the {len(oof_stack)} training patients: **{n_wrong_oof} mistakes**"
        f" (each patient predicted by a model that never saw them). Held-out test set: **{n_wrong_test} mistakes**"
        f" out of {len(test_stack)}.\n",
        ("**Check:** at least 10 wrong predictions analysed - " + ("YES." if n_wrong_oof >= 10 else "NO, fewer than 10 exist; say so in the README.") + "\n"),
        "## Types of mistake (out-of-fold)\n",
        df_to_md(pd.DataFrame([conf_oof])),
        f"\n## The {len(worst)} most confidently wrong predictions (out-of-fold)\n",
        df_to_md(worst),
        "\n## Error rate by patient group (out-of-fold)\n",
    ]
    for key, label in [("sex_label", "Sex"), ("age_band", "Age band"), ("cp_label", "Chest pain type")]:
        err_md += [f"**{label}**\n", df_to_md(groups[key]), ""]
    err_md += [
        f"## Do stacking and {best_single} fail on the same patients? (out-of-fold)\n",
        df_to_md(pd.DataFrame([{"wrong in both": overlap["wrong_in_both"],
                                "only stacking wrong": overlap["only_a_wrong"],
                                f"only {best_single} wrong": overlap["only_b_wrong"]}])),
        "\n## Test-set mistakes\n",
        (df_to_md(ea.worst_errors(test_stack, len(test_stack))[show_cols].reset_index(names="patient_row"))
         if n_wrong_test else "None."),
        "\n## Interpretation (auto-written from the numbers above)\n",
        interpretation,
    ]
    (out / "error_analysis.md").write_text("\n".join(err_md), encoding="utf-8")

    # ------------------------------------------------------------ 7. write summary
    print("7/7 Writing summary ...")
    test_tbl = test_results.reset_index().rename(columns={"model": "Model"})
    cmp_show = comparison.copy()
    cmp_show["comparison"] = cmp_show["model_a"] + " minus " + cmp_show["model_b"]
    cmp_show = cmp_show[["comparison", "metric", "mean_diff", "std_diff", "a_wins", "b_wins", "ties", "p_value"]]

    results_md = "\n".join([
        f"_Auto-generated by `run_experiment.py` (seed {seed}, primary metric **{primary}**)._\n",
        "**Data**\n", df_to_md(data_summary), "",
        f"**Cross-validation on the training set** ({len(X_train)} patients, "
        f"{cfg['cv']['n_splits']}-fold x {cfg['cv']['n_repeats']} repeats, same folds for every model; mean ± std)\n",
        cv_table_md(cv_summary, ev.METRICS), "",
        f"Best single model chosen by CV {primary} (no test data used): **{best_single}**\n",
        "**Paired comparison on identical folds** (Nadeau-Bengio corrected t-test; positive = first model better)\n",
        df_to_md(cmp_show, 4), "",
        f"**Held-out test set** ({len(X_test)} patients, scored once)\n",
        df_to_md(test_tbl), "",
        "**Verdict**\n", verdict, "",
        "**What the meta-learner learned**\n", df_to_md(meta_table), "",
        "Figures: `results/figures/cv_boxplot.png`, `roc_test.png`, `confusion_stack_test.png`, `meta_weights.png`. "
        "Full error analysis: `results/error_analysis.md`.",
    ])
    (out / "summary.md").write_text("# Results summary\n\n" + results_md + "\n", encoding="utf-8")
    if update_readme_results(PROJECT_ROOT / "README.md", results_md):
        print("    README.md results section updated.")
    if update_readme_results(PROJECT_ROOT / "README.md", interpretation, INTERP_START, INTERP_END):
        print("    README.md interpretation section updated.")

    (out / "environment.txt").write_text(
        f"python {platform.python_version()}\nnumpy {np.__version__}\npandas {pd.__version__}\n"
        f"scikit-learn {sklearn.__version__}\nseed {seed}\n", encoding="utf-8")

    print("\nDONE.\n" + verdict)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
