"""Figures saved as PNG files (no pop-up windows, so it runs anywhere)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, not to the screen
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve


def plot_cv_boxplot(cv_folds, metric: str, order: list[str], path: Path) -> None:
    """Spread of the 25 CV scores per model."""
    data = [cv_folds.loc[cv_folds["model"] == m, metric].to_numpy() for m in order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(data, vert=False, tick_labels=order)
    ax.set_xlabel(f"{metric} (higher is better)")
    ax.set_title(f"Cross-validated {metric} on the training set (5-fold x 5 repeats)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_test, probas: dict, names: list[str], path: Path) -> None:
    """ROC curves of the chosen models on the held-out test set."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name in names:
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, probas[name]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, proba, title: str, path: Path, threshold: float = 0.5) -> None:
    pred = (np.asarray(proba) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, display_labels=["no disease", "disease"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_meta_weights(names: list[str], weights, path: Path) -> None:
    """How much does the stacking meta-learner trust each base model?"""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(names, weights, color="steelblue")
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Meta-learner coefficient (larger = trusted more)")
    ax.set_title("What the stacking meta-learner learned")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
