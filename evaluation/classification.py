"""Classification metrics for the tri-modal region/niche evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def stratified_knn_accuracy(features: np.ndarray, labels: np.ndarray,
                            k: int = 15, folds: int = 5, seed: int = 0
                            ) -> tuple[float, list[float]]:
    """Stratified k-NN cross-validation, returning balanced accuracy.

    Used to evaluate whether an embedding preserves pathology-region or
    tumour-niche identity without training a deep classifier.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    per_fold = []
    for train_idx, test_idx in skf.split(features, labels):
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        clf.fit(features[train_idx], labels[train_idx])
        pred = clf.predict(features[test_idx])
        per_fold.append(float(balanced_accuracy_score(labels[test_idx], pred)))
    return float(np.mean(per_fold)), per_fold


def macro_f1(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    return float(f1_score(labels_true, labels_pred, average="macro"))


def per_class_report(labels_true: np.ndarray, labels_pred: np.ndarray
                     ) -> dict[str, dict[str, float]]:
    """Per-class precision / recall / F1, returned as a dict."""
    return classification_report(labels_true, labels_pred, output_dict=True, zero_division=0)
