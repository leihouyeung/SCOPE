"""Clustering metrics: ARI, NMI, spatial coherence."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray,
                        ignore: str | None = None) -> float:
    """ARI computed only over cells whose reference label is not `ignore`."""
    labels_true = np.asarray(labels_true).astype(str)
    labels_pred = np.asarray(labels_pred).astype(str)
    if ignore is not None:
        mask = labels_true != ignore
        labels_true, labels_pred = labels_true[mask], labels_pred[mask]
    return float(adjusted_rand_score(labels_true, labels_pred))


def normalized_mutual_information(labels_true: np.ndarray, labels_pred: np.ndarray,
                                  ignore: str | None = None) -> float:
    labels_true = np.asarray(labels_true).astype(str)
    labels_pred = np.asarray(labels_pred).astype(str)
    if ignore is not None:
        mask = labels_true != ignore
        labels_true, labels_pred = labels_true[mask], labels_pred[mask]
    return float(normalized_mutual_info_score(labels_true, labels_pred))


def spatial_coherence(coords: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Mean fraction of each cell's k physically nearest neighbours that share
    its inferred cluster label.

    Captures how often physically adjacent cells share the predicted label,
    independent of any manual annotation.
    """
    nn = NearestNeighbors(n_neighbors=min(k + 1, coords.shape[0])).fit(coords)
    _, idx = nn.kneighbors(coords)
    # Drop self.
    if idx.shape[1] > 1:
        idx = idx[:, 1:]
    labels = np.asarray(labels)
    matches = (labels[idx] == labels[:, None]).astype(np.float32)
    return float(matches.mean())
