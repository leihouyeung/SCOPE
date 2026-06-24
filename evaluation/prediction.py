"""Cross-modal prediction metrics: per-feature PCC and Spearman correlation."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def per_feature_correlation(predictions: np.ndarray, ground_truth: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature Pearson and Spearman correlation across cells.

    Args:
        predictions, ground_truth: (N_cells, N_features) arrays.

    Returns:
        pcc: (N_features,) Pearson correlation per feature (NaN when variance is zero).
        spearman: (N_features,) Spearman correlation per feature.
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError("predictions and ground_truth must have the same shape")
    n_features = predictions.shape[1]
    pcc = np.full(n_features, np.nan, dtype=np.float64)
    spear = np.full(n_features, np.nan, dtype=np.float64)
    for j in range(n_features):
        p, g = predictions[:, j], ground_truth[:, j]
        if p.std() > 0 and g.std() > 0:
            pcc[j] = float(np.corrcoef(p, g)[0, 1])
            spear[j] = float(spearmanr(p, g).correlation)
    return pcc, spear


def cell_level_pcc(predictions: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Per-cell Pearson correlation across features (averaged across cells = cell-level PCC)."""
    if predictions.shape != ground_truth.shape:
        raise ValueError("predictions and ground_truth must have the same shape")
    n_cells = predictions.shape[0]
    pcc = np.full(n_cells, np.nan, dtype=np.float64)
    for i in range(n_cells):
        p, g = predictions[i], ground_truth[i]
        if p.std() > 0 and g.std() > 0:
            pcc[i] = float(np.corrcoef(p, g)[0, 1])
    return pcc
