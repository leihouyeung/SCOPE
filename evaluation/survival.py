"""Survival metrics: Harrell concordance index, log-rank test, RMST."""
from __future__ import annotations

import numpy as np


def concordance_index(times: np.ndarray, risk: np.ndarray, events: np.ndarray) -> float:
    """Harrell's concordance index.

    Falls back to a vectorised plain-NumPy implementation when `lifelines`
    is not available. Considers only comparable pairs (i, j) where the
    earlier-failure cell has its event observed.
    """
    try:
        from lifelines.utils import concordance_index as _ci
        return float(_ci(times, -risk, events))
    except Exception:
        pass
    times = np.asarray(times); risk = np.asarray(risk); events = np.asarray(events).astype(bool)
    n = len(times)
    num = den = 0.0
    for i in range(n):
        if not events[i]:
            continue
        comparable = times[i] < times
        num += float(((risk[i] > risk) & comparable).sum())
        den += float(comparable.sum())
    return num / den if den > 0 else float("nan")


def log_rank_test(time_a: np.ndarray, event_a: np.ndarray,
                  time_b: np.ndarray, event_b: np.ndarray) -> tuple[float, float]:
    """Return the log-rank chi-square statistic and p-value for two groups."""
    from lifelines.statistics import logrank_test
    res = logrank_test(time_a, time_b, event_observed_A=event_a, event_observed_B=event_b)
    return float(res.test_statistic), float(res.p_value)


def restricted_mean_survival_time(time: np.ndarray, event: np.ndarray,
                                  tau: float) -> float:
    """RMST up to horizon `tau` from a single survival sample."""
    from lifelines import KaplanMeierFitter
    km = KaplanMeierFitter()
    km.fit(time, event)
    s = km.survival_function_.iloc[:, 0]
    s_clipped = s[s.index <= tau]
    if s_clipped.empty:
        return 0.0
    deltas = np.diff(np.r_[0.0, s_clipped.index.to_numpy(), tau])
    values = np.r_[1.0, s_clipped.to_numpy()]
    return float((values * deltas).sum())
