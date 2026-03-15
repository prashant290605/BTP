"""
Reusable evaluation metrics for early warning systems.

Metrics:
- Lead Time
- False Alarm Rate
- Regime Separation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def _to_2d(array_like: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    """
    Normalize inputs to shape (n_series, time_steps).
    """
    arr = np.asarray(array_like, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError("Expected 1D or 2D array-like scores.")


def _normalize_collapse_times(
    collapse_times: int | Sequence[int],
    n_series: int,
) -> np.ndarray:
    """
    Normalize collapse times to length n_series.
    """
    if np.isscalar(collapse_times):
        return np.full(n_series, int(collapse_times), dtype=int)
    ct = np.asarray(collapse_times, dtype=int).reshape(-1)
    if ct.size != n_series:
        raise ValueError("collapse_times length must match number of score series.")
    return ct


def _first_persistent_alarm(
    scores: np.ndarray,
    threshold: float,
    min_persistence: int = 1,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> Optional[int]:
    """
    Return first index where score >= threshold for min_persistence steps.
    """
    if end_index is None:
        end_index = scores.size
    if min_persistence < 1:
        raise ValueError("min_persistence must be >= 1.")

    run_length = 0
    run_start = None
    for t in range(start_index, min(end_index, scores.size)):
        s = scores[t]
        if np.isfinite(s) and s >= threshold:
            if run_length == 0:
                run_start = t
            run_length += 1
            if run_length >= min_persistence:
                return run_start
        else:
            run_length = 0
            run_start = None
    return None


def compute_lead_time(
    scores: np.ndarray,
    collapse_time: int,
    threshold: float = 0.5,
    min_persistence: int = 1,
    start_index: int = 0,
) -> float:
    """
    Lead time = collapse_time - first_alarm_time (alarm before collapse only).
    Returns np.nan if no valid pre-collapse alarm is found.
    """
    t_alarm = _first_persistent_alarm(
        scores=scores,
        threshold=threshold,
        min_persistence=min_persistence,
        start_index=start_index,
        end_index=collapse_time,
    )
    if t_alarm is None:
        return float("nan")
    return float(collapse_time - t_alarm)


def compute_false_alarm_rate(
    scores: np.ndarray,
    collapse_time: int,
    threshold: float = 0.5,
    warning_horizon: int = 200,
    min_persistence: int = 1,
    start_index: int = 0,
) -> float:
    """
    False Alarm Rate (FAR) computed over the pre-warning region.

    Pre-warning region:
        [start_index, collapse_time - warning_horizon)

    FAR definition:
        FAR = (# alarm points in pre-warning region) / (# points in pre-warning region)
    """
    pre_warning_end = max(start_index, collapse_time - warning_horizon)
    region = scores[start_index:pre_warning_end]
    if region.size == 0:
        return 0.0

    if min_persistence <= 1:
        alarms = np.isfinite(region) & (region >= threshold)
        return float(np.mean(alarms))

    # Persistence-aware FAR: count points belonging to persistent alarm runs.
    alarm_points = np.zeros(region.size, dtype=bool)
    run_length = 0
    run_start = None
    for i, s in enumerate(region):
        if np.isfinite(s) and s >= threshold:
            if run_length == 0:
                run_start = i
            run_length += 1
        else:
            if run_length >= min_persistence and run_start is not None:
                alarm_points[run_start:i] = True
            run_length = 0
            run_start = None

    if run_length >= min_persistence and run_start is not None:
        alarm_points[run_start:region.size] = True

    return float(np.mean(alarm_points))


def compute_regime_separation(
    scores: np.ndarray,
    collapse_time: int,
    pre_collapse_window: int = 400,
    start_index: int = 0,
    epsilon: float = 1e-8,
) -> float:
    """
    Regime separation between stable and pre-collapse windows using effect size:

        separation = (mu_precollapse - mu_stable) / pooled_std

    where:
      stable window      = [start_index, collapse_time - pre_collapse_window)
      pre-collapse window = [collapse_time - pre_collapse_window, collapse_time)

    Returns 0.0 when one of the windows is empty.
    """
    split = max(start_index, collapse_time - pre_collapse_window)
    stable = scores[start_index:split]
    pre = scores[split:collapse_time]

    stable = stable[np.isfinite(stable)]
    pre = pre[np.isfinite(pre)]

    if stable.size == 0 or pre.size == 0:
        return 0.0

    mu_s = float(np.mean(stable))
    mu_p = float(np.mean(pre))
    var_s = float(np.var(stable))
    var_p = float(np.var(pre))
    pooled_var = 0.5 * (var_s + var_p)
    if pooled_var < epsilon:
        # Degenerate case: both windows nearly constant.
        # Fall back to raw mean difference to avoid exploding scores.
        return float(mu_p - mu_s)

    pooled_std = np.sqrt(pooled_var)
    return float((mu_p - mu_s) / pooled_std)


def evaluate_ews_series(
    scores: np.ndarray,
    collapse_time: int,
    threshold: float = 0.5,
    warning_horizon: int = 200,
    pre_collapse_window: int = 400,
    min_persistence: int = 1,
    start_index: int = 0,
) -> Dict[str, float]:
    """
    Evaluate one score time series.
    """
    return {
        "lead_time": compute_lead_time(
            scores=scores,
            collapse_time=collapse_time,
            threshold=threshold,
            min_persistence=min_persistence,
            start_index=start_index,
        ),
        "false_alarm_rate": compute_false_alarm_rate(
            scores=scores,
            collapse_time=collapse_time,
            threshold=threshold,
            warning_horizon=warning_horizon,
            min_persistence=min_persistence,
            start_index=start_index,
        ),
        "regime_separation": compute_regime_separation(
            scores=scores,
            collapse_time=collapse_time,
            pre_collapse_window=pre_collapse_window,
            start_index=start_index,
        ),
    }


def evaluate_ews_batch(
    scores: np.ndarray | Sequence[np.ndarray],
    collapse_times: int | Sequence[int],
    threshold: float = 0.5,
    warning_horizon: int = 200,
    pre_collapse_window: int = 400,
    min_persistence: int = 1,
    start_index: int = 0,
) -> Dict[str, object]:
    """
    Evaluate a batch of score time series and return per-series and aggregate metrics.
    """
    scores_2d = _to_2d(scores)
    collapse_arr = _normalize_collapse_times(collapse_times, scores_2d.shape[0])

    per_series: List[Dict[str, float]] = []
    for i in range(scores_2d.shape[0]):
        ct = int(np.clip(collapse_arr[i], 1, scores_2d.shape[1]))
        metrics = evaluate_ews_series(
            scores=scores_2d[i],
            collapse_time=ct,
            threshold=threshold,
            warning_horizon=warning_horizon,
            pre_collapse_window=pre_collapse_window,
            min_persistence=min_persistence,
            start_index=start_index,
        )
        per_series.append(metrics)

    lead_times = np.array([m["lead_time"] for m in per_series], dtype=float)
    fars = np.array([m["false_alarm_rate"] for m in per_series], dtype=float)
    seps = np.array([m["regime_separation"] for m in per_series], dtype=float)

    aggregate = {
        "lead_time_mean": float(np.nanmean(lead_times)) if np.any(np.isfinite(lead_times)) else float("nan"),
        "lead_time_std": float(np.nanstd(lead_times)) if np.any(np.isfinite(lead_times)) else float("nan"),
        "detection_rate": float(np.mean(np.isfinite(lead_times))),
        "false_alarm_rate_mean": float(np.mean(fars)),
        "false_alarm_rate_std": float(np.std(fars)),
        "regime_separation_mean": float(np.mean(seps)),
        "regime_separation_std": float(np.std(seps)),
    }

    return {
        "per_series": per_series,
        "aggregate": aggregate,
    }
