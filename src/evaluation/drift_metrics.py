"""
Distribution drift metrics for benchmark experiments.

All metrics compare:
    baseline_distribution (reference / training)
    current_distribution  (stream / test window)

Implemented metrics:
1. Wasserstein distance (1D empirical W1)
2. KL divergence (histogram-based)
3. Maximum Mean Discrepancy (RBF-kernel MMD^2)
4. Mean and variance shift
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def _as_1d_array(x: np.ndarray) -> np.ndarray:
    """Convert input to finite 1D numpy array."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Input distribution has no finite values.")
    return arr


def _as_2d_samples(x: np.ndarray) -> np.ndarray:
    """
    Convert input to 2D array: (n_samples, n_features).

    Rules:
    - 1D input -> (n, 1)
    - 2D input -> unchanged
    - >2D input -> flatten trailing dimensions per sample
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        raise ValueError("Input must contain at least one sample.")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    finite_mask = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite_mask]
    if arr.shape[0] == 0:
        raise ValueError("Input distribution has no finite samples.")
    return arr


def wasserstein_distance(
    baseline_distribution: np.ndarray,
    current_distribution: np.ndarray,
) -> float:
    """
    Empirical 1D Wasserstein-1 distance (Earth Mover's Distance).

    For multivariate inputs, values are flattened into 1D.
    """
    x = np.sort(_as_1d_array(baseline_distribution))
    y = np.sort(_as_1d_array(current_distribution))

    all_vals = np.sort(np.concatenate([x, y]))
    if all_vals.size <= 1:
        return 0.0

    deltas = np.diff(all_vals)
    if deltas.size == 0:
        return float(abs(np.mean(x) - np.mean(y)))

    cdf_x = np.searchsorted(x, all_vals[:-1], side="right") / x.size
    cdf_y = np.searchsorted(y, all_vals[:-1], side="right") / y.size
    w1 = np.sum(np.abs(cdf_x - cdf_y) * deltas)
    return float(w1)


def kl_divergence(
    baseline_distribution: np.ndarray,
    current_distribution: np.ndarray,
    bins: int = 64,
    epsilon: float = 1e-12,
) -> float:
    """
    KL divergence D_KL(P || Q), where:
      P = baseline_distribution
      Q = current_distribution

    Uses shared histogram bins with additive epsilon smoothing.
    """
    p_samples = _as_1d_array(baseline_distribution)
    q_samples = _as_1d_array(current_distribution)

    lo = min(float(np.min(p_samples)), float(np.min(q_samples)))
    hi = max(float(np.max(p_samples)), float(np.max(q_samples)))

    if np.isclose(lo, hi):
        return 0.0

    p_hist, bin_edges = np.histogram(p_samples, bins=bins, range=(lo, hi), density=False)
    q_hist, _ = np.histogram(q_samples, bins=bin_edges, density=False)

    p = p_hist.astype(float) + epsilon
    q = q_hist.astype(float) + epsilon
    p /= np.sum(p)
    q /= np.sum(q)

    kl = np.sum(p * np.log(p / q))
    return float(kl)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF kernel matrix exp(-gamma * ||x-y||^2)."""
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    sq_dists = np.maximum(x_norm + y_norm - 2.0 * (x @ y.T), 0.0)
    return np.exp(-gamma * sq_dists)


def _median_heuristic_gamma(z: np.ndarray) -> float:
    """
    Estimate RBF gamma via median pairwise squared distance:
      gamma = 1 / (2 * median_sq_dist)
    """
    n = z.shape[0]
    if n < 2:
        return 1.0

    z_norm = np.sum(z * z, axis=1, keepdims=True)
    sq_dists = np.maximum(z_norm + z_norm.T - 2.0 * (z @ z.T), 0.0)
    upper = sq_dists[np.triu_indices(n, k=1)]
    upper = upper[upper > 0]
    if upper.size == 0:
        return 1.0
    median_sq = float(np.median(upper))
    if median_sq <= 0:
        return 1.0
    return 1.0 / (2.0 * median_sq)


def maximum_mean_discrepancy(
    baseline_distribution: np.ndarray,
    current_distribution: np.ndarray,
    gamma: float | None = None,
    max_samples: int = 1000,
    seed: int = 0,
) -> float:
    """
    Compute RBF-kernel MMD^2 between two sample sets.

    Returns non-negative scalar drift score.
    """
    x = _as_2d_samples(baseline_distribution)
    y = _as_2d_samples(current_distribution)

    if max_samples is not None and max_samples > 0:
        rng = np.random.default_rng(seed)
        if x.shape[0] > max_samples:
            idx = rng.choice(x.shape[0], size=max_samples, replace=False)
            x = x[idx]
        if y.shape[0] > max_samples:
            idx = rng.choice(y.shape[0], size=max_samples, replace=False)
            y = y[idx]

    if gamma is None:
        z = np.vstack([x, y])
        gamma = _median_heuristic_gamma(z)

    k_xx = _rbf_kernel(x, x, gamma)
    k_yy = _rbf_kernel(y, y, gamma)
    k_xy = _rbf_kernel(x, y, gamma)

    mmd2 = np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy)
    return float(max(mmd2, 0.0))


def mean_shift(
    baseline_distribution: np.ndarray,
    current_distribution: np.ndarray,
) -> float:
    """Absolute mean shift |mu_current - mu_baseline|."""
    x = _as_1d_array(baseline_distribution)
    y = _as_1d_array(current_distribution)
    return float(abs(np.mean(y) - np.mean(x)))


def variance_shift(
    baseline_distribution: np.ndarray,
    current_distribution: np.ndarray,
) -> float:
    """Absolute variance shift |var_current - var_baseline|."""
    x = _as_1d_array(baseline_distribution)
    y = _as_1d_array(current_distribution)
    return float(abs(np.var(y) - np.var(x)))


def mean_and_variance_shift(
    baseline_distribution: np.ndarray,
    current_distribution: np.ndarray,
) -> Tuple[float, float]:
    """
    Convenience function returning:
      (mean_shift, variance_shift)
    """
    return (
        mean_shift(baseline_distribution, current_distribution),
        variance_shift(baseline_distribution, current_distribution),
    )


def compute_all_drift_metrics(
    baseline_distribution: np.ndarray,
    current_distribution: np.ndarray,
) -> Dict[str, float]:
    """Compute all drift metrics and return a single dictionary."""
    mu_shift, var_shift = mean_and_variance_shift(baseline_distribution, current_distribution)
    return {
        "wasserstein": wasserstein_distance(baseline_distribution, current_distribution),
        "kl_divergence": kl_divergence(baseline_distribution, current_distribution),
        "mmd": maximum_mean_discrepancy(baseline_distribution, current_distribution),
        "mean_shift": mu_shift,
        "variance_shift": var_shift,
    }
