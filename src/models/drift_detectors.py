"""
Pluggable drift detectors for streaming early-warning models.

Implemented detectors:
- VarianceDriftDetector (legacy baseline used in this project)
- ADWINDetector
- CUSUMDetector
- PageHinkleyDetector
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, Optional
import numpy as np


class BaseDriftDetector(ABC):
    """Unified interface for drift detectors."""

    def __init__(self) -> None:
        self._last_drift = False

    @abstractmethod
    def update(self, value: float) -> bool:
        """Update detector with one value. Returns True when drift is detected."""

    def detect(self) -> bool:
        """
        Compatibility helper for detectors used in update-then-detect style.
        """
        return self._last_drift

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""

    def reset_baseline(self) -> None:
        """
        Compatibility with existing code. By default same as reset().
        Detectors may override to keep recent state and only reset reference.
        """
        self.reset()

    def get_state(self) -> Dict[str, float]:
        """Return diagnostics/state snapshot."""
        return {"last_drift": float(self._last_drift)}


class VarianceDriftDetector(BaseDriftDetector):
    """
    Legacy variance-ratio detector (kept for backward compatibility).
    """

    def __init__(self, threshold: float = 2.5, window_size: int = 100, buffer_maxlen: int = 1000):
        super().__init__()
        self.threshold = threshold
        self.window_size = window_size
        self.score_buffer: Deque[float] = deque(maxlen=buffer_maxlen)
        self.baseline_var: Optional[float] = None
        self.baseline_mean: Optional[float] = None
        self.initialized = False

    def update(self, value: float) -> bool:
        self.score_buffer.append(float(value))
        self._last_drift = self._detect_internal()
        return self._last_drift

    def _detect_internal(self) -> bool:
        if len(self.score_buffer) < self.window_size:
            return False

        recent_scores = np.array(list(self.score_buffer)[-self.window_size:], dtype=float)
        recent_var = float(np.var(recent_scores))
        recent_mean = float(np.mean(recent_scores))

        if not self.initialized:
            self.baseline_var = recent_var
            self.baseline_mean = recent_mean
            self.initialized = True
            return False

        baseline = max(self.baseline_var if self.baseline_var is not None else 1e-12, 1e-12)
        return bool(recent_var > self.threshold * baseline)

    def reset_baseline(self) -> None:
        if len(self.score_buffer) >= self.window_size:
            recent_scores = np.array(list(self.score_buffer)[-self.window_size:], dtype=float)
            self.baseline_var = float(np.var(recent_scores))
            self.baseline_mean = float(np.mean(recent_scores))
            self.initialized = True
        self._last_drift = False

    def reset(self) -> None:
        self.score_buffer.clear()
        self.baseline_var = None
        self.baseline_mean = None
        self.initialized = False
        self._last_drift = False

    def get_state(self) -> Dict[str, float]:
        return {
            "last_drift": float(self._last_drift),
            "buffer_size": float(len(self.score_buffer)),
            "baseline_var": float(self.baseline_var) if self.baseline_var is not None else np.nan,
            "threshold": float(self.threshold),
        }


class ADWINDetector(BaseDriftDetector):
    """
    Simplified ADWIN-style detector based on adaptive window mean-difference tests.
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_window_length: int = 40,
        clock: int = 1,
        max_window_length: int = 1000,
    ):
        super().__init__()
        self.delta = delta
        self.min_window_length = min_window_length
        self.clock = max(1, int(clock))
        self.max_window_length = max_window_length

        self.window: Deque[float] = deque()
        self.n_seen = 0

    def update(self, value: float) -> bool:
        self.n_seen += 1
        self.window.append(float(value))
        if len(self.window) > self.max_window_length:
            self.window.popleft()

        drift = False
        if self.n_seen % self.clock == 0 and len(self.window) >= 2 * self.min_window_length:
            drift = self._check_drift()

        self._last_drift = drift
        return drift

    def _check_drift(self) -> bool:
        w = np.array(self.window, dtype=float)
        n = w.size
        csum = np.cumsum(w)
        best_cut = None

        # Evaluate candidate cuts with minimum size constraints.
        for cut in range(self.min_window_length, n - self.min_window_length + 1):
            n0 = cut
            n1 = n - cut
            mu0 = csum[cut - 1] / n0
            mu1 = (csum[-1] - csum[cut - 1]) / n1

            eps = np.sqrt(0.5 * np.log(4.0 / self.delta) * (1.0 / n0 + 1.0 / n1))
            if abs(mu0 - mu1) > eps:
                best_cut = cut
                break

        if best_cut is not None:
            # Drop old segment, keep recent subwindow.
            new_w = w[best_cut:]
            self.window = deque(new_w.tolist())
            return True
        return False

    def reset(self) -> None:
        self.window.clear()
        self.n_seen = 0
        self._last_drift = False

    def reset_baseline(self) -> None:
        # ADWIN naturally adapts by trimming older data.
        self._last_drift = False

    def get_state(self) -> Dict[str, float]:
        return {
            "last_drift": float(self._last_drift),
            "window_size": float(len(self.window)),
            "delta": float(self.delta),
        }


class CUSUMDetector(BaseDriftDetector):
    """
    Two-sided CUSUM drift detector on standardized stream values.
    """

    def __init__(
        self,
        k: float = 0.5,
        h: float = 8.0,
        warmup_steps: int = 100,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.k = k
        self.h = h
        self.warmup_steps = warmup_steps
        self.eps = eps

        self.values: Deque[float] = deque(maxlen=max(10, warmup_steps * 5))
        self.mu0: Optional[float] = None
        self.sigma0: Optional[float] = None
        self.s_pos = 0.0
        self.s_neg = 0.0

    def update(self, value: float) -> bool:
        x = float(value)
        self.values.append(x)

        if self.mu0 is None and len(self.values) >= self.warmup_steps:
            arr = np.array(self.values, dtype=float)
            self.mu0 = float(np.mean(arr))
            self.sigma0 = float(np.std(arr))
            self._last_drift = False
            return False

        if self.mu0 is None or self.sigma0 is None:
            self._last_drift = False
            return False

        z = (x - self.mu0) / max(self.sigma0, self.eps)
        self.s_pos = max(0.0, self.s_pos + z - self.k)
        self.s_neg = min(0.0, self.s_neg + z + self.k)

        drift = (self.s_pos > self.h) or (abs(self.s_neg) > self.h)
        self._last_drift = bool(drift)
        return self._last_drift

    def reset_baseline(self) -> None:
        if len(self.values) >= self.warmup_steps:
            arr = np.array(self.values, dtype=float)
            self.mu0 = float(np.mean(arr))
            self.sigma0 = float(np.std(arr))
        self.s_pos = 0.0
        self.s_neg = 0.0
        self._last_drift = False

    def reset(self) -> None:
        self.values.clear()
        self.mu0 = None
        self.sigma0 = None
        self.s_pos = 0.0
        self.s_neg = 0.0
        self._last_drift = False

    def get_state(self) -> Dict[str, float]:
        return {
            "last_drift": float(self._last_drift),
            "mu0": float(self.mu0) if self.mu0 is not None else np.nan,
            "sigma0": float(self.sigma0) if self.sigma0 is not None else np.nan,
            "s_pos": float(self.s_pos),
            "s_neg": float(self.s_neg),
            "k": float(self.k),
            "h": float(self.h),
        }


class PageHinkleyDetector(BaseDriftDetector):
    """
    Page-Hinkley detector with optional forgetting factor.
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 20.0,
        alpha: float = 1.0,
        warmup_steps: int = 100,
    ):
        super().__init__()
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.warmup_steps = warmup_steps

        self.n = 0
        self.mean = 0.0
        self.m_t = 0.0
        self.min_m_t = 0.0
        self.max_m_t = 0.0
        self._warmup_values: Deque[float] = deque(maxlen=max(10, warmup_steps * 2))

    def update(self, value: float) -> bool:
        x = float(value)
        self.n += 1
        self._warmup_values.append(x)

        if self.n <= self.warmup_steps:
            # Running mean during warmup.
            self.mean += (x - self.mean) / self.n
            self._last_drift = False
            return False

        # Exponentially weighted running mean if alpha < 1.
        if self.alpha < 1.0:
            self.mean = self.alpha * self.mean + (1.0 - self.alpha) * x
        else:
            self.mean += (x - self.mean) / self.n

        self.m_t += x - self.mean - self.delta
        self.min_m_t = min(self.min_m_t, self.m_t)
        self.max_m_t = max(self.max_m_t, self.m_t)

        upward_change = (self.m_t - self.min_m_t) > self.lambda_
        downward_change = (self.max_m_t - self.m_t) > self.lambda_
        self._last_drift = bool(upward_change or downward_change)
        return self._last_drift

    def reset_baseline(self) -> None:
        # Keep warm recent values; reset cumulative statistics.
        if len(self._warmup_values) > 0:
            arr = np.array(self._warmup_values, dtype=float)
            self.mean = float(np.mean(arr))
            self.n = max(self.warmup_steps, len(arr))
        else:
            self.n = 0
            self.mean = 0.0
        self.m_t = 0.0
        self.min_m_t = 0.0
        self.max_m_t = 0.0
        self._last_drift = False

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m_t = 0.0
        self.min_m_t = 0.0
        self.max_m_t = 0.0
        self._warmup_values.clear()
        self._last_drift = False

    def get_state(self) -> Dict[str, float]:
        return {
            "last_drift": float(self._last_drift),
            "mean": float(self.mean),
            "m_t": float(self.m_t),
            "min_m_t": float(self.min_m_t),
            "max_m_t": float(self.max_m_t),
            "delta": float(self.delta),
            "lambda": float(self.lambda_),
        }


def create_drift_detector(detector_type: str = "variance", **kwargs) -> BaseDriftDetector:
    """
    Factory for drift detectors.

    detector_type:
      - 'variance' (legacy/default)
      - 'adwin'
      - 'cusum'
      - 'page_hinkley'
    """
    detector_type = detector_type.lower().strip()

    if detector_type == "variance":
        return VarianceDriftDetector(**kwargs)
    if detector_type == "adwin":
        return ADWINDetector(**kwargs)
    if detector_type == "cusum":
        return CUSUMDetector(**kwargs)
    if detector_type in ("page_hinkley", "page-hinkley", "ph"):
        return PageHinkleyDetector(**kwargs)

    raise ValueError(f"Unknown detector_type: {detector_type}")

