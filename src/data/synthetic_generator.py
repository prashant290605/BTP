"""
Synthetic Time Series Generator for Early Warning System.

This module generates synthetic time series with three phases:
1. Stable regime
2. Approaching transition
3. Post-transition regime

It supports modular concept drift injections:
- Mean drift
- Variance drift
- Noise drift
- Autocorrelation drift
- Control parameter drift
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import numpy as np


class SyntheticTimeSeriesGenerator:
    """
    Generates synthetic time series with structural transitions and drift.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    # ---------------------------------------------------------------------
    # Shared drift utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _ramp(t: int, start: int, end: int, power: float = 1.0) -> float:
        """
        Shared ramp(t) in [0, 1] controlling gradual drift progression.
        """
        if end <= start:
            return 1.0 if t >= start else 0.0
        progress = (t - start) / float(end - start)
        progress = float(np.clip(progress, 0.0, 1.0))
        return progress**power

    @staticmethod
    def _resolve_magnitude(drift_config: Optional[Dict], params: Dict) -> float:
        """
        Resolve drift magnitude with backward compatibility.
        """
        if drift_config is None:
            return 1.0
        if "magnitude" in drift_config:
            return float(drift_config["magnitude"])
        if "magnitude" in params:
            return float(params["magnitude"])
        return 1.0

    def _apply_control_parameter_drift(
        self,
        base_r: float,
        t: int,
        start: int,
        end: int,
        magnitude: float = 1.0,
        shift_amount: float = 0.2,
        direction: float = 1.0,
        power: float = 1.0,
    ) -> float:
        """
        Add gradual drift directly to control parameter r(t).
        """
        delta_r = direction * shift_amount * magnitude * self._ramp(t, start, end, power)
        return base_r + delta_r

    # ---------------------------------------------------------------------
    # Dynamical systems
    # ---------------------------------------------------------------------
    def generate_fold_bifurcation(
        self,
        length: int = 2500,
        stable_length: int = 1000,
        transition_length: int = 800,
        dt: float = 0.01,
        noise_std: float = 0.05,
        control_drift_config: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fold bifurcation:
            dx/dt = r(t) - x^2 - gamma*x^4 + noise
        """
        time_series = np.zeros(length)
        labels = np.zeros(length, dtype=int)

        r0 = 1.0
        r_end = -0.2
        r_schedule = np.linspace(r0, r_end, length)
        x = np.sqrt(r0)
        gamma = 0.001

        if control_drift_config is not None:
            d_start = control_drift_config.get("start", length // 4)
            d_end = control_drift_config.get("end", length // 2)
            d_params = control_drift_config.get("params", {})
            d_mag = self._resolve_magnitude(control_drift_config, d_params)
        else:
            d_start = d_end = 0
            d_params = {}
            d_mag = 0.0

        for i in range(length):
            if i < stable_length:
                labels[i] = 0
            elif i < stable_length + transition_length:
                labels[i] = 1
            else:
                labels[i] = 2

            r = float(r_schedule[i])
            if control_drift_config is not None:
                r = self._apply_control_parameter_drift(
                    base_r=r,
                    t=i,
                    start=d_start,
                    end=d_end,
                    magnitude=d_mag,
                    shift_amount=d_params.get("shift_amount", 0.2),
                    direction=d_params.get("direction", 1.0),
                    power=d_params.get("power", 1.0),
                )

            dx = (r - x**2 - gamma * x**4) * dt + noise_std * np.sqrt(dt) * np.random.randn()
            x = x + dx

            if np.abs(x) > 10.0:
                time_series[i:] = np.nan
                break
            time_series[i] = x

        return time_series, labels

    def generate_saddle_node(
        self,
        length: int = 2500,
        stable_length: int = 1000,
        transition_length: int = 800,
        dt: float = 0.1,
        noise_std: float = 0.1,
        control_drift_config: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Saddle-node bifurcation:
            dx/dt = r + x - x^3 + noise
        """
        time_series = np.zeros(length)
        labels = np.zeros(length, dtype=int)
        x = -1.0

        r_stable = -0.3
        r_critical = 0.0
        r_post = 0.5

        if control_drift_config is not None:
            d_start = control_drift_config.get("start", length // 4)
            d_end = control_drift_config.get("end", length // 2)
            d_params = control_drift_config.get("params", {})
            d_mag = self._resolve_magnitude(control_drift_config, d_params)
        else:
            d_start = d_end = 0
            d_params = {}
            d_mag = 0.0

        for i in range(length):
            if i < stable_length:
                r = r_stable
                labels[i] = 0
            elif i < stable_length + transition_length:
                progress = (i - stable_length) / transition_length
                r = r_stable + progress * (r_critical - r_stable)
                labels[i] = 1
            else:
                r = r_post
                labels[i] = 2

            if control_drift_config is not None:
                r = self._apply_control_parameter_drift(
                    base_r=r,
                    t=i,
                    start=d_start,
                    end=d_end,
                    magnitude=d_mag,
                    shift_amount=d_params.get("shift_amount", 0.15),
                    direction=d_params.get("direction", 1.0),
                    power=d_params.get("power", 1.0),
                )

            dx = (r + x - x**3) * dt + noise_std * np.sqrt(dt) * np.random.randn()
            x = x + dx
            x = np.clip(x, -5.0, 5.0)
            time_series[i] = x

        return time_series, labels

    def generate_hopf_bifurcation(
        self,
        length: int = 2500,
        stable_length: int = 1000,
        transition_length: int = 800,
        dt: float = 0.05,
        noise_std: float = 0.05,
        control_drift_config: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hopf bifurcation:
            dx/dt = r*x - y - x*(x^2 + y^2)
            dy/dt = x + r*y - y*(x^2 + y^2)
        """
        time_series = np.zeros(length)
        labels = np.zeros(length, dtype=int)
        x, y = 0.1, 0.1

        r_stable = -0.5
        r_critical = 0.0
        r_post = 0.3

        if control_drift_config is not None:
            d_start = control_drift_config.get("start", length // 4)
            d_end = control_drift_config.get("end", length // 2)
            d_params = control_drift_config.get("params", {})
            d_mag = self._resolve_magnitude(control_drift_config, d_params)
        else:
            d_start = d_end = 0
            d_params = {}
            d_mag = 0.0

        for i in range(length):
            if i < stable_length:
                r = r_stable
                labels[i] = 0
            elif i < stable_length + transition_length:
                progress = (i - stable_length) / transition_length
                r = r_stable + progress * (r_critical - r_stable)
                labels[i] = 1
            else:
                r = r_post
                labels[i] = 2

            if control_drift_config is not None:
                r = self._apply_control_parameter_drift(
                    base_r=r,
                    t=i,
                    start=d_start,
                    end=d_end,
                    magnitude=d_mag,
                    shift_amount=d_params.get("shift_amount", 0.12),
                    direction=d_params.get("direction", 1.0),
                    power=d_params.get("power", 1.0),
                )

            r_sq = x**2 + y**2
            dx = (r * x - y - x * r_sq) * dt + noise_std * np.sqrt(dt) * np.random.randn()
            dy = (x + r * y - y * r_sq) * dt + noise_std * np.sqrt(dt) * np.random.randn()
            x = np.clip(x + dx, -5.0, 5.0)
            y = np.clip(y + dy, -5.0, 5.0)
            time_series[i] = x

        return time_series, labels

    # ---------------------------------------------------------------------
    # Modular drift injections (post-generation)
    # ---------------------------------------------------------------------
    def add_mean_drift(
        self,
        time_series: np.ndarray,
        drift_start: int,
        drift_end: int,
        magnitude: float = 1.0,
        shift_amount: float = 0.5,
        power: float = 1.0,
    ) -> np.ndarray:
        """
        Mean drift:
            x_t <- x_t + ramp(t) * (magnitude * shift_amount)
        """
        ts = time_series.copy()
        total_shift = magnitude * shift_amount

        for i in range(drift_start, len(ts)):
            ts[i] += self._ramp(i, drift_start, drift_end, power) * total_shift

        return ts

    def add_variance_drift(
        self,
        time_series: np.ndarray,
        drift_start: int,
        drift_end: int,
        magnitude: float = 1.0,
        initial_scale: float = 1.0,
        final_scale: float = 1.5,
        power: float = 1.0,
    ) -> np.ndarray:
        """
        Variance drift:
            x_t <- mu + s(t) * (x_t - mu)
        """
        ts = time_series.copy()
        ref_mean = float(np.mean(ts[:max(drift_start, 1)]))
        target_scale = initial_scale + magnitude * (final_scale - initial_scale)

        for i in range(drift_start, len(ts)):
            ramp_val = self._ramp(i, drift_start, drift_end, power)
            scale_t = initial_scale + ramp_val * (target_scale - initial_scale)
            ts[i] = ref_mean + (ts[i] - ref_mean) * scale_t

        return ts

    def add_noise_drift(
        self,
        time_series: np.ndarray,
        drift_start: int,
        drift_end: int,
        magnitude: float = 1.0,
        initial_std: float = 0.1,
        final_std: float = 0.3,
        power: float = 1.0,
    ) -> np.ndarray:
        """
        Noise drift:
            x_t <- x_t + epsilon_t, epsilon_t ~ N(0, sigma(t)^2)
        """
        ts = time_series.copy()
        target_std = initial_std + magnitude * (final_std - initial_std)

        for i in range(drift_start, len(ts)):
            ramp_val = self._ramp(i, drift_start, drift_end, power)
            std_t = initial_std + ramp_val * (target_std - initial_std)
            ts[i] += std_t * np.random.randn()

        return ts

    def add_autocorrelation_drift(
        self,
        time_series: np.ndarray,
        drift_start: int,
        drift_end: int,
        magnitude: float = 1.0,
        rho_max: float = 0.8,
        power: float = 1.0,
    ) -> np.ndarray:
        """
        Autocorrelation drift:
            x_t <- (1-rho(t))*x_t + rho(t)*x_{t-1}
        """
        ts = time_series.copy()
        rho_target = float(np.clip(magnitude * rho_max, 0.0, 0.99))
        start_idx = max(drift_start, 1)

        for i in range(start_idx, len(ts)):
            rho_t = self._ramp(i, drift_start, drift_end, power) * rho_target
            ts[i] = (1.0 - rho_t) * ts[i] + rho_t * ts[i - 1]

        return ts

    # ---------------------------------------------------------------------
    # Backward-compatible drift wrappers
    # ---------------------------------------------------------------------
    def add_noise_variance_drift(
        self,
        time_series: np.ndarray,
        drift_start: int,
        drift_end: int,
        initial_std: float = 0.1,
        final_std: float = 0.3,
    ) -> np.ndarray:
        return self.add_noise_drift(
            time_series=time_series,
            drift_start=drift_start,
            drift_end=drift_end,
            magnitude=1.0,
            initial_std=initial_std,
            final_std=final_std,
        )

    def add_mean_shift_drift(
        self,
        time_series: np.ndarray,
        drift_start: int,
        drift_end: int,
        shift_amount: float = 0.5,
    ) -> np.ndarray:
        return self.add_mean_drift(
            time_series=time_series,
            drift_start=drift_start,
            drift_end=drift_end,
            magnitude=1.0,
            shift_amount=shift_amount,
        )

    def add_scale_drift(
        self,
        time_series: np.ndarray,
        drift_start: int,
        drift_end: int,
        initial_scale: float = 1.0,
        final_scale: float = 1.5,
    ) -> np.ndarray:
        return self.add_variance_drift(
            time_series=time_series,
            drift_start=drift_start,
            drift_end=drift_end,
            magnitude=1.0,
            initial_scale=initial_scale,
            final_scale=final_scale,
        )

    # ---------------------------------------------------------------------
    # Dataset generation
    # ---------------------------------------------------------------------
    def generate_dataset(
        self,
        n_realizations: int = 100,
        length: int = 2500,
        output_dir: str = "data/raw",
        dynamics_type: str = "fold",
        drift_config: Optional[Dict] = None,
    ) -> None:
        """
        Generate multiple realizations and save to disk.

        drift_config keys:
            - type:
                new: 'mean', 'variance', 'noise', 'autocorrelation', 'control_parameter'
                legacy: 'mean_shift', 'scale', 'noise_variance'
            - magnitude: scalar severity (default 1.0)
            - start: drift start index
            - end: drift end index
            - params: drift-specific parameters
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if dynamics_type == "fold":
            generator_func = self.generate_fold_bifurcation
        elif dynamics_type == "saddle_node":
            generator_func = self.generate_saddle_node
        elif dynamics_type == "hopf":
            generator_func = self.generate_hopf_bifurcation
        else:
            raise ValueError(f"Unknown dynamics type: {dynamics_type}")

        drift_type = None
        drift_start = length // 4
        drift_end = length // 2
        params: Dict = {}
        magnitude = 1.0

        if drift_config is not None:
            drift_type = drift_config.get("type")
            drift_start = drift_config.get("start", drift_start)
            drift_end = drift_config.get("end", drift_end)
            params = drift_config.get("params", {})
            magnitude = self._resolve_magnitude(drift_config, params)

        drift_aliases = {
            "mean_shift": "mean",
            "noise_variance": "noise",
            "scale": "variance",
            "control_param": "control_parameter",
        }
        normalized_drift_type = drift_aliases.get(drift_type, drift_type)

        all_series = []
        all_labels = []

        for _ in range(n_realizations):
            control_drift = None
            if normalized_drift_type == "control_parameter" and drift_config is not None:
                control_drift = {
                    "start": drift_start,
                    "end": drift_end,
                    "magnitude": magnitude,
                    "params": params,
                }

            ts, labels = generator_func(length=length, control_drift_config=control_drift)

            if normalized_drift_type == "mean":
                ts = self.add_mean_drift(
                    ts,
                    drift_start,
                    drift_end,
                    magnitude=magnitude,
                    shift_amount=params.get("shift_amount", 0.5),
                    power=params.get("power", 1.0),
                )
            elif normalized_drift_type == "variance":
                ts = self.add_variance_drift(
                    ts,
                    drift_start,
                    drift_end,
                    magnitude=magnitude,
                    initial_scale=params.get("initial_scale", 1.0),
                    final_scale=params.get("final_scale", 1.5),
                    power=params.get("power", 1.0),
                )
            elif normalized_drift_type == "noise":
                ts = self.add_noise_drift(
                    ts,
                    drift_start,
                    drift_end,
                    magnitude=magnitude,
                    initial_std=params.get("initial_std", 0.1),
                    final_std=params.get("final_std", 0.3),
                    power=params.get("power", 1.0),
                )
            elif normalized_drift_type == "autocorrelation":
                ts = self.add_autocorrelation_drift(
                    ts,
                    drift_start,
                    drift_end,
                    magnitude=magnitude,
                    rho_max=params.get("rho_max", 0.8),
                    power=params.get("power", 1.0),
                )
            elif normalized_drift_type == "control_parameter":
                # Applied inside generator dynamics.
                pass
            elif normalized_drift_type is not None:
                raise ValueError(f"Unknown drift type: {drift_type}")

            all_series.append(ts)
            all_labels.append(labels)

        all_series = np.array(all_series)
        all_labels = np.array(all_labels)

        drift_suffix = f"_{drift_type}" if drift_type else ""
        np.save(output_path / f"time_series_{dynamics_type}{drift_suffix}.npy", all_series)
        np.save(output_path / f"labels_{dynamics_type}{drift_suffix}.npy", all_labels)

        metadata = {
            "n_realizations": n_realizations,
            "length": length,
            "dynamics_type": dynamics_type,
            "drift_config": drift_config,
            "seed": self.seed,
        }
        with open(output_path / f"metadata_{dynamics_type}{drift_suffix}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Generated {n_realizations} realizations of {dynamics_type} dynamics")
        print(f"Saved to {output_path}")
        print(f"Shape: {all_series.shape}")


if __name__ == "__main__":
    generator = SyntheticTimeSeriesGenerator(seed=42)

    print("Generating baseline datasets...")
    generator.generate_dataset(
        n_realizations=100,
        length=2500,
        output_dir="data/raw",
        dynamics_type="fold",
    )
    generator.generate_dataset(
        n_realizations=100,
        length=2500,
        output_dir="data/raw",
        dynamics_type="saddle_node",
    )
    generator.generate_dataset(
        n_realizations=100,
        length=2500,
        output_dir="data/raw",
        dynamics_type="hopf",
    )

    print("\nGenerating datasets with concept drift...")
    generator.generate_dataset(
        n_realizations=100,
        length=2500,
        output_dir="data/drift_scenarios",
        dynamics_type="fold",
        drift_config={
            "type": "noise_variance",
            "start": 500,
            "end": 1000,
            "params": {"initial_std": 0.1, "final_std": 0.3},
        },
    )
    generator.generate_dataset(
        n_realizations=100,
        length=2500,
        output_dir="data/drift_scenarios",
        dynamics_type="fold",
        drift_config={
            "type": "mean_shift",
            "start": 500,
            "end": 1000,
            "params": {"shift_amount": 0.5},
        },
    )
    generator.generate_dataset(
        n_realizations=100,
        length=2500,
        output_dir="data/drift_scenarios",
        dynamics_type="fold",
        drift_config={
            "type": "scale",
            "start": 500,
            "end": 1000,
            "params": {"initial_scale": 1.0, "final_scale": 1.5},
        },
    )

    print("\nData generation complete!")
