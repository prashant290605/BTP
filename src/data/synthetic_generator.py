"""
Synthetic Time Series Generator for Early Warning System

This module generates synthetic time series with:
1. Stable regime - Normal dynamics before transition
2. Gradual approach to critical transition - Warning phase
3. Post-transition regime - New stable state after structural change

Optional concept drift mechanisms:
- Noise variance drift
- Mean shift without collapse
- Scale changes
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
import json


class SyntheticTimeSeriesGenerator:
    """
    Generates synthetic time series with structural transitions.
    
    The generator uses a fold bifurcation model where the system
    gradually loses stability before transitioning to a new state.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
    
    def generate_fold_bifurcation(
        self,
        length: int = 2500,
        stable_length: int = 1000,
        transition_length: int = 800,
        dt: float = 0.01,
        noise_std: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time series using fold bifurcation dynamics.
        
        Model: dx/dt = r(t) - x^2 - gamma*x^4 + noise
        where r(t) decreases slowly from positive to near zero,
        causing the system to approach a critical transition (fold bifurcation).
        
        Args:
            length: Total time series length
            stable_length: Length of stable regime
            transition_length: Length of gradual approach to transition
            dt: Time step (default 0.01 for numerical stability)
            noise_std: Standard deviation of noise (default 0.05)
            
        Returns:
            time_series: Generated time series
            labels: Phase labels (0=stable, 1=approaching, 2=post-transition)
        """
        time_series = np.zeros(length)
        labels = np.zeros(length, dtype=int)
        
        # Control parameter schedule: decreasing from r0 to r_end
        r0 = 1.0
        r_end = -0.2
        r_schedule = np.linspace(r0, r_end, length)
        
        # Initial condition: equilibrium at r0
        x = np.sqrt(r0)
        
        # Soft confining parameter
        gamma = 0.001
        
        for i in range(length):
            # Determine regime labels based on time
            if i < stable_length:
                labels[i] = 0  # Stable regime
            elif i < stable_length + transition_length:
                labels[i] = 1  # Approaching transition
            else:
                labels[i] = 2  # Post-transition
            
            # Get current control parameter
            r = r_schedule[i]
            
            # Fold bifurcation dynamics: dx/dt = r - x^2 - gamma*x^4
            dx = (r - x**2 - gamma * x**4) * dt + noise_std * np.sqrt(dt) * np.random.randn()
            x = x + dx
            
            # Numerical guard: terminate if trajectory diverges
            if np.abs(x) > 10.0:
                # Fill remaining with NaN and break
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
        noise_std: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time series using saddle-node bifurcation.
        
        Model: dx/dt = r + x - x^3 + noise
        
        Args:
            length: Total time series length
            stable_length: Length of stable regime
            transition_length: Length of gradual approach to transition
            dt: Time step
            noise_std: Standard deviation of noise
            
        Returns:
            time_series: Generated time series
            labels: Phase labels (0=stable, 1=approaching, 2=post-transition)
        """
        time_series = np.zeros(length)
        labels = np.zeros(length, dtype=int)
        
        x = -1.0  # Initial condition
        
        r_stable = -0.3
        r_critical = 0.0
        r_post = 0.5
        
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
            
            # Saddle-node dynamics
            dx = (r + x - x**3) * dt + noise_std * np.sqrt(dt) * np.random.randn()
            x = x + dx
            # Numerical stability: clip to prevent blow-up
            x = np.clip(x, -5.0, 5.0)
            time_series[i] = x
        
        return time_series, labels
    
    def generate_hopf_bifurcation(
        self,
        length: int = 2500,
        stable_length: int = 1000,
        transition_length: int = 800,
        dt: float = 0.05,
        noise_std: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time series using Hopf bifurcation (oscillatory transition).
        
        Model: dx/dt = r*x - y - x*(x^2 + y^2)
               dy/dt = x + r*y - y*(x^2 + y^2)
        
        Returns only x component as the time series.
        
        Args:
            length: Total time series length
            stable_length: Length of stable regime
            transition_length: Length of gradual approach to transition
            dt: Time step
            noise_std: Standard deviation of noise
            
        Returns:
            time_series: Generated time series (x component)
            labels: Phase labels (0=stable, 1=approaching, 2=post-transition)
        """
        time_series = np.zeros(length)
        labels = np.zeros(length, dtype=int)
        
        x, y = 0.1, 0.1  # Initial conditions
        
        r_stable = -0.5
        r_critical = 0.0
        r_post = 0.3
        
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
            
            # Hopf bifurcation dynamics
            r_sq = x**2 + y**2
            dx = (r * x - y - x * r_sq) * dt + noise_std * np.sqrt(dt) * np.random.randn()
            dy = (x + r * y - y * r_sq) * dt + noise_std * np.sqrt(dt) * np.random.randn()
            
            x = x + dx
            y = y + dy
            # Numerical stability: clip to prevent blow-up
            x = np.clip(x, -5.0, 5.0)
            y = np.clip(y, -5.0, 5.0)
            time_series[i] = x
        
        return time_series, labels
    
    def add_noise_variance_drift(
        self,
        time_series: np.ndarray,
        drift_start: int,
        drift_end: int,
        initial_std: float = 0.1,
        final_std: float = 0.3
    ) -> np.ndarray:
        """
        Add concept drift via changing noise variance.
        
        Args:
            time_series: Original time series
            drift_start: Start index of drift
            drift_end: End index of drift
            initial_std: Initial noise standard deviation
            final_std: Final noise standard deviation
            
        Returns:
            Modified time series with noise variance drift
        """
        ts_drift = time_series.copy()
        
        for i in range(drift_start, min(drift_end, len(time_series))):
            progress = (i - drift_start) / (drift_end - drift_start)
            current_std = initial_std + progress * (final_std - initial_std)
            ts_drift[i] += current_std * np.random.randn()
        
        # Continue with final std after drift period
        for i in range(drift_end, len(time_series)):
            ts_drift[i] += final_std * np.random.randn()
        
        return ts_drift
    
    def add_mean_shift_drift(
        self,
        time_series: np.ndarray,
        drift_start: int,
        drift_end: int,
        shift_amount: float = 0.5
    ) -> np.ndarray:
        """
        Add concept drift via gradual mean shift.
        
        Args:
            time_series: Original time series
            drift_start: Start index of drift
            drift_end: End index of drift
            shift_amount: Total amount of mean shift
            
        Returns:
            Modified time series with mean shift drift
        """
        ts_drift = time_series.copy()
        
        for i in range(drift_start, min(drift_end, len(time_series))):
            progress = (i - drift_start) / (drift_end - drift_start)
            ts_drift[i] += progress * shift_amount
        
        # Apply full shift after drift period
        ts_drift[drift_end:] += shift_amount
        
        return ts_drift
    
    def add_scale_drift(
        self,
        time_series: np.ndarray,
        drift_start: int,
        drift_end: int,
        initial_scale: float = 1.0,
        final_scale: float = 1.5
    ) -> np.ndarray:
        """
        Add concept drift via changing scale/amplitude.
        
        Args:
            time_series: Original time series
            drift_start: Start index of drift
            drift_end: End index of drift
            initial_scale: Initial scale factor
            final_scale: Final scale factor
            
        Returns:
            Modified time series with scale drift
        """
        ts_drift = time_series.copy()
        
        # Calculate mean to preserve it during scaling
        mean_val = np.mean(time_series[:drift_start])
        
        for i in range(drift_start, min(drift_end, len(time_series))):
            progress = (i - drift_start) / (drift_end - drift_start)
            current_scale = initial_scale + progress * (final_scale - initial_scale)
            ts_drift[i] = mean_val + (ts_drift[i] - mean_val) * current_scale
        
        # Apply final scale after drift period
        for i in range(drift_end, len(time_series)):
            ts_drift[i] = mean_val + (ts_drift[i] - mean_val) * final_scale
        
        return ts_drift
    
    def generate_dataset(
        self,
        n_realizations: int = 100,
        length: int = 2500,
        output_dir: str = "data/raw",
        dynamics_type: str = "fold",
        drift_config: Optional[Dict] = None
    ) -> None:
        """
        Generate multiple realizations and save to disk.
        
        Args:
            n_realizations: Number of time series to generate
            length: Length of each time series
            output_dir: Directory to save data
            dynamics_type: Type of dynamics ('fold', 'saddle_node', 'hopf')
            drift_config: Optional drift configuration dict with keys:
                - type: 'noise_variance', 'mean_shift', or 'scale'
                - start: drift start index
                - end: drift end index
                - params: dict of drift-specific parameters
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Select generator function
        if dynamics_type == "fold":
            generator_func = self.generate_fold_bifurcation
        elif dynamics_type == "saddle_node":
            generator_func = self.generate_saddle_node
        elif dynamics_type == "hopf":
            generator_func = self.generate_hopf_bifurcation
        else:
            raise ValueError(f"Unknown dynamics type: {dynamics_type}")
        
        all_series = []
        all_labels = []
        
        for i in range(n_realizations):
            # Generate base time series
            ts, labels = generator_func(length=length)
            
            # Apply drift if configured
            if drift_config is not None:
                drift_type = drift_config.get('type')
                drift_start = drift_config.get('start', length // 4)
                drift_end = drift_config.get('end', length // 2)
                params = drift_config.get('params', {})
                
                if drift_type == 'noise_variance':
                    ts = self.add_noise_variance_drift(
                        ts, drift_start, drift_end, **params
                    )
                elif drift_type == 'mean_shift':
                    ts = self.add_mean_shift_drift(
                        ts, drift_start, drift_end, **params
                    )
                elif drift_type == 'scale':
                    ts = self.add_scale_drift(
                        ts, drift_start, drift_end, **params
                    )
            
            all_series.append(ts)
            all_labels.append(labels)
        
        # Convert to arrays
        all_series = np.array(all_series)
        all_labels = np.array(all_labels)
        
        # Save data
        drift_suffix = f"_{drift_config['type']}" if drift_config else ""
        np.save(output_path / f"time_series_{dynamics_type}{drift_suffix}.npy", all_series)
        np.save(output_path / f"labels_{dynamics_type}{drift_suffix}.npy", all_labels)
        
        # Save metadata
        metadata = {
            'n_realizations': n_realizations,
            'length': length,
            'dynamics_type': dynamics_type,
            'drift_config': drift_config,
            'seed': self.seed
        }
        
        with open(output_path / f"metadata_{dynamics_type}{drift_suffix}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Generated {n_realizations} realizations of {dynamics_type} dynamics")
        print(f"Saved to {output_path}")
        print(f"Shape: {all_series.shape}")


if __name__ == "__main__":
    # Example usage
    generator = SyntheticTimeSeriesGenerator(seed=42)
    
    # Generate baseline datasets (no drift)
    print("Generating baseline datasets...")
    generator.generate_dataset(
        n_realizations=100,
        length=2500,
        output_dir="data/raw",
        dynamics_type="fold"
    )
    
    generator.generate_dataset(
        n_realizations=100,
        length=2500,
        output_dir="data/raw",
        dynamics_type="saddle_node"
    )
    
    generator.generate_dataset(
        n_realizations=100,
        length=2500,
        output_dir="data/raw",
        dynamics_type="hopf"
    )
    
    # Generate datasets with concept drift
    print("\nGenerating datasets with concept drift...")
    
    # Noise variance drift
    generator.generate_dataset(
        n_realizations=100,
        length=2500,
        output_dir="data/drift_scenarios",
        dynamics_type="fold",
        drift_config={
            'type': 'noise_variance',
            'start': 500,
            'end': 1000,
            'params': {'initial_std': 0.1, 'final_std': 0.3}
        }
    )
    
    # Mean shift drift
    generator.generate_dataset(
        n_realizations=100,
        length=2500,
        output_dir="data/drift_scenarios",
        dynamics_type="fold",
        drift_config={
            'type': 'mean_shift',
            'start': 500,
            'end': 1000,
            'params': {'shift_amount': 0.5}
        }
    )
    
    # Scale drift
    generator.generate_dataset(
        n_realizations=100,
        length=2500,
        output_dir="data/drift_scenarios",
        dynamics_type="fold",
        drift_config={
            'type': 'scale',
            'start': 500,
            'end': 1000,
            'params': {'initial_scale': 1.0, 'final_scale': 1.5}
        }
    )
    
    print("\nData generation complete!")
