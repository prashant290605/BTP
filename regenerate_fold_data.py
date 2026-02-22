"""
Script to regenerate only fold bifurcation datasets with soft stabilization.
"""

import sys
sys.path.append('src')

from data import SyntheticTimeSeriesGenerator

# Initialize generator with same seed for reproducibility
generator = SyntheticTimeSeriesGenerator(seed=42)

print("Regenerating fold bifurcation datasets with soft stabilization...")
print("=" * 60)

# Generate baseline fold dataset
print("\n1. Baseline fold bifurcation...")
generator.generate_dataset(
    n_realizations=100,
    length=2500,
    output_dir="data/raw",
    dynamics_type="fold"
)

# Generate fold datasets with concept drift
print("\n2. Fold with noise variance drift...")
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

print("\n3. Fold with mean shift drift...")
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

print("\n4. Fold with scale drift...")
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

print("\n" + "=" * 60)
print("Fold bifurcation datasets regenerated successfully!")
print("Soft stabilization (x^4 term) preserves early-warning signals.")
