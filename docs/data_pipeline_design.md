# Data Pipeline Design Document

## Overview

This document explains the data generation logic for the Online Adaptive Deep-Learning-Based Early Warning System project.

## Data Generation Logic

### 1. Bifurcation Dynamics Models

The synthetic time series generator implements three types of critical transitions based on bifurcation theory:

#### A. Fold Bifurcation
**Mathematical Model:**
```
dx/dt = r - x² + noise
```

- **Stable regime** (r = -0.5): System has stable equilibrium
- **Approaching transition** (r: -0.5 → 0.0): Control parameter r increases gradually
- **Post-transition** (r = 0.5): System transitions to new stable state

**Characteristics:**
- Gradual loss of stability
- Increasing variance and autocorrelation (early warning signals)
- Sudden transition at critical point

#### B. Saddle-Node Bifurcation
**Mathematical Model:**
```
dx/dt = r + x - x³ + noise
```

- **Stable regime** (r = -0.3): System in lower stable branch
- **Approaching transition** (r: -0.3 → 0.0): Gradual approach to saddle-node point
- **Post-transition** (r = 0.5): System jumps to upper stable branch

**Characteristics:**
- Similar to fold bifurcation
- Critical slowing down before transition
- Hysteresis effects

#### C. Hopf Bifurcation
**Mathematical Model:**
```
dx/dt = r*x - y - x*(x² + y²)
dy/dt = x + r*y - y*(x² + y²)
```

- **Stable regime** (r = -0.5): Fixed point is stable
- **Approaching transition** (r: -0.5 → 0.0): Loss of stability
- **Post-transition** (r = 0.3): Oscillatory behavior emerges

**Characteristics:**
- Transition from fixed point to limit cycle
- Emergence of oscillations
- Different type of early warning signals

### 2. Regime Labels

Each time series is labeled with three phases:

- **Label 0 (Stable)**: First 1000 time steps - normal dynamics
- **Label 1 (Approaching Transition)**: Next 800 time steps - warning phase
- **Label 2 (Post-Transition)**: Remaining ~700 time steps - new state

**Important:** Labels indicate phases only, NOT exact tipping points. This reflects real-world scenarios where exact transition timing is unknown.

### 3. Concept Drift Mechanisms

Three types of concept drift are implemented to test model robustness:

#### A. Noise Variance Drift
**Purpose:** Simulate changing measurement noise or environmental variability

**Implementation:**
- Gradually increase noise standard deviation from initial_std to final_std
- Applied over drift period (e.g., steps 500-1000)
- Continues with final_std after drift period

**Effect:** Increases uncertainty without changing underlying dynamics

#### B. Mean Shift Drift
**Purpose:** Simulate gradual baseline changes without structural collapse

**Implementation:**
- Linearly shift mean by shift_amount over drift period
- Applied as additive offset
- Maintains full shift after drift period

**Effect:** Changes baseline level while preserving transition dynamics

#### C. Scale Drift
**Purpose:** Simulate changing amplitude or measurement scale

**Implementation:**
- Gradually change scale factor from initial_scale to final_scale
- Applied multiplicatively around mean
- Preserves mean while changing variance

**Effect:** Changes signal amplitude without affecting mean

### 4. Data Generation Parameters

**Default Configuration:**
- Total length: 2500 time steps
- Stable regime: 1000 steps
- Transition approach: 800 steps
- Post-transition: 700 steps
- Time step (dt): 0.1 (0.05 for Hopf)
- Noise std: 0.1 (0.05 for Hopf)
- Number of realizations: 100 per configuration

**Drift Configuration Example:**
```python
drift_config = {
    'type': 'noise_variance',
    'start': 500,
    'end': 1000,
    'params': {'initial_std': 0.1, 'final_std': 0.3}
}
```

## Folder Structure

```
data/
├── raw/                                    # Baseline datasets (no drift)
│   ├── time_series_fold.npy              # Shape: (100, 2500)
│   ├── labels_fold.npy                    # Shape: (100, 2500)
│   ├── metadata_fold.json
│   ├── time_series_saddle_node.npy
│   ├── labels_saddle_node.npy
│   ├── metadata_saddle_node.json
│   ├── time_series_hopf.npy
│   ├── labels_hopf.npy
│   └── metadata_hopf.json
│
├── drift_scenarios/                        # Datasets with concept drift
│   ├── time_series_fold_noise_variance.npy
│   ├── labels_fold_noise_variance.npy
│   ├── metadata_fold_noise_variance.json
│   ├── time_series_fold_mean_shift.npy
│   ├── labels_fold_mean_shift.npy
│   ├── metadata_fold_mean_shift.json
│   ├── time_series_fold_scale.npy
│   ├── labels_fold_scale.npy
│   └── metadata_fold_scale.json
│
└── processed/                              # Will contain preprocessed data
    └── (to be created in next step)
```

## Data Format

**Time Series Files (.npy):**
- Shape: (n_realizations, length)
- Type: float64
- Values: Continuous time series data

**Label Files (.npy):**
- Shape: (n_realizations, length)
- Type: int
- Values: 0 (stable), 1 (approaching), 2 (post-transition)

**Metadata Files (.json):**
```json
{
  "n_realizations": 100,
  "length": 2500,
  "dynamics_type": "fold",
  "drift_config": null,
  "seed": 42
}
```

## Early Warning Signals

The generated data exhibits classic early warning signals before transitions:

1. **Increasing Variance:** Fluctuations grow as system loses stability
2. **Increasing Autocorrelation:** Critical slowing down - system takes longer to recover from perturbations
3. **Changing Skewness:** Distribution becomes asymmetric near transition

These signals are what the CNN-LSTM model will learn to detect.

## Usage Example

```python
from src.data import SyntheticTimeSeriesGenerator

# Initialize generator
generator = SyntheticTimeSeriesGenerator(seed=42)

# Generate baseline dataset
generator.generate_dataset(
    n_realizations=100,
    length=2500,
    output_dir="data/raw",
    dynamics_type="fold"
)

# Generate dataset with drift
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
```

## Next Steps

1. **Preprocessing Pipeline:** Create rolling windows for CNN-LSTM input
2. **Data Normalization:** Standardize features for model training
3. **Train/Test Split:** Separate data for offline baseline and online evaluation
4. **Augmentation (Optional):** Generate additional realizations if needed

## References

- Scheffer, M., et al. (2009). "Early-warning signals for critical transitions." Nature.
- Dakos, V., et al. (2012). "Methods for detecting early warnings of critical transitions in time series." PLoS ONE.
