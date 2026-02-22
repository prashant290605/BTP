# Online Adaptive Deep-Learning-Based Early Warning System for Structural Change in Streaming Time Series

BTech Final Year Project - Research-Grade Implementation

## Project Overview

This project implements an early warning system that operates in real-time, adapts to concept drift, and produces continuous warning scores before structural transitions occur in streaming time series data.

## Architecture

- **Model**: CNN–LSTM hybrid architecture
- **Learning**: Offline baseline + Online adaptive learning
- **Drift Handling**: Explicit concept drift detection and adaptation

## Project Structure

```
BTP Project/
├── data/
│   ├── raw/                    # Raw synthetic datasets
│   ├── processed/              # Preprocessed data with rolling windows
│   └── drift_scenarios/        # Datasets with concept drift
├── src/
│   ├── data/                   # Data generation and preprocessing
│   ├── models/                 # CNN-LSTM model implementations
│   ├── evaluation/             # Evaluation metrics and scripts
│   └── utils/                  # Utility functions
├── notebooks/                  # Jupyter notebooks for experiments
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Data Generation

The synthetic data generator creates time series with three distinct phases:

1. **Stable Regime**: Normal dynamics before transition
2. **Approaching Transition**: Gradual approach to critical point (warning phase)
3. **Post-Transition**: New stable state after structural change

### Dynamics Types

- **Fold Bifurcation**: `dx/dt = r - x^2 + noise`
- **Saddle-Node Bifurcation**: `dx/dt = r + x - x^3 + noise`
- **Hopf Bifurcation**: Oscillatory transition dynamics

### Concept Drift Mechanisms

- **Noise Variance Drift**: Gradual change in noise level
- **Mean Shift**: Gradual shift in mean without collapse
- **Scale Drift**: Changes in amplitude/variance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Data

```python
from src.data import SyntheticTimeSeriesGenerator

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

## Evaluation Metrics

- **Lead Time**: Time between warning and actual transition
- **False Alarm Rate**: Proportion of false warnings
- **Warning Signal Stability**: Consistency of warning scores
- **Performance Under Drift**: Degradation metrics during concept drift

## Next Steps

- [ ] Implement preprocessing pipeline with rolling windows
- [ ] Build offline baseline CNN–LSTM model
- [ ] Develop online adaptive learning module
- [ ] Create warning score generation system
- [ ] Implement evaluation framework
- [ ] Run experiments and generate results

## License

Academic/Research Use
