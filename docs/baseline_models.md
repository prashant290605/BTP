# Baseline Models Documentation

## Overview

Two baseline models for early warning signal detection:

1. **Classical EWS**: Statistical indicators (variance, autocorrelation)
2. **Offline CNN-LSTM**: Deep learning model trained once on full dataset

## Baseline 1: Classical Early Warning Signals

### Architecture

**Method**: Sliding window statistical indicators

**Components**:
- Rolling variance computation
- Rolling lag-1 autocorrelation computation
- Normalization to [0, 1] range
- Combination into warning score

### Implementation

**File**: `src/models/classical_ews.py`

**Class**: `ClassicalEWS`

### How It Works

#### 1. Rolling Variance

For each time step t, compute variance over window [t-w, t]:

```python
variance[t] = var(x[t-w:t])
```

**Why**: Variance increases as system loses stability (critical slowing down)

#### 2. Rolling Autocorrelation

For each time step t, compute lag-1 autocorrelation:

```python
autocorr[t] = corr(x[t-w:t-1], x[t-w+1:t])
```

**Why**: Autocorrelation increases as recovery time from perturbations grows

#### 3. Normalization

Normalize indicators to [0, 1] using percentiles from training data:

```python
var_norm = (variance - min_var) / (max_var - min_var)
ac_norm = (autocorr - min_ac) / (max_ac - min_ac)
```

Uses 1st and 99th percentiles for robustness to outliers.

#### 4. Warning Score Computation

Combine normalized indicators with equal weighting:

```python
warning_score = 0.5 * var_norm + 0.5 * ac_norm
```

**Output**: Continuous warning score ∈ [0, 1] for each time step

### Usage

```python
from src.models import ClassicalEWS

# Initialize
ews = ClassicalEWS(window_size=100)

# Fit on training data
ews.fit(train_time_series)

# Predict on test data
warning_scores = ews.predict(test_time_series)
```

### Parameters

- **window_size**: Size of rolling window (default: 100)
  - Larger windows: smoother signals, less responsive
  - Smaller windows: noisier signals, more responsive

### Advantages

- ✓ Interpretable (based on known EWS theory)
- ✓ No training required (just normalization fitting)
- ✓ Fast computation
- ✓ Works with limited data

### Limitations

- ✗ Assumes specific bifurcation dynamics
- ✗ May not capture complex patterns
- ✗ Sensitive to noise
- ✗ Requires manual window size tuning

---

## Baseline 2: Offline CNN-LSTM Model

### Architecture

**File**: `src/models/cnn_lstm_baseline.py`

**Class**: `CNNLSTM_EWS`

### Network Structure

```
Input: (batch, window_size, 1)
    ↓
CNN Block 1:
    Conv1D(32 filters, kernel=3, relu)
    MaxPooling1D(pool=2)
    ↓
CNN Block 2:
    Conv1D(64 filters, kernel=3, relu)
    MaxPooling1D(pool=2)
    ↓
LSTM Block:
    LSTM(50 units, return_sequences=True)
    LSTM(50 units, return_sequences=False)
    ↓
Dense Output:
    Dense(25 units, relu)
    Dropout(0.3)
    Dense(1 unit, sigmoid)
    ↓
Output: Warning score ∈ [0, 1]
```

### Layer Details

#### CNN Layers (Feature Extraction)

**Purpose**: Extract local patterns and features from time series

- **Conv1D Layer 1**: 32 filters, kernel size 3
  - Learns low-level features (local trends, fluctuations)
  - Padding='same' preserves temporal dimension
  
- **MaxPooling1D**: Pool size 2
  - Reduces dimensionality by 2x
  - Retains most salient features
  
- **Conv1D Layer 2**: 64 filters, kernel size 3
  - Learns higher-level features
  - Combines patterns from first layer

**Output**: Feature maps of shape (batch, window_size/4, 64)

#### LSTM Layers (Temporal Modeling)

**Purpose**: Model temporal dependencies and long-range patterns

- **LSTM Layer 1**: 50 units, return_sequences=True
  - Processes feature maps sequentially
  - Maintains temporal structure
  
- **LSTM Layer 2**: 50 units, return_sequences=False
  - Aggregates temporal information
  - Produces fixed-length representation

**Output**: Context vector of shape (batch, 50)

#### Dense Layers (Warning Score)

**Purpose**: Map learned representation to warning score

- **Dense Layer**: 25 units, ReLU activation
  - Non-linear transformation
  
- **Dropout**: 0.3 rate
  - Regularization to prevent overfitting
  
- **Output Layer**: 1 unit, sigmoid activation
  - Produces warning score ∈ [0, 1]

### Training Details

#### Data Preparation

**Window Creation**:
```python
from src.utils import load_and_prepare_data

X_train, X_val, y_train, y_val, normalizer = load_and_prepare_data(
    data_path="data/raw",
    dynamics_type="fold",
    window_size=100,
    test_size=0.2
)
```

**Label Mapping**:
- Stable regime (0) → 0.0
- Approaching transition (1) → 0.5
- Post-transition (2) → 1.0

**Normalization**: StandardScaler on training windows

#### Training Configuration

- **Loss Function**: Binary cross-entropy
  - Treats as regression problem
  - Penalizes deviation from target scores
  
- **Optimizer**: Adam (lr=0.001)
  - Adaptive learning rate
  - Momentum for faster convergence
  
- **Batch Size**: 32
  - Balance between speed and stability
  
- **Early Stopping**: Patience 10 epochs
  - Monitors validation loss
  - Restores best weights
  
- **Learning Rate Reduction**: Factor 0.5, patience 5
  - Reduces lr when validation loss plateaus

### Warning Score Computation

For a new time series at time step t:

1. **Extract window**: `x[t-w:t]`
2. **Normalize**: Using training statistics
3. **Reshape**: To (1, window_size, 1)
4. **Forward pass**: Through CNN-LSTM
5. **Output**: Warning score ∈ [0, 1]

### Usage

```python
from src.models import CNNLSTM_EWS, train_cnn_lstm_baseline

# Train model
model, metrics = train_cnn_lstm_baseline(
    data_path="data/raw",
    dynamics_type="fold",
    window_size=100,
    epochs=50,
    save_path="models/cnn_lstm_baseline"
)

# Or load pre-trained model
model = CNNLSTM_EWS.load("models/cnn_lstm_baseline")

# Predict on new data
warning_scores = model.predict(X_test)
```

### Parameters

- **window_size**: Input window size (default: 100)
- **cnn_filters**: CNN filter sizes (default: (32, 64))
- **lstm_units**: LSTM hidden units (default: (50, 50))
- **dense_units**: Dense layer size (default: 25)
- **dropout_rate**: Dropout rate (default: 0.3)
- **learning_rate**: Adam learning rate (default: 0.001)

### Advantages

- ✓ Learns complex patterns automatically
- ✓ No manual feature engineering
- ✓ Can capture non-linear dynamics
- ✓ Good generalization with enough data

### Limitations

- ✗ Requires training data
- ✗ Black-box (less interpretable)
- ✗ Computationally expensive
- ✗ No online adaptation (offline baseline)

---

## Comparison

| Aspect | Classical EWS | CNN-LSTM |
|--------|---------------|----------|
| **Interpretability** | High | Low |
| **Training Required** | Minimal | Yes |
| **Computation** | Fast | Moderate |
| **Data Requirements** | Low | High |
| **Flexibility** | Low | High |
| **Generalization** | Limited | Good |

## Expected Performance

### Classical EWS

- **Stable regime**: Low scores (0.0 - 0.3)
- **Approaching transition**: Increasing scores (0.3 - 0.7)
- **Post-transition**: High scores (0.7 - 1.0)

### CNN-LSTM

- **Stable regime**: Low scores (0.0 - 0.2)
- **Approaching transition**: Increasing scores (0.4 - 0.6)
- **Post-transition**: High scores (0.8 - 1.0)

## Files Created

```
src/
├── models/
│   ├── __init__.py
│   ├── classical_ews.py          # Classical EWS implementation
│   └── cnn_lstm_baseline.py      # CNN-LSTM model
└── utils/
    ├── __init__.py
    └── preprocessing.py           # Data preprocessing utilities

models/                            # Saved model directory
```

## Next Steps

1. Train CNN-LSTM baseline on fold bifurcation data
2. Evaluate both baselines on test set
3. Compare performance metrics (lead time, false alarm rate)
4. Test on drift scenarios
5. Proceed to online adaptive learning implementation
