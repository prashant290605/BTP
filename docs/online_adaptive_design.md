# Online Adaptive CNN-LSTM - Design Documentation

## Overview

Online adaptive early warning system that maintains performance under concept drift through unsupervised drift detection and selective fine-tuning.

## Architecture

**Base Model**: Same CNN-LSTM as offline baseline
- Conv1D(16, kernel=5, relu) → MaxPool(2)
- Conv1D(32, kernel=3, relu) → MaxPool(2)
- LSTM(32)
- Dense(1, sigmoid)

**Initialization**: Trained offline model weights

## Key Components

### 1. Drift Detector (Unsupervised)

**Method**: Score variance monitoring

**Logic**:
```python
recent_var = var(scores[-100:])
drift_detected = recent_var > 2.5 * baseline_var
```

**Why it works**:
- Under drift, prediction distribution shifts
- Variance spike indicates distribution change
- No labels required (fully unsupervised)

**Parameters**:
- Threshold: 2.5× baseline variance
- Detection window: 100 steps
- Minimum steps: 100 (for baseline estimation)

### 2. Rolling Buffer

**Purpose**: Store recent data for adaptation

**Capacity**: 1000 windows (FIFO)

**Contents**:
- Normalized windows
- Corresponding labels (for fine-tuning only)

### 3. Adaptive Update Strategy

**Trigger**: Drift detection + cooldown check

**Procedure**:
1. Extract 250 most recent windows from buffer
2. Fine-tune model for 3 epochs
3. Use MSE loss, lr=0.0001, batch_size=32
4. Reset drift detector baseline
5. Set cooldown timer

**Why 250 windows**:
- Sufficient data for meaningful update
- Not too large (prevents overfitting to recent noise)
- Balances adaptation speed vs stability

**Why 3 epochs**:
- Enough for weight adjustment
- Prevents catastrophic forgetting
- Maintains stability

### 4. Stability Controls

**Cooldown Period**: 200 steps
- Prevents rapid repeated adaptations
- Allows model to stabilize after update
- Reduces computational overhead

**Small Learning Rate**: 0.0001
- 10× smaller than offline training
- Gradual weight updates
- Preserves learned patterns

**Limited Epochs**: 3
- Quick adaptation
- Prevents overfitting
- Maintains generalization

**MSE Loss**: Consistent with offline training
- Same objective function
- Smooth gradients
- Stable optimization

## Streaming Inference Pipeline

### Process Flow

```
For each time step t:
  1. Extract window [t-100, t]
  2. Normalize using original statistics
  3. Predict warning score
  4. Update drift detector (score only)
  5. Add window to buffer
  6. Check drift detection
  7. If drift AND cooldown passed:
       → Adapt model on recent 250 windows
       → Reset drift baseline
       → Set cooldown timer
  8. Output warning score
```

### Key Properties

- **One-step-at-a-time**: True streaming mode
- **No future data**: Only uses past windows
- **Unsupervised**: No label access during inference
- **Adaptive**: Updates when drift detected
- **Stable**: Cooldown prevents over-adaptation

## Drift Detection Logic

### Baseline Establishment

```python
# First 100 steps
baseline_var = var(scores[0:100])
baseline_mean = mean(scores[0:100])
```

### Ongoing Monitoring

```python
# Every step after 100
recent_var = var(scores[t-100:t])
if recent_var > 2.5 * baseline_var:
    drift_detected = True
```

### After Adaptation

```python
# Reset baseline to new distribution
baseline_var = var(scores[t-100:t])
```

## Adaptation Mechanism

### Data Collection

```python
# Get recent windows from buffer
windows, labels = buffer.get_recent(250)
```

### Fine-Tuning

```python
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='mse'
)

model.fit(
    windows, labels,
    epochs=3,
    batch_size=32,
    verbose=0
)
```

### State Update

```python
last_adaptation_step = current_step
adaptation_count += 1
drift_detector.reset_baseline()
```

## Expected Behavior

### No Drift
- **Adaptations**: 0-2 (minimal)
- **Performance**: Same as offline baseline
- **Stability**: No degradation

### Noise Variance Drift
- **Adaptations**: 5-10
- **Effect**: Recalibrates to new noise level
- **Improvement**: Reduced variance, stable warnings

### Mean Shift Drift
- **Adaptations**: 3-7
- **Effect**: Relearns shifted patterns
- **Improvement**: Recovery from failure (0.0 → 0.5-0.7)

### Scale Drift
- **Adaptations**: 4-8
- **Effect**: Adjusts sensitivity
- **Improvement**: Better regime separation

## Usage

### Basic Usage

```python
from src.models.online_adaptive_ews import load_online_adaptive_model

# Load model
online_model = load_online_adaptive_model(
    model_path="models/cnn_lstm_offline.keras",
    norm_path="models/cnn_lstm_offline_norm.json"
)

# Process stream
scores, adaptation_points = online_model.process_stream(
    time_series, labels
)

# Get statistics
stats = online_model.get_stats()
print(f"Adaptations: {stats['adaptation_count']}")
```

### Evaluation

```bash
# Evaluate on all scenarios
python evaluate_online_adaptive.py

# Compare with baselines
python compare_all_methods.py
```

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Drift threshold | 2.5 | Balance sensitivity vs false positives |
| Detection window | 100 | Sufficient for variance estimation |
| Adaptation windows | 250 | Balance data vs overfitting |
| Adaptation epochs | 3 | Quick adaptation, prevent forgetting |
| Learning rate | 0.0001 | Gradual updates, stability |
| Cooldown period | 200 | Prevent over-adaptation |
| Buffer size | 1000 | Memory for adaptation |

## Advantages

✓ **Unsupervised**: No labels needed for drift detection  
✓ **Adaptive**: Maintains performance under drift  
✓ **Stable**: Cooldown and small lr prevent instability  
✓ **Efficient**: Selective updates only when needed  
✓ **Streaming**: True one-step-at-a-time processing  

## Limitations

✗ **Latency**: Adaptation takes time (3 epochs)  
✗ **Memory**: Requires buffer storage  
✗ **Computation**: Fine-tuning adds overhead  
✗ **Threshold**: Fixed threshold may not suit all drifts  

## Future Improvements

- Adaptive threshold based on recent statistics
- Multiple drift detection methods (ensemble)
- Faster adaptation (knowledge distillation)
- Adaptive buffer size
- Drift severity estimation

## Files

- `src/models/online_adaptive_ews.py` - Implementation
- `evaluate_online_adaptive.py` - Evaluation script
- `compare_all_methods.py` - Comparison script
- `docs/online_adaptive_design.md` - This documentation
