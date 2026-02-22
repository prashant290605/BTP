# Offline CNN-LSTM Baseline - Instructions

## Overview

Offline CNN-LSTM early warning system baseline with exact specified architecture.

## Architecture

```
Input: (100, 1)
    ↓
Conv1D(16 filters, kernel=5, relu)
    ↓
MaxPooling1D(pool=2)
    ↓
Conv1D(32 filters, kernel=3, relu)
    ↓
MaxPooling1D(pool=2)
    ↓
LSTM(32 units)
    ↓
Dense(1 unit, sigmoid)
    ↓
Output: Warning score ∈ [0, 1]
```

## Training Configuration

- **Loss**: Binary cross-entropy
- **Optimizer**: Adam
- **Epochs**: 30
- **Batch size**: 64
- **Window size**: 100
- **Train once**: No online updates

## Label Mapping

- Stable regime (0) → 0.0
- Approaching transition (1) → 0.5
- Post-transition (2) → 1.0

## Files

- **`train_cnn_lstm.py`**: Training script
- **`evaluate_cnn_lstm.py`**: Evaluation and visualization script

## Usage

### Step 1: Install TensorFlow

```bash
pip install tensorflow>=2.13.0
```

### Step 2: Train Model

```bash
python train_cnn_lstm.py
```

**Output**:
- Model saved to `models/cnn_lstm_offline.keras`
- Training history saved to `models/cnn_lstm_offline_history.json`
- Normalization parameters saved to `models/cnn_lstm_offline_norm.json`

**Expected training time**: 5-10 minutes (depending on hardware)

### Step 3: Evaluate Model

```bash
python evaluate_cnn_lstm.py
```

**Output**:
- Warning score statistics by regime
- Visualization plot saved to `results/cnn_lstm_warning_scores.png`
- Statistics saved to `results/cnn_lstm_stats.json`

## Expected Results

### Warning Score Statistics

**Stable Regime (label=0)**:
- Mean: 0.0 - 0.2
- Low variance
- Scores should be consistently low

**Approaching Transition (label=1)**:
- Mean: 0.3 - 0.6
- Increasing trend
- Scores should rise gradually

**Post-Transition (label=2)**:
- Mean: 0.7 - 1.0
- High variance possible
- Scores should be consistently high

### Visualization

The evaluation script generates plots showing:
- Warning scores over time
- Regime boundaries (shaded regions)
- Clear increase in scores before transitions

## Data Split

- **Train**: 70% of data (70 samples)
- **Validation**: 15% of data (15 samples)
- **Test**: 15% of data (15 samples)

## Model Parameters

Total parameters: ~10,000 (exact count depends on window size after pooling)

## Troubleshooting

### TensorFlow not found

```bash
pip install tensorflow>=2.13.0
```

### Out of memory

Reduce batch size in `train_cnn_lstm.py`:
```python
batch_size=32  # or 16
```

### Model not converging

- Check data normalization (mean ≈ 0, std ≈ 1)
- Verify label mapping (0→0.0, 1→0.5, 2→1.0)
- Ensure no NaN values in data

## Next Steps

After baseline evaluation:
1. Compare with classical EWS baseline
2. Test on concept drift scenarios
3. Implement online adaptive learning
4. Develop comprehensive evaluation framework

## Constraints Followed

✓ Exact architecture as specified  
✓ No transformers, attention, TDA, or causal models  
✓ No concept drift handling (offline baseline)  
✓ No hyperparameter tuning  
✓ Fixed window size (100)  
✓ Train once, no retraining  
