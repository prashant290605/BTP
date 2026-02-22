"""
Test script for baseline models.

Tests both classical EWS and CNN-LSTM baseline on synthetic data.
"""

import sys
sys.path.append('src')

import numpy as np
from models import ClassicalEWS
from utils import load_and_prepare_data

print("=" * 70)
print("BASELINE MODELS TEST")
print("=" * 70)

# Test 1: Classical EWS
print("\n1. Testing Classical EWS Baseline")
print("-" * 70)

# Load data
print("Loading fold bifurcation data...")
time_series = np.load("data/raw/time_series_fold.npy")
labels = np.load("data/raw/labels_fold.npy")

print(f"Data shape: {time_series.shape}")
print(f"Labels shape: {labels.shape}")

# Split train/test
n_train = 80
train_ts = time_series[:n_train]
test_ts = time_series[n_train:n_train+5]  # Use 5 test samples
test_labels = labels[n_train:n_train+5]

print(f"\nTrain samples: {n_train}")
print(f"Test samples: {len(test_ts)}")

# Initialize and fit
print("\nFitting Classical EWS...")
ews = ClassicalEWS(window_size=100)
ews.fit(train_ts)

# Predict on test data
print("\nPredicting on test data...")
warning_scores = ews.predict(test_ts)

print(f"Warning scores shape: {warning_scores.shape}")

# Analyze scores per regime
for i in range(len(test_ts)):
    ts_labels = test_labels[i]
    ts_scores = warning_scores[i]
    
    # Get scores per regime
    stable_scores = ts_scores[ts_labels == 0]
    approaching_scores = ts_scores[ts_labels == 1]
    post_scores = ts_scores[ts_labels == 2]
    
    print(f"\nTest sample {i+1}:")
    if len(stable_scores) > 0:
        print(f"  Stable regime: mean={np.nanmean(stable_scores):.3f}, "
              f"std={np.nanstd(stable_scores):.3f}")
    if len(approaching_scores) > 0:
        print(f"  Approaching: mean={np.nanmean(approaching_scores):.3f}, "
              f"std={np.nanstd(approaching_scores):.3f}")
    if len(post_scores) > 0:
        print(f"  Post-transition: mean={np.nanmean(post_scores):.3f}, "
              f"std={np.nanstd(post_scores):.3f}")

print("\n✓ Classical EWS test completed successfully!")

# Test 2: Preprocessing utilities
print("\n" + "=" * 70)
print("2. Testing Preprocessing Utilities")
print("-" * 70)

print("\nLoading and preparing data for CNN-LSTM...")
X_train, X_test, y_train, y_test, normalizer = load_and_prepare_data(
    data_path="data/raw",
    dynamics_type="fold",
    window_size=100,
    test_size=0.2,
    random_state=42
)

print(f"\nData shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test: {y_test.shape}")

print(f"\nLabel distribution (train):")
print(f"  0.0 (stable): {np.sum(y_train == 0.0)}")
print(f"  0.5 (approaching): {np.sum(y_train == 0.5)}")
print(f"  1.0 (post-transition): {np.sum(y_train == 1.0)}")

print(f"\nData statistics (train):")
print(f"  Mean: {np.mean(X_train):.4f}")
print(f"  Std: {np.std(X_train):.4f}")
print(f"  Min: {np.min(X_train):.4f}")
print(f"  Max: {np.max(X_train):.4f}")

print("\n✓ Preprocessing test completed successfully!")

# Test 3: CNN-LSTM Model Architecture
print("\n" + "=" * 70)
print("3. Testing CNN-LSTM Model Architecture")
print("-" * 70)

from models import CNNLSTM_EWS

print("\nBuilding CNN-LSTM model...")
model = CNNLSTM_EWS(window_size=100)

print("\n✓ Model built successfully!")
print(f"  Total parameters: {model.model.count_params():,}")

# Test forward pass
print("\nTesting forward pass...")
test_input = X_test[:5]  # Use 5 samples
predictions = model.predict(test_input)

print(f"Input shape: {test_input.shape}")
print(f"Output shape: {predictions.shape}")
print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

print("\n✓ CNN-LSTM architecture test completed successfully!")

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\nBaseline models are ready for training and evaluation.")
print("\nNext steps:")
print("  1. Train CNN-LSTM: python src/models/cnn_lstm_baseline.py")
print("  2. Evaluate baselines on test data")
print("  3. Compare performance metrics")
