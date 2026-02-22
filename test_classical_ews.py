"""
Simple test for classical EWS (no TensorFlow required).
"""

import sys
sys.path.append('src')

import numpy as np
from models.classical_ews import ClassicalEWS

print("=" * 70)
print("CLASSICAL EWS BASELINE TEST")
print("=" * 70)

# Load data
print("\nLoading fold bifurcation data...")
time_series = np.load("data/raw/time_series_fold.npy")
labels = np.load("data/raw/labels_fold.npy")

print(f"Data shape: {time_series.shape}")

# Split train/test
n_train = 80
train_ts = time_series[:n_train]
test_ts = time_series[n_train:n_train+3]
test_labels = labels[n_train:n_train+3]

print(f"Train samples: {n_train}")
print(f"Test samples: {len(test_ts)}")

# Initialize and fit
print("\nFitting Classical EWS (window_size=100)...")
ews = ClassicalEWS(window_size=100)
ews.fit(train_ts)

# Predict
print("\nPredicting warning scores...")
warning_scores = ews.predict(test_ts)

print(f"Warning scores shape: {warning_scores.shape}")

# Analyze results
print("\n" + "-" * 70)
print("RESULTS")
print("-" * 70)

for i in range(len(test_ts)):
    ts_labels = test_labels[i]
    ts_scores = warning_scores[i]
    
    # Get scores per regime (skip NaN values)
    valid_mask = ~np.isnan(ts_scores)
    
    stable_mask = (ts_labels == 0) & valid_mask
    approaching_mask = (ts_labels == 1) & valid_mask
    post_mask = (ts_labels == 2) & valid_mask
    
    print(f"\nTest Sample {i+1}:")
    
    if np.any(stable_mask):
        stable_scores = ts_scores[stable_mask]
        print(f"  Stable (0):       mean={np.mean(stable_scores):.3f}, "
              f"std={np.std(stable_scores):.3f}")
    
    if np.any(approaching_mask):
        approaching_scores = ts_scores[approaching_mask]
        print(f"  Approaching (1):  mean={np.mean(approaching_scores):.3f}, "
              f"std={np.std(approaching_scores):.3f}")
    
    if np.any(post_mask):
        post_scores = ts_scores[post_mask]
        print(f"  Post-trans (2):   mean={np.mean(post_scores):.3f}, "
              f"std={np.std(post_scores):.3f}")

print("\n" + "=" * 70)
print("✓ Classical EWS test completed successfully!")
print("=" * 70)

print("\nExpected behavior:")
print("  - Stable regime: Low scores (0.0 - 0.3)")
print("  - Approaching: Increasing scores (0.3 - 0.7)")
print("  - Post-transition: High scores (0.7 - 1.0)")
