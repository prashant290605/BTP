"""
Evaluation script for offline CNN-LSTM baseline.

Loads trained model and evaluates on test data.
Computes statistics and visualizes warning scores.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import json
import os


def load_model_and_params(model_path="models/cnn_lstm_offline"):
    """Load trained model and normalization parameters."""
    model = keras.models.load_model(f"{model_path}.keras")
    
    with open(f"{model_path}_norm.json", 'r') as f:
        norm_params = json.load(f)
    
    return model, norm_params['mean'], norm_params['std']


def predict_on_time_series(model, time_series, window_size, mean, std):
    """
    Predict warning scores for entire time series using batched prediction.
    
    Args:
        model: Trained model
        time_series: 1D time series
        window_size: Window size
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Warning scores array (length = len(time_series))
    """
    scores = np.full(len(time_series), np.nan)
    
    # Create all windows at once
    n_windows = len(time_series) - window_size
    windows = np.zeros((n_windows, window_size, 1))
    
    for i in range(n_windows):
        window = time_series[i:i + window_size]
        # Normalize
        window_norm = (window - mean) / std
        windows[i, :, 0] = window_norm
    
    # Batched prediction (much faster than per-window)
    predictions = model.predict(windows, batch_size=256, verbose=0).flatten()
    
    # Assign scores to correct positions
    scores[window_size:] = predictions
    
    return scores


def evaluate_model(
    model_path="models/cnn_lstm_offline",
    data_path="data/raw",
    dynamics_type="fold",
    window_size=100
):
    """
    Evaluate trained model on test data.
    
    Args:
        model_path: Path to saved model
        data_path: Path to data
        dynamics_type: Type of dynamics
        window_size: Window size
    """
    print("=" * 70)
    print("OFFLINE CNN-LSTM BASELINE - EVALUATION")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model, mean, std = load_model_and_params(model_path)
    print(f"Normalization: mean={mean:.4f}, std={std:.4f}")
    
    # Load test data
    print("\nLoading test data...")
    time_series = np.load(os.path.join(data_path, f"time_series_{dynamics_type}.npy"))
    labels = np.load(os.path.join(data_path, f"labels_{dynamics_type}.npy"))
    
    # Use last 15% as test set (same as training script)
    n_total = len(time_series)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    test_ts = time_series[n_train + n_val:]
    test_labels = labels[n_train + n_val:]
    
    print(f"Test samples: {len(test_ts)}")
    
    # Predict on test samples
    print("\nPredicting warning scores...")
    all_scores = []
    all_labels = []
    
    for i in range(len(test_ts)):
        ts = test_ts[i]
        label = test_labels[i]
        
        # Skip if contains NaN
        if np.any(np.isnan(ts)):
            continue
        
        scores = predict_on_time_series(model, ts, window_size, mean, std)
        all_scores.append(scores)
        all_labels.append(label)
    
    print(f"Evaluated {len(all_scores)} test samples")
    
    # Compute statistics per regime
    print("\n" + "=" * 70)
    print("WARNING SCORES BY REGIME")
    print("=" * 70)
    
    regime_stats = {}
    
    for regime_val, regime_name in [(0, "Stable"), (1, "Approaching"), (2, "Post-transition")]:
        regime_scores = []
        
        for scores, label in zip(all_scores, all_labels):
            mask = (label == regime_val) & (~np.isnan(scores))
            regime_scores.extend(scores[mask])
        
        if len(regime_scores) > 0:
            regime_scores = np.array(regime_scores)
            stats = {
                'mean': np.mean(regime_scores),
                'std': np.std(regime_scores),
                'min': np.min(regime_scores),
                'max': np.max(regime_scores)
            }
            regime_stats[regime_val] = stats
            
            print(f"\n{regime_name} (label={regime_val}):")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")
            print(f"  Count: {len(regime_scores)}")
    
    # Visualize warning scores
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    
    # Plot first 5 test samples
    n_plot = min(5, len(all_scores))
    
    fig, axes = plt.subplots(n_plot, 1, figsize=(15, 3 * n_plot))
    if n_plot == 1:
        axes = [axes]
    
    for i in range(n_plot):
        scores = all_scores[i]
        label = all_labels[i]
        
        # Plot warning score
        axes[i].plot(scores, linewidth=1.5, color='darkred', label='Warning Score')
        
        # Shade regimes
        axes[i].axvspan(0, 1000, alpha=0.15, color='green', label='Stable')
        axes[i].axvspan(1000, 1800, alpha=0.15, color='orange', label='Approaching')
        axes[i].axvspan(1800, len(scores), alpha=0.15, color='red', label='Post-transition')
        
        axes[i].set_ylabel('Warning Score', fontsize=11)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].set_title(f'Test Sample {i+1}', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        if i == 0:
            axes[i].legend(loc='upper left', fontsize=9)
    
    axes[-1].set_xlabel('Time Step', fontsize=11)
    plt.tight_layout()
    
    # Save plot
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/cnn_lstm_warning_scores.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to results/cnn_lstm_warning_scores.png")
    
    plt.show()
    
    # Save statistics
    with open("results/cnn_lstm_stats.json", 'w') as f:
        json.dump(regime_stats, f, indent=2)
    
    print(f"Statistics saved to results/cnn_lstm_stats.json")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    
    return all_scores, all_labels, regime_stats


if __name__ == "__main__":
    scores, labels, stats = evaluate_model(
        model_path="models/cnn_lstm_offline",
        data_path="data/raw",
        dynamics_type="fold",
        window_size=100
    )
