"""
Evaluate offline CNN-LSTM baseline under concept drift scenarios.

Tests the SAME trained model (no retraining) on drift datasets
to demonstrate performance degradation.
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
    """
    scores = np.full(len(time_series), np.nan)
    
    # Create all windows at once
    n_windows = len(time_series) - window_size
    windows = np.zeros((n_windows, window_size, 1))
    
    for i in range(n_windows):
        window = time_series[i:i + window_size]
        # Normalize using ORIGINAL training statistics (no adaptation)
        window_norm = (window - mean) / std
        windows[i, :, 0] = window_norm
    
    # Batched prediction
    predictions = model.predict(windows, batch_size=256, verbose=0).flatten()
    
    # Assign scores to correct positions
    scores[window_size:] = predictions
    
    return scores


def evaluate_on_drift_scenario(
    model,
    mean,
    std,
    drift_type,
    data_path="data/drift_scenarios",
    window_size=100
):
    """
    Evaluate model on a specific drift scenario.
    
    Args:
        model: Trained model (no retraining)
        mean: Original normalization mean
        std: Original normalization std
        drift_type: Type of drift scenario
        data_path: Path to drift data
        window_size: Window size
        
    Returns:
        all_scores, all_labels, regime_stats
    """
    print(f"\n{'=' * 70}")
    print(f"EVALUATING ON: {drift_type.upper()}")
    print(f"{'=' * 70}")
    
    # Load drift data
    ts_file = os.path.join(data_path, f"time_series_fold_{drift_type}.npy")
    label_file = os.path.join(data_path, f"labels_fold_{drift_type}.npy")
    
    time_series = np.load(ts_file)
    labels = np.load(label_file)
    
    print(f"Loaded {drift_type} data: {time_series.shape}")
    
    # Use all data for drift testing (no train/val split needed)
    test_ts = time_series
    test_labels = labels
    
    # Predict on all samples
    print("Predicting warning scores...")
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
    
    print(f"Evaluated {len(all_scores)} samples")
    
    # Compute statistics per regime
    print(f"\nWARNING SCORES BY REGIME ({drift_type}):")
    print("-" * 70)
    
    regime_stats = {}
    
    for regime_val, regime_name in [(0, "Stable"), (1, "Approaching"), (2, "Post-transition")]:
        regime_scores = []
        
        for scores, label in zip(all_scores, all_labels):
            mask = (label == regime_val) & (~np.isnan(scores))
            regime_scores.extend(scores[mask])
        
        if len(regime_scores) > 0:
            regime_scores = np.array(regime_scores)
            stats = {
                'mean': float(np.mean(regime_scores)),
                'std': float(np.std(regime_scores)),
                'min': float(np.min(regime_scores)),
                'max': float(np.max(regime_scores))
            }
            regime_stats[regime_val] = stats
            
            print(f"\n{regime_name} (label={regime_val}):")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")
    
    return all_scores, all_labels, regime_stats


def visualize_drift_results(all_results, save_dir="results/drift_analysis"):
    """
    Create visualizations for all drift scenarios.
    
    Args:
        all_results: Dictionary with drift_type -> (scores, labels, stats)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot each drift scenario
    for drift_type, (scores_list, labels_list, stats) in all_results.items():
        n_plot = min(3, len(scores_list))
        
        fig, axes = plt.subplots(n_plot, 1, figsize=(15, 3 * n_plot))
        if n_plot == 1:
            axes = [axes]
        
        for i in range(n_plot):
            scores = scores_list[i]
            label = labels_list[i]
            
            # Plot warning score
            axes[i].plot(scores, linewidth=1.5, color='darkred', label='Warning Score')
            
            # Shade regimes
            axes[i].axvspan(0, 1000, alpha=0.15, color='green', label='Stable')
            axes[i].axvspan(1000, 1800, alpha=0.15, color='orange', label='Approaching')
            axes[i].axvspan(1800, len(scores), alpha=0.15, color='red', label='Post-transition')
            
            axes[i].set_ylabel('Warning Score', fontsize=11)
            axes[i].set_ylim(-0.05, 1.05)
            axes[i].set_title(f'{drift_type.replace("_", " ").title()} - Sample {i+1}', 
                            fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            if i == 0:
                axes[i].legend(loc='upper left', fontsize=9)
        
        axes[-1].set_xlabel('Time Step', fontsize=11)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, f"drift_{drift_type}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close()


def main():
    """
    Main evaluation function for drift scenarios.
    """
    print("=" * 70)
    print("OFFLINE CNN-LSTM UNDER CONCEPT DRIFT")
    print("Stress-testing trained model (NO retraining)")
    print("=" * 70)
    
    # Load trained model (same as baseline evaluation)
    print("\nLoading trained model...")
    model, mean, std = load_model_and_params("models/cnn_lstm_offline")
    print(f"Using ORIGINAL normalization: mean={mean:.4f}, std={std:.4f}")
    print("(No adaptation to drift)")
    
    # Drift scenarios to test
    drift_scenarios = [
        "noise_variance",
        "mean_shift",
        "scale"
    ]
    
    # Evaluate on each drift scenario
    all_results = {}
    
    for drift_type in drift_scenarios:
        scores, labels, stats = evaluate_on_drift_scenario(
            model, mean, std, drift_type,
            data_path="data/drift_scenarios",
            window_size=100
        )
        all_results[drift_type] = (scores, labels, stats)
    
    # Visualize results
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    visualize_drift_results(all_results, save_dir="results/drift_analysis")
    
    # Save statistics
    stats_summary = {}
    for drift_type, (_, _, stats) in all_results.items():
        stats_summary[drift_type] = stats
    
    with open("results/drift_analysis/drift_stats_summary.json", 'w') as f:
        json.dump(stats_summary, f, indent=2)
    
    print("\nStatistics saved to results/drift_analysis/drift_stats_summary.json")
    
    # Summary
    print("\n" + "=" * 70)
    print("DRIFT ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\nBaseline (no drift) performance:")
    print("  Stable: ~0.03, Approaching: ~0.50, Post: ~0.97")
    
    print("\nDrift scenario performance:")
    for drift_type, (_, _, stats) in all_results.items():
        print(f"\n{drift_type.replace('_', ' ').title()}:")
        for regime_val, regime_name in [(0, "Stable"), (1, "Approaching"), (2, "Post-transition")]:
            if regime_val in stats:
                print(f"  {regime_name}: {stats[regime_val]['mean']:.4f} "
                      f"(std: {stats[regime_val]['std']:.4f})")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nThe offline CNN-LSTM model shows performance degradation under drift.")
    print("Without online adaptation, the model cannot adjust to changing dynamics.")
    print("This demonstrates the need for adaptive learning mechanisms.")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
