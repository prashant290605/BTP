"""
Evaluate online adaptive CNN-LSTM on all scenarios.

Compares performance against offline baseline.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from models.online_adaptive_ews import load_online_adaptive_model
import json
import os


def evaluate_online_adaptive(
    data_path,
    dynamics_type,
    model_path="models/cnn_lstm_offline.keras",
    norm_path="models/cnn_lstm_offline_norm.json"
):
    """
    Evaluate online adaptive model on a dataset.
    
    Args:
        data_path: Path to data directory
        dynamics_type: Type of dynamics (e.g., 'fold', 'fold_noise_variance')
        model_path: Path to offline model
        norm_path: Path to normalization parameters
        
    Returns:
        all_scores, all_labels, regime_stats, adaptation_stats
    """
    print(f"\n{'=' * 70}")
    print(f"ONLINE ADAPTIVE CNN-LSTM: {dynamics_type.upper()}")
    print(f"{'=' * 70}")
    
    # Load data
    if 'fold_' in dynamics_type:
        # Drift scenario
        ts_file = os.path.join(data_path, f"time_series_{dynamics_type}.npy")
        label_file = os.path.join(data_path, f"labels_{dynamics_type}.npy")
    else:
        # No drift
        ts_file = os.path.join(data_path, f"time_series_{dynamics_type}.npy")
        label_file = os.path.join(data_path, f"labels_{dynamics_type}.npy")
    
    time_series = np.load(ts_file)
    labels = np.load(label_file)
    
    print(f"Loaded data: {time_series.shape}")
    
    # Create model ONCE (reuse for all samples)
    online_model = load_online_adaptive_model(model_path, norm_path)
    
    # Process all samples
    all_scores = []
    all_labels = []
    all_adaptations = []
    
    for i in range(len(time_series)):
        ts = time_series[i]
        label = labels[i]
        
        # Skip if contains NaN
        if np.any(np.isnan(ts)):
            continue
        
        # Reset model state for new sample (but reuse loaded model)
        if hasattr(online_model, "reset_stream_state"):
            online_model.reset_stream_state()
        else:
            online_model.buffer.windows.clear()
            online_model.buffer.labels.clear()
            online_model.current_step = 0
            online_model.last_adaptation_step = -online_model.cooldown_period
            online_model.adaptation_count = 0
            online_model.adaptation_points = []
        
        # Reload original model weights (reset adaptations)
        online_model.model.load_weights(model_path.replace('.keras', '_weights.h5') 
                                       if os.path.exists(model_path.replace('.keras', '_weights.h5'))
                                       else model_path)
        
        # Process stream
        scores, adaptation_points = online_model.process_stream(ts, label)
        
        all_scores.append(scores)
        all_labels.append(label)
        all_adaptations.append(adaptation_points)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(time_series)} samples...")
    
    print(f"Evaluated {len(all_scores)} samples")
    
    # Compute statistics per regime
    print(f"\nWARNING SCORES BY REGIME:")
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
    
    # Adaptation statistics
    total_adaptations = sum(len(a) for a in all_adaptations)
    avg_adaptations = total_adaptations / len(all_adaptations) if all_adaptations else 0
    
    adaptation_stats = {
        'total_adaptations': total_adaptations,
        'avg_per_sample': avg_adaptations,
        'samples_with_adaptation': sum(1 for a in all_adaptations if len(a) > 0)
    }
    
    print(f"\nADAPTATION STATISTICS:")
    print(f"  Total adaptations: {total_adaptations}")
    print(f"  Avg per sample: {avg_adaptations:.2f}")
    print(f"  Samples with adaptation: {adaptation_stats['samples_with_adaptation']}/{len(all_adaptations)}")
    
    return all_scores, all_labels, regime_stats, adaptation_stats, all_adaptations


def visualize_online_adaptive(
    all_scores,
    all_labels,
    all_adaptations,
    scenario_name,
    save_dir="results/online_adaptive"
):
    """
    Visualize online adaptive results.
    
    Args:
        all_scores: List of score arrays
        all_labels: List of label arrays
        all_adaptations: List of adaptation points
        scenario_name: Name of scenario
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot first 3 samples
    n_plot = min(3, len(all_scores))
    
    fig, axes = plt.subplots(n_plot, 1, figsize=(15, 3 * n_plot))
    if n_plot == 1:
        axes = [axes]
    
    for i in range(n_plot):
        scores = all_scores[i]
        label = all_labels[i]
        adaptations = all_adaptations[i]
        
        # Plot warning score
        axes[i].plot(scores, linewidth=1.5, color='darkred', label='Warning Score')
        
        # Mark adaptation points
        for adapt_point in adaptations:
            axes[i].axvline(adapt_point, color='blue', linestyle='--', 
                          alpha=0.7, linewidth=1.5, label='Adaptation' if adapt_point == adaptations[0] else '')
        
        # Shade regimes
        axes[i].axvspan(0, 1000, alpha=0.15, color='green', label='Stable')
        axes[i].axvspan(1000, 1800, alpha=0.15, color='orange', label='Approaching')
        axes[i].axvspan(1800, len(scores), alpha=0.15, color='red', label='Post-transition')
        
        axes[i].set_ylabel('Warning Score', fontsize=11)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].set_title(f'{scenario_name} - Sample {i+1} ({len(adaptations)} adaptations)', 
                        fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        if i == 0:
            axes[i].legend(loc='upper left', fontsize=9)
    
    axes[-1].set_xlabel('Time Step', fontsize=11)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f"online_{scenario_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def main():
    """
    Main evaluation function.
    """
    print("=" * 70)
    print("ONLINE ADAPTIVE CNN-LSTM EVALUATION")
    print("=" * 70)
    
    # Scenarios to evaluate
    scenarios = {
        'no_drift': ('data/raw', 'fold'),
        'noise_variance': ('data/drift_scenarios', 'fold_noise_variance'),
        'mean_shift': ('data/drift_scenarios', 'fold_mean_shift'),
        'scale': ('data/drift_scenarios', 'fold_scale')
    }
    
    all_results = {}
    
    for scenario_name, (data_path, dynamics_type) in scenarios.items():
        scores, labels, regime_stats, adaptation_stats, adaptations = evaluate_online_adaptive(
            data_path=data_path,
            dynamics_type=dynamics_type
        )
        
        all_results[scenario_name] = {
            'scores': scores,
            'labels': labels,
            'regime_stats': regime_stats,
            'adaptation_stats': adaptation_stats,
            'adaptations': adaptations
        }
        
        # Visualize
        visualize_online_adaptive(
            scores, labels, adaptations,
            scenario_name,
            save_dir="results/online_adaptive"
        )
    
    # Save statistics
    stats_summary = {}
    for scenario_name, results in all_results.items():
        stats_summary[scenario_name] = {
            'regime_stats': results['regime_stats'],
            'adaptation_stats': results['adaptation_stats']
        }
    
    with open("results/online_adaptive/online_stats_summary.json", 'w') as f:
        json.dump(stats_summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print("\nResults saved to results/online_adaptive/")


if __name__ == "__main__":
    main()
