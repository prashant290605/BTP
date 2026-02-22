"""
Evaluate Early Warning Signals on Real Climate Data

Analyzes warning signal stability under seasonal distribution shift using ERA5 data.

Research Question:
    Does a forecasting-based early warning signal remain stable under seasonal shifts,
    or does it show structural instability?

Approach:
    1. Load ERA5 2m temperature (2015, Delhi)
    2. Split: Train (Jan-Aug) / Test (Sep-Dec)
    3. Train CNN-LSTM forecaster on train period
    4. Compute prediction residuals as warning scores
    5. Compute classical EWS (variance + autocorr)
    6. Analyze distribution shift between train and test
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

from data.era5_loader import load_and_preprocess_era5, create_forecast_windows, normalize_series
from models.cnn_lstm_forecast import CNNLSTMForecaster
from models.classical_ews_real import compute_classical_ews


def split_train_test(data, train_months=8):
    """
    Split data into train (Jan-Aug) and test (Sep-Dec).
    
    Args:
        data: pandas Series with daily data
        train_months: Number of months for training (default: 8)
        
    Returns:
        train_data, test_data
    """
    # Assuming 365 days, ~30 days per month
    train_days = train_months * 30
    
    train_data = data[:train_days]
    test_data = data[train_days:]
    
    return train_data, test_data


def main():
    """Main evaluation pipeline."""
    
    print("="*70)
    print("REAL CLIMATE DATA EWS ANALYSIS")
    print("="*70)
    
    # Create results directory
    results_dir = Path("results/real_climate")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # 1. Load ERA5 Data
    # ========================================
    print("\n[1/6] Loading ERA5 data...")
    
    file_path = "Dataset/data_stream-oper_stepType-instant.nc"
    temp_daily = load_and_preprocess_era5(file_path)
    
    print(f"  [OK] Loaded {len(temp_daily)} daily samples")
    print(f"  Date range: {temp_daily.index[0]} to {temp_daily.index[-1]}")
    print(f"  Temperature: {temp_daily.min():.2f}C to {temp_daily.max():.2f}C (mean: {temp_daily.mean():.2f}C)")
    
    # ========================================
    # 2. Split Train/Test
    # ========================================
    print("\n[2/6] Splitting train/test...")
    
    train_data, test_data = split_train_test(temp_daily, train_months=8)
    
    print(f"  Train: {len(train_data)} days (Jan-Aug)")
    print(f"  Test: {len(test_data)} days (Sep-Dec)")
    
    # ========================================
    # 3. Normalize and Create Windows
    # ========================================
    print("\n[3/6] Creating forecast windows...")
    
    # Normalize using train statistics
    train_norm, train_mean, train_std = normalize_series(train_data, return_params=True)
    test_norm = (test_data - train_mean) / train_std
    
    # Create windows
    window_size = 30
    X_train, y_train = create_forecast_windows(train_norm, window_size=window_size)
    X_test, y_test = create_forecast_windows(test_norm, window_size=window_size)
    
    print(f"  Window size: {window_size} days")
    print(f"  Train windows: {len(X_train)}")
    print(f"  Test windows: {len(X_test)}")
    
    # ========================================
    # 4. Train CNN-LSTM Forecaster
    # ========================================
    print("\n[4/6] Training CNN-LSTM forecaster...")
    
    forecaster = CNNLSTMForecaster(window_size=window_size)
    history = forecaster.train(
        X_train, y_train,
        X_test, y_test,
        epochs=20,
        batch_size=32,
        verbose=0
    )
    
    train_loss = history.history['loss'][-1]
    test_loss = history.history['val_loss'][-1]
    
    print(f"  [OK] Training complete")
    print(f"  Final train loss: {train_loss:.4f}")
    print(f"  Final test loss: {test_loss:.4f}")
    
    # ========================================
    # 5. Compute Warning Scores
    # ========================================
    print("\n[5/6] Computing warning scores...")
    
    # CNN residuals
    train_residuals = forecaster.compute_residuals(X_train, y_train)
    test_residuals = forecaster.compute_residuals(X_test, y_test)
    
    print(f"  CNN-LSTM residuals:")
    print(f"    Train: {np.mean(train_residuals):.4f} +/- {np.std(train_residuals):.4f}")
    print(f"    Test: {np.mean(test_residuals):.4f} +/- {np.std(test_residuals):.4f}")
    
    # Classical EWS
    ews_train = compute_classical_ews(train_data, window=30)
    ews_test = compute_classical_ews(test_data, window=30)
    
    print(f"  Classical EWS computed")
    
    # ========================================
    # 6. Analyze Distribution Shift
    # ========================================
    print("\n[6/6] Analyzing distribution shift...")
    
    # Monthly statistics
    monthly_stats = {}
    
    # Group by month
    temp_with_month = temp_daily.to_frame('temp')
    temp_with_month['month'] = temp_with_month.index.month
    
    for month in range(1, 13):
        month_data = temp_with_month[temp_with_month['month'] == month]['temp']
        if len(month_data) > 0:
            monthly_stats[int(month)] = {
                'mean': float(month_data.mean()),
                'std': float(month_data.std()),
                'min': float(month_data.min()),
                'max': float(month_data.max())
            }
    
    # Residual statistics
    residual_stats = {
        'train': {
            'mean': float(np.mean(train_residuals)),
            'std': float(np.std(train_residuals)),
            'median': float(np.median(train_residuals))
        },
        'test': {
            'mean': float(np.mean(test_residuals)),
            'std': float(np.std(test_residuals)),
            'median': float(np.median(test_residuals))
        }
    }
    
    # Distribution shift assessment
    residual_shift = np.mean(test_residuals) / np.mean(train_residuals) if np.mean(train_residuals) > 0 else 0
    
    print(f"\n  Distribution Shift Analysis:")
    print(f"    Residual ratio (test/train): {residual_shift:.2f}x")
    
    if residual_shift > 1.5:
        print(f"    [WARNING] Significant distribution shift detected!")
    elif residual_shift > 1.2:
        print(f"    [INFO] Moderate distribution shift detected")
    else:
        print(f"    [OK] Warning signal relatively stable")
    
    # ========================================
    # 7. Generate Plots
    # ========================================
    print("\n[7/7] Generating plots...")
    
    # Plot 1: Temperature time series
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(temp_daily.index, temp_daily.values, linewidth=1, color='darkblue')
    ax.axvline(train_data.index[-1], color='red', linestyle='--', linewidth=2, label='Train/Test Split')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Temperature (C)', fontsize=11)
    ax.set_title('ERA5 2m Temperature - Delhi 2015', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / 'temperature_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Classical EWS
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    # Variance
    axes[0].plot(ews_train['variance'].index, ews_train['variance'].values, label='Train', color='blue', linewidth=1)
    axes[0].plot(ews_test['variance'].index, ews_test['variance'].values, label='Test', color='orange', linewidth=1)
    axes[0].set_ylabel('Rolling Variance', fontsize=10)
    axes[0].set_title('Classical EWS: Variance', fontsize=11, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Autocorrelation
    axes[1].plot(ews_train['autocorrelation'].index, ews_train['autocorrelation'].values, label='Train', color='blue', linewidth=1)
    axes[1].plot(ews_test['autocorrelation'].index, ews_test['autocorrelation'].values, label='Test', color='orange', linewidth=1)
    axes[1].set_xlabel('Date', fontsize=10)
    axes[1].set_ylabel('Rolling Autocorrelation', fontsize=10)
    axes[1].set_title('Classical EWS: Autocorrelation (lag=1)', fontsize=11, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'classical_ews.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: CNN residual warning score
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Create time indices for residuals (offset by window_size)
    train_indices = train_data.index[window_size:]
    test_indices = test_data.index[window_size:]
    
    ax.plot(train_indices, train_residuals, label='Train Residuals', color='blue', linewidth=1, alpha=0.7)
    ax.plot(test_indices, test_residuals, label='Test Residuals', color='orange', linewidth=1, alpha=0.7)
    ax.axhline(np.mean(train_residuals), color='blue', linestyle='--', linewidth=1.5, label=f'Train Mean: {np.mean(train_residuals):.3f}')
    ax.axhline(np.mean(test_residuals), color='orange', linestyle='--', linewidth=1.5, label=f'Test Mean: {np.mean(test_residuals):.3f}')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Prediction Error (Residual)', fontsize=11)
    ax.set_title('CNN-LSTM Forecasting Residuals (Warning Score)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'cnn_residual.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Comparison overlay
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    
    # Temperature
    axes[0].plot(temp_daily.index, temp_daily.values, linewidth=1, color='darkblue')
    axes[0].axvline(train_data.index[-1], color='red', linestyle='--', linewidth=2, alpha=0.5)
    axes[0].set_ylabel('Temperature (C)', fontsize=10)
    axes[0].set_title('Temperature Time Series', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Classical EWS (combined)
    axes[1].plot(ews_train['combined'].index, ews_train['combined'].values, label='Train', color='blue', linewidth=1)
    axes[1].plot(ews_test['combined'].index, ews_test['combined'].values, label='Test', color='orange', linewidth=1)
    axes[1].set_ylabel('Classical EWS', fontsize=10)
    axes[1].set_title('Classical EWS (Variance + Autocorr)', fontsize=11, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # CNN residuals
    axes[2].plot(train_indices, train_residuals, label='Train', color='blue', linewidth=1, alpha=0.7)
    axes[2].plot(test_indices, test_residuals, label='Test', color='orange', linewidth=1, alpha=0.7)
    axes[2].set_xlabel('Date', fontsize=10)
    axes[2].set_ylabel('CNN Residual', fontsize=10)
    axes[2].set_title('CNN-LSTM Forecasting Residuals', fontsize=11, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison_overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Plots saved to {results_dir}/")
    
    # ========================================
    # 8. Save Statistics
    # ========================================
    statistics = {
        'data_summary': {
            'total_days': int(len(temp_daily)),
            'train_days': int(len(train_data)),
            'test_days': int(len(test_data)),
            'temp_min': float(temp_daily.min()),
            'temp_max': float(temp_daily.max()),
            'temp_mean': float(temp_daily.mean()),
            'temp_std': float(temp_daily.std())
        },
        'model_performance': {
            'final_train_loss': float(train_loss),
            'final_test_loss': float(test_loss)
        },
        'residual_statistics': residual_stats,
        'monthly_statistics': monthly_stats,
        'distribution_shift': {
            'residual_ratio': float(residual_shift),
            'shift_detected': bool(residual_shift > 1.2)
        }
    }
    
    with open(results_dir / 'monthly_statistics.json', 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"  [OK] Statistics saved to {results_dir}/monthly_statistics.json")
    
    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    print(f"\nData Summary:")
    print(f"  - Total samples: {len(temp_daily)} days")
    print(f"  - Train/Test split: {len(train_data)}/{len(test_data)} days")
    print(f"  - Temperature range: {temp_daily.min():.2f}C to {temp_daily.max():.2f}C")
    
    print(f"\nModel Performance:")
    print(f"  - Train loss: {train_loss:.4f}")
    print(f"  - Test loss: {test_loss:.4f}")
    
    print(f"\nWarning Score Analysis:")
    print(f"  - Train residual: {np.mean(train_residuals):.4f} +/- {np.std(train_residuals):.4f}")
    print(f"  - Test residual: {np.mean(test_residuals):.4f} +/- {np.std(test_residuals):.4f}")
    print(f"  - Residual ratio (test/train): {residual_shift:.2f}x")
    
    print(f"\nDistribution Shift Assessment:")
    if residual_shift > 1.5:
        print(f"  [WARNING] SIGNIFICANT SHIFT: Warning signal shows structural instability")
        print(f"      Forecasting errors increase substantially in test period")
    elif residual_shift > 1.2:
        print(f"  [INFO] MODERATE SHIFT: Warning signal shows some instability")
        print(f"      Forecasting errors moderately higher in test period")
    else:
        print(f"  [OK] STABLE: Warning signal remains relatively stable")
        print(f"      Forecasting errors consistent across train/test")
    
    print(f"\nResults saved to: {results_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
