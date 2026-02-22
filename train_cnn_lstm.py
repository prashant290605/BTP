"""
Offline CNN-LSTM Early Warning System Baseline

Exact architecture as specified:
- Conv1D(16, kernel=5, relu) → MaxPool(2)
- Conv1D(32, kernel=3, relu) → MaxPool(2)
- LSTM(32)
- Dense(1, sigmoid)

Train once, no online updates.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
import json


def build_cnn_lstm_model(window_size=100):
    """
    Build CNN-LSTM model with exact specified architecture.
    
    Args:
        window_size: Input window size (default: 100)
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(window_size, 1)),
        
        # CNN Block 1
        layers.Conv1D(filters=16, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        # CNN Block 2
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        # LSTM layer
        layers.LSTM(units=32),
        
        # Dense output
        layers.Dense(units=1, activation='sigmoid')
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['mae', 'mse']
    )
    
    return model


def load_and_prepare_data(data_path="data/raw", dynamics_type="fold", window_size=100):
    """
    Load data and create rolling windows.
    
    Args:
        data_path: Path to data directory
        dynamics_type: Type of dynamics
        window_size: Window size
        
    Returns:
        X_train, X_val, y_train, y_val, X_test, y_test, normalizer_mean, normalizer_std
    """
    # Load data
    ts_file = os.path.join(data_path, f"time_series_{dynamics_type}.npy")
    label_file = os.path.join(data_path, f"labels_{dynamics_type}.npy")
    
    time_series = np.load(ts_file)
    labels = np.load(label_file)
    
    print(f"Loaded {dynamics_type} data: {time_series.shape}")
    
    # Split: 70% train, 15% val, 15% test
    n_total = len(time_series)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    train_ts = time_series[:n_train]
    train_labels = labels[:n_train]
    
    val_ts = time_series[n_train:n_train+n_val]
    val_labels = labels[n_train:n_train+n_val]
    
    test_ts = time_series[n_train+n_val:]
    test_labels = labels[n_train+n_val:]
    
    print(f"Train: {len(train_ts)}, Val: {len(val_ts)}, Test: {len(test_ts)}")
    
    # Create rolling windows
    def create_windows(ts_array, label_array):
        X_list = []
        y_list = []
        
        for i in range(len(ts_array)):
            ts = ts_array[i]
            label = label_array[i]
            
            # Skip if contains NaN
            if np.any(np.isnan(ts)):
                continue
            
            for j in range(window_size, len(ts)):
                window = ts[j - window_size:j]
                X_list.append(window)
                # Map labels: 0→0.0, 1→0.5, 2→1.0
                y_list.append(label[j] * 0.5)
        
        return np.array(X_list), np.array(y_list)
    
    print("Creating rolling windows...")
    X_train, y_train = create_windows(train_ts, train_labels)
    X_val, y_val = create_windows(val_ts, val_labels)
    X_test, y_test = create_windows(test_ts, test_labels)
    
    print(f"Train windows: {len(X_train)}")
    print(f"Val windows: {len(X_val)}")
    print(f"Test windows: {len(X_test)}")
    
    # Reshape for CNN: (n_samples, window_size, 1)
    X_train = X_train.reshape(-1, window_size, 1)
    X_val = X_val.reshape(-1, window_size, 1)
    X_test = X_test.reshape(-1, window_size, 1)
    
    # Normalize using training statistics
    mean = np.mean(X_train)
    std = np.std(X_train)
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    print(f"\nNormalization: mean={mean:.4f}, std={std:.4f}")
    
    return X_train, X_val, y_train, y_val, X_test, y_test, mean, std


def train_model(
    X_train, y_train,
    X_val, y_val,
    window_size=100,
    epochs=30,
    batch_size=64,
    save_path="models/cnn_lstm_offline"
):
    """
    Train CNN-LSTM model.
    
    Args:
        X_train: Training windows
        y_train: Training labels
        X_val: Validation windows
        y_val: Validation labels
        window_size: Window size
        epochs: Number of epochs
        batch_size: Batch size
        save_path: Path to save model
        
    Returns:
        Trained model and history
    """
    # Build model
    print("\nBuilding CNN-LSTM model...")
    model = build_cnn_lstm_model(window_size=window_size)
    model.summary()
    
    # Train
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(f"{save_path}.keras")
    
    # Save history
    with open(f"{save_path}_history.json", 'w') as f:
        json.dump(history.history, f)
    
    print(f"\nModel saved to {save_path}.keras")
    
    return model, history


if __name__ == "__main__":
    print("=" * 70)
    print("OFFLINE CNN-LSTM BASELINE - TRAINING")
    print("=" * 70)
    
    # Load and prepare data
    X_train, X_val, y_train, y_val, X_test, y_test, mean, std = load_and_prepare_data(
        data_path="data/raw",
        dynamics_type="fold",
        window_size=100
    )
    
    # Train model
    model, history = train_model(
        X_train, y_train,
        X_val, y_val,
        window_size=100,
        epochs=30,
        batch_size=64,
        save_path="models/cnn_lstm_offline"
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)
    
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    # Predict on test set
    y_pred = model.predict(X_test, verbose=0).squeeze()
    
    # Compute statistics per regime
    print("\n" + "-" * 70)
    print("WARNING SCORES BY REGIME")
    print("-" * 70)
    
    for regime_val, regime_name in [(0.0, "Stable"), (0.5, "Approaching"), (1.0, "Post-transition")]:
        mask = y_test == regime_val
        if np.any(mask):
            scores = y_pred[mask]
            print(f"\n{regime_name} (target={regime_val}):")
            print(f"  Mean: {np.mean(scores):.4f}")
            print(f"  Std:  {np.std(scores):.4f}")
            print(f"  Min:  {np.min(scores):.4f}")
            print(f"  Max:  {np.max(scores):.4f}")
    
    # Save normalization parameters
    norm_params = {'mean': float(mean), 'std': float(std)}
    with open("models/cnn_lstm_offline_norm.json", 'w') as f:
        json.dump(norm_params, f)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext: Run evaluation script to visualize warning scores")
