"""
CNN-LSTM Forecaster for Self-Supervised Early Warning

Trains a CNN-LSTM to forecast next day from last 30 days.
Prediction error magnitude serves as warning signal.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CNNLSTMForecaster:
    """
    Self-supervised forecasting model for early warning signals.
    
    Architecture:
        Input (30, 1) → Conv1D(16, k=5) → MaxPool → LSTM(32) → Dense(1)
    
    Warning score = |actual - predicted|
    """
    
    def __init__(self, window_size=30):
        """
        Initialize forecaster.
        
        Args:
            window_size: Number of past days to use for forecasting
        """
        self.window_size = window_size
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build CNN-LSTM architecture."""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.window_size, 1)),
            
            # Conv1D layer
            layers.Conv1D(filters=16, kernel_size=5, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            
            # LSTM layer
            layers.LSTM(32),
            
            # Output layer (linear for regression)
            layers.Dense(1, activation='linear')
        ])
        
        # Compile with MSE loss
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32, verbose=1):
        """
        Train forecaster.
        
        Args:
            X_train: Training windows (n_samples, window_size)
            y_train: Training targets (n_samples,)
            X_val: Validation windows (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Reshape inputs
        X_train = X_train.reshape(-1, self.window_size, 1)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val = X_val.reshape(-1, self.window_size, 1)
            validation_data = (X_val, y_val)
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """
        Generate predictions.
        
        Args:
            X: Input windows (n_samples, window_size)
            
        Returns:
            Predictions (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() or train() first.")
        
        X = X.reshape(-1, self.window_size, 1)
        predictions = self.model.predict(X, verbose=0).flatten()
        
        return predictions
    
    def compute_residuals(self, X, y_true):
        """
        Compute prediction residuals (warning scores).
        
        Args:
            X: Input windows
            y_true: True values
            
        Returns:
            Residuals (absolute prediction errors)
        """
        predictions = self.predict(X)
        residuals = np.abs(y_true - predictions)
        
        return residuals
    
    def save(self, path):
        """Save model to file."""
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path):
        """Load model from file."""
        self.model = keras.models.load_model(path)


if __name__ == "__main__":
    # Test forecaster
    print("Testing CNN-LSTM Forecaster...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    data = np.sin(np.linspace(0, 10, n_samples)) + np.random.normal(0, 0.1, n_samples)
    
    # Create windows
    window_size = 30
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split train/test
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train forecaster
    forecaster = CNNLSTMForecaster(window_size=window_size)
    forecaster.train(X_train, y_train, X_test, y_test, epochs=10, verbose=1)
    
    # Compute residuals
    train_residuals = forecaster.compute_residuals(X_train, y_train)
    test_residuals = forecaster.compute_residuals(X_test, y_test)
    
    print(f"\n✓ Forecaster trained successfully")
    print(f"  Train residual: {np.mean(train_residuals):.4f} ± {np.std(train_residuals):.4f}")
    print(f"  Test residual: {np.mean(test_residuals):.4f} ± {np.std(test_residuals):.4f}")
