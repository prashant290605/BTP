"""
Offline CNN-LSTM baseline model for early warning signals.

Architecture:
- CNN layers for feature extraction
- LSTM layers for temporal modeling
- Dense output for warning score prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from typing import Tuple, Optional, Dict
import os
import json


class CNNLSTM_EWS:
    """
    Offline CNN-LSTM model for early warning signal detection.
    
    Trained once on full dataset, no online updates.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        cnn_filters: Tuple[int, int] = (32, 64),
        lstm_units: Tuple[int, int] = (50, 50),
        dense_units: int = 25,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001
    ):
        """
        Initialize CNN-LSTM model.
        
        Args:
            window_size: Input window size
            cnn_filters: Number of filters for each CNN layer
            lstm_units: Number of units for each LSTM layer
            dense_units: Number of units in dense layer
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        self.window_size = window_size
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.history = None
        self.build_model()
    
    def build_model(self):
        """Build CNN-LSTM architecture."""
        
        # Input layer
        inputs = layers.Input(shape=(self.window_size, 1))
        
        # CNN Block 1
        x = layers.Conv1D(
            filters=self.cnn_filters[0],
            kernel_size=3,
            activation='relu',
            padding='same'
        )(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        # CNN Block 2
        x = layers.Conv1D(
            filters=self.cnn_filters[1],
            kernel_size=3,
            activation='relu',
            padding='same'
        )(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        # LSTM Block
        x = layers.LSTM(
            units=self.lstm_units[0],
            return_sequences=True
        )(x)
        x = layers.LSTM(
            units=self.lstm_units[1],
            return_sequences=False
        )(x)
        
        # Dense Output
        x = layers.Dense(units=self.dense_units, activation='relu')(x)
        x = layers.Dropout(rate=self.dropout_rate)(x)
        outputs = layers.Dense(units=1, activation='sigmoid')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['mae', 'mse']
        )
        
        print("Model built successfully!")
        self.model.summary()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        verbose: int = 1
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training windows of shape (n_samples, window_size, 1)
            y_train: Training labels of shape (n_samples,)
            X_val: Validation windows
            y_val: Validation labels
            epochs: Maximum number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train
        print(f"\nTraining CNN-LSTM model...")
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict warning scores.
        
        Args:
            X: Input windows of shape (n_samples, window_size, 1)
            
        Returns:
            Warning scores of shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.squeeze()
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test windows
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Compute metrics
        loss, mae, mse = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Additional metrics
        correlation = np.corrcoef(y_pred, y_test)[0, 1]
        
        # Mean scores per regime
        regime_scores = {}
        for regime_val in [0.0, 0.5, 1.0]:
            regime_mask = y_test == regime_val
            if np.any(regime_mask):
                regime_scores[regime_val] = np.mean(y_pred[regime_mask])
        
        metrics = {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'correlation': correlation,
            'mean_score_stable': regime_scores.get(0.0, np.nan),
            'mean_score_approaching': regime_scores.get(0.5, np.nan),
            'mean_score_post': regime_scores.get(1.0, np.nan)
        }
        
        return metrics
    
    def save(self, filepath: str):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model (without extension)
        """
        # Save model weights
        self.model.save(f"{filepath}.keras")
        
        # Save configuration
        config = {
            'window_size': self.window_size,
            'cnn_filters': self.cnn_filters,
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }
        
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'CNNLSTM_EWS':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model (without extension)
            
        Returns:
            Loaded model instance
        """
        # Load configuration
        with open(f"{filepath}_config.json", 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(**config)
        
        # Load weights
        instance.model = keras.models.load_model(f"{filepath}.keras")
        
        print(f"Model loaded from {filepath}")
        return instance


def train_cnn_lstm_baseline(
    data_path: str = "data/raw",
    dynamics_type: str = "fold",
    window_size: int = 100,
    test_size: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    save_path: Optional[str] = None
) -> Tuple[CNNLSTM_EWS, Dict]:
    """
    Train CNN-LSTM baseline model.
    
    Args:
        data_path: Path to data directory
        dynamics_type: Type of dynamics
        window_size: Rolling window size
        test_size: Fraction for validation
        epochs: Training epochs
        batch_size: Batch size
        save_path: Path to save model (optional)
        
    Returns:
        Trained model and metrics
    """
    # Import preprocessing here to avoid circular import
    import sys
    sys.path.append('src')
    from utils.preprocessing import load_and_prepare_data
    
    # Load and prepare data
    print("Loading and preparing data...")
    X_train, X_val, y_train, y_val, normalizer = load_and_prepare_data(
        data_path=data_path,
        dynamics_type=dynamics_type,
        window_size=window_size,
        test_size=test_size
    )
    
    # Create and train model
    model = CNNLSTM_EWS(window_size=window_size)
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate
    print("\nEvaluating on validation set...")
    metrics = model.evaluate(X_val, y_val)
    
    print("\nValidation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save if requested
    if save_path:
        model.save(save_path)
    
    return model, metrics


if __name__ == "__main__":
    # Train baseline model
    print("Training CNN-LSTM Baseline Model")
    print("=" * 60)
    
    model, metrics = train_cnn_lstm_baseline(
        data_path="data/raw",
        dynamics_type="fold",
        window_size=100,
        test_size=0.2,
        epochs=50,
        batch_size=32,
        save_path="models/cnn_lstm_baseline"
    )
    
    print("\nTraining complete!")
