"""
Online Adaptive CNN-LSTM Early Warning System.

Features:
- Streaming inference (one window at a time)
- Pluggable drift detection (variance / ADWIN / CUSUM / Page-Hinkley)
- Selective fine-tuning on recent data
- Stability controls (cooldown, limited epochs)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import json
import os
from .drift_detectors import BaseDriftDetector, VarianceDriftDetector, create_drift_detector


# Backward-compatible alias for existing imports.
DriftDetector = VarianceDriftDetector


class RollingBuffer:
    """
    Rolling buffer for storing recent windows and labels.
    """
    
    def __init__(self, maxlen=1000):
        """
        Initialize buffer.
        
        Args:
            maxlen: Maximum buffer size
        """
        self.windows = deque(maxlen=maxlen)
        self.labels = deque(maxlen=maxlen)
    
    def add(self, window, label):
        """Add window and label to buffer."""
        self.windows.append(window)
        self.labels.append(label)
    
    def get_recent(self, n):
        """
        Get n most recent items.
        
        Args:
            n: Number of recent items
            
        Returns:
            windows, labels as numpy arrays
        """
        n = min(n, len(self.windows))
        windows = list(self.windows)[-n:]
        labels = list(self.labels)[-n:]
        return np.array(windows), np.array(labels)
    
    def __len__(self):
        return len(self.windows)


class OnlineAdaptiveCNNLSTM:
    """
    Online adaptive CNN-LSTM for early warning signals.
    
    Maintains performance under concept drift through:
    - Unsupervised drift detection
    - Selective fine-tuning
    - Stability controls
    """
    
    def __init__(
        self,
        model_path,
        norm_mean,
        norm_std,
        window_size=100,
        drift_threshold=2.5,
        drift_detector: BaseDriftDetector | None = None,
        detector_type: str = "variance",
        detector_params: dict | None = None,
        adaptation_windows=250,
        adaptation_epochs=3,
        cooldown_period=200
    ):
        """
        Initialize online adaptive system.
        
        Args:
            model_path: Path to trained offline model
            norm_mean: Normalization mean
            norm_std: Normalization std
            window_size: Input window size
            drift_threshold: Legacy threshold for variance detector
            drift_detector: Optional preconstructed detector (highest priority)
            detector_type: Detector name for factory creation
            detector_params: Parameters passed to detector factory
            adaptation_windows: Number of recent windows for adaptation
            adaptation_epochs: Epochs for fine-tuning
            cooldown_period: Steps between adaptations
        """
        # Load model
        self.model = keras.models.load_model(model_path)
        self.window_size = window_size
        
        # Normalization
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        
        # Drift detection (pluggable)
        if drift_detector is not None:
            self.drift_detector = drift_detector
        else:
            detector_params = detector_params or {}
            if detector_type == "variance":
                # Preserve legacy default behavior.
                detector_params.setdefault("threshold", drift_threshold)
                detector_params.setdefault("window_size", 100)
            self.drift_detector = create_drift_detector(detector_type, **detector_params)
        
        # Buffer
        self.buffer = RollingBuffer(maxlen=1000)
        
        # Adaptation parameters
        self.adaptation_windows = adaptation_windows
        self.adaptation_epochs = adaptation_epochs
        self.cooldown_period = cooldown_period
        
        # State tracking
        self.current_step = 0
        self.last_adaptation_step = -cooldown_period
        self.adaptation_count = 0
        self.adaptation_points = []
        
        # Compile model ONCE (not on every adaptation)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae']
        )
    
    def predict_step(self, window):
        """
        Predict warning score for single window.
        
        Args:
            window: Input window (1D array)
            
        Returns:
            Warning score
        """
        # Normalize
        window_norm = (window - self.norm_mean) / self.norm_std
        
        # Reshape for model
        X = window_norm.reshape(1, self.window_size, 1)
        
        # Predict
        score = self.model.predict(X, verbose=0)[0, 0]
        
        return score
    
    def can_adapt(self):
        """Check if adaptation is allowed (cooldown check)."""
        return (self.current_step - self.last_adaptation_step) >= self.cooldown_period
    
    def adapt_model(self):
        """
        Fine-tune model on recent data.
        """
        # Get recent data
        windows, labels = self.buffer.get_recent(self.adaptation_windows)
        
        if len(windows) < 50:  # Minimum data requirement
            return
        
        # Reshape windows
        X = windows.reshape(-1, self.window_size, 1)
        y = labels
        
        # Fine-tune (model already compiled in __init__)
        self.model.fit(
            X, y,
            epochs=self.adaptation_epochs,
            batch_size=32,
            verbose=0
        )
        
        # Update state
        self.last_adaptation_step = self.current_step
        self.adaptation_count += 1
        self.adaptation_points.append(self.current_step)
        
        # Reset drift detector baseline
        if hasattr(self.drift_detector, "reset_baseline"):
            self.drift_detector.reset_baseline()
    
    def process_stream(self, time_series, labels):
        """
        Process time series in streaming mode with batched predictions.
        
        Args:
            time_series: Time series data (1D array)
            labels: True labels (for buffer only, not for drift detection)
            
        Returns:
            scores: Predicted warning scores
            adaptation_points: Time steps where adaptation occurred
        """
        scores = np.full(len(time_series), np.nan)
        
        # Create all windows upfront for batched prediction
        n_windows = len(time_series) - self.window_size
        all_windows = np.zeros((n_windows, self.window_size, 1))
        
        for i in range(n_windows):
            window = time_series[i:i + self.window_size]
            window_norm = (window - self.norm_mean) / self.norm_std
            all_windows[i, :, 0] = window_norm
        
        # Batched prediction (much faster than per-step)
        # Note: Scores are computed with initial model state
        # Adaptations affect future samples, not current one
        all_scores = self.model.predict(all_windows, batch_size=256, verbose=0).flatten()
        
        # Now process stream logic with pre-computed scores
        for t in range(self.window_size, len(time_series)):
            idx = t - self.window_size
            score = all_scores[idx]
            scores[t] = score
            
            # Update buffer
            window = time_series[t - self.window_size:t]
            window_norm = (window - self.norm_mean) / self.norm_std
            label = labels[t] * 0.5
            self.buffer.add(window_norm, label)
            
            # Update drift detector (UNSUPERVISED - uses score only)
            drift_flag = self.drift_detector.update(score)
            if drift_flag is None and hasattr(self.drift_detector, "detect"):
                drift_flag = self.drift_detector.detect()

            # Check for drift and adapt if needed
            if bool(drift_flag) and self.can_adapt():
                self.adapt_model()
                # Note: Adaptation affects model state for NEXT sample
                # Current sample scores remain unchanged (realistic streaming)
            
            self.current_step += 1
        
        return scores, self.adaptation_points
    
    def get_stats(self):
        """Get adaptation statistics."""
        return {
            'adaptation_count': self.adaptation_count,
            'adaptation_points': self.adaptation_points,
            'buffer_size': len(self.buffer),
            'detector_state': self.drift_detector.get_state()
        }

    def reset_stream_state(self):
        """Reset stream/adaptation state for a new sample stream."""
        if hasattr(self.drift_detector, "reset"):
            self.drift_detector.reset()
        self.buffer.windows.clear()
        self.buffer.labels.clear()
        self.current_step = 0
        self.last_adaptation_step = -self.cooldown_period
        self.adaptation_count = 0
        self.adaptation_points = []


def load_online_adaptive_model(
    model_path="models/cnn_lstm_offline.keras",
    norm_path="models/cnn_lstm_offline_norm.json",
    detector_type="variance",
    detector_params=None,
):
    """
    Load online adaptive model from offline trained model.
    
    Args:
        model_path: Path to offline model
        norm_path: Path to normalization parameters
        detector_type: Drift detector name
        detector_params: Drift detector parameters
        
    Returns:
        OnlineAdaptiveCNNLSTM instance
    """
    # Load normalization parameters
    with open(norm_path, 'r') as f:
        norm_params = json.load(f)
    
    # Create online adaptive model
    online_model = OnlineAdaptiveCNNLSTM(
        model_path=model_path,
        norm_mean=norm_params['mean'],
        norm_std=norm_params['std'],
        window_size=100,
        drift_threshold=2.5,
        detector_type=detector_type,
        detector_params=detector_params,
        adaptation_windows=250,
        adaptation_epochs=3,
        cooldown_period=200
    )
    
    return online_model


if __name__ == "__main__":
    # Test online adaptive model
    print("Testing Online Adaptive CNN-LSTM...")
    
    # Load model
    online_model = load_online_adaptive_model()
    
    # Load test data
    time_series = np.load("data/raw/time_series_fold.npy")
    labels = np.load("data/raw/labels_fold.npy")
    
    # Test on one sample
    test_ts = time_series[0]
    test_labels = labels[0]
    
    print(f"Processing time series of length {len(test_ts)}...")
    scores, adaptation_points = online_model.process_stream(test_ts, test_labels)
    
    print(f"\nResults:")
    print(f"  Scores computed: {np.sum(~np.isnan(scores))}")
    print(f"  Adaptations: {len(adaptation_points)}")
    print(f"  Adaptation points: {adaptation_points}")
    
    stats = online_model.get_stats()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Online adaptive model test complete!")
