"""
Classical Early Warning Signals (EWS) baseline.

Computes rolling variance and autocorrelation as indicators
of approaching critical transitions.
"""

import numpy as np
from typing import Tuple, Optional


class ClassicalEWS:
    """
    Classical early warning signals based on statistical indicators.
    
    Computes:
    - Rolling variance
    - Rolling lag-1 autocorrelation
    
    Combines into a continuous warning score.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize classical EWS detector.
        
        Args:
            window_size: Size of rolling window for statistics
        """
        self.window_size = window_size
        self.min_var = None
        self.max_var = None
        self.min_ac = None
        self.max_ac = None
        self.fitted = False
    
    def compute_rolling_variance(self, time_series: np.ndarray) -> np.ndarray:
        """
        Compute rolling variance.
        
        Args:
            time_series: 1D time series
            
        Returns:
            Rolling variance array
        """
        n = len(time_series)
        variance = np.zeros(n)
        variance[:self.window_size] = np.nan
        
        for i in range(self.window_size, n):
            window = time_series[i - self.window_size:i]
            variance[i] = np.var(window)
        
        return variance
    
    def compute_rolling_autocorrelation(self, time_series: np.ndarray) -> np.ndarray:
        """
        Compute rolling lag-1 autocorrelation.
        
        Args:
            time_series: 1D time series
            
        Returns:
            Rolling autocorrelation array
        """
        n = len(time_series)
        autocorr = np.zeros(n)
        autocorr[:self.window_size] = np.nan
        
        for i in range(self.window_size, n):
            window = time_series[i - self.window_size:i]
            
            # Lag-1 autocorrelation
            if len(window) > 1:
                x1 = window[:-1]
                x2 = window[1:]
                
                # Pearson correlation
                if np.std(x1) > 0 and np.std(x2) > 0:
                    autocorr[i] = np.corrcoef(x1, x2)[0, 1]
                else:
                    autocorr[i] = 0.0
            else:
                autocorr[i] = 0.0
        
        return autocorr
    
    def fit(self, time_series: np.ndarray) -> 'ClassicalEWS':
        """
        Fit normalizer on training data to learn min/max values.
        
        Args:
            time_series: Training time series of shape (n_samples, length)
            
        Returns:
            self
        """
        all_var = []
        all_ac = []
        
        for ts in time_series:
            # Skip if contains NaN
            if np.any(np.isnan(ts)):
                continue
            
            var = self.compute_rolling_variance(ts)
            ac = self.compute_rolling_autocorrelation(ts)
            
            # Collect non-NaN values
            all_var.extend(var[~np.isnan(var)])
            all_ac.extend(ac[~np.isnan(ac)])
        
        # Compute min/max for normalization
        all_var = np.array(all_var)
        all_ac = np.array(all_ac)
        
        self.min_var = np.percentile(all_var, 1)  # Use percentiles for robustness
        self.max_var = np.percentile(all_var, 99)
        self.min_ac = np.percentile(all_ac, 1)
        self.max_ac = np.percentile(all_ac, 99)
        
        self.fitted = True
        
        print(f"Fitted classical EWS:")
        print(f"  Variance range: [{self.min_var:.4f}, {self.max_var:.4f}]")
        print(f"  Autocorr range: [{self.min_ac:.4f}, {self.max_ac:.4f}]")
        
        return self
    
    def predict(self, time_series: np.ndarray) -> np.ndarray:
        """
        Compute warning scores for time series.
        
        Args:
            time_series: Time series of shape (length,) or (n_samples, length)
            
        Returns:
            Warning scores of same shape as input
        """
        if not self.fitted:
            raise ValueError("ClassicalEWS must be fitted before prediction")
        
        # Handle single time series
        if time_series.ndim == 1:
            time_series = time_series.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        warning_scores = np.zeros_like(time_series)
        
        for i, ts in enumerate(time_series):
            # Skip if contains NaN
            if np.any(np.isnan(ts)):
                warning_scores[i, :] = np.nan
                continue
            
            # Compute indicators
            variance = self.compute_rolling_variance(ts)
            autocorr = self.compute_rolling_autocorrelation(ts)
            
            # Normalize to [0, 1]
            var_norm = (variance - self.min_var) / (self.max_var - self.min_var + 1e-10)
            ac_norm = (autocorr - self.min_ac) / (self.max_ac - self.min_ac + 1e-10)
            
            # Clip to [0, 1]
            var_norm = np.clip(var_norm, 0, 1)
            ac_norm = np.clip(ac_norm, 0, 1)
            
            # Combine into warning score (equal weighting)
            warning_score = 0.5 * var_norm + 0.5 * ac_norm
            
            warning_scores[i, :] = warning_score
        
        if squeeze_output:
            warning_scores = warning_scores.squeeze()
        
        return warning_scores
    
    def fit_predict(self, time_series: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step.
        
        Args:
            time_series: Time series data
            
        Returns:
            Warning scores
        """
        self.fit(time_series)
        return self.predict(time_series)


def evaluate_classical_ews(
    time_series: np.ndarray,
    labels: np.ndarray,
    window_size: int = 100
) -> Tuple[np.ndarray, dict]:
    """
    Evaluate classical EWS on test data.
    
    Args:
        time_series: Test time series of shape (n_samples, length)
        labels: True labels of shape (n_samples, length)
        window_size: Rolling window size
        
    Returns:
        warning_scores: Predicted warning scores
        metrics: Dictionary of evaluation metrics
    """
    # Initialize and fit
    ews = ClassicalEWS(window_size=window_size)
    ews.fit(time_series)
    
    # Predict
    warning_scores = ews.predict(time_series)
    
    # Compute metrics
    # Flatten for metric computation
    scores_flat = warning_scores.flatten()
    labels_flat = labels.flatten()
    
    # Remove NaN values
    valid_mask = ~np.isnan(scores_flat)
    scores_flat = scores_flat[valid_mask]
    labels_flat = labels_flat[valid_mask]
    
    # Compute correlation with labels
    correlation = np.corrcoef(scores_flat, labels_flat)[0, 1]
    
    # Compute mean score per regime
    regime_scores = {}
    for regime in [0, 1, 2]:
        regime_mask = labels_flat == regime
        if np.any(regime_mask):
            regime_scores[regime] = np.mean(scores_flat[regime_mask])
    
    metrics = {
        'correlation': correlation,
        'mean_score_stable': regime_scores.get(0, np.nan),
        'mean_score_approaching': regime_scores.get(1, np.nan),
        'mean_score_post': regime_scores.get(2, np.nan)
    }
    
    return warning_scores, metrics


if __name__ == "__main__":
    # Test classical EWS
    print("Testing Classical EWS...")
    
    # Load data
    time_series = np.load("data/raw/time_series_fold.npy")
    labels = np.load("data/raw/labels_fold.npy")
    
    print(f"Data shape: {time_series.shape}")
    
    # Split train/test
    n_train = 80
    train_ts = time_series[:n_train]
    test_ts = time_series[n_train:]
    test_labels = labels[n_train:]
    
    # Evaluate
    warning_scores, metrics = evaluate_classical_ews(
        train_ts, labels[:n_train], window_size=100
    )
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test on single time series
    print("\nTesting on single time series...")
    ews = ClassicalEWS(window_size=100)
    ews.fit(train_ts)
    
    test_score = ews.predict(test_ts[0])
    print(f"Warning score shape: {test_score.shape}")
    print(f"Score range: [{np.nanmin(test_score):.4f}, {np.nanmax(test_score):.4f}]")
