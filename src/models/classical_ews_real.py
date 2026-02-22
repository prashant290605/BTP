"""
Classical Early Warning Signals for Real Climate Data

Computes rolling variance and autocorrelation as early warning indicators.
"""

import numpy as np
import pandas as pd


def rolling_variance(data, window=30):
    """
    Compute rolling variance.
    
    Args:
        data: pandas Series or numpy array
        window: Window size for rolling computation
        
    Returns:
        pandas Series with rolling variance
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    return data.rolling(window=window, min_periods=window).var()


def rolling_autocorrelation(data, window=30, lag=1):
    """
    Compute rolling autocorrelation.
    
    Args:
        data: pandas Series or numpy array
        window: Window size for rolling computation
        lag: Lag for autocorrelation
        
    Returns:
        pandas Series with rolling autocorrelation
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    def autocorr(x):
        """Compute autocorrelation for a window."""
        if len(x) < lag + 1:
            return np.nan
        return pd.Series(x).autocorr(lag=lag)
    
    return data.rolling(window=window, min_periods=window).apply(autocorr, raw=False)


def normalize_indicator(indicator):
    """
    Normalize indicator to [0, 1] range using min-max scaling.
    
    Args:
        indicator: pandas Series
        
    Returns:
        Normalized indicator
    """
    min_val = indicator.min()
    max_val = indicator.max()
    
    if max_val == min_val:
        return pd.Series(np.zeros(len(indicator)), index=indicator.index)
    
    normalized = (indicator - min_val) / (max_val - min_val)
    return normalized


def combine_indicators(variance, autocorr, weights=(0.5, 0.5)):
    """
    Combine variance and autocorrelation into single warning score.
    
    Args:
        variance: pandas Series with variance indicator
        autocorr: pandas Series with autocorrelation indicator
        weights: Tuple of (variance_weight, autocorr_weight)
        
    Returns:
        Combined warning score
    """
    # Normalize both indicators
    var_norm = normalize_indicator(variance)
    ac_norm = normalize_indicator(autocorr)
    
    # Combine with weights
    combined = weights[0] * var_norm + weights[1] * ac_norm
    
    return combined


def compute_classical_ews(data, window=30, lag=1):
    """
    Compute all classical EWS indicators.
    
    Args:
        data: pandas Series or numpy array
        window: Window size
        lag: Lag for autocorrelation
        
    Returns:
        Dictionary with variance, autocorr, and combined indicators
    """
    # Compute indicators
    variance = rolling_variance(data, window=window)
    autocorr = rolling_autocorrelation(data, window=window, lag=lag)
    
    # Combine
    combined = combine_indicators(variance, autocorr)
    
    return {
        'variance': variance,
        'autocorrelation': autocorr,
        'combined': combined
    }


if __name__ == "__main__":
    # Test classical EWS
    print("Testing Classical EWS...")
    
    # Generate synthetic data with increasing variance
    np.random.seed(42)
    n = 200
    t = np.linspace(0, 10, n)
    noise_scale = np.linspace(0.1, 1.0, n)  # Increasing noise
    data = np.sin(t) + np.random.normal(0, noise_scale)
    
    # Compute EWS
    ews = compute_classical_ews(data, window=30)
    
    print(f"\n✓ Classical EWS computed")
    print(f"  Variance range: {ews['variance'].min():.4f} to {ews['variance'].max():.4f}")
    print(f"  Autocorr range: {ews['autocorrelation'].min():.4f} to {ews['autocorrelation'].max():.4f}")
    print(f"  Combined range: {ews['combined'].min():.4f} to {ews['combined'].max():.4f}")
    print(f"  Non-NaN samples: {ews['combined'].notna().sum()}/{len(data)}")
