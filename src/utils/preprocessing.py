"""
Preprocessing utilities for time series data.

Includes:
- Rolling window creation
- Data normalization
- Train/test splitting
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


def create_rolling_windows(
    time_series: np.ndarray,
    labels: np.ndarray,
    window_size: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create rolling windows from time series data.
    
    Args:
        time_series: Time series array of shape (n_samples, length)
        labels: Label array of shape (n_samples, length)
        window_size: Size of rolling window
        
    Returns:
        X: Windows of shape (n_windows, window_size, 1)
        y: Labels of shape (n_windows,)
    """
    X_list = []
    y_list = []
    
    for i in range(time_series.shape[0]):
        ts = time_series[i]
        label = labels[i]
        
        # Skip if contains NaN
        if np.any(np.isnan(ts)):
            continue
        
        # Create windows for this time series
        for j in range(window_size, len(ts)):
            window = ts[j - window_size:j]
            X_list.append(window)
            y_list.append(label[j])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Reshape for CNN input: (n_windows, window_size, 1)
    X = X.reshape(-1, window_size, 1)
    
    return X, y


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """
    Convert regime labels to warning scores.
    
    0 (stable) -> 0.0
    1 (approaching) -> 0.5
    2 (post-transition) -> 1.0
    
    Args:
        labels: Integer labels (0, 1, 2)
        
    Returns:
        Normalized labels as floats
    """
    label_map = {0: 0.0, 1: 0.5, 2: 1.0}
    return np.vectorize(label_map.get)(labels).astype(np.float32)


def train_test_split_windows(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split windowed data into train and test sets.
    
    Args:
        X: Window features
        y: Labels
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


class TimeSeriesNormalizer:
    """
    Normalizer for time series windows using StandardScaler.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'TimeSeriesNormalizer':
        """
        Fit normalizer on training data.
        
        Args:
            X: Training windows of shape (n_samples, window_size, 1)
            
        Returns:
            self
        """
        # Reshape to 2D for StandardScaler
        X_2d = X.reshape(-1, X.shape[1])
        self.scaler.fit(X_2d)
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted normalizer.
        
        Args:
            X: Windows of shape (n_samples, window_size, 1)
            
        Returns:
            Normalized windows
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        original_shape = X.shape
        X_2d = X.reshape(-1, X.shape[1])
        X_normalized = self.scaler.transform(X_2d)
        return X_normalized.reshape(original_shape)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Training windows
            
        Returns:
            Normalized windows
        """
        self.fit(X)
        return self.transform(X)


def load_and_prepare_data(
    data_path: str = "data/raw",
    dynamics_type: str = "fold",
    window_size: int = 100,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TimeSeriesNormalizer]:
    """
    Load data and prepare for model training.
    
    Args:
        data_path: Path to data directory
        dynamics_type: Type of dynamics ('fold', 'saddle_node', 'hopf')
        window_size: Rolling window size
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test, normalizer
    """
    import os
    
    # Load data
    ts_file = os.path.join(data_path, f"time_series_{dynamics_type}.npy")
    label_file = os.path.join(data_path, f"labels_{dynamics_type}.npy")
    
    time_series = np.load(ts_file)
    labels = np.load(label_file)
    
    print(f"Loaded {dynamics_type} data: {time_series.shape}")
    
    # Create rolling windows
    print(f"Creating rolling windows (size={window_size})...")
    X, y = create_rolling_windows(time_series, labels, window_size)
    print(f"Created {len(X)} windows")
    
    # Normalize labels
    y = normalize_labels(y)
    
    # Split into train/test
    print(f"Splitting into train/test (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split_windows(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set: {len(X_train)} windows")
    print(f"Test set: {len(X_test)} windows")
    
    # Normalize features
    print("Normalizing features...")
    normalizer = TimeSeriesNormalizer()
    X_train = normalizer.fit_transform(X_train)
    X_test = normalizer.transform(X_test)
    
    return X_train, X_test, y_train, y_test, normalizer


if __name__ == "__main__":
    # Test preprocessing
    X_train, X_test, y_train, y_test, normalizer = load_and_prepare_data()
    
    print("\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    print("\nLabel distribution (train):")
    print(f"  0.0 (stable): {np.sum(y_train == 0.0)}")
    print(f"  0.5 (approaching): {np.sum(y_train == 0.5)}")
    print(f"  1.0 (post-transition): {np.sum(y_train == 1.0)}")
