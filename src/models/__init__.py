"""Baseline models for early warning signals."""

from .classical_ews import ClassicalEWS, evaluate_classical_ews
from .drift_detectors import (
    ADWINDetector,
    CUSUMDetector,
    PageHinkleyDetector,
    VarianceDriftDetector,
    create_drift_detector,
)

# Optional TensorFlow-dependent imports
try:
    from .cnn_lstm_baseline import CNNLSTM_EWS, train_cnn_lstm_baseline
    __all__ = [
        'ClassicalEWS',
        'evaluate_classical_ews',
        'VarianceDriftDetector',
        'ADWINDetector',
        'CUSUMDetector',
        'PageHinkleyDetector',
        'create_drift_detector',
        'CNNLSTM_EWS',
        'train_cnn_lstm_baseline'
    ]
except ImportError:
    __all__ = [
        'ClassicalEWS',
        'evaluate_classical_ews',
        'VarianceDriftDetector',
        'ADWINDetector',
        'CUSUMDetector',
        'PageHinkleyDetector',
        'create_drift_detector',
    ]
