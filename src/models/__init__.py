"""Baseline models for early warning signals."""

from .classical_ews import ClassicalEWS, evaluate_classical_ews

# Optional TensorFlow-dependent imports
try:
    from .cnn_lstm_baseline import CNNLSTM_EWS, train_cnn_lstm_baseline
    __all__ = [
        'ClassicalEWS',
        'evaluate_classical_ews',
        'CNNLSTM_EWS',
        'train_cnn_lstm_baseline'
    ]
except ImportError:
    __all__ = [
        'ClassicalEWS',
        'evaluate_classical_ews'
    ]
