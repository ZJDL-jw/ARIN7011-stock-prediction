"""Probability calibration utilities."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path


class Calibrator:
    """Probability calibrator using Isotonic Regression."""
    
    def __init__(self, method: str = 'isotonic'):
        """
        Args:
            method: 'isotonic' or 'platt'
        """
        self.method = method
        self.calibrators = {}  # One per horizon
    
    def fit(self, y_pred: np.ndarray, y_true: np.ndarray, horizon: int = 1):
        """Fit calibrator for a specific horizon.
        
        Args:
            y_pred: Uncalibrated probabilities
            y_true: True binary labels
            horizon: Prediction horizon
        """
        if self.method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.method == 'platt':
            calibrator = LogisticRegression()
            y_pred = y_pred.reshape(-1, 1)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        calibrator.fit(y_pred, y_true)
        self.calibrators[horizon] = calibrator
    
    def predict(self, y_pred: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Calibrate probabilities.
        
        Args:
            y_pred: Uncalibrated probabilities
            horizon: Prediction horizon
        
        Returns:
            Calibrated probabilities
        """
        if horizon not in self.calibrators:
            return y_pred
        
        calibrator = self.calibrators[horizon]
        
        if self.method == 'platt':
            y_pred = y_pred.reshape(-1, 1)
        
        calibrated = calibrator.predict(y_pred)
        
        # Ensure probabilities are in [0, 1]
        calibrated = np.clip(calibrated, 0, 1)
        
        return calibrated
    
    def save(self, path: str):
        """Save calibrator to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """Load calibrator from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)

