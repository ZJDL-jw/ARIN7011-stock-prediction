"""Evaluation metrics for classification and quantile regression."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from scipy.stats import norm
from scipy.optimize import minimize_scalar


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray, metric='f1') -> float:
    """Find optimal threshold for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        metric: Metric to optimize ('f1' or 'accuracy')
    
    Returns:
        Optimal threshold value
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    
    # Use more reasonable threshold range: 0.3 to 0.7 (avoid extreme thresholds)
    # This prevents all models from predicting the same class
    thresholds = np.linspace(0.3, 0.7, 41)
    best_threshold = 0.5
    best_score = -1
    
    # Also check if using median of predictions gives better balance
    median_pred = np.median(y_pred)
    if 0.3 <= median_pred <= 0.7:
        thresholds = np.concatenate([thresholds, [median_pred]])
    
    for threshold in thresholds:
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Check if predictions are balanced (not all 0 or all 1)
        unique_pred = np.unique(y_pred_binary)
        if len(unique_pred) < 2:
            continue  # Skip if all predictions are the same
        
        if metric == 'f1':
            try:
                score = f1_score(y_true, y_pred_binary)
            except:
                score = 0.0
        else:  # accuracy
            score = accuracy_score(y_true, y_pred_binary)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    # If no good threshold found, use median of predictions or 0.5
    if best_score <= 0:
        median_pred = np.median(y_pred)
        if 0.3 <= median_pred <= 0.7:
            best_threshold = median_pred
        else:
            best_threshold = 0.5
    
    return best_threshold


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Binary classification accuracy."""
    return accuracy_score(y_true, (y_pred > threshold).astype(int))


def f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Binary classification F1 score."""
    return f1_score(y_true, (y_pred > threshold).astype(int))


def auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ROC AUC score."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.0


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score for probability calibration."""
    return brier_score_loss(y_true, y_pred)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Pinball loss for quantile regression.
    
    Args:
        y_true: True values
        y_pred: Predicted quantile values
        tau: Quantile level (0 < tau < 1)
    """
    error = y_true - y_pred
    return np.mean(np.maximum(tau * error, (tau - 1) * error))


def crps_approx(y_true: np.ndarray, quantiles: dict) -> float:
    """Approximate CRPS using multiple quantiles.
    
    Args:
        y_true: True values
        quantiles: Dict mapping tau -> predicted quantile values
    """
    taus = sorted(quantiles.keys())
    if len(taus) < 2:
        return np.nan
    
    crps = 0.0
    for i, tau in enumerate(taus):
        q_pred = quantiles[tau]
        if i == 0:
            # First quantile
            crps += tau * np.mean(np.maximum(y_true - q_pred, 0))
        elif i == len(taus) - 1:
            # Last quantile
            crps += (1 - tau) * np.mean(np.maximum(q_pred - y_true, 0))
        else:
            # Middle quantiles
            q_prev = quantiles[taus[i-1]]
            crps += (tau - taus[i-1]) * np.mean(np.maximum(y_true - q_pred, 0))
    
    return crps


def compute_all_metrics(y_true_cls: dict, y_pred_cls: dict,
                       y_true_reg: dict, y_pred_reg: dict,
                       horizons: list, quantiles: list,
                       optimal_thresholds: dict = None) -> dict:
    """Compute all metrics for all horizons and quantiles.
    
    Args:
        y_true_cls: True classification labels
        y_pred_cls: Predicted classification probabilities
        y_true_reg: True regression values
        y_pred_reg: Predicted quantile values
        horizons: List of horizons
        quantiles: List of quantile levels
        optimal_thresholds: Optional dict of optimal thresholds per horizon
    
    Returns:
        Dictionary with metrics for each horizon
    """
    results = {}
    
    for h in horizons:
        # Find optimal threshold if not provided
        if optimal_thresholds is None or h not in optimal_thresholds:
            threshold = find_optimal_threshold(y_true_cls[h], y_pred_cls[h], metric='f1')
        else:
            threshold = optimal_thresholds[h]
        
        results[f'h{h}'] = {
            'threshold': threshold,
            'accuracy': accuracy(y_true_cls[h], y_pred_cls[h], threshold),
            'f1': f1(y_true_cls[h], y_pred_cls[h], threshold),
            'auc': auc(y_true_cls[h], y_pred_cls[h]),
            'brier': brier_score(y_true_cls[h], y_pred_cls[h]),
        }
        
        # Quantile metrics - improved key matching
        quantile_results = {}
        quantile_dict = {}
        
        for tau in quantiles:
            # Try to find the key in multiple formats
            pred_array = None
            for key in y_pred_reg[h].keys():
                try:
                    if abs(float(key) - tau) < 1e-6:
                        pred_array = y_pred_reg[h][key]
                        break
                except (ValueError, TypeError):
                    if key == tau or key == str(tau):
                        pred_array = y_pred_reg[h][key]
                        break
            
            if pred_array is not None:
                # Ensure arrays are same length
                min_len = min(len(y_true_reg[h]), len(pred_array))
                y_true_trimmed = y_true_reg[h][:min_len]
                pred_trimmed = pred_array[:min_len]
                
                # Remove NaN and inf
                valid_mask = np.isfinite(y_true_trimmed) & np.isfinite(pred_trimmed)
                if np.sum(valid_mask) > 0:
                    y_true_clean = y_true_trimmed[valid_mask]
                    pred_clean = pred_trimmed[valid_mask]
                    
                    try:
                        quantile_results[f'pinball_{tau}'] = pinball_loss(
                            y_true_clean, pred_clean, tau
                        )
                        quantile_dict[tau] = pred_clean
                    except Exception as e:
                        quantile_results[f'pinball_{tau}'] = np.nan
                else:
                    quantile_results[f'pinball_{tau}'] = np.nan
            else:
                quantile_results[f'pinball_{tau}'] = np.nan
        
        # CRPS approximation
        if len(quantile_dict) >= 2:
            # Ensure all quantiles have same length
            min_len = min(len(y_true_reg[h]), *[len(q) for q in quantile_dict.values()])
            y_true_trimmed = y_true_reg[h][:min_len]
            quantile_dict_trimmed = {tau: arr[:min_len] for tau, arr in quantile_dict.items()}
            
            # Remove NaN and inf
            valid_mask = np.isfinite(y_true_trimmed)
            for tau in quantile_dict_trimmed:
                valid_mask = valid_mask & np.isfinite(quantile_dict_trimmed[tau])
            
            if np.sum(valid_mask) > 0:
                y_true_clean = y_true_trimmed[valid_mask]
                quantile_dict_clean = {tau: arr[valid_mask] for tau, arr in quantile_dict_trimmed.items()}
                try:
                    results[f'h{h}']['crps'] = crps_approx(y_true_clean, quantile_dict_clean)
                except Exception as e:
                    results[f'h{h}']['crps'] = np.nan
            else:
                results[f'h{h}']['crps'] = np.nan
        else:
            results[f'h{h}']['crps'] = np.nan
        
        results[f'h{h}'].update(quantile_results)
    
    return results

