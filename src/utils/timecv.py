"""Time series cross-validation utilities."""
import numpy as np
import pandas as pd
from typing import List, Tuple


def walk_forward_split(dates: pd.DatetimeIndex, train_end: str, 
                      test_start: str, test_end: str) -> Tuple[np.ndarray, np.ndarray]:
    """Split data into train and test sets based on dates.
    
    Args:
        dates: DatetimeIndex of all dates
        train_end: Last date for training (inclusive)
        test_start: First date for testing (inclusive)
        test_end: Last date for testing (inclusive)
    
    Returns:
        train_indices, test_indices
    """
    dates = pd.to_datetime(dates)
    train_end = pd.to_datetime(train_end)
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)
    
    train_mask = dates <= train_end
    test_mask = (dates >= test_start) & (dates <= test_end)
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    return train_indices, test_indices


def annual_walk_forward(dates: pd.DatetimeIndex, start_year: int, 
                       end_year: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate annual walk-forward splits.
    
    Args:
        dates: DatetimeIndex of all dates
        start_year: First year for testing
        end_year: Last year for testing
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    dates = pd.to_datetime(dates)
    splits = []
    
    for year in range(start_year, end_year + 1):
        train_end = f"{year-1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"
        
        train_idx, test_idx = walk_forward_split(dates, train_end, test_start, test_end)
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    
    return splits


def market_regime_split(dates: pd.DatetimeIndex, index_returns: pd.Series,
                       window: int = 60) -> dict:
    """Split test period into bull/bear/sideways regimes.
    
    Args:
        dates: DatetimeIndex
        index_returns: Index returns series
        window: Rolling window for regime classification
    
    Returns:
        Dict mapping regime -> indices
    """
    dates = pd.to_datetime(dates)
    rolling_ret = index_returns.rolling(window=window).sum()
    
    regimes = {
        'bull': [],
        'bear': [],
        'sideways': []
    }
    
    threshold_high = rolling_ret.quantile(0.67)
    threshold_low = rolling_ret.quantile(0.33)
    
    for i, date in enumerate(dates):
        if date not in rolling_ret.index:
            continue
        
        ret = rolling_ret.loc[date]
        if ret > threshold_high:
            regimes['bull'].append(i)
        elif ret < threshold_low:
            regimes['bear'].append(i)
        else:
            regimes['sideways'].append(i)
    
    return {k: np.array(v) for k, v in regimes.items()}

