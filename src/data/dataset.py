"""Dataset construction: sliding windows and labels."""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from typing import Tuple, Dict
from src.utils.io import load_config, ensure_dir
from src.utils.timecv import walk_forward_split
from src.data.factors import build_factor_dataset


def compute_labels(df: pd.DataFrame, horizons: list, quantiles: list) -> Dict:
    """Compute classification and regression labels.
    
    Args:
        df: DataFrame with 'close' column
        horizons: List of prediction horizons (e.g., [1, 7, 30])
        quantiles: List of quantile levels (e.g., [0.1, 0.5, 0.9])
    
    Returns:
        Dict with 'classification' and 'regression' labels
    """
    labels = {
        'classification': {},
        'regression': {}
    }
    
    # Compute returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    for h in horizons:
        # Forward return: r_{t -> t+h} (sum of next h days)
        # Shift by -h to get return from t to t+h
        forward_ret = log_returns.rolling(window=h).sum().shift(-h)
        
        # Improved classification: use adaptive threshold based on percentile
        # Different thresholds for different horizons (日线/周线/月线)
        if h == 1:
            # 日线：使用55th百分位数，更平衡的类别
            percentile_threshold = forward_ret.quantile(0.55)
            median_threshold = forward_ret.median()
            threshold = percentile_threshold if not np.isnan(percentile_threshold) else median_threshold
        elif h == 7:
            # 周线：使用60th百分位数
            percentile_threshold = forward_ret.quantile(0.6)
            median_threshold = forward_ret.median()
            threshold = percentile_threshold if not np.isnan(percentile_threshold) else median_threshold
        else:  # h == 30 (月线)
            # 月线：使用60th百分位数
            percentile_threshold = forward_ret.quantile(0.6)
            median_threshold = forward_ret.median()
            threshold = percentile_threshold if not np.isnan(percentile_threshold) else median_threshold
        
        if np.isnan(threshold):
            threshold = 0.0
        labels['classification'][h] = (forward_ret > threshold).astype(int).values
        
        # Regression: actual return values (handle NaN)
        labels['regression'][h] = np.nan_to_num(forward_ret.values, nan=0.0)
        
        # Also handle NaN in classification labels
        labels['classification'][h] = np.nan_to_num(
            labels['classification'][h], nan=0
        )
        
        # Quantile labels (for training, we'll predict quantiles)
        labels['quantiles'] = quantiles
    
    return labels


class StockDataset(Dataset):
    """Dataset for stock prediction with sliding windows."""
    
    def __init__(self, features: np.ndarray, labels: Dict, indices: np.ndarray = None):
        """
        Args:
            features: Array of shape [n_samples, T, D]
            labels: Dict with classification and regression labels
            indices: Optional indices to subset
        """
        self.features = features
        self.labels = labels
        self.indices = indices if indices is not None else np.arange(len(features))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        x = torch.FloatTensor(self.features[actual_idx])
        
        # Get labels for all horizons
        y_cls = {}
        y_reg = {}
        
        for h in self.labels['classification'].keys():
            y_cls[h] = torch.LongTensor([self.labels['classification'][h][actual_idx]])
            y_reg[h] = torch.FloatTensor([self.labels['regression'][h][actual_idx]])
        
        return {
            'features': x,
            'classification': y_cls,
            'regression': y_reg
        }


def build_sliding_windows(factor_df: pd.DataFrame, window_T: int,
                         horizons: list, quantiles: list) -> Tuple[np.ndarray, Dict]:
    """Build sliding window samples.
    
    Args:
        factor_df: DataFrame with factors, indexed by date, with 'ticker' column
        window_T: Window length
        horizons: Prediction horizons
        quantiles: Quantile levels
    
    Returns:
        features: Array [n_samples, T, D]
        labels: Dict with classification and regression labels
    """
    # Get factor columns (exclude 'ticker' and 'close')
    factor_cols = [c for c in factor_df.columns if c not in ['ticker', 'close']]
    n_features = len(factor_cols)
    
    all_features = []
    all_labels_cls = {h: [] for h in horizons}
    all_labels_reg = {h: [] for h in horizons}
    
    # Process each ticker separately
    for ticker in factor_df['ticker'].unique():
        ticker_df = factor_df[factor_df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_index()
        
        # Extract factor values (exclude 'close' and 'ticker')
        factor_cols_clean = [c for c in factor_cols if c not in ['close', 'ticker']]
        factor_values = ticker_df[factor_cols_clean].values
        
        # Fill NaN with 0
        factor_values = np.nan_to_num(factor_values, nan=0.0)
        
        # Compute labels from close prices
        if 'close' not in ticker_df.columns:
            print(f"Warning: Close prices not found for {ticker}. Using dummy values.")
            ticker_df['close'] = 100.0
        
        # Compute labels
        labels = compute_labels(ticker_df, horizons, quantiles)
        
        # Build sliding windows
        n_samples = len(factor_values) - window_T + 1
        if n_samples <= 0:
            continue
        
        for i in range(n_samples):
            window_features = factor_values[i:i+window_T]
            all_features.append(window_features)
            
            label_idx = i + window_T - 1
            if label_idx < len(labels['classification'][horizons[0]]):
                for h in horizons:
                    all_labels_cls[h].append(labels['classification'][h][label_idx])
                    all_labels_reg[h].append(labels['regression'][h][label_idx])
        
        # Adjust labels to match window indices
        # Labels should align with the last timestep of each window
        for h in horizons:
            if len(all_labels_cls[h]) < len(all_features):
                # Pad or trim
                n_needed = len(all_features) - len(all_labels_cls[h])
                if n_needed > 0:
                    # Pad with last value or 0
                    last_val_cls = all_labels_cls[h][-1] if all_labels_cls[h] else 0
                    last_val_reg = all_labels_reg[h][-1] if all_labels_reg[h] else 0.0
                    all_labels_cls[h].extend([last_val_cls] * n_needed)
                    all_labels_reg[h].extend([last_val_reg] * n_needed)
    
    # Convert to arrays
    features = np.array(all_features)  # [n_samples, T, D]
    
    labels_dict = {
        'classification': {h: np.array(all_labels_cls[h]) for h in horizons},
        'regression': {h: np.array(all_labels_reg[h]) for h in horizons},
        'quantiles': quantiles
    }
    
    return features, labels_dict


def load_dataset(config: dict):
    """Load preprocessed dataset."""
    import pandas as pd
    data_dir = config['data']['out_dir']
    processed_dir = Path(data_dir) / 'processed'
    
    # Load data
    data = np.load(processed_dir / 'dataset.npz', allow_pickle=True)
    features = data['features']
    
    labels_cls = {int(k): v for k, v in data['labels_cls'].item().items()}
    labels_reg = {int(k): v for k, v in data['labels_reg'].item().items()}
    
    labels = {
        'classification': labels_cls,
        'regression': labels_reg,
        'quantiles': data['quantiles'].tolist()
    }
    
    # Load dates for splitting
    dates = np.load(processed_dir / 'dates.npy')
    dates = pd.to_datetime(dates)
    
    return features, labels, dates


def build_dataset(config: dict):
    """Main function to build and save dataset."""
    # Load factors
    data_dir = config['data']['out_dir']
    processed_dir = Path(data_dir) / 'processed'
    
    factor_path = processed_dir / 'factors.csv'
    if not factor_path.exists():
        print("Factors not found. Building factors first...")
        build_factor_dataset(config)
    
    factor_df = pd.read_csv(factor_path, index_col=0, parse_dates=True)
    
    # We need close prices for labels - load from interim data
    tickers = config['data']['tickers']
    interim_dir = Path(data_dir) / 'interim'
    
    # Add close prices to factor_df
    for ticker in tickers:
        ticker_path = interim_dir / f"{ticker.replace('.', '_')}_preprocessed.csv"
        if ticker_path.exists():
            df_ticker = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
            ticker_mask = factor_df['ticker'] == ticker
            ticker_factors = factor_df[ticker_mask].copy()
            
            # Align dates
            common_dates = ticker_factors.index.intersection(df_ticker.index)
            if len(common_dates) > 0:
                close_values = df_ticker.loc[common_dates, 'close']
                # Map back to factor_df
                for date in common_dates:
                    factor_df.loc[(factor_df.index == date) & (factor_df['ticker'] == ticker), 'close'] = close_values.loc[date]
    
    # If close not found, use a workaround
    if 'close' not in factor_df.columns:
        print("Warning: Close prices not found. Using ret_5 as proxy.")
        # This is a limitation - in production we'd ensure close is available
        factor_df['close'] = 100.0
    else:
        # Fill NaN with forward fill then backward fill
        factor_df['close'] = factor_df.groupby('ticker')['close'].fillna(method='ffill').fillna(method='bfill')
        factor_df['close'] = factor_df['close'].fillna(100.0)
    
    # Build sliding windows
    window_T = config['features']['window_T']
    horizons = config['features']['horizons']
    quantiles = config['features']['quantiles']
    
    print("Building sliding windows...")
    features, labels = build_sliding_windows(factor_df, window_T, horizons, quantiles)
    
    # Save
    np.savez_compressed(
        processed_dir / 'dataset.npz',
        features=features,
        labels_cls={k: v for k, v in labels['classification'].items()},
        labels_reg={k: v for k, v in labels['regression'].items()},
        quantiles=quantiles
    )
    
    # Also save date index for splitting
    dates = factor_df.index.unique()
    np.save(processed_dir / 'dates.npy', dates.values)
    
    print(f"Dataset built: {features.shape} samples")
    return features, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--factors', type=str, default='configs/factors.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    factors_config = load_config(args.factors)
    config['factors'] = factors_config
    
    if args.build:
        build_dataset(config)
        print("Dataset construction complete!")

