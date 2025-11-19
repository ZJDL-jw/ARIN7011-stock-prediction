"""Preprocess raw stock data: alignment, winsorization, rolling normalization."""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from scipy.stats import mstats
from tqdm import tqdm
from src.utils.io import load_config, ensure_dir


def load_raw_data(data_dir: str, ticker: str) -> pd.DataFrame:
    """Load raw CSV data for a ticker."""
    file_path = Path(data_dir) / 'raw' / f"{ticker.replace('.', '_')}.csv"
    if not file_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


def winsorize_series(series: pd.Series, pct: float = 0.01) -> pd.Series:
    """Winsorize series at pct and (1-pct) percentiles."""
    lower = series.quantile(pct)
    upper = series.quantile(1 - pct)
    return series.clip(lower=lower, upper=upper)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score using historical mean/std (no lookahead).
    
    For each point t, uses data from [t-window, t-1] to compute mean/std.
    """
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(len(series)):
        if i < window:
            # Not enough history, use available data
            window_data = series.iloc[:i]
        else:
            # Use [i-window, i-1] (exclude current point)
            window_data = series.iloc[i-window:i]
        
        if len(window_data) > 0:
            mean = window_data.mean()
            std = window_data.std()
            if std > 0:
                result.iloc[i] = (series.iloc[i] - mean) / std
            else:
                result.iloc[i] = 0.0
        else:
            result.iloc[i] = 0.0
    
    return result


def preprocess_ticker(data_dir: str, ticker: str, config: dict) -> pd.DataFrame:
    """Preprocess data for a single ticker.
    
    Steps:
    1. Load raw data
    2. Compute log returns
    3. Winsorize at 1%-99%
    4. Rolling z-score normalization
    """
    df = load_raw_data(data_dir, ticker)
    if df.empty:
        return df
    
    # Compute log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Winsorize key columns
    winsor_pct = config['features']['winsor_pct']
    for col in ['close', 'volume', 'log_return']:
        if col in df.columns:
            df[col] = winsorize_series(df[col], winsor_pct)
    
    # Rolling z-score normalization
    zscore_window = config['features']['zscore_window']
    df['close_z'] = rolling_zscore(df['close'], zscore_window)
    df['volume_z'] = rolling_zscore(df['volume'], zscore_window)
    df['log_return_z'] = rolling_zscore(df['log_return'], zscore_window)
    
    return df


def align_dates(dataframes: list) -> pd.DataFrame:
    """Align all dataframes to common date index."""
    if not dataframes:
        return pd.DataFrame()
    
    # Get common date range
    all_dates = set(dataframes[0].index)
    for df in dataframes[1:]:
        all_dates = all_dates.intersection(set(df.index))
    
    common_dates = sorted(list(all_dates))
    
    # Align all dataframes
    aligned = []
    for df in dataframes:
        df_aligned = df.loc[common_dates].copy()
        aligned.append(df_aligned)
    
    return pd.concat(aligned, axis=0, keys=[df['ticker'].iloc[0] for df in aligned])


def preprocess_all(config: dict):
    """Preprocess all tickers."""
    tickers = config['data']['tickers']
    index = config['data']['index']
    data_dir = config['data']['out_dir']
    
    interim_dir = Path(data_dir) / 'interim'
    ensure_dir(str(interim_dir))
    
    all_dfs = []
    
    # Preprocess stocks
    print("Preprocessing stocks...")
    for ticker in tqdm(tickers):
        df = preprocess_ticker(data_dir, ticker, config)
        if not df.empty:
            df.to_csv(interim_dir / f"{ticker.replace('.', '_')}_preprocessed.csv")
            all_dfs.append(df)
    
    # Preprocess index
    if index:
        print(f"Preprocessing index {index}...")
        df_index = preprocess_ticker(data_dir, index, config)
        if not df_index.empty:
            df_index.to_csv(interim_dir / f"{index.replace('.', '_')}_preprocessed.csv")
            all_dfs.append(df_index)
    
    print(f"Preprocessed {len(all_dfs)} datasets")
    return all_dfs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--factors', type=str, default='configs/factors.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    factors_config = load_config(args.factors)
    config['factors'] = factors_config
    
    preprocess_all(config)
    print("Preprocessing complete!")

