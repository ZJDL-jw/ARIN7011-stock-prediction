"""Download stock data using yfinance."""
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from src.utils.io import load_config, ensure_dir


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download data for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        start: Start date
        end: End date
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True)
        
        if df.empty:
            print(f"Warning: No data for {ticker}")
            return pd.DataFrame()
        
        # Rename columns to lowercase
        df.columns = [c.lower() for c in df.columns]
        df.index.name = 'date'
        df.reset_index(inplace=True)
        df['ticker'] = ticker
        
        return df
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return pd.DataFrame()


def download_all(config: dict):
    """Download data for all tickers and index.
    
    Args:
        config: Configuration dictionary
    """
    tickers = config['data']['tickers']
    index = config['data']['index']
    start = config['data']['start']
    end = config['data']['end']
    out_dir = config['data']['out_dir']
    
    raw_dir = Path(out_dir) / 'raw'
    ensure_dir(str(raw_dir))
    
    all_data = []
    
    # Download stocks
    print(f"Downloading {len(tickers)} stocks...")
    for ticker in tqdm(tickers):
        df = download_ticker(ticker, start, end)
        if not df.empty:
            df.to_csv(raw_dir / f"{ticker.replace('.', '_')}.csv", index=False)
            all_data.append(df)
    
    # Download index
    if index:
        print(f"Downloading index {index}...")
        df_index = download_ticker(index, start, end)
        if not df_index.empty:
            df_index.to_csv(raw_dir / f"{index.replace('.', '_')}.csv", index=False)
            all_data.append(df_index)
    
    print(f"Downloaded {len(all_data)} datasets")
    return all_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    download_all(config)
    print("Download complete!")

