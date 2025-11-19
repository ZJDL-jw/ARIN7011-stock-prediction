"""Factor engineering: compute extensive factors + representation learning."""
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from typing import Dict, List
import warnings
from src.utils.io import load_config

# Suppress overflow warnings for numerical stability
warnings.filterwarnings('ignore', category=RuntimeWarning)


def compute_returns(df: pd.DataFrame, periods: List[int]) -> Dict[str, pd.Series]:
    """Compute returns for multiple periods.
    
    Args:
        df: DataFrame with 'close' column
        periods: List of periods (e.g., [1, 5, 10])
    
    Returns:
        Dict mapping period -> return series
    """
    results = {}
    for p in periods:
        results[f'ret_{p}'] = df['close'].pct_change(p)
    return results


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100.0  # Normalize to [0, 1]


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                 signal: int = 9) -> pd.Series:
    """Compute MACD histogram."""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, 
                            num_std: float = 2.0) -> pd.Series:
    """Compute Bollinger Band width."""
    ma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    bandwidth = (upper - lower) / ma
    return bandwidth


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR (Average True Range)."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    # Normalize by price
    atr_pct = atr / df['close']
    return atr_pct


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.Series:
    """Compute Stochastic Oscillator %K."""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    return stoch_k / 100.0  # Normalize to [0, 1]


def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Williams %R."""
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    wr = -100 * (high_max - df['close']) / (high_max - low_min)
    return wr / 100.0  # Normalize to [-1, 0]


def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute Commodity Channel Index."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    return cci / 100.0  # Normalize


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index."""
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    atr = compute_atr(df, period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx / 100.0  # Normalize


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Compute On-Balance Volume (normalized)."""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    # Normalize by rolling max
    obv_norm = obv / obv.rolling(window=60).max().replace(0, 1)
    return obv_norm


def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Money Flow Index."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + positive_sum / negative_sum.replace(0, 1)))
    return mfi / 100.0  # Normalize to [0, 1]


def compute_roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """Compute Rate of Change."""
    roc = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
    return roc / 100.0  # Normalize


def compute_momentum(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Compute Momentum."""
    momentum = df['close'] - df['close'].shift(period)
    # Normalize by price
    momentum_pct = momentum / df['close'].shift(period)
    return momentum_pct


def compute_ema_ratios(df: pd.DataFrame, periods: List[int] = [12, 26, 50]) -> Dict[str, pd.Series]:
    """Compute EMA ratios."""
    results = {}
    emas = {}
    for p in periods:
        emas[p] = df['close'].ewm(span=p, adjust=False).mean()
    
    # EMA ratios
    if len(periods) >= 2:
        results['ema_ratio_12_26'] = emas[12] / emas[26] - 1
    if len(periods) >= 3:
        results['ema_ratio_26_50'] = emas[26] / emas[50] - 1
        results['ema_ratio_12_50'] = emas[12] / emas[50] - 1
    
    return results


def compute_price_position(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute price position within recent range."""
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    position = (df['close'] - low_min) / (high_max - low_min).replace(0, 1)
    return position


def compute_volatility_ratios(df: pd.DataFrame, periods: List[int] = [10, 20, 60]) -> Dict[str, pd.Series]:
    """Compute volatility ratios at different periods."""
    results = {}
    log_ret = np.log(df['close'] / df['close'].shift(1))
    
    vols = {}
    for p in periods:
        vols[p] = log_ret.rolling(window=p).std()
    
    if len(periods) >= 2:
        results['vol_ratio_10_20'] = vols[10] / vols[20].replace(0, 1)
    if len(periods) >= 3:
        results['vol_ratio_20_60'] = vols[20] / vols[60].replace(0, 1)
        results['vol_ratio_10_60'] = vols[10] / vols[60].replace(0, 1)
    
    return results


def compute_trend_strength(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute trend strength using linear regression slope."""
    def slope(x):
        if len(x) < 2:
            return 0
        return np.polyfit(range(len(x)), x, 1)[0]
    
    trend = df['close'].rolling(window=period).apply(slope, raw=True)
    # Normalize by price
    trend_norm = trend / df['close']
    return trend_norm


def winsorize_factor(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize factor to remove extreme values."""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    """Safe division that handles zero and inf."""
    result = numerator / denominator.replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], default)
    return result.fillna(default)


def compute_all_factors(df: pd.DataFrame, index_df: pd.DataFrame = None,
                       config: dict = None) -> pd.DataFrame:
    """Compute extensive factors (40+ factors) with numerical stability.
    
    Args:
        df: Stock DataFrame with OHLCV
        index_df: Index DataFrame (optional)
        config: Configuration dict
    
    Returns:
        DataFrame with factor columns (preserves 'close' column)
    """
    factors_df = df.copy()
    # Preserve close column for label computation
    if 'close' not in factors_df.columns and 'close' in df.columns:
        factors_df['close'] = df['close']
    
    # === Price-based factors ===
    # 1-3. Returns (multiple periods)
    rets = compute_returns(df, [1, 5, 10, 20])
    for name, series in rets.items():
        factors_df[name] = series
    
    # 4. Reverse return
    factors_df['rev_1'] = -df['close'].pct_change(1)
    
    # 5. Short-term momentum (for Horizon 1 prediction)
    factors_df['momentum_1'] = df['close'].pct_change(1)
    factors_df['momentum_2'] = df['close'].pct_change(2)
    factors_df['momentum_3'] = df['close'].pct_change(3)
    
    # 6. Intraday volatility (high-low range)
    factors_df['intraday_range'] = (df['high'] - df['low']) / df['close']
    factors_df['intraday_range_ma3'] = factors_df['intraday_range'].rolling(3).mean()
    
    # 7. Price acceleration (second derivative)
    price_change = df['close'].pct_change(1)
    factors_df['price_acceleration'] = price_change.diff()
    
    # 8. Recent trend strength (short window)
    factors_df['trend_3'] = compute_trend_strength(df, period=3)
    factors_df['trend_5'] = compute_trend_strength(df, period=5)
    
    # 5-7. Moving average biases
    for period in [5, 10, 20]:
        ma = df['close'].rolling(window=period).mean()
        factors_df[f'bias_ma{period}'] = (df['close'] / ma) - 1
    
    # 8-9. EMA ratios
    ema_ratios = compute_ema_ratios(df, periods=[12, 26, 50])
    for name, series in ema_ratios.items():
        factors_df[name] = series
    
    # 10. Price position
    factors_df['price_position_20'] = compute_price_position(df, period=20)
    
    # === Momentum & Oscillators ===
    # 11. RSI (multiple periods)
    for period in [14, 21]:
        factors_df[f'rsi{period}'] = compute_rsi(df, period=period)
    
    # 12. MACD histogram
    factors_df['macd_hist'] = compute_macd(df)
    
    # 13-14. Stochastic & Williams %R
    factors_df['stoch_k'] = compute_stochastic(df, k_period=14)
    factors_df['williams_r'] = compute_williams_r(df, period=14)
    
    # 15. CCI
    factors_df['cci'] = compute_cci(df, period=20)
    
    # 16. ADX
    factors_df['adx'] = compute_adx(df, period=14)
    
    # 17. ROC
    factors_df['roc'] = compute_roc(df, period=12)
    
    # 18. Momentum
    factors_df['momentum'] = compute_momentum(df, period=10)
    
    # === Volatility factors ===
    # 19-21. Bollinger Bands (multiple periods)
    for period in [20, 30]:
        factors_df[f'bb_bw{period}'] = compute_bollinger_bands(df, period=period)
    
    # 22-24. ATR (multiple periods)
    for period in [14, 21]:
        factors_df[f'ATR{period}'] = compute_atr(df, period=period)
    
    # 25-27. Volatility (multiple periods)
    log_ret = np.log(df['close'] / df['close'].shift(1))
    for period in [10, 20, 30]:
        factors_df[f'volsigma_{period}'] = log_ret.rolling(window=period).std()
    
    # 28-30. Volatility ratios
    vol_ratios = compute_volatility_ratios(df, periods=[10, 20, 60])
    for name, series in vol_ratios.items():
        factors_df[name] = series
    
    # 31. Trend strength
    factors_df['trend_strength'] = compute_trend_strength(df, period=14)
    
    # === Volume factors ===
    # 32-34. Volume ratios (multiple periods)
    for period in [5, 10, 20]:
        ma_vol = df['volume'].rolling(window=period).mean()
        factors_df[f'vol_ma_ratio{period}'] = df['volume'] / ma_vol.replace(0, 1)
    
    # 35. Turnover
    factors_df['turnover'] = df['volume'] / df['close']
    
    # 36. OBV
    factors_df['obv'] = compute_obv(df)
    
    # 37. MFI
    factors_df['mfi'] = compute_mfi(df, period=14)
    
    # === Market relative factors ===
    # 38-40. Relative strength vs index (multiple periods)
    if index_df is not None and 'close' in index_df.columns:
        index_rets = compute_returns(index_df, [1, 5, 10])
        for period in [1, 5, 10]:
            ret_key = f'ret_{period}'
            if ret_key in rets and ret_key in index_rets:
                factors_df[f'rs_index_{period}'] = rets[ret_key] - index_rets[ret_key]
            else:
                factors_df[f'rs_index_{period}'] = 0.0
    else:
        for period in [1, 5, 10]:
            factors_df[f'rs_index_{period}'] = 0.0
    
    # Apply winsorization to all factors (except close and ticker) for numerical stability
    factor_cols = [col for col in factors_df.columns if col not in ['close', 'ticker']]
    for col in factor_cols:
        if col in factors_df.columns:
            # Clip extreme values to prevent overflow
            factors_df[col] = winsorize_factor(factors_df[col], lower=0.01, upper=0.99)
            # Replace any remaining inf/nan
            factors_df[col] = factors_df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            # Clip to reasonable range
            factors_df[col] = factors_df[col].clip(lower=-10.0, upper=10.0)
    
    return factors_df


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder for factor compression."""
    
    def __init__(self, input_dim: int, latent_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


def compute_ae_embeddings(factor_data: np.ndarray, window: int = 60,
                         latent_dim: int = 2, epochs: int = 50) -> np.ndarray:
    """Compute AE embeddings for factor data.
    
    Args:
        factor_data: Array of shape [n_samples, n_features]
        window: Training window size
        latent_dim: Latent dimension
        epochs: Training epochs
    
    Returns:
        Embeddings of shape [n_samples, latent_dim]
    """
    n_samples, n_features = factor_data.shape
    embeddings = np.zeros((n_samples, latent_dim))
    
    # Train AE on rolling windows
    device = torch.device('cpu')
    model = SimpleAutoencoder(n_features, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for i in range(window, n_samples):
        # Use [i-window, i-1] for training (no lookahead)
        train_data = factor_data[i-window:i].copy()
        
        # Handle inf and NaN, clip extreme values
        train_data = np.nan_to_num(train_data, nan=0.0, posinf=0.0, neginf=0.0)
        train_data = np.clip(train_data, -10.0, 10.0)
        
        # Normalize with robust method
        median = np.median(train_data, axis=0, keepdims=True)
        mad = np.median(np.abs(train_data - median), axis=0, keepdims=True) + 1e-8
        train_norm = (train_data - median) / mad
        # Clip after normalization
        train_norm = np.clip(train_norm, -5.0, 5.0)
        train_norm = np.nan_to_num(train_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Train for a few epochs
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            x_tensor = torch.FloatTensor(train_norm).to(device)
            recon, encoded = model(x_tensor)
            loss = criterion(recon, x_tensor)
            loss.backward()
            optimizer.step()
        
        # Encode current point
        model.eval()
        with torch.no_grad():
            current = factor_data[i:i+1].copy()
            current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
            current = np.clip(current, -10.0, 10.0)
            current_norm = (current - median) / mad
            current_norm = np.clip(current_norm, -5.0, 5.0)
            current_norm = np.nan_to_num(current_norm, nan=0.0, posinf=0.0, neginf=0.0)
            x_tensor = torch.FloatTensor(current_norm).to(device)
            _, encoded = model(x_tensor)
            encoded_np = encoded.cpu().numpy()[0]
            # Clip embeddings
            encoded_np = np.clip(encoded_np, -5.0, 5.0)
            embeddings[i] = encoded_np
    
    return embeddings


def compute_pca_factors(factor_data: np.ndarray, window: int = 60,
                       n_components: int = 5) -> np.ndarray:
    """Compute PCA factors using rolling window with robust scaling.
    
    Args:
        factor_data: Array of shape [n_samples, n_features]
        window: Training window size
        n_components: Number of PCA components
    
    Returns:
        PCA factors of shape [n_samples, n_components]
    """
    n_samples, n_features = factor_data.shape
    pca_factors = np.zeros((n_samples, n_components))
    
    # Use RobustScaler for better numerical stability
    scaler = RobustScaler()
    
    for i in range(window, n_samples):
        # Use [i-window, i-1] for training (no lookahead)
        train_data = factor_data[i-window:i].copy()
        
        # Handle inf and NaN, clip extreme values
        train_data = np.nan_to_num(train_data, nan=0.0, posinf=0.0, neginf=0.0)
        train_data = np.clip(train_data, -10.0, 10.0)
        
        # Normalize with robust scaler
        try:
            train_norm = scaler.fit_transform(train_data)
            # Clip after normalization to prevent overflow
            train_norm = np.clip(train_norm, -5.0, 5.0)
            train_norm = np.nan_to_num(train_norm, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            # If scaling fails, use raw data (already clipped)
            train_norm = train_data
        
        # Fit PCA
        try:
            n_comp = min(n_components, train_norm.shape[1], train_norm.shape[0])
            if n_comp < 1:
                pca_factors[i] = 0.0
                continue
                
            pca = PCA(n_components=n_comp)
            pca.fit(train_norm)
            
            # Transform current point
            current = factor_data[i:i+1].copy()
            current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
            current = np.clip(current, -10.0, 10.0)
            current_norm = scaler.transform(current)
            current_norm = np.clip(current_norm, -5.0, 5.0)
            current_norm = np.nan_to_num(current_norm, nan=0.0, posinf=0.0, neginf=0.0)
            
            transformed = pca.transform(current_norm)[0]
            # Clip transformed values
            transformed = np.clip(transformed, -5.0, 5.0)
            pca_factors[i, :len(transformed)] = transformed[:n_components]
        except Exception as e:
            # If PCA fails, use zeros
            pca_factors[i] = 0.0
    
    return pca_factors


def add_ae_factors(factors_df: pd.DataFrame, base_factor_cols: List[str],
                  window: int = 60, latent_dim: int = 5) -> pd.DataFrame:
    """Add AE embeddings as additional factors.
    
    Args:
        factors_df: DataFrame with base factors
        base_factor_cols: List of base factor column names
        window: Training window for AE
        latent_dim: Latent dimension for AE
    
    Returns:
        DataFrame with added 'ae_1', 'ae_2', ... columns
    """
    # Extract base factors
    factor_data = factors_df[base_factor_cols].values
    
    # Fill NaN with 0
    factor_data = np.nan_to_num(factor_data, nan=0.0)
    
    # Compute embeddings
    embeddings = compute_ae_embeddings(factor_data, window=window, latent_dim=latent_dim)
    
    for i in range(latent_dim):
        factors_df[f'ae_{i+1}'] = embeddings[:, i]
    
    return factors_df


def add_representation_factors(factors_df: pd.DataFrame, base_factor_cols: List[str],
                              window: int = 60, use_pca: bool = True, use_ae: bool = True,
                              pca_components: int = 5, ae_latent_dim: int = 5) -> pd.DataFrame:
    """Add representation learning factors (PCA + AE).
    
    Args:
        factors_df: DataFrame with base factors
        base_factor_cols: List of base factor column names
        window: Training window size
        use_pca: Whether to use PCA
        use_ae: Whether to use Autoencoder
        pca_components: Number of PCA components
        ae_latent_dim: Latent dimension for AE
    
    Returns:
        DataFrame with added representation factors
    """
    # Extract base factors
    factor_data = factors_df[base_factor_cols].values
    
    # Fill NaN with 0
    factor_data = np.nan_to_num(factor_data, nan=0.0)
    
    # Add PCA factors
    if use_pca:
        pca_factors = compute_pca_factors(factor_data, window=window, n_components=pca_components)
        for i in range(pca_components):
            factors_df[f'pca_{i+1}'] = pca_factors[:, i]
    
    # Add AE factors
    if use_ae:
        factors_df = add_ae_factors(factors_df, base_factor_cols, window=window, latent_dim=ae_latent_dim)
    
    return factors_df


def build_factor_dataset(config: dict) -> pd.DataFrame:
    """Build complete factor dataset for all tickers.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Combined DataFrame with all factors for all tickers
    """
    from pathlib import Path
    
    tickers = config['data']['tickers']
    index_ticker = config['data']['index']
    data_dir = config['data']['out_dir']
    factors_config = config.get('factors', {})
    
    use_pca = factors_config.get('use_pca', True)
    use_ae = factors_config.get('use_autoencoder', True)
    pca_components = factors_config.get('pca_components', 5)
    ae_latent_dim = factors_config.get('ae_latent_dim', 5)
    window = factors_config.get('representation_window', 60)
    
    interim_dir = Path(data_dir) / 'interim'
    
    # Load index data
    index_df = None
    if index_ticker:
        index_path = interim_dir / f"{index_ticker.replace('.', '_')}_preprocessed.csv"
        if index_path.exists():
            index_df = pd.read_csv(index_path, index_col=0, parse_dates=True)
    
    all_factors = []
    # Updated base factor columns (50+ factors including short-term features)
    base_factor_cols = [
        'ret_1', 'ret_5', 'ret_10', 'ret_20', 'rev_1',
        'momentum_1', 'momentum_2', 'momentum_3',  # Short-term momentum
        'intraday_range', 'intraday_range_ma3',  # Intraday volatility
        'price_acceleration',  # Price acceleration
        'trend_3', 'trend_5',  # Short-term trend
        'bias_ma5', 'bias_ma10', 'bias_ma20',
        'ema_ratio_12_26', 'ema_ratio_26_50', 'ema_ratio_12_50',
        'price_position_20',
        'rsi14', 'rsi21', 'macd_hist',
        'stoch_k', 'williams_r', 'cci', 'adx', 'roc', 'momentum',
        'bb_bw20', 'bb_bw30',
        'ATR14', 'ATR21',
        'volsigma_10', 'volsigma_20', 'volsigma_30',
        'vol_ratio_10_20', 'vol_ratio_20_60', 'vol_ratio_10_60',
        'trend_strength',
        'vol_ma_ratio5', 'vol_ma_ratio10', 'vol_ma_ratio20',
        'turnover', 'obv', 'mfi',
        'rs_index_1', 'rs_index_5', 'rs_index_10'
    ]
    
    for ticker in tickers:
        ticker_path = interim_dir / f"{ticker.replace('.', '_')}_preprocessed.csv"
        if not ticker_path.exists():
            continue
        
        df = pd.read_csv(ticker_path, index_col=0, parse_dates=True)
        
        # Compute base factors
        factors_df = compute_all_factors(df, index_df, config)
        
        # Add representation learning factors
        if use_pca or use_ae:
            factors_df = add_representation_factors(
                factors_df, base_factor_cols,
                window=window,
                use_pca=use_pca,
                use_ae=use_ae,
                pca_components=pca_components,
                ae_latent_dim=ae_latent_dim
            )
        
        # Collect all factor columns
        factor_cols = base_factor_cols.copy()
        if use_pca:
            factor_cols.extend([f'pca_{i+1}' for i in range(pca_components)])
        if use_ae:
            factor_cols.extend([f'ae_{i+1}' for i in range(ae_latent_dim)])
        
        # Select only factor columns (keep ticker for later use)
        selected_cols = [col for col in factor_cols if col in factors_df.columns]
        selected_cols.append('ticker')
        if 'close' in factors_df.columns:
            selected_cols.append('close')
        
        factors_df = factors_df[selected_cols].copy()
        factors_df['ticker'] = ticker
        
        all_factors.append(factors_df)
    
    # Combine
    combined = pd.concat(all_factors, axis=0)
    
    # Save
    processed_dir = Path(data_dir) / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(processed_dir / 'factors.csv')
    
    return combined

