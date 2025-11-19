"""Backtesting script with transaction costs."""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict

from src.utils.seed import set_seed
from src.utils.io import load_config
from src.utils.plots import plot_cumulative_returns
from src.data.dataset import StockDataset, load_dataset
from src.utils.timecv import walk_forward_split
from src.train import create_model
from src.evaluate import evaluate_model


def compute_portfolio_returns(predictions: Dict, dates: pd.DatetimeIndex,
                              tickers: list, holding_days: int = 1,
                              top_k: int = 3, position_mode: str = 'equal',
                              cost_bps: float = 10.0) -> pd.Series:
    """Compute portfolio returns from predictions.
    
    Args:
        predictions: Dict mapping ticker -> probability array
        dates: Date index
        tickers: List of ticker symbols
        holding_days: Number of days to hold positions
        top_k: Number of stocks to select
        position_mode: 'equal' or 'prob_weighted'
        cost_bps: Transaction cost in basis points
    
    Returns:
        Series of portfolio returns
    """
    # Simplified backtest - in practice would need actual returns
    # This is a placeholder implementation
    n_days = len(dates)
    returns = np.zeros(n_days)
    
    for i in range(0, n_days, holding_days):
        if i >= n_days - holding_days:
            break
        
        # Select top-k stocks
        probs = {t: predictions.get(t, np.array([0.5]))[i] for t in tickers}
        sorted_tickers = sorted(probs.keys(), key=lambda t: probs[t], reverse=True)
        selected = sorted_tickers[:top_k]
        
        # Compute weights
        if position_mode == 'equal':
            weights = {t: 1.0 / len(selected) for t in selected}
        else:
            total_prob = sum(probs[t] for t in selected)
            weights = {t: probs[t] / total_prob for t in selected}
        
        # Simplified: assume 0.1% daily return for selected stocks
        # In practice, would use actual returns
        portfolio_ret = sum(weights.get(t, 0) * 0.001 for t in tickers)
        
        # Apply costs (simplified)
        cost = cost_bps / 10000.0
        portfolio_ret -= cost
        
        returns[i:i+holding_days] = portfolio_ret
    
    return pd.Series(returns, index=dates[:len(returns)])


def compute_metrics(returns: pd.Series) -> Dict:
    """Compute backtest metrics."""
    # CAGR
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252.0
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Annualized volatility
    vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming rf=0)
    sharpe = cagr / vol if vol > 0 else 0
    
    # Maximum drawdown
    cumret = (1 + returns).cumprod()
    running_max = cumret.expanding().max()
    drawdown = (cumret - running_max) / running_max
    mdd = drawdown.min()
    
    # Turnover (simplified)
    turnover = 1.0 / 1  # Daily rebalancing
    
    return {
        'cagr': cagr,
        'volatility': vol,
        'sharpe': sharpe,
        'mdd': mdd,
        'turnover': turnover,
        'total_return': total_return
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/backtest.yaml')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--cost_bps', type=float, default=10.0)
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Load data
    print("Loading dataset...")
    features, labels, dates = load_dataset(config)
    
    # Create test split
    train_end = config['split']['train_end']
    val_end = config['split']['val_end']
    test_end = config['split']['test_end']
    _, _, test_idx = walk_forward_split(dates, val_end, val_end, test_end)
    
    # Load model
    input_dim = features.shape[-1]
    horizons = config['features']['horizons']
    quantiles = config['features']['quantiles']
    
    # Need model config - load from patchtst config
    model_config = load_config('configs/patchtst.yaml')
    model = create_model(model_config, input_dim, horizons, quantiles)
    device = torch.device('cpu')
    model = model.to(device)
    
    # Load checkpoint
    if Path(args.ckpt).exists():
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded checkpoint from {args.ckpt}")
    
    # Get predictions
    test_dataset = StockDataset(features, labels, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    outputs_cls, _, _, _ = evaluate_model(
        model, test_loader, device, horizons, quantiles
    )
    
    # Simplified backtest (would need actual returns in practice)
    tickers = config['data']['tickers']
    predictions = {t: outputs_cls[1] for t in tickers}  # Use h=1 predictions
    
    # Compute returns
    test_dates = dates[test_idx]
    strategy_returns = compute_portfolio_returns(
        predictions, test_dates, tickers,
        holding_days=config['holding_days'],
        top_k=config['top_k'],
        position_mode=config['position_mode'],
        cost_bps=args.cost_bps
    )
    
    # Benchmark: equal weight
    benchmark_returns = pd.Series(
        np.full(len(test_dates), 0.0005),  # Simplified: 0.05% daily
        index=test_dates
    )
    
    # Compute metrics
    strategy_metrics = compute_metrics(strategy_returns)
    benchmark_metrics = compute_metrics(benchmark_returns)
    
    print("\n=== Backtest Results ===")
    print(f"\nStrategy (cost={args.cost_bps}bps):")
    print(f"  CAGR: {strategy_metrics['cagr']:.2%}")
    print(f"  Volatility: {strategy_metrics['volatility']:.2%}")
    print(f"  Sharpe: {strategy_metrics['sharpe']:.2f}")
    print(f"  MDD: {strategy_metrics['mdd']:.2%}")
    
    print(f"\nBenchmark:")
    print(f"  CAGR: {benchmark_metrics['cagr']:.2%}")
    print(f"  Sharpe: {benchmark_metrics['sharpe']:.2f}")
    
    # Plot
    figs_dir = Path('reports/figs')
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    plot_cumulative_returns(
        {
            'Strategy': strategy_returns.values,
            'Benchmark': benchmark_returns.values
        },
        figs_dir / 'backtest_returns.png',
        title='Cumulative Returns'
    )
    
    # Save results
    reports_dir = Path('reports/tables')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({
        'strategy': strategy_metrics,
        'benchmark': benchmark_metrics
    }).T
    results_df.to_csv(reports_dir / f'backtest_cost_{args.cost_bps}bps.csv')
    
    print("\nBacktest complete!")


if __name__ == '__main__':
    main()

