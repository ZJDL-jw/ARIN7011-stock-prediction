"""Walk-forward evaluation script."""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict

from src.utils.seed import set_seed
from src.utils.io import load_config
from src.utils.metrics import compute_all_metrics
from src.utils.plots import plot_walkforward_boxplot
from src.utils.timecv import annual_walk_forward
from src.data.dataset import StockDataset, load_dataset
from src.train import create_model
from src.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Load data
    print("Loading dataset...")
    features, labels, dates = load_dataset(config)
    
    # Annual walk-forward splits
    start_year = 2020
    end_year = 2024
    splits = annual_walk_forward(dates, start_year, end_year)
    
    # Load model
    input_dim = features.shape[-1]
    horizons = config['features']['horizons']
    quantiles = config['features']['quantiles']
    
    model = create_model(config, input_dim, horizons, quantiles)
    device = torch.device('cpu')
    model = model.to(device)
    
    # Load checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        output_dir = Path(config.get('output_dir', 'runs')) / config['model']
        ckpt_path = output_dir / 'checkpoints' / 'best.pt'
    
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint from {ckpt_path}")
    
    # Evaluate on each year
    results_by_year = defaultdict(lambda: defaultdict(list))
    
    print("Running walk-forward evaluation...")
    for train_idx, test_idx in splits:
        if len(test_idx) == 0:
            continue
        
        year = dates[test_idx[0]].year
        print(f"\nEvaluating year {year}...")
        
        # Create test dataset
        test_dataset = StockDataset(features, labels, test_idx)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Evaluate
        outputs_cls, outputs_reg, targets_cls, targets_reg = evaluate_model(
            model, test_loader, device, horizons, quantiles
        )
        
        # Compute metrics
        metrics = compute_all_metrics(
            targets_cls, outputs_cls, targets_reg, outputs_reg, horizons, quantiles
        )
        
        # Store results
        for h in horizons:
            results_by_year[year][f'auc_h{h}'].append(metrics[f'h{h}']['auc'])
            results_by_year[year][f'f1_h{h}'].append(metrics[f'h{h}']['f1'])
            results_by_year[year][f'crps_h{h}'].append(metrics[f'h{h}']['crps'])
    
    # Convert to format for plotting
    auc_results = {year: results_by_year[year]['auc_h1'] for year in sorted(results_by_year.keys())}
    
    # Plot
    figs_dir = Path('reports/figs')
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    plot_walkforward_boxplot(
        auc_results,
        figs_dir / 'walkforward_auc.png',
        metric='AUC',
        title='Walk-forward AUC by Year'
    )
    
    # Save results
    reports_dir = Path('reports/tables')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame({
        year: {
            'auc_mean': np.mean(results_by_year[year]['auc_h1']),
            'auc_std': np.std(results_by_year[year]['auc_h1']),
            'f1_mean': np.mean(results_by_year[year]['f1_h1']),
            'crps_mean': np.mean(results_by_year[year]['crps_h1'])
        }
        for year in sorted(results_by_year.keys())
    }).T
    
    results_df.to_csv(reports_dir / 'walkforward_results.csv')
    print("\nWalk-forward evaluation complete!")


if __name__ == '__main__':
    main()

