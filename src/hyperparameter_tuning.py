"""Hyperparameter tuning using grid search."""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from itertools import product
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.io import load_config, save_json
from src.data.dataset import StockDataset, load_dataset
from src.utils.timecv import walk_forward_split
from src.train import create_model, train_model
from src.evaluate import evaluate_model
from src.utils.metrics import compute_all_metrics, find_optimal_threshold


def grid_search(config, param_grid, max_combinations=20):
    """Perform grid search over hyperparameters.
    
    Args:
        config: Base config dict
        param_grid: Dict of parameter names -> list of values
        max_combinations: Maximum number of combinations to try
    
    Returns:
        Best config and results
    """
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    
    # Limit combinations
    if len(all_combinations) > max_combinations:
        print(f"Limiting to {max_combinations} random combinations from {len(all_combinations)} total")
        np.random.seed(42)
        all_combinations = np.random.choice(len(all_combinations), max_combinations, replace=False)
        all_combinations = [all_combinations[i] if isinstance(all_combinations[i], tuple) 
                           else all_combinations[i] for i in range(max_combinations)]
        # Reconstruct from indices
        indices = np.random.choice(len(list(product(*param_values))), max_combinations, replace=False)
        all_combinations = [list(product(*param_values))[i] for i in indices]
    
    print(f"Testing {len(all_combinations)} hyperparameter combinations...")
    
    # Load data once
    print("Loading dataset...")
    features, labels, dates = load_dataset(config)
    
    train_end = config['split']['train_end']
    val_end = config['split']['val_end']
    test_end = config['split']['test_end']
    train_idx, val_idx = walk_forward_split(dates, train_end, train_end, val_end)
    _, test_idx = walk_forward_split(dates, val_end, val_end, test_end)
    
    train_dataset = StockDataset(features, labels, train_idx)
    val_dataset = StockDataset(features, labels, val_idx)
    test_dataset = StockDataset(features, labels, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=config['trainer']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['trainer']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['trainer']['batch_size'], shuffle=False)
    
    results = []
    best_score = -np.inf
    best_config = None
    best_metrics = None
    
    for i, combination in enumerate(tqdm(all_combinations, desc="Grid search")):
        # Create config with this combination
        test_config = config.copy()
        for param_name, param_value in zip(param_names, combination):
            # Update nested config
            if '.' in param_name:
                parts = param_name.split('.')
                current = test_config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = param_value
            else:
                test_config[param_name] = param_value
        
        try:
            # Train model
            input_dim = features.shape[-1]
            horizons = config['features']['horizons']
            quantiles = config['features']['quantiles']
            
            model = create_model(test_config, input_dim, horizons, quantiles)
            device = torch.device('cpu')
            model = model.to(device)
            
            # Train with early stopping (simplified version)
            from src.models.common.losses import MultiTaskLoss
            criterion = MultiTaskLoss(
                horizons=horizons,
                quantiles=quantiles,
                bce_weight=test_config.get('loss', {}).get('bce_weight', 1.0),
                pinball_weight=test_config.get('loss', {}).get('pinball_weight', 1.0),
                use_focal=test_config.get('loss', {}).get('use_focal', True),
                focal_alpha=test_config.get('loss', {}).get('focal_alpha', 1.0),
                focal_gamma=test_config.get('loss', {}).get('focal_gamma', 2.0)
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=test_config['trainer']['lr'],
                weight_decay=test_config['trainer']['weight_decay']
            )
            
            # Simple training loop
            best_val_loss = float('inf')
            max_epochs = 20
            for epoch in range(max_epochs):
                model.train()
                train_loss = 0.0
                for batch in train_loader:
                    x = batch['features'].to(device)
                    targets = {
                        'classification': {h: batch['classification'][h].to(device) for h in horizons},
                        'regression': {h: batch['regression'][h].to(device) for h in horizons}
                    }
                    outputs = model(x)
                    losses = criterion(outputs, targets)
                    loss = losses['total']
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), test_config['trainer'].get('grad_clip', 1.0))
                    optimizer.step()
                    train_loss += loss.item() if hasattr(loss, 'item') else loss
                
                # Validate
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch['features'].to(device)
                        targets = {
                            'classification': {h: batch['classification'][h].to(device) for h in horizons},
                            'regression': {h: batch['regression'][h].to(device) for h in horizons}
                        }
                        outputs = model(x)
                        losses = criterion(outputs, targets)
                        loss = losses['total']
                        val_loss += loss.item() if hasattr(loss, 'item') else loss
                
                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if epoch >= 5:  # Early stopping after at least 5 epochs
                        break
            
            # Evaluate on validation set
            val_outputs_cls, val_outputs_reg, val_targets_cls, val_targets_reg = evaluate_model(
                model, val_loader, device, horizons, quantiles
            )
            
            # Find optimal thresholds
            optimal_thresholds = {}
            for h in horizons:
                threshold = find_optimal_threshold(val_targets_cls[h], val_outputs_cls[h], metric='f1')
                optimal_thresholds[h] = threshold
            
            val_metrics = compute_all_metrics(
                val_targets_cls, val_outputs_cls, val_targets_reg, val_outputs_reg,
                horizons, quantiles, optimal_thresholds=optimal_thresholds
            )
            
            # Score: average F1 across horizons
            score = np.mean([val_metrics[f'h{h}']['f1'] for h in horizons])
            
            # Store results
            result = {
                'combination': i,
                'score': score,
                'params': {name: val for name, val in zip(param_names, combination)},
                'metrics': {h: val_metrics[f'h{h}'] for h in horizons}
            }
            results.append(result)
            
            if score > best_score:
                best_score = score
                best_config = test_config.copy()
                best_metrics = val_metrics.copy()
                
                # Also evaluate on test set
                test_outputs_cls, test_outputs_reg, test_targets_cls, test_targets_reg = evaluate_model(
                    model, test_loader, device, horizons, quantiles
                )
                test_metrics = compute_all_metrics(
                    test_targets_cls, test_outputs_cls, test_targets_reg, test_outputs_reg,
                    horizons, quantiles, optimal_thresholds=optimal_thresholds
                )
                result['test_metrics'] = {h: test_metrics[f'h{h}'] for h in horizons}
                best_metrics = test_metrics
            
            print(f"  Combination {i+1}/{len(all_combinations)}: Score={score:.4f}")
            
        except Exception as e:
            print(f"  Combination {i+1} failed: {e}")
            continue
    
    return best_config, best_metrics, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Base config file')
    parser.add_argument('--param_grid', type=str, required=True,
                       help='JSON file with parameter grid')
    parser.add_argument('--max_combinations', type=int, default=20,
                       help='Maximum combinations to try')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Load parameter grid
    with open(args.param_grid, 'r') as f:
        param_grid = json.load(f)
    
    print(f"Parameter grid: {param_grid}")
    
    # Perform grid search
    best_config, best_metrics, results = grid_search(
        config, param_grid, max_combinations=args.max_combinations
    )
    
    # Save results
    output_dir = Path('runs/hyperparameter_tuning')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best config
    save_json(best_config, output_dir / 'best_config.json')
    
    # Save all results
    results_df = pd.DataFrame([
        {
            'combination': r['combination'],
            'score': r['score'],
            **r['params'],
            **{f'h{h}_{metric}': r['metrics'][h].get(metric, np.nan) 
               for h in config['features']['horizons'] 
               for metric in ['f1', 'auc', 'crps']}
        }
        for r in results
    ])
    results_df = results_df.sort_values('score', ascending=False)
    results_df.to_csv(output_dir / 'grid_search_results.csv', index=False)
    
    # Save best metrics
    save_json(best_metrics, output_dir / 'best_metrics.json')
    
    print(f"\n=== Best Configuration ===")
    print(f"Score: {best_score:.4f}")
    print(f"Parameters:")
    for name, values in param_grid.items():
        print(f"  {name}: {best_config.get(name, 'N/A')}")
    print(f"\nBest metrics:")
    for h in config['features']['horizons']:
        print(f"  Horizon {h}: F1={best_metrics[f'h{h}']['f1']:.4f}, "
              f"AUC={best_metrics[f'h{h}']['auc']:.4f}")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()

