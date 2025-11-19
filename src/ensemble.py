"""Script to train and evaluate ensemble models."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
import json

from src.utils.seed import set_seed
from src.utils.io import load_config, ensure_dir, save_json
from src.data.dataset import StockDataset, load_dataset
from src.utils.timecv import walk_forward_split
from src.train import create_model
from src.models.ensemble import EnsembleModel
from src.evaluate import evaluate_model
from src.utils.metrics import compute_all_metrics


def load_trained_models(config: dict, model_names: list, input_dim: int,
                        horizons: list, quantiles: list, device):
    """Load trained models from checkpoints."""
    models = []
    for model_name in model_names:
        # Load model-specific config
        model_config_path = f'configs/{model_name}.yaml'
        if Path(model_config_path).exists():
            model_config = load_config(model_config_path)
        else:
            # Fallback: use base config and set model name
            model_config = config.copy()
            model_config['model'] = model_name
        
        # Load model
        model = create_model(model_config, input_dim, horizons, quantiles)
        model = model.to(device)
        
        # Load checkpoint
        output_dir = Path(model_config.get('output_dir', f'runs/{model_name}'))
        ckpt_path = output_dir / model_name / 'checkpoints' / 'best.pt'
        
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded {model_name} from {ckpt_path}")
        else:
            print(f"Warning: Checkpoint not found for {model_name} at {ckpt_path}")
        
        models.append(model)
    
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mlp.yaml')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['lstm', 'gru', 'patchtst'],
                       help='List of model names to ensemble')
    parser.add_argument('--method', type=str, choices=['weighted', 'average', 'stacking'],
                       default='weighted', help='Ensemble method')
    parser.add_argument('--weights', type=str, default=None,
                       help='Comma-separated weights for models (e.g., "0.4,0.3,0.3")')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])
    
    # Load data
    print("Loading dataset...")
    features, labels, dates = load_dataset(config)
    
    # Create splits
    train_end = config['split']['train_end']
    val_end = config['split']['val_end']
    test_end = config['split']['test_end']
    train_idx, val_idx = walk_forward_split(dates, train_end, train_end, val_end)
    _, test_idx = walk_forward_split(dates, val_end, val_end, test_end)
    
    test_dataset = StockDataset(features, labels, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Model parameters
    input_dim = features.shape[-1]
    horizons = config['features']['horizons']
    quantiles = config['features']['quantiles']
    device = torch.device('cpu')
    
    # Load base models
    print(f"\nLoading base models: {args.models}")
    base_models = load_trained_models(
        config, args.models, input_dim, horizons, quantiles, device
    )
    
    # Optimize weights on validation set
    print("\nOptimizing ensemble weights on validation set...")
    val_dataset = StockDataset(features, labels, val_idx)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Collect base model predictions on validation set
    val_base_outputs = []
    for model in base_models:
        model.eval()
        val_outputs_cls, val_outputs_reg, val_targets_cls, val_targets_reg = evaluate_model(
            model, val_loader, device, horizons, quantiles
        )
        val_base_outputs.append({
            'cls': val_outputs_cls,
            'reg': val_outputs_reg,
            'targets_cls': val_targets_cls,
            'targets_reg': val_targets_reg
        })
    
    # Optimize weights using scipy
    from scipy.optimize import minimize
    from src.utils.metrics import compute_all_metrics, find_optimal_threshold
    
    def objective(weights_array):
        """Objective function: negative F1 score (we want to maximize F1)"""
        weights_dict = {f'model_{i}': w for i, w in enumerate(weights_array)}
        
        # Combine predictions with these weights
        combined_cls = {h: np.zeros_like(val_base_outputs[0]['cls'][h]) for h in horizons}
        combined_reg = {h: {tau: np.zeros_like(val_base_outputs[0]['reg'][h][tau]) 
                            for tau in quantiles} for h in horizons}
        
        for i, outputs in enumerate(val_base_outputs):
            w = weights_array[i]
            for h in horizons:
                combined_cls[h] += w * outputs['cls'][h]
                for tau in quantiles:
                    combined_reg[h][tau] += w * outputs['reg'][h][tau]
        
        # Compute metrics
        metrics = compute_all_metrics(
            val_base_outputs[0]['targets_cls'], combined_cls,
            val_base_outputs[0]['targets_reg'], combined_reg,
            horizons, quantiles
        )
        
        # Return negative average F1 (we want to maximize F1)
        avg_f1 = np.mean([metrics[f'h{h}']['f1'] for h in horizons])
        return -avg_f1
    
    # Initial weights (equal)
    initial_weights = np.ones(len(base_models)) / len(base_models)
    
    # Constraints: weights sum to 1, all >= 0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0, 1) for _ in range(len(base_models))]
    
    # Optimize
    result = minimize(objective, initial_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints, options={'maxiter': 100})
    
    if result.success:
        optimal_weights = result.x
        weights = {f'model_{i}': w for i, w in enumerate(optimal_weights)}
        print("  Optimal weights:")
        for i, model_name in enumerate(args.models):
            print(f"    {model_name}: {optimal_weights[i]:.4f}")
    else:
        print("  Warning: Weight optimization failed, using equal weights")
        weights = {f'model_{i}': 1.0 / len(base_models) for i in range(len(base_models))}
    
    # Parse manual weights if provided (override optimization)
    if args.weights:
        weight_list = [float(w) for w in args.weights.split(',')]
        if len(weight_list) == len(args.models):
            weights = {f'model_{i}': w for i, w in enumerate(weight_list)}
            print("  Using manually specified weights")
    
    # Create ensemble
    use_stacking = (args.method == 'stacking')
    ensemble = EnsembleModel(
        base_models, horizons, quantiles,
        method=args.method if not use_stacking else 'average',
        weights=weights,
        use_stacking=use_stacking
    )
    ensemble = ensemble.to(device)
    ensemble.eval()
    
    # If stacking, train stacking networks on validation set
    if use_stacking:
        print("\nTraining stacking networks on validation set...")
        val_dataset = StockDataset(features, labels, val_idx)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        # Collect base model predictions
        all_base_cls = {h: [] for h in horizons}
        all_base_quantiles = {h: {tau: [] for tau in quantiles} for h in horizons}
        all_targets_cls = {h: [] for h in horizons}
        all_targets_reg = {h: [] for h in horizons}
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['features'].to(device)
                for model in base_models:
                    model.eval()
                    outputs = model(x)
                    for h in horizons:
                        all_base_cls[h].append(outputs['classification'][h].cpu())
                        for tau in quantiles:
                            if tau in outputs['quantiles'][h]:
                                all_base_quantiles[h][tau].append(outputs['quantiles'][h][tau].cpu())
                
                for h in horizons:
                    all_targets_cls[h].extend(batch['classification'][h].numpy())
                    all_targets_reg[h].extend(batch['regression'][h].numpy())
        
        # Stack predictions
        stacked_cls = {h: torch.cat(all_base_cls[h], dim=0) for h in horizons}
        stacked_quantiles = {
            h: {tau: torch.cat(all_base_quantiles[h][tau], dim=0) for tau in quantiles}
            for h in horizons
        }
        
        # Train stacking networks
        optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.001)
        criterion_cls = nn.BCEWithLogitsLoss()
        criterion_reg = nn.MSELoss()
        
        for epoch in range(20):
            ensemble.train()
            total_loss = 0.0
            
            for h in horizons:
                # Classification
                logits = ensemble.stacking_networks[str(h)](stacked_cls[h])
                targets = torch.FloatTensor(all_targets_cls[h]).unsqueeze(1)
                loss_cls = criterion_cls(logits, targets)
                
                # Quantiles
                loss_quant = 0.0
                for tau in quantiles:
                    key = f"h{h}_tau{tau}".replace('.', '_')
                    pred = ensemble.stacking_quantile_networks[key](stacked_quantiles[h][tau])
                    targets_reg = torch.FloatTensor(all_targets_reg[h]).unsqueeze(1)
                    loss_quant += criterion_reg(pred, targets_reg)
                
                total_loss += loss_cls + loss_quant / len(quantiles)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"Stacking epoch {epoch+1}/20, loss: {total_loss.item():.4f}")
    
    # Find optimal thresholds on validation set
    print("\nFinding optimal thresholds on validation set...")
    val_outputs_cls, _, val_targets_cls, _ = evaluate_model(
        ensemble, val_loader, device, horizons, quantiles
    )
    
    from src.utils.metrics import find_optimal_threshold
    optimal_thresholds = {}
    for h in horizons:
        threshold = find_optimal_threshold(val_targets_cls[h], val_outputs_cls[h], metric='f1')
        optimal_thresholds[h] = threshold
        print(f"  Horizon {h}: optimal threshold = {threshold:.4f}")
    
    # Evaluate ensemble on test set
    print("\nEvaluating ensemble model on test set...")
    outputs_cls, outputs_reg, targets_cls, targets_reg = evaluate_model(
        ensemble, test_loader, device, horizons, quantiles
    )
    
    # Compute metrics with optimal thresholds
    metrics = compute_all_metrics(
        targets_cls, outputs_cls, targets_reg, outputs_reg, horizons, quantiles,
        optimal_thresholds=optimal_thresholds
    )
    
    # Print results
    print("\n=== Ensemble Test Set Metrics (with optimal thresholds) ===")
    for h in horizons:
        print(f"\nHorizon {h}:")
        print(f"  Optimal Threshold: {metrics[f'h{h}']['threshold']:.4f}")
        print(f"  Accuracy: {metrics[f'h{h}']['accuracy']:.4f}")
        print(f"  F1: {metrics[f'h{h}']['f1']:.4f}")
        print(f"  AUC: {metrics[f'h{h}']['auc']:.4f}")
        print(f"  Brier: {metrics[f'h{h}']['brier']:.4f}")
        if 'crps' in metrics[f'h{h}'] and not np.isnan(metrics[f'h{h}']['crps']):
            print(f"  CRPS: {metrics[f'h{h}']['crps']:.4f}")
        for tau in quantiles:
            key = f'pinball_{tau}'
            if key in metrics[f'h{h}'] and not np.isnan(metrics[f'h{h}'][key]):
                print(f"  Pinball {tau}: {metrics[f'h{h}'][key]:.4f}")
    
    # Save results
    output_dir = Path('runs/ensemble')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    import pandas as pd
    metrics_df = pd.DataFrame({
        h: metrics[f'h{h}'] for h in horizons
    }).T
    metrics_df.to_csv(output_dir / 'ensemble_test_metrics.csv')
    print(f"\nResults saved to {output_dir / 'ensemble_test_metrics.csv'}")


if __name__ == '__main__':
    main()

