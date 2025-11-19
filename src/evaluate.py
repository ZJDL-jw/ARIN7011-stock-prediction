"""Evaluation script with calibration and visualization."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from src.utils.seed import set_seed
from src.utils.io import load_config, load_json
from src.utils.metrics import compute_all_metrics
from src.utils.plots import plot_roc_curve, plot_pr_curve, plot_reliability
from src.utils.calibration import Calibrator
from src.data.dataset import StockDataset, load_dataset
from src.utils.timecv import walk_forward_split
from src.train import create_model


def evaluate_model(model, dataloader, device, horizons, quantiles):
    """Evaluate model and return predictions."""
    model.eval()
    
    all_outputs_cls = {h: [] for h in horizons}
    all_outputs_reg = {h: {tau: [] for tau in quantiles} for h in horizons}
    all_targets_cls = {h: [] for h in horizons}
    all_targets_reg = {h: [] for h in horizons}
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['features'].to(device)
            outputs = model(x)
            
            # Classification outputs
            for h in horizons:
                logits = outputs['classification'][h].cpu().numpy()
                probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
                all_outputs_cls[h].extend(probs)
                all_targets_cls[h].extend(batch['classification'][h].numpy())
            
            # Quantile outputs - ensure proper key matching
            for h in horizons:
                quantile_dict = outputs['quantiles'][h]
                for tau in quantiles:
                    # Try multiple key formats
                    quant_pred = None
                    if tau in quantile_dict:
                        quant_pred = quantile_dict[tau]
                    elif str(tau) in quantile_dict:
                        quant_pred = quantile_dict[str(tau)]
                    else:
                        # Try to find by converting keys
                        for key in quantile_dict.keys():
                            try:
                                if abs(float(key) - tau) < 1e-6:
                                    quant_pred = quantile_dict[key]
                                    break
                            except:
                                continue
                    
                    if quant_pred is not None:
                        quant_pred = quant_pred.cpu().numpy().squeeze()
                        # Ensure 1D array
                        if quant_pred.ndim == 0:
                            quant_pred = quant_pred.reshape(1)
                        elif quant_pred.ndim > 1:
                            quant_pred = quant_pred.flatten()
                        all_outputs_reg[h][tau].extend(quant_pred)
                    else:
                        # If no match found, use zeros
                        all_outputs_reg[h][tau].extend(np.zeros(x.shape[0]))
                
                # Targets (add once per batch, not per quantile)
                if len(all_targets_reg[h]) < len(all_outputs_reg[h][quantiles[0]]):
                    reg_targets = batch['regression'][h].numpy()
                    if reg_targets.ndim > 1:
                        reg_targets = reg_targets.squeeze()
                    all_targets_reg[h].extend(reg_targets)
    
    # Convert to arrays
    outputs_cls = {h: np.array(all_outputs_cls[h]).squeeze() for h in horizons}
    outputs_reg = {
        h: {tau: np.array(all_outputs_reg[h][tau]).squeeze() for tau in quantiles}
        for h in horizons
    }
    targets_cls = {h: np.array(all_targets_cls[h]).squeeze() for h in horizons}
    targets_reg = {h: np.array(all_targets_reg[h]).squeeze() for h in horizons}
    
    return outputs_cls, outputs_reg, targets_cls, targets_reg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--calibrate', type=str, choices=['isotonic', 'platt', 'none'], default='none')
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
        output_dir = Path(config.get('output_dir', f"runs/{config.get('model', 'mlp')}"))
        ckpt_path = output_dir / 'checkpoints' / 'best.pt'
    
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}")
    
    # Find optimal thresholds on validation set
    print("Finding optimal thresholds on validation set...")
    val_dataset = StockDataset(features, labels, val_idx)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    val_outputs_cls, _, val_targets_cls, _ = evaluate_model(
        model, val_loader, device, horizons, quantiles
    )
    
    # Find optimal thresholds for each horizon
    from src.utils.metrics import find_optimal_threshold
    optimal_thresholds = {}
    for h in horizons:
        threshold = find_optimal_threshold(val_targets_cls[h], val_outputs_cls[h], metric='f1')
        optimal_thresholds[h] = threshold
        print(f"  Horizon {h}: optimal threshold = {threshold:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    outputs_cls, outputs_reg, targets_cls, targets_reg = evaluate_model(
        model, test_loader, device, horizons, quantiles
    )
    
    # Compute metrics with optimal thresholds
    print("Computing metrics...")
    metrics = compute_all_metrics(
        targets_cls, outputs_cls, targets_reg, outputs_reg, horizons, quantiles,
        optimal_thresholds=optimal_thresholds
    )
    
    # Print metrics
    print("\n=== Test Set Metrics (with optimal thresholds) ===")
    for h in horizons:
        print(f"\nHorizon {h}:")
        print(f"  Optimal Threshold: {metrics[f'h{h}']['threshold']:.4f}")
        print(f"  Accuracy: {metrics[f'h{h}']['accuracy']:.4f}")
        print(f"  F1: {metrics[f'h{h}']['f1']:.4f}")
        print(f"  AUC: {metrics[f'h{h}']['auc']:.4f}")
        print(f"  Brier: {metrics[f'h{h}']['brier']:.4f}")
        print(f"  CRPS: {metrics[f'h{h}']['crps']:.4f}")
        for tau in quantiles:
            print(f"  Pinball {tau}: {metrics[f'h{h}'][f'pinball_{tau}']:.4f}")
    
    # Calibration
    calibrators = {}
    if args.calibrate != 'none':
        print(f"\nCalibrating probabilities using {args.calibrate}...")
        
        # Use validation set for calibration
        train_end = config['split']['train_end']
        val_end = config['split']['val_end']
        _, val_idx = walk_forward_split(dates, train_end, train_end, val_end)
        val_dataset = StockDataset(features, labels, val_idx)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        val_outputs_cls, _, val_targets_cls, _ = evaluate_model(
            model, val_loader, device, horizons, quantiles
        )
        
        # Fit calibrators
        for h in horizons:
            calibrator = Calibrator(method=args.calibrate)
            calibrator.fit(val_outputs_cls[h], val_targets_cls[h], horizon=h)
            calibrators[h] = calibrator
        
        # Apply calibration to test set
        outputs_cls_cal = {}
        for h in horizons:
            outputs_cls_cal[h] = calibrators[h].predict(outputs_cls[h], horizon=h)
        
        # Recompute metrics
        metrics_cal = compute_all_metrics(
            targets_cls, outputs_cls_cal, targets_reg, outputs_reg, horizons, quantiles
        )
        
        print("\n=== Calibrated Metrics ===")
        for h in horizons:
            print(f"\nHorizon {h}:")
            print(f"  Brier (before): {metrics[f'h{h}']['brier']:.4f}")
            print(f"  Brier (after): {metrics_cal[f'h{h}']['brier']:.4f}")
    
    # Save metrics
    output_dir = Path(config.get('output_dir', f"runs/{config.get('model', 'mlp')}"))
    reports_dir = Path('reports/tables')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(reports_dir / f"{config['model']}_test_metrics.csv")
    
    # Plotting
    print("\nGenerating plots...")
    model_name = config.get('model', 'mlp')
    figs_dir = Path('reports/figs') / model_name
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    for h in horizons:
        # ROC curve
        plot_roc_curve(
            targets_cls[h], outputs_cls[h],
            figs_dir / f"roc_h{h}.png",
            title=f"ROC Curve (Horizon {h})"
        )
        
        # PR curve
        plot_pr_curve(
            targets_cls[h], outputs_cls[h],
            figs_dir / f"pr_h{h}.png",
            title=f"Precision-Recall Curve (Horizon {h})"
        )
        
        # Reliability diagram
        plot_reliability(
            targets_cls[h], outputs_cls[h],
            figs_dir / f"reliability_h{h}_before.png",
            title=f"Reliability Diagram (Horizon {h}, Before Calibration)"
        )
        
        if args.calibrate != 'none':
            plot_reliability(
                targets_cls[h], outputs_cls_cal[h],
                figs_dir / f"reliability_h{h}_after.png",
                title=f"Reliability Diagram (Horizon {h}, After Calibration)"
            )
    
    # Attention visualization (for Transformer models)
    if config['model'] in ['patchtst', 'simple_transformer', 'transformer']:
        print("Extracting attention weights...")
        model.eval()
        sample_batch = next(iter(test_loader))
        x_sample = sample_batch['features'][:1].to(device)
        
        with torch.no_grad():
            outputs = model(x_sample, return_attention=True)
            if 'attention' in outputs:
                from src.utils.plots import plot_attention_heatmap
                # Average over batches and heads
                attn = outputs['attention'][-1]  # Last layer
                if isinstance(attn, tuple):
                    attn = attn[0]
                attn_np = attn.cpu().numpy()
                plot_attention_heatmap(
                    attn_np[0],  # First sample
                    figs_dir / "attention_heatmap.png",
                    title="PatchTST Attention Heatmap"
                )
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

