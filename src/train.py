"""Training script for stock prediction models."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json

from src.utils.seed import set_seed
from src.utils.io import load_config, ensure_dir, save_json
from src.data.dataset import StockDataset
from src.models.mlp import MLP
from src.models.lstm import LSTM
from src.models.gru import GRU
from src.models.cnn1d import CNN1D
from src.models.patchtst import PatchTST
from src.models.simple_transformer import SimpleTransformer
from src.models.transformer import Transformer
from src.models.common.losses import MultiTaskLoss


def load_dataset(config: dict):
    """Load preprocessed dataset."""
    from src.data.dataset import load_dataset as _load_dataset
    return _load_dataset(config)


def create_data_splits(features, labels, dates, config):
    """Create train/val/test splits."""
    from src.utils.timecv import walk_forward_split
    
    train_end = config['split']['train_end']
    val_end = config['split']['val_end']
    test_end = config['split']['test_end']
    
    # Train/val split
    train_idx, val_idx = walk_forward_split(dates, train_end, train_end, val_end)
    
    # Val/test split (not used in training, but computed for reference)
    _, test_idx = walk_forward_split(dates, val_end, val_end, test_end)
    
    return train_idx, val_idx, test_idx


def create_model(config: dict, input_dim: int, horizons: list, quantiles: list):
    """Create model based on config."""
    model_name = config['model']
    
    if model_name == 'mlp':
        model = MLP(
            input_dim=input_dim,
            horizons=horizons,
            quantiles=quantiles,
            hidden_dims=config['mlp']['hidden_dims'],
            dropout=config['mlp']['dropout'],
            activation=config['mlp']['activation']
        )
    elif model_name == 'lstm':
        model = LSTM(
            input_dim=input_dim,
            horizons=horizons,
            quantiles=quantiles,
            hidden_size=config['lstm']['hidden_size'],
            num_layers=config['lstm']['num_layers'],
            dropout=config['lstm']['dropout'],
            bidirectional=config['lstm']['bidirectional']
        )
    elif model_name == 'gru':
        model = GRU(
            input_dim=input_dim,
            horizons=horizons,
            quantiles=quantiles,
            hidden_size=config['gru']['hidden_size'],
            num_layers=config['gru']['num_layers'],
            dropout=config['gru']['dropout'],
            bidirectional=config['gru']['bidirectional']
        )
    elif model_name == 'cnn1d':
        model = CNN1D(
            input_dim=input_dim,
            horizons=horizons,
            quantiles=quantiles,
            num_filters=config['cnn1d']['num_filters'],
            kernel_sizes=config['cnn1d']['kernel_sizes'],
            dropout=config['cnn1d']['dropout'],
            pool_size=config['cnn1d'].get('pool_size', 2)
        )
    elif model_name == 'patchtst':
        model = PatchTST(
            input_dim=input_dim,
            horizons=horizons,
            quantiles=quantiles,
            d_model=config['patchtst']['d_model'],
            n_heads=config['patchtst']['n_heads'],
            depth=config['patchtst']['depth'],
            patch_len=config['patchtst']['patch_len'],
            stride=config['patchtst']['stride'],
            dropout=config['patchtst']['dropout'],
            channel_independent=config['patchtst']['channel_independent']
        )
    elif model_name == 'simple_transformer':
        model = SimpleTransformer(
            input_dim=input_dim,
            horizons=horizons,
            quantiles=quantiles,
            d_model=config['simple_transformer']['d_model'],
            n_heads=config['simple_transformer']['n_heads'],
            depth=config['simple_transformer']['depth'],
            dropout=config['simple_transformer']['dropout']
        )
    elif model_name == 'transformer':
        model = Transformer(
            input_dim=input_dim,
            horizons=horizons,
            quantiles=quantiles,
            d_model=config['transformer']['d_model'],
            n_heads=config['transformer']['n_heads'],
            depth=config['transformer']['depth'],
            d_ff=config['transformer'].get('d_ff', None),
            dropout=config['transformer']['dropout']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_dict = {}
    
    for batch in tqdm(dataloader, desc='Training'):
        x = batch['features'].to(device)
        targets = {
            'classification': {h: batch['classification'][h].to(device) for h in batch['classification']},
            'regression': {h: batch['regression'][h].to(device) for h in batch['regression']}
        }
        
        optimizer.zero_grad()
        outputs = model(x)
        
        losses = criterion(outputs, targets)
        loss = losses['total']
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip
            )
        
        optimizer.step()
        
        total_loss += loss.item() if hasattr(loss, 'item') else loss
        for k, v in losses.items():
            if k not in loss_dict:
                loss_dict[k] = 0.0
            loss_dict[k] += v.item() if hasattr(v, 'item') else v
    
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_loss_dict = {k: v / n_batches for k, v in loss_dict.items()}
    
    return avg_loss, avg_loss_dict


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    loss_dict = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            x = batch['features'].to(device)
            targets = {
                'classification': {h: batch['classification'][h].to(device) for h in batch['classification']},
                'regression': {h: batch['regression'][h].to(device) for h in batch['regression']}
            }
            
            outputs = model(x)
            losses = criterion(outputs, targets)
            loss = losses['total']
            
            total_loss += loss.item() if hasattr(loss, 'item') else loss
            for k, v in losses.items():
                if k not in loss_dict:
                    loss_dict[k] = 0.0
                loss_dict[k] += v.item() if hasattr(v, 'item') else v
    
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_loss_dict = {k: v / n_batches for k, v in loss_dict.items()}
    
    return avg_loss, avg_loss_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Load data
    print("Loading dataset...")
    import pandas as pd
    features, labels, dates = load_dataset(config)
    
    # Create splits
    print("Creating data splits...")
    train_idx, val_idx, test_idx = create_data_splits(features, labels, dates, config)
    
    # Create datasets
    train_dataset = StockDataset(features, labels, train_idx)
    val_dataset = StockDataset(features, labels, val_idx)
    
    # Create dataloaders
    batch_size = config['trainer']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_dim = features.shape[-1]
    horizons = config['features']['horizons']
    quantiles = config['features']['quantiles']
    
    print(f"Creating {config['model']} model...")
    model = create_model(config, input_dim, horizons, quantiles)
    
    device = torch.device('cpu')
    model = model.to(device)
    
    # Loss and optimizer
    loss_config = config.get('loss', {})
    criterion = MultiTaskLoss(
        horizons=horizons,
        quantiles=quantiles,
        bce_weight=loss_config.get('bce_weight', 1.0),
        pinball_weight=loss_config.get('pinball_weight', 1.0),
        use_focal=loss_config.get('use_focal', True),
        focal_alpha=loss_config.get('focal_alpha', 1.0),
        focal_gamma=loss_config.get('focal_gamma', 2.0)
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['trainer']['lr'],
        weight_decay=config['trainer']['weight_decay']
    )
    
    # Learning rate scheduler
    if config['trainer']['lr_scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
    else:
        scheduler = None
    
    # Training loop
    output_dir = Path(config.get('output_dir', 'runs')) / config['model']
    ensure_dir(str(output_dir))
    checkpoint_dir = output_dir / 'checkpoints'
    ensure_dir(str(checkpoint_dir))
    
    best_val_loss = float('inf')
    patience = config['trainer']['early_stop_patience']
    patience_counter = 0
    epochs = config['trainer']['epochs']
    
    logs = []
    
    print("Starting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        grad_clip = config['trainer'].get('grad_clip', 0)
        train_loss, train_loss_dict = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        
        # Validate
        val_loss, val_loss_dict = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
        
        # Logging
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{f'train_{k}': v for k, v in train_loss_dict.items()},
            **{f'val_{k}': v for k, v in val_loss_dict.items()}
        }
        logs.append(log_entry)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / 'best.pt')
            print(f"Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save logs
    save_json(logs, output_dir / 'logs.json')
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

