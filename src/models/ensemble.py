"""Ensemble model combining multiple base models."""
import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np


class EnsembleModel(nn.Module):
    """Ensemble model that combines predictions from multiple base models.
    
    Supports both weighted averaging and learned stacking.
    """
    
    def __init__(self, models: List[nn.Module], horizons: list, quantiles: list,
                 method: str = 'weighted', weights: Dict[str, float] = None,
                 use_stacking: bool = False, stacking_dim: int = 32):
        """
        Args:
            models: List of trained base models
            horizons: List of prediction horizons
            quantiles: List of quantile levels
            method: 'weighted' or 'average'
            weights: Optional dict mapping model_name -> weight
            use_stacking: Whether to use learned stacking
            stacking_dim: Hidden dimension for stacking network
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.horizons = horizons
        self.quantiles = quantiles
        self.method = method
        self.use_stacking = use_stacking
        
        # Default weights (equal if not specified)
        if weights is None:
            self.weights = {f'model_{i}': 1.0 / len(models) for i in range(len(models))}
        else:
            # Normalize weights
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}
        
        # Stacking network (optional)
        if use_stacking:
            n_models = len(models)
            self.stacking_networks = nn.ModuleDict({
                str(h): nn.Sequential(
                    nn.Linear(n_models, stacking_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(stacking_dim, 1)
                ) for h in horizons
            })
            self.stacking_quantile_networks = nn.ModuleDict({
                f"h{h}_tau{tau}".replace('.', '_'): nn.Sequential(
                    nn.Linear(n_models, stacking_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(stacking_dim, 1)
                ) for h in horizons for tau in quantiles
            })
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tensor [batch_size, T, D]
        
        Returns:
            Dict with ensemble predictions
        """
        # Get predictions from all models
        all_classification = {h: [] for h in self.horizons}
        all_quantiles = {h: {tau: [] for tau in self.quantiles} for h in self.horizons}
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                outputs = model(x)
            
            # Collect classification outputs
            for h in self.horizons:
                all_classification[h].append(outputs['classification'][h])
            
            # Collect quantile outputs
            for h in self.horizons:
                for tau in self.quantiles:
                    if tau in outputs['quantiles'][h]:
                        all_quantiles[h][tau].append(outputs['quantiles'][h][tau])
                    elif str(tau) in outputs['quantiles'][h]:
                        all_quantiles[h][tau].append(outputs['quantiles'][h][str(tau)])
        
        # Combine predictions
        ensemble_outputs = {
            'classification': {},
            'quantiles': {}
        }
        
        if self.use_stacking:
            # Learned stacking
            for h in self.horizons:
                # Classification stacking
                cls_stack = torch.stack(all_classification[h], dim=-1)  # [B, 1, N]
                cls_stack = cls_stack.squeeze(1)  # [B, N]
                ensemble_outputs['classification'][h] = self.stacking_networks[str(h)](cls_stack)
                
                # Quantile stacking
                ensemble_outputs['quantiles'][h] = {}
                for tau in self.quantiles:
                    if len(all_quantiles[h][tau]) > 0:
                        quant_stack = torch.stack(all_quantiles[h][tau], dim=-1)  # [B, 1, N]
                        quant_stack = quant_stack.squeeze(1)  # [B, N]
                        key = f"h{h}_tau{tau}".replace('.', '_')
                        ensemble_outputs['quantiles'][h][tau] = self.stacking_quantile_networks[key](quant_stack)
        else:
            # Weighted or simple averaging
            for h in self.horizons:
                # Classification: weighted average of logits
                cls_list = all_classification[h]
                if self.method == 'weighted':
                    weights_tensor = torch.tensor([
                        self.weights.get(f'model_{i}', 1.0 / len(self.models))
                        for i in range(len(cls_list))
                    ], device=cls_list[0].device, dtype=cls_list[0].dtype)
                    weights_tensor = weights_tensor.view(1, 1, -1)
                    cls_stack = torch.stack(cls_list, dim=-1)  # [B, 1, N]
                    ensemble_outputs['classification'][h] = (cls_stack * weights_tensor).sum(dim=-1)
                else:
                    # Simple average
                    ensemble_outputs['classification'][h] = torch.stack(cls_list, dim=-1).mean(dim=-1)
                
                # Quantiles: weighted average
                ensemble_outputs['quantiles'][h] = {}
                for tau in self.quantiles:
                    if len(all_quantiles[h][tau]) > 0:
                        quant_list = all_quantiles[h][tau]
                        if self.method == 'weighted':
                            weights_tensor = torch.tensor([
                                self.weights.get(f'model_{i}', 1.0 / len(self.models))
                                for i in range(len(quant_list))
                            ], device=quant_list[0].device, dtype=quant_list[0].dtype)
                            weights_tensor = weights_tensor.view(1, 1, -1)
                            quant_stack = torch.stack(quant_list, dim=-1)  # [B, 1, N]
                            ensemble_outputs['quantiles'][h][tau] = (quant_stack * weights_tensor).sum(dim=-1)
                        else:
                            ensemble_outputs['quantiles'][h][tau] = torch.stack(quant_list, dim=-1).mean(dim=-1)
        
        return ensemble_outputs

