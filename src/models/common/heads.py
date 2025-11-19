"""Output heads for multi-task prediction."""
import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    """Multi-task output head: classification + quantile regression."""
    
    def __init__(self, input_dim: int, horizons: list, quantiles: list,
                 hidden_dim: int = 32):
        """
        Args:
            input_dim: Input feature dimension
            horizons: List of prediction horizons
            quantiles: List of quantile levels
            hidden_dim: Hidden dimension for shared layers
        """
        super().__init__()
        self.horizons = horizons
        self.quantiles = quantiles
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification heads (one per horizon)
        self.classification_heads = nn.ModuleDict({
            str(h): nn.Linear(hidden_dim, 1) for h in horizons
        })
        
        # Quantile regression heads (one per horizon, multiple quantiles)
        self.quantile_heads = nn.ModuleDict({
            f"h{h}_tau{tau}".replace('.', '_'): nn.Linear(hidden_dim, 1)
            for h in horizons
            for tau in quantiles
        })
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            Dict with 'classification' and 'quantiles' outputs
        """
        shared = self.shared(x)
        
        outputs = {
            'classification': {},
            'quantiles': {}
        }
        
        # Classification outputs
        for h in self.horizons:
            outputs['classification'][h] = self.classification_heads[str(h)](shared)
        
        # Quantile outputs
        for h in self.horizons:
            outputs['quantiles'][h] = {
                tau: self.quantile_heads[f"h{h}_tau{tau}".replace('.', '_')](shared)
                for tau in self.quantiles
            }
        
        return outputs

