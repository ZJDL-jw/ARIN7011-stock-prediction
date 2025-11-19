"""MLP model for stock prediction."""
import torch
import torch.nn as nn
from src.models.common.heads import MultiTaskHead


class MLP(nn.Module):
    """Multi-layer perceptron for stock prediction.
    
    Uses only the last timestep (t) as input.
    """
    
    def __init__(self, input_dim: int, horizons: list, quantiles: list,
                 hidden_dims: list = [64, 32], dropout: float = 0.2,
                 activation: str = 'relu'):
        """
        Args:
            input_dim: Feature dimension D
            horizons: List of prediction horizons
            quantiles: List of quantile levels
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', etc.)
        """
        super().__init__()
        self.input_dim = input_dim
        self.horizons = horizons
        self.quantiles = quantiles
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Output head
        self.head = MultiTaskHead(prev_dim, horizons, quantiles)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tensor [batch_size, T, D]
        
        Returns:
            Dict with classification and quantile outputs
        """
        # Use only last timestep
        x_last = x[:, -1, :]  # [batch_size, D]
        
        # Forward through backbone
        features = self.backbone(x_last)
        
        # Forward through head
        outputs = self.head(features)
        
        return outputs

