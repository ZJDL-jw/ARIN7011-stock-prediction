"""GRU model for stock prediction."""
import torch
import torch.nn as nn
from src.models.common.heads import MultiTaskHead


class GRU(nn.Module):
    """GRU model for sequential stock prediction."""
    
    def __init__(self, input_dim: int, horizons: list, quantiles: list,
                 hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        """
        Args:
            input_dim: Feature dimension D
            horizons: List of prediction horizons
            quantiles: List of quantile levels
            hidden_size: GRU hidden size
            num_layers: Number of GRU layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__()
        self.input_dim = input_dim
        self.horizons = horizons
        self.quantiles = quantiles
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU
        # Note: dropout only applies between layers, not after the last layer
        self.gru = nn.GRU(
            input_dim, hidden_size, num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Additional dropout after GRU if needed
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Output dimension
        gru_output_dim = hidden_size * (2 if bidirectional else 1)
        
        # Output head
        self.head = MultiTaskHead(gru_output_dim, horizons, quantiles)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tensor [batch_size, T, D]
        
        Returns:
            Dict with classification and quantile outputs
        """
        # GRU forward
        gru_out, h_n = self.gru(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            last_hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            last_hidden = h_n[-1]
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Forward through head
        outputs = self.head(last_hidden)
        
        return outputs

