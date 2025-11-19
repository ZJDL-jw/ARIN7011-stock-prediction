"""Standard Transformer model for stock prediction."""
import torch
import torch.nn as nn
import math
from src.models.common.heads import MultiTaskHead


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer encoder layer (with FFN)."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None,
                 dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass."""
        # Self-attention
        attn_out, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x, attn_weights


class Transformer(nn.Module):
    """Standard Transformer model (with full encoder layers)."""
    
    def __init__(self, input_dim: int, horizons: list, quantiles: list,
                 d_model: int = 128, n_heads: int = 8, depth: int = 3,
                 d_ff: int = None, dropout: float = 0.1):
        """
        Args:
            input_dim: Feature dimension D
            horizons: List of prediction horizons
            quantiles: List of quantile levels
            d_model: Model dimension
            n_heads: Number of attention heads
            depth: Number of transformer layers
            d_ff: Feed-forward dimension (default: 4 * d_model)
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.horizons = horizons
        self.quantiles = quantiles
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Pooling
        self.pooling = 'mean'
        
        # Output head
        self.head = MultiTaskHead(d_model, horizons, quantiles)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> dict:
        """
        Args:
            x: Input tensor [batch_size, T, D]
            return_attention: Whether to return attention weights
        
        Returns:
            Dict with outputs and optionally attention weights
        """
        batch_size, T, D = x.shape
        
        # Input projection
        x = self.input_proj(x)  # [batch_size, T, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:T, :].unsqueeze(0)
        
        # Transformer encoder
        all_attentions = []
        for encoder in self.encoder_layers:
            x, attn_weights = encoder(x)
            if return_attention:
                all_attentions.append(attn_weights)
        
        # Pooling
        if self.pooling == 'mean':
            pooled = x.mean(dim=1)  # [batch_size, d_model]
        else:
            pooled = x[:, 0, :]  # First token
        
        # Output head
        outputs = self.head(pooled)
        
        if return_attention:
            outputs['attention'] = all_attentions
        
        return outputs

