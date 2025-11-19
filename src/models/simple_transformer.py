"""Simplified Transformer model for stock prediction."""
import torch
import torch.nn as nn
import math
from src.models.common.heads import MultiTaskHead


class SimpleTransformerEncoder(nn.Module):
    """Simplified Transformer encoder layer (no feed-forward, just attention)."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x


class SimpleTransformer(nn.Module):
    """Simplified Transformer model (attention only, no FFN)."""
    
    def __init__(self, input_dim: int, horizons: list, quantiles: list,
                 d_model: int = 128, n_heads: int = 8, depth: int = 3,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Feature dimension D
            horizons: List of prediction horizons
            quantiles: List of quantile levels
            d_model: Model dimension
            n_heads: Number of attention heads
            depth: Number of transformer layers
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
        
        # Transformer encoder layers (simplified, no FFN)
        self.encoder_layers = nn.ModuleList([
            SimpleTransformerEncoder(d_model, n_heads, dropout=dropout)
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
            # Get attention weights before applying
            if return_attention:
                _, attn_weights = encoder.self_attn(x, x, x)
                all_attentions.append(attn_weights)
            x = encoder(x)
        
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

