"""PatchTST model for stock prediction."""
import torch
import torch.nn as nn
import math
from src.models.common.heads import MultiTaskHead


class PatchEmbedding(nn.Module):
    """Patch embedding for time series."""
    
    def __init__(self, d_model: int, patch_len: int, stride: int):
        """
        Args:
            d_model: Model dimension
            patch_len: Patch length
            stride: Stride for patching
        """
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1000, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [batch_size, seq_len, n_vars]
        
        Returns:
            Embedded patches [batch_size, n_patches, d_model]
        """
        n_vars = x.shape[-1]
        
        # Padding
        x = self.padding_patch_layer(x)
        
        # Unfold to patches
        x = x.unfold(dimension=-2, size=self.patch_len, step=self.stride)
        # x: [batch_size, n_vars, n_patches, patch_len]
        
        # Reshape
        batch_size, n_vars, n_patches, patch_len = x.shape
        x = x.permute(0, 2, 1, 3)  # [batch_size, n_patches, n_vars, patch_len]
        x = x.reshape(batch_size * n_patches, n_vars, patch_len)
        
        # Channel-independent: process each channel separately
        # x: [batch_size * n_patches, n_vars, patch_len]
        # We'll process each var independently
        embedded_patches = []
        for i in range(n_vars):
            var_patches = x[:, i, :]  # [batch_size * n_patches, patch_len]
            var_embedded = self.value_embedding(var_patches)  # [batch_size * n_patches, d_model]
            embedded_patches.append(var_embedded)
        
        # Stack: [batch_size * n_patches, n_vars, d_model]
        x = torch.stack(embedded_patches, dim=1)
        
        # Flatten for channel-independent processing
        # We'll average over channels or use separate processing
        # For simplicity, average over channels
        x = x.mean(dim=1)  # [batch_size * n_patches, d_model]
        
        # Reshape back
        x = x.reshape(batch_size, n_patches, self.value_embedding.out_features)
        
        # Add position embedding
        n_patches = x.shape[1]
        x = x + self.position_embedding[:n_patches, :].unsqueeze(0)
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder layer."""
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Self-attention
        attn_out, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x, attn_weights


class PatchTST(nn.Module):
    """PatchTST model for time series forecasting."""
    
    def __init__(self, input_dim: int, horizons: list, quantiles: list,
                 d_model: int = 64, n_heads: int = 4, depth: int = 2,
                 patch_len: int = 6, stride: int = 3, dropout: float = 0.1,
                 channel_independent: bool = True):
        """
        Args:
            input_dim: Feature dimension D
            horizons: List of prediction horizons
            quantiles: List of quantile levels
            d_model: Model dimension
            n_heads: Number of attention heads
            depth: Number of transformer layers
            patch_len: Patch length
            stride: Stride for patching
            dropout: Dropout rate
            channel_independent: Whether to process channels independently
        """
        super().__init__()
        self.input_dim = input_dim
        self.horizons = horizons
        self.quantiles = quantiles
        self.channel_independent = channel_independent
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(d_model, patch_len, stride)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model, n_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Pooling (use CLS token or average)
        self.pooling = 'mean'  # or 'cls'
        
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
        # Patch embedding
        # x: [batch_size, T, D]
        # For channel-independent, we process each channel separately
        if self.channel_independent:
            # Process each channel
            batch_size, T, D = x.shape
            patch_embeddings = []
            
            for d in range(D):
                x_channel = x[:, :, d:d+1]  # [batch_size, T, 1]
                patch_emb = self.patch_embed(x_channel)  # [batch_size, n_patches, d_model]
                patch_embeddings.append(patch_emb)
            
            # Average over channels
            x = torch.stack(patch_embeddings, dim=0).mean(dim=0)  # [batch_size, n_patches, d_model]
        else:
            x = self.patch_embed(x)
        
        # Transformer encoder
        all_attentions = []
        for encoder in self.encoder_layers:
            x, attn_weights = encoder(x)
            all_attentions.append(attn_weights)
        
        # Pooling
        if self.pooling == 'mean':
            pooled = x.mean(dim=1)  # [batch_size, d_model]
        else:
            pooled = x[:, 0, :]  # CLS token
        
        # Output head
        outputs = self.head(pooled)
        
        if return_attention:
            outputs['attention'] = all_attentions
        
        return outputs

