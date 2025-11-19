"""1D CNN model for stock prediction."""
import torch
import torch.nn as nn
from src.models.common.heads import MultiTaskHead


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for sequential stock prediction."""
    
    def __init__(self, input_dim: int, horizons: list, quantiles: list,
                 num_filters: list = [64, 128, 256], kernel_sizes: list = [3, 5, 7],
                 dropout: float = 0.3, pool_size: int = 2):
        """
        Args:
            input_dim: Feature dimension D
            horizons: List of prediction horizons
            quantiles: List of quantile levels
            num_filters: List of filter numbers for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout: Dropout rate
            pool_size: Pooling size
        """
        super().__init__()
        self.input_dim = input_dim
        self.horizons = horizons
        self.quantiles = quantiles
        
        # Ensure num_filters and kernel_sizes have same length
        if len(num_filters) != len(kernel_sizes):
            raise ValueError("num_filters and kernel_sizes must have same length")
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # Same padding
                )
            )
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            if i < len(num_filters) - 1:  # Don't pool after last conv
                conv_layers.append(nn.MaxPool1d(kernel_size=pool_size))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output dimension (last filter number)
        cnn_output_dim = num_filters[-1]
        
        # Output head
        self.head = MultiTaskHead(cnn_output_dim, horizons, quantiles)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tensor [batch_size, T, D]
        
        Returns:
            Dict with classification and quantile outputs
        """
        # Convert from [B, T, D] to [B, D, T] for Conv1d
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Global average pooling: [B, C, T] -> [B, C, 1]
        x = self.global_pool(x)
        
        # Squeeze: [B, C, 1] -> [B, C]
        x = x.squeeze(-1)
        
        # Forward through head
        outputs = self.head(x)
        
        return outputs

