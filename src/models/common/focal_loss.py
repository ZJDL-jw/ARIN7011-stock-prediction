"""Focal Loss implementation for class imbalance."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        """
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model [batch_size]
            targets: True labels [batch_size]
        
        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

