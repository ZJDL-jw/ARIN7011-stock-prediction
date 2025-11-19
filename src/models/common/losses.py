"""Loss functions for multi-task learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.common.focal_loss import FocalLoss


class PinballLoss(nn.Module):
    """Pinball loss for quantile regression."""
    
    def __init__(self, tau: float):
        """
        Args:
            tau: Quantile level (0 < tau < 1)
        """
        super().__init__()
        self.tau = tau
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted quantile values [batch_size]
            y_true: True values [batch_size]
        
        Returns:
            Loss value
        """
        error = y_true - y_pred
        loss = torch.max(self.tau * error, (self.tau - 1) * error)
        return loss.mean()


class MultiQuantileLoss(nn.Module):
    """Multi-quantile pinball loss."""
    
    def __init__(self, quantiles: list, weights: dict = None):
        """
        Args:
            quantiles: List of quantile levels
            weights: Optional dict mapping quantile -> weight
        """
        super().__init__()
        self.quantiles = quantiles
        self.weights = weights or {tau: 1.0 for tau in quantiles}
        self.losses = nn.ModuleDict({
            str(tau).replace('.', '_'): PinballLoss(tau) for tau in quantiles
        })
    
    def forward(self, y_pred: dict, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Dict mapping quantile -> predicted values [batch_size]
            y_true: True values [batch_size]
        
        Returns:
            Weighted sum of pinball losses
        """
        total_loss = 0.0
        for tau in self.quantiles:
            tau_str = str(tau)
            tau_key = tau_str.replace('.', '_')
            if tau_str in y_pred:
                loss = self.losses[tau_key](y_pred[tau_str].squeeze(), y_true.squeeze())
                total_loss += self.weights.get(tau, 1.0) * loss
        
        return total_loss


class CRPSLoss(nn.Module):
    """Approximate CRPS using multiple quantiles."""
    
    def __init__(self, quantiles: list):
        """
        Args:
            quantiles: List of quantile levels (sorted)
        """
        super().__init__()
        self.quantiles = sorted(quantiles)
    
    def forward(self, y_pred: dict, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Dict mapping quantile -> predicted values [batch_size]
            y_true: True values [batch_size]
        
        Returns:
            Approximate CRPS
        """
        crps = 0.0
        
        for i, tau in enumerate(self.quantiles):
            tau_str = str(tau)
            q_pred = y_pred[tau_str].squeeze()
            
            if i == 0:
                # First quantile
                crps += tau * torch.mean(torch.clamp(y_true.squeeze() - q_pred, min=0))
            elif i == len(self.quantiles) - 1:
                # Last quantile
                crps += (1 - tau) * torch.mean(torch.clamp(q_pred - y_true.squeeze(), min=0))
            else:
                # Middle quantiles
                q_prev = y_pred[str(self.quantiles[i-1])].squeeze()
                crps += (tau - self.quantiles[i-1]) * torch.mean(
                    torch.clamp(y_true.squeeze() - q_pred, min=0)
                )
        
        return crps


class MultiTaskLoss(nn.Module):
    """Combined loss for classification and quantile regression."""
    
    def __init__(self, horizons: list, quantiles: list,
                 bce_weight: float = 1.0, pinball_weight: float = 1.0,
                 use_focal: bool = True, focal_alpha: float = 1.0, focal_gamma: float = 2.0):
        """
        Args:
            horizons: List of prediction horizons
            quantiles: List of quantile levels
            bce_weight: Weight for BCE loss
            pinball_weight: Weight for pinball loss (increased default to 1.0)
            use_focal: Whether to use Focal Loss instead of BCE
            focal_alpha: Alpha parameter for Focal Loss
            focal_gamma: Gamma parameter for Focal Loss
        """
        super().__init__()
        self.horizons = horizons
        self.quantiles = quantiles
        self.bce_weight = bce_weight
        self.pinball_weight = pinball_weight
        
        if use_focal:
            self.bce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
        self.pinball_loss = MultiQuantileLoss(quantiles)
    
    def forward(self, outputs: dict, targets: dict) -> dict:
        """
        Args:
            outputs: Dict with 'classification' and 'quantiles' keys
            targets: Dict with 'classification' and 'regression' keys
        
        Returns:
            Dict with individual and total losses
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Classification losses
        for h in self.horizons:
            if h in outputs.get('classification', {}) and h in targets.get('classification', {}):
                logits = outputs['classification'][h]
                labels = targets['classification'][h].float()
                bce = self.bce_loss(logits, labels)
                loss_dict[f'bce_h{h}'] = bce
                total_loss += self.bce_weight * bce
        
        # Quantile regression losses
        for h in self.horizons:
            if h in outputs.get('quantiles', {}) and h in targets.get('regression', {}):
                quantile_preds = outputs['quantiles'][h]
                y_true = targets['regression'][h]
                
                # 确保quantile_preds的key是str格式，与MultiQuantileLoss期望的格式一致
                quantile_preds_str = {}
                for tau in self.quantiles:
                    tau_str = str(tau)
                    if tau in quantile_preds:
                        quantile_preds_str[tau_str] = quantile_preds[tau]
                    elif tau_str in quantile_preds:
                        quantile_preds_str[tau_str] = quantile_preds[tau_str]
                
                if len(quantile_preds_str) > 0:
                    pinball = self.pinball_loss(quantile_preds_str, y_true)
                    loss_dict[f'pinball_h{h}'] = pinball
                    total_loss += self.pinball_weight * pinball
                else:
                    # 如果没有有效的分位数预测，记录0损失
                    loss_dict[f'pinball_h{h}'] = torch.tensor(0.0, device=y_true.device)
        
        loss_dict['total'] = total_loss
        return loss_dict

