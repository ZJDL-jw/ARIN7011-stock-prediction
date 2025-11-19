"""Plotting utilities for visualization."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from pathlib import Path
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, 
                   save_path: str, title: str = "ROC Curve"):
    """Plot ROC curve."""
    if len(np.unique(y_true)) < 2:
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_pred: np.ndarray,
                  save_path: str, title: str = "Precision-Recall Curve"):
    """Plot Precision-Recall curve."""
    if len(np.unique(y_true)) < 2:
        return
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_reliability(y_true: np.ndarray, y_pred: np.ndarray,
                    save_path: str, title: str = "Reliability Diagram",
                    n_bins: int = 10):
    """Plot reliability diagram for probability calibration."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence = y_pred[in_bin].mean()
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence)
            counts.append(prop_in_bin)
        else:
            accuracies.append(0)
            confidences.append((bin_lower + bin_upper) / 2)
            counts.append(0)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(confidences, accuracies, 'o-', label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_heatmap(attention: np.ndarray, save_path: str,
                           title: str = "Attention Heatmap"):
    """Plot attention weights as heatmap.
    
    Args:
        attention: Attention matrix of shape [n_heads, seq_len, seq_len] or [seq_len, seq_len]
    """
    if attention.ndim == 3:
        # Average over heads
        attention = attention.mean(axis=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, cmap='viridis', cbar=True)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title(title)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_walkforward_boxplot(results: dict, save_path: str,
                            metric: str = 'auc', title: str = "Walk-forward Results"):
    """Plot boxplot of walk-forward evaluation results.
    
    Args:
        results: Dict mapping year -> list of metric values
        metric: Metric name to plot
        save_path: Path to save figure
    """
    years = sorted(results.keys())
    data = [results[year] for year in years]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=years)
    plt.xlabel('Year')
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cumulative_returns(returns: dict, save_path: str,
                           title: str = "Cumulative Returns"):
    """Plot cumulative returns for strategy and benchmarks.
    
    Args:
        returns: Dict mapping strategy name -> array of returns
    """
    plt.figure(figsize=(12, 6))
    for name, rets in returns.items():
        cumret = np.cumprod(1 + rets) - 1
        plt.plot(cumret, label=name)
    
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_waterfall(baseline: float, drops: dict, save_path: str,
                           title: str = "Ablation Study"):
    """Plot waterfall chart for ablation study.
    
    Args:
        baseline: Baseline metric value
        drops: Dict mapping factor group -> metric value after dropping
    """
    groups = list(drops.keys())
    values = [drops[g] for g in groups]
    
    x_pos = np.arange(len(groups))
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_pos, values, alpha=0.7)
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.3f})')
    plt.xlabel('Dropped Factor Group')
    plt.ylabel('Metric Value')
    plt.title(title)
    plt.xticks(x_pos, groups, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

