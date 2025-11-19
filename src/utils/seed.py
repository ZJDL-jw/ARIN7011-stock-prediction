"""Set random seeds for reproducibility."""
import random
import numpy as np
import torch
import os


def set_seed(seed: int = 2025):
    """Set random seeds for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

