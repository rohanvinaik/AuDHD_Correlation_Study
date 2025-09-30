"""Deterministic seeding for reproducibility"""
import random
import numpy as np
from typing import Optional


def set_seed(seed: Optional[int] = None) -> None:
    """
    Set random seed for Python, NumPy for reproducibility

    Args:
        seed: Random seed value. If None, uses default 42
    """
    if seed is None:
        seed = 42

    random.seed(seed)
    np.random.seed(seed)

    # If torch is available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass