"""Feature scaling"""
from typing import Dict, Any
from ..config.schema import AppConfig


def apply(X: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """Apply feature scaling (z-score, robust, etc.)"""
    # TODO: Implement scaling
    return X