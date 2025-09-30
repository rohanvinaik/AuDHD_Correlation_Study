"""Covariate adjustment"""
from typing import Dict, Any
from ..config.schema import AppConfig


def partial_out(X: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """Partial out covariates (age, sex, ancestry PCs)"""
    # TODO: Implement covariate adjustment
    return X