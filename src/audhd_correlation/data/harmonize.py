"""Site/batch harmonization"""
from typing import Dict, Any
from ..config.schema import AppConfig


def harmonize_all(tables: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """Harmonize across sites/batches"""
    # TODO: Implement harmonization
    return tables