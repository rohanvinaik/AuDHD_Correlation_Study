"""Quality control per modality"""
from typing import Dict, Any
from ..config.schema import AppConfig


def run_all(tables: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """Run QC on all modalities"""
    # TODO: Implement QC
    return tables