"""Data loaders for SPARK, SSC, ABCD, UK Biobank"""
from typing import Dict, Any
from ..config.schema import AppConfig


def fetch_all_datasets(cfg: AppConfig) -> None:
    """Fetch raw datasets"""
    # TODO: Implement dataset downloads
    pass


def fetch_references(cfg: AppConfig) -> None:
    """Fetch reference data (ontologies, pathways)"""
    # TODO: Implement reference downloads
    pass


def load_all(cfg: AppConfig) -> Dict[str, Any]:
    """Load all raw data tables"""
    # TODO: Implement data loading
    return {}