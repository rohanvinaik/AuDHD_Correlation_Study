"""Ontology mapping (HPO, SNOMED, RxNorm, ExO)"""
from typing import Dict, Any
from ..config.schema import AppConfig


def map_all(tables: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """Map all tables to ontologies"""
    # TODO: Implement ontology mapping
    return tables