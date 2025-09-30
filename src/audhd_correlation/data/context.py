"""Context variable tagging"""
from typing import Dict, Any
from ..config.schema import AppConfig


def add_tags(tables: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """Add context tags (fasting, time-of-day, etc.)"""
    # TODO: Implement context tagging
    return tables