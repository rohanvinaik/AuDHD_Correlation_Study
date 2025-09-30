"""
AuDHD Correlation Study
Multi-omics integration pipeline for ADHD/Autism subtyping
"""

__version__ = "0.1.0"
__author__ = "Rohan Vinaik"
__email__ = "your.email@example.com"

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Default paths
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
INTERIM_DATA_DIR = DATA_ROOT / "interim"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
EXTERNAL_DATA_DIR = DATA_ROOT / "external"

__all__ = [
    "__version__",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "DATA_ROOT",
]