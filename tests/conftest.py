"""Pytest configuration and fixtures"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def random_seed():
    """Set random seed for tests"""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_data():
    """Generate small synthetic dataset for testing"""
    n_samples = 100
    n_features = 50

    return {
        "genetics": pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"gene_{i}" for i in range(n_features)]
        ),
        "metabolomics": pd.DataFrame(
            np.random.randn(n_samples, 20),
            columns=[f"metabolite_{i}" for i in range(20)]
        ),
        "clinical": pd.DataFrame(
            np.random.randn(n_samples, 10),
            columns=[f"clinical_{i}" for i in range(10)]
        ),
    }


@pytest.fixture
def config_path(tmp_path):
    """Create temporary config file"""
    config = tmp_path / "config.yaml"
    config.write_text("seed: 42\nn_jobs: 1\n")
    return config