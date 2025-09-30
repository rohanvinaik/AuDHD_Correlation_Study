"""Tests for config loader"""
import pytest
from pathlib import Path
import tempfile
import yaml

from audhd_correlation.config.loader import load_config, load_yaml, save_config
from audhd_correlation.config.schema import AppConfig
from pydantic import ValidationError


@pytest.fixture
def temp_config_file():
    """Create temporary config file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "seed": 42,
            "n_jobs": 4,
            "data": {
                "roots": {"raw": "data/raw"},
                "datasets": ["SPARK"],
                "context_fields": ["fasting"],
                "ancestry_cols": ["PC1"],
            },
            "features": {},
            "preprocess": {
                "imputation": "mice_delta",
                "batch_method": "combat",
                "scaling": {"genetic": "standard"},
                "adjust_covariates": ["age"],
            },
            "integrate": {
                "method": "mofa2",
                "weights": {"genetic": 0.5, "metabolomic": 0.5},
            },
            "cluster": {},
            "validate": {},
            "causal": {},
        }
        yaml.safe_dump(config, f)
        yield Path(f.name)
        Path(f.name).unlink()


def test_load_yaml(temp_config_file):
    """Test YAML loading"""
    data = load_yaml(temp_config_file)
    assert isinstance(data, dict)
    assert "seed" in data
    assert data["seed"] == 42


def test_load_yaml_missing_file():
    """Test loading non-existent file raises error"""
    with pytest.raises(FileNotFoundError):
        load_yaml("nonexistent_file.yaml")


def test_load_config(temp_config_file):
    """Test config loading and validation"""
    config = load_config(temp_config_file)
    assert isinstance(config, AppConfig)
    assert config.seed == 42
    assert config.n_jobs == 4
    assert "SPARK" in config.data.datasets


def test_save_config(tmp_path):
    """Test config saving"""
    from audhd_correlation.config.schema import DataConfig, FeatureConfig

    config = AppConfig(
        seed=123,
        n_jobs=2,
        data=DataConfig(
            roots={"raw": "data/raw"},
            datasets=["SPARK"],
            context_fields=[],
            ancestry_cols=[],
        ),
        features=FeatureConfig(),
        preprocess={
            "imputation": "mice_delta",
            "batch_method": "combat",
            "scaling": {"genetic": "standard"},
            "adjust_covariates": [],
        },
        integrate={
            "method": "mofa2",
            "weights": {"a": 0.5, "b": 0.5},
        },
        cluster={},
        validate={},
        causal={},
    )

    output_path = tmp_path / "test_config.yaml"
    save_config(config, output_path)

    assert output_path.exists()

    # Load back and verify
    loaded = load_config(output_path)
    assert loaded.seed == 123
    assert loaded.n_jobs == 2


def test_load_invalid_config(tmp_path):
    """Test loading invalid config raises ValidationError"""
    invalid_config = tmp_path / "invalid.yaml"
    with open(invalid_config, "w") as f:
        yaml.safe_dump({
            "seed": 42,
            "data": {
                "roots": {},
                "datasets": ["INVALID_DATASET"],  # Invalid
                "context_fields": [],
                "ancestry_cols": [],
            }
        }, f)

    with pytest.raises(ValidationError):
        load_config(invalid_config)