"""Tests for configuration schemas and loading"""
import pytest
from pathlib import Path
from pydantic import ValidationError

from audhd_correlation.config.schema import (
    AppConfig,
    DataConfig,
    FeatureConfig,
    PreprocessConfig,
    IntegrateConfig,
    ClusterConfig,
    ValidateConfig,
    CausalConfig,
)


def test_data_config_valid():
    """Test DataConfig with valid inputs"""
    config = DataConfig(
        roots={"raw": "data/raw", "processed": "data/processed"},
        datasets=["SPARK", "SSC"],
        context_fields=["fasting", "clock_time"],
        ancestry_cols=["PC1", "PC2", "PC3"],
    )
    assert len(config.datasets) == 2
    assert "SPARK" in config.datasets


def test_data_config_invalid_dataset():
    """Test DataConfig rejects invalid datasets"""
    with pytest.raises(ValidationError):
        DataConfig(
            roots={"raw": "data/raw"},
            datasets=["INVALID_DATASET"],
            context_fields=[],
            ancestry_cols=[],
        )


def test_preprocess_config_missing_threshold():
    """Test PreprocessConfig validates missing_threshold range"""
    with pytest.raises(ValidationError):
        PreprocessConfig(
            imputation="mice_delta",
            batch_method="combat",
            scaling={"genetic": "standard"},
            adjust_covariates=[],
            missing_threshold=1.5,  # Invalid: > 1.0
        )


def test_integrate_config_weights_sum():
    """Test IntegrateConfig validates weight sum"""
    with pytest.raises(ValidationError):
        IntegrateConfig(
            method="stack",
            weights={
                "genetic": 0.5,
                "metabolomic": 0.8,  # Sum > 1.0
            }
        )


def test_integrate_config_valid_weights():
    """Test IntegrateConfig with valid weights"""
    config = IntegrateConfig(
        method="stack",
        weights={
            "genetic": 0.3,
            "metabolomic": 0.4,
            "clinical": 0.3,
        }
    )
    assert sum(config.weights.values()) == pytest.approx(1.0)


def test_cluster_config_defaults():
    """Test ClusterConfig with defaults"""
    config = ClusterConfig()
    assert config.min_gap_score >= 1.0
    assert 0.5 <= config.min_stability <= 1.0
    assert config.topology_enabled is True


def test_causal_config_mediation_triplets():
    """Test CausalConfig validates mediation triplets"""
    valid_triplets = [
        {
            "exposure": "PRS_ADHD",
            "mediator": "dopamine",
            "outcome": "ADHD_RS"
        }
    ]
    config = CausalConfig(mediation_triplets=valid_triplets)
    assert len(config.mediation_triplets) == 1


def test_causal_config_invalid_mediation():
    """Test CausalConfig rejects invalid mediation triplets"""
    with pytest.raises(ValidationError):
        CausalConfig(
            mediation_triplets=[
                {"exposure": "PRS_ADHD", "outcome": "ADHD_RS"}  # Missing mediator
            ]
        )


def test_app_config_minimal():
    """Test AppConfig with minimal required fields"""
    config = AppConfig(
        data=DataConfig(
            roots={"raw": "data/raw"},
            datasets=["SPARK"],
            context_fields=[],
            ancestry_cols=[],
        ),
        features=FeatureConfig(),
        preprocess=PreprocessConfig(
            imputation="mice_delta",
            batch_method="combat",
            scaling={"genetic": "standard"},
            adjust_covariates=[],
        ),
        integrate=IntegrateConfig(
            method="mofa2",
            weights={"genetic": 0.5, "metabolomic": 0.5},
        ),
        cluster=ClusterConfig(),
        validate=ValidateConfig(),
        causal=CausalConfig(),
    )
    assert config.seed == 42  # Default value
    assert config.n_jobs >= 1


def test_app_config_forbids_extra_fields():
    """Test AppConfig rejects unknown fields"""
    with pytest.raises(ValidationError):
        AppConfig(
            unknown_field="invalid",  # Should raise error
            data=DataConfig(roots={}, datasets=[], context_fields=[], ancestry_cols=[]),
            features=FeatureConfig(),
            preprocess=PreprocessConfig(
                imputation="mice_delta",
                batch_method="combat",
                scaling={},
                adjust_covariates=[],
            ),
            integrate=IntegrateConfig(method="mofa2", weights={"a": 0.5, "b": 0.5}),
            cluster=ClusterConfig(),
            validate=ValidateConfig(),
            causal=CausalConfig(),
        )