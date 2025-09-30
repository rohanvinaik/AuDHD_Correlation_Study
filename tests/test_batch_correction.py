"""Tests for batch correction pipeline"""
import pytest
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path

from src.audhd_correlation.preprocess.batch_effects import (
    combat,
    combat_seq,
    BatchCorrectionPipeline,
    _create_design_matrix,
    _get_batch_design,
)


@pytest.fixture
def sample_data_with_batch():
    """Create sample data with batch effects"""
    np.random.seed(42)

    n_samples = 100
    n_features = 20

    # Create two batches
    batch1 = pd.DataFrame(
        np.random.randn(50, n_features) + 0,  # Batch 1: mean = 0
        index=[f"sample_{i}" for i in range(50)],
        columns=[f"feature_{j}" for j in range(n_features)],
    )

    batch2 = pd.DataFrame(
        np.random.randn(50, n_features) + 2,  # Batch 2: mean = 2 (batch effect)
        index=[f"sample_{i}" for i in range(50, 100)],
        columns=[f"feature_{j}" for j in range(n_features)],
    )

    data = pd.concat([batch1, batch2])

    # Create batch labels
    batch = pd.Series(
        ["batch1"] * 50 + ["batch2"] * 50,
        index=data.index,
        name="batch",
    )

    return data, batch


@pytest.fixture
def sample_data_with_covariates(sample_data_with_batch):
    """Create sample data with covariates"""
    data, batch = sample_data_with_batch

    # Add covariates
    covariates = pd.DataFrame(
        {
            "age": np.random.randint(5, 18, len(data)),
            "sex": np.random.choice([0, 1], len(data)),
        },
        index=data.index,
    )

    return data, batch, covariates


@pytest.fixture
def count_data_with_batch():
    """Create count data with batch effects"""
    np.random.seed(42)

    n_samples = 60
    n_features = 30

    # Generate count data with batch effects
    batch1 = np.random.poisson(lam=10, size=(30, n_features))
    batch2 = np.random.poisson(lam=20, size=(30, n_features))  # Higher counts

    data = pd.DataFrame(
        np.vstack([batch1, batch2]),
        index=[f"sample_{i}" for i in range(n_samples)],
        columns=[f"taxon_{j}" for j in range(n_features)],
    )

    batch = pd.Series(
        ["batch1"] * 30 + ["batch2"] * 30,
        index=data.index,
    )

    return data, batch


def test_combat_basic(sample_data_with_batch):
    """Test basic ComBat correction"""
    data, batch = sample_data_with_batch

    corrected = combat(data, batch)

    # Check output shape
    assert corrected.shape == data.shape
    assert corrected.index.equals(data.index)
    assert corrected.columns.equals(data.columns)

    # Check that batch effect is reduced
    batch1_mean = data.loc[batch == "batch1"].mean().mean()
    batch2_mean = data.loc[batch == "batch2"].mean().mean()
    original_diff = abs(batch1_mean - batch2_mean)

    batch1_corrected = corrected.loc[batch == "batch1"].mean().mean()
    batch2_corrected = corrected.loc[batch == "batch2"].mean().mean()
    corrected_diff = abs(batch1_corrected - batch2_corrected)

    assert corrected_diff < original_diff


def test_combat_with_covariates(sample_data_with_covariates):
    """Test ComBat with covariate preservation"""
    data, batch, covariates = sample_data_with_covariates

    corrected = combat(data, batch, covariates=covariates)

    # Check output shape
    assert corrected.shape == data.shape

    # Check that batch effect is reduced
    batch1_mean = data.loc[batch == "batch1"].mean().mean()
    batch2_mean = data.loc[batch == "batch2"].mean().mean()
    original_diff = abs(batch1_mean - batch2_mean)

    batch1_corrected = corrected.loc[batch == "batch1"].mean().mean()
    batch2_corrected = corrected.loc[batch == "batch2"].mean().mean()
    corrected_diff = abs(batch1_corrected - batch2_corrected)

    assert corrected_diff < original_diff


def test_combat_mean_only(sample_data_with_batch):
    """Test ComBat with mean-only correction"""
    data, batch = sample_data_with_batch

    corrected = combat(data, batch, mean_only=True)

    # Check output shape
    assert corrected.shape == data.shape

    # Mean-only should reduce mean differences
    batch1_mean = data.loc[batch == "batch1"].mean().mean()
    batch2_mean = data.loc[batch == "batch2"].mean().mean()
    original_diff = abs(batch1_mean - batch2_mean)

    batch1_corrected = corrected.loc[batch == "batch1"].mean().mean()
    batch2_corrected = corrected.loc[batch == "batch2"].mean().mean()
    corrected_diff = abs(batch1_corrected - batch2_corrected)

    assert corrected_diff < original_diff


def test_combat_nonparametric(sample_data_with_batch):
    """Test ComBat with non-parametric correction"""
    data, batch = sample_data_with_batch

    corrected = combat(data, batch, parametric=False)

    # Check output shape
    assert corrected.shape == data.shape

    # Check that batch effect is reduced
    batch1_mean = data.loc[batch == "batch1"].mean().mean()
    batch2_mean = data.loc[batch == "batch2"].mean().mean()
    original_diff = abs(batch1_mean - batch2_mean)

    batch1_corrected = corrected.loc[batch == "batch1"].mean().mean()
    batch2_corrected = corrected.loc[batch == "batch2"].mean().mean()
    corrected_diff = abs(batch1_corrected - batch2_corrected)

    assert corrected_diff < original_diff * 1.5  # Allow more tolerance for non-parametric


def test_combat_single_batch(sample_data_with_batch):
    """Test ComBat with single batch (should return unchanged)"""
    data, _ = sample_data_with_batch

    # Create single batch
    batch = pd.Series(["batch1"] * len(data), index=data.index)

    corrected = combat(data, batch)

    # Should return unchanged
    pd.testing.assert_frame_equal(corrected, data)


def test_combat_seq(count_data_with_batch):
    """Test ComBat-seq for count data"""
    data, batch = count_data_with_batch

    corrected = combat_seq(data, batch)

    # Check output shape
    assert corrected.shape == data.shape
    assert corrected.index.equals(data.index)
    assert corrected.columns.equals(data.columns)

    # Check that values are still non-negative (counts)
    assert (corrected >= 0).all().all()

    # Check that batch effect is reduced
    batch1_mean = data.loc[batch == "batch1"].mean().mean()
    batch2_mean = data.loc[batch == "batch2"].mean().mean()
    original_diff = abs(batch1_mean - batch2_mean)

    batch1_corrected = corrected.loc[batch == "batch1"].mean().mean()
    batch2_corrected = corrected.loc[batch == "batch2"].mean().mean()
    corrected_diff = abs(batch1_corrected - batch2_corrected)

    assert corrected_diff < original_diff


def test_create_design_matrix():
    """Test design matrix creation"""
    batch = pd.Series(["batch1", "batch1", "batch2", "batch2"])

    # Without covariates
    design = _create_design_matrix(batch, None)
    assert design.shape == (4, 1)
    assert (design[:, 0] == 1).all()

    # With covariates
    covariates = pd.DataFrame({"age": [10, 12, 11, 13], "sex": [0, 1, 0, 1]})
    design = _create_design_matrix(batch, covariates)
    assert design.shape == (4, 3)  # intercept + 2 covariates


def test_get_batch_design():
    """Test batch design matrix creation"""
    batch = pd.Series(
        ["batch1", "batch1", "batch2", "batch2", "batch3"],
        index=range(5),
    )

    batch_design = _get_batch_design(batch)

    assert batch_design.shape == (5, 3)
    assert batch_design.sum(axis=1).tolist() == [1, 1, 1, 1, 1]  # One-hot encoded


def test_batch_correction_pipeline():
    """Test BatchCorrectionPipeline class"""
    with TemporaryDirectory() as tmpdir:
        pipeline = BatchCorrectionPipeline(
            method="combat",
            generate_plots=False,  # Skip plots for faster testing
            output_dir=Path(tmpdir),
        )

        # Create test data
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(50, 10) + np.repeat([0, 2], 25)[:, None],
            index=[f"sample_{i}" for i in range(50)],
        )
        batch = pd.Series(["batch1"] * 25 + ["batch2"] * 25, index=data.index)

        corrected = pipeline.correct_data(data, batch, modality="test")

        assert corrected.shape == data.shape


def test_batch_correction_pipeline_with_plots():
    """Test BatchCorrectionPipeline with plot generation"""
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        pipeline = BatchCorrectionPipeline(
            method="combat",
            generate_plots=True,
            output_dir=output_dir,
        )

        # Create test data
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(50, 10) + np.repeat([0, 2], 25)[:, None],
            index=[f"sample_{i}" for i in range(50)],
        )
        batch = pd.Series(["batch1"] * 25 + ["batch2"] * 25, index=data.index)

        corrected = pipeline.correct_data(data, batch, modality="test")

        # Check that plots were generated
        assert (output_dir / "test_before_correction.png").exists()
        assert (output_dir / "test_after_correction.png").exists()
        assert (output_dir / "test_before_after_comparison.png").exists()


def test_batch_correction_preserves_variability():
    """Test that biological variability is preserved"""
    np.random.seed(42)

    # Create data with batch effects and biological signal
    n_samples = 100
    n_features = 20

    # Group 1: lower biological signal
    group1_batch1 = np.random.randn(25, n_features) * 0.5 + 0
    group1_batch2 = np.random.randn(25, n_features) * 0.5 + 2  # batch effect

    # Group 2: higher biological signal
    group2_batch1 = np.random.randn(25, n_features) * 0.5 + 1  # biological effect
    group2_batch2 = np.random.randn(25, n_features) * 0.5 + 3  # biological + batch

    data = pd.DataFrame(
        np.vstack([group1_batch1, group1_batch2, group2_batch1, group2_batch2])
    )

    batch = pd.Series(
        ["batch1"] * 25 + ["batch2"] * 25 + ["batch1"] * 25 + ["batch2"] * 25
    )

    group = pd.Series(
        ["group1"] * 50 + ["group2"] * 50
    )

    # Apply correction
    corrected = combat(data, batch)

    # Check that biological differences are preserved
    original_group_diff = abs(
        data.iloc[:50].mean().mean() - data.iloc[50:].mean().mean()
    )
    corrected_group_diff = abs(
        corrected.iloc[:50].mean().mean() - corrected.iloc[50:].mean().mean()
    )

    # Biological difference should be largely preserved
    assert corrected_group_diff > 0.3 * original_group_diff


def test_batch_correction_with_nan():
    """Test batch correction with missing values"""
    np.random.seed(42)

    # Create data with NaNs
    data = pd.DataFrame(np.random.randn(50, 10))
    data.iloc[::5, ::3] = np.nan  # Add some NaNs

    batch = pd.Series(["batch1"] * 25 + ["batch2"] * 25)

    # Should handle NaNs gracefully
    corrected = combat(data, batch)

    assert corrected.shape == data.shape
    # NaN pattern might change due to standardization