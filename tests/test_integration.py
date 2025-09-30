"""Tests for multi-omics integration"""
import pytest
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path

from src.audhd_correlation.integrate.mofa import MOFAIntegration
from src.audhd_correlation.integrate.methods import (
    _integrate_mofa,
    _integrate_stack,
    save_integration_results,
    load_integration_results,
)


@pytest.fixture
def multi_omics_data():
    """Create synthetic multi-omics data"""
    np.random.seed(42)

    n_samples = 50
    n_factors = 3

    # Generate latent factors
    Z_true = np.random.randn(n_samples, n_factors)

    # Generate data for each omics layer
    # Genetic data
    n_genetic = 20
    W_genetic = np.random.randn(n_genetic, n_factors) * 0.5
    genetic_data = Z_true @ W_genetic.T + np.random.randn(n_samples, n_genetic) * 0.1

    # Metabolomic data
    n_metabol = 15
    W_metabol = np.random.randn(n_metabol, n_factors) * 0.5
    metabol_data = Z_true @ W_metabol.T + np.random.randn(n_samples, n_metabol) * 0.1

    # Clinical data
    n_clinical = 10
    W_clinical = np.random.randn(n_clinical, n_factors) * 0.5
    clinical_data = Z_true @ W_clinical.T + np.random.randn(n_samples, n_clinical) * 0.1

    # Create DataFrames
    data_dict = {
        "genetic": pd.DataFrame(
            genetic_data,
            index=[f"sample_{i}" for i in range(n_samples)],
            columns=[f"gene_{j}" for j in range(n_genetic)],
        ),
        "metabolomic": pd.DataFrame(
            metabol_data,
            index=[f"sample_{i}" for i in range(n_samples)],
            columns=[f"metabolite_{j}" for j in range(n_metabol)],
        ),
        "clinical": pd.DataFrame(
            clinical_data,
            index=[f"sample_{i}" for i in range(n_samples)],
            columns=[f"clinical_{j}" for j in range(n_clinical)],
        ),
    }

    return data_dict, Z_true


@pytest.fixture
def multi_omics_data_with_missing(multi_omics_data):
    """Create multi-omics data with missing values"""
    data_dict, Z_true = multi_omics_data

    # Add missing values (20% missing in each view)
    for view, df in data_dict.items():
        mask = np.random.rand(*df.shape) < 0.2
        data_dict[view] = df.mask(mask)

    return data_dict, Z_true


@pytest.fixture
def partially_overlapping_samples():
    """Create data with partial sample overlap"""
    np.random.seed(42)

    n_samples_total = 60
    n_factors = 3

    # Generate latent factors for all samples
    Z_true = np.random.randn(n_samples_total, n_factors)

    # View 1: samples 0-40
    view1_data = Z_true[:40] @ np.random.randn(n_factors, 10).T
    view1 = pd.DataFrame(
        view1_data,
        index=[f"sample_{i}" for i in range(40)],
        columns=[f"view1_feat_{j}" for j in range(10)],
    )

    # View 2: samples 20-60
    view2_data = Z_true[20:] @ np.random.randn(n_factors, 8).T
    view2 = pd.DataFrame(
        view2_data,
        index=[f"sample_{i}" for i in range(20, 60)],
        columns=[f"view2_feat_{j}" for j in range(8)],
    )

    return {"view1": view1, "view2": view2}, Z_true


def test_mofa_initialization():
    """Test MOFA initialization"""
    mofa = MOFAIntegration(n_factors=5, n_iterations=10)

    assert mofa.n_factors == 5
    assert mofa.n_iterations == 10
    assert mofa.Z is None  # Not fitted yet


def test_mofa_fit(multi_omics_data):
    """Test MOFA fitting"""
    data_dict, _ = multi_omics_data

    mofa = MOFAIntegration(n_factors=3, n_iterations=50)
    mofa.fit(data_dict)

    # Check that parameters are fitted
    assert mofa.Z is not None
    assert mofa.Z.shape == (50, 3)

    # Check loadings for each view
    assert "genetic" in mofa.W
    assert "metabolomic" in mofa.W
    assert "clinical" in mofa.W

    assert mofa.W["genetic"].shape == (20, 3)
    assert mofa.W["metabolomic"].shape == (15, 3)
    assert mofa.W["clinical"].shape == (10, 3)


def test_mofa_with_missing_data(multi_omics_data_with_missing):
    """Test MOFA with missing data"""
    data_dict, _ = multi_omics_data_with_missing

    mofa = MOFAIntegration(n_factors=3, n_iterations=50)
    mofa.fit(data_dict)

    # Should handle missing data
    assert mofa.Z is not None
    assert not np.isnan(mofa.Z).any()


def test_mofa_with_partial_overlap(partially_overlapping_samples):
    """Test MOFA with partially overlapping samples"""
    data_dict, _ = partially_overlapping_samples

    mofa = MOFAIntegration(n_factors=3, n_iterations=50)
    mofa.fit(data_dict)

    # Should handle all 60 samples
    assert mofa.Z.shape[0] == 60
    assert len(mofa.sample_ids) == 60


def test_mofa_transform(multi_omics_data):
    """Test MOFA transform on new data"""
    data_dict, _ = multi_omics_data

    mofa = MOFAIntegration(n_factors=3, n_iterations=50)
    mofa.fit(data_dict)

    # Transform same data (should work)
    factors = mofa.transform(data_dict)

    assert factors.shape == (50, 3)
    assert factors.index.equals(data_dict["genetic"].index)


def test_mofa_get_latent_factors(multi_omics_data):
    """Test getting latent factors"""
    data_dict, _ = multi_omics_data

    mofa = MOFAIntegration(n_factors=3, n_iterations=50)
    mofa.fit(data_dict)

    factors = mofa.get_latent_factors()

    assert isinstance(factors, pd.DataFrame)
    assert factors.shape == (50, 3)
    assert list(factors.columns) == ["Factor1", "Factor2", "Factor3"]


def test_mofa_get_factor_loadings(multi_omics_data):
    """Test getting factor loadings"""
    data_dict, _ = multi_omics_data

    mofa = MOFAIntegration(n_factors=3, n_iterations=50)
    mofa.fit(data_dict)

    # Get all loadings
    loadings = mofa.get_factor_loadings()

    assert "genetic" in loadings
    assert "metabolomic" in loadings
    assert "clinical" in loadings

    # Check genetic loadings
    genetic_loadings = loadings["genetic"]
    assert genetic_loadings.shape == (20, 3)
    assert list(genetic_loadings.columns) == ["Factor1", "Factor2", "Factor3"]

    # Get loadings for specific view
    genetic_only = mofa.get_factor_loadings(view="genetic")
    assert "genetic" in genetic_only
    assert len(genetic_only) == 1


def test_mofa_get_factor_loadings_with_threshold(multi_omics_data):
    """Test getting factor loadings with threshold"""
    data_dict, _ = multi_omics_data

    mofa = MOFAIntegration(n_factors=3, n_iterations=50)
    mofa.fit(data_dict)

    # Get loadings with threshold
    loadings = mofa.get_factor_loadings(threshold=0.1)

    # Should have filtered some loadings
    for view, view_loadings in loadings.items():
        assert view_loadings.shape[0] <= data_dict[view].shape[1]


def test_mofa_variance_explained(multi_omics_data):
    """Test variance explained calculation"""
    data_dict, _ = multi_omics_data

    mofa = MOFAIntegration(n_factors=3, n_iterations=50)
    mofa.fit(data_dict)

    # Check variance explained is calculated
    assert len(mofa.variance_explained) > 0
    assert "genetic" in mofa.variance_explained

    # Check structure
    genetic_var = mofa.variance_explained["genetic"]
    assert "Factor1" in genetic_var
    assert "Factor2" in genetic_var
    assert "Factor3" in genetic_var

    # Variance should be between 0 and 1
    for factor, var in genetic_var.items():
        assert 0 <= var <= 1


def test_mofa_with_view_weights():
    """Test MOFA with view weights"""
    np.random.seed(42)

    # Create simple data
    data_dict = {
        "view1": pd.DataFrame(np.random.randn(30, 10)),
        "view2": pd.DataFrame(np.random.randn(30, 8)),
    }

    # Fit with equal weights
    mofa1 = MOFAIntegration(n_factors=2, n_iterations=50)
    mofa1.fit(data_dict)

    # Fit with weighted views
    mofa2 = MOFAIntegration(
        n_factors=2,
        n_iterations=50,
        view_weights={"view1": 2.0, "view2": 0.5},
    )
    mofa2.fit(data_dict)

    # Results should be different
    assert not np.allclose(mofa1.Z, mofa2.Z)


def test_mofa_convergence():
    """Test MOFA convergence monitoring"""
    data_dict = {
        "view1": pd.DataFrame(np.random.randn(30, 10)),
    }

    mofa = MOFAIntegration(n_factors=2, n_iterations=1000, convergence_threshold=1e-4)
    mofa.fit(data_dict)

    # Should have ELBO history
    assert len(mofa.elbo_history) > 0

    # ELBO should generally increase (with noise)
    # Check that later ELBOs are not much worse than earlier ones
    if len(mofa.elbo_history) > 10:
        early_elbo = np.mean(mofa.elbo_history[:5])
        late_elbo = np.mean(mofa.elbo_history[-5:])
        # Late should be better (more positive) or similar
        assert late_elbo >= early_elbo - 100  # Allow some numerical noise


def test_integrate_stack():
    """Test simple concatenation integration"""
    data_dict = {
        "view1": pd.DataFrame(np.random.randn(30, 10)),
        "view2": pd.DataFrame(np.random.randn(30, 8)),
    }

    # Create mock config
    cfg = type("Config", (), {
        "integrate": type("Integrate", (), {
            "method": "stack",
            "weights": {"view1": 1.0, "view2": 0.5},
        })()
    })()

    result = _integrate_stack(data_dict, cfg)

    assert "concatenated" in result
    assert result["concatenated"].shape == (30, 18)  # 10 + 8 columns


def test_save_load_integration_results(multi_omics_data):
    """Test saving and loading integration results"""
    with TemporaryDirectory() as tmpdir:
        data_dict, _ = multi_omics_data

        # Fit MOFA
        mofa = MOFAIntegration(n_factors=3, n_iterations=50)
        mofa.fit(data_dict)

        results = {
            "factors": mofa.get_latent_factors(),
            "loadings": mofa.get_factor_loadings(),
            "variance_explained": mofa.variance_explained,
        }

        # Save
        output_dir = Path(tmpdir)
        save_integration_results(results, output_dir, method="mofa")

        # Check files exist
        assert (output_dir / "latent_factors.csv").exists()
        assert (output_dir / "loadings_genetic.csv").exists()
        assert (output_dir / "variance_explained.csv").exists()

        # Load
        loaded = load_integration_results(output_dir, method="mofa")

        # Check loaded data
        assert "factors" in loaded
        assert "variance_explained" in loaded

        pd.testing.assert_frame_equal(
            results["factors"], loaded["factors"], check_dtype=False
        )