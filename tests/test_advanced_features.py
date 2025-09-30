"""Tests for advanced batch correction and integration features"""
import pytest
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path

from src.audhd_correlation.preprocess.mixed_effects import (
    correct_mixed_effects,
    correct_partial_timeseries,
    MixedEffectsBatchCorrection,
)
from src.audhd_correlation.preprocess.qc_reports import BatchCorrectionQCReport
from src.audhd_correlation.integrate.adversarial import (
    DomainAdversarialIntegration,
    apply_adversarial_correction,
    evaluate_site_confounding,
)
from src.audhd_correlation.integrate.group_specific import GroupSpecificMOFA


@pytest.fixture
def site_platform_data():
    """Create data with site and platform effects"""
    np.random.seed(42)

    n_samples = 60
    n_features = 15

    # Create combinations of site and platform
    sites = np.repeat(["site1", "site2", "site3"], 20)
    platforms = np.tile(["platformA", "platformB"], 30)

    # Base data + site effects + platform effects + interaction
    data = np.random.randn(n_samples, n_features)

    for i, (site, platform) in enumerate(zip(sites, platforms)):
        # Site effect
        site_effect = {"site1": 0, "site2": 1, "site3": 2}[site]
        data[i] += site_effect

        # Platform effect
        platform_effect = {"platformA": 0, "platformB": 0.5}[platform]
        data[i] += platform_effect

        # Interaction
        if site == "site2" and platform == "platformB":
            data[i] += 0.3

    data_df = pd.DataFrame(
        data, index=[f"sample_{i}" for i in range(n_samples)]
    )

    site_series = pd.Series(sites, index=data_df.index)
    platform_series = pd.Series(platforms, index=data_df.index)

    return data_df, site_series, platform_series


@pytest.fixture
def timeseries_data():
    """Create time-series data with batch effects"""
    np.random.seed(42)

    n_samples = 50
    n_features = 10

    timepoints = np.linspace(0, 10, n_samples)
    sites = np.repeat(["site1", "site2"], 25)
    batches = np.repeat(["batch1", "batch2"], 25)

    # Create data with temporal trend + batch effects
    data = np.zeros((n_samples, n_features))

    for j in range(n_features):
        # Temporal trend (quadratic)
        trend = 0.5 * timepoints + 0.1 * timepoints ** 2

        # Add noise
        trend += np.random.randn(n_samples) * 0.2

        # Add batch effect
        for i, batch in enumerate(batches):
            if batch == "batch2":
                trend[i] += 1.5

        data[:, j] = trend

    data_df = pd.DataFrame(data, index=[f"sample_{i}" for i in range(n_samples)])

    return data_df, pd.Series(timepoints, index=data_df.index), pd.Series(sites, index=data_df.index), pd.Series(batches, index=data_df.index)


def test_mixed_effects_correction(site_platform_data):
    """Test mixed effects batch correction"""
    data, site, platform = site_platform_data

    corrected = correct_mixed_effects(data, site, platform)

    assert corrected.shape == data.shape
    assert corrected.index.equals(data.index)

    # Check that site effects are reduced
    site1_mean = data.loc[site == "site1"].mean().mean()
    site2_mean = data.loc[site == "site2"].mean().mean()
    original_diff = abs(site1_mean - site2_mean)

    site1_corrected = corrected.loc[site == "site1"].mean().mean()
    site2_corrected = corrected.loc[site == "site2"].mean().mean()
    corrected_diff = abs(site1_corrected - site2_corrected)

    assert corrected_diff < original_diff


def test_mixed_effects_with_covariates(site_platform_data):
    """Test mixed effects with covariate preservation"""
    data, site, platform = site_platform_data

    # Add covariates
    covariates = pd.DataFrame({
        "age": np.random.randint(5, 18, len(data)),
        "sex": np.random.choice([0, 1], len(data)),
    }, index=data.index)

    corrected = correct_mixed_effects(data, site, platform, covariates=covariates)

    assert corrected.shape == data.shape


def test_mixed_effects_partial_correction(site_platform_data):
    """Test partial batch correction"""
    data, site, platform = site_platform_data

    corrected_full = correct_mixed_effects(data, site, platform, partial_correction=False)
    corrected_partial = correct_mixed_effects(data, site, platform, partial_correction=True)

    # Just check that both corrections work without errors
    # The partial correction parameter adjusts how much variance is removed
    assert corrected_full.shape == data.shape
    assert corrected_partial.shape == data.shape
    assert not corrected_full.isna().all().all()
    assert not corrected_partial.isna().all().all()


def test_mixed_effects_batch_correction_class(site_platform_data):
    """Test MixedEffectsBatchCorrection class"""
    data, site, platform = site_platform_data

    corrector = MixedEffectsBatchCorrection(partial_correction=True)
    corrected = corrector.correct(data, site, platform)

    assert corrected.shape == data.shape


def test_partial_timeseries_correction(timeseries_data):
    """Test partial batch correction for time-series"""
    data, timepoints, sites, batches = timeseries_data

    corrected = correct_partial_timeseries(
        data, timepoints, sites, batches, preserve_trend=True
    )

    assert corrected.shape == data.shape

    # Check that temporal trend is preserved
    # Original trend should correlate with corrected values
    for j in range(data.shape[1]):
        corr = np.corrcoef(timepoints, corrected.iloc[:, j])[0, 1]
        assert abs(corr) > 0.5  # Strong temporal correlation preserved


def test_timeseries_without_trend_preservation(timeseries_data):
    """Test time-series correction without trend preservation"""
    data, timepoints, sites, batches = timeseries_data

    corrected = correct_partial_timeseries(
        data, timepoints, sites, batches, preserve_trend=False
    )

    assert corrected.shape == data.shape


def test_qc_report_generation():
    """Test QC report generation"""
    with TemporaryDirectory() as tmpdir:
        # Create test data
        np.random.seed(42)
        data_before = pd.DataFrame(np.random.randn(50, 10) + np.repeat([0, 2], 25)[:, None])
        data_after = pd.DataFrame(np.random.randn(50, 10))
        batch = pd.Series(["batch1"] * 25 + ["batch2"] * 25)

        data_before.index = [f"sample_{i}" for i in range(50)]
        data_after.index = [f"sample_{i}" for i in range(50)]
        batch.index = data_before.index

        # Generate report
        qc_report = BatchCorrectionQCReport(Path(tmpdir))
        report_path = qc_report.generate_report(
            data_before, data_after, batch, modality="test"
        )

        # Check that report was generated
        assert report_path.exists()
        assert report_path.suffix == ".html"

        # Check that plots were generated
        assert len(qc_report.plots) > 0

        # Check metrics
        assert "test" in qc_report.metrics
        metrics = qc_report.metrics["test"]
        assert "batch_effect_before" in metrics
        assert "batch_effect_after" in metrics
        assert "batch_effect_reduction" in metrics


def test_qc_report_with_covariates():
    """Test QC report with covariate preservation plots"""
    with TemporaryDirectory() as tmpdir:
        np.random.seed(42)
        data_before = pd.DataFrame(np.random.randn(50, 10))
        data_after = pd.DataFrame(np.random.randn(50, 10))
        batch = pd.Series(["batch1"] * 25 + ["batch2"] * 25)
        covariates = pd.DataFrame({
            "age": np.random.randint(5, 18, 50),
            "sex": np.random.choice([0, 1], 50),
        })

        data_before.index = [f"sample_{i}" for i in range(50)]
        data_after.index = [f"sample_{i}" for i in range(50)]
        batch.index = data_before.index
        covariates.index = data_before.index

        qc_report = BatchCorrectionQCReport(Path(tmpdir))
        report_path = qc_report.generate_report(
            data_before, data_after, batch, modality="test", covariates=covariates
        )

        assert report_path.exists()


def test_domain_adversarial_integration():
    """Test domain adversarial integration"""
    np.random.seed(42)

    # Create data with site effects
    n_samples = 60
    data_dict = {
        "view1": pd.DataFrame(
            np.random.randn(n_samples, 10) + np.repeat([0, 1, 2], 20)[:, None],
            index=[f"sample_{i}" for i in range(n_samples)],
        ),
        "view2": pd.DataFrame(
            np.random.randn(n_samples, 8) + np.repeat([0, 1, 2], 20)[:, None],
            index=[f"sample_{i}" for i in range(n_samples)],
        ),
    }

    site_labels = pd.Series(
        np.repeat(["site1", "site2", "site3"], 20),
        index=data_dict["view1"].index,
    )

    # Fit adversarial model
    adv_model = DomainAdversarialIntegration(
        n_factors=3, adversarial_weight=0.1, n_iterations=50
    )

    factors = adv_model.fit_transform(data_dict, site_labels)

    assert factors.shape == (n_samples, 3)
    assert list(factors.columns) == ["Factor1", "Factor2", "Factor3"]


def test_apply_adversarial_correction():
    """Test post-hoc adversarial correction"""
    np.random.seed(42)

    # Create latent factors with site effects
    n_samples = 60
    factors = pd.DataFrame(
        np.random.randn(n_samples, 5) + np.repeat([0, 1, 2], 20)[:, None],
        index=[f"sample_{i}" for i in range(n_samples)],
    )

    site_labels = pd.Series(
        np.repeat(["site1", "site2", "site3"], 20),
        index=factors.index,
    )

    # Apply correction
    corrected = apply_adversarial_correction(
        factors, site_labels, correction_strength=0.5
    )

    assert corrected.shape == factors.shape

    # Site differences should be reduced
    site1_mean = factors.loc[site_labels == "site1"].mean().mean()
    site2_mean = factors.loc[site_labels == "site2"].mean().mean()
    original_diff = abs(site1_mean - site2_mean)

    site1_corrected = corrected.loc[site_labels == "site1"].mean().mean()
    site2_corrected = corrected.loc[site_labels == "site2"].mean().mean()
    corrected_diff = abs(site1_corrected - site2_corrected)

    assert corrected_diff < original_diff


def test_evaluate_site_confounding():
    """Test site confounding evaluation"""
    np.random.seed(42)

    # Create factors with moderate site signal
    n_samples = 90
    factors = pd.DataFrame(
        np.random.randn(n_samples, 5) + np.repeat([0, 0.5, 1], 30)[:, None],
        index=[f"sample_{i}" for i in range(n_samples)],
    )

    site_labels = pd.Series(
        np.repeat(["site1", "site2", "site3"], 30),
        index=factors.index,
    )

    # Evaluate
    metrics = evaluate_site_confounding(factors, site_labels)

    assert "site_classification_accuracy" in metrics
    assert "random_baseline" in metrics
    assert "excess_site_information" in metrics

    # Should be able to classify sites better than random
    assert metrics["site_classification_accuracy"] > metrics["random_baseline"]


def test_group_specific_mofa():
    """Test group-specific MOFA"""
    np.random.seed(42)

    # Create data with group-specific patterns
    n_samples = 60
    n_factors = 3

    # Shared factors
    Z_shared = np.random.randn(n_samples, n_factors)

    data_dict = {
        "view1": pd.DataFrame(
            Z_shared @ np.random.randn(n_factors, 10) + np.random.randn(n_samples, 10) * 0.1,
            index=[f"sample_{i}" for i in range(n_samples)],
        ),
        "view2": pd.DataFrame(
            Z_shared @ np.random.randn(n_factors, 8) + np.random.randn(n_samples, 8) * 0.1,
            index=[f"sample_{i}" for i in range(n_samples)],
        ),
    }

    # Add group-specific patterns
    group_labels = pd.Series(
        np.repeat(["group1", "group2"], 30),
        index=data_dict["view1"].index,
    )

    # Fit group-specific MOFA
    gs_mofa = GroupSpecificMOFA(
        n_shared_factors=2, n_group_factors=2, n_iterations=50
    )
    gs_mofa.fit(data_dict, group_labels)

    # Get all factors
    all_factors = gs_mofa.get_all_factors(data_dict, group_labels)

    # Should have shared + group-specific factors
    assert "Factor1" in all_factors.columns  # Shared
    assert "Group_group1_Factor1" in all_factors.columns  # Group-specific

    # Group-specific factors should be mostly zero for other groups
    group1_mask = group_labels == "group1"
    group1_factor_in_group2 = all_factors.loc[~group1_mask, "Group_group1_Factor1"]
    # Allow some numerical tolerance
    assert (np.abs(group1_factor_in_group2) < 0.1).sum() >= len(group1_factor_in_group2) * 0.9


def test_group_specific_mofa_variance():
    """Test variance explained in group-specific MOFA"""
    np.random.seed(42)

    # Simple data
    data_dict = {
        "view1": pd.DataFrame(np.random.randn(40, 10)),
    }

    group_labels = pd.Series(np.repeat(["group1", "group2"], 20))
    group_labels.index = data_dict["view1"].index

    gs_mofa = GroupSpecificMOFA(
        n_shared_factors=2, n_group_factors=1, n_iterations=30
    )
    gs_mofa.fit(data_dict, group_labels)

    # Should have fitted shared and group models
    assert gs_mofa.shared_model is not None
    assert "group1" in gs_mofa.group_models
    assert "group2" in gs_mofa.group_models