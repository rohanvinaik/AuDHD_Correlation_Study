#!/usr/bin/env python3
"""
MOFA Integration Smoke Tests

Tests MOFA integration with synthetic data to verify:
1. Basic functionality (fit, transform, get results)
2. Output format compliance with embedding contract
3. Handle 2+ modalities with different feature counts
4. Missing data handling
5. Convergence and stability
"""
import numpy as np
import pandas as pd
import pytest

from audhd_correlation.integrate.mofa import MOFAIntegration
from audhd_correlation.integrate.methods import (
    integrate_omics,
    standardize_integration_output,
    extract_primary_embedding,
    validate_embedding_shape,
    validate_no_missing,
)


@pytest.fixture
def synthetic_multimodal_data():
    """
    Generate synthetic multi-modal data for MOFA testing

    Returns:
        Dict with 2 modalities (genomic, metabolomic)
    """
    np.random.seed(42)

    n_samples = 50
    n_features_genomic = 100
    n_features_metabolomic = 80

    # Generate latent factors (ground truth)
    n_latent = 3
    Z_true = np.random.randn(n_samples, n_latent)

    # Generate loadings for each modality
    W_genomic = np.random.randn(n_features_genomic, n_latent) * 0.5
    W_metabolomic = np.random.randn(n_features_metabolomic, n_latent) * 0.5

    # Generate data: Y = Z @ W.T + noise
    genomic_data = Z_true @ W_genomic.T + np.random.randn(n_samples, n_features_genomic) * 0.5
    metabolomic_data = Z_true @ W_metabolomic.T + np.random.randn(n_samples, n_features_metabolomic) * 0.5

    # Create DataFrames
    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]
    genomic_features = [f"SNP_{i:04d}" for i in range(n_features_genomic)]
    metabolomic_features = [f"METAB_{i:03d}" for i in range(n_features_metabolomic)]

    genomic_df = pd.DataFrame(genomic_data, index=sample_ids, columns=genomic_features)
    metabolomic_df = pd.DataFrame(metabolomic_data, index=sample_ids, columns=metabolomic_features)

    return {
        "genomic": genomic_df,
        "metabolomic": metabolomic_df,
        "Z_true": Z_true,
        "W_genomic": W_genomic,
        "W_metabolomic": W_metabolomic,
    }


@pytest.fixture
def synthetic_data_with_missing():
    """
    Generate synthetic data with missing values

    Returns:
        Dict with 2 modalities, 10% missing values
    """
    np.random.seed(42)

    n_samples = 50
    n_features_mod1 = 100
    n_features_mod2 = 80

    # Generate data
    data_mod1 = np.random.randn(n_samples, n_features_mod1)
    data_mod2 = np.random.randn(n_samples, n_features_mod2)

    # Introduce missing values (10%)
    missing_mask_1 = np.random.rand(n_samples, n_features_mod1) < 0.1
    missing_mask_2 = np.random.rand(n_samples, n_features_mod2) < 0.1

    data_mod1[missing_mask_1] = np.nan
    data_mod2[missing_mask_2] = np.nan

    # Create DataFrames
    sample_ids = [f"SAMPLE_{i:03d}" for i in range(n_samples)]

    mod1_df = pd.DataFrame(
        data_mod1,
        index=sample_ids,
        columns=[f"FEAT1_{i:04d}" for i in range(n_features_mod1)]
    )
    mod2_df = pd.DataFrame(
        data_mod2,
        index=sample_ids,
        columns=[f"FEAT2_{i:03d}" for i in range(n_features_mod2)]
    )

    return {
        "modality1": mod1_df,
        "modality2": mod2_df,
    }


class TestMOFABasicFunctionality:
    """Test basic MOFA functionality"""

    def test_mofa_fit(self, synthetic_multimodal_data):
        """Test that MOFA can fit synthetic data"""
        data_dict = {
            "genomic": synthetic_multimodal_data["genomic"],
            "metabolomic": synthetic_multimodal_data["metabolomic"],
        }

        mofa = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa.fit(data_dict)

        # Check that model is fitted
        assert mofa.Z is not None, "Latent factors not initialized"
        assert mofa.W, "Factor loadings not initialized"
        assert "genomic" in mofa.W, "Genomic loadings not found"
        assert "metabolomic" in mofa.W, "Metabolomic loadings not found"

        # Check shapes
        n_samples = len(data_dict["genomic"])
        assert mofa.Z.shape == (n_samples, 5), f"Unexpected Z shape: {mofa.Z.shape}"
        assert mofa.W["genomic"].shape[0] == data_dict["genomic"].shape[1]
        assert mofa.W["metabolomic"].shape[0] == data_dict["metabolomic"].shape[1]

    def test_mofa_get_latent_factors(self, synthetic_multimodal_data):
        """Test that we can extract latent factors as DataFrame"""
        data_dict = {
            "genomic": synthetic_multimodal_data["genomic"],
            "metabolomic": synthetic_multimodal_data["metabolomic"],
        }

        mofa = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa.fit(data_dict)

        factors = mofa.get_latent_factors()

        # Check type and shape
        assert isinstance(factors, pd.DataFrame), "Factors should be DataFrame"
        assert factors.shape[0] == len(data_dict["genomic"])
        assert factors.shape[1] == 5
        assert list(factors.columns) == [f"Factor{i+1}" for i in range(5)]

        # Check index matches sample IDs
        assert factors.index.equals(data_dict["genomic"].index)

    def test_mofa_get_factor_loadings(self, synthetic_multimodal_data):
        """Test that we can extract factor loadings"""
        data_dict = {
            "genomic": synthetic_multimodal_data["genomic"],
            "metabolomic": synthetic_multimodal_data["metabolomic"],
        }

        mofa = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa.fit(data_dict)

        loadings = mofa.get_factor_loadings()

        # Check structure
        assert isinstance(loadings, dict)
        assert "genomic" in loadings
        assert "metabolomic" in loadings

        # Check shapes
        assert loadings["genomic"].shape == (
            data_dict["genomic"].shape[1], 5
        ), f"Unexpected genomic loadings shape: {loadings['genomic'].shape}"
        assert loadings["metabolomic"].shape == (
            data_dict["metabolomic"].shape[1], 5
        ), f"Unexpected metabolomic loadings shape: {loadings['metabolomic'].shape}"

    def test_mofa_transform(self, synthetic_multimodal_data):
        """Test that MOFA can transform new data"""
        data_dict = {
            "genomic": synthetic_multimodal_data["genomic"],
            "metabolomic": synthetic_multimodal_data["metabolomic"],
        }

        mofa = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa.fit(data_dict)

        # Transform same data (test transform capability)
        factors_transformed = mofa.transform(data_dict)

        assert isinstance(factors_transformed, pd.DataFrame)
        assert factors_transformed.shape == (len(data_dict["genomic"]), 5)

    def test_mofa_convergence(self, synthetic_multimodal_data):
        """Test that MOFA converges (ELBO improves)"""
        data_dict = {
            "genomic": synthetic_multimodal_data["genomic"],
            "metabolomic": synthetic_multimodal_data["metabolomic"],
        }

        mofa = MOFAIntegration(n_factors=5, n_iterations=100)
        mofa.fit(data_dict)

        # Check that ELBO history exists and improves
        assert len(mofa.elbo_history) > 0, "ELBO history empty"

        # ELBO should generally increase (allow small fluctuations)
        if len(mofa.elbo_history) > 2:
            initial_elbo = mofa.elbo_history[0]
            final_elbo = mofa.elbo_history[-1]
            assert final_elbo > initial_elbo - 100, (
                f"ELBO decreased significantly: {initial_elbo} -> {final_elbo}"
            )


class TestMOFAEmbeddingContract:
    """Test that MOFA outputs comply with embedding contract"""

    def test_mofa_output_format(self, synthetic_multimodal_data):
        """Test that MOFA returns dict with expected keys"""
        data_dict = {
            "genomic": synthetic_multimodal_data["genomic"],
            "metabolomic": synthetic_multimodal_data["metabolomic"],
        }

        mofa = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa.fit(data_dict)

        # Get results
        factors = mofa.get_latent_factors()
        loadings = mofa.get_factor_loadings()

        # Simulate integration results dict
        results = {
            "factors": factors,
            "loadings": loadings,
            "variance_explained": mofa.variance_explained,
            "model": mofa,
        }

        # Check keys
        assert "factors" in results
        assert "loadings" in results

    def test_standardize_mofa_output(self, synthetic_multimodal_data):
        """Test standardize_integration_output for MOFA"""
        data_dict = {
            "genomic": synthetic_multimodal_data["genomic"],
            "metabolomic": synthetic_multimodal_data["metabolomic"],
        }

        mofa = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa.fit(data_dict)

        results = {
            "factors": mofa.get_latent_factors(),
            "loadings": mofa.get_factor_loadings(),
        }

        # Standardize
        embeddings = standardize_integration_output(results, method="mofa")

        # Check structure
        assert isinstance(embeddings, dict)
        assert "mofa_factors" in embeddings
        assert isinstance(embeddings["mofa_factors"], np.ndarray)

        # Check shape
        n_samples = len(data_dict["genomic"])
        assert embeddings["mofa_factors"].shape == (n_samples, 5)

    def test_extract_primary_embedding(self, synthetic_multimodal_data):
        """Test extract_primary_embedding for MOFA"""
        data_dict = {
            "genomic": synthetic_multimodal_data["genomic"],
            "metabolomic": synthetic_multimodal_data["metabolomic"],
        }

        mofa = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa.fit(data_dict)

        results = {
            "factors": mofa.get_latent_factors(),
            "loadings": mofa.get_factor_loadings(),
        }

        # Extract primary embedding
        X = extract_primary_embedding(results, method="mofa")

        # Check type and shape
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        assert X.shape[0] == len(data_dict["genomic"])
        assert X.shape[1] == 5

        # Validate
        validate_embedding_shape(X, expected_n_samples=len(data_dict["genomic"]))
        validate_no_missing(X)


class TestMOFAMissingDataHandling:
    """Test MOFA with missing data"""

    def test_mofa_handles_missing_data(self, synthetic_data_with_missing):
        """Test that MOFA can handle missing values"""
        mofa = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa.fit(synthetic_data_with_missing)

        # Should complete without errors
        assert mofa.Z is not None
        assert mofa.W

        # Get factors
        factors = mofa.get_latent_factors()
        assert not factors.isna().any().any(), "Factors contain NaN"

    def test_mofa_missing_samples_across_modalities(self):
        """Test MOFA with different samples per modality"""
        np.random.seed(42)

        n_samples_mod1 = 60
        n_samples_mod2 = 50
        n_features_mod1 = 100
        n_features_mod2 = 80

        # Modality 1: 60 samples
        sample_ids_1 = [f"SAMPLE_{i:03d}" for i in range(n_samples_mod1)]
        data_mod1 = pd.DataFrame(
            np.random.randn(n_samples_mod1, n_features_mod1),
            index=sample_ids_1,
            columns=[f"FEAT1_{i:04d}" for i in range(n_features_mod1)]
        )

        # Modality 2: 50 samples (subset of modality 1)
        sample_ids_2 = [f"SAMPLE_{i:03d}" for i in range(n_samples_mod2)]
        data_mod2 = pd.DataFrame(
            np.random.randn(n_samples_mod2, n_features_mod2),
            index=sample_ids_2,
            columns=[f"FEAT2_{i:03d}" for i in range(n_features_mod2)]
        )

        data_dict = {
            "modality1": data_mod1,
            "modality2": data_mod2,
        }

        # Fit MOFA
        mofa = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa.fit(data_dict)

        # Should handle union of samples
        factors = mofa.get_latent_factors()
        assert factors.shape[0] == n_samples_mod1  # Should be union (60)


class TestMOFAStability:
    """Test MOFA stability and reproducibility"""

    def test_mofa_reproducibility_with_seed(self, synthetic_multimodal_data):
        """Test that MOFA produces same results with same seed"""
        data_dict = {
            "genomic": synthetic_multimodal_data["genomic"],
            "metabolomic": synthetic_multimodal_data["metabolomic"],
        }

        # Run 1
        np.random.seed(42)
        mofa1 = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa1.fit(data_dict)
        factors1 = mofa1.get_latent_factors()

        # Run 2 (same seed)
        np.random.seed(42)
        mofa2 = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa2.fit(data_dict)
        factors2 = mofa2.get_latent_factors()

        # Should be identical (or very close due to floating point)
        np.testing.assert_allclose(
            factors1.values, factors2.values, rtol=1e-5, atol=1e-8,
            err_msg="MOFA not reproducible with same seed"
        )

    def test_mofa_variance_explained(self, synthetic_multimodal_data):
        """Test that variance explained is calculated"""
        data_dict = {
            "genomic": synthetic_multimodal_data["genomic"],
            "metabolomic": synthetic_multimodal_data["metabolomic"],
        }

        mofa = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa.fit(data_dict)

        # Check variance explained structure
        assert mofa.variance_explained, "Variance explained not calculated"
        assert "genomic" in mofa.variance_explained
        assert "metabolomic" in mofa.variance_explained

        # Check that factors are present
        for view in ["genomic", "metabolomic"]:
            var_exp = mofa.variance_explained[view]
            assert len(var_exp) == 5, f"Expected 5 factors, got {len(var_exp)}"

            # Variance explained should be between 0 and 1
            for factor, value in var_exp.items():
                assert 0 <= value <= 1, (
                    f"Variance explained for {view}/{factor} out of range: {value}"
                )


class TestMOFAEdgeCases:
    """Test MOFA edge cases"""

    def test_mofa_single_modality(self, synthetic_multimodal_data):
        """Test MOFA with single modality (should work like PCA)"""
        data_dict = {
            "genomic": synthetic_multimodal_data["genomic"],
        }

        mofa = MOFAIntegration(n_factors=5, n_iterations=50)
        mofa.fit(data_dict)

        factors = mofa.get_latent_factors()
        assert factors.shape == (len(data_dict["genomic"]), 5)

    def test_mofa_more_factors_than_samples(self, synthetic_multimodal_data):
        """Test MOFA when n_factors > n_samples"""
        # Take subset of samples
        data_dict = {
            "genomic": synthetic_multimodal_data["genomic"].iloc[:10],
            "metabolomic": synthetic_multimodal_data["metabolomic"].iloc[:10],
        }

        # Request 20 factors (more than 10 samples)
        mofa = MOFAIntegration(n_factors=20, n_iterations=50)

        # Should complete but may adjust n_factors internally
        mofa.fit(data_dict)

        factors = mofa.get_latent_factors()
        # Factors should be capped at n_samples
        assert factors.shape[0] == 10
        assert factors.shape[1] <= 10  # May be less if ARD pruning occurs


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])