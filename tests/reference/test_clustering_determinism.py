#!/usr/bin/env python3
"""
Deterministic Reference Tests for Clustering

Tests that clustering produces fixed labels on toy datasets across runs
and minor library version shifts (with tolerance).

Purpose:
- Catch regressions in clustering algorithms
- Verify reproducibility with random_state
- Tolerance-based comparison for numerical stability

Reference datasets saved in tests/reference/fixtures/
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from audhd_correlation.modeling.clustering import (
    ConsensusClusteringPipeline,
    HDBSCANParameterSweep,
    SpectralCoAssignmentClustering,
    BayesianGaussianMixtureClustering,
)
from sklearn.metrics import adjusted_rand_score


# Reference data directory
REFERENCE_DIR = Path(__file__).parent / "fixtures"
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def toy_dataset_3clusters():
    """
    Generate deterministic toy dataset with 3 well-separated clusters

    Used as reference for determinism tests.
    """
    np.random.seed(42)

    # 3 clusters, 30 samples each
    cluster1 = np.random.randn(30, 5) + np.array([0, 0, 0, 0, 0])
    cluster2 = np.random.randn(30, 5) + np.array([10, 0, 0, 0, 0])
    cluster3 = np.random.randn(30, 5) + np.array([0, 10, 0, 0, 0])

    X = np.vstack([cluster1, cluster2, cluster3])
    y_true = np.array([0] * 30 + [1] * 30 + [2] * 30)

    return X, y_true


@pytest.fixture
def toy_dataset_imbalanced():
    """
    Generate toy dataset with imbalanced clusters

    Cluster sizes: 50, 20, 10
    """
    np.random.seed(42)

    cluster1 = np.random.randn(50, 5) + np.array([0, 0, 0, 0, 0])
    cluster2 = np.random.randn(20, 5) + np.array([10, 0, 0, 0, 0])
    cluster3 = np.random.randn(10, 5) + np.array([0, 10, 0, 0, 0])

    X = np.vstack([cluster1, cluster2, cluster3])
    y_true = np.array([0] * 50 + [1] * 20 + [2] * 10)

    return X, y_true


@pytest.fixture
def toy_dataset_with_noise():
    """
    Generate toy dataset with noise points

    3 clusters + 10 noise points
    """
    np.random.seed(42)

    cluster1 = np.random.randn(30, 5) + np.array([0, 0, 0, 0, 0])
    cluster2 = np.random.randn(30, 5) + np.array([10, 0, 0, 0, 0])
    cluster3 = np.random.randn(30, 5) + np.array([0, 10, 0, 0, 0])
    noise = np.random.randn(10, 5) * 10  # Outliers

    X = np.vstack([cluster1, cluster2, cluster3, noise])
    y_true = np.array([0] * 30 + [1] * 30 + [2] * 30 + [-1] * 10)

    return X, y_true


def save_reference_labels(
    name: str,
    labels: np.ndarray,
    metadata: dict
) -> None:
    """
    Save reference labels to disk

    Args:
        name: Reference name
        labels: Cluster labels
        metadata: Additional metadata (method, params, etc.)
    """
    output_file = REFERENCE_DIR / f"{name}_labels.npz"

    np.savez(
        output_file,
        labels=labels,
        metadata=np.array([metadata], dtype=object)
    )

    print(f"Reference saved: {output_file}")


def load_reference_labels(name: str) -> tuple:
    """
    Load reference labels from disk

    Args:
        name: Reference name

    Returns:
        Tuple of (labels, metadata)
    """
    input_file = REFERENCE_DIR / f"{name}_labels.npz"

    if not input_file.exists():
        return None, None

    data = np.load(input_file, allow_pickle=True)
    labels = data["labels"]
    metadata = data["metadata"][0]

    return labels, metadata


def compare_labels_with_tolerance(
    labels1: np.ndarray,
    labels2: np.ndarray,
    tolerance: float = 0.95
) -> bool:
    """
    Compare labels with tolerance (using ARI)

    Args:
        labels1: First set of labels
        labels2: Second set of labels
        tolerance: Minimum ARI to consider equivalent (default: 0.95)

    Returns:
        True if ARI >= tolerance
    """
    # Handle noise labels
    # ARI handles different label IDs, so just compare structure
    ari = adjusted_rand_score(labels1, labels2)

    return ari >= tolerance


class TestConsensusPipelineDeterminism:
    """Test that ConsensusClusteringPipeline is deterministic"""

    def test_identical_runs_same_seed(self, toy_dataset_3clusters):
        """Test that identical runs with same seed produce identical labels"""
        X, _ = toy_dataset_3clusters

        # Run 1
        np.random.seed(42)
        pipeline1 = ConsensusClusteringPipeline(
            use_hdbscan=True,
            use_spectral=True,
            use_bgmm=True,
            n_bootstrap=50,
            random_state=42,
        )
        pipeline1.fit(X, generate_embeddings=False)
        labels1 = pipeline1.consensus_labels_

        # Run 2 (same seed)
        np.random.seed(42)
        pipeline2 = ConsensusClusteringPipeline(
            use_hdbscan=True,
            use_spectral=True,
            use_bgmm=True,
            n_bootstrap=50,
            random_state=42,
        )
        pipeline2.fit(X, generate_embeddings=False)
        labels2 = pipeline2.consensus_labels_

        # Should be identical
        np.testing.assert_array_equal(labels1, labels2)

    def test_different_seeds_different_results(self, toy_dataset_3clusters):
        """Test that different seeds produce different results (but similar ARI)"""
        X, _ = toy_dataset_3clusters

        # Run 1 (seed=42)
        pipeline1 = ConsensusClusteringPipeline(
            use_hdbscan=True,
            use_spectral=True,
            n_bootstrap=50,
            random_state=42,
        )
        pipeline1.fit(X, generate_embeddings=False)
        labels1 = pipeline1.consensus_labels_

        # Run 2 (seed=99)
        pipeline2 = ConsensusClusteringPipeline(
            use_hdbscan=True,
            use_spectral=True,
            n_bootstrap=50,
            random_state=99,
        )
        pipeline2.fit(X, generate_embeddings=False)
        labels2 = pipeline2.consensus_labels_

        # Labels may differ, but ARI should be high (both find 3 clusters)
        ari = adjusted_rand_score(labels1, labels2)
        assert ari > 0.8, f"ARI too low: {ari:.3f}"

    @pytest.mark.reference
    def test_reference_3clusters_consensus(self, toy_dataset_3clusters):
        """
        Reference test: 3-cluster dataset with consensus clustering

        This test creates/updates reference labels. Run with:
        pytest -m reference --update-references
        """
        X, y_true = toy_dataset_3clusters

        # Run clustering
        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=True,
            use_spectral=True,
            use_bgmm=True,
            n_bootstrap=50,
            random_state=42,
        )
        pipeline.fit(X, generate_embeddings=False)
        labels = pipeline.consensus_labels_

        # Load reference
        ref_labels, ref_metadata = load_reference_labels("consensus_3clusters")

        if ref_labels is None or pytest.config.getoption("--update-references", default=False):
            # Create/update reference
            metadata = {
                "method": "consensus",
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_clusters": len(set(labels)),
                "random_state": 42,
            }
            save_reference_labels("consensus_3clusters", labels, metadata)
            pytest.skip("Reference labels created/updated")

        # Compare with reference (tolerance-based)
        assert compare_labels_with_tolerance(labels, ref_labels, tolerance=0.95), (
            "Clustering results differ from reference"
        )

    @pytest.mark.reference
    def test_reference_imbalanced_consensus(self, toy_dataset_imbalanced):
        """Reference test: Imbalanced clusters"""
        X, y_true = toy_dataset_imbalanced

        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=True,
            use_spectral=True,
            n_bootstrap=50,
            random_state=42,
        )
        pipeline.fit(X, generate_embeddings=False)
        labels = pipeline.consensus_labels_

        ref_labels, _ = load_reference_labels("consensus_imbalanced")

        if ref_labels is None or pytest.config.getoption("--update-references", default=False):
            metadata = {
                "method": "consensus",
                "cluster_sizes": [50, 20, 10],
                "random_state": 42,
            }
            save_reference_labels("consensus_imbalanced", labels, metadata)
            pytest.skip("Reference labels created/updated")

        assert compare_labels_with_tolerance(labels, ref_labels, tolerance=0.90)

    @pytest.mark.reference
    def test_reference_with_noise_consensus(self, toy_dataset_with_noise):
        """Reference test: Dataset with noise points"""
        X, y_true = toy_dataset_with_noise

        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=True,
            use_spectral=True,
            n_bootstrap=50,
            random_state=42,
        )
        pipeline.fit(X, generate_embeddings=False)
        labels = pipeline.consensus_labels_

        ref_labels, _ = load_reference_labels("consensus_with_noise")

        if ref_labels is None or pytest.config.getoption("--update-references", default=False):
            metadata = {
                "method": "consensus",
                "n_noise": (labels == -1).sum(),
                "random_state": 42,
            }
            save_reference_labels("consensus_with_noise", labels, metadata)
            pytest.skip("Reference labels created/updated")

        assert compare_labels_with_tolerance(labels, ref_labels, tolerance=0.90)


class TestHDBSCANDeterminism:
    """Test HDBSCAN determinism"""

    def test_hdbscan_reproducibility(self, toy_dataset_3clusters):
        """Test HDBSCAN reproduces same results with same seed"""
        X, _ = toy_dataset_3clusters

        try:
            import hdbscan

            # Run 1
            clusterer1 = hdbscan.HDBSCAN(
                min_cluster_size=10,
                min_samples=5,
                random_state=42,
            )
            labels1 = clusterer1.fit_predict(X)

            # Run 2 (same seed)
            clusterer2 = hdbscan.HDBSCAN(
                min_cluster_size=10,
                min_samples=5,
                random_state=42,
            )
            labels2 = clusterer2.fit_predict(X)

            # Should be identical
            np.testing.assert_array_equal(labels1, labels2)

        except ImportError:
            pytest.skip("hdbscan not available")

    @pytest.mark.reference
    def test_reference_hdbscan_sweep(self, toy_dataset_3clusters):
        """Reference test: HDBSCAN parameter sweep"""
        X, _ = toy_dataset_3clusters

        try:
            sweep = HDBSCANParameterSweep(
                min_cluster_sizes=[5, 10],
                min_samples_list=[1, 5],
            )
            sweep.fit(X, scoring="silhouette")
            labels = sweep.predict()

            ref_labels, _ = load_reference_labels("hdbscan_sweep_3clusters")

            if ref_labels is None or pytest.config.getoption("--update-references", default=False):
                metadata = {
                    "method": "hdbscan_sweep",
                    "best_params": sweep.best_params_,
                }
                save_reference_labels("hdbscan_sweep_3clusters", labels, metadata)
                pytest.skip("Reference labels created/updated")

            assert compare_labels_with_tolerance(labels, ref_labels, tolerance=0.95)

        except ImportError:
            pytest.skip("hdbscan not available")


class TestSpectralDeterminism:
    """Test spectral clustering determinism"""

    def test_spectral_reproducibility(self, toy_dataset_3clusters):
        """Test spectral clustering reproduces same results"""
        X, _ = toy_dataset_3clusters

        from sklearn.cluster import SpectralClustering

        # Run 1
        sc1 = SpectralClustering(
            n_clusters=3,
            affinity="nearest_neighbors",
            random_state=42,
        )
        labels1 = sc1.fit_predict(X)

        # Run 2
        sc2 = SpectralClustering(
            n_clusters=3,
            affinity="nearest_neighbors",
            random_state=42,
        )
        labels2 = sc2.fit_predict(X)

        # Should be identical
        np.testing.assert_array_equal(labels1, labels2)

    @pytest.mark.reference
    def test_reference_spectral_coassignment(self, toy_dataset_3clusters):
        """Reference test: Spectral co-assignment clustering"""
        X, _ = toy_dataset_3clusters

        clusterer = SpectralCoAssignmentClustering(
            n_clusters=3,
            n_resamples=50,
            threshold=0.5,
        )

        np.random.seed(42)
        clusterer.fit(X)
        labels = clusterer.predict()

        ref_labels, _ = load_reference_labels("spectral_coassignment_3clusters")

        if ref_labels is None or pytest.config.getoption("--update-references", default=False):
            metadata = {
                "method": "spectral_coassignment",
                "n_resamples": 50,
            }
            save_reference_labels("spectral_coassignment_3clusters", labels, metadata)
            pytest.skip("Reference labels created/updated")

        assert compare_labels_with_tolerance(labels, ref_labels, tolerance=0.90)


class TestBGMMDeterminism:
    """Test Bayesian GMM determinism"""

    def test_bgmm_reproducibility(self, toy_dataset_3clusters):
        """Test BGMM reproduces same results with same seed"""
        X, _ = toy_dataset_3clusters

        # Run 1
        bgmm1 = BayesianGaussianMixtureClustering(
            max_components=10,
            n_init=5,
        )
        bgmm1.fit(X)
        labels1 = bgmm1.predict(X)

        # Run 2 (same seed in model)
        bgmm2 = BayesianGaussianMixtureClustering(
            max_components=10,
            n_init=5,
        )
        bgmm2.fit(X)
        labels2 = bgmm2.predict(X)

        # Should be very similar (ARI > 0.95)
        ari = adjusted_rand_score(labels1, labels2)
        assert ari > 0.95, f"BGMM not reproducible: ARI={ari:.3f}"


def pytest_addoption(parser):
    """Add --update-references option to pytest"""
    parser.addoption(
        "--update-references",
        action="store_true",
        default=False,
        help="Update reference labels",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "reference", "--update-references"])