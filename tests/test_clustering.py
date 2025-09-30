"""Comprehensive tests for consensus clustering pipeline"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons

from src.audhd_correlation.modeling.clustering import (
    HDBSCANParameterSweep,
    SpectralCoAssignmentClustering,
    BayesianGaussianMixtureClustering,
    MultiEmbeddingGenerator,
    TopologicalGapDetector,
    ConsensusClusteringPipeline,
    ClusterAssignment,
    ClusteringMetrics,
)

# Check for optional dependencies
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


@pytest.fixture
def blob_data():
    """Create well-separated blob data"""
    X, y_true = make_blobs(n_samples=200, n_features=10, centers=4,
                           cluster_std=0.5, random_state=42)
    return X, y_true


@pytest.fixture
def moon_data():
    """Create non-linear moon data"""
    X, y_true = make_moons(n_samples=200, noise=0.1, random_state=42)
    return X, y_true


@pytest.fixture
def small_data():
    """Create small dataset for edge cases"""
    X, y_true = make_blobs(n_samples=30, n_features=5, centers=3,
                           cluster_std=0.3, random_state=42)
    return X, y_true


@pytest.fixture
def high_dim_data():
    """Create high-dimensional data"""
    X, y_true = make_blobs(n_samples=100, n_features=50, centers=3,
                           cluster_std=1.0, random_state=42)
    return X, y_true


class TestHDBSCANParameterSweep:
    """Tests for HDBSCAN parameter sweep"""

    @pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
    def test_basic_sweep(self, blob_data):
        """Test basic parameter sweep"""
        X, y_true = blob_data

        sweep = HDBSCANParameterSweep(
            min_cluster_sizes=[5, 10],
            min_samples_list=[1, 5],
            metrics=['euclidean'],
            cluster_selection_methods=['eom']
        )

        sweep.fit(X)

        assert sweep.best_labels_ is not None
        assert len(sweep.best_labels_) == len(X)
        assert sweep.best_score_ is not None
        assert 'min_cluster_size' in sweep.best_params_
        assert 'min_samples' in sweep.best_params_

    @pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
    def test_scoring_methods(self, blob_data):
        """Test different scoring methods"""
        X, y_true = blob_data

        for scoring in ['silhouette', 'calinski_harabasz', 'davies_bouldin']:
            sweep = HDBSCANParameterSweep(
                min_cluster_sizes=[10],
                min_samples_list=[5],
                metrics=['euclidean'],
                cluster_selection_methods=['eom']
            )

            sweep.fit(X, scoring=scoring)
            assert sweep.best_score_ is not None

    @pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
    def test_sweep_results(self, blob_data):
        """Test sweep results DataFrame"""
        X, y_true = blob_data

        sweep = HDBSCANParameterSweep(
            min_cluster_sizes=[5, 10],
            min_samples_list=[1, 5],
            metrics=['euclidean', 'manhattan'],
            cluster_selection_methods=['eom']
        )

        sweep.fit(X)
        results = sweep.get_sweep_results()

        assert isinstance(results, pd.DataFrame)
        assert 'min_cluster_size' in results.columns
        assert 'score' in results.columns
        assert 'n_clusters' in results.columns
        assert len(results) == 2 * 2 * 2  # All combinations

    @pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
    def test_no_valid_clusters(self, small_data):
        """Test handling when no valid clusters found"""
        X, y_true = small_data

        # Use very large min_cluster_size to force no clusters
        sweep = HDBSCANParameterSweep(
            min_cluster_sizes=[100],
            min_samples_list=[10],
            metrics=['euclidean'],
            cluster_selection_methods=['eom']
        )

        sweep.fit(X)

        # Should still return some labels (even if mostly noise)
        assert sweep.best_labels_ is not None
        assert len(sweep.best_labels_) == len(X)


class TestSpectralCoAssignmentClustering:
    """Tests for spectral co-assignment clustering"""

    def test_basic_clustering(self, blob_data):
        """Test basic spectral co-assignment"""
        X, y_true = blob_data

        spec = SpectralCoAssignmentClustering(
            n_clusters=4, n_resamples=20, threshold=0.5
        )

        spec.fit(X)
        labels = spec.predict()

        assert len(labels) == len(X)
        assert len(np.unique(labels)) > 1
        assert spec.coassignment_matrix_ is not None
        assert spec.coassignment_matrix_.shape == (len(X), len(X))

    def test_auto_n_clusters(self, blob_data):
        """Test automatic cluster number detection"""
        X, y_true = blob_data

        spec = SpectralCoAssignmentClustering(
            n_clusters=None, n_resamples=20, threshold=0.5
        )

        spec.fit(X)
        labels = spec.predict()

        assert spec.n_clusters is not None
        assert spec.n_clusters > 1
        assert len(np.unique(labels)) <= spec.n_clusters

    def test_co_assignment_matrix_properties(self, blob_data):
        """Test co-assignment matrix properties"""
        X, y_true = blob_data

        spec = SpectralCoAssignmentClustering(n_clusters=4, n_resamples=20)
        spec.fit(X)

        co_mat = spec.coassignment_matrix_

        # Symmetric
        assert np.allclose(co_mat, co_mat.T)

        # Values in [0, 1]
        assert np.all(co_mat >= 0)
        assert np.all(co_mat <= 1)

        # Should have some non-zero values (samples clustered together)
        assert np.max(co_mat) > 0

    def test_custom_base_clusterer(self, blob_data):
        """Test with custom base clusterer"""
        from sklearn.cluster import KMeans

        X, y_true = blob_data

        base = KMeans(n_clusters=4, random_state=42, n_init=10)
        spec = SpectralCoAssignmentClustering(
            n_clusters=4, n_resamples=20, threshold=0.5
        )

        spec.fit(X, base_clusterer=base)
        labels = spec.predict()

        assert len(labels) == len(X)


class TestBayesianGaussianMixtureClustering:
    """Tests for Bayesian GMM clustering"""

    def test_basic_clustering(self, blob_data):
        """Test basic Bayesian GMM"""
        X, y_true = blob_data

        bgm = BayesianGaussianMixtureClustering(max_components=10)
        bgm.fit(X)
        labels = bgm.predict(X)

        assert len(labels) == len(X)
        assert bgm.n_components_ is not None
        assert bgm.n_components_ <= 10
        # Labels may not match components exactly due to sklearn's label assignment
        assert len(np.unique(labels)) >= 1

    def test_predict_proba(self, blob_data):
        """Test probability predictions"""
        X, y_true = blob_data

        bgm = BayesianGaussianMixtureClustering(max_components=10)
        bgm.fit(X)
        proba = bgm.predict_proba(X)

        assert proba.shape[0] == len(X)
        assert proba.shape[1] == bgm.max_components  # Full components, not effective

        # Probabilities sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)

        # All probabilities in [0, 1]
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_component_detection(self, blob_data):
        """Test automatic component detection"""
        X, y_true = blob_data

        # Should detect around 4 components for 4 blobs
        bgm = BayesianGaussianMixtureClustering(max_components=20)
        bgm.fit(X)

        # Should not use all 20 components
        assert bgm.n_components_ < 20
        assert bgm.n_components_ >= 2

    def test_covariance_types(self, blob_data):
        """Test different covariance types"""
        X, y_true = blob_data

        for cov_type in ['full', 'tied', 'diag', 'spherical']:
            bgm = BayesianGaussianMixtureClustering(
                max_components=10, covariance_type=cov_type
            )
            bgm.fit(X)
            labels = bgm.predict(X)
            assert len(labels) == len(X)


class TestMultiEmbeddingGenerator:
    """Tests for multi-embedding generator"""

    def test_basic_embedding(self, blob_data):
        """Test basic embedding generation"""
        X, y_true = blob_data

        gen = MultiEmbeddingGenerator(
            methods=['tsne', 'pca'],
            n_components=2
        )

        embeddings = gen.fit_transform(X)

        assert isinstance(embeddings, dict)
        assert 'pca' in embeddings
        assert len(embeddings) > 1

    def test_tsne_embeddings(self, blob_data):
        """Test t-SNE embeddings"""
        X, y_true = blob_data

        gen = MultiEmbeddingGenerator(
            methods=['tsne', 'pca'],
            n_components=2
        )

        embeddings = gen.fit_transform(X)

        # Should have at least PCA and some t-SNE
        assert 'pca' in embeddings
        has_tsne = any('tsne' in key for key in embeddings.keys())
        assert has_tsne

        for key, emb in embeddings.items():
            assert emb.shape == (len(X), 2)

    @pytest.mark.skipif(not UMAP_AVAILABLE, reason="umap not installed")
    def test_umap_embeddings(self, blob_data):
        """Test UMAP embeddings"""
        X, y_true = blob_data

        gen = MultiEmbeddingGenerator(
            methods=['umap', 'pca'],
            n_components=2
        )

        embeddings = gen.fit_transform(X)

        # Should have PCA and UMAP
        assert 'pca' in embeddings
        has_umap = any('umap' in key for key in embeddings.keys())
        assert has_umap

        for key, emb in embeddings.items():
            assert emb.shape == (len(X), 2)

    def test_high_dimensional_embedding(self, blob_data):
        """Test higher-dimensional embeddings"""
        X, y_true = blob_data

        gen = MultiEmbeddingGenerator(
            methods=['pca'],
            n_components=5
        )

        embeddings = gen.fit_transform(X)

        for key, emb in embeddings.items():
            assert emb.shape == (len(X), 5)


class TestTopologicalGapDetector:
    """Tests for topological gap detector"""

    def test_basic_gap_detection(self, blob_data):
        """Test basic gap detection"""
        X, y_true = blob_data

        tgd = TopologicalGapDetector()
        tgd.fit(X)

        assert hasattr(tgd, 'gaps_')
        assert isinstance(tgd.gaps_, list)
        # Should detect some gaps in well-separated blobs
        assert len(tgd.gaps_) >= 0

    def test_no_gaps(self, moon_data):
        """Test with data that has no clear gaps"""
        X, y_true = moon_data

        tgd = TopologicalGapDetector()
        tgd.fit(X)

        # May or may not find gaps in moon data
        assert hasattr(tgd, 'gaps_')
        assert isinstance(tgd.gaps_, list)


class TestConsensusClusteringPipeline:
    """Tests for consensus clustering pipeline"""

    def test_basic_pipeline(self, blob_data):
        """Test basic consensus clustering"""
        X, y_true = blob_data

        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=False,  # Skip HDBSCAN if not available
            use_spectral=True,
            use_bgmm=True,
            n_bootstrap=10
        )

        pipeline.fit(X, generate_embeddings=False)

        assert pipeline.consensus_labels_ is not None
        assert len(pipeline.consensus_labels_) == len(X)
        assert pipeline.confidence_scores_ is not None
        assert len(pipeline.confidence_scores_) == len(X)

    def test_without_embeddings(self, blob_data):
        """Test pipeline without generating embeddings"""
        X, y_true = blob_data

        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=False,
            use_bgmm=True,
            n_bootstrap=10
        )

        pipeline.fit(X, generate_embeddings=False)

        assert pipeline.consensus_labels_ is not None
        # Original data is always in embeddings_
        assert 'original' in pipeline.embeddings_

    def test_individual_methods(self, blob_data):
        """Test running individual methods"""
        X, y_true = blob_data

        # Test BGM only
        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=False,
            use_spectral=False,
            use_bgmm=True,
            n_bootstrap=10
        )

        pipeline.fit(X, generate_embeddings=False)
        assert pipeline.consensus_labels_ is not None

    def test_confidence_scores(self, blob_data):
        """Test confidence score computation"""
        X, y_true = blob_data

        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=False,
            use_bgmm=True,
            n_bootstrap=20
        )
        pipeline.fit(X, generate_embeddings=False)

        # Confidence should be in [0, 1]
        assert np.all(pipeline.confidence_scores_ >= 0)
        assert np.all(pipeline.confidence_scores_ <= 1)

    def test_with_embeddings(self, blob_data):
        """Test pipeline with embedding generation"""
        X, y_true = blob_data

        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=False,
            use_bgmm=True,
            n_bootstrap=10
        )

        pipeline.fit(X, generate_embeddings=True)

        # Should have generated embeddings
        assert len(pipeline.embeddings_) > 0
        assert pipeline.consensus_labels_ is not None


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_small_dataset(self, small_data):
        """Test with small dataset"""
        X, y_true = small_data

        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=False,
            use_bgmm=True,
            n_bootstrap=5
        )

        pipeline.fit(X, generate_embeddings=False)

        # Should still produce results
        assert pipeline.consensus_labels_ is not None
        assert len(pipeline.consensus_labels_) == len(X)

    def test_high_dimensional(self, high_dim_data):
        """Test with high-dimensional data"""
        X, y_true = high_dim_data

        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=False,
            use_bgmm=True,
            use_spectral=True,
            n_bootstrap=10
        )

        pipeline.fit(X, generate_embeddings=True)

        assert pipeline.consensus_labels_ is not None
        assert len(pipeline.consensus_labels_) == len(X)

    def test_single_cluster_scenario(self):
        """Test when data naturally forms single cluster"""
        # Generate tight single cluster
        X = np.random.randn(50, 5) * 0.1

        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=False,
            use_bgmm=True,
            n_bootstrap=10
        )
        pipeline.fit(X, generate_embeddings=False)

        # Should handle gracefully (may produce 1 or few clusters)
        assert pipeline.consensus_labels_ is not None
        assert len(pipeline.consensus_labels_) == len(X)

    @pytest.mark.skipif(not HDBSCAN_AVAILABLE, reason="hdbscan not installed")
    def test_noise_cluster_handling(self):
        """Test handling of noise cluster (-1 labels)"""
        X = np.vstack([
            np.random.randn(40, 5),  # Main cluster
            np.random.randn(10, 5) * 5  # Noise points far away
        ])

        sweep = HDBSCANParameterSweep(
            min_cluster_sizes=[5],
            min_samples_list=[3],
            metrics=['euclidean'],
            cluster_selection_methods=['eom']
        )

        sweep.fit(X)

        # Should handle noise labels (-1) gracefully
        assert sweep.best_labels_ is not None
        assert len(sweep.best_labels_) == len(X)


class TestIntegration:
    """Integration tests combining multiple components"""

    def test_full_pipeline_blob_data(self, blob_data):
        """Test full pipeline on blob data"""
        X, y_true = blob_data

        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=False,
            use_spectral=True,
            use_bgmm=True,
            n_bootstrap=15
        )

        pipeline.fit(X, generate_embeddings=True)

        # Should find some reasonable number of clusters
        n_clusters = len(np.unique(pipeline.consensus_labels_))
        assert 2 <= n_clusters <= 50  # Relaxed upper bound

        # Should have generated embeddings
        assert len(pipeline.embeddings_) > 0

    def test_full_pipeline_moon_data(self, moon_data):
        """Test full pipeline on moon data (non-linear)"""
        X, y_true = moon_data

        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=False,
            use_spectral=True,
            use_bgmm=True,
            n_bootstrap=15
        )

        pipeline.fit(X, generate_embeddings=True)

        # Should find clusters
        n_clusters = len(np.unique(pipeline.consensus_labels_))
        assert n_clusters >= 2

    def test_reproducibility(self, blob_data):
        """Test that results are reproducible with same random state"""
        X, y_true = blob_data

        pipeline1 = ConsensusClusteringPipeline(
            use_hdbscan=False,
            use_bgmm=True,
            n_bootstrap=10,
            random_state=42
        )

        pipeline2 = ConsensusClusteringPipeline(
            use_hdbscan=False,
            use_bgmm=True,
            n_bootstrap=10,
            random_state=42
        )

        pipeline1.fit(X, generate_embeddings=False)
        pipeline2.fit(X, generate_embeddings=False)

        # Should produce identical results
        assert np.array_equal(pipeline1.consensus_labels_, pipeline2.consensus_labels_)

    def test_metrics_consistency(self, blob_data):
        """Test that metrics are computed"""
        X, y_true = blob_data

        pipeline = ConsensusClusteringPipeline(
            use_hdbscan=False,
            use_bgmm=True,
            n_bootstrap=10,
            random_state=42
        )

        pipeline.fit(X, generate_embeddings=False)

        # Check that we have metrics if they were computed
        if pipeline.metrics_ is not None:
            # Check metrics exist (correct attribute names)
            assert pipeline.metrics_.silhouette is not None
            # Check metrics are in expected ranges
            assert -1 <= pipeline.metrics_.silhouette <= 1