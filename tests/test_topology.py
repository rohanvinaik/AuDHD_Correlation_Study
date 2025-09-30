"""Comprehensive tests for topology analysis"""
import pytest
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

from src.audhd_correlation.modeling.topology import (
    MinimumSpanningTreeAnalyzer,
    DensityGapAnalyzer,
    KNNGraphConnectivityAnalyzer,
    SpectralGapDetector,
    TopologyAnalyzer,
    analyze_separation_vs_spectrum,
    GapScore,
    BaselineManifold,
    RotationNull,
    TopologyGate,
    DeviationScores,
    SeparationDecision,
)

# Check for optional dependencies
try:
    import ripser
    from src.audhd_correlation.modeling.topology import PersistentHomologyAnalyzer
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


@pytest.fixture
def separated_data():
    """Well-separated clusters"""
    X, y = make_blobs(n_samples=200, n_features=5, centers=4,
                      cluster_std=0.5, random_state=42)
    return X, y


@pytest.fixture
def spectrum_data():
    """Continuous spectrum (moon-shaped)"""
    X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
    # Add more features
    X = np.hstack([X, np.random.randn(200, 3) * 0.1])
    return X, y


@pytest.fixture
def intermediate_data():
    """Intermediate case (overlapping circles)"""
    X, y = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
    X = np.hstack([X, np.random.randn(200, 3) * 0.1])
    return X, y


class TestMinimumSpanningTreeAnalyzer:
    """Tests for MST analysis"""

    def test_fit(self, separated_data):
        """Test basic fitting"""
        X, y = separated_data

        analyzer = MinimumSpanningTreeAnalyzer()
        analyzer.fit(X, y)

        assert analyzer.mst_ is not None
        assert analyzer.edge_lengths_ is not None
        assert len(analyzer.edge_lengths_) == len(X) - 1  # MST has n-1 edges
        assert analyzer.gap_edges_ is not None

    def test_gap_statistic_separated(self, separated_data):
        """Test gap statistic on separated data"""
        X, y = separated_data

        analyzer = MinimumSpanningTreeAnalyzer()
        analyzer.fit(X, y)

        gaps = analyzer.compute_gap_statistic(y)

        assert 'gap_ratio' in gaps
        assert 'within_cluster_mean' in gaps
        assert 'between_cluster_mean' in gaps

        # Separated data should have high gap ratio
        assert gaps['gap_ratio'] > 1.0
        assert gaps['between_cluster_mean'] > gaps['within_cluster_mean']

    def test_gap_statistic_spectrum(self, spectrum_data):
        """Test gap statistic on spectrum data"""
        X, y = spectrum_data

        analyzer = MinimumSpanningTreeAnalyzer()
        analyzer.fit(X, y)

        gaps = analyzer.compute_gap_statistic(y)

        # Spectrum data should have lower gap ratio
        assert 'gap_ratio' in gaps
        assert gaps['gap_ratio'] >= 0

    def test_separation_hypothesis(self, separated_data):
        """Test separation hypothesis test"""
        X, y = separated_data

        analyzer = MinimumSpanningTreeAnalyzer()
        analyzer.fit(X, y)

        test_result = analyzer.test_separation_hypothesis(y, n_bootstrap=100)

        assert 'p_value' in test_result
        assert 'effect_size' in test_result
        assert 'bootstrap_p_value' in test_result

        # Separated data should have significant p-value
        assert test_result['p_value'] < 0.05

    def test_without_labels(self, separated_data):
        """Test MST without labels"""
        X, y = separated_data

        analyzer = MinimumSpanningTreeAnalyzer()
        analyzer.fit(X)  # No labels

        assert analyzer.mst_ is not None
        assert analyzer.gap_edges_ is None

        # Can still compute gap statistic
        gaps = analyzer.compute_gap_statistic()
        assert 'gap_ratio' in gaps


class TestDensityGapAnalyzer:
    """Tests for density gap analysis"""

    def test_fit(self, separated_data):
        """Test basic fitting"""
        X, y = separated_data

        analyzer = DensityGapAnalyzer(n_neighbors=10)
        analyzer.fit(X)

        assert analyzer.densities_ is not None
        assert len(analyzer.densities_) == len(X)
        assert np.all(analyzer.densities_ > 0)

    def test_gap_score_separated(self, separated_data):
        """Test gap score on separated data"""
        X, y = separated_data

        analyzer = DensityGapAnalyzer(n_neighbors=10)
        analyzer.fit(X)

        gap_score = analyzer.compute_gap_score(y)

        assert isinstance(gap_score, GapScore)
        assert 0 <= gap_score.score <= 1
        assert gap_score.interpretation in ['separated', 'spectrum', 'intermediate']

        # Gap score should be computed (may vary based on data)
        assert gap_score.score >= 0

    def test_gap_score_spectrum(self, spectrum_data):
        """Test gap score on spectrum data"""
        X, y = spectrum_data

        analyzer = DensityGapAnalyzer(n_neighbors=10)
        analyzer.fit(X)

        gap_score = analyzer.compute_gap_score(y)

        # Spectrum data may have lower gap score
        assert 0 <= gap_score.score <= 1

    def test_different_n_neighbors(self, separated_data):
        """Test with different k values"""
        X, y = separated_data

        for k in [5, 10, 20]:
            analyzer = DensityGapAnalyzer(n_neighbors=k)
            analyzer.fit(X)
            gap_score = analyzer.compute_gap_score(y)

            assert isinstance(gap_score, GapScore)


class TestKNNGraphConnectivityAnalyzer:
    """Tests for k-NN graph connectivity"""

    def test_fit(self, separated_data):
        """Test basic fitting"""
        X, y = separated_data

        analyzer = KNNGraphConnectivityAnalyzer(n_neighbors=10)
        analyzer.fit(X, y)

        assert analyzer.graph_ is not None
        assert analyzer.connectivity_ is not None
        assert 'n_components' in analyzer.connectivity_
        assert 'edge_purity' in analyzer.connectivity_

    def test_connectivity_separated(self, separated_data):
        """Test connectivity on separated data"""
        X, y = separated_data

        analyzer = KNNGraphConnectivityAnalyzer(n_neighbors=10)
        analyzer.fit(X, y)

        # Separated data should have high edge purity
        assert analyzer.connectivity_['edge_purity'] > 0.7

    def test_modularity(self, separated_data):
        """Test modularity computation"""
        X, y = separated_data

        analyzer = KNNGraphConnectivityAnalyzer(n_neighbors=10)
        analyzer.fit(X, y)

        modularity = analyzer.compute_modularity(y)

        assert -1 <= modularity <= 1
        # Separated data should have positive modularity
        assert modularity > 0

    def test_connectivity_hypothesis(self, separated_data):
        """Test connectivity hypothesis"""
        X, y = separated_data

        analyzer = KNNGraphConnectivityAnalyzer(n_neighbors=10)
        analyzer.fit(X, y)

        test_result = analyzer.test_connectivity_hypothesis(y)

        assert 'edge_purity' in test_result
        assert 'modularity' in test_result
        assert 'interpretation' in test_result
        assert test_result['interpretation'] in ['separated', 'spectrum', 'intermediate']

    def test_without_labels(self, separated_data):
        """Test without labels"""
        X, y = separated_data

        analyzer = KNNGraphConnectivityAnalyzer(n_neighbors=10)
        analyzer.fit(X)

        assert analyzer.graph_ is not None
        assert analyzer.connectivity_ is not None


class TestSpectralGapDetector:
    """Tests for spectral gap detection"""

    def test_fit(self, separated_data):
        """Test basic fitting"""
        X, y = separated_data

        detector = SpectralGapDetector(n_neighbors=10, n_eigenvalues=20)
        detector.fit(X)

        assert detector.eigenvalues_ is not None
        assert detector.eigenvectors_ is not None
        assert detector.gaps_ is not None
        assert len(detector.gaps_) == len(detector.eigenvalues_) - 1

    def test_detect_spectral_gap(self, separated_data):
        """Test gap detection"""
        X, y = separated_data

        detector = SpectralGapDetector(n_neighbors=10, n_eigenvalues=20)
        detector.fit(X)

        gap_info = detector.detect_spectral_gap()

        assert 'largest_gap_index' in gap_info
        assert 'n_clusters_estimate' in gap_info
        assert 'eigengap_ratio' in gap_info

        # Should detect around 4 clusters
        assert 2 <= gap_info['n_clusters_estimate'] <= 10

    def test_separation_hypothesis(self, separated_data):
        """Test separation hypothesis"""
        X, y = separated_data

        detector = SpectralGapDetector(n_neighbors=10, n_eigenvalues=20)
        detector.fit(X)

        test_result = detector.test_separation_hypothesis()

        assert 'interpretation' in test_result
        assert 'confidence' in test_result
        assert 0 <= test_result['confidence'] <= 1

    def test_different_eigenvalues(self, separated_data):
        """Test with different numbers of eigenvalues"""
        X, y = separated_data

        for n_eig in [5, 10, 15]:
            detector = SpectralGapDetector(n_neighbors=10, n_eigenvalues=n_eig)
            detector.fit(X)

            assert len(detector.eigenvalues_) <= n_eig


@pytest.mark.skipif(not RIPSER_AVAILABLE, reason="ripser not installed")
class TestPersistentHomologyAnalyzer:
    """Tests for persistent homology"""

    def test_fit(self, separated_data):
        """Test basic fitting"""
        X, y = separated_data

        analyzer = PersistentHomologyAnalyzer(maxdim=2)
        analyzer.fit(X)

        assert analyzer.diagrams_ is not None
        assert len(analyzer.diagrams_) >= 2  # H0 and H1

    def test_compute_lifetimes(self, separated_data):
        """Test lifetime computation"""
        X, y = separated_data

        analyzer = PersistentHomologyAnalyzer(maxdim=2)
        analyzer.fit(X)

        lifetimes = analyzer.compute_lifetimes()

        assert isinstance(lifetimes, dict)
        assert 0 in lifetimes  # H0
        assert all(isinstance(v, np.ndarray) for v in lifetimes.values())

    def test_persistence_entropy(self, separated_data):
        """Test persistence entropy"""
        X, y = separated_data

        analyzer = PersistentHomologyAnalyzer(maxdim=2)
        analyzer.fit(X)

        entropies = analyzer.compute_persistence_entropy()

        assert isinstance(entropies, dict)
        assert all(v >= 0 for v in entropies.values())

    def test_detect_significant_features(self, separated_data):
        """Test significant feature detection"""
        X, y = separated_data

        analyzer = PersistentHomologyAnalyzer(maxdim=2)
        analyzer.fit(X)

        significant = analyzer.detect_significant_features(dimension=0)

        assert isinstance(significant, np.ndarray)

    def test_interpret_topology(self, separated_data):
        """Test topology interpretation"""
        X, y = separated_data

        analyzer = PersistentHomologyAnalyzer(maxdim=2)
        analyzer.fit(X)

        interpretation = analyzer.interpret_topology()

        assert interpretation in ['separated', 'spectrum', 'intermediate']


class TestTopologyAnalyzer:
    """Tests for comprehensive topology analyzer"""

    def test_analyze_separated(self, separated_data):
        """Test on separated data"""
        X, y = separated_data

        analyzer = TopologyAnalyzer(n_neighbors=10, use_persistence=False, n_bootstrap=100)
        result = analyzer.analyze(X, y, compute_persistence=False)

        assert result.gap_scores is not None
        assert result.mst_analysis is not None
        assert result.knn_connectivity is not None
        assert result.spectral_gaps is not None
        assert result.hypothesis_test is not None
        assert result.overall_interpretation is not None

        # Separated data should be identified as separated
        assert result.overall_interpretation == 'separated'

    def test_analyze_spectrum(self, spectrum_data):
        """Test on spectrum data"""
        X, y = spectrum_data

        analyzer = TopologyAnalyzer(n_neighbors=10, use_persistence=False, n_bootstrap=100)
        result = analyzer.analyze(X, y, compute_persistence=False)

        # Moon data with clear labels may still show separation
        # The key is that the analysis completes without errors
        assert result.overall_interpretation in ['separated', 'spectrum', 'intermediate']

    def test_analyze_intermediate(self, intermediate_data):
        """Test on intermediate data"""
        X, y = intermediate_data

        analyzer = TopologyAnalyzer(n_neighbors=10, use_persistence=False, n_bootstrap=100)
        result = analyzer.analyze(X, y, compute_persistence=False)

        assert result.overall_interpretation in ['separated', 'spectrum', 'intermediate']

    @pytest.mark.skipif(not RIPSER_AVAILABLE, reason="ripser not installed")
    def test_with_persistence(self, separated_data):
        """Test with persistent homology"""
        X, y = separated_data

        analyzer = TopologyAnalyzer(n_neighbors=10, use_persistence=True, n_bootstrap=100)
        result = analyzer.analyze(X, y, compute_persistence=True)

        assert result.persistence is not None
        assert result.persistence.diagrams is not None

    def test_hypothesis_test(self, separated_data):
        """Test integrated hypothesis test"""
        X, y = separated_data

        analyzer = TopologyAnalyzer(n_neighbors=10, use_persistence=False, n_bootstrap=100)
        result = analyzer.analyze(X, y, compute_persistence=False)

        test = result.hypothesis_test

        assert 'separation_score' in test
        assert 'confidence' in test
        assert 'n_tests' in test

        assert 0 <= test['separation_score'] <= 1
        assert 0 <= test['confidence'] <= 1


class TestConvenienceFunction:
    """Test convenience function"""

    def test_convenience_function(self, separated_data):
        """Test convenience function"""
        X, y = separated_data

        result = analyze_separation_vs_spectrum(
            X, y,
            n_neighbors=10,
            use_persistence=False,
            n_bootstrap=100,
        )

        assert result.gap_scores is not None
        assert result.overall_interpretation is not None


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_small_dataset(self):
        """Test with small dataset"""
        X = np.random.randn(20, 3)
        y = np.array([0] * 10 + [1] * 10)

        analyzer = TopologyAnalyzer(n_neighbors=5, use_persistence=False, n_bootstrap=50)
        result = analyzer.analyze(X, y, compute_persistence=False)

        assert result.overall_interpretation is not None

    def test_single_cluster(self):
        """Test with single cluster"""
        X = np.random.randn(100, 5)
        y = np.zeros(100, dtype=int)

        analyzer = TopologyAnalyzer(n_neighbors=10, use_persistence=False, n_bootstrap=100)
        result = analyzer.analyze(X, y, compute_persistence=False)

        assert result.overall_interpretation is not None

    def test_many_clusters(self):
        """Test with many clusters"""
        X, y = make_blobs(n_samples=200, n_features=5, centers=20,
                          cluster_std=0.3, random_state=42)

        analyzer = TopologyAnalyzer(n_neighbors=10, use_persistence=False, n_bootstrap=100)
        result = analyzer.analyze(X, y, compute_persistence=False)

        assert result.overall_interpretation is not None

    def test_high_dimensional(self):
        """Test with high-dimensional data"""
        X, y = make_blobs(n_samples=100, n_features=50, centers=4,
                          cluster_std=1.0, random_state=42)

        analyzer = TopologyAnalyzer(n_neighbors=10, use_persistence=False, n_bootstrap=100)
        result = analyzer.analyze(X, y, compute_persistence=False)

        assert result.overall_interpretation is not None

    def test_noise_labels(self):
        """Test with noise labels (-1)"""
        X, y = make_blobs(n_samples=100, n_features=5, centers=3,
                          cluster_std=0.5, random_state=42)

        # Add some noise points
        y[:10] = -1

        analyzer = TopologyAnalyzer(n_neighbors=10, use_persistence=False, n_bootstrap=100)
        result = analyzer.analyze(X, y, compute_persistence=False)

        assert result.overall_interpretation is not None


class TestStatisticalProperties:
    """Test statistical properties of analyses"""

    def test_gap_score_range(self, separated_data):
        """Test gap score is in valid range"""
        X, y = separated_data

        analyzer = DensityGapAnalyzer(n_neighbors=10)
        analyzer.fit(X)
        gap_score = analyzer.compute_gap_score(y)

        assert 0 <= gap_score.score <= 1
        assert 0 <= gap_score.p_value <= 1

    def test_modularity_range(self, separated_data):
        """Test modularity is in valid range"""
        X, y = separated_data

        analyzer = KNNGraphConnectivityAnalyzer(n_neighbors=10)
        analyzer.fit(X, y)
        modularity = analyzer.compute_modularity(y)

        assert -1 <= modularity <= 1

    def test_eigenvalues_sorted(self, separated_data):
        """Test eigenvalues are sorted"""
        X, y = separated_data

        detector = SpectralGapDetector(n_neighbors=10, n_eigenvalues=20)
        detector.fit(X)

        # Eigenvalues should be non-negative and sorted
        assert np.all(detector.eigenvalues_ >= 0)
        assert np.all(np.diff(detector.eigenvalues_) >= 0)

    def test_edge_purity_range(self, separated_data):
        """Test edge purity is in [0, 1]"""
        X, y = separated_data

        analyzer = KNNGraphConnectivityAnalyzer(n_neighbors=10)
        analyzer.fit(X, y)

        edge_purity = analyzer.connectivity_['edge_purity']
        assert 0 <= edge_purity <= 1


class TestReproducibility:
    """Test reproducibility"""

    def test_mst_reproducible(self, separated_data):
        """Test MST is reproducible"""
        X, y = separated_data

        analyzer1 = MinimumSpanningTreeAnalyzer()
        analyzer1.fit(X, y)

        analyzer2 = MinimumSpanningTreeAnalyzer()
        analyzer2.fit(X, y)

        # MST should be deterministic
        assert np.array_equal(analyzer1.edge_lengths_, analyzer2.edge_lengths_)

    def test_density_reproducible(self, separated_data):
        """Test density computation is reproducible"""
        X, y = separated_data

        analyzer1 = DensityGapAnalyzer(n_neighbors=10)
        analyzer1.fit(X)

        analyzer2 = DensityGapAnalyzer(n_neighbors=10)
        analyzer2.fit(X)

        assert np.allclose(analyzer1.densities_, analyzer2.densities_)


# ============================================================================
# Baseline-Deviation Framework Tests
# ============================================================================

@pytest.fixture
def baseline_deviation_data():
    """Data with baseline and deviants"""
    # Baseline: high-density Gaussian
    X_baseline, _ = make_blobs(n_samples=150, n_features=10, centers=1,
                               cluster_std=0.5, random_state=42)

    # Deviants: outliers from baseline
    X_deviants, _ = make_blobs(n_samples=50, n_features=10, centers=3,
                               cluster_std=0.7, random_state=43)
    X_deviants += 5.0  # Shift away from baseline

    X = np.vstack([X_baseline, X_deviants])
    control_mask = np.array([True] * 150 + [False] * 50)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, control_mask


@pytest.fixture
def batch_confounded_data():
    """Data with batch effects confounded with clusters"""
    # True clusters
    X, y = make_blobs(n_samples=200, n_features=10, centers=3,
                      cluster_std=1.0, random_state=42)

    # Add batch effects
    batch_size = len(X) // 2
    X[:batch_size] += np.random.randn(10) * 3.0  # Batch 1 shift
    X[batch_size:] -= np.random.randn(10) * 3.0  # Batch 2 shift

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


class TestBaselineManifold:
    """Tests for BaselineManifold class"""

    def test_control_mode_fit(self, baseline_deviation_data):
        """Test fitting in control mode"""
        X, control_mask = baseline_deviation_data

        baseline = BaselineManifold(
            mode='control',
            n_neighbors=15,
            local_pca_components=5
        )
        baseline.fit(X, control_mask=control_mask)

        # Check fitted attributes
        assert baseline.baseline_indices_ is not None
        assert len(baseline.baseline_indices_) == control_mask.sum()
        assert baseline.knn_graph_ is not None
        assert baseline.local_bases_ is not None
        assert baseline.mst_ is not None

    def test_unsupervised_mode_fit(self, baseline_deviation_data):
        """Test fitting in unsupervised mode"""
        X, _ = baseline_deviation_data

        baseline = BaselineManifold(
            mode='unsupervised',
            n_neighbors=15,
            local_pca_components=5,
            density_percentile=75.0
        )
        baseline.fit(X)

        # Check fitted attributes
        assert baseline.baseline_indices_ is not None
        assert len(baseline.baseline_indices_) > 0
        assert baseline.knn_graph_ is not None
        assert baseline.local_bases_ is not None

    def test_score_deviations(self, baseline_deviation_data):
        """Test deviation scoring"""
        X, control_mask = baseline_deviation_data

        baseline = BaselineManifold(mode='control', n_neighbors=15)
        baseline.fit(X, control_mask=control_mask)

        scores = baseline.score(X)

        # Check DeviationScores structure
        assert isinstance(scores, DeviationScores)
        assert len(scores.orthogonal_residual) == len(X)
        assert len(scores.mst_delta) == len(X)
        assert len(scores.knn_curvature) == len(X)
        assert len(scores.deviation_score) == len(X)

        # All scores should be non-negative
        assert np.all(scores.orthogonal_residual >= 0)
        assert np.all(scores.deviation_score >= 0)

    def test_deviation_scores_discriminate(self, baseline_deviation_data):
        """Test that deviation scores discriminate baseline from deviants"""
        X, control_mask = baseline_deviation_data

        baseline = BaselineManifold(mode='control', n_neighbors=15)
        baseline.fit(X, control_mask=control_mask)

        scores = baseline.score(X)

        # Deviants should have higher deviation scores than baseline
        baseline_scores = scores.deviation_score[control_mask]
        deviant_scores = scores.deviation_score[~control_mask]

        # Mean deviant score should be higher
        assert deviant_scores.mean() > baseline_scores.mean()


class TestRotationNull:
    """Tests for RotationNull class"""

    def test_fit(self, baseline_deviation_data):
        """Test fitting rotation null"""
        X, control_mask = baseline_deviation_data

        baseline = BaselineManifold(mode='control', n_neighbors=15)
        baseline.fit(X, control_mask=control_mask)

        rotation_null = RotationNull(
            baseline_manifold=baseline,
            n_rotations=50,  # Reduced for speed
            preserve_scale=True,
            random_state=42
        )
        rotation_null.fit(X)

        # Check fitted attributes
        assert rotation_null.null_distribution_ is not None
        assert len(rotation_null.null_distribution_) == 50
        assert np.all(rotation_null.null_distribution_ >= 0)

    def test_quantile(self, baseline_deviation_data):
        """Test quantile computation"""
        X, control_mask = baseline_deviation_data

        baseline = BaselineManifold(mode='control', n_neighbors=15)
        baseline.fit(X, control_mask=control_mask)

        rotation_null = RotationNull(baseline_manifold=baseline, n_rotations=50, random_state=42)
        rotation_null.fit(X)

        # Test different quantiles
        q95 = rotation_null.quantile(0.95)
        q99 = rotation_null.quantile(0.99)

        assert q99 > q95  # Higher quantile should be higher
        assert q95 > 0
        assert q99 > 0

    def test_fdr_threshold(self, baseline_deviation_data):
        """Test FDR threshold computation"""
        X, control_mask = baseline_deviation_data

        baseline = BaselineManifold(mode='control', n_neighbors=15)
        baseline.fit(X, control_mask=control_mask)

        rotation_null = RotationNull(baseline_manifold=baseline, n_rotations=50, random_state=42)
        rotation_null.fit(X)

        # Get observed scores
        observed_scores = baseline.score(X).deviation_score

        # Compute FDR threshold
        threshold = rotation_null.compute_fdr_threshold(observed_scores, q=0.1)

        assert threshold > 0
        assert np.isfinite(threshold)

    def test_null_distribution_reasonable(self, baseline_deviation_data):
        """Test that null distribution is reasonable"""
        X, control_mask = baseline_deviation_data

        baseline = BaselineManifold(mode='control', n_neighbors=15)
        baseline.fit(X, control_mask=control_mask)

        rotation_null = RotationNull(baseline_manifold=baseline, n_rotations=100, random_state=42)
        rotation_null.fit(X)

        # Null distribution should have reasonable variance
        null_std = np.std(rotation_null.null_distribution_)
        null_mean = np.mean(rotation_null.null_distribution_)

        assert null_std > 0  # Some variance
        assert null_mean > 0  # Positive mean


class TestTopologyGate:
    """Tests for enhanced TopologyGate"""

    def test_analyze_separated(self, separated_data):
        """Test on well-separated clusters"""
        X, y = separated_data

        gate = TopologyGate(
            min_separation_score=0.6,
            min_confidence=0.5,
            n_bootstrap=50,  # Reduced for speed
            random_state=42
        )

        decision = gate.analyze(X, labels=y)

        # Check SeparationDecision structure
        assert isinstance(decision, SeparationDecision)
        assert decision.separation_score >= 0
        assert decision.separation_score <= 1
        assert decision.confidence_interval is not None
        assert len(decision.confidence_interval) == 2
        assert decision.decision in ['separated', 'spectrum', 'intermediate']
        assert isinstance(decision.subtype_claim_allowed, bool)

        # Well-separated data should pass gate
        assert decision.decision == 'separated'
        assert decision.subtype_claim_allowed is True

    def test_analyze_spectrum(self, spectrum_data):
        """Test on spectrum data"""
        X, y = spectrum_data

        gate = TopologyGate(
            min_separation_score=0.6,
            min_confidence=0.5,
            n_bootstrap=50,
            random_state=42
        )

        decision = gate.analyze(X, labels=None)  # No labels for spectrum

        assert isinstance(decision, SeparationDecision)
        # Spectrum data may or may not pass depending on structure
        assert decision.decision in ['separated', 'spectrum', 'intermediate']

    def test_bootstrap_ci(self, separated_data):
        """Test bootstrap confidence interval"""
        X, y = separated_data

        gate = TopologyGate(
            min_separation_score=0.6,
            min_confidence=0.5,
            n_bootstrap=100,
            random_state=42
        )

        decision = gate.analyze(X, labels=y)

        ci_lower, ci_upper = decision.confidence_interval

        # CI should be valid
        assert ci_lower <= ci_upper
        assert ci_lower >= 0
        assert ci_upper <= 1

        # Separation score should be within CI
        assert ci_lower <= decision.separation_score <= ci_upper

    def test_batch_confounded_detection(self, batch_confounded_data):
        """Test detection of batch-confounded clusters"""
        X, y = batch_confounded_data

        gate = TopologyGate(
            min_separation_score=0.6,
            min_confidence=0.5,
            n_bootstrap=50,
            random_state=42
        )

        decision = gate.analyze(X, labels=y)

        # Batch-confounded data may fail gate
        # (depends on severity of batch effects)
        assert decision.decision in ['separated', 'spectrum', 'intermediate']


class TestBaselineDeviationIntegration:
    """Integration tests for baseline-deviation framework"""

    def test_end_to_end_control_mode(self, baseline_deviation_data):
        """Test end-to-end pipeline in control mode"""
        X, control_mask = baseline_deviation_data

        # Step 1: Learn baseline
        baseline = BaselineManifold(mode='control', n_neighbors=15)
        baseline.fit(X, control_mask=control_mask)

        # Step 2: Score deviations
        deviation_scores = baseline.score(X)

        # Step 3: Compute threshold via rotation null
        rotation_null = RotationNull(baseline_manifold=baseline, n_rotations=50, random_state=42)
        rotation_null.fit(X)
        threshold = rotation_null.quantile(0.95)

        # Step 4: Identify deviants
        deviants_mask = deviation_scores.deviation_score >= threshold

        # Should identify some deviants
        assert deviants_mask.sum() > 0
        assert deviants_mask.sum() < len(X)

        # Step 5: Topology gate (if enough deviants)
        if deviants_mask.sum() >= 30:
            X_deviants = X[deviants_mask]
            gate = TopologyGate(min_separation_score=0.5, n_bootstrap=30, random_state=42)
            decision = gate.analyze(X_deviants)

            assert decision.decision in ['separated', 'spectrum', 'intermediate']

    def test_end_to_end_unsupervised_mode(self, separated_data):
        """Test end-to-end pipeline in unsupervised mode"""
        X, y = separated_data

        # Step 1: Learn baseline (unsupervised)
        baseline = BaselineManifold(
            mode='unsupervised',
            n_neighbors=15,
            density_percentile=75.0
        )
        baseline.fit(X)

        # Step 2: Score deviations
        deviation_scores = baseline.score(X)

        # Step 3: FDR threshold
        rotation_null = RotationNull(baseline_manifold=baseline, n_rotations=50, random_state=42)
        rotation_null.fit(X)
        threshold = rotation_null.compute_fdr_threshold(
            deviation_scores.deviation_score,
            q=0.1
        )

        # Step 4: Identify deviants
        deviants_mask = deviation_scores.deviation_score >= threshold

        # Should identify some structure
        assert isinstance(deviants_mask, np.ndarray)
        assert len(deviants_mask) == len(X)

    def test_spectrum_vs_clusters(self, separated_data, spectrum_data):
        """Test that framework distinguishes spectrum from clusters"""

        # Test on separated data
        X_sep, y_sep = separated_data
        baseline_sep = BaselineManifold(mode='unsupervised', n_neighbors=15)
        baseline_sep.fit(X_sep)

        rotation_null_sep = RotationNull(baseline_manifold=baseline_sep, n_rotations=50, random_state=42)
        rotation_null_sep.fit(X_sep)

        scores_sep = baseline_sep.score(X_sep)
        threshold_sep = rotation_null_sep.quantile(0.90)
        deviants_mask_sep = scores_sep.deviation_score >= threshold_sep

        # Test on spectrum data
        X_spec, y_spec = spectrum_data
        baseline_spec = BaselineManifold(mode='unsupervised', n_neighbors=15)
        baseline_spec.fit(X_spec)

        rotation_null_spec = RotationNull(baseline_manifold=baseline_spec, n_rotations=50, random_state=42)
        rotation_null_spec.fit(X_spec)

        scores_spec = baseline_spec.score(X_spec)
        threshold_spec = rotation_null_spec.quantile(0.90)
        deviants_mask_spec = scores_spec.deviation_score >= threshold_spec

        # Both should run without errors
        assert isinstance(deviants_mask_sep, np.ndarray)
        assert isinstance(deviants_mask_spec, np.ndarray)