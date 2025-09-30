"""Comprehensive tests for validation framework"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from src.audhd_correlation.validation.metrics import (
    compute_internal_metrics,
    compute_stability_metrics,
    compute_outlier_robustness,
    compute_balanced_metrics,
)

from src.audhd_correlation.validation.stability import (
    bootstrap_stability,
    subsampling_stability,
    noise_stability,
    feature_stability,
    permutation_test_stability,
)

from src.audhd_correlation.validation.biological import (
    pathway_enrichment_analysis,
    clinical_relevance_analysis,
    symptom_severity_analysis,
    diagnostic_concordance,
    functional_outcome_prediction,
)

from src.audhd_correlation.validation.cross_validation import (
    cross_site_validation,
    stratified_cross_validation,
    batch_effect_validation,
)


@pytest.fixture
def clustering_data():
    """Generate test clustering data"""
    X, y = make_blobs(n_samples=200, n_features=10, centers=4,
                      cluster_std=1.0, random_state=42)
    return X, y


@pytest.fixture
def imbalanced_data():
    """Generate imbalanced clustering data"""
    X, y = make_blobs(n_samples=200, n_features=10, centers=4,
                      cluster_std=1.0, center_box=(-10, 10), random_state=42)

    # Make clusters imbalanced
    mask = (y == 0) & (np.arange(len(y)) % 3 != 0)
    X = X[~mask]
    y = y[~mask]

    return X, y


@pytest.fixture
def clinical_data(clustering_data):
    """Generate synthetic clinical data"""
    X, y = clustering_data

    # Continuous variables
    symptom_severity = np.random.randn(len(y)) + y * 0.5
    age = np.random.randint(5, 18, len(y))

    # Categorical variable
    diagnosis = np.array(['ADHD', 'ASD', 'Both', 'Neither'])[y]

    clinical_df = pd.DataFrame({
        'symptom_severity': symptom_severity,
        'age': age,
        'diagnosis': diagnosis,
    })

    return clinical_df


class TestInternalMetrics:
    """Tests for internal validation metrics"""

    def test_compute_internal_metrics(self, clustering_data):
        """Test internal metrics computation"""
        X, y = clustering_data

        metrics = compute_internal_metrics(X, y)

        assert -1 <= metrics.silhouette <= 1
        assert metrics.davies_bouldin >= 0
        assert metrics.calinski_harabasz >= 0
        assert metrics.dunn_index >= 0
        assert metrics.variance_ratio >= 0
        assert 0 <= metrics.overall_quality <= 1

    def test_per_cluster_silhouette(self, clustering_data):
        """Test per-cluster silhouette scores"""
        X, y = clustering_data

        metrics = compute_internal_metrics(X, y)

        assert len(metrics.silhouette_per_cluster) == len(np.unique(y))
        for score in metrics.silhouette_per_cluster.values():
            assert -1 <= score <= 1

    def test_robust_metrics(self, clustering_data):
        """Test robust metrics with outlier handling"""
        X, y = clustering_data

        metrics = compute_internal_metrics(X, y, handle_outliers=True)

        assert metrics.robust_silhouette is not None
        assert metrics.median_silhouette is not None

    def test_noise_labels(self, clustering_data):
        """Test with noise labels (-1)"""
        X, y = clustering_data

        # Add some noise points
        y[:10] = -1

        metrics = compute_internal_metrics(X, y)

        assert metrics.silhouette is not None
        assert not np.isnan(metrics.silhouette)

    def test_single_cluster(self):
        """Test with single cluster"""
        X = np.random.randn(50, 5)
        y = np.zeros(50)

        metrics = compute_internal_metrics(X, y)

        # Should return default metrics
        assert metrics.silhouette == 0.0


class TestStabilityMetrics:
    """Tests for stability metrics"""

    def test_compute_stability_metrics(self, clustering_data):
        """Test stability metrics computation"""
        X, y = clustering_data

        metrics = compute_stability_metrics(X, y)

        assert 'coefficient_of_variation' in metrics
        assert 'imbalance_ratio' in metrics
        assert metrics['imbalance_ratio'] >= 0.99  # Allow floating point tolerance

    def test_imbalanced_data(self, imbalanced_data):
        """Test with imbalanced clusters"""
        X, y = imbalanced_data

        metrics = compute_stability_metrics(X, y)

        # Should have high imbalance ratio
        assert metrics['imbalance_ratio'] > 1.5


class TestOutlierRobustness:
    """Tests for outlier robustness"""

    def test_outlier_robustness(self, clustering_data):
        """Test robustness to outliers"""
        X, y = clustering_data

        robustness = compute_outlier_robustness(
            X, y, contamination=0.1, n_iterations=5
        )

        assert 'silhouette_degradation' in robustness
        assert 'silhouette_robustness' in robustness
        assert 0 <= robustness['silhouette_robustness'] <= 1

    def test_different_contamination(self, clustering_data):
        """Test with different contamination levels"""
        X, y = clustering_data

        for contamination in [0.05, 0.1, 0.2]:
            robustness = compute_outlier_robustness(
                X, y, contamination=contamination, n_iterations=3
            )

            assert 'silhouette_degradation' in robustness


class TestBalancedMetrics:
    """Tests for balanced metrics"""

    def test_balanced_metrics(self, clustering_data):
        """Test balanced metrics"""
        X, y = clustering_data

        metrics = compute_balanced_metrics(X, y)

        assert 'balanced_silhouette' in metrics
        assert 'weighted_compactness' in metrics
        assert -1 <= metrics['balanced_silhouette'] <= 1

    def test_imbalanced_clusters(self, imbalanced_data):
        """Test with imbalanced clusters"""
        X, y = imbalanced_data

        metrics = compute_balanced_metrics(X, y)

        # Balanced silhouette should differ from standard silhouette
        standard_metrics = compute_internal_metrics(X, y)

        # Both should be valid
        assert -1 <= metrics['balanced_silhouette'] <= 1
        assert -1 <= standard_metrics.silhouette <= 1


class TestBootstrapStability:
    """Tests for bootstrap stability"""

    def clustering_func(self, X):
        """Simple clustering function for testing"""
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        return kmeans.fit_predict(X)

    def test_bootstrap_stability(self, clustering_data):
        """Test bootstrap stability"""
        X, y = clustering_data

        result = bootstrap_stability(
            X, y, self.clustering_func, n_bootstrap=10, sample_fraction=0.8
        )

        assert len(result.ari_scores) > 0
        assert 0 <= result.ari_mean <= 1
        assert 0 <= result.stability_score <= 1
        assert result.interpretation in ['excellent', 'good', 'moderate', 'poor']

    def test_confidence_intervals(self, clustering_data):
        """Test confidence interval computation"""
        X, y = clustering_data

        result = bootstrap_stability(
            X, y, self.clustering_func, n_bootstrap=20, sample_fraction=0.8
        )

        ci_ari = result.ari_ci
        ci_ami = result.ami_ci

        assert ci_ari[0] <= result.ari_mean <= ci_ari[1]
        assert ci_ami[0] <= result.ami_mean <= ci_ami[1]


class TestSubsamplingStability:
    """Tests for subsampling stability"""

    def clustering_func(self, X):
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        return kmeans.fit_predict(X)

    def test_subsampling_stability(self, clustering_data):
        """Test subsampling stability"""
        X, y = clustering_data

        results = subsampling_stability(
            X, y, self.clustering_func,
            subsample_fractions=[0.6, 0.8],
            n_iterations=5
        )

        assert 'fraction_0.6' in results
        assert 'fraction_0.8' in results

        for result in results.values():
            assert result.stability_score >= 0


class TestNoiseStability:
    """Tests for noise stability"""

    def clustering_func(self, X):
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        return kmeans.fit_predict(X)

    def test_noise_stability(self, clustering_data):
        """Test stability to noise"""
        X, y = clustering_data

        results = noise_stability(
            X, y, self.clustering_func,
            noise_levels=[0.05, 0.1],
            n_iterations=5
        )

        assert 'noise_0.05' in results
        assert 'noise_0.10' in results

        # Lower noise should have higher stability
        assert results['noise_0.05'].stability_score >= results['noise_0.10'].stability_score - 0.3


class TestFeatureStability:
    """Tests for feature stability"""

    def clustering_func(self, X):
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        return kmeans.fit_predict(X)

    def test_feature_stability(self, clustering_data):
        """Test stability with feature subsets"""
        X, y = clustering_data

        results = feature_stability(
            X, y, self.clustering_func,
            feature_fractions=[0.7, 0.9],
            n_iterations=5
        )

        assert 'features_0.7' in results
        assert 'features_0.9' in results


class TestPermutationTest:
    """Tests for permutation test"""

    def clustering_func(self, X):
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        return kmeans.fit_predict(X)

    def test_permutation_test(self, clustering_data):
        """Test permutation test for significance"""
        X, y = clustering_data

        result = permutation_test_stability(
            X, y, self.clustering_func, n_permutations=20
        )

        assert 'p_value' in result
        assert 'observed_ari' in result
        assert 0 <= result['p_value'] <= 1

        # Real clustering should be better than random
        assert result['observed_ari'] > result['random_ari_mean']


class TestPathwayEnrichment:
    """Tests for pathway enrichment"""

    def test_pathway_enrichment(self, clustering_data):
        """Test pathway enrichment analysis"""
        X, y = clustering_data

        # Create mock pathway data
        feature_names = [f'gene_{i}' for i in range(X.shape[1])]

        pathways = {
            'pathway1': ['gene_0', 'gene_1', 'gene_2'],
            'pathway2': ['gene_5', 'gene_6', 'gene_7'],
        }

        # Assign features to clusters (use feature indices as cluster labels)
        feature_labels = np.array([i % 4 for i in range(len(feature_names))])

        results = pathway_enrichment_analysis(
            feature_labels, feature_names, pathways, min_pathway_size=2
        )

        assert len(results) > 0

        for result in results:
            assert 0 <= result.p_value <= 1
            assert result.odds_ratio >= 0

    def test_empty_pathways(self, clustering_data):
        """Test with empty pathways"""
        X, y = clustering_data

        feature_names = [f'gene_{i}' for i in range(X.shape[1])]
        pathways = {}

        feature_labels = np.array([i % 4 for i in range(len(feature_names))])

        results = pathway_enrichment_analysis(
            feature_labels, feature_names, pathways
        )

        assert len(results) == 0


class TestClinicalRelevance:
    """Tests for clinical relevance analysis"""

    def test_clinical_relevance(self, clustering_data, clinical_data):
        """Test clinical relevance analysis"""
        X, y = clustering_data

        results = clinical_relevance_analysis(
            y, clinical_data, clinical_variables=['symptom_severity', 'age']
        )

        assert len(results) > 0

        for result in results:
            assert result.p_value >= 0
            assert result.effect_size >= 0

    def test_continuous_variable(self, clustering_data, clinical_data):
        """Test continuous variable analysis"""
        X, y = clustering_data

        results = clinical_relevance_analysis(
            y, clinical_data, clinical_variables=['symptom_severity']
        )

        assert len(results) == 1
        assert 'symptom_severity' in [r.clinical_variable for r in results]

    def test_categorical_variable(self, clustering_data, clinical_data):
        """Test categorical variable analysis"""
        X, y = clustering_data

        results = clinical_relevance_analysis(
            y, clinical_data, clinical_variables=['diagnosis']
        )

        assert len(results) == 1
        assert 'diagnosis' in [r.clinical_variable for r in results]


class TestDiagnosticConcordance:
    """Tests for diagnostic concordance"""

    def test_diagnostic_concordance(self, clustering_data):
        """Test diagnostic concordance"""
        X, y = clustering_data

        # Create mock diagnoses that somewhat align with clusters
        diagnoses = np.array(['ADHD', 'ASD', 'Both', 'Neither'])[y]

        concordance = diagnostic_concordance(y, diagnoses)

        assert 'cluster_purity' in concordance
        assert 'diagnosis_purity' in concordance
        assert 'f_measure' in concordance

        assert 0 <= concordance['cluster_purity'] <= 1
        assert 0 <= concordance['f_measure'] <= 1


class TestFunctionalOutcome:
    """Tests for functional outcome prediction"""

    def test_functional_outcome(self, clustering_data):
        """Test functional outcome prediction"""
        X, y = clustering_data

        # Create mock outcomes correlated with clusters
        outcomes = y + np.random.randn(len(y)) * 0.3

        result = functional_outcome_prediction(y, outcomes)

        assert 'predictive_r2' in result
        assert 'kruskal_p_value' in result

        # Should have some predictive power
        assert result['predictive_r2'] > 0


class TestCrossSiteValidation:
    """Tests for cross-site validation"""

    def clustering_func(self, X):
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        return kmeans.fit_predict(X)

    def test_cross_site_validation(self, clustering_data):
        """Test leave-one-site-out validation"""
        X, y = clustering_data

        # Create mock site labels
        site_labels = np.array([i % 3 for i in range(len(y))])

        result = cross_site_validation(
            X, y, site_labels, self.clustering_func
        )

        assert result.ari_mean >= 0
        assert result.generalization_score >= 0


class TestStratifiedCV:
    """Tests for stratified cross-validation"""

    def clustering_func(self, X):
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        return kmeans.fit_predict(X)

    def test_stratified_cv(self, clustering_data):
        """Test stratified cross-validation"""
        X, y = clustering_data

        result = stratified_cross_validation(
            X, y, self.clustering_func, n_folds=3
        )

        assert len(result.fold_ari_scores) > 0
        assert result.ari_mean >= 0


class TestBatchEffectValidation:
    """Tests for batch effect validation"""

    def clustering_func(self, X):
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        return kmeans.fit_predict(X)

    def test_batch_effect_validation(self, clustering_data):
        """Test batch effect robustness"""
        X, y = clustering_data

        # Create mock batch labels
        batch_labels = np.array([i % 2 for i in range(len(y))])

        result = batch_effect_validation(
            X, y, batch_labels, self.clustering_func
        )

        assert 'within_batch_ari' in result
        assert 'cross_batch_ari' in result
        assert 'batch_robustness' in result


class TestEdgeCases:
    """Test edge cases"""

    def test_small_dataset(self):
        """Test with very small dataset"""
        X = np.random.randn(10, 5)
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])

        metrics = compute_internal_metrics(X, y)

        assert metrics is not None

    def test_high_dimensional(self):
        """Test with high-dimensional data"""
        X, y = make_blobs(n_samples=100, n_features=100, centers=3,
                          cluster_std=2.0, random_state=42)

        metrics = compute_internal_metrics(X, y)

        assert metrics is not None

    def test_all_noise(self):
        """Test with all noise labels"""
        X = np.random.randn(50, 5)
        y = np.full(50, -1)

        metrics = compute_internal_metrics(X, y)

        # Should return defaults
        assert metrics.silhouette == 0.0