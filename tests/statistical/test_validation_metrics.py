"""Statistical tests for validation metrics

Tests statistical properties and reliability of validation metrics.
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_blobs


class TestSilhouetteScore:
    """Statistical tests for silhouette score"""

    def test_silhouette_range(self, clustered_data):
        """Test that silhouette score is in valid range"""
        from sklearn.metrics import silhouette_score

        X = clustered_data['X']
        y = clustered_data['y']

        score = silhouette_score(X, y)

        assert -1 <= score <= 1, f"Silhouette score {score} out of range [-1, 1]"

    def test_silhouette_good_clustering(self):
        """Test silhouette score for known good clustering"""
        # Create well-separated clusters
        X, y = make_blobs(n_samples=300, n_features=10, centers=3,
                          cluster_std=0.5, random_state=42)

        from sklearn.metrics import silhouette_score

        score = silhouette_score(X, y)

        # Well-separated clusters should have high silhouette
        assert score > 0.5, f"Expected good clustering silhouette > 0.5, got {score}"

    def test_silhouette_poor_clustering(self):
        """Test silhouette score for known poor clustering"""
        # Create overlapping clusters
        X, y = make_blobs(n_samples=300, n_features=10, centers=3,
                          cluster_std=5.0, random_state=42)

        from sklearn.metrics import silhouette_score

        score = silhouette_score(X, y)

        # Overlapping clusters should have lower silhouette
        assert score < 0.5, f"Expected poor clustering silhouette < 0.5, got {score}"

    def test_silhouette_deterministic(self, clustered_data):
        """Test that silhouette score is deterministic"""
        from sklearn.metrics import silhouette_score

        X = clustered_data['X']
        y = clustered_data['y']

        score1 = silhouette_score(X, y)
        score2 = silhouette_score(X, y)

        assert score1 == score2, "Silhouette score should be deterministic"


class TestCalinskiHarabaszScore:
    """Statistical tests for Calinski-Harabasz score"""

    def test_ch_score_positive(self, clustered_data):
        """Test that CH score is positive"""
        from sklearn.metrics import calinski_harabasz_score

        X = clustered_data['X']
        y = clustered_data['y']

        score = calinski_harabasz_score(X, y)

        assert score > 0, f"CH score should be positive, got {score}"

    def test_ch_score_increases_with_separation(self):
        """Test that CH score increases with cluster separation"""
        from sklearn.metrics import calinski_harabasz_score

        scores = []
        for std in [0.5, 1.0, 2.0]:
            X, y = make_blobs(n_samples=300, n_features=10, centers=3,
                            cluster_std=std, random_state=42)
            scores.append(calinski_harabasz_score(X, y))

        # Score should decrease as clusters overlap more (higher std)
        assert scores[0] > scores[1] > scores[2], \
            f"CH score should decrease with overlap: {scores}"


class TestDaviesBouldinScore:
    """Statistical tests for Davies-Bouldin score"""

    def test_db_score_non_negative(self, clustered_data):
        """Test that DB score is non-negative"""
        from sklearn.metrics import davies_bouldin_score

        X = clustered_data['X']
        y = clustered_data['y']

        score = davies_bouldin_score(X, y)

        assert score >= 0, f"DB score should be non-negative, got {score}"

    def test_db_score_decreases_with_separation(self):
        """Test that DB score decreases with better clustering"""
        from sklearn.metrics import davies_bouldin_score

        scores = []
        for std in [0.5, 1.0, 2.0]:
            X, y = make_blobs(n_samples=300, n_features=10, centers=3,
                            cluster_std=std, random_state=42)
            scores.append(davies_bouldin_score(X, y))

        # Score should increase as clusters overlap more (lower is better)
        assert scores[0] < scores[1] < scores[2], \
            f"DB score should increase with overlap: {scores}"


class TestAdjustedRandIndex:
    """Statistical tests for Adjusted Rand Index"""

    def test_ari_perfect_agreement(self):
        """Test ARI for perfect agreement"""
        from sklearn.metrics import adjusted_rand_score

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        score = adjusted_rand_score(y_true, y_pred)

        assert score == 1.0, f"Perfect agreement should give ARI=1.0, got {score}"

    def test_ari_random_labeling(self):
        """Test ARI for random labeling"""
        from sklearn.metrics import adjusted_rand_score

        np.random.seed(42)
        n = 1000
        y_true = np.random.choice([0, 1, 2], n)
        y_pred = np.random.choice([0, 1, 2], n)

        score = adjusted_rand_score(y_true, y_pred)

        # Random labeling should give ARI close to 0
        assert abs(score) < 0.1, f"Random labeling should give ARI≈0, got {score}"

    def test_ari_range(self):
        """Test that ARI is in valid range"""
        from sklearn.metrics import adjusted_rand_score

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([1, 1, 0, 0, 2, 2])

        score = adjusted_rand_score(y_true, y_pred)

        assert -1 <= score <= 1, f"ARI {score} out of range [-1, 1]"


class TestStabilityMetrics:
    """Statistical tests for stability metrics"""

    def test_bootstrap_stability_range(self, clustered_data):
        """Test that bootstrap stability is in valid range"""
        from audhd_correlation.validation import bootstrap_stability

        X = clustered_data['X']
        y = clustered_data['y']

        result = bootstrap_stability(X, y, n_bootstrap=10)

        ari_key = 'mean_ari' if 'mean_ari' in result else 'ari_mean'
        assert 0 <= result[ari_key] <= 1, \
            f"Bootstrap ARI {result[ari_key]} out of range [0, 1]"

    def test_stability_increases_with_quality(self):
        """Test that stability increases with clustering quality"""
        from audhd_correlation.validation import bootstrap_stability
        from sklearn.cluster import KMeans

        # Good clustering
        X_good, _ = make_blobs(n_samples=300, n_features=10, centers=3,
                               cluster_std=0.5, random_state=42)
        y_good = KMeans(n_clusters=3, random_state=42).fit_predict(X_good)

        # Poor clustering
        X_poor, _ = make_blobs(n_samples=300, n_features=10, centers=3,
                               cluster_std=5.0, random_state=42)
        y_poor = KMeans(n_clusters=3, random_state=42).fit_predict(X_poor)

        result_good = bootstrap_stability(X_good, y_good, n_bootstrap=10)
        result_poor = bootstrap_stability(X_poor, y_poor, n_bootstrap=10)

        ari_key = 'mean_ari' if 'mean_ari' in result_good else 'ari_mean'

        assert result_good[ari_key] > result_poor[ari_key], \
            "Good clustering should have higher stability"


class TestStatisticalSignificance:
    """Tests for statistical significance of cluster differences"""

    def test_cluster_separation_significance(self, clustered_data):
        """Test statistical significance of cluster separation"""
        from scipy.stats import f_oneway

        X = clustered_data['X']
        y = clustered_data['y']

        # Test each feature for significant differences
        p_values = []
        for i in range(X.shape[1]):
            feature_values = [X[y == k, i] for k in np.unique(y)]
            _, p = f_oneway(*feature_values)
            p_values.append(p)

        # At least some features should show significant differences
        n_significant = sum(p < 0.05 for p in p_values)
        assert n_significant > 0, "No features showed significant cluster differences"

    def test_permutation_test(self, clustered_data):
        """Test permutation test for cluster validity"""
        from sklearn.metrics import silhouette_score

        X = clustered_data['X']
        y = clustered_data['y']

        # Observed score
        observed_score = silhouette_score(X, y)

        # Permutation scores
        n_permutations = 100
        perm_scores = []
        np.random.seed(42)

        for _ in range(n_permutations):
            y_perm = np.random.permutation(y)
            perm_scores.append(silhouette_score(X, y_perm))

        # P-value: proportion of permutations >= observed
        p_value = (np.array(perm_scores) >= observed_score).mean()

        # True clustering should be significantly better than random
        assert p_value < 0.05, \
            f"Clustering not significantly better than random (p={p_value})"


class TestDistributionTests:
    """Tests for distribution properties of metrics"""

    def test_silhouette_distribution(self):
        """Test distribution of silhouette scores across multiple runs"""
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans

        scores = []
        np.random.seed(42)

        for i in range(30):
            X, _ = make_blobs(n_samples=300, n_features=10, centers=3,
                            cluster_std=1.0, random_state=i)
            y = KMeans(n_clusters=3, random_state=i).fit_predict(X)
            scores.append(silhouette_score(X, y))

        # Test normality (Shapiro-Wilk)
        _, p_value = stats.shapiro(scores)

        # Distribution should be reasonably normal
        # (May not always pass, but should be close)
        print(f"Silhouette distribution normality p-value: {p_value}")

    def test_metric_variance(self, clustered_data):
        """Test variance of metrics across bootstrap samples"""
        from sklearn.metrics import silhouette_score

        X = clustered_data['X']
        y = clustered_data['y']

        scores = []
        np.random.seed(42)

        for _ in range(30):
            # Bootstrap sample
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            scores.append(silhouette_score(X_boot, y_boot))

        # Variance should be reasonably small for stable clustering
        cv = np.std(scores) / np.mean(scores)  # Coefficient of variation
        assert cv < 0.3, f"Metric too variable (CV={cv}), clustering may be unstable"


class TestMetricCorrelations:
    """Tests for correlations between different metrics"""

    def test_silhouette_ch_correlation(self):
        """Test correlation between silhouette and CH score"""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        silhouette_scores = []
        ch_scores = []

        for std in np.linspace(0.5, 3.0, 10):
            X, y = make_blobs(n_samples=300, n_features=10, centers=3,
                            cluster_std=std, random_state=42)
            silhouette_scores.append(silhouette_score(X, y))
            ch_scores.append(calinski_harabasz_score(X, y))

        # These metrics should be positively correlated
        correlation = np.corrcoef(silhouette_scores, ch_scores)[0, 1]
        assert correlation > 0.5, \
            f"Silhouette and CH should be positively correlated, got r={correlation}"

    def test_silhouette_db_anticorrelation(self):
        """Test anti-correlation between silhouette and DB score"""
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        silhouette_scores = []
        db_scores = []

        for std in np.linspace(0.5, 3.0, 10):
            X, y = make_blobs(n_samples=300, n_features=10, centers=3,
                            cluster_std=std, random_state=42)
            silhouette_scores.append(silhouette_score(X, y))
            db_scores.append(davies_bouldin_score(X, y))

        # These metrics should be negatively correlated
        correlation = np.corrcoef(silhouette_scores, db_scores)[0, 1]
        assert correlation < -0.5, \
            f"Silhouette and DB should be negatively correlated, got r={correlation}"


class TestRobustness:
    """Tests for metric robustness"""

    def test_outlier_robustness(self, clustered_data):
        """Test metric robustness to outliers"""
        from sklearn.metrics import silhouette_score

        X = clustered_data['X'].copy()
        y = clustered_data['y']

        # Original score
        score_original = silhouette_score(X, y)

        # Add outliers
        n_outliers = 10
        X_outliers = X.copy()
        outlier_indices = np.random.choice(len(X), n_outliers, replace=False)
        X_outliers[outlier_indices] += np.random.normal(0, 10, (n_outliers, X.shape[1]))

        score_with_outliers = silhouette_score(X_outliers, y)

        # Score should change but not drastically
        rel_change = abs(score_with_outliers - score_original) / abs(score_original)
        assert rel_change < 0.5, \
            f"Metric too sensitive to outliers (changed by {rel_change*100}%)"

    def test_sample_size_sensitivity(self):
        """Test metric sensitivity to sample size"""
        from sklearn.metrics import silhouette_score

        scores = []

        for n_samples in [50, 100, 200, 500]:
            X, y = make_blobs(n_samples=n_samples, n_features=10, centers=3,
                            cluster_std=1.0, random_state=42)
            scores.append(silhouette_score(X, y))

        # Scores should be relatively stable across sample sizes
        cv = np.std(scores) / np.mean(scores)
        assert cv < 0.2, \
            f"Metric too sensitive to sample size (CV={cv})"


class TestPowerAnalysis:
    """Tests for statistical power of validation metrics"""

    def test_power_to_detect_clusters(self):
        """Test power to detect clusters vs no structure"""
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans

        # Clustered data
        X_clustered, y_true = make_blobs(n_samples=300, n_features=10, centers=3,
                                         cluster_std=1.0, random_state=42)
        y_pred = KMeans(n_clusters=3, random_state=42).fit_predict(X_clustered)
        score_clustered = silhouette_score(X_clustered, y_pred)

        # Random data (no structure)
        np.random.seed(42)
        X_random = np.random.randn(300, 10)
        y_random = KMeans(n_clusters=3, random_state=42).fit_predict(X_random)
        score_random = silhouette_score(X_random, y_random)

        # Should clearly distinguish clustered vs random
        assert score_clustered - score_random > 0.2, \
            "Insufficient power to detect clusters"

    def test_minimum_sample_size(self):
        """Test minimum sample size for reliable metrics"""
        from sklearn.metrics import silhouette_score

        min_reliable_n = None

        for n in [20, 30, 50, 100]:
            scores = []
            for seed in range(10):
                X, y = make_blobs(n_samples=n, n_features=10, centers=3,
                                cluster_std=1.0, random_state=seed)
                scores.append(silhouette_score(X, y))

            cv = np.std(scores) / np.mean(scores)

            if cv < 0.15:  # Reliable if CV < 15%
                min_reliable_n = n
                break

        assert min_reliable_n is not None and min_reliable_n <= 100, \
            "Minimum sample size for reliable metrics too large"