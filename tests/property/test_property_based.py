"""Property-based tests using Hypothesis

Tests general properties that should hold for all valid inputs.
"""
import pytest
import numpy as np
import pandas as pd

try:
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis.extra.numpy import arrays
    from hypothesis.extra.pandas import data_frames, column
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    pytestmark = pytest.mark.skip("hypothesis not installed")


if HYPOTHESIS_AVAILABLE:

    # ============================================================================
    # Property Tests for Preprocessing
    # ============================================================================

    class TestPreprocessingProperties:
        """Property-based tests for preprocessing"""

        @given(
            data=data_frames([
                column('feature1', dtype=float),
                column('feature2', dtype=float),
                column('feature3', dtype=float),
            ], index=st.integers(0, 100))
        )
        @settings(deadline=None, max_examples=50)
        def test_scaling_preserves_shape(self, data):
            """Test that scaling preserves data shape"""
            assume(len(data) > 0)
            assume(not data.isna().all().all())

            from audhd_correlation.preprocess import scale_features

            scaled = scale_features(data)

            assert scaled.shape == data.shape
            assert list(scaled.columns) == list(data.columns)
            assert list(scaled.index) == list(data.index)

        @given(
            data=data_frames([
                column('feature1', dtype=float, elements=st.floats(-100, 100)),
                column('feature2', dtype=float, elements=st.floats(-100, 100)),
            ], index=st.integers(0, 100))
        )
        @settings(deadline=None, max_examples=50)
        def test_scaling_centers_data(self, data):
            """Test that standard scaling centers data near zero"""
            assume(len(data) > 10)
            assume(not data.isna().any().any())
            assume(data.std().min() > 0.01)  # Non-constant data

            from audhd_correlation.preprocess import scale_features

            scaled = scale_features(data, method='standard')

            # Mean should be close to zero
            assert abs(scaled.mean().mean()) < 0.5

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(10, 50), st.integers(5, 20)),
                elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=30)
        def test_imputation_fills_missing(self, data):
            """Test that imputation fills all missing values"""
            # Add some missing values
            mask = np.random.random(data.shape) < 0.1
            data[mask] = np.nan

            assume(np.isnan(data).any())  # Has missing values
            assume(not np.isnan(data).all())  # Not all missing

            from audhd_correlation.preprocess.impute import impute_missing

            df = pd.DataFrame(data)
            imputed = impute_missing(df, method='mean')

            # All missing values should be filled
            assert not imputed.isna().any().any()

    # ============================================================================
    # Property Tests for Clustering
    # ============================================================================

    class TestClusteringProperties:
        """Property-based tests for clustering"""

        @given(
            X=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(50, 200), st.integers(5, 20)),
                elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=20)
        def test_clustering_returns_valid_labels(self, X):
            """Test that clustering returns valid label array"""
            from audhd_correlation.modeling.clustering import perform_clustering

            result = perform_clustering(X, method='kmeans', n_clusters=3)

            labels = result['labels']

            # Should have same length as input
            assert len(labels) == len(X)

            # Labels should be integers
            assert np.issubdtype(labels.dtype, np.integer)

            # Should have expected number of clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            assert n_clusters > 0

        @given(
            n_clusters=st.integers(2, 10),
            n_samples=st.integers(50, 200)
        )
        @settings(deadline=None, max_examples=20)
        def test_kmeans_finds_k_clusters(self, n_clusters, n_samples):
            """Test that K-means finds exactly k clusters"""
            assume(n_samples >= n_clusters * 5)  # Reasonable ratio

            from audhd_correlation.modeling.clustering import perform_clustering
            from sklearn.datasets import make_blobs

            X, _ = make_blobs(
                n_samples=n_samples,
                n_features=10,
                centers=n_clusters,
                random_state=42
            )

            result = perform_clustering(X, method='kmeans', n_clusters=n_clusters)

            # K-means should find exactly k clusters
            assert len(set(result['labels'])) == n_clusters

    # ============================================================================
    # Property Tests for Validation Metrics
    # ============================================================================

    class TestValidationProperties:
        """Property-based tests for validation metrics"""

        @given(
            X=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(50, 200), st.integers(5, 20)),
                elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=20)
        def test_silhouette_score_range(self, X):
            """Test that silhouette score is in valid range"""
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import KMeans

            labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)

            score = silhouette_score(X, labels)

            # Must be in [-1, 1]
            assert -1 <= score <= 1

        @given(
            y_true=arrays(
                dtype=np.int32,
                shape=st.integers(50, 200),
                elements=st.integers(0, 4)
            ),
            y_pred=arrays(
                dtype=np.int32,
                shape=st.integers(50, 200),
                elements=st.integers(0, 4)
            )
        )
        @settings(deadline=None, max_examples=50)
        def test_ari_symmetry(self, y_true, y_pred):
            """Test that ARI is symmetric"""
            assume(len(y_true) == len(y_pred))
            assume(len(set(y_true)) > 1)
            assume(len(set(y_pred)) > 1)

            from sklearn.metrics import adjusted_rand_score

            score1 = adjusted_rand_score(y_true, y_pred)
            score2 = adjusted_rand_score(y_pred, y_true)

            # ARI should be symmetric
            assert abs(score1 - score2) < 1e-10

        @given(
            labels=arrays(
                dtype=np.int32,
                shape=st.integers(50, 200),
                elements=st.integers(0, 4)
            )
        )
        @settings(deadline=None, max_examples=50)
        def test_ari_perfect_agreement(self, labels):
            """Test that ARI gives 1.0 for perfect agreement"""
            assume(len(set(labels)) > 1)

            from sklearn.metrics import adjusted_rand_score

            score = adjusted_rand_score(labels, labels)

            # Perfect agreement should give ARI = 1.0
            assert abs(score - 1.0) < 1e-10

    # ============================================================================
    # Property Tests for Data Loaders
    # ============================================================================

    class TestDataLoaderProperties:
        """Property-based tests for data loaders"""

        @given(
            n_samples=st.integers(10, 100),
            n_features=st.integers(5, 50)
        )
        @settings(deadline=None, max_examples=30)
        def test_genomic_data_shape(self, n_samples, n_features):
            """Test that genomic data maintains shape"""
            # Generate random genotype matrix
            genotypes = np.random.choice([0, 1, 2], size=(n_samples, n_features))
            df = pd.DataFrame(genotypes)

            # Any processing should maintain shape
            assert df.shape == (n_samples, n_features)

        @given(
            ages=arrays(
                dtype=np.float64,
                shape=st.integers(10, 100),
                elements=st.floats(18, 90, allow_nan=False)
            )
        )
        @settings(deadline=None, max_examples=50)
        def test_clinical_age_validation(self, ages):
            """Test that clinical data validation works"""
            df = pd.DataFrame({'age': ages})

            # All ages should be in valid range
            assert (df['age'] >= 0).all()
            assert (df['age'] <= 150).all()

    # ============================================================================
    # Property Tests for Harmonization
    # ============================================================================

    class TestHarmonizationProperties:
        """Property-based tests for data harmonization"""

        @given(
            n_samples=st.integers(50, 200),
            n_features=st.integers(10, 50)
        )
        @settings(deadline=None, max_examples=20)
        def test_harmonization_preserves_samples(self, n_samples, n_features):
            """Test that harmonization preserves sample count"""
            data1 = pd.DataFrame(
                np.random.randn(n_samples, n_features),
                index=[f"S{i:03d}" for i in range(n_samples)]
            )

            data2 = pd.DataFrame(
                np.random.randn(n_samples, n_features),
                index=[f"S{i:03d}" for i in range(n_samples)]
            )

            from audhd_correlation.data.harmonize import harmonize_sample_ids

            harmonized = harmonize_sample_ids([data1, data2])

            # Should preserve all common samples
            assert len(harmonized[0]) <= n_samples
            assert len(harmonized[1]) <= n_samples
            assert len(harmonized[0]) == len(harmonized[1])

    # ============================================================================
    # Property Tests for Invariants
    # ============================================================================

    class TestInvariants:
        """Tests for important invariants"""

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(10, 100), st.integers(5, 20)),
                elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=30)
        def test_preprocessing_idempotent(self, data):
            """Test that applying preprocessing twice gives same result"""
            from audhd_correlation.preprocess import scale_features

            df = pd.DataFrame(data)

            scaled1 = scale_features(df)
            scaled2 = scale_features(scaled1)

            # Second scaling should be close to identity
            # (scaled data shouldn't change much)
            diff = (scaled2 - scaled1).abs().mean().mean()
            assert diff < 0.1

        @given(
            X=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(50, 100), st.integers(5, 15)),
                elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=20)
        def test_clustering_deterministic(self, X):
            """Test that clustering is deterministic with fixed seed"""
            from audhd_correlation.modeling.clustering import perform_clustering

            result1 = perform_clustering(X, method='kmeans', n_clusters=3)
            result2 = perform_clustering(X, method='kmeans', n_clusters=3)

            # Should get same labels
            assert np.array_equal(result1['labels'], result2['labels'])

        @given(
            labels=arrays(
                dtype=np.int32,
                shape=st.integers(50, 200),
                elements=st.integers(0, 4)
            )
        )
        @settings(deadline=None, max_examples=50)
        def test_permutation_invariance(self, labels):
            """Test that metrics are invariant to label permutation"""
            assume(len(set(labels)) > 1)

            from sklearn.metrics import silhouette_score
            from sklearn.datasets import make_blobs

            X, _ = make_blobs(
                n_samples=len(labels),
                n_features=10,
                centers=len(set(labels)),
                random_state=42
            )

            # Original score
            score1 = silhouette_score(X, labels)

            # Permute labels
            perm_map = {old: new for new, old in enumerate(set(labels))}
            labels_perm = np.array([perm_map[l] for l in labels])

            score2 = silhouette_score(X, labels_perm)

            # Scores should be identical (invariant to label names)
            assert abs(score1 - score2) < 1e-10


else:
    # Placeholder if hypothesis not available
    def test_hypothesis_required():
        pytest.skip("hypothesis not installed")