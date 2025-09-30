"""Property-based tests for preprocessing invariants

These tests use Hypothesis to verify that preprocessing operations preserve
critical mathematical properties and don't introduce artifacts.
"""
import pytest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis.extra.numpy import arrays
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    pytestmark = pytest.mark.skip("hypothesis not installed")


if HYPOTHESIS_AVAILABLE:

    # ==========================================================================
    # Transformation Properties
    # ==========================================================================

    class TestTransformationInvariants:
        """Test that transformations preserve key properties"""

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(10, 100), st.integers(5, 20)),
                elements=st.floats(0.1, 1000.0, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=50)
        def test_log_transform_preserves_rank_order(self, data):
            """Log2 transform must preserve rank order within each feature"""
            # Log transform
            log_data = np.log2(data + 1)

            # Check rank order preserved for each feature
            for col in range(data.shape[1]):
                original_ranks = data[:, col].argsort()
                log_ranks = log_data[:, col].argsort()

                # Ranks should be identical
                assert np.array_equal(original_ranks, log_ranks), \
                    "Log transform changed rank order"

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(10, 100), st.integers(5, 20)),
                elements=st.floats(0.1, 1000.0, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=50)
        def test_log_transform_monotonic(self, data):
            """Log2 transform is monotonically increasing"""
            log_data = np.log2(data + 1)

            # For each feature, check if x1 < x2 => log(x1) < log(x2)
            for col in range(data.shape[1]):
                sorted_idx = data[:, col].argsort()
                sorted_original = data[sorted_idx, col]
                sorted_log = log_data[sorted_idx, col]

                # Log of sorted values should also be sorted
                assert np.all(np.diff(sorted_log) >= 0), \
                    "Log transform not monotonic"

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(10, 50), st.integers(5, 15)),
                elements=st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=30)
        def test_clr_transform_preserves_relative_order(self, data):
            """CLR transform preserves relative ordering"""
            from skbio.stats.composition import clr

            # Add small pseudocount and make compositional
            data_comp = data / data.sum(axis=1, keepdims=True)

            # CLR transform
            clr_data = clr(data_comp + 1e-6)

            # Check rank correlation is very high
            for col in range(data.shape[1]):
                corr, _ = spearmanr(data_comp[:, col], clr_data[:, col])
                assert corr > 0.95, "CLR changed relative order significantly"

    # ==========================================================================
    # Imputation Properties
    # ==========================================================================

    class TestImputationInvariants:
        """Test that imputation preserves non-missing values"""

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(20, 100), st.integers(5, 20)),
                elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=30)
        def test_imputation_preserves_non_missing(self, data):
            """Imputation must not change non-missing values"""
            # Create missing mask (10% missing)
            mask = np.random.random(data.shape) < 0.1
            data_with_missing = data.copy()
            data_with_missing[mask] = np.nan

            assume(np.isnan(data_with_missing).any())  # Has missing
            assume(not np.isnan(data_with_missing).all())  # Not all missing

            # Impute using mean
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            imputed = imputer.fit_transform(data_with_missing)

            # Check non-missing values unchanged
            non_missing_mask = ~mask
            np.testing.assert_array_almost_equal(
                data[non_missing_mask],
                imputed[non_missing_mask],
                decimal=10,
                err_msg="Imputation changed non-missing values"
            )

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(20, 100), st.integers(5, 20)),
                elements=st.floats(0, 100, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=30)
        def test_knn_imputation_within_range(self, data):
            """KNN imputation should not create values outside observed range"""
            # Create missing mask
            mask = np.random.random(data.shape) < 0.2
            data_with_missing = data.copy()
            data_with_missing[mask] = np.nan

            assume(np.isnan(data_with_missing).any())
            assume(not np.isnan(data_with_missing).all())

            # KNN impute
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            imputed = imputer.fit_transform(data_with_missing)

            # Check imputed values within observed range per feature
            for col in range(data.shape[1]):
                observed_min = np.nanmin(data_with_missing[:, col])
                observed_max = np.nanmax(data_with_missing[:, col])

                # All imputed values should be in [min, max]
                assert np.all(imputed[:, col] >= observed_min - 1e-6), \
                    f"KNN imputed value below observed minimum in column {col}"
                assert np.all(imputed[:, col] <= observed_max + 1e-6), \
                    f"KNN imputed value above observed maximum in column {col}"

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(20, 50), st.integers(5, 15)),
                elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=30)
        def test_imputation_deterministic_with_seed(self, data):
            """Imputation with fixed seed should be deterministic"""
            # Create missing mask
            np.random.seed(42)
            mask = np.random.random(data.shape) < 0.15
            data_with_missing = data.copy()
            data_with_missing[mask] = np.nan

            assume(np.isnan(data_with_missing).any())

            # Impute twice with same seed
            from sklearn.impute import KNNImputer

            np.random.seed(42)
            imputer1 = KNNImputer(n_neighbors=5)
            imputed1 = imputer1.fit_transform(data_with_missing)

            np.random.seed(42)
            imputer2 = KNNImputer(n_neighbors=5)
            imputed2 = imputer2.fit_transform(data_with_missing)

            # Should be identical
            np.testing.assert_array_equal(imputed1, imputed2,
                err_msg="Imputation not deterministic with fixed seed")

    # ==========================================================================
    # Scaling Properties
    # ==========================================================================

    class TestScalingInvariants:
        """Test that scaling preserves rank order"""

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(10, 100), st.integers(5, 20)),
                elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=50)
        def test_standard_scaling_preserves_rank(self, data):
            """Standard scaling preserves rank order within features"""
            assume(np.std(data, axis=0).min() > 0.01)  # Non-constant features

            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)

            # Check rank order preserved for each feature
            for col in range(data.shape[1]):
                original_ranks = data[:, col].argsort()
                scaled_ranks = scaled[:, col].argsort()

                assert np.array_equal(original_ranks, scaled_ranks), \
                    f"Standard scaling changed rank order in column {col}"

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(10, 100), st.integers(5, 20)),
                elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=50)
        def test_robust_scaling_preserves_rank(self, data):
            """Robust scaling preserves rank order"""
            assume(np.std(data, axis=0).min() > 0.01)

            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
            scaled = scaler.fit_transform(data)

            # Check rank order preserved
            for col in range(data.shape[1]):
                corr, _ = spearmanr(data[:, col], scaled[:, col])
                assert corr > 0.999, \
                    f"Robust scaling changed rank order in column {col}"

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(10, 100), st.integers(5, 20)),
                elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=30)
        def test_standard_scaling_achieves_target(self, data):
            """Standard scaling achieves mean=0, std=1"""
            assume(np.std(data, axis=0).min() > 0.01)

            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)

            # Check mean ≈ 0, std ≈ 1
            means = scaled.mean(axis=0)
            stds = scaled.std(axis=0)

            np.testing.assert_array_almost_equal(
                means,
                np.zeros_like(means),
                decimal=10,
                err_msg="Standard scaling did not achieve mean=0"
            )

            np.testing.assert_array_almost_equal(
                stds,
                np.ones_like(stds),
                decimal=10,
                err_msg="Standard scaling did not achieve std=1"
            )

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(10, 50), st.integers(5, 15)),
                elements=st.floats(0, 100, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=30)
        def test_minmax_scaling_achieves_range(self, data):
            """MinMax scaling achieves range [0, 1]"""
            assume(np.ptp(data, axis=0).min() > 0.01)  # Non-constant

            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data)

            # Check min=0, max=1
            mins = scaled.min(axis=0)
            maxs = scaled.max(axis=0)

            np.testing.assert_array_almost_equal(
                mins,
                np.zeros_like(mins),
                decimal=10,
                err_msg="MinMax scaling did not achieve min=0"
            )

            np.testing.assert_array_almost_equal(
                maxs,
                np.ones_like(maxs),
                decimal=10,
                err_msg="MinMax scaling did not achieve max=1"
            )

    # ==========================================================================
    # Batch Correction Properties
    # ==========================================================================

    class TestBatchCorrectionInvariants:
        """Test that batch correction reduces batch effects without over-smoothing"""

        @given(
            n_samples_per_batch=st.integers(20, 50),
            n_features=st.integers(10, 30)
        )
        @settings(deadline=None, max_examples=20)
        def test_batch_correction_reduces_batch_variance(self, n_samples_per_batch, n_features):
            """Batch correction should reduce variance explained by batch"""
            from sklearn.decomposition import PCA
            from sklearn.metrics import r2_score

            # Create data with batch effects
            n_batches = 3
            data_list = []
            batch_labels = []

            for batch_id in range(n_batches):
                # Add batch-specific shift
                batch_data = np.random.randn(n_samples_per_batch, n_features)
                batch_data += batch_id * 2.0  # Batch effect

                data_list.append(batch_data)
                batch_labels.extend([batch_id] * n_samples_per_batch)

            data = np.vstack(data_list)
            batch_labels = np.array(batch_labels)

            # PCA on original data
            pca_before = PCA(n_components=1)
            pc1_before = pca_before.fit_transform(data).ravel()

            # Variance explained by batch
            batch_one_hot = pd.get_dummies(batch_labels).values
            r2_batch_before = r2_score(
                pc1_before,
                batch_one_hot @ np.linalg.lstsq(batch_one_hot, pc1_before, rcond=None)[0]
            )

            # Apply simple batch correction (center each batch)
            data_corrected = data.copy()
            for batch_id in range(n_batches):
                batch_mask = batch_labels == batch_id
                data_corrected[batch_mask] -= data_corrected[batch_mask].mean(axis=0)

            # PCA on corrected data
            pca_after = PCA(n_components=1)
            pc1_after = pca_after.fit_transform(data_corrected).ravel()

            # Variance explained by batch after correction
            r2_batch_after = r2_score(
                pc1_after,
                batch_one_hot @ np.linalg.lstsq(batch_one_hot, pc1_after, rcond=None)[0]
            )

            # Batch effect should be reduced
            assert r2_batch_after < r2_batch_before * 0.5, \
                f"Batch correction did not reduce batch variance: {r2_batch_before:.3f} -> {r2_batch_after:.3f}"

        def test_batch_correction_preserves_biological_signal(self):
            """Batch correction should not remove biological signal"""
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score

            # Create data with both batch effects and biological groups
            n_samples_per_group = 30
            n_features = 20

            data_list = []
            true_labels = []
            batch_labels = []

            # Group 0 with batch 0 and 1
            group0_batch0 = np.random.randn(n_samples_per_group, n_features) + [1, 0] * (n_features // 2)
            group0_batch1 = np.random.randn(n_samples_per_group, n_features) + [1, 0] * (n_features // 2) + 3.0  # Batch shift

            # Group 1 with batch 0 and 1
            group1_batch0 = np.random.randn(n_samples_per_group, n_features) + [-1, 0] * (n_features // 2)
            group1_batch1 = np.random.randn(n_samples_per_group, n_features) + [-1, 0] * (n_features // 2) + 3.0  # Batch shift

            data = np.vstack([group0_batch0, group0_batch1, group1_batch0, group1_batch1])
            true_labels = np.array([0] * (n_samples_per_group * 2) + [1] * (n_samples_per_group * 2))
            batch_labels = np.array([0] * n_samples_per_group + [1] * n_samples_per_group + [0] * n_samples_per_group + [1] * n_samples_per_group)

            # Silhouette before
            sil_before = silhouette_score(data, true_labels)

            # Simple batch correction
            data_corrected = data.copy()
            for batch_id in [0, 1]:
                batch_mask = batch_labels == batch_id
                data_corrected[batch_mask] -= data_corrected[batch_mask].mean(axis=0)

            # Silhouette after
            sil_after = silhouette_score(data_corrected, true_labels)

            # Biological signal should be mostly preserved
            assert sil_after > sil_before * 0.8, \
                f"Batch correction removed biological signal: {sil_before:.3f} -> {sil_after:.3f}"

    # ==========================================================================
    # End-to-End Pipeline Properties
    # ==========================================================================

    class TestPipelineInvariants:
        """Test properties of complete preprocessing pipeline"""

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(20, 50), st.integers(10, 20)),
                elements=st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=20)
        def test_pipeline_deterministic_with_seed(self, data):
            """Complete pipeline should be deterministic with fixed seed"""
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler

            # Add missing values
            np.random.seed(42)
            mask = np.random.random(data.shape) < 0.1
            data_with_missing = data.copy()
            data_with_missing[mask] = np.nan

            assume(np.isnan(data_with_missing).any())

            # Run pipeline twice with same seed
            def run_pipeline(data_input, seed):
                np.random.seed(seed)

                # Log transform
                data_log = np.log2(data_input + 1)

                # Impute
                imputer = SimpleImputer(strategy='mean')
                data_imputed = imputer.fit_transform(data_log)

                # Scale
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_imputed)

                return data_scaled

            result1 = run_pipeline(data_with_missing, 42)
            result2 = run_pipeline(data_with_missing, 42)

            # Should be identical
            np.testing.assert_array_equal(result1, result2,
                err_msg="Pipeline not deterministic with fixed seed")

        @given(
            data=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(20, 50), st.integers(10, 20)),
                elements=st.floats(1.0, 100.0, allow_nan=False, allow_infinity=False)
            )
        )
        @settings(deadline=None, max_examples=20)
        def test_pipeline_preserves_sample_relationships(self, data):
            """Pipeline should preserve relative relationships between samples"""
            assume(np.std(data, axis=0).min() > 0.1)

            from sklearn.preprocessing import StandardScaler
            from scipy.spatial.distance import pdist, squareform
            from scipy.stats import spearmanr

            # Compute pairwise distances before
            dist_before = squareform(pdist(data, metric='euclidean'))

            # Simple pipeline: log + scale
            data_log = np.log2(data + 1)

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_log)

            # Compute pairwise distances after
            dist_after = squareform(pdist(data_scaled, metric='euclidean'))

            # Rank correlation should be very high
            corr, _ = spearmanr(dist_before.ravel(), dist_after.ravel())

            assert corr > 0.9, \
                f"Pipeline changed sample relationships too much: correlation = {corr:.3f}"


else:
    # Placeholder if hypothesis not available
    def test_hypothesis_required():
        pytest.skip("hypothesis not installed")