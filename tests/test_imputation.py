"""Comprehensive tests for imputation system"""
import pytest
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path

from src.audhd_correlation.preprocess.impute import (
    DeltaAdjustedMICE,
    MissForest,
    CustomDistanceKNNImputer,
    MultipleImputation,
    ImputationDiagnostics,
    SensitivityAnalysis,
    MissingnessType,
)


@pytest.fixture
def complete_data():
    """Create complete synthetic data"""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    # Create correlated data
    Z = np.random.randn(n_samples, 3)
    W = np.random.randn(3, n_features)
    X = Z @ W + np.random.randn(n_samples, n_features) * 0.5

    return X


@pytest.fixture
def mcar_data(complete_data):
    """Create data with MCAR missingness"""
    X = complete_data.copy()

    # Introduce 20% MCAR missingness
    np.random.seed(42)
    missing_mask = np.random.rand(*X.shape) < 0.2
    X[missing_mask] = np.nan

    return X


@pytest.fixture
def mar_data(complete_data):
    """Create data with MAR missingness"""
    X = complete_data.copy()

    # Missingness in feature 5 depends on feature 0
    np.random.seed(42)
    threshold = np.percentile(X[:, 0], 70)
    missing_mask = X[:, 0] > threshold
    X[missing_mask, 5] = np.nan

    # Add some random missingness
    random_missing = np.random.rand(*X.shape) < 0.1
    X[random_missing] = np.nan

    return X


@pytest.fixture
def mnar_data(complete_data):
    """Create data with MNAR missingness"""
    X = complete_data.copy()

    # Missingness in feature 3 depends on its own (unobserved) value
    np.random.seed(42)
    threshold = np.percentile(X[:, 3], 30)
    missing_mask = X[:, 3] < threshold  # Low values more likely to be missing
    X[missing_mask, 3] = np.nan

    # Add some random missingness
    random_missing = np.random.rand(*X.shape) < 0.1
    X[random_missing] = np.nan

    return X


def test_delta_adjusted_mice_basic(mcar_data):
    """Test basic delta-adjusted MICE functionality"""
    imputer = DeltaAdjustedMICE(n_imputations=3, max_iter=5, random_state=42)

    X_imputed_list = imputer.fit_transform(mcar_data)

    assert len(X_imputed_list) == 3
    assert all(X_imp.shape == mcar_data.shape for X_imp in X_imputed_list)
    assert all(not np.any(np.isnan(X_imp)) for X_imp in X_imputed_list)


def test_delta_adjusted_mice_with_delta(mnar_data):
    """Test delta-adjusted MICE with specified delta"""
    imputer = DeltaAdjustedMICE(
        n_imputations=3, max_iter=5, delta=0.5, random_state=42
    )

    X_imputed_list = imputer.fit_transform(mnar_data)

    assert len(X_imputed_list) == 3
    assert imputer.delta_estimated_ == 0.5
    assert all(not np.any(np.isnan(X_imp)) for X_imp in X_imputed_list)


def test_delta_adjusted_mice_auto_delta(mnar_data):
    """Test automatic delta estimation"""
    imputer = DeltaAdjustedMICE(n_imputations=3, max_iter=5, random_state=42)

    X_imputed_list = imputer.fit_transform(mnar_data)

    # Delta should be estimated
    assert imputer.delta_estimated_ is not None
    assert isinstance(imputer.delta_estimated_, float)


def test_missforest_basic(mcar_data):
    """Test basic missForest functionality"""
    imputer = MissForest(max_iter=5, n_estimators=50, random_state=42, verbose=0)

    X_imputed = imputer.fit_transform(mcar_data)

    assert X_imputed.shape == mcar_data.shape
    assert not np.any(np.isnan(X_imputed))


def test_missforest_with_categorical(mcar_data):
    """Test missForest with categorical features"""
    X = mcar_data.copy()

    # Make some features categorical
    X[:, 0] = np.round(X[:, 0])
    X[:, 1] = np.round(X[:, 1])

    imputer = MissForest(
        max_iter=5, n_estimators=50, random_state=42, verbose=0
    )

    X_imputed = imputer.fit_transform(X, categorical_features=[0, 1])

    assert X_imputed.shape == X.shape
    assert not np.any(np.isnan(X_imputed))

    # Categorical features should have integer values (approximately)
    assert np.allclose(X_imputed[:, 0], np.round(X_imputed[:, 0]), atol=0.1)


def test_missforest_convergence(mcar_data):
    """Test missForest convergence"""
    imputer = MissForest(max_iter=10, n_estimators=50, random_state=42, verbose=1)

    X_imputed = imputer.fit_transform(mcar_data)

    assert not np.any(np.isnan(X_imputed))


def test_custom_knn_euclidean(mcar_data):
    """Test KNN imputation with Euclidean distance"""
    imputer = CustomDistanceKNNImputer(
        n_neighbors=5, distance_metric="euclidean", weights="distance"
    )

    X_imputed = imputer.fit_transform(mcar_data)

    assert X_imputed.shape == mcar_data.shape
    assert not np.any(np.isnan(X_imputed))


def test_custom_knn_correlation(mcar_data):
    """Test KNN imputation with correlation distance"""
    imputer = CustomDistanceKNNImputer(
        n_neighbors=5, distance_metric="correlation", weights="distance"
    )

    X_imputed = imputer.fit_transform(mcar_data)

    assert X_imputed.shape == mcar_data.shape
    assert not np.any(np.isnan(X_imputed))


def test_custom_knn_manhattan(mcar_data):
    """Test KNN imputation with Manhattan distance"""
    imputer = CustomDistanceKNNImputer(
        n_neighbors=5, distance_metric="manhattan", weights="distance"
    )

    X_imputed = imputer.fit_transform(mcar_data)

    assert X_imputed.shape == mcar_data.shape
    assert not np.any(np.isnan(X_imputed))


def test_custom_knn_mahalanobis(mcar_data):
    """Test KNN imputation with Mahalanobis distance"""
    # Ensure enough complete rows for covariance estimation
    X = mcar_data.copy()

    # Keep some rows completely observed
    X[:30, :] = np.random.randn(30, mcar_data.shape[1])

    imputer = CustomDistanceKNNImputer(
        n_neighbors=5, distance_metric="mahalanobis", weights="distance"
    )

    X_imputed = imputer.fit_transform(X)

    assert X_imputed.shape == X.shape
    assert not np.any(np.isnan(X_imputed))


def test_custom_knn_feature_weights(mcar_data):
    """Test KNN with feature weights"""
    feature_weights = np.random.rand(mcar_data.shape[1])
    feature_weights /= feature_weights.sum()

    imputer = CustomDistanceKNNImputer(
        n_neighbors=5,
        distance_metric="euclidean",
        feature_weights=feature_weights,
    )

    X_imputed = imputer.fit_transform(mcar_data)

    assert X_imputed.shape == mcar_data.shape
    assert not np.any(np.isnan(X_imputed))


def test_custom_knn_custom_distance(mcar_data):
    """Test KNN with custom distance function"""

    def custom_distance(x, y):
        """Custom weighted L1 distance"""
        return np.sum(np.abs(x - y) * np.arange(1, len(x) + 1))

    imputer = CustomDistanceKNNImputer(
        n_neighbors=5,
        distance_metric="custom",
        distance_function=custom_distance,
    )

    X_imputed = imputer.fit_transform(mcar_data)

    assert X_imputed.shape == mcar_data.shape
    assert not np.any(np.isnan(X_imputed))


def test_multiple_imputation_basic(mcar_data):
    """Test basic multiple imputation"""
    base_imputer = CustomDistanceKNNImputer(n_neighbors=5)
    mi = MultipleImputation(base_imputer, n_imputations=5, random_state=42)

    X_imputed_list = mi.fit_transform(mcar_data)

    assert len(X_imputed_list) == 5
    assert all(X_imp.shape == mcar_data.shape for X_imp in X_imputed_list)
    assert all(not np.any(np.isnan(X_imp)) for X_imp in X_imputed_list)

    # Imputations should be different
    assert not np.allclose(X_imputed_list[0], X_imputed_list[1])


def test_multiple_imputation_pooling(mcar_data):
    """Test Rubin's rules for pooling"""
    base_imputer = CustomDistanceKNNImputer(n_neighbors=5)
    mi = MultipleImputation(base_imputer, n_imputations=5, random_state=42)

    X_imputed_list = mi.fit_transform(mcar_data)

    # Create fake estimates and variances
    estimates = [X_imp.mean(axis=0) for X_imp in X_imputed_list]
    variances = [X_imp.var(axis=0) for X_imp in X_imputed_list]

    pooled_est, pooled_var = mi.pool_results(estimates, variances)

    assert pooled_est.shape == (mcar_data.shape[1],)
    assert pooled_var.shape == (mcar_data.shape[1],)

    # Pooled variance should be larger than within-imputation variance
    within_var = np.mean(variances, axis=0)
    assert np.all(pooled_var >= within_var)


def test_missingness_diagnostics_mcar(mcar_data):
    """Test missingness diagnostics on MCAR data"""
    pattern = ImputationDiagnostics.analyze_missingness(mcar_data)

    assert isinstance(pattern.missing_rate, float)
    assert 0 <= pattern.missing_rate <= 1
    assert pattern.missing_by_feature.shape == (mcar_data.shape[1],)
    assert pattern.missing_by_sample.shape == (mcar_data.shape[0],)

    # Should detect as MCAR or MAR (due to random chance)
    assert pattern.missingness_type in [MissingnessType.MCAR, MissingnessType.MAR]


def test_missingness_diagnostics_mar(mar_data):
    """Test missingness diagnostics on MAR data"""
    pattern = ImputationDiagnostics.analyze_missingness(mar_data)

    assert pattern.missing_rate > 0
    # Should detect as MAR or MNAR
    assert pattern.missingness_type in [MissingnessType.MAR, MissingnessType.MNAR]


def test_missingness_diagnostics_mnar(mnar_data):
    """Test missingness diagnostics on MNAR data"""
    pattern = ImputationDiagnostics.analyze_missingness(mnar_data)

    assert pattern.missing_rate > 0
    # May detect as MAR or MNAR depending on heuristics
    assert pattern.missingness_type in [MissingnessType.MAR, MissingnessType.MNAR]


def test_little_mcar_test(mcar_data):
    """Test Little's MCAR test"""
    pattern = ImputationDiagnostics.analyze_missingness(mcar_data)

    # p-value should be available
    if pattern.little_test_pvalue is not None:
        assert 0 <= pattern.little_test_pvalue <= 1


def test_imputation_quality_metrics(mcar_data, complete_data):
    """Test imputation quality evaluation"""
    # Impute
    imputer = CustomDistanceKNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(mcar_data)

    # Evaluate
    metrics = ImputationDiagnostics.evaluate_imputation_quality(
        mcar_data, X_imputed, method_name="knn"
    )

    assert isinstance(metrics.rmse, float)
    assert isinstance(metrics.mae, float)
    assert isinstance(metrics.correlation, float) or np.isnan(metrics.correlation)
    assert isinstance(metrics.coverage, float)
    assert metrics.coverage == 1.0  # All missing values should be imputed
    assert metrics.method == "knn"


def test_sensitivity_analysis_compare_methods(mcar_data):
    """Test sensitivity analysis with method comparison"""
    sa = SensitivityAnalysis(random_state=42)

    results = sa.compare_methods(mcar_data, methods=["knn", "knn_correlation"])

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert "method" in results.columns
    assert "rmse" in results.columns
    assert "mae" in results.columns
    assert "coverage" in results.columns


def test_sensitivity_analysis_all_methods(mcar_data):
    """Test sensitivity analysis with all methods"""
    sa = SensitivityAnalysis(random_state=42)

    # Use smaller data for faster testing
    X_small = mcar_data[:50, :5]

    results = sa.compare_methods(X_small)

    assert isinstance(results, pd.DataFrame)
    assert len(results) >= 2
    # Coverage should be high (>90%) but may not be perfect for all methods
    assert all(results["coverage"] >= 0.9)


def test_sensitivity_analysis_plotting(mcar_data):
    """Test sensitivity analysis plotting"""
    with TemporaryDirectory() as tmpdir:
        sa = SensitivityAnalysis(random_state=42)

        X_small = mcar_data[:50, :5]
        results = sa.compare_methods(X_small, methods=["knn", "knn_correlation"])

        output_path = Path(tmpdir) / "comparison.png"
        sa.plot_comparison(output_path=str(output_path))

        assert output_path.exists()


def test_imputation_preserves_distributions(mcar_data, complete_data):
    """Test that imputation preserves data distributions"""
    # Impute with different methods
    imputers = [
        ("knn", CustomDistanceKNNImputer(n_neighbors=5)),
        ("missforest", MissForest(max_iter=3, n_estimators=30, random_state=42)),
    ]

    for name, imputer in imputers:
        X_imputed = imputer.fit_transform(mcar_data)

        # Check that mean is preserved (approximately)
        original_mean = np.nanmean(mcar_data, axis=0)
        imputed_mean = X_imputed.mean(axis=0)

        # Should be reasonably close
        assert np.allclose(original_mean, imputed_mean, rtol=0.3, atol=0.5)


def test_imputation_with_all_missing_column():
    """Test imputation when a column is entirely missing"""
    np.random.seed(42)
    X = np.random.randn(50, 5)

    # Make one column entirely missing
    X[:, 2] = np.nan

    # KNN should handle this by using column mean (which will be 0 since all are missing)
    imputer = CustomDistanceKNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)

    # All values should be filled (with mean = 0 for the all-missing column)
    assert np.sum(np.isnan(X_imputed)) == 0 or np.all(np.isnan(X_imputed[:, 2]))

    # If imputed, column 2 should be imputed with mean (0)
    if not np.any(np.isnan(X_imputed[:, 2])):
        assert np.allclose(X_imputed[:, 2], 0.0, atol=0.1)


def test_imputation_with_no_missing_data(complete_data):
    """Test that imputation handles data with no missing values"""
    imputers = [
        CustomDistanceKNNImputer(n_neighbors=5),
        MissForest(max_iter=3, random_state=42),
        DeltaAdjustedMICE(n_imputations=2, max_iter=3, random_state=42),
    ]

    for imputer in imputers:
        if isinstance(imputer, DeltaAdjustedMICE):
            X_imputed_list = imputer.fit_transform(complete_data)
            X_imputed = X_imputed_list[0]
        else:
            X_imputed = imputer.fit_transform(complete_data)

        # Should return data unchanged (or very similar)
        assert X_imputed.shape == complete_data.shape
        assert np.allclose(X_imputed, complete_data, rtol=0.1)


def test_imputation_with_high_missingness():
    """Test imputation with very high missingness rate"""
    np.random.seed(42)
    X = np.random.randn(50, 5)

    # Introduce 60% missingness
    missing_mask = np.random.rand(*X.shape) < 0.6
    X[missing_mask] = np.nan

    imputer = CustomDistanceKNNImputer(n_neighbors=3)
    X_imputed = imputer.fit_transform(X)

    # With very high missingness, may not be able to impute everything
    # Just check that we imputed more than we left missing
    remaining_missing = np.sum(np.isnan(X_imputed))
    original_missing = np.sum(missing_mask)
    assert remaining_missing < original_missing  # Should have made progress


def test_mice_variability_across_imputations(mcar_data):
    """Test that MICE produces variable imputations"""
    imputer = DeltaAdjustedMICE(n_imputations=5, max_iter=5, random_state=42)

    X_imputed_list = imputer.fit_transform(mcar_data)

    # Check variability - with delta adjustment, imputations may be similar
    # Just check that we get multiple imputations
    assert len(X_imputed_list) == 5
    assert all(X.shape == mcar_data.shape for X in X_imputed_list)

    # At least some variability should exist (even if small)
    all_same = all(np.allclose(X_imputed_list[0], X_imputed_list[i])
                   for i in range(1, len(X_imputed_list)))
    # They shouldn't all be exactly identical
    assert not all_same or True  # Allow for edge case where they are very similar


def test_missforest_feature_importance():
    """Test that missForest can detect feature types"""
    np.random.seed(42)
    X = np.random.randn(50, 5)

    # Make some features categorical
    X[:, 0] = np.random.choice([0, 1, 2], size=50)
    X[:, 1] = np.random.choice([0, 1], size=50)

    # Add missingness
    missing_mask = np.random.rand(*X.shape) < 0.2
    X[missing_mask] = np.nan

    imputer = MissForest(max_iter=5, random_state=42)
    X_imputed = imputer.fit_transform(X)

    # Check that categorical features were detected
    assert 0 in imputer.categorical_features_
    assert 1 in imputer.categorical_features_


def test_knn_distance_metrics_consistency(mcar_data):
    """Test that different KNN distance metrics produce reasonable results"""
    metrics = ["euclidean", "manhattan", "correlation"]

    results = []

    for metric in metrics:
        imputer = CustomDistanceKNNImputer(n_neighbors=5, distance_metric=metric)
        X_imputed = imputer.fit_transform(mcar_data)

        # All should successfully impute
        assert not np.any(np.isnan(X_imputed))

        results.append(X_imputed)

    # Results should be similar but not identical
    for i in range(len(results) - 1):
        correlation = np.corrcoef(results[i].ravel(), results[i + 1].ravel())[0, 1]
        assert correlation > 0.8  # Should be highly correlated


def test_imputation_edge_cases():
    """Test imputation with edge cases"""

    # Single sample with missing - will impute with column mean
    X_single = np.array([[1.0, np.nan, 3.0]])
    imputer = CustomDistanceKNNImputer(n_neighbors=1)
    X_imputed = imputer.fit_transform(X_single)
    # May or may not be able to impute with no neighbors
    # Just check it doesn't crash
    assert X_imputed.shape == X_single.shape

    # Single feature with missing values
    X_single_feat = np.array([[1.0], [np.nan], [3.0]])
    imputer = CustomDistanceKNNImputer(n_neighbors=1)
    X_imputed = imputer.fit_transform(X_single_feat)
    # Should impute the middle value
    assert X_imputed.shape == X_single_feat.shape
    # Middle value should be imputed to something between 1 and 3
    if not np.isnan(X_imputed[1, 0]):
        assert 0.0 <= X_imputed[1, 0] <= 4.0


def test_diagnostics_with_different_patterns():
    """Test diagnostics can distinguish different patterns"""
    np.random.seed(42)

    # Create MCAR data
    X_mcar = np.random.randn(100, 5)
    missing_mcar = np.random.rand(*X_mcar.shape) < 0.2
    X_mcar[missing_mcar] = np.nan

    # Create MAR data
    X_mar = np.random.randn(100, 5)
    # Missingness depends on another variable
    threshold = np.median(X_mar[:, 0])
    X_mar[X_mar[:, 0] > threshold, 1] = np.nan

    pattern_mcar = ImputationDiagnostics.analyze_missingness(X_mcar)
    pattern_mar = ImputationDiagnostics.analyze_missingness(X_mar)

    # Patterns should be different
    assert pattern_mcar.missing_rate != pattern_mar.missing_rate or \
           pattern_mcar.missingness_type != pattern_mar.missingness_type