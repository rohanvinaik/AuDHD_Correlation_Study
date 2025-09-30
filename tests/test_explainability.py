"""Comprehensive tests for explainability framework"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_blobs

from src.audhd_correlation.explainability.classifier import (
    train_cluster_classifier,
    get_top_features,
    predict_cluster,
    get_cluster_separability,
    analyze_misclassifications,
)

from src.audhd_correlation.explainability.shap_analysis import (
    compute_shap_values,
    get_top_features_per_cluster,
    get_feature_interactions,
    identify_cluster_prototypes,
    get_cluster_signature,
    explain_sample,
    compare_clusters,
)

from src.audhd_correlation.explainability.visualization import (
    plot_shap_waterfall,
    plot_shap_summary,
    plot_shap_beeswarm,
    plot_feature_importance,
    plot_shap_importance_comparison,
    plot_interaction_heatmap,
    plot_partial_dependence,
    plot_cluster_prototypes,
    plot_cluster_comparison,
)


@pytest.fixture
def clustering_data():
    """Generate test clustering data with clear structure"""
    np.random.seed(42)
    X, y = make_blobs(
        n_samples=100,  # Reduced for speed
        n_features=10,  # Reduced for speed
        centers=3,  # Reduced for speed
        cluster_std=1.5,
        random_state=42
    )
    return X, y


@pytest.fixture
def feature_names():
    """Generate feature names"""
    return [f"feature_{i}" for i in range(10)]


class TestClusterClassifier:
    """Tests for cluster classifier"""

    def test_train_classifier(self, clustering_data, feature_names):
        """Test basic classifier training"""
        X, y = clustering_data

        result = train_cluster_classifier(
            X, y,
            feature_names=feature_names,
            n_estimators=100,
            random_state=42
        )

        assert result.classifier is not None
        assert result.train_accuracy > 0.5
        assert result.cv_accuracy_mean > 0.3
        assert len(result.feature_importance) == len(feature_names)

    def test_feature_importance(self, clustering_data, feature_names):
        """Test feature importance extraction"""
        X, y = clustering_data

        result = train_cluster_classifier(
            X, y,
            feature_names=feature_names,
            n_estimators=100,
            random_state=42
        )

        # Check feature importance
        assert all(0 <= v <= 1 for v in result.feature_importance.values())
        assert abs(sum(result.feature_importance.values()) - 1.0) < 0.01

        # Get top features
        top_features = get_top_features(result, n_features=10)
        assert len(top_features) == 10
        assert all(isinstance(f[0], str) for f in top_features)
        assert all(isinstance(f[1], float) for f in top_features)

    def test_per_cluster_metrics(self, clustering_data, feature_names):
        """Test per-cluster performance metrics"""
        X, y = clustering_data

        result = train_cluster_classifier(
            X, y,
            feature_names=feature_names,
            n_estimators=100,
            random_state=42
        )

        # Should have metrics for each cluster
        assert len(result.per_cluster_precision) == len(np.unique(y))
        assert len(result.per_cluster_recall) == len(np.unique(y))
        assert len(result.per_cluster_f1) == len(np.unique(y))

        # All metrics should be between 0 and 1
        for cluster_id in result.per_cluster_precision:
            assert 0 <= result.per_cluster_precision[cluster_id] <= 1
            assert 0 <= result.per_cluster_recall[cluster_id] <= 1
            assert 0 <= result.per_cluster_f1[cluster_id] <= 1

    def test_prediction(self, clustering_data, feature_names):
        """Test prediction on new samples"""
        X, y = clustering_data

        # Split data
        X_train = X[:80]
        X_test = X[80:]
        y_train = y[:80]

        result = train_cluster_classifier(
            X_train, y_train,
            feature_names=feature_names,
            n_estimators=50,
            cv_folds=2,
            random_state=42
        )

        # Predict labels
        y_pred = predict_cluster(result, X_test, return_probabilities=False)
        assert len(y_pred) == len(X_test)
        assert all(pred in np.unique(y_train) for pred in y_pred)

        # Predict probabilities
        y_proba = predict_cluster(result, X_test, return_probabilities=True)
        assert y_proba.shape == (len(X_test), len(np.unique(y_train)))
        assert np.allclose(np.sum(y_proba, axis=1), 1.0)

    def test_cluster_separability(self, clustering_data, feature_names):
        """Test cluster separability metrics"""
        X, y = clustering_data

        result = train_cluster_classifier(
            X, y,
            feature_names=feature_names,
            n_estimators=100,
            random_state=42
        )

        separability = get_cluster_separability(result)

        assert 'accuracy' in separability
        assert 'cv_accuracy' in separability
        assert 'separability_score' in separability
        assert 0 <= separability['separability_score'] <= 1

    def test_misclassification_analysis(self, clustering_data, feature_names):
        """Test misclassification analysis"""
        X, y = clustering_data

        result = train_cluster_classifier(
            X, y,
            feature_names=feature_names,
            n_estimators=100,
            random_state=42
        )

        analysis = analyze_misclassifications(result, X)

        assert 'misclassified_indices' in analysis
        assert 'mean_confidence' in analysis
        assert 0 <= analysis['mean_confidence'] <= 1

    def test_imbalanced_clusters(self, feature_names):
        """Test with imbalanced clusters"""
        # Create imbalanced data
        X, y = make_blobs(n_samples=300, n_features=20, centers=3, random_state=42)

        # Make imbalanced
        mask = (y == 0) & (np.arange(len(y)) % 3 != 0)
        X = X[~mask]
        y = y[~mask]

        result = train_cluster_classifier(
            X, y,
            feature_names=feature_names,
            class_weight='balanced',
            n_estimators=100,
            random_state=42
        )

        assert result.train_accuracy > 0.3

    def test_noise_labels(self, feature_names):
        """Test handling of noise labels (-1)"""
        X, y = make_blobs(n_samples=300, n_features=20, centers=3, random_state=42)

        # Add noise points
        y[:20] = -1

        result = train_cluster_classifier(
            X, y,
            feature_names=feature_names,
            n_estimators=100,
            random_state=42
        )

        # Should only train on valid clusters
        assert result.train_accuracy > 0.3


class TestShapAnalysis:
    """Tests for SHAP analysis"""

    def test_compute_shap_values(self, clustering_data, feature_names):
        """Test SHAP value computation"""
        X, y = clustering_data

        # Use even smaller subset for SHAP
        X_small = X[:30]
        y_small = y[:30]

        # Train classifier
        classifier_result = train_cluster_classifier(
            X_small, y_small,
            feature_names=feature_names,
            n_estimators=20,  # Smaller for speed
            cv_folds=2,  # Fewer folds
            random_state=42
        )

        # Compute SHAP values
        shap_result = compute_shap_values(
            classifier_result,
            X_small,
            approximate=True  # Faster
        )

        assert shap_result.shap_values is not None
        assert shap_result.shap_values.shape[0] == len(X_small)
        assert shap_result.shap_values.shape[1] == X_small.shape[1]
        assert len(shap_result.feature_names) == X_small.shape[1]

    def test_top_features_per_cluster(self, clustering_data, feature_names):
        """Test top features extraction per cluster"""
        X, y = clustering_data
        X_small = X[:30]
        y_small = y[:30]

        classifier_result = train_cluster_classifier(
            X_small, y_small, feature_names=feature_names, n_estimators=20, cv_folds=2, random_state=42
        )

        shap_result = compute_shap_values(
            classifier_result, X_small, approximate=True
        )

        # Get top features for each cluster
        top_features = get_top_features_per_cluster(shap_result, n_features=10)

        assert isinstance(top_features, dict)
        assert len(top_features) > 0

        for cluster_id, features in top_features.items():
            assert len(features) == 10
            assert all(isinstance(f[0], str) for f in features)
            assert all(isinstance(f[1], float) for f in features)

    @pytest.mark.slow
    def test_feature_interactions(self, clustering_data, feature_names):
        """Test feature interaction computation (slow)"""
        X, y = clustering_data
        X_small = X[:20]
        y_small = y[:20]

        classifier_result = train_cluster_classifier(
            X_small, y_small, feature_names=feature_names, n_estimators=10, cv_folds=2, random_state=42
        )

        shap_result = compute_shap_values(
            classifier_result, X_small, approximate=True
        )

        cluster_id = int(np.unique(y_small)[0])

        interaction_df = get_feature_interactions(
            shap_result,
            cluster_id=cluster_id,
            n_top_features=3,
            sample_size=10
        )

        assert isinstance(interaction_df, pd.DataFrame)
        assert interaction_df.shape[0] == interaction_df.shape[1]

    def test_cluster_prototypes(self, clustering_data, feature_names):
        """Test cluster prototype identification"""
        X, y = clustering_data
        X_small = X[:30]
        y_small = y[:30]

        classifier_result = train_cluster_classifier(
            X_small, y_small, feature_names=feature_names, n_estimators=20, cv_folds=2, random_state=42
        )

        shap_result = compute_shap_values(
            classifier_result, X_small, approximate=True
        )

        cluster_id = int(np.unique(y)[0])

        # Test different methods
        for method in ['highest_probability', 'most_consistent', 'median']:
            prototypes = identify_cluster_prototypes(
                shap_result,
                cluster_id=cluster_id,
                n_prototypes=3,
                method=method
            )

            assert 'indices' in prototypes
            assert len(prototypes['indices']) <= 3
            assert 'shap_values' in prototypes
            assert 'data' in prototypes

    def test_cluster_signature(self, clustering_data, feature_names):
        """Test cluster signature extraction"""
        X, y = clustering_data

        classifier_result = train_cluster_classifier(
            X, y, feature_names=feature_names, n_estimators=50, random_state=42
        )

        shap_result = compute_shap_values(
            classifier_result, X, approximate=True
        )

        cluster_id = int(np.unique(y)[0])

        signature = get_cluster_signature(
            shap_result,
            cluster_id=cluster_id,
            n_features=10
        )

        assert 'positive' in signature
        assert 'negative' in signature
        assert isinstance(signature['positive'], dict)
        assert isinstance(signature['negative'], dict)

    def test_explain_sample(self, clustering_data, feature_names):
        """Test individual sample explanation"""
        X, y = clustering_data

        classifier_result = train_cluster_classifier(
            X, y, feature_names=feature_names, n_estimators=50, random_state=42
        )

        shap_result = compute_shap_values(
            classifier_result, X, approximate=True
        )

        explanation = explain_sample(
            shap_result,
            sample_index=0,
            n_features=10
        )

        assert 'sample_index' in explanation
        assert 'cluster_id' in explanation
        assert 'top_features' in explanation
        assert len(explanation['top_features']) == 10

        for feature in explanation['top_features']:
            assert 'feature' in feature
            assert 'value' in feature
            assert 'shap_value' in feature
            assert 'contribution' in feature

    def test_compare_clusters(self, clustering_data, feature_names):
        """Test cluster comparison"""
        X, y = clustering_data

        classifier_result = train_cluster_classifier(
            X, y, feature_names=feature_names, n_estimators=50, random_state=42
        )

        shap_result = compute_shap_values(
            classifier_result, X, approximate=True
        )

        unique_clusters = np.unique(y)
        cluster_a = int(unique_clusters[0])
        cluster_b = int(unique_clusters[1])

        comparison = compare_clusters(
            shap_result,
            cluster_a=cluster_a,
            cluster_b=cluster_b,
            n_features=15
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 15
        assert 'feature' in comparison.columns
        assert 'shap_difference' in comparison.columns


class TestVisualization:
    """Tests for visualization functions"""

    def test_feature_importance_plot(self, clustering_data, feature_names, tmp_path):
        """Test feature importance plot"""
        X, y = clustering_data

        classifier_result = train_cluster_classifier(
            X, y, feature_names=feature_names, n_estimators=50, random_state=42
        )

        fig = plot_feature_importance(
            classifier_result,
            n_features=15,
            show=False,
            save_path=tmp_path / 'feature_importance.png'
        )

        assert fig is not None
        assert (tmp_path / 'feature_importance.png').exists()

    def test_shap_waterfall(self, clustering_data, feature_names, tmp_path):
        """Test SHAP waterfall plot"""
        X, y = clustering_data

        # Use small subset
        X_small = X[:50]
        y_small = y[:50]

        classifier_result = train_cluster_classifier(
            X_small, y_small, feature_names=feature_names, n_estimators=20, random_state=42
        )

        shap_result = compute_shap_values(
            classifier_result, X_small, approximate=True
        )

        fig = plot_shap_waterfall(
            shap_result,
            sample_index=0,
            max_display=10,
            show=False,
            save_path=tmp_path / 'waterfall.png'
        )

        assert fig is not None
        assert (tmp_path / 'waterfall.png').exists()

    def test_shap_summary(self, clustering_data, feature_names, tmp_path):
        """Test SHAP summary plot"""
        X, y = clustering_data

        X_small = X[:50]
        y_small = y[:50]

        classifier_result = train_cluster_classifier(
            X_small, y_small, feature_names=feature_names, n_estimators=20, random_state=42
        )

        shap_result = compute_shap_values(
            classifier_result, X_small, approximate=True
        )

        cluster_id = int(np.unique(y_small)[0])

        fig = plot_shap_summary(
            shap_result,
            cluster_id=cluster_id,
            max_display=10,
            show=False,
            save_path=tmp_path / 'summary.png'
        )

        assert fig is not None
        assert (tmp_path / 'summary.png').exists()

    def test_shap_beeswarm(self, clustering_data, feature_names, tmp_path):
        """Test SHAP beeswarm plot"""
        X, y = clustering_data

        X_small = X[:50]
        y_small = y[:50]

        classifier_result = train_cluster_classifier(
            X_small, y_small, feature_names=feature_names, n_estimators=20, random_state=42
        )

        shap_result = compute_shap_values(
            classifier_result, X_small, approximate=True
        )

        cluster_id = int(np.unique(y_small)[0])

        fig = plot_shap_beeswarm(
            shap_result,
            cluster_id=cluster_id,
            max_display=10,
            show=False,
            save_path=tmp_path / 'beeswarm.png'
        )

        assert fig is not None
        assert (tmp_path / 'beeswarm.png').exists()

    def test_shap_importance_comparison(self, clustering_data, feature_names, tmp_path):
        """Test SHAP importance comparison plot"""
        X, y = clustering_data

        classifier_result = train_cluster_classifier(
            X, y, feature_names=feature_names, n_estimators=50, random_state=42
        )

        shap_result = compute_shap_values(
            classifier_result, X, approximate=True
        )

        fig = plot_shap_importance_comparison(
            shap_result,
            n_features=10,
            show=False,
            save_path=tmp_path / 'importance_comparison.png'
        )

        assert fig is not None
        assert (tmp_path / 'importance_comparison.png').exists()

    def test_partial_dependence_plot(self, clustering_data, feature_names, tmp_path):
        """Test partial dependence plot"""
        X, y = clustering_data

        classifier_result = train_cluster_classifier(
            X, y, feature_names=feature_names, n_estimators=50, random_state=42
        )

        cluster_id = int(np.unique(y)[0])

        fig = plot_partial_dependence(
            classifier_result,
            X,
            features=[0, 1, 2],
            cluster_id=cluster_id,
            show=False,
            save_path=tmp_path / 'partial_dependence.png'
        )

        assert fig is not None
        assert (tmp_path / 'partial_dependence.png').exists()

    def test_cluster_comparison_plot(self, clustering_data, feature_names, tmp_path):
        """Test cluster comparison plot"""
        X, y = clustering_data

        classifier_result = train_cluster_classifier(
            X, y, feature_names=feature_names, n_estimators=50, random_state=42
        )

        shap_result = compute_shap_values(
            classifier_result, X, approximate=True
        )

        unique_clusters = np.unique(y)
        cluster_a = int(unique_clusters[0])
        cluster_b = int(unique_clusters[1])

        comparison = compare_clusters(
            shap_result,
            cluster_a=cluster_a,
            cluster_b=cluster_b,
            n_features=15
        )

        fig = plot_cluster_comparison(
            comparison,
            cluster_a=cluster_a,
            cluster_b=cluster_b,
            n_features=15,
            show=False,
            save_path=tmp_path / 'comparison.png'
        )

        assert fig is not None
        assert (tmp_path / 'comparison.png').exists()


class TestEdgeCases:
    """Test edge cases"""

    def test_binary_classification(self):
        """Test with 2 clusters"""
        X, y = make_blobs(n_samples=200, n_features=10, centers=2, random_state=42)
        feature_names = [f"feature_{i}" for i in range(10)]

        result = train_cluster_classifier(
            X, y, feature_names=feature_names, n_estimators=50, random_state=42
        )

        assert result.train_accuracy > 0.5

    def test_small_dataset(self):
        """Test with small dataset"""
        X, y = make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)
        feature_names = [f"feature_{i}" for i in range(10)]

        result = train_cluster_classifier(
            X, y, feature_names=feature_names, n_estimators=20, random_state=42
        )

        assert result is not None

    def test_high_dimensional(self):
        """Test with high-dimensional data"""
        X, y = make_blobs(n_samples=200, n_features=50, centers=3, random_state=42)
        feature_names = [f"feature_{i}" for i in range(50)]

        result = train_cluster_classifier(
            X, y, feature_names=feature_names, n_estimators=50, random_state=42
        )

        assert result.train_accuracy > 0.3