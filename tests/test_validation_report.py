"""Tests for validation report generation"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from src.audhd_correlation.validation.report import (
    generate_validation_report,
    save_report,
    ValidationReport,
)


@pytest.fixture
def clustering_data():
    """Generate test clustering data"""
    X, y = make_blobs(n_samples=200, n_features=10, centers=4,
                      cluster_std=1.0, random_state=42)
    return X, y


@pytest.fixture
def clustering_func():
    """Simple clustering function"""
    def func(X):
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        return kmeans.fit_predict(X)
    return func


@pytest.fixture
def clinical_data(clustering_data):
    """Generate synthetic clinical data"""
    X, y = clustering_data

    symptom_severity = np.random.randn(len(y)) + y * 0.5
    age = np.random.randint(5, 18, len(y))
    diagnosis = np.array(['ADHD', 'ASD', 'Both', 'Neither'])[y]

    return pd.DataFrame({
        'symptom_severity': symptom_severity,
        'age': age,
        'diagnosis': diagnosis,
    })


class TestReportGeneration:
    """Tests for report generation"""

    def test_basic_report(self, clustering_data, clustering_func):
        """Test basic report generation"""
        X, y = clustering_data

        report = generate_validation_report(
            X, y,
            clustering_func=clustering_func,
            include_bootstrap=False,
            include_subsampling=False,
            include_noise=False,
            include_features=False,
            include_permutation=False,
            include_stratified_cv=False,
        )

        assert isinstance(report, ValidationReport)
        assert report.internal_metrics is not None
        assert report.stability_metrics is not None
        assert report.overall_quality is not None
        assert report.recommendations is not None

    def test_full_report(self, clustering_data, clustering_func):
        """Test full report with all components"""
        X, y = clustering_data

        report = generate_validation_report(
            X, y,
            clustering_func=clustering_func,
            include_bootstrap=True,
            include_subsampling=True,
            include_noise=True,
            include_features=True,
            include_permutation=True,
            include_stratified_cv=True,
            n_bootstrap=10,  # Small number for speed
        )

        assert report.bootstrap_result is not None
        assert report.subsampling_results is not None
        assert report.noise_results is not None
        assert report.feature_results is not None
        assert report.permutation_result is not None
        assert report.stratified_cv_result is not None

    def test_with_clinical_data(self, clustering_data, clustering_func, clinical_data):
        """Test report with clinical data"""
        X, y = clustering_data

        report = generate_validation_report(
            X, y,
            clustering_func=clustering_func,
            clinical_data=clinical_data,
            include_clinical=True,
            include_bootstrap=False,
            include_subsampling=False,
            include_noise=False,
            include_features=False,
            include_permutation=False,
            include_stratified_cv=False,
        )

        assert report.clinical_results is not None
        assert len(report.clinical_results) > 0

    def test_with_site_labels(self, clustering_data, clustering_func):
        """Test report with site labels"""
        X, y = clustering_data

        # Create mock site labels
        site_labels = np.array([i % 3 for i in range(len(y))])

        report = generate_validation_report(
            X, y,
            clustering_func=clustering_func,
            site_labels=site_labels,
            include_cross_site=True,
            include_bootstrap=False,
            include_subsampling=False,
            include_noise=False,
            include_features=False,
            include_permutation=False,
            include_stratified_cv=False,
        )

        assert report.cross_site_result is not None

    def test_recommendations(self, clustering_data, clustering_func):
        """Test that recommendations are generated"""
        X, y = clustering_data

        report = generate_validation_report(
            X, y,
            clustering_func=clustering_func,
            include_bootstrap=True,
            include_permutation=True,
            n_bootstrap=10,
        )

        assert report.recommendations is not None
        assert len(report.recommendations) > 0

    def test_overall_quality(self, clustering_data, clustering_func):
        """Test overall quality score"""
        X, y = clustering_data

        report = generate_validation_report(
            X, y,
            clustering_func=clustering_func,
            include_bootstrap=True,
            include_permutation=True,
            include_stratified_cv=True,
            n_bootstrap=10,
        )

        assert report.overall_quality is not None
        assert 0 <= report.overall_quality <= 1


class TestReportSaving:
    """Tests for report saving"""

    def test_save_json(self, clustering_data, clustering_func, tmp_path):
        """Test saving report as JSON"""
        X, y = clustering_data

        report = generate_validation_report(
            X, y,
            clustering_func=clustering_func,
            include_bootstrap=False,
            include_subsampling=False,
            include_noise=False,
            include_features=False,
            include_permutation=False,
            include_stratified_cv=False,
        )

        save_report(report, tmp_path, format='json')

        json_file = tmp_path / 'validation_report.json'
        assert json_file.exists()

    def test_save_text(self, clustering_data, clustering_func, tmp_path):
        """Test saving report as text"""
        X, y = clustering_data

        report = generate_validation_report(
            X, y,
            clustering_func=clustering_func,
            include_bootstrap=True,
            include_permutation=True,
            n_bootstrap=10,
        )

        save_report(report, tmp_path, format='txt')

        txt_file = tmp_path / 'validation_report.txt'
        assert txt_file.exists()

        # Check content
        content = txt_file.read_text()
        assert 'VALIDATION REPORT' in content
        assert 'INTERNAL METRICS' in content
        assert 'RECOMMENDATIONS' in content

    def test_save_html(self, clustering_data, clustering_func, tmp_path):
        """Test saving report as HTML"""
        X, y = clustering_data

        report = generate_validation_report(
            X, y,
            clustering_func=clustering_func,
            include_bootstrap=True,
            n_bootstrap=10,
        )

        save_report(report, tmp_path, format='html')

        html_file = tmp_path / 'validation_report.html'
        assert html_file.exists()

        # Check content
        content = html_file.read_text()
        assert '<html>' in content
        assert 'Validation Report' in content
        assert 'Overall Quality Score' in content


class TestEdgeCases:
    """Test edge cases"""

    def test_no_clustering_func(self, clustering_data):
        """Test report without clustering function"""
        X, y = clustering_data

        # Should still work but skip stability tests
        report = generate_validation_report(
            X, y,
            clustering_func=None,
            include_bootstrap=True,
            include_permutation=True,
        )

        assert report.internal_metrics is not None
        assert report.bootstrap_result is None
        assert report.permutation_result is None

    def test_small_dataset(self):
        """Test with very small dataset"""
        X = np.random.randn(20, 5)
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 2)

        def clustering_func(X):
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            return kmeans.fit_predict(X)

        report = generate_validation_report(
            X, y,
            clustering_func=clustering_func,
            include_bootstrap=True,
            n_bootstrap=5,
        )

        assert report is not None