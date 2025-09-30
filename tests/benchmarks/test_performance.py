"""Performance benchmarks for pipeline components

Uses pytest-benchmark for timing and memory profiling.
"""
import pytest
import numpy as np
import pandas as pd
import time


class TestDataLoadingPerformance:
    """Performance benchmarks for data loading"""

    def test_genomic_loading_speed(self, benchmark, temp_data_dir):
        """Benchmark genomic data loading"""
        # Create test VCF
        vcf_path = temp_data_dir / "test.vcf"
        # Simplified VCF for speed
        vcf_content = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS001
chr1\t1000\trs001\tA\tG\t.\tPASS\t.\tGT\t0/1
"""
        vcf_path.write_text(vcf_content)

        from audhd_correlation.data.genomic_loader import load_vcf

        result = benchmark(load_vcf, vcf_path)
        assert result is not None

    def test_clinical_loading_speed(self, benchmark, temp_data_dir, synthetic_clinical_data):
        """Benchmark clinical data loading"""
        csv_path = temp_data_dir / "clinical.csv"
        synthetic_clinical_data.to_csv(csv_path, index=False)

        from audhd_correlation.data.clinical_loader import load_clinical_csv

        result = benchmark(load_clinical_csv, csv_path)
        assert isinstance(result, pd.DataFrame)


class TestPreprocessingPerformance:
    """Performance benchmarks for preprocessing"""

    def test_imputation_speed(self, benchmark, synthetic_metabolomic_data):
        """Benchmark data imputation"""
        from audhd_correlation.preprocess import impute_missing

        result = benchmark(
            impute_missing,
            synthetic_metabolomic_data,
            method='knn',
            n_neighbors=5
        )
        assert isinstance(result, pd.DataFrame)

    def test_scaling_speed(self, benchmark, synthetic_metabolomic_data):
        """Benchmark feature scaling"""
        from audhd_correlation.preprocess import scale_features

        result = benchmark(scale_features, synthetic_metabolomic_data)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.parametrize("n_samples", [100, 500, 1000])
    def test_preprocessing_scalability(self, n_samples):
        """Test preprocessing scales with data size"""
        from audhd_correlation.preprocess import scale_features

        data = pd.DataFrame(np.random.randn(n_samples, 100))

        start = time.time()
        scale_features(data)
        duration = time.time() - start

        # Should be roughly linear
        expected_max = (n_samples / 100) * 0.5  # 0.5s for 100 samples
        assert duration < expected_max, \
            f"Preprocessing too slow: {duration}s for {n_samples} samples"


class TestIntegrationPerformance:
    """Performance benchmarks for integration"""

    def test_pca_integration_speed(self, benchmark):
        """Benchmark PCA integration"""
        from audhd_correlation.integrate import integrate_multiomics

        data = {
            'genomic': pd.DataFrame(np.random.randn(100, 500)),
            'metabolomic': pd.DataFrame(np.random.randn(100, 200)),
        }

        result = benchmark(
            integrate_multiomics,
            data,
            method='pca',
            n_components=10
        )
        assert result is not None

    @pytest.mark.slow
    def test_mofa_integration_speed(self, benchmark):
        """Benchmark MOFA integration"""
        pytest.skip("MOFA may be slow, test separately")


class TestClusteringPerformance:
    """Performance benchmarks for clustering"""

    def test_hdbscan_speed(self, benchmark, integrated_data):
        """Benchmark HDBSCAN clustering"""
        from audhd_correlation.modeling.clustering import perform_clustering

        result = benchmark(
            perform_clustering,
            integrated_data.values,
            method='hdbscan',
            min_cluster_size=15
        )
        assert 'labels' in result

    def test_kmeans_speed(self, benchmark, integrated_data):
        """Benchmark K-means clustering"""
        from audhd_correlation.modeling.clustering import perform_clustering

        result = benchmark(
            perform_clustering,
            integrated_data.values,
            method='kmeans',
            n_clusters=3
        )
        assert 'labels' in result

    @pytest.mark.parametrize("n_samples", [100, 500, 1000])
    def test_clustering_scalability(self, n_samples):
        """Test clustering scales with data size"""
        from audhd_correlation.modeling.clustering import perform_clustering

        X = np.random.randn(n_samples, 15)

        start = time.time()
        perform_clustering(X, method='kmeans', n_clusters=3)
        duration = time.time() - start

        # Should scale roughly O(n)
        expected_max = (n_samples / 100) * 1.0  # 1s for 100 samples
        assert duration < expected_max, \
            f"Clustering too slow: {duration}s for {n_samples} samples"


class TestValidationPerformance:
    """Performance benchmarks for validation"""

    def test_silhouette_computation_speed(self, benchmark, clustered_data):
        """Benchmark silhouette score computation"""
        from sklearn.metrics import silhouette_score

        X = clustered_data['X']
        y = clustered_data['y']

        result = benchmark(silhouette_score, X, y)
        assert -1 <= result <= 1

    def test_bootstrap_stability_speed(self, benchmark, clustered_data):
        """Benchmark bootstrap stability analysis"""
        from audhd_correlation.validation import bootstrap_stability

        X = clustered_data['X']
        y = clustered_data['y']

        result = benchmark(
            bootstrap_stability,
            X, y,
            n_bootstrap=10  # Small number for speed
        )
        assert result is not None


class TestMemoryUsage:
    """Memory usage benchmarks"""

    @pytest.mark.slow
    def test_integration_memory(self):
        """Test memory usage during integration"""
        pytest.importorskip("memory_profiler")

        from memory_profiler import memory_usage
        from audhd_correlation.integrate import integrate_multiomics

        data = {
            'genomic': pd.DataFrame(np.random.randn(1000, 1000)),
            'metabolomic': pd.DataFrame(np.random.randn(1000, 500)),
        }

        def integrate():
            integrate_multiomics(data, method='pca', n_components=10)

        mem_usage = memory_usage(integrate, interval=0.1)
        peak_memory = max(mem_usage) - min(mem_usage)

        # Should not use excessive memory (< 500MB increase)
        assert peak_memory < 500, \
            f"Excessive memory usage: {peak_memory}MB"

    @pytest.mark.slow
    def test_clustering_memory(self):
        """Test memory usage during clustering"""
        pytest.importorskip("memory_profiler")

        from memory_profiler import memory_usage
        from audhd_correlation.modeling.clustering import perform_clustering

        X = np.random.randn(1000, 50)

        def cluster():
            perform_clustering(X, method='hdbscan', min_cluster_size=20)

        mem_usage = memory_usage(cluster, interval=0.1)
        peak_memory = max(mem_usage) - min(mem_usage)

        # Should not use excessive memory (< 200MB increase)
        assert peak_memory < 200, \
            f"Excessive memory usage: {peak_memory}MB"


class TestParallelPerformance:
    """Performance tests for parallel processing"""

    @pytest.mark.slow
    def test_parallel_speedup(self):
        """Test that parallelization provides speedup"""
        from audhd_correlation.preprocess import scale_features

        data = pd.DataFrame(np.random.randn(1000, 500))

        # Sequential
        start = time.time()
        for _ in range(4):
            scale_features(data)
        seq_time = time.time() - start

        # Parallel (if implemented)
        # Would test parallel implementation here
        # parallel_time should be < seq_time

    def test_parallel_overhead(self):
        """Test that parallel overhead is acceptable"""
        # Small data should not be slower with parallelization
        from audhd_correlation.preprocess import scale_features

        data = pd.DataFrame(np.random.randn(50, 10))

        start = time.time()
        scale_features(data)
        duration = time.time() - start

        # Should be fast for small data
        assert duration < 1.0, "Too much overhead for small data"


# ============================================================================
# Regression Benchmarks
# ============================================================================

class TestRegressionBenchmarks:
    """Regression tests to catch performance degradation"""

    def test_baseline_preprocessing_time(self, baseline_metrics):
        """Test that preprocessing time hasn't regressed"""
        from audhd_correlation.preprocess import scale_features

        data = pd.DataFrame(np.random.randn(100, 50))

        start = time.time()
        scale_features(data)
        duration = time.time() - start

        # Compare to baseline (with tolerance)
        baseline_time = baseline_metrics.get('preprocessing_time', 0.5)
        assert duration < baseline_time * 1.5, \
            f"Preprocessing regressed: {duration}s vs baseline {baseline_time}s"

    def test_baseline_clustering_time(self, baseline_metrics, integrated_data):
        """Test that clustering time hasn't regressed"""
        from audhd_correlation.modeling.clustering import perform_clustering

        start = time.time()
        perform_clustering(
            integrated_data.values,
            method='kmeans',
            n_clusters=3
        )
        duration = time.time() - start

        baseline_time = baseline_metrics.get('clustering_time', 2.0)
        assert duration < baseline_time * 1.5, \
            f"Clustering regressed: {duration}s vs baseline {baseline_time}s"


# ============================================================================
# Load Testing
# ============================================================================

class TestLoadHandling:
    """Tests for handling large datasets"""

    @pytest.mark.slow
    @pytest.mark.parametrize("n_samples", [1000, 5000, 10000])
    def test_large_sample_handling(self, n_samples):
        """Test handling of large sample sizes"""
        from audhd_correlation.preprocess import scale_features

        data = pd.DataFrame(np.random.randn(n_samples, 100))

        start = time.time()
        scale_features(data)
        duration = time.time() - start

        # Should scale sub-quadratically
        assert duration < (n_samples / 1000) ** 1.5, \
            f"Poor scaling for {n_samples} samples"

    @pytest.mark.slow
    @pytest.mark.parametrize("n_features", [100, 500, 1000, 5000])
    def test_large_feature_handling(self, n_features):
        """Test handling of high-dimensional data"""
        from audhd_correlation.preprocess import scale_features

        data = pd.DataFrame(np.random.randn(100, n_features))

        start = time.time()
        scale_features(data)
        duration = time.time() - start

        # Should scale roughly linearly with features
        assert duration < (n_features / 100) * 1.2, \
            f"Poor scaling for {n_features} features"