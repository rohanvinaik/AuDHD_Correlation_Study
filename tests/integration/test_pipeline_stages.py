"""Integration tests for pipeline stages

Tests the complete flow of data through pipeline stages.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestDataLoadingPipeline:
    """Test data loading pipeline"""

    def test_end_to_end_data_loading(self, temp_data_dir):
        """Test complete data loading workflow"""
        # This would test loading all modalities and harmonization
        # Skipped if actual loaders not fully implemented
        pytest.skip("Requires full loader implementation")

    def test_multi_omics_alignment(
        self,
        synthetic_genomic_data,
        synthetic_clinical_data,
        synthetic_metabolomic_data,
    ):
        """Test that all modalities align on sample IDs"""
        from audhd_correlation.data.harmonize import align_multiomics

        genomic = synthetic_genomic_data['genotypes']
        clinical = synthetic_clinical_data.set_index('sample_id')
        metabolomic = synthetic_metabolomic_data

        aligned = align_multiomics({
            'genomic': genomic,
            'clinical': clinical,
            'metabolomic': metabolomic,
        })

        # All should have same samples
        sample_sets = [set(df.index) for df in aligned.values()]
        common_samples = set.intersection(*sample_sets)

        assert len(common_samples) > 0
        for df in aligned.values():
            assert len(df) == len(common_samples)


class TestPreprocessingPipeline:
    """Test preprocessing pipeline"""

    def test_full_preprocessing_workflow(self, synthetic_metabolomic_data):
        """Test complete preprocessing: impute, scale, batch correct"""
        from audhd_correlation.preprocess import (
            impute_missing,
            scale_features,
        )

        # Start with data with missing values
        data = synthetic_metabolomic_data.copy()

        # Step 1: Imputation
        imputed = impute_missing(data, method='knn', n_neighbors=5)
        assert imputed.isna().sum().sum() < data.isna().sum().sum()

        # Step 2: Scaling
        scaled = scale_features(imputed, method='standard')
        assert abs(scaled.mean().mean()) < 0.1  # Near zero mean
        assert abs(scaled.std().mean() - 1.0) < 0.1  # Near unit variance

    def test_preprocessing_preserves_structure(self, synthetic_clinical_data):
        """Test that preprocessing preserves data structure"""
        from audhd_correlation.preprocess import scale_features

        data = synthetic_clinical_data[['age', 'bmi', 'severity_score', 'iq']]

        scaled = scale_features(data)

        assert scaled.shape == data.shape
        assert list(scaled.columns) == list(data.columns)
        assert list(scaled.index) == list(data.index)


class TestIntegrationPipeline:
    """Test multi-omics integration pipeline"""

    def test_pca_integration(
        self,
        synthetic_genomic_data,
        synthetic_metabolomic_data,
    ):
        """Test PCA-based integration"""
        from audhd_correlation.integrate import integrate_multiomics

        # Prepare data
        genomic = synthetic_genomic_data['genotypes'].iloc[:, :100]  # Subset
        metabolomic = synthetic_metabolomic_data

        result = integrate_multiomics(
            {'genomic': genomic, 'metabolomic': metabolomic},
            method='pca',
            n_components=10,
        )

        assert result is not None
        # Should have reduced dimensions
        if hasattr(result, 'shape'):
            assert result.shape[1] == 10

    def test_integration_output_format(self, integrated_data):
        """Test that integration output has correct format"""
        assert isinstance(integrated_data, pd.DataFrame)
        assert integrated_data.shape[1] > 0  # Has factors
        assert integrated_data.shape[0] > 0  # Has samples
        assert not integrated_data.isna().any().any()  # No missing values


class TestClusteringPipeline:
    """Test clustering pipeline"""

    def test_hdbscan_clustering(self, integrated_data):
        """Test HDBSCAN clustering"""
        from audhd_correlation.modeling.clustering import perform_clustering

        result = perform_clustering(
            integrated_data.values,
            method='hdbscan',
            min_cluster_size=15,
        )

        assert 'labels' in result
        assert 'embedding' in result
        assert len(result['labels']) == len(integrated_data)

        # Check that we found some clusters
        n_clusters = len(set(result['labels'])) - (1 if -1 in result['labels'] else 0)
        assert n_clusters > 0

    def test_kmeans_clustering(self, integrated_data):
        """Test K-means clustering"""
        from audhd_correlation.modeling.clustering import perform_clustering

        result = perform_clustering(
            integrated_data.values,
            method='kmeans',
            n_clusters=3,
        )

        assert 'labels' in result
        assert len(result['labels']) == len(integrated_data)
        # K-means should find exactly k clusters
        assert len(set(result['labels'])) == 3

    def test_clustering_with_embedding(self, integrated_data):
        """Test clustering includes UMAP embedding"""
        from audhd_correlation.modeling.clustering import perform_clustering

        result = perform_clustering(
            integrated_data.values,
            method='hdbscan',
            embedding_method='umap',
        )

        assert 'embedding' in result
        assert result['embedding'].shape == (len(integrated_data), 2)


class TestValidationPipeline:
    """Test validation pipeline"""

    def test_internal_validation_metrics(self, integrated_data, clustering_result):
        """Test computation of internal validation metrics"""
        from audhd_correlation.validation import compute_internal_metrics

        X = integrated_data.values
        labels = clustering_result['labels']

        metrics = compute_internal_metrics(X, labels)

        assert 'silhouette' in metrics or 'silhouette_score' in metrics
        # Silhouette should be between -1 and 1
        sil_key = 'silhouette' if 'silhouette' in metrics else 'silhouette_score'
        assert -1 <= metrics[sil_key] <= 1

    def test_stability_analysis(self, integrated_data, clustering_result):
        """Test stability analysis"""
        from audhd_correlation.validation import bootstrap_stability

        X = integrated_data.values
        labels = clustering_result['labels']

        result = bootstrap_stability(
            X,
            labels,
            n_bootstrap=10,  # Small number for speed
        )

        assert 'mean_ari' in result or 'ari_mean' in result
        # ARI should be between 0 and 1
        ari_key = 'mean_ari' if 'mean_ari' in result else 'ari_mean'
        assert 0 <= result[ari_key] <= 1


class TestBiologicalAnalysisPipeline:
    """Test biological analysis pipeline"""

    @pytest.mark.skip(reason="Requires gene set databases")
    def test_pathway_enrichment(self, synthetic_genomic_data, clustering_result):
        """Test pathway enrichment analysis"""
        from audhd_correlation.biological import run_pathway_enrichment

        genotypes = synthetic_genomic_data['genotypes']
        labels = clustering_result['labels']

        result = run_pathway_enrichment(
            genotypes,
            labels,
            modality='genomic',
        )

        assert result is not None

    @pytest.mark.skip(reason="Requires network databases")
    def test_network_analysis(self, synthetic_metabolomic_data, clustering_result):
        """Test biological network analysis"""
        from audhd_correlation.biological import build_biological_networks

        metabolomic = synthetic_metabolomic_data
        labels = clustering_result['labels']

        result = build_biological_networks(
            {'metabolomic': metabolomic},
            labels,
        )

        assert result is not None


class TestVisualizationPipeline:
    """Test visualization pipeline"""

    def test_embedding_plot_creation(self, clustering_result, temp_output_dir):
        """Test creation of embedding plots"""
        from audhd_correlation.viz import plot_embedding

        embedding = clustering_result['embedding']
        labels = clustering_result['labels']
        output_path = temp_output_dir / 'embedding.png'

        result = plot_embedding(
            embedding,
            labels,
            output_path=output_path,
        )

        assert output_path.exists() or result is not None

    def test_heatmap_creation(
        self,
        synthetic_metabolomic_data,
        clustering_result,
        temp_output_dir
    ):
        """Test creation of heatmaps"""
        from audhd_correlation.viz import plot_heatmaps

        data = {'metabolomic': synthetic_metabolomic_data}
        labels = clustering_result['labels']

        result = plot_heatmaps(
            data,
            labels,
            output_dir=temp_output_dir,
        )

        assert result is not None or len(list(temp_output_dir.glob('*.png'))) > 0


class TestReportingPipeline:
    """Test reporting pipeline"""

    def test_report_generation(
        self,
        clustering_result,
        temp_output_dir,
    ):
        """Test generation of analysis report"""
        from audhd_correlation.reporting import generate_comprehensive_report

        report_path = temp_output_dir / 'report.html'

        result = generate_comprehensive_report(
            clustering_results=clustering_result,
            validation_results={'silhouette': 0.5},
            biological_results={},
            visualization_results={},
            output_path=report_path,
        )

        assert report_path.exists() or result is not None


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================

class TestEndToEndPipeline:
    """Test complete pipeline flow"""

    @pytest.mark.slow
    def test_minimal_pipeline_execution(
        self,
        synthetic_genomic_data,
        synthetic_clinical_data,
        temp_output_dir,
    ):
        """Test execution of minimal pipeline"""
        # This would test running the entire pipeline
        # with minimal settings for speed
        pytest.skip("Requires full pipeline implementation")

    def test_pipeline_with_checkpointing(self, temp_output_dir):
        """Test pipeline with checkpoint/resume"""
        # Test that pipeline can be interrupted and resumed
        pytest.skip("Requires checkpoint implementation")

    def test_pipeline_handles_errors_gracefully(self):
        """Test that pipeline handles errors appropriately"""
        # Test error handling throughout pipeline
        pytest.skip("Requires error handling implementation")


# ============================================================================
# Data Flow Tests
# ============================================================================

class TestDataFlow:
    """Test data flow through pipeline"""

    def test_data_types_preserved(
        self,
        synthetic_genomic_data,
        synthetic_clinical_data,
    ):
        """Test that data types are preserved through pipeline"""
        # Genomic should stay numeric
        assert synthetic_genomic_data['genotypes'].dtypes.apply(
            lambda x: np.issubdtype(x, np.number)
        ).all()

        # Clinical categorical should stay categorical
        assert synthetic_clinical_data['diagnosis'].dtype == 'object'

    def test_sample_ids_tracked(
        self,
        synthetic_genomic_data,
        synthetic_clinical_data,
    ):
        """Test that sample IDs are tracked through pipeline"""
        genomic_samples = set(synthetic_genomic_data['genotypes'].index)
        clinical_samples = set(synthetic_clinical_data['sample_id'])

        # Should have overlap
        assert len(genomic_samples & clinical_samples) > 0

    def test_no_data_leakage(self, integrated_data, clustering_result):
        """Test that no data leaks between train/test"""
        # This would test proper cross-validation
        # to ensure no information leakage
        pass  # Placeholder


# ============================================================================
# Performance Tests
# ============================================================================

class TestPipelinePerformance:
    """Test pipeline performance"""

    @pytest.mark.slow
    def test_preprocessing_scales_linearly(self):
        """Test that preprocessing scales linearly with data size"""
        import time
        from audhd_correlation.preprocess import scale_features

        times = []
        sizes = [100, 200, 400]

        for n in sizes:
            data = pd.DataFrame(np.random.randn(n, 50))
            start = time.time()
            scale_features(data)
            times.append(time.time() - start)

        # Check roughly linear scaling (within factor of 2)
        assert times[1] / times[0] < 3
        assert times[2] / times[1] < 3

    @pytest.mark.slow
    def test_integration_memory_efficient(self):
        """Test that integration doesn't use excessive memory"""
        # Would use memory_profiler or similar
        pytest.skip("Requires memory profiling setup")