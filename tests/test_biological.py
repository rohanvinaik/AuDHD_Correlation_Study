"""Comprehensive tests for biological interpretation framework"""
import pytest
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

from src.audhd_correlation.biological.gsea import (
    run_gsea,
    prerank_gsea,
    load_gene_sets,
    compare_cluster_enrichments,
    GSEAResult,
    GSEAResults,
)

from src.audhd_correlation.biological.networks import (
    reconstruct_metabolic_network,
    analyze_ppi_network,
    find_hub_nodes,
    community_detection,
    NetworkResult,
)

from src.audhd_correlation.biological.pathway_integration import (
    integrate_multiomics_pathways,
    combined_pathway_score,
    cross_omics_enrichment,
    MultiOmicsPathwayResult,
)

from src.audhd_correlation.biological.drug_targets import (
    predict_drug_targets,
    rank_druggable_targets,
    find_approved_drugs,
    drug_repurposing_candidates,
    DrugTargetResult,
)


@pytest.fixture
def expression_data():
    """Create synthetic expression data"""
    np.random.seed(42)
    n_samples = 100
    n_genes = 50

    # Generate gene names
    genes = [f'GENE{i}' for i in range(n_genes)]

    # Create expression matrix
    data = np.random.randn(n_samples, n_genes)

    df = pd.DataFrame(data, columns=genes)
    return df


@pytest.fixture
def cluster_labels():
    """Create cluster labels"""
    np.random.seed(42)
    labels = np.random.choice([0, 1, 2], size=100)
    return labels


@pytest.fixture
def gene_sets():
    """Create example gene sets"""
    gene_sets = {
        'Pathway_A': {f'GENE{i}' for i in range(0, 15)},
        'Pathway_B': {f'GENE{i}' for i in range(5, 20)},
        'Pathway_C': {f'GENE{i}' for i in range(10, 25)},
        'Pathway_D': {f'GENE{i}' for i in range(15, 30)},
    }
    return gene_sets


@pytest.fixture
def metabolite_data():
    """Create synthetic metabolite data"""
    np.random.seed(42)
    n_samples = 100
    n_metabolites = 20

    metabolites = [f'MET{i}' for i in range(n_metabolites)]
    data = np.random.randn(n_samples, n_metabolites)

    # Add some correlations
    data[:, 1] = data[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2
    data[:, 2] = data[:, 0] * 0.7 + np.random.randn(n_samples) * 0.3

    df = pd.DataFrame(data, columns=metabolites)
    return df


@pytest.fixture
def protein_list():
    """Create protein list for PPI"""
    return ['IL6', 'TNF', 'NFKB1', 'JAK1', 'STAT3', 'IL1B', 'IL6R']


class TestGSEA:
    """Tests for GSEA analysis"""

    def test_load_gene_sets(self):
        """Test loading gene sets"""
        gene_sets = load_gene_sets(database='GO_Biological_Process')

        assert isinstance(gene_sets, dict)
        assert len(gene_sets) > 0
        assert all(isinstance(v, set) for v in gene_sets.values())

    def test_prerank_gsea(self, gene_sets):
        """Test preranked GSEA"""
        # Create ranked gene list
        np.random.seed(42)
        genes = [f'GENE{i}' for i in range(50)]
        ranks = np.random.randn(50)

        ranked_genes = pd.Series(ranks, index=genes).sort_values(ascending=False)

        # Run GSEA
        results = prerank_gsea(
            ranked_genes=ranked_genes,
            gene_sets=gene_sets,
            n_permutations=100,  # Small for speed
            min_size=3,
            max_size=100,
        )

        assert isinstance(results, GSEAResults)
        assert len(results.results) > 0
        assert results.n_permutations == 100

        # Check result structure
        for result in results.results:
            assert isinstance(result, GSEAResult)
            assert result.pathway_id is not None
            assert result.pathway_name is not None
            assert result.es is not None
            assert result.nes is not None
            assert 0 <= result.pval <= 1
            assert 0 <= result.fdr <= 1

    def test_run_gsea(self, expression_data, cluster_labels, gene_sets):
        """Test standard GSEA"""
        results = run_gsea(
            expression_data=expression_data,
            labels=cluster_labels,
            gene_sets=gene_sets,
            cluster_id=0,
            n_permutations=100,
        )

        assert isinstance(results, GSEAResults)
        assert len(results.results) > 0

    def test_gsea_results_methods(self, gene_sets):
        """Test GSEAResults methods"""
        # Create ranked genes
        np.random.seed(42)
        genes = [f'GENE{i}' for i in range(50)]
        ranks = np.random.randn(50)
        ranked_genes = pd.Series(ranks, index=genes).sort_values(ascending=False)

        results = prerank_gsea(
            ranked_genes=ranked_genes,
            gene_sets=gene_sets,
            n_permutations=50,
            min_size=3,
        )

        # Test top pathways
        top = results.top_pathways(n=2)
        assert len(top) <= 2

        # Test significant pathways
        sig = results.significant_pathways()
        assert all(r.fdr <= 0.25 for r in sig)

        # Test to_dataframe
        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert 'pathway_name' in df.columns
        assert 'NES' in df.columns

    def test_compare_cluster_enrichments(self, expression_data, cluster_labels, gene_sets):
        """Test comparing enrichments across clusters"""
        comparison = compare_cluster_enrichments(
            expression_data=expression_data,
            labels=cluster_labels,
            gene_sets=gene_sets,
            n_permutations=50,
        )

        assert isinstance(comparison, pd.DataFrame)
        # Should have columns for each cluster
        assert any('cluster_' in col for col in comparison.columns)


class TestNetworks:
    """Tests for network analysis"""

    def test_reconstruct_metabolic_network(self, metabolite_data):
        """Test metabolic network reconstruction"""
        result = reconstruct_metabolic_network(
            metabolite_data=metabolite_data,
            correlation_threshold=0.6,
            method='pearson',
        )

        assert isinstance(result, NetworkResult)
        assert isinstance(result.graph, nx.Graph)
        assert result.n_nodes > 0
        assert result.n_edges >= 0
        assert len(result.degree_centrality) == result.n_nodes

    def test_analyze_ppi_network(self, protein_list):
        """Test PPI network analysis"""
        result = analyze_ppi_network(
            proteins=protein_list,
            ppi_database=None,  # Use example database
        )

        assert isinstance(result, NetworkResult)
        assert result.n_nodes > 0
        assert len(result.degree_centrality) == result.n_nodes

    def test_network_centrality_measures(self, metabolite_data):
        """Test centrality measures"""
        result = reconstruct_metabolic_network(
            metabolite_data=metabolite_data,
            correlation_threshold=0.5,
        )

        assert isinstance(result.degree_centrality, dict)
        assert isinstance(result.betweenness_centrality, dict)
        assert isinstance(result.closeness_centrality, dict)
        assert isinstance(result.eigenvector_centrality, dict)

        # All values should be between 0 and 1
        for centrality in [
            result.degree_centrality,
            result.betweenness_centrality,
            result.closeness_centrality,
        ]:
            for value in centrality.values():
                assert 0 <= value <= 1

    def test_find_hub_nodes(self, metabolite_data):
        """Test hub node identification"""
        result = reconstruct_metabolic_network(
            metabolite_data=metabolite_data,
            correlation_threshold=0.6,
        )

        hubs = find_hub_nodes(result, n_hubs=5, metric='degree')

        assert isinstance(hubs, list)
        assert len(hubs) <= 5

    def test_community_detection(self, metabolite_data):
        """Test community detection"""
        result = reconstruct_metabolic_network(
            metabolite_data=metabolite_data,
            correlation_threshold=0.5,
        )

        result_with_communities = community_detection(
            result,
            method='greedy',
        )

        assert result_with_communities.communities is not None
        assert result_with_communities.modularity is not None
        assert len(result_with_communities.communities) > 0

    def test_network_statistics(self, metabolite_data):
        """Test network statistics calculation"""
        result = reconstruct_metabolic_network(
            metabolite_data=metabolite_data,
            correlation_threshold=0.6,
        )

        assert result.avg_degree is not None
        assert result.density is not None
        assert result.avg_clustering is not None
        assert result.avg_path_length is not None

        # Check reasonable ranges
        assert 0 <= result.density <= 1
        assert 0 <= result.avg_clustering <= 1

    def test_empty_network(self):
        """Test handling of empty network"""
        # Create data with no correlations
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f'M{i}' for i in range(5)]
        )

        result = reconstruct_metabolic_network(
            metabolite_data=data,
            correlation_threshold=0.99,  # Very high threshold
        )

        assert result.n_nodes > 0
        assert result.n_edges >= 0  # May be 0


class TestPathwayIntegration:
    """Tests for multi-omics pathway integration"""

    def test_integrate_multiomics_pathways(self):
        """Test multi-omics integration"""
        # Create mock enrichment results
        genomics = pd.DataFrame({
            'pathway_name': ['Pathway_A', 'Pathway_B', 'Pathway_C'],
            'NES': [2.5, -1.8, 1.5],
            'FDR': [0.01, 0.05, 0.15],
        })

        transcriptomics = pd.DataFrame({
            'pathway_name': ['Pathway_A', 'Pathway_B', 'Pathway_D'],
            'NES': [2.2, -2.0, 1.8],
            'FDR': [0.02, 0.03, 0.10],
        })

        results = integrate_multiomics_pathways(
            genomics_results=genomics,
            transcriptomics_results=transcriptomics,
            method='weighted_average',
            fdr_threshold=0.25,
        )

        assert len(results.results) > 0
        assert results.integration_method == 'weighted_average'

        # Check result structure
        for result in results.results:
            assert isinstance(result, MultiOmicsPathwayResult)
            assert result.pathway_name is not None
            assert result.combined_score is not None

    def test_integration_methods(self):
        """Test different integration methods"""
        genomics = pd.DataFrame({
            'pathway_name': ['Pathway_A'],
            'NES': [2.5],
            'FDR': [0.01],
        })

        transcriptomics = pd.DataFrame({
            'pathway_name': ['Pathway_A'],
            'NES': [2.2],
            'FDR': [0.02],
        })

        for method in ['weighted_average', 'rank_aggregation', 'stouffer']:
            results = integrate_multiomics_pathways(
                genomics_results=genomics,
                transcriptomics_results=transcriptomics,
                method=method,
            )

            assert len(results.results) > 0
            assert results.integration_method == method

    def test_consistent_pathways(self):
        """Test identifying consistent pathways"""
        genomics = pd.DataFrame({
            'pathway_name': ['Pathway_A', 'Pathway_B'],
            'NES': [2.5, 1.5],
            'FDR': [0.01, 0.15],
        })

        transcriptomics = pd.DataFrame({
            'pathway_name': ['Pathway_A', 'Pathway_C'],
            'NES': [2.2, 1.8],
            'FDR': [0.02, 0.10],
        })

        results = integrate_multiomics_pathways(
            genomics_results=genomics,
            transcriptomics_results=transcriptomics,
        )

        # Pathway_A is in both omics and significant
        consistent = results.consistent_pathways(min_omics=2)
        pathway_names = [r.pathway_name for r in consistent]

        assert 'Pathway_A' in pathway_names

    def test_cross_omics_enrichment(self):
        """Test cross-omics enrichment analysis"""
        genomics_genes = {'GENE1', 'GENE2', 'GENE3', 'GENE4'}
        transcriptomics_genes = {'GENE2', 'GENE3', 'GENE4', 'GENE5'}
        proteomics_genes = {'GENE3', 'GENE4', 'GENE5', 'GENE6'}

        pathway_gene_sets = {
            'Pathway_A': {'GENE3', 'GENE4', 'GENE7', 'GENE8'},
            'Pathway_B': {'GENE1', 'GENE2', 'GENE9', 'GENE10'},
        }

        results = cross_omics_enrichment(
            genomics_genes=genomics_genes,
            transcriptomics_genes=transcriptomics_genes,
            proteomics_genes=proteomics_genes,
            pathway_gene_sets=pathway_gene_sets,
            min_overlap=1,
        )

        assert isinstance(results, pd.DataFrame)
        if len(results) > 0:
            assert 'cross_omics_overlap' in results.columns
            assert 'p_combined' in results.columns


class TestDrugTargets:
    """Tests for drug target prediction"""

    def test_predict_drug_targets(self):
        """Test drug target prediction"""
        genes = ['IL6', 'TNF', 'DRD2', 'MTOR', 'BDNF']

        results = predict_drug_targets(
            genes=genes,
            ranking_method='weighted_score',
        )

        assert len(results.results) == len(genes)

        # Check result structure
        for result in results.results:
            assert isinstance(result, DrugTargetResult)
            assert result.target_gene in genes
            assert 0 <= result.combined_score <= 1
            assert 0 <= result.druggability_score <= 1

    def test_drug_target_with_expression(self, expression_data):
        """Test drug target prediction with expression data"""
        # Use genes that exist in expression data
        genes = ['GENE0', 'GENE1', 'GENE2']

        results = predict_drug_targets(
            genes=genes,
            expression_data=expression_data,
        )

        assert len(results.results) == len(genes)

        # Expression scores should be computed
        for result in results.results:
            assert result.expression_score > 0

    def test_rank_druggable_targets(self):
        """Test target ranking"""
        genes = ['IL6', 'TNF', 'DRD2', 'BDNF']

        results = predict_drug_targets(genes=genes)

        # Test different prioritization strategies
        for strategy in ['novel', 'approved', 'hubs']:
            reranked = rank_druggable_targets(results, prioritize=strategy)

            assert len(reranked.results) == len(results.results)
            assert strategy in reranked.ranking_method

    def test_find_approved_drugs(self):
        """Test finding approved drugs"""
        genes = ['IL6', 'TNF', 'DRD2']

        results = predict_drug_targets(genes=genes)
        drugs_df = find_approved_drugs(results)

        assert isinstance(drugs_df, pd.DataFrame)

        if len(drugs_df) > 0:
            assert 'drug_name' in drugs_df.columns
            assert 'target_gene' in drugs_df.columns
            assert 'clinical_phase' in drugs_df.columns

    def test_novel_targets(self):
        """Test identifying novel targets"""
        genes = ['IL6', 'TNF', 'BDNF', 'GABA']

        results = predict_drug_targets(genes=genes)
        novel = results.novel_targets()

        # BDNF and GABA have no known drugs
        novel_genes = [r.target_gene for r in novel]
        assert 'BDNF' in novel_genes

    def test_approved_drug_targets(self):
        """Test identifying targets with approved drugs"""
        genes = ['IL6', 'TNF', 'DRD2', 'BDNF']

        results = predict_drug_targets(genes=genes)
        approved = results.approved_drug_targets()

        # IL6, TNF, DRD2 have approved drugs
        approved_genes = [r.target_gene for r in approved]
        assert 'IL6' in approved_genes
        assert 'TNF' in approved_genes
        assert 'DRD2' in approved_genes

    def test_drug_repurposing_candidates(self):
        """Test drug repurposing identification"""
        cluster_targets = {
            0: ['IL6', 'TNF', 'NFKB1'],
            1: ['DRD2', 'HTR2A', 'SLC6A4'],
            2: ['MTOR', 'JAK1', 'STAT3'],
        }

        candidates = drug_repurposing_candidates(
            cluster_targets=cluster_targets,
            min_druggability=0.7,
        )

        assert isinstance(candidates, pd.DataFrame)

        if len(candidates) > 0:
            assert 'cluster_id' in candidates.columns
            assert 'target_gene' in candidates.columns
            assert 'drug_name' in candidates.columns
            assert 'current_indication' in candidates.columns

    def test_to_dataframe(self):
        """Test converting results to DataFrame"""
        genes = ['IL6', 'TNF', 'DRD2']

        results = predict_drug_targets(genes=genes)
        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(genes)
        assert 'target_gene' in df.columns
        assert 'combined_score' in df.columns


class TestEdgeCases:
    """Test edge cases"""

    def test_empty_gene_list(self):
        """Test with empty gene list"""
        results = predict_drug_targets(genes=[])
        assert len(results.results) == 0

    def test_unknown_genes(self):
        """Test with unknown genes"""
        genes = ['UNKNOWN1', 'UNKNOWN2']

        results = predict_drug_targets(genes=genes)

        # Should still return results with default scores
        assert len(results.results) == len(genes)

        for result in results.results:
            # Unknown genes get default druggability score
            assert result.druggability_score == 0.5

    def test_single_gene(self):
        """Test with single gene"""
        results = predict_drug_targets(genes=['IL6'])

        assert len(results.results) == 1
        assert results.results[0].target_gene == 'IL6'

    def test_gsea_no_significant_pathways(self, gene_sets):
        """Test GSEA with no significant pathways"""
        # Create ranked genes with no strong signal
        np.random.seed(42)
        genes = [f'GENE{i}' for i in range(50)]
        ranks = np.random.randn(50) * 0.1  # Very small ranks

        ranked_genes = pd.Series(ranks, index=genes).sort_values(ascending=False)

        results = prerank_gsea(
            ranked_genes=ranked_genes,
            gene_sets=gene_sets,
            n_permutations=50,
            min_size=3,
        )

        # Should still return results, but may have no significant pathways
        assert isinstance(results, GSEAResults)

    def test_network_insufficient_samples(self):
        """Test network reconstruction with insufficient samples"""
        data = pd.DataFrame(
            np.random.randn(5, 10),
            columns=[f'M{i}' for i in range(10)]
        )

        with pytest.raises(ValueError, match="Need at least"):
            reconstruct_metabolic_network(data, min_samples=10)